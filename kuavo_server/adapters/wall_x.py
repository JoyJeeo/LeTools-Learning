from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from ..runtime import register_adapter
from .base import ModelServerAdapter, resolve_model_repo_root


_STANDARD_CAMERA_KEYS = {
    "observation.images.head_cam_h": "observation.images.head_cam_h",
    "observation.images.wrist_cam_l": "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r": "observation.images.wrist_cam_r",
    "face_view": "observation.images.head_cam_h",
    "front_view": "observation.images.head_cam_h",
    "left_wrist_view": "observation.images.wrist_cam_l",
    "right_wrist_view": "observation.images.wrist_cam_r",
}


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _as_hwc_uint8(image: Any) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"Wall-X image must be HWC, got shape={array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Wall-X image must have 3 channels, got shape={array.shape}")
    return np.ascontiguousarray(array)


def _layout(config: dict[str, Any], key: str) -> dict[str, int]:
    task = config.get("task") or {}
    value = config.get(key) or task.get(key) or (config.get("data") or {}).get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Wall-X train config is missing task.{key}")
    return {str(name): int(dim) for name, dim in value.items()}


def _real_dim(layout: dict[str, int]) -> int:
    return sum(dim for key, dim in layout.items() if key != "action_padding")


def _strip_padding(chunk: np.ndarray, layout: dict[str, int]) -> np.ndarray:
    parts: list[np.ndarray] = []
    start = 0
    for key, dim in layout.items():
        end = start + dim
        if end > chunk.shape[1]:
            raise ValueError(
                f"Wall-X output width {chunk.shape[1]} is smaller than configured layout width {end}"
            )
        if key != "action_padding":
            parts.append(chunk[:, start:end])
        start = end
    if start != chunk.shape[1]:
        raise ValueError(
            f"Wall-X output width {chunk.shape[1]} does not match task.dof_config width {start}"
        )
    return np.concatenate(parts, axis=1) if parts else np.empty((len(chunk), 0), dtype=chunk.dtype)


@register_adapter
class WallXAdapter(ModelServerAdapter):
    """Serve a native Wall-X flow checkpoint with Kuavo's stable ZMQ payload."""

    name = "wall_x"

    def __init__(
        self,
        *,
        checkpoint: str,
        train_config_path: str,
        model_repo_root: str,
        which_arm: str,
        action_horizon: int,
        execution_horizon: int,
        device: str,
        num_inference_timesteps: int,
        norm_key: str,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("--action_horizon must be a positive integer")
        if execution_horizon < 0:
            raise ValueError("--execution_horizon must be zero or a positive integer")

        self.model_repo_root = resolve_model_repo_root("wall_x", model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Wall-X checkpoint not found at: {self.checkpoint}")
        self.train_config_path = Path(train_config_path).expanduser().resolve() if train_config_path else None
        if self.train_config_path is not None and not self.train_config_path.is_file():
            raise FileNotFoundError(f"Wall-X train config not found at: {self.train_config_path}")

        self.which_arm = which_arm
        self.action_horizon = action_horizon
        self.execution_horizon = execution_horizon
        self.device = device
        self.num_inference_timesteps = num_inference_timesteps
        self.norm_key = norm_key
        self._pending_actions: list[np.ndarray] = []

        if str(self.model_repo_root) not in sys.path:
            sys.path.insert(0, str(self.model_repo_root))

        from wall_x._vendor.harrix.serving._wallx_infer.base_dataclass import (  # type: ignore
            RobotStateActionData,
        )
        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig  # type: ignore
        from wall_x._vendor.harrix.serving.policy.wall_x_policy import WallXPolicy  # type: ignore

        config = InferConfig(
            checkpoint_path=str(self.checkpoint),
            train_config_path=str(self.train_config_path) if self.train_config_path else None,
            robot_type="desktop",
            robot_use_joint_angle_control=True,
            action_horizon=action_horizon,
            action_dim=None,
            model_device=device,
            num_inference_timesteps=num_inference_timesteps,
            norm_key=norm_key,
        )
        self.model = WallXPolicy(
            config=config,
            image_passing_mode="numpy",
            default_infer_mode="flow",
            serialize_actions=False,
        )
        self.config = config
        self._state_action_data_cls = RobotStateActionData
        self.agent_pos_layout = _layout(config.train_config, "agent_pos_config")
        self.action_layout = _layout(config.train_config, "dof_config")
        self.expected_state_dim = _real_dim(self.agent_pos_layout)
        self.expected_action_dim = _real_dim(self.action_layout)
        self.camera_sources = self._resolve_camera_sources(config.train_config, config.cam_names)

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint", type=str, required=True, help="Merged Wall-X checkpoint directory"
        )
        parser.add_argument(
            "--train_config_path",
            type=str,
            default="",
            help="Training YAML used by the checkpoint; empty uses config.yml/config.yaml beside it",
        )
        parser.add_argument(
            "--model_repo_root",
            type=str,
            default="",
            help="Optional Wall-X source override; defaults to vendored kuavo_model/external_models/wall-x",
        )
        parser.add_argument("--which_arm", type=str, default="both", choices=["left", "right", "both"])
        parser.add_argument("--action_horizon", type=int, default=32, help="Wall-X generated chunk length")
        parser.add_argument(
            "--execution_horizon",
            type=int,
            default=8,
            help="Number of generated actions exposed/cached; 0 keeps the full chunk",
        )
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--num_inference_timesteps", type=int, default=10)
        parser.add_argument(
            "--norm_key",
            type=str,
            default="kuavo",
            help="Normalizer dataset key; checkpoint-local norm_stats.json accepts any single key",
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "WallXAdapter":
        return cls(
            checkpoint=args.checkpoint,
            train_config_path=args.train_config_path,
            model_repo_root=args.model_repo_root,
            which_arm=args.which_arm,
            action_horizon=args.action_horizon,
            execution_horizon=args.execution_horizon,
            device=args.device,
            num_inference_timesteps=args.num_inference_timesteps,
            norm_key=args.norm_key,
        )

    @staticmethod
    def _resolve_camera_sources(
        train_config: dict[str, Any], cam_names: list[str]
    ) -> dict[str, str]:
        camera_mapping = (((train_config.get("data") or {}).get("key_mappings") or {}).get("camera") or {})
        sources: dict[str, str] = {}
        for dataset_key, model_key in camera_mapping.items():
            standard_key = _STANDARD_CAMERA_KEYS.get(str(dataset_key))
            if standard_key:
                sources[str(model_key)] = standard_key
        for model_key in cam_names:
            if model_key not in sources:
                standard_key = _STANDARD_CAMERA_KEYS.get(model_key)
                if standard_key:
                    sources[model_key] = standard_key
        missing = [name for name in cam_names if name not in sources]
        if missing:
            raise ValueError(
                "Cannot map Wall-X camera names to Kuavo payload keys: "
                f"{missing}. Set data.key_mappings.camera with standard Kuavo keys."
            )
        return sources

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "model_repo_root": str(self.model_repo_root),
            "checkpoint": str(self.checkpoint),
            "train_config_path": str(self.train_config_path or "<checkpoint-local>"),
            "which_arm": self.which_arm,
            "action_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "norm_key": self.config.norm_key,
            "state_dim": self.expected_state_dim,
            "action_dim": self.expected_action_dim,
            "camera_sources": self.camera_sources,
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        self.model.reset()
        return {"status": "ok", "message": "Wall-X action chunk cleared"}

    def _adapt_state(self, state: Any) -> np.ndarray:
        array = _to_numpy(state).astype(np.float32).reshape(-1)
        if array.shape[0] == self.expected_state_dim:
            return array
        if array.shape[0] == 16 and self.expected_state_dim == 8:
            if self.which_arm == "left":
                return array[:8]
            if self.which_arm == "right":
                return array[8:16]
        raise ValueError(
            f"Wall-X checkpoint expects {self.expected_state_dim} state dims, got {array.shape[0]}. "
            "Use a train config matching the kuavo_data which_arm/eef layout."
        )

    def _build_native_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = self._adapt_state(obs["observation.state"])
        native_state = self._state_action_data_cls(config=self.config)
        offset = 0
        for key, dim in self.agent_pos_layout.items():
            if key == "action_padding":
                continue
            native_state.save_state_data_with_key(state[offset : offset + dim], key, gt_dim=dim)
            offset += dim

        total_action_dim = sum(self.action_layout.values())
        dof_mask = np.ones((1, self.action_horizon, total_action_dim), dtype=np.float32)
        offset = 0
        for key, dim in self.action_layout.items():
            if key == "action_padding":
                dof_mask[:, :, offset : offset + dim] = 0
            offset += dim
        native_state.dof_mask = dof_mask

        native_obs: dict[str, Any] = {"robot_state_action_data": native_state}
        for model_key, kuavo_key in self.camera_sources.items():
            fallback = obs["observation.images.head_cam_h"]
            native_obs[model_key] = _as_hwc_uint8(obs.get(kuavo_key, fallback))
        return native_obs

    def _convert_chunk(self, action: Any) -> np.ndarray:
        chunk = _to_numpy(action).astype(np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2:
            raise ValueError(f"Wall-X predict_action must be [H,D] or [1,H,D], got {chunk.shape}")
        chunk = _strip_padding(chunk, self.action_layout)

        if chunk.shape[1] == 16 and self.expected_action_dim != 16:
            if self.which_arm == "left":
                chunk = chunk[:, :8]
            elif self.which_arm == "right":
                chunk = chunk[:, 8:16]
        expected = self.expected_action_dim
        if chunk.shape[1] != expected:
            raise ValueError(
                f"Kuavo {self.which_arm} mode requires {expected} action dims, "
                f"but Wall-X produced {chunk.shape[1]} after removing padding"
            )
        return chunk.astype(np.float64)

    def _predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        native_obs = self._build_native_observation(obs)
        prompt = str(obs.get("prompt", ""))
        result = self.model.model_wrapper.infer_flow_action(native_obs, prompt)
        if "predict_action" not in result:
            raise KeyError("Wall-X native inference result is missing predict_action")
        chunk = self._convert_chunk(result["predict_action"])
        if self.execution_horizon:
            chunk = chunk[: self.execution_horizon]
        if not len(chunk):
            raise ValueError("Wall-X returned an empty action chunk")
        return chunk

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._pending_actions:
            self._pending_actions = [row.copy() for row in self._predict_chunk(obs)]
        return self._pending_actions.pop(0)

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._predict_chunk(obs)
