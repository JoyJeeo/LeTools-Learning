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


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value if isinstance(value, np.ndarray) else np.asarray(value)


def _as_chw_uint8(image: Any) -> np.ndarray:
    arr = _to_numpy(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"G0.5 image must be 3D CHW or HWC, got {arr.shape}")
    if arr.shape[-1] in (1, 3) and arr.shape[0] not in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    if arr.shape[0] != 3:
        raise ValueError(f"G0.5 image must have 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


@register_adapter
class G05KuavoAdapter(ModelServerAdapter):
    """Serve a Kuavo-fine-tuned G0.5 checkpoint through the KLS ZMQ contract."""

    name = "g05"
    def __init__(
        self,
        *,
        checkpoint: str,
        model_repo_root: str,
        execution_horizon: int,
        device: str,
    ) -> None:
        self.model_repo_root = resolve_model_repo_root(self.name, model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"G0.5 checkpoint does not exist: {self.checkpoint}")
        if execution_horizon < 0:
            raise ValueError("execution_horizon must be >= 0")
        self.execution_horizon = execution_horizon
        self.device = device
        self._pending_actions: list[np.ndarray] = []

        for path in (self.model_repo_root, self.model_repo_root / "src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        scripts_dir = self.model_repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from g05.models.g05.inferencer import PolicyInferencer, resolve_processor
        from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
        from g05.utils.eval.eval_utils import filter_embodiment
        from serve_policy import setup

        run_dir = find_run_dir(str(self.checkpoint))
        cfg = load_config_from_run_dir(run_dir, str(self.checkpoint), [])
        if "embodiment_datasets" not in cfg.data:
            raise ValueError("G0.5 checkpoint config has no embodiment_datasets")
        filter_embodiment(cfg, "kuavo")
        self.action_horizon = int(cfg.data.action_size)
        self.policy, self.processor = setup(cfg, device=device)
        self.inferencer = PolicyInferencer(self.policy, self.processor, device=device)
        self.kuavo_processor = resolve_processor(
            self.processor, {"embodiment_type": "kuavo"}
        )
        self.state_layout = self._layout_from_shape_meta(self.kuavo_processor.shape_meta["state"])
        self.action_layout = self._layout_from_shape_meta(self.kuavo_processor.shape_meta["action"])
        self.state_dim = max(end for _, _, end in self.state_layout)
        self.action_dim = max(end for _, _, end in self.action_layout)

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--checkpoint", required=True, help="G0.5 model_state_dict.pt")
        parser.add_argument(
            "--model_repo_root",
            default="",
            help="GalaxeaVLA checkout; defaults to kuavo_model/external_models/GalaxeaVLA.",
        )
        parser.add_argument(
            "--execution_horizon",
            type=int,
            default=16,
            help="Returned chunk length. 0 uses the complete trained horizon.",
        )
        parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")

    @classmethod
    def from_args(cls, args: Namespace) -> "G05KuavoAdapter":
        return cls(
            checkpoint=args.checkpoint,
            model_repo_root=args.model_repo_root,
            execution_horizon=args.execution_horizon,
            device=args.device,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "model_repo_root": str(self.model_repo_root),
            "checkpoint": str(self.checkpoint),
            "execution_horizon": self.execution_horizon or self.action_horizon,
            "device": self.device,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_parts": [key for key, _, _ in self.state_layout],
            "action_parts": [key for key, _, _ in self.action_layout],
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        return {"status": "ok", "message": "G0.5 action cache cleared"}

    @staticmethod
    def _layout_from_shape_meta(shape_meta: Any) -> tuple[tuple[str, int, int], ...]:
        layout = []
        occupied: set[int] = set()
        for meta in shape_meta:
            key = str(meta["key"])
            start = int(meta["start_index"])
            end = start + int(meta["raw_shape"])
            indices = set(range(start, end))
            if start < 0 or end <= start or occupied.intersection(indices):
                raise ValueError(f"Invalid or overlapping G0.5 shape_meta entry for {key!r}")
            occupied.update(indices)
            layout.append((key, start, end))
        if not layout:
            raise ValueError("G0.5 checkpoint has an empty shape_meta layout")
        return tuple(layout)

    def _split_state(self, state: Any) -> dict[str, np.ndarray]:
        flat = _to_numpy(state).astype(np.float32).reshape(-1)
        if flat.size != self.state_dim:
            raise ValueError(f"G0.5 Kuavo adapter expects {self.state_dim}D state, got {flat.shape}")
        return {key: flat[start:end] for key, start, end in self.state_layout}

    def _pack_action(self, action: dict[str, Any], step: int) -> np.ndarray:
        packed = np.zeros(self.action_dim, dtype=np.float64)
        for key, start, end in self.action_layout:
            width = end - start
            if key not in action:
                raise ValueError(f"G0.5 output is missing Kuavo action key {key!r}")
            arr = _to_numpy(action[key])
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or arr.shape[1] != width:
                raise ValueError(f"G0.5 action[{key!r}] must be [T,{width}], got {arr.shape}")
            packed[start:end] = arr[min(step, arr.shape[0] - 1)]
        return packed

    def _build_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        head = obs["observation.images.head_cam_h"]
        left = obs.get("observation.images.wrist_cam_l", head)
        right = obs.get("observation.images.wrist_cam_r", head)
        return {
            "images": {
                "head_rgb": _as_chw_uint8(head),
                "left_wrist_rgb": _as_chw_uint8(left),
                "right_wrist_rgb": _as_chw_uint8(right),
            },
            "state": self._split_state(obs["observation.state"]),
            "task": str(obs.get("prompt", "robot manipulation")),
            "frequency": float(obs.get("frequency", 15.0)),
            "embodiment_type": "kuavo",
        }

    def _predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        from serve_policy import build_obs_dict

        model_obs = build_obs_dict(self._build_obs(obs), self.processor)
        action = self.inferencer.infer_one(model_obs)
        action.pop("_cot_text", None)
        action.pop("_absent_keys", None)
        first = next(iter(action.values()), None)
        if first is None:
            raise ValueError("G0.5 returned no Kuavo action parts")
        arr = _to_numpy(first)
        horizon = arr.shape[-2] if arr.ndim >= 2 else 1
        limit = horizon if self.execution_horizon == 0 else min(horizon, self.execution_horizon)
        if limit <= 0:
            raise ValueError("G0.5 returned an empty action chunk")
        return np.stack([self._pack_action(action, step) for step in range(limit)])

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if self._pending_actions:
            return self._pending_actions.pop(0)
        chunk = self._predict_chunk(obs)
        self._pending_actions = [step for step in chunk[1:]]
        return chunk[0]

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._predict_chunk(obs)
