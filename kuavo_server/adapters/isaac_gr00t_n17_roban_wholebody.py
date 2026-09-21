"""Isaac GR00T N1.7 server adapter for Roban wholebody 23D/32D IO."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import register_adapter
from .base import ModelServerAdapter
from .isaac_gr00t_n17 import (
    _Gr00tRuntime,
    _as_hwc_uint8,
    _resolve_repo_root,
    _to_numpy,
)


STATE_LAYOUT = {
    "left_arm": 4,
    "left_hand": 1,
    "right_arm": 4,
    "right_hand": 1,
    "left_leg": 6,
    "right_leg": 6,
    "waist": 1,
}
ACTION_LAYOUT = {
    **STATE_LAYOUT,
    "root_xyz": 3,
    "root_rot6d": 6,
}
STATE_DIM = sum(STATE_LAYOUT.values())
ACTION_DIM = sum(ACTION_LAYOUT.values())


def _validate_modality_layout(
    keys: list[str], dimensions: dict[str, int], expected: dict[str, int], name: str
) -> None:
    if set(keys) != set(expected):
        raise ValueError(
            f"Roban wholebody GR00T {name} keys must be {list(expected)}, got {keys}"
        )
    wrong = {
        key: (dimensions.get(key), width)
        for key, width in expected.items()
        if dimensions.get(key) != width
    }
    if wrong:
        raise ValueError(
            f"Roban wholebody GR00T {name} dimensions mismatch: {wrong}"
        )


@register_adapter
class IsaacGr00tN17RobanWholebodyAdapter(ModelServerAdapter):
    name = "isaac_gr00t_n17_roban_wholebody"

    def __init__(
        self,
        *,
        model_repo_root: str,
        checkpoint: str,
        embodiment_tag: str,
        execution_horizon: int | None,
        device: str,
        strict: bool,
    ) -> None:
        self.model_repo_root = _resolve_repo_root(model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint does not exist: {self.checkpoint}")
        if execution_horizon is not None and execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive or omitted")
        self.execution_horizon = execution_horizon
        self._pending_actions: list[np.ndarray] = []
        self.model = _Gr00tRuntime(
            repo_root=self.model_repo_root,
            model_path=self.checkpoint,
            embodiment_tag_raw=embodiment_tag,
            device=device,
            strict=strict,
        )
        _validate_modality_layout(
            self.model.state_keys, self.model.state_dims, STATE_LAYOUT, "state"
        )
        _validate_modality_layout(
            self.model.action_keys, self.model.action_dims, ACTION_LAYOUT, "action"
        )
        if not self.model.video_keys:
            raise ValueError("Roban wholebody checkpoint must define a video modality")
        print(
            "[isaac-gr00t-n17-roban-wholebody] "
            f"checkpoint={self.checkpoint} horizon="
            f"{self.execution_horizon or self.model.action_horizon}",
            flush=True,
        )

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--model_repo_root", type=str, default="")
        parser.add_argument("--checkpoint", type=str, required=True)
        parser.add_argument("--embodiment_tag", type=str, default="NEW_EMBODIMENT")
        parser.add_argument("--execution_horizon", type=int, default=50)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--strict", action="store_true")

    @classmethod
    def from_args(
        cls, args: Namespace
    ) -> "IsaacGr00tN17RobanWholebodyAdapter":
        return cls(
            model_repo_root=args.model_repo_root,
            checkpoint=args.checkpoint,
            embodiment_tag=args.embodiment_tag,
            execution_horizon=args.execution_horizon,
            device=args.device,
            strict=args.strict,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "platform_type": "roban",
            "checkpoint": str(self.checkpoint),
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "model_action_horizon": self.model.action_horizon,
            "execution_horizon": self.execution_horizon,
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        self.model.reset()
        return {"status": "ok", "message": "Roban wholebody adapter reset"}

    @staticmethod
    def _split_state(raw_state: Any) -> dict[str, np.ndarray]:
        state = _to_numpy(raw_state).astype(np.float32).reshape(-1)
        if state.size != STATE_DIM:
            raise ValueError(f"Roban wholebody state must be 23-D, got {state.size}")
        return {
            "left_arm": state[0:4],
            "left_hand": state[4:5],
            "right_arm": state[5:9],
            "right_hand": state[9:10],
            "left_leg": state[10:16],
            "right_leg": state[16:22],
            "waist": state[22:23],
        }

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "observation.state" not in obs:
            raise ValueError("Missing `observation.state`")
        image_key = "observation.images.head_cam_h"
        if image_key not in obs:
            raise ValueError(f"Missing `{image_key}`")
        parts = self._split_state(obs["observation.state"])
        state = {key: parts[key][None, None, :] for key in self.model.state_keys}
        image = _as_hwc_uint8(obs[image_key])
        video = {key: image[None, None, ...] for key in self.model.video_keys}
        language = {self.model.language_key: [[str(obs.get("prompt", ""))]]}
        return {"video": video, "state": state, "language": language}

    def _convert_action_chunk(self, action_dict: dict[str, np.ndarray]) -> np.ndarray:
        missing = [key for key in ACTION_LAYOUT if key not in action_dict]
        if missing:
            raise ValueError(f"GR00T wholebody output is missing keys: {missing}")
        arrays: list[np.ndarray] = []
        horizon: int | None = None
        for key, width in ACTION_LAYOUT.items():
            value = _to_numpy(action_dict[key])
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width:
                raise ValueError(
                    f"Expected action[{key}] [1,H,{width}], got {value.shape}"
                )
            if horizon is None:
                horizon = int(value.shape[1])
            elif value.shape[1] != horizon:
                raise ValueError("GR00T wholebody action keys have different horizons")
            arrays.append(value[0].astype(np.float32))
        chunk = np.concatenate(arrays, axis=1)
        if horizon is None or chunk.shape != (horizon, ACTION_DIM):
            raise ValueError(f"Converted wholebody chunk must be [H,32], got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("Converted wholebody chunk contains non-finite values")
        return chunk

    def _predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        output = self.model.infer(self._build_model_obs(obs))
        if not isinstance(output, dict):
            raise ValueError(f"Unexpected GR00T output type: {type(output)}")
        chunk = self._convert_action_chunk(output)
        if self.execution_horizon is not None:
            chunk = chunk[: self.execution_horizon]
        if not len(chunk):
            raise ValueError("GR00T returned an empty wholebody chunk")
        return chunk

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._pending_actions:
            chunk = self._predict_chunk(obs)
            self._pending_actions = [step.copy() for step in chunk]
        return self._pending_actions.pop(0)

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._predict_chunk(obs)
