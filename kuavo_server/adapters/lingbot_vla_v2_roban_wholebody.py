"""LingBot-VLA v2 server adapter for Roban wholebody 23-D state / 32-D action."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import register_adapter
from .lingbot_vla_v2 import (
    LingBotVlaV2Adapter,
    _as_hwc_uint8,
    _resolve_lingbot_root,
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

# FeatureTransform.unapply may return either the reconstructed flat `action`
# tensor or the unified LingBot joint keys used during training.
SPLIT_ACTION_KEYS = (
    ("action.arm.position", 8),
    ("action.effector.position", 2),
    ("action.leg.position", 12),
    ("action.waist.position", 1),
    ("action.root_xyz.position", 3),
    ("action.root_rot6d.position", 6),
)


def _resolve_training_config_path(lingbot_root: Path, training_config_path: str) -> str:
    if training_config_path:
        return training_config_path
    default_path = lingbot_root / "configs" / "vla" / "roban_wholebody" / "roban_wholebody.yaml"
    if not default_path.is_file():
        raise FileNotFoundError(
            f"Default Roban wholebody LingBot-VLA v2 config not found at: {default_path}. "
            "Please provide `--training_config_path`."
        )
    return str(default_path)


def _resolve_robot_config_path(robot_config_path: str) -> str:
    if robot_config_path:
        return robot_config_path
    return str(
        Path(__file__).resolve().parents[1] / "configs" / "lingbotvla_v2" / "roban_wholebody.yaml"
    )


@register_adapter
class LingBotVlaV2RobanWholebodyAdapter(LingBotVlaV2Adapter):
    """Serve Roban wholebody LingBot-VLA v2 through the standardized chunk API."""

    name = "lingbot_vla_v2_roban_wholebody"

    def __init__(
        self,
        *,
        checkpoint: str,
        model_repo_root: str,
        execution_horizon: int,
        qwen3vl_path: str,
        training_config_path: str,
        robot_config_path: str,
        robot_norm_path: str,
        use_compile: bool,
        use_fp32: bool,
    ) -> None:
        lingbot_root = _resolve_lingbot_root(model_repo_root)
        super().__init__(
            checkpoint=checkpoint,
            model_repo_root=str(lingbot_root),
            which_arm="both",
            execution_horizon=execution_horizon,
            qwen3vl_path=qwen3vl_path,
            training_config_path=_resolve_training_config_path(lingbot_root, training_config_path),
            robot_config_path=_resolve_robot_config_path(robot_config_path),
            robot_norm_path=robot_norm_path,
            use_compile=use_compile,
            use_fp32=use_fp32,
        )
        self.which_arm = "wholebody"

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--checkpoint",
            type=str,
            required=True,
            help="Path to a LingBot-VLA v2 hf_ckpt/model dir",
        )
        parser.add_argument(
            "--model_repo_root",
            type=str,
            default="",
            help="Path to LingBot-VLA v2 source checkout. Defaults to kuavo_model/external_models/lingbot-vla-v2.",
        )
        parser.add_argument(
            "--robot_norm_path",
            type=str,
            required=True,
            help="Path to Roban wholebody norm stats JSON",
        )
        parser.add_argument(
            "--training_config_path",
            type=str,
            default="",
            help="LingBot-VLA v2 YAML; defaults to configs/vla/roban_wholebody/roban_wholebody.yaml.",
        )
        parser.add_argument(
            "--robot_config_path",
            type=str,
            default="",
            help="Roban wholebody feature mapping YAML; defaults to kuavo_server/configs/lingbotvla_v2/roban_wholebody.yaml.",
        )
        parser.add_argument(
            "--qwen3vl_path",
            type=str,
            default="",
            help="Optional local Qwen3-VL tokenizer/model dir. If empty, uses training_config_path model.tokenizer_path.",
        )
        parser.add_argument(
            "--execution_horizon",
            type=int,
            default=50,
            help="-1 returns the model's full chunk. Default 50 matches the 50 Hz wholebody chunk.",
        )
        parser.add_argument("--use_compile", action="store_true")
        parser.add_argument(
            "--use_fp32",
            action="store_true",
            help="Use fp32 inference instead of bf16",
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "LingBotVlaV2RobanWholebodyAdapter":
        return cls(
            checkpoint=args.checkpoint,
            model_repo_root=args.model_repo_root,
            execution_horizon=args.execution_horizon,
            qwen3vl_path=args.qwen3vl_path,
            training_config_path=args.training_config_path,
            robot_config_path=args.robot_config_path,
            robot_norm_path=args.robot_norm_path,
            use_compile=args.use_compile,
            use_fp32=args.use_fp32,
        )

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        meta.update(
            {
                "adapter": self.name,
                "platform_type": "roban",
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
            }
        )
        meta.pop("which_arm", None)
        return meta

    def _build_model_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "observation.state" not in obs:
            raise ValueError("Missing `observation.state`")
        image_key = "observation.images.head_cam_h"
        if image_key not in obs:
            raise ValueError(f"Missing `{image_key}`")
        state = _to_numpy(obs["observation.state"]).astype(np.float32).reshape(-1)
        if state.size != STATE_DIM:
            raise ValueError(f"Roban wholebody state must be {STATE_DIM}-D, got {state.size}")
        return {
            image_key: _as_hwc_uint8(obs[image_key]),
            "observation.state": state,
            "task": str(obs.get("prompt", "")),
        }

    def _convert_action(self, action: Any) -> np.ndarray:
        result = _to_numpy(action).astype(np.float32).reshape(-1)
        if result.size != ACTION_DIM:
            raise ValueError(
                f"Roban wholebody LingBot-VLA v2 action must be {ACTION_DIM}-D, got {result.size}"
            )
        if not np.isfinite(result).all():
            raise ValueError("Roban wholebody LingBot-VLA v2 action contains non-finite values")
        return result

    def _compose_action_from_dict(self, out: dict[str, Any]) -> Any:
        if "action" in out:
            return out["action"]

        missing = [key for key, _ in SPLIT_ACTION_KEYS if key not in out]
        if missing:
            raise ValueError(
                "Unexpected LingBot-VLA v2 wholebody output keys: "
                f"{list(out)}; missing {missing}"
            )

        arrays: list[np.ndarray] = []
        horizon: int | None = None
        for key, width in SPLIT_ACTION_KEYS:
            value = _to_numpy(out[key])
            if value.ndim == 1:
                value = value[None, :]
            if value.ndim != 2 or value.shape[-1] != width:
                raise ValueError(
                    f"Expected {key} [H,{width}] or [{width}], got {value.shape}"
                )
            if horizon is None:
                horizon = int(value.shape[0])
            elif value.shape[0] != horizon:
                raise ValueError("LingBot-VLA v2 wholebody action keys have different horizons")
            arrays.append(value.astype(np.float32))
        return np.concatenate(arrays, axis=-1)
