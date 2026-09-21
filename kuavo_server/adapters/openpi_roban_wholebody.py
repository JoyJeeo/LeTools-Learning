"""OpenPI server adapter for Roban wholebody 23-D state / 32-D action."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import register_adapter
from .openpi import (
    OpenPiJaxLejuAdapter,
    _as_hwc_uint8,
    _OpenPiRuntime,
    _resolve_repo_root,
    _to_numpy,
)


STATE_DIM = 23
ACTION_DIM = 32
ROBAN_REPACK_STRUCTURE = {
    "head": "observation.images.head_cam_h",
    "state": "observation.state",
    "prompt": "prompt",
}


@register_adapter
class OpenPiRobanWholebodyAdapter(OpenPiJaxLejuAdapter):
    """Serve ``pi0_roban_wholebody`` through the standardized chunk API."""

    name = "openpi_roban_wholebody"

    def __init__(
        self,
        *,
        checkpoint: str,
        model_repo_root: str,
        policy_config_name: str,
        execution_horizon: int,
        device: str,
        asset_id: str,
    ) -> None:
        self.model_repo_root = _resolve_repo_root(model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint dir does not exist: {self.checkpoint}")
        if execution_horizon < 0:
            raise ValueError("execution_horizon must be non-negative")

        self.policy_config_name = policy_config_name
        self.which_arm = "wholebody"
        self.device = device
        self.asset_id = asset_id or self._detect_asset_id(self.checkpoint)
        self.expected_state_dim = self._detect_norm_dim(
            self.checkpoint, self.asset_id, "state"
        )
        self.expected_action_dim = self._detect_norm_dim(
            self.checkpoint, self.asset_id, "actions"
        )
        if self.expected_state_dim not in (None, STATE_DIM):
            raise ValueError(
                f"Roban wholebody checkpoint state stats must be {STATE_DIM}-D, "
                f"got {self.expected_state_dim}"
            )
        if self.expected_action_dim not in (None, ACTION_DIM):
            raise ValueError(
                f"Roban wholebody checkpoint action stats must be {ACTION_DIM}-D, "
                f"got {self.expected_action_dim}"
            )
        self._pending_actions: list[np.ndarray] = []
        self.model = _OpenPiRuntime(
            repo_root=self.model_repo_root,
            policy_config_name=policy_config_name,
            checkpoint_dir=self.checkpoint,
            execution_horizon=execution_horizon,
            pytorch_device=device,
            asset_id=self.asset_id,
            repack_structure=ROBAN_REPACK_STRUCTURE,
        )

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--model_repo_root", type=str, default="")
        parser.add_argument("--checkpoint", type=str, required=True)
        parser.add_argument(
            "--policy_config_name", type=str, default="pi0_roban_wholebody"
        )
        parser.add_argument(
            "--execution_horizon",
            type=int,
            default=0,
            help="0 uses the model's complete 50-step action chunk.",
        )
        parser.add_argument("--device", type=str, default="")
        parser.add_argument("--asset_id", type=str, default="")

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenPiRobanWholebodyAdapter":
        return cls(
            checkpoint=args.checkpoint,
            model_repo_root=args.model_repo_root,
            policy_config_name=args.policy_config_name,
            execution_horizon=args.execution_horizon,
            device=args.device,
            asset_id=args.asset_id,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "platform_type": "roban",
            "model_repo_root": str(self.model_repo_root),
            "checkpoint": str(self.checkpoint),
            "policy_config_name": self.policy_config_name,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "execution_horizon": self.model.execution_horizon,
            "asset_id": self.asset_id,
        }

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
            "prompt": str(obs.get("prompt", "")),
        }

    def _convert_action(self, action: Any) -> np.ndarray:
        result = _to_numpy(action).astype(np.float32).reshape(-1)
        if result.size != ACTION_DIM:
            raise ValueError(
                f"Roban wholebody OpenPI action must be {ACTION_DIM}-D, got {result.size}"
            )
        if not np.isfinite(result).all():
            raise ValueError("Roban wholebody OpenPI action contains non-finite values")
        return result
