from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import register_adapter
from .base import ModelServerAdapter, resolve_model_repo_root
from .xiaomi_robotics_0 import _as_rgb_image, _flat_state


class _XiaomiRuntime:
    def __init__(self, repo_root: Path, checkpoint: Path, processor_path: str, device: str) -> None:
        xr1_root = repo_root / "xr1"
        for path in (xr1_root, repo_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        import torch
        from PIL import Image
        from transformers import AutoProcessor

        from mibot.server.deploy import load_model, load_stats
        from mibot.server.runtime.client import Client
        from mibot.utils.action_layout import ActionLayout
        from mibot.utils.io import (
            compose_state,
            denormalize_action,
            normalize_quantile,
            resize_image,
        )

        self.torch = torch
        self.Image = Image
        self.Client = Client
        self.compose_state = compose_state
        self.denormalize_action = denormalize_action
        self.normalize_quantile = normalize_quantile
        self.resize_image = resize_image
        self.device = device

        config, self.model = load_model(str(checkpoint), device, vlm_path=processor_path)
        data = config.data.params.train_datasets
        self.layout = ActionLayout(data.modality) if data.get("modality") else None
        self.camera_keys = self.layout.camera_keys if self.layout else XiaomiRobotics1Adapter.camera_keys
        self.mean, self.std, self.q01, self.q99, self.action_mask = load_stats(config, device)
        self.q01_numpy = self.q01.cpu().numpy()
        self.q99_numpy = self.q99.cpu().numpy()
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.processor.tokenizer.padding_side = "right"

    def infer(
        self,
        state: np.ndarray,
        images: tuple[np.ndarray, np.ndarray, np.ndarray],
        prompt: str,
    ) -> np.ndarray:
        pictures = [
            self.resize_image(self.Image.fromarray(image), factor=32, max_pixels=160000)
            for image in images
        ]
        payload = self.processor.apply_chat_template(
            [self.Client._messages(prompt, *pictures)],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )
        if self.layout:
            packed_state = self.layout.pack_state(state)
        else:
            packed_state = self.compose_state(
                left_gripper=state[7],
                left_joint=state[:7],
                right_gripper=state[15],
                right_joint=state[8:15],
            )
        payload["state"] = self.torch.from_numpy(
            self.normalize_quantile(packed_state, self.q01_numpy, self.q99_numpy)
        )[None]
        batch = {
            key: value.to(self.device) if isinstance(value, self.torch.Tensor) else value
            for key, value in payload.items()
        }
        mask = self.action_mask.unsqueeze(0).expand(batch["input_ids"].shape[0], -1, -1)
        batch["action"] = self.torch.zeros(
            (batch["input_ids"].shape[0], *self.mean.shape),
            device=self.device,
            dtype=self.torch.bfloat16,
        )
        batch["action_mask"] = mask
        with self.torch.inference_mode():
            action = self.model.generate(batch)
            action = self.denormalize_action(action * mask, self.mean, self.std) * mask
        return action[0].float().cpu().numpy()


@register_adapter
class XiaomiRobotics1Adapter(ModelServerAdapter):
    name = "xiaomi_robotics_1"
    camera_keys = (
        "observation.images.head_cam_h",
        "observation.images.wrist_cam_l",
        "observation.images.wrist_cam_r",
    )

    def __init__(
        self,
        *,
        model_repo_root: str,
        checkpoint: str,
        processor_path: str,
        execution_horizon: int,
        device: str,
    ) -> None:
        self.model_repo_root = resolve_model_repo_root(self.name, model_repo_root)
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if execution_horizon < 0:
            raise ValueError("execution_horizon must be zero or positive")
        processor_dir = Path(processor_path).expanduser()
        self.processor_path = (
            str(processor_dir.resolve()) if processor_dir.exists() else processor_path
        )
        self.execution_horizon = execution_horizon
        self._pending_actions: list[np.ndarray] = []
        self.model = _XiaomiRuntime(
            self.model_repo_root, self.checkpoint, self.processor_path, device
        )

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--model_repo_root", type=str, default="")
        parser.add_argument("--checkpoint", type=str, required=True)
        parser.add_argument("--processor_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
        parser.add_argument("--execution_horizon", type=int, default=10)
        parser.add_argument("--device", type=str, default="cuda:0")

    @classmethod
    def from_args(cls, args: Namespace) -> "XiaomiRobotics1Adapter":
        return cls(
            model_repo_root=args.model_repo_root,
            checkpoint=args.checkpoint,
            processor_path=args.processor_path,
            execution_horizon=args.execution_horizon,
            device=args.device,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.name,
            "model_repo_root": str(self.model_repo_root),
            "checkpoint": str(self.checkpoint),
            "execution_horizon": self.execution_horizon,
            "action_representation": (
                self.model.layout.representation if self.model.layout else "xr1_eef_delta_60d"
            ),
            "state_dim": self.model.layout.state_source_dim if self.model.layout else 16,
            "action_dim": self.model.layout.action_source_dim if self.model.layout else 60,
        }

    def reset(self) -> dict[str, Any]:
        self._pending_actions.clear()
        return {"status": "ok", "message": "adapter action cache cleared"}

    def _predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        state_dim = self.model.layout.state_source_dim if self.model.layout else 16
        state = _flat_state(obs["observation.state"], state_dim)
        images = tuple(_as_rgb_image(obs[key]) for key in self.model.camera_keys)
        raw = self.model.infer(state, images, str(obs.get("prompt", "")))
        chunk = self.model.layout.recover_action(raw, state) if self.model.layout else raw
        return chunk[: self.execution_horizon] if self.execution_horizon > 0 else chunk

    def select_action(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._pending_actions:
            self._pending_actions.extend(self._predict_chunk(obs))
        return self._pending_actions.pop(0)

    def select_action_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        self._pending_actions.clear()
        return self._predict_chunk(obs)
