# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers.utils import logging

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.action_layout import ActionLayout
from mibot.utils.io import normalize_action, normalize_quantile, validate_quantiles, validate_stats

logger = logging.get_logger(__name__)


class XR1LeRobotDataset(JsonDataset):
    """Config-driven LeRobot dataset for XR1 post-training."""

    def __init__(self, params):
        data = params["train_datasets"]
        self.layout = ActionLayout(data["modality"])
        self.action_length = int(data.get("action_length", params.get("action_length", 30)))
        self.batch_size = int(data.get("batch_size", 16))
        self.max_samples = (
            int(params.get("max_steps", 1000))
            * self.batch_size
            * int(os.environ.get("WORLD_SIZE", 1))
        )
        self.mean, self.std = validate_stats(data["mean"], data["std"], self.action_length)
        self.q01, self.q99 = validate_quantiles(data["q01"], data["q99"])

        root = Path(data["root"]).expanduser().resolve()
        repo_id = str(data.get("repo_id") or f"lerobot/{root.name}")
        fps = int(data["fps"])
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            delta_timestamps={
                self.layout.action_key: [step / fps for step in range(self.action_length)]
            },
            video_backend=str(data.get("video_backend", "pyav")),
        )
        if self.dataset.fps != fps:
            raise ValueError(f"Configured fps={fps}, dataset fps={self.dataset.fps}")
        features = {self.layout.state_key, self.layout.action_key, *self.layout.camera_keys}
        missing = features - set(self.dataset.features)
        if missing:
            raise ValueError(f"LeRobot dataset is missing features: {sorted(missing)}")
        if not len(self.dataset):
            raise ValueError(f"LeRobot dataset is empty: {root}")
        self.num_samples = max(len(self.dataset), self.max_samples)
        logger.info(f"LeRobot samples: {len(self.dataset)}; training samples: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        item = self.dataset[index % len(self.dataset)]
        state = self._numpy(item[self.layout.state_key])
        future = self._numpy(item[self.layout.action_key])
        is_pad = self._numpy(item[f"{self.layout.action_key}_is_pad"]).astype(bool)
        steps = int((~is_pad).sum())
        action = self.layout.pack_action(state, future)
        action = normalize_action(action, self.mean, self.std)
        action_mask = self.layout.action_mask(self.action_length, ~is_pad)
        conversations = [
            {
                "from": "human",
                "value": "<image>" * len(self.layout.camera_keys) + f"\n{item['task']} /no_cot",
            },
            {"from": "gpt", "value": "<cot></cot>"},
            {"from": "human", "value": "Robot state: <state>"},
            {
                "from": "gpt",
                "value": "".join(f"<a_{step}>" for step in range(steps)) + "<score>",
            },
        ]
        images = [self._to_pil(item[key]) for key in self.layout.camera_keys]

        return {
            "messages": self._messages(conversations, self._augment(images)),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "state": torch.from_numpy(
                normalize_quantile(self.layout.pack_state(state), self.q01, self.q99)
            ),
            "vlm_action_target": torch.from_numpy(action[:steps]),
            "vlm_action_mask": torch.from_numpy(action_mask[:steps]),
            "vlm_action_actual_length": steps,
        }

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    @classmethod
    def _to_pil(cls, value):
        image = cls._numpy(value)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)
        if np.issubdtype(image.dtype, np.floating):
            image = image * 255.0 if image.max(initial=0) <= 1.0 else image
        image = image.clip(0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        from PIL import Image

        return Image.fromarray(image)
