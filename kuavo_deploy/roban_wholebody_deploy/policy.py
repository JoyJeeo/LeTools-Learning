"""LeRobot-local and standardized-server wholebody policy facade."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from kuavo_deploy.roban_wholebody_deploy.config import InferenceConfig


LOGGER = logging.getLogger("roban_wholebody.policy")


def _to_chunk_tensor(actions: Any):
    import torch

    if isinstance(actions, dict):
        if "action" not in actions:
            raise ValueError(f"Policy output is missing `action`: {list(actions)}")
        actions = actions["action"]
    tensor = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected policy chunk [H,D], got {tuple(tensor.shape)}")
    return tensor


class WholebodyPolicy:
    def __init__(self, config: InferenceConfig) -> None:
        import torch

        self.config = config
        self.is_server = config.policy_type == "client"
        if self.is_server:
            from kuavo_deploy.kuavo_service.client import PolicyClient

            self.policy = PolicyClient(
                host=config.server_host,
                port=config.server_port,
                task_prompt=config.task_prompt,
                api_token=config.api_token or None,
            )
            self.policy.reset()
            self.preprocessor = lambda obs: obs
            self.postprocessor = lambda action: action
        else:
            from kuavo_deploy.utils.policy_loader import load_native_policy_bundle

            device = torch.device(config.device)
            (
                self.policy,
                self.preprocessor,
                self.postprocessor,
                model_dir,
            ) = load_native_policy_bundle(
                pretrained_path=Path(config.pretrained_path),
                device=device,
                strict=True,
            )
            LOGGER.info("Loaded Roban wholebody policy from %s", model_dir)

    def reset(self) -> None:
        """Reset any model/server-side action queue before a fresh run session."""
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def _prepare_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.is_server:
            return self.preprocessor(observation)
        from kuavo_deploy.utils.policy_loader import inject_task_prompt

        return self.preprocessor(
            inject_task_prompt(observation, self.config.task_prompt)
        )

    def predict_chunk(self, observation: dict[str, Any]) -> np.ndarray:
        import torch

        processed = self._prepare_observation(observation)
        with torch.inference_mode():
            if hasattr(self.policy, "select_action_chunk"):
                actions = self.policy.select_action_chunk(processed)
            elif hasattr(self.policy, "predict_action_chunk"):
                actions = self.policy.predict_action_chunk(processed)
            else:
                actions = self.policy.select_action(processed)

        chunk = _to_chunk_tensor(actions)
        output: list[np.ndarray] = []
        for index in range(chunk.shape[0]):
            step = self.postprocessor(chunk[index : index + 1])
            step_tensor = _to_chunk_tensor(step)
            output.append(step_tensor[0].detach().cpu().numpy().astype(np.float32))
        result = np.stack(output, axis=0)
        if result.shape[1] != 32:
            raise ValueError(f"Wholebody policy must output [H,32], got {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError("Wholebody policy returned non-finite actions")
        return result
