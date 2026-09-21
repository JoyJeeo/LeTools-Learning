"""OpenPI transforms for the Roban wholebody LeRobot dataset.

The flat state/action layouts match
``kuavo_data/modality_templates/roban_wholebody_modality.json`` exactly.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_LAYOUT = (
    ("left_arm", 4),
    ("left_hand", 1),
    ("right_arm", 4),
    ("right_hand", 1),
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 1),
)
ACTION_LAYOUT = (*STATE_LAYOUT, ("root_xyz", 3), ("root_rot6d", 6))
STATE_DIM = sum(width for _, width in STATE_LAYOUT)
ACTION_DIM = sum(width for _, width in ACTION_LAYOUT)


def _parse_image(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Roban head image must be 3-D, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Roban head image must have three channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _require_last_dim(name: str, value: np.ndarray, expected: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[-1] != expected:
        raise ValueError(f"Roban {name} must end in dimension {expected}, got {array.shape}")
    return array


@dataclasses.dataclass(frozen=True)
class RobanWholebodyInputs(transforms.DataTransformFn):
    """Convert Roban's single-camera flat observation to OpenPI format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["head"])
        padded_camera = np.zeros_like(head_image)
        padded_camera_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        result = {
            "state": _require_last_dim("state", data["state"], STATE_DIM),
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": padded_camera,
                "right_wrist_0_rgb": padded_camera,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": padded_camera_mask,
                "right_wrist_0_rgb": padded_camera_mask,
            },
            "prompt": data["prompt"],
        }
        if "actions" in data:
            result["actions"] = _require_last_dim("actions", data["actions"], ACTION_DIM)
        return result


@dataclasses.dataclass(frozen=True)
class RobanWholebodyOutputs(transforms.DataTransformFn):
    """Remove any model padding while preserving the modality action order."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"OpenPI actions must contain at least {ACTION_DIM} dimensions, got {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
