"""Configuration model for the Roban data conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # Allows lightweight message-processing tests without Hydra.
    DictConfig = Any
    OmegaConf = None


# Roban ROS topic defaults. Only the camera topic remains user-configurable in
# YAML; robot-layout details belong to this platform-specific module.
JOINT_STATE_TOPIC = "/rt/joint_state"
HEFT_REFERENCE_TOPIC = "/rt/pico/heft_reference"
HAND_CMD_TOPIC = "/rt/hand_cmd"
HAND_STATE_TOPIC = "/rt/hand_state"

# q slice conventions are Python half-open ranges.
LEFT_LEG_SLICE = (0, 6)
RIGHT_LEG_SLICE = (6, 12)
WAIST_STATE_INDEX = 12
LEFT_ARM_SLICE = (13, 17)
RIGHT_ARM_SLICE = (17, 21)

WAIST_CMD_INDEX = 0
LEFT_LEG_CMD_SLICE = (1, 7)
RIGHT_LEG_CMD_SLICE = (7, 13)
LEFT_ARM_CMD_SLICE = (13, 17)
RIGHT_ARM_CMD_SLICE = (17, 21)

HAND_DOF_INDICES = (0, 6)
# Wholebody hand messages use their full range: 0=open, 100=closed.
HAND_OPEN_VALUES = (0.0, 0.0)
HAND_CLOSED_VALUES = (100.0, 100.0)

WHOLEBODY_STATE_NAMES = (
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_hand",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_hand",
    "left_leg_1",
    "left_leg_2",
    "left_leg_3",
    "left_leg_4",
    "left_leg_5",
    "left_leg_6",
    "right_leg_1",
    "right_leg_2",
    "right_leg_3",
    "right_leg_4",
    "right_leg_5",
    "right_leg_6",
    "waist",
)
WHOLEBODY_ACTION_NAMES = WHOLEBODY_STATE_NAMES + (
    "root_x",
    "root_y",
    "root_z",
    "root_rot6d_1",
    "root_rot6d_2",
    "root_rot6d_3",
    "root_rot6d_4",
    "root_rot6d_5",
    "root_rot6d_6",
)


def get_roban_feature_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return WHOLEBODY_STATE_NAMES, WHOLEBODY_ACTION_NAMES


@dataclass(frozen=True)
class RobanDataConfig:
    platform_type: str
    task_description: str
    train_hz: int
    main_timeline: str
    main_timeline_fps: int
    sample_drop: int
    resize_width: int
    resize_height: int
    camera_topic: str
    joint_state_topic: str
    heft_reference_topic: str
    hand_cmd_topic: str
    hand_state_topic: str
    left_arm_slice: tuple[int, int]
    right_arm_slice: tuple[int, int]
    hand_dof_indices: tuple[int, int]
    hand_open_values: tuple[float, float]
    hand_closed_values: tuple[float, float]


def load_roban_config(cfg: DictConfig) -> RobanDataConfig:
    """Validate and load the Roban section of a Hydra config."""
    if OmegaConf is None:
        raise ImportError("omegaconf is required to load Roban YAML configuration")
    platform_type = str(
        OmegaConf.select(cfg, "dataset.platform_type", default="roban")
    ).lower()
    if platform_type != "roban":
        raise ValueError(
            "Roban converter requires dataset.platform_type=roban, "
            f"got: {platform_type}"
        )
    train_hz = int(OmegaConf.select(cfg, "dataset.train_hz", default=10))
    main_timeline_fps = int(
        OmegaConf.select(cfg, "dataset.main_timeline_fps", default=30)
    )
    if train_hz <= 0 or main_timeline_fps <= 0:
        raise ValueError("dataset.train_hz and dataset.main_timeline_fps must be positive")
    if train_hz > main_timeline_fps:
        raise ValueError("dataset.train_hz cannot exceed dataset.main_timeline_fps")

    return RobanDataConfig(
        platform_type=platform_type,
        task_description=str(
            OmegaConf.select(cfg, "dataset.task_description", default="Pick and Place")
        ),
        train_hz=train_hz,
        main_timeline=str(
            OmegaConf.select(cfg, "dataset.main_timeline", default="head_cam_h")
        ),
        main_timeline_fps=main_timeline_fps,
        sample_drop=int(OmegaConf.select(cfg, "dataset.sample_drop", default=10)),
        resize_width=int(OmegaConf.select(cfg, "dataset.resize.width", default=640)),
        resize_height=int(OmegaConf.select(cfg, "dataset.resize.height", default=480)),
        camera_topic=str(
            OmegaConf.select(
                cfg,
                "dataset.topics.camera",
                default="/cam_h/color/image_raw/compressed",
            )
        ),
        joint_state_topic=JOINT_STATE_TOPIC,
        heft_reference_topic=HEFT_REFERENCE_TOPIC,
        hand_cmd_topic=HAND_CMD_TOPIC,
        hand_state_topic=HAND_STATE_TOPIC,
        left_arm_slice=LEFT_ARM_SLICE,
        right_arm_slice=RIGHT_ARM_SLICE,
        hand_dof_indices=HAND_DOF_INDICES,
        hand_open_values=HAND_OPEN_VALUES,
        hand_closed_values=HAND_CLOSED_VALUES,
    )
