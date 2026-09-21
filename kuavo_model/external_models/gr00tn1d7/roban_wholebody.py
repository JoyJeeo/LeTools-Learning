"""GR00T modality registration for the Roban wholebody dataset."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


STATE_KEYS = [
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
    "left_leg",
    "right_leg",
    "waist",
]
ACTION_KEYS = STATE_KEYS + ["root_xyz", "root_rot6d"]

roban_wholebody_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["head"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
    "action": ModalityConfig(
        # Fifty samples represent one second at the HEFT control frequency.
        delta_indices=list(range(40)),
        modality_keys=ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    roban_wholebody_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
