"""Roban message processing and ROS bag reading.

The generic chunked time aligner and the existing Kuavo color-image decoder
are reused. Only Roban-specific topic and field semantics live here.
"""

from __future__ import annotations

import glob
import os
from typing import Callable, Optional

import numpy as np

from kuavo_data.common import kuavo_dataset as kuavo_common
from kuavo_data.common.chunk_process import ChunkedRosbagProcessor
from kuavo_data.roban_data.config import (
    LEFT_ARM_CMD_SLICE,
    LEFT_ARM_SLICE,
    LEFT_LEG_CMD_SLICE,
    LEFT_LEG_SLICE,
    RIGHT_ARM_CMD_SLICE,
    RIGHT_ARM_SLICE,
    RIGHT_LEG_CMD_SLICE,
    RIGHT_LEG_SLICE,
    WAIST_CMD_INDEX,
    WAIST_STATE_INDEX,
    RobanDataConfig,
)


CAMERA_KEY = "head_cam_h"


def _aligned_array(aligned_frame: dict, key: str, expected: int) -> np.ndarray:
    item = aligned_frame.get(key)
    values = np.asarray(item.get("data", []) if item else [], dtype=np.float32)
    if values.size != expected:
        raise ValueError(
            f"Aligned field `{key}` must be {expected}-D, got {values.size}"
        )
    return values


def build_wholebody_state_action(
    aligned_frame: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the wholebody 23-D state and 32-D action in policy order."""

    joint_state = _aligned_array(aligned_frame, "observation.joints", 21)
    state_hands = _aligned_array(aligned_frame, "observation.gripper", 2)
    # action.wholebody stores source q[0:21], root_xyz and root_rot6d.
    wholebody_action = _aligned_array(aligned_frame, "action.wholebody", 30)
    action_hands = _aligned_array(aligned_frame, "action.gripper", 2)
    joint_cmd = wholebody_action[:21]
    root_xyz_rot6d = wholebody_action[21:]

    state = np.concatenate(
        (
            joint_state[slice(*LEFT_ARM_SLICE)],
            state_hands[:1],
            joint_state[slice(*RIGHT_ARM_SLICE)],
            state_hands[1:],
            joint_state[slice(*LEFT_LEG_SLICE)],
            joint_state[slice(*RIGHT_LEG_SLICE)],
            joint_state[WAIST_STATE_INDEX : WAIST_STATE_INDEX + 1],
        )
    ).astype(np.float32)
    action = np.concatenate(
        (
            joint_cmd[slice(*LEFT_ARM_CMD_SLICE)],
            action_hands[:1],
            joint_cmd[slice(*RIGHT_ARM_CMD_SLICE)],
            action_hands[1:],
            joint_cmd[slice(*LEFT_LEG_CMD_SLICE)],
            joint_cmd[slice(*RIGHT_LEG_CMD_SLICE)],
            joint_cmd[WAIST_CMD_INDEX : WAIST_CMD_INDEX + 1],
            root_xyz_rot6d,
        )
    ).astype(np.float32)
    return state, action


def quaternion_wxyz_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to the first two rows of its rotation matrix."""

    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"root_quat must be 4-D wxyz, got {quat.size}")
    if not np.isfinite(quat).all():
        raise ValueError("root_quat contains a non-finite value")
    norm = float(np.linalg.norm(quat))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("root_quat must have a non-zero norm")
    w, x, y, z = quat / norm
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
        ],
        dtype=np.float32,
    )
    return rotation.reshape(6)


class RobanMsgProcessor:
    def __init__(self, config: RobanDataConfig):
        self.config = config

    @staticmethod
    def _timestamp(msg) -> float:
        try:
            return float(msg.header.stamp.to_sec())
        except AttributeError:
            # ChunkedRosbagProcessor replaces this value with the bag timestamp.
            return 0.0

    @staticmethod
    def _field(msg, name: str):
        if not hasattr(msg, name):
            raise ValueError(
                f"Message {type(msg).__name__} is missing required field `{name}`"
            )
        return getattr(msg, name)

    def process_joint_state_wholebody(self, msg) -> dict:
        q = np.asarray(self._field(msg, "q"), dtype=np.float32).reshape(-1)
        if q.size < 21:
            raise ValueError(
                f"/rt/joint_state.q needs at least 21 values, got {q.size}"
            )
        return {"data": q[:21], "timestamp": self._timestamp(msg)}

    @staticmethod
    def _vector(values, field_name: str, names: tuple[str, ...]) -> np.ndarray:
        if all(hasattr(values, name) for name in names):
            values = [getattr(values, name) for name in names]
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.size != len(names):
            raise ValueError(
                f"{field_name} must be {len(names)}-D "
                f"({''.join(names)}), got {array.size}"
            )
        return array

    def process_heft_reference(self, msg) -> dict:
        q = np.asarray(self._field(msg, "q"), dtype=np.float32).reshape(-1)
        if q.size < 21:
            raise ValueError(
                f"/rt/pico/heft_reference.q needs at least 21 values, got {q.size}"
            )
        root_xyz = self._vector(
            self._field(msg, "root_pos"), "root_pos", ("x", "y", "z")
        )
        root_wxyz = self._vector(
            self._field(msg, "root_quat"), "root_quat", ("w", "x", "y", "z")
        )
        root_rot6d = quaternion_wxyz_to_rotation6d(root_wxyz)
        data = np.concatenate((q[:21], root_xyz, root_rot6d)).astype(np.float32)
        return {"data": data, "timestamp": self._timestamp(msg)}

    def process_hand_cmd(self, msg) -> dict:
        return self._process_hand_field(msg, "data")

    def process_hand_state(self, msg) -> dict:
        return self._process_hand_field(msg, "position")

    def _process_hand_field(self, msg, field_name: str) -> dict:
        values = np.asarray(self._field(msg, field_name), dtype=np.float32).reshape(-1)
        indices = np.asarray(self.config.hand_dof_indices, dtype=np.int64)
        if values.size <= int(indices.max()):
            raise ValueError(
                f"Hand data needs index {int(indices.max())}, got {values.size} values"
            )

        selected = values[indices]
        opened = np.asarray(self.config.hand_open_values, dtype=np.float32)
        closed = np.asarray(self.config.hand_closed_values, dtype=np.float32)
        denominator = closed - opened
        if np.any(np.isclose(denominator, 0.0)):
            raise ValueError("Hand open_values and closed_values must differ")

        # 0 means fully open and 1 means fully closed for both hands. Values
        # outside the declared physical range are clipped to a valid command.
        normalized = np.clip((selected - opened) / denominator, 0.0, 1.0)
        return {
            "data": normalized.astype(np.float32),
            "timestamp": self._timestamp(msg),
        }


class RobanRosbagReader:
    """Read Roban topics while reusing the repository's chunked aligner."""

    def __init__(self, config: RobanDataConfig):
        self.config = config
        self._msg_processor = RobanMsgProcessor(config)

        # process_color_image reads these existing resize settings. Assigning
        # them here lets Roban reuse the tested decoder without changing it.
        kuavo_common.RESIZE_W = config.resize_width
        kuavo_common.RESIZE_H = config.resize_height
        color_processor = kuavo_common.KuavoMsgProcesser.process_color_image

        shared_topic_map = {
            CAMERA_KEY: {
                "topic": config.camera_topic,
                "msg_process_fn": color_processor,
            },
        }
        self._topic_process_map = {
            **shared_topic_map,
            **self._build_wholebody_topic_map(),
        }

    def _build_wholebody_topic_map(self) -> dict:
        return {
            "observation.joints": {
                "topic": self.config.joint_state_topic,
                "msg_process_fn": self._msg_processor.process_joint_state_wholebody,
            },
            "observation.gripper": {
                "topic": self.config.hand_state_topic,
                "msg_process_fn": self._msg_processor.process_hand_state,
            },
            "action.wholebody": {
                "topic": self.config.heft_reference_topic,
                "msg_process_fn": self._msg_processor.process_heft_reference,
            },
            "action.gripper": {
                "topic": self.config.hand_cmd_topic,
                "msg_process_fn": self._msg_processor.process_hand_cmd,
            },
        }

    @staticmethod
    def list_bag_files(bag_dir: str) -> list[str]:
        return sorted(glob.glob(os.path.join(str(bag_dir), "*.bag")))

    def process_rosbag_chunked(
        self,
        bag_file: str,
        frame_callback: Callable[[dict, int], None],
        chunk_size: int = 100,
        save_callback: Optional[Callable[[], None]] = None,
    ) -> int:
        processor = ChunkedRosbagProcessor(
            msg_processer=self._msg_processor,
            topic_process_map=self._topic_process_map,
            camera_names=[CAMERA_KEY],
            train_hz=self.config.train_hz,
            main_timeline=self.config.main_timeline,
            main_timeline_fps=self.config.main_timeline_fps,
            sample_drop=self.config.sample_drop,
        )
        _, main_timestamps, all_timestamps = processor.scan_timestamps_only(bag_file)
        return processor.process_in_chunks(
            bag_file=bag_file,
            main_timestamps=main_timestamps,
            all_timestamps=all_timestamps,
            frame_callback=frame_callback,
            chunk_size=chunk_size,
            save_callback=save_callback,
        )
