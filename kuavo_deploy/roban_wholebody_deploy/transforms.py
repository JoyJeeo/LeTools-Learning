"""Roban wholebody observation, policy-action and HEFT conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


POLICY_STATE_DIM = 23
POLICY_ACTION_DIM = 32
HEFT_Q_DIM = 21


@dataclass(frozen=True)
class ActionFrame:
    heft_q: np.ndarray
    root_xyz: np.ndarray
    root_wxyz: np.ndarray
    left_gripper: np.ndarray
    right_gripper: np.ndarray


def build_policy_state(joint_q: Any, hand_position: Any) -> np.ndarray:
    joints = np.asarray(joint_q, dtype=np.float32).reshape(-1)
    hands = np.asarray(hand_position, dtype=np.float32).reshape(-1)
    if joints.size < 21:
        raise ValueError(f"Roban joint state must contain at least 21 values, got {joints.size}")
    if hands.size != 12:
        raise ValueError(f"Roban hand state must contain 12 values, got {hands.size}")
    hand_pair = np.clip(hands[[0, 6]] / 100.0, 0.0, 1.0)
    state = np.concatenate(
        (
            joints[13:17],
            hand_pair[0:1],
            joints[17:21],
            hand_pair[1:2],
            joints[0:6],
            joints[6:12],
            joints[12:13],
        )
    ).astype(np.float32)
    if state.shape != (POLICY_STATE_DIM,):
        raise RuntimeError(f"Wholebody policy state must be 23-D, got {state.shape}")
    return state


def _to_action_chunk(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "action" not in value:
            raise ValueError(f"Policy output dict is missing `action`: {list(value)}")
        value = value["action"]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[1] != POLICY_ACTION_DIM:
        raise ValueError(f"Wholebody action chunk must be [H,32], got {chunk.shape}")
    if chunk.shape[0] == 0 or not np.isfinite(chunk).all():
        raise ValueError("Wholebody action chunk is empty or contains non-finite values")
    return chunk


def rotation6d_rows_to_wxyz(value: Any) -> np.ndarray:
    """Invert roban_data's first-two-rows rotation6d convention."""
    rows = np.asarray(value, dtype=np.float64).reshape(-1)
    if rows.size != 6 or not np.isfinite(rows).all():
        raise ValueError("root rotation6d must be a finite 6-vector")
    row1 = rows[:3]
    row2 = rows[3:]
    norm1 = float(np.linalg.norm(row1))
    if norm1 < 1e-8:
        raise ValueError("root rotation6d first row has zero norm")
    row1 = row1 / norm1
    row2 = row2 - float(np.dot(row1, row2)) * row1
    norm2 = float(np.linalg.norm(row2))
    if norm2 < 1e-8:
        raise ValueError("root rotation6d rows are collinear")
    row2 = row2 / norm2
    rotation = np.row_stack((row1, row2, np.cross(row1, row2)))
    return _rotation_matrix_to_wxyz(rotation)


def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        next_axis = (axis + 1) % 3
        last_axis = (axis + 2) % 3
        scale = math.sqrt(
            max(
                0.0,
                1.0
                + rotation[axis, axis]
                - rotation[next_axis, next_axis]
                - rotation[last_axis, last_axis],
            )
        ) * 2.0
        if scale < 1e-12:
            raise ValueError("rotation matrix cannot be converted to a quaternion")
        xyz = np.zeros(3, dtype=np.float64)
        xyz[axis] = 0.25 * scale
        xyz[next_axis] = (
            rotation[axis, next_axis] + rotation[next_axis, axis]
        ) / scale
        xyz[last_axis] = (
            rotation[axis, last_axis] + rotation[last_axis, axis]
        ) / scale
        w = (rotation[last_axis, next_axis] - rotation[next_axis, last_axis]) / scale
        quat = np.concatenate(([w], xyz))
    quat /= float(np.linalg.norm(quat))
    if quat[0] < 0.0:
        quat = -quat
    return quat


def parse_action_chunk(value: Any) -> list[ActionFrame]:
    chunk = _to_action_chunk(value)
    frames: list[ActionFrame] = []
    previous_quat: np.ndarray | None = None
    for action in chunk:
        # Policy: LArm4,LHand1,RArm4,RHand1,LLeg6,RLeg6,Waist1,XYZ3,Rot6D6.
        root_wxyz = rotation6d_rows_to_wxyz(action[26:32])
        if previous_quat is not None and np.dot(previous_quat, root_wxyz) < 0.0:
            root_wxyz = -root_wxyz
        previous_quat = root_wxyz
        heft_q = np.concatenate(
            (
                action[22:23],
                action[10:16],
                action[16:22],
                action[0:4],
                action[5:9],
            )
        ).astype(np.float64)
        frames.append(
            ActionFrame(
                heft_q=heft_q,
                root_xyz=action[23:26].astype(np.float64),
                root_wxyz=root_wxyz,
                left_gripper=action[4:5].astype(np.float64),
                right_gripper=action[9:10].astype(np.float64),
            )
        )
    return frames


def expand_hand_command(action: ActionFrame, scale: float = 100.0) -> np.ndarray:
    left = float(np.clip(action.left_gripper[0], 0.0, 1.0)) * scale
    right = float(np.clip(action.right_gripper[0], 0.0, 1.0)) * scale
    return np.asarray([left] * 6 + [right] * 6, dtype=np.float64)
