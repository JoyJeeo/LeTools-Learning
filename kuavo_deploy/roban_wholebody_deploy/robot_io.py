"""ROS/DDS I/O used by the Roban wholebody deployment runner."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np


LOGGER = logging.getLogger("roban_wholebody.io")

HEFT_TOPIC = "/rt/pico/heft_reference"
JOINT_STATE_TOPIC = "/rt/joint_state"
HAND_STATE_TOPIC = "/rt/hand_state"
HAND_CMD_TOPIC = "/rt/hand_cmd"

# /rt/joint_state order: left leg 6, right leg 6, waist 1,
# left arm 4, right arm 4. Any following head joints are intentionally ignored.
WHOLEBODY_JOINT_COUNT = 21


def _wall_stamp() -> tuple[int, int]:
    now = time.time()
    sec = int(now)
    return sec, int((now - sec) * 1e9)


class LatestInputs:
    """Thread-safe latest camera, joint-state and hand-state samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._camera: dict[str, Any] | None = None
        self._joint: dict[str, Any] | None = None
        self._hand: dict[str, Any] | None = None

    def set_camera(self, image: np.ndarray, compressed: bytes, ros_stamp: float) -> None:
        with self._lock:
            self._camera = {
                "image_bgr": image,
                "image_compressed": compressed,
                "source_stamp": ros_stamp,
                "recv_monotonic": time.monotonic(),
            }

    def set_joint(self, msg: Any) -> None:
        if len(msg.q) < WHOLEBODY_JOINT_COUNT:
            LOGGER.warning(
                "Dropping joint state: q has %d elements, need at least %d",
                len(msg.q),
                WHOLEBODY_JOINT_COUNT,
            )
            return
        with self._lock:
            self._joint = {
                "position": np.asarray(msg.q[:WHOLEBODY_JOINT_COUNT], dtype=np.float64),
                "velocity": np.asarray(msg.v[:WHOLEBODY_JOINT_COUNT], dtype=np.float64),
                "acceleration": np.asarray(msg.vd[:WHOLEBODY_JOINT_COUNT], dtype=np.float64),
                "effort": np.asarray(msg.tau[:WHOLEBODY_JOINT_COUNT], dtype=np.float64),
                "source_stamp": float(msg.header_sec)
                + float(msg.header_nanosec) * 1e-9,
                "recv_monotonic": time.monotonic(),
            }

    def set_hand(self, msg: Any) -> None:
        if len(msg.position) != 12:
            LOGGER.warning(
                "Dropping hand state: position has %d elements, need 12",
                len(msg.position),
            )
            return
        with self._lock:
            self._hand = {
                "left_valid": bool(msg.left_valid),
                "right_valid": bool(msg.right_valid),
                "left_sample_age_ms": int(msg.left_sample_age_ms),
                "right_sample_age_ms": int(msg.right_sample_age_ms),
                "position": np.asarray(msg.position, dtype=np.float64),
                "velocity": np.asarray(msg.velocity, dtype=np.float64),
                "current": np.asarray(msg.current, dtype=np.float64),
                "state": np.asarray(msg.state, dtype=np.uint8),
                "source_stamp": float(msg.header_sec)
                + float(msg.header_nanosec) * 1e-9,
                "recv_monotonic": time.monotonic(),
            }

    def snapshot(self, camera_max_age: float, state_max_age: float) -> dict[str, Any]:
        with self._lock:
            camera = None if self._camera is None else dict(self._camera)
            joint = None if self._joint is None else dict(self._joint)
            hand = None if self._hand is None else dict(self._hand)
        if camera is None or joint is None or hand is None:
            raise RuntimeError("waiting for camera, joint state, and hand state")
        now = time.monotonic()
        if now - camera["recv_monotonic"] > camera_max_age:
            raise RuntimeError("camera sample is stale")
        if now - joint["recv_monotonic"] > state_max_age:
            raise RuntimeError("joint state is stale")
        if now - hand["recv_monotonic"] > state_max_age:
            raise RuntimeError("hand state is stale")
        return {"camera": camera, "joint_state": joint, "hand_state": hand}


class RosCompressedCamera:
    """Subscribe to a ROS1 sensor_msgs/CompressedImage topic."""

    def __init__(self, topic: str, store: LatestInputs) -> None:
        try:
            import cv2
            import rospy
            from sensor_msgs.msg import CompressedImage
        except ImportError as exc:
            raise RuntimeError(
                "ROS1 rospy/sensor_msgs and OpenCV are required for camera input"
            ) from exc

        if not rospy.core.is_initialized():
            rospy.init_node("roban_wholebody_deploy", anonymous=True, disable_signals=True)

        def callback(msg: Any) -> None:
            encoded = bytes(msg.data)
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.warning("Failed to decode compressed camera frame")
                return
            stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
            store.set_camera(image, encoded, stamp)

        self._subscriber = rospy.Subscriber(
            topic,
            CompressedImage,
            callback,
            queue_size=1,
            buff_size=16 * 1024 * 1024,
        )


class DdsIo:
    """Read Roban state topics and publish HEFT/hand commands."""

    def __init__(self, domain_id: int, store: LatestInputs) -> None:
        # Keep CycloneDDS imports lazy: runner.run() first applies CYCLONEDDS_URI,
        # and unit tests can exercise scheduling without a DDS installation.
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from cyclonedds.util import duration

        from kuavo_deploy.roban_wholebody_deploy.dds_types import (
            Float64Array,
            HandState,
            HeftReference,
            JointState,
        )

        self._float64_array_type = Float64Array
        self._heft_reference_type = HeftReference
        self._duration = duration
        self.participant = DomainParticipant(domain_id)
        # Lower-computer feedback writers offer BEST_EFFORT, so the readers
        # must not request RELIABLE QoS.
        state_qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.History.KeepLast(1),
        )
        self.joint_reader = DataReader(
            self.participant,
            Topic(self.participant, JOINT_STATE_TOPIC, JointState),
            qos=state_qos,
        )
        self.hand_reader = DataReader(
            self.participant,
            Topic(self.participant, HAND_STATE_TOPIC, HandState),
            qos=state_qos,
        )
        self.heft_writer = DataWriter(
            self.participant,
            Topic(self.participant, HEFT_TOPIC, HeftReference),
        )
        self.hand_writer = DataWriter(
            self.participant,
            Topic(self.participant, HAND_CMD_TOPIC, Float64Array),
        )
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._read_loop,
                args=(self.joint_reader, store.set_joint, "joint"),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_loop,
                args=(self.hand_reader, store.set_hand, "hand"),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def _read_loop(self, reader: Any, callback: Callable[[Any], None], name: str) -> None:
        while not self._stop.is_set():
            try:
                sample = reader.take_one(timeout=self._duration(seconds=0.1))
                if sample is not None:
                    callback(sample)
            except Exception:
                LOGGER.exception("%s DDS reader failed", name)
                time.sleep(0.1)

    def publish_heft(self, seq: int, epoch: int, action: Any) -> None:
        sec, nsec = _wall_stamp()
        self.heft_writer.write(
            self._heft_reference_type(
                header_sec=sec,
                header_nanosec=nsec,
                schema_version=1,
                robot_version="roban_s17",
                seq=seq,
                calibration_epoch=epoch,
                capture_timestamp_ms=int(time.time() * 1000),
                valid=True,
                calibrated=True,
                root_pos=action.root_xyz.tolist(),
                root_quat=action.root_wxyz.tolist(),
                q=action.heft_q.tolist(),
                qd=[],
            )
        )

    def publish_hand(self, hand_position: np.ndarray) -> None:
        sec, nsec = _wall_stamp()
        self.hand_writer.write(
            self._float64_array_type(
                header_sec=sec,
                header_nanosec=nsec,
                data=np.clip(hand_position, 0.0, 100.0).tolist(),
            )
        )

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=0.3)


def invoke_switch_cli(command: str, controller: str, library_path: str = "") -> bool:
    """Invoke the lejulab controller switch client with optional DDS libraries."""
    env = None
    if library_path:
        env = dict(os.environ)
        inherited = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{library_path}:{inherited}" if inherited else library_path
        )
    try:
        completed = subprocess.run(
            [command, controller],
            check=False,
            text=True,
            capture_output=True,
            timeout=5.0,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.error("Controller switch command failed: %s", exc)
        return False
    output = (completed.stdout + completed.stderr).strip()
    if output:
        LOGGER.info("Controller switch: %s", output)
    return completed.returncode == 0
