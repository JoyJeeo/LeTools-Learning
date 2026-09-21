#!/usr/bin/env python3
"""Run a Roban wholebody policy and stream actions to HEFT at 50 Hz."""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kuavo_deploy.roban_wholebody_deploy.action_buffer import ActionBuffer
from kuavo_deploy.roban_wholebody_deploy.config import (
    RobanWholebodyDeployConfig,
    load_roban_wholebody_deploy_config,
)
from kuavo_deploy.roban_wholebody_deploy.policy import WholebodyPolicy
from kuavo_deploy.roban_wholebody_deploy.transforms import (
    build_policy_state,
    expand_hand_command,
    parse_action_chunk,
)


LOGGER = logging.getLogger("roban_wholebody")


class RunControl:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.pause = threading.Event()
        # The worker may finish an old request after pause/resume. A monotonically
        # increasing generation lets it discard that result instead of putting it
        # back into an already-cleared action buffer.
        self.inference_enabled = threading.Event()
        self.inference_enabled.set()
        self._generation = 0

    def generation(self) -> int:
        return self._generation

    def toggle_pause(self) -> bool:
        if self.pause.is_set():
            self.pause.clear()
            paused = False
        else:
            self.pause.set()
            paused = True
        self._generation += 1
        return paused

    def request_stop(self) -> None:
        self._generation += 1
        self.stop.set()

    def install_signal_handlers(self) -> None:
        def toggle_pause(_signum, _frame) -> None:
            if not self.toggle_pause():
                LOGGER.info("Wholebody execution resumed")
            else:
                LOGGER.info("Wholebody execution pause requested")

        def request_stop(_signum, _frame) -> None:
            LOGGER.info("Wholebody execution stop requested")
            self.request_stop()

        signal.signal(signal.SIGUSR1, toggle_pause)
        signal.signal(signal.SIGUSR2, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)


def build_observation(
    snapshot: dict[str, Any], config: RobanWholebodyDeployConfig
) -> dict[str, Any]:
    import cv2
    import torch

    image_bgr = np.asarray(snapshot["camera"]["image_bgr"])
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Camera image must be HWC BGR, got {image_bgr.shape}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        (config.camera.width, config.camera.height),
        interpolation=cv2.INTER_LINEAR,
    )
    state = build_policy_state(
        snapshot["joint_state"]["position"],
        snapshot["hand_state"]["position"],
    )
    return {
        "observation.images.head_cam_h": (
            torch.from_numpy(np.ascontiguousarray(image_rgb))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .unsqueeze(0)
        ),
        "observation.state": torch.from_numpy(state).unsqueeze(0),
    }


def _snapshot(store, config: RobanWholebodyDeployConfig) -> dict[str, Any]:
    return store.snapshot(config.camera.max_age, config.dds.state_max_age)


def _wait_for_inputs(store, config: RobanWholebodyDeployConfig, stop) -> None:
    deadline = time.monotonic() + config.camera.startup_timeout
    while not stop.is_set():
        try:
            _snapshot(store, config)
            return
        except RuntimeError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for ROS/DDS inputs: {exc}") from exc
            time.sleep(0.05)


def _async_inference_loop(
    *,
    config: RobanWholebodyDeployConfig,
    store,
    policy: WholebodyPolicy,
    buffer: ActionBuffer,
    control: RunControl,
    errors: list[BaseException],
) -> None:
    execution = config.execution
    initialized = False
    observed_generation = control.generation()
    reset_pending = False
    try:
        while not control.stop.is_set():
            generation = control.generation()
            if generation != observed_generation:
                observed_generation = generation
                initialized = False
                reset_pending = True

            if control.pause.is_set() or not control.inference_enabled.is_set():
                time.sleep(0.02)
                continue

            if initialized:
                buffer.wait_until_at_most(execution.prefetch_steps, timeout=0.1)
                if (
                    control.stop.is_set()
                    or control.pause.is_set()
                    or not control.inference_enabled.is_set()
                    or control.generation() != observed_generation
                    or buffer.size() > execution.prefetch_steps
                ):
                    continue

            if reset_pending:
                policy.reset()
                reset_pending = False

            request_generation = observed_generation
            consumed_before = buffer.total_consumed()
            started = time.monotonic()
            observation = build_observation(_snapshot(store, config), config)
            chunk = policy.predict_chunk(observation)
            elapsed = time.monotonic() - started

            if (
                control.stop.is_set()
                or control.pause.is_set()
                or not control.inference_enabled.is_set()
                or control.generation() != request_generation
            ):
                initialized = False
                LOGGER.info(
                    "Discarded stale chunk from generation %d after control transition",
                    request_generation,
                )
                continue

            consumed_during_inference = max(
                0, buffer.total_consumed() - consumed_before
            )

            if not initialized:
                stats = buffer.replace(chunk)
                delay_steps = 0
                initialized = True
            else:
                measured_delay = math.ceil(elapsed * execution.action_hz)
                delay_steps = min(
                    max(consumed_during_inference, measured_delay),
                    execution.max_inference_delay_steps,
                )
                stats = buffer.merge(
                    chunk,
                    delay_steps=delay_steps,
                    overlap_steps=(
                        execution.blend_overlap_steps
                        if execution.blend_enabled
                        else 0
                    ),
                    ramp=execution.blend_ramp,
                )
            LOGGER.info(
                "Chunk ready: H=%d infer=%.1fms consumed=%d delay=%d "
                "blended=%s queued=%d",
                len(chunk),
                elapsed * 1000.0,
                consumed_during_inference,
                delay_steps,
                stats.get("blended", 0),
                stats["queued"],
            )
            # A short model horizon can remain below the low watermark. Wait
            # for real consumption before permitting another inference.
            consumed_after_submit = buffer.total_consumed()
            buffer.wait_for_consumption_after(
                consumed_after_submit, timeout=0.1
            )
    except BaseException as exc:
        errors.append(exc)
        LOGGER.exception("Asynchronous wholebody inference failed")
        control.request_stop()


class JointOutputFilter:
    """Optional 50 Hz one-pole low-pass/deadband filter for HEFT q21."""

    def __init__(self, enabled: bool, cutoff_hz: float, deadband: float, sample_hz: float):
        self.enabled = bool(enabled)
        self.cutoff_hz = float(cutoff_hz)
        self.deadband = float(deadband)
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz / float(sample_hz))
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def apply(self, frame):
        if not self.enabled:
            return frame
        current = np.asarray(frame.heft_q, dtype=np.float64)
        if self._previous is None:
            self._previous = current.copy()
        else:
            target = current
            if self.deadband > 0.0:
                target = np.where(
                    np.abs(current - self._previous) < self.deadband,
                    self._previous,
                    current,
                )
            self._previous += self.alpha * (target - self._previous)
        return replace(frame, heft_q=self._previous.copy())


def _publish_action(
    dds,
    action_vector: np.ndarray,
    seq: int,
    epoch: int,
    hand_scale: float,
    joint_filter: JointOutputFilter | None = None,
):
    frame = parse_action_chunk(action_vector)[0]
    if joint_filter is not None:
        frame = joint_filter.apply(frame)
    dds.publish_heft(seq, epoch, frame)
    dds.publish_hand(expand_hand_command(frame, hand_scale))
    return frame


def run(config: RobanWholebodyDeployConfig, control: RunControl) -> None:
    from kuavo_deploy.roban_wholebody_deploy.robot_io import (
        DdsIo,
        LatestInputs,
        RosCompressedCamera,
        invoke_switch_cli,
    )

    if config.dds.config_uri:
        os.environ["CYCLONEDDS_URI"] = config.dds.config_uri
        LOGGER.info("Using CycloneDDS config: %s", config.dds.config_uri)

    policy = WholebodyPolicy(config.inference)
    store = LatestInputs()
    camera = RosCompressedCamera(config.camera.topic, store)
    dds = DdsIo(config.dds.domain_id, store)
    buffer = ActionBuffer(config.execution.buffer_capacity)
    joint_filter = JointOutputFilter(
        config.heft.joint_filter_enabled,
        config.heft.joint_filter_cutoff_hz,
        config.heft.joint_filter_deadband,
        config.execution.action_hz,
    )
    inference_errors: list[BaseException] = []
    worker: threading.Thread | None = None
    switched = False
    last_action: np.ndarray | None = None

    try:
        LOGGER.info("Waiting for ROS camera and DDS joint/hand state")
        _wait_for_inputs(store, config, control.stop)
        if config.execution.mode == "async":
            worker = threading.Thread(
                target=_async_inference_loop,
                kwargs={
                    "config": config,
                    "store": store,
                    "policy": policy,
                    "buffer": buffer,
                    "control": control,
                    "errors": inference_errors,
                },
                name="roban-wholebody-inference",
                daemon=True,
            )
            worker.start()

        period = 1.0 / config.execution.action_hz
        deadline = time.monotonic()
        epoch = max(1, int(time.time()))
        first_publish: float | None = None
        hold_steps = 0
        published_steps = 0
        handled_generation = control.generation()
        fresh_chunk_deadline = time.monotonic() + config.execution.startup_timeout
        hold_limit = min(
            config.execution.max_hold_steps,
            max(1, int(config.execution.action_timeout * config.execution.action_hz)),
        )

        while published_steps < config.inference.max_steps:
            generation = control.generation()
            if generation != handled_generation:
                # The generation change already makes in-flight results stale.
                # Move the robot to AMP before disabling production and clearing
                # the local queue, matching the requested safety ordering.
                if switched and config.heft.switch_cli:
                    if not invoke_switch_cli(
                        config.heft.switch_cli,
                        config.heft.fallback_controller,
                        config.heft.switch_library_path,
                    ):
                        raise RuntimeError(
                            "Failed to switch controller to "
                            f"{config.heft.fallback_controller}"
                        )
                    switched = False

                    control.inference_enabled.clear()
                    joint_filter.reset()
                dropped = buffer.clear()
                last_action = None
                first_publish = None
                hold_steps = 0
                # A new epoch makes the restarted HEFT stream distinguishable
                # from references published before the pause.
                epoch = (epoch + 1) & 0xFFFFFFFF
                if epoch == 0:
                    epoch = 1
                handled_generation = generation
                deadline = time.monotonic()

                if control.stop.is_set():
                    LOGGER.info("Stop transition cleared %d queued actions", dropped)
                    break
                if control.pause.is_set():
                    LOGGER.info(
                        "Paused in %s; cleared %d queued actions",
                        config.heft.fallback_controller,
                        dropped,
                    )
                    continue

                if config.execution.mode == "sync":
                    policy.reset()
                fresh_chunk_deadline = (
                    time.monotonic() + config.execution.startup_timeout
                )
                control.inference_enabled.set()
                LOGGER.info(
                    "Resume requested; action buffer cleared and fresh warm-up required"
                )

            if control.stop.is_set():
                break
            if control.pause.is_set():
                # AMP owns the robot while paused. Do not refresh HEFT with a
                # held action and do not consume max_steps during the pause.
                time.sleep(min(period, 0.02))
                deadline = time.monotonic()
                continue

            if config.execution.mode == "sync" and buffer.size() == 0:
                request_generation = control.generation()
                observation = build_observation(_snapshot(store, config), config)
                chunk = policy.predict_chunk(observation)
                if (
                    control.stop.is_set()
                    or control.pause.is_set()
                    or control.generation() != request_generation
                ):
                    LOGGER.info(
                        "Discarded stale synchronous chunk from generation %d",
                        request_generation,
                    )
                    continue
                buffer.replace(chunk)

            # After resume, wait without publishing until a new-generation chunk
            # is available. The first fresh frame starts a complete warm-up.
            if buffer.size() == 0 and last_action is None:
                if inference_errors:
                    raise RuntimeError("Asynchronous inference stopped") from inference_errors[0]
                if time.monotonic() >= fresh_chunk_deadline:
                    raise TimeoutError("Timed out waiting for a fresh action chunk")
                time.sleep(min(period, 0.02))
                deadline = time.monotonic()
                continue

            # Never block the 50 Hz publisher waiting for inference. A transient
            # underrun is handled by a bounded last-action hold.
            action = buffer.take(timeout=0.0)
            if action is None:
                if last_action is None or hold_steps >= hold_limit:
                    raise TimeoutError("Wholebody action buffer underrun")
                action = last_action
                hold_steps += 1
                LOGGER.warning("Action buffer empty; holding the last action (%d)", hold_steps)
            else:
                hold_steps = 0
                last_action = action

            published_steps += 1
            _publish_action(
                dds,
                action,
                published_steps,
                epoch,
                config.heft.hand_scale,
                joint_filter,
            )
            if first_publish is None:
                first_publish = time.monotonic()
                LOGGER.info("HEFT 50Hz reference warm-up started")
            if (
                config.heft.switch_cli
                and not switched
                and time.monotonic() - first_publish >= config.heft.warmup_sec
            ):
                switched = invoke_switch_cli(
                    config.heft.switch_cli,
                    config.heft.controller,
                    config.heft.switch_library_path,
                )
                if not switched:
                    raise RuntimeError(
                        f"Failed to switch controller to {config.heft.controller}"
                    )

            deadline += period
            sleep_seconds = deadline - time.monotonic()
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
            else:
                LOGGER.warning(
                    "50Hz HEFT deadline missed by %.1fms at step %d",
                    -sleep_seconds * 1000.0,
                    published_steps,
                )
                deadline = time.monotonic()

        if inference_errors:
            raise RuntimeError("Asynchronous inference stopped") from inference_errors[0]
    finally:
        # Switch away from HEFT before stopping inference/DDS. This path covers
        # SIGUSR2, SIGTERM, Ctrl-C, max_steps, and runtime exceptions.
        if switched and config.heft.switch_cli:
            if not invoke_switch_cli(
                config.heft.switch_cli,
                config.heft.fallback_controller,
                config.heft.switch_library_path,
            ):
                LOGGER.error(
                    "Failed to switch controller to %s during shutdown",
                    config.heft.fallback_controller,
                )
        dropped = buffer.clear()
        control.inference_enabled.clear()
        control.request_stop()
        if worker is not None:
            worker.join(timeout=1.0)
        dds.close()
        del camera
        LOGGER.info("Cleared %d queued actions during shutdown", dropped)
        LOGGER.info("Wholebody runner stopped; HEFT stream is no longer refreshed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="run", choices=("run",))
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_roban_wholebody_deploy_config(args.config)
    control = RunControl()
    control.install_signal_handlers()
    run(config, control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
