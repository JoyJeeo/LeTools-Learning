"""Thread-safe action buffer with delay-aware overlap blending."""

from __future__ import annotations

import math
import time
from collections import deque
from threading import Condition
from typing import Any

import numpy as np


def blend_weights(count: int, ramp: str) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    # Do not reach either endpoint inside the overlap. The sample immediately
    # before the blend remains fully old; the tail immediately after is new.
    phase = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
    if ramp == "linear":
        return phase
    if ramp == "cosine":
        return 0.5 - 0.5 * np.cos(math.pi * phase)
    raise ValueError(f"Unsupported blend ramp: {ramp}")


class ActionBuffer:
    """A single-consumer queue whose merge never rewinds consumed actions."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._actions: deque[np.ndarray] = deque()
        self._condition = Condition()
        self._total_consumed = 0

    def size(self) -> int:
        with self._condition:
            return len(self._actions)

    def total_consumed(self) -> int:
        with self._condition:
            return self._total_consumed

    def snapshot(self) -> list[np.ndarray]:
        with self._condition:
            return [action.copy() for action in self._actions]

    def clear(self) -> int:
        """Drop every queued action and wake producers waiting on queue size."""
        with self._condition:
            dropped = len(self._actions)
            self._actions.clear()
            self._condition.notify_all()
            return dropped

    def wait_until_at_most(self, size: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._actions) > size:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def wait_until_nonempty(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._actions:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def wait_for_consumption_after(self, consumed: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._total_consumed <= consumed:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def take(self, timeout: float) -> np.ndarray | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._actions:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            action = self._actions.popleft()
            self._total_consumed += 1
            self._condition.notify_all()
            return action

    @staticmethod
    def _validate_chunk(actions: Any) -> list[np.ndarray]:
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"Action buffer requires a non-empty [H,D] chunk, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Action chunk contains non-finite values")
        return [row.copy() for row in array]

    def replace(self, actions: Any) -> dict[str, int | str]:
        new = self._validate_chunk(actions)[: self.capacity]
        with self._condition:
            old_size = len(self._actions)
            self._actions.clear()
            self._actions.extend(new)
            self._condition.notify_all()
            return {
                "mode": "replace",
                "old_size": old_size,
                "produced": len(new),
                "queued": len(self._actions),
            }

    def merge(
        self,
        actions: Any,
        *,
        delay_steps: int,
        overlap_steps: int,
        ramp: str,
    ) -> dict[str, int | str]:
        """Align a new chunk to now, then blend it with the unconsumed queue.

        Actions consumed while inference ran are represented only by
        ``delay_steps`` and are dropped from the new chunk. The current queue
        already starts at the next not-yet-executed control step, so there is no
        freeze index and no action-index rewind.
        """
        produced = self._validate_chunk(actions)
        delay = min(max(0, int(delay_steps)), len(produced))
        new = produced[delay:]
        with self._condition:
            old = list(self._actions)
            old_size = len(old)
            if not new:
                return {
                    "mode": "blend_replace",
                    "old_size": old_size,
                    "produced": len(produced),
                    "delay_steps": delay,
                    "blended": 0,
                    "queued": old_size,
                }

            blend_count = min(max(0, int(overlap_steps)), len(old), len(new))
            weights = blend_weights(blend_count, ramp)
            blended = [
                ((1.0 - weight) * old[index] + weight * new[index]).astype(
                    np.float32, copy=False
                )
                for index, weight in enumerate(weights)
            ]
            merged = blended + new[blend_count:]
            merged = merged[: self.capacity]
            self._actions.clear()
            self._actions.extend(merged)
            self._condition.notify_all()
            return {
                "mode": "blend_replace",
                "old_size": old_size,
                "produced": len(produced),
                "delay_steps": delay,
                "blended": blend_count,
                "queued": len(self._actions),
            }
