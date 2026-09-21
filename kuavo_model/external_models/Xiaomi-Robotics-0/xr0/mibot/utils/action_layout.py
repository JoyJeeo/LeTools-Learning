# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Component:
    name: str
    source: slice
    target: slice
    state_component: str | None = None


class ActionLayout:
    """Pack a flat robot state/action space into XR0 model tensors."""

    def __init__(self, config: Mapping):
        self.signature = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        model = config["model"]
        source = config["source"]
        self.state_dim = int(model["state_dim"])
        self.action_dim = int(model["action_dim"])
        self.state_key = str(source["state"])
        self.action_key = str(source["action"])
        self.representation = str(config.get("action_representation", "delta"))
        self.state = self._components(config["state"], self.state_dim, is_action=False)
        self.action = self._components(config["action"], self.action_dim, is_action=True)
        self.state_by_name = {component.name: component for component in self.state}
        self.state_source_dim = max(component.source.stop for component in self.state)
        self.action_source_dim = max(component.source.stop for component in self.action)
        self.camera_keys = tuple(value["original_key"] for value in config["video"].values())

        if self.representation not in {"absolute", "delta"}:
            raise ValueError(f"Unsupported action representation: {self.representation}")
        if self.representation == "delta":
            for component in self.action:
                state_component = self.state_by_name[component.state_component]
                if self._size(component.source) != self._size(state_component.source):
                    raise ValueError(f"State/action size mismatch for {component.name}")

    @classmethod
    def _components(cls, values: Mapping, model_dim: int, is_action: bool) -> tuple[Component, ...]:
        components = []
        occupied = np.zeros(model_dim, dtype=bool)
        for name, value in values.items():
            source = slice(int(value["start"]), int(value["end"]))
            target_start = int(value["model_start"])
            target = slice(target_start, target_start + cls._size(source))
            if source.start < 0 or source.stop <= source.start or target.stop > model_dim:
                raise ValueError(f"Invalid slice for component {name}")
            if occupied[target].any():
                raise ValueError(f"Overlapping model slice for component {name}")
            occupied[target] = True
            components.append(
                Component(
                    name=name,
                    source=source,
                    target=target,
                    state_component=str(value.get("state_component", name)) if is_action else None,
                )
            )
        if not components:
            raise ValueError("Layout must contain at least one component")
        return tuple(components)

    @staticmethod
    def _size(value: slice) -> int:
        return value.stop - value.start

    @staticmethod
    def _vector(value, expected_dim: int, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32).reshape(-1)
        if result.shape != (expected_dim,):
            raise ValueError(f"{name} expected shape {(expected_dim,)}, got {result.shape}")
        return result

    def pack_state(self, state) -> np.ndarray:
        state = self._vector(state, self.state_source_dim, self.state_key)
        output = np.zeros((1, self.state_dim), dtype=np.float32)
        for component in self.state:
            output[0, component.target] = state[component.source]
        return output

    def pack_action(self, state, future_action) -> np.ndarray:
        state = self._vector(state, self.state_source_dim, self.state_key)
        future = np.asarray(future_action, dtype=np.float32)
        if future.ndim != 2 or future.shape[1] != self.action_source_dim:
            raise ValueError(f"{self.action_key} expected shape [T, {self.action_source_dim}], got {future.shape}")
        output = np.zeros((future.shape[0], self.action_dim), dtype=np.float32)
        for component in self.action:
            value = future[:, component.source]
            if self.representation == "delta":
                value = value - state[self.state_by_name[component.state_component].source]
            output[:, component.target] = value
        return output

    def recover_action(self, action, state) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        state = self._vector(state, self.state_source_dim, self.state_key)
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"Model action expected shape [T, {self.action_dim}], got {action.shape}")
        output = np.zeros((action.shape[0], self.action_source_dim), dtype=np.float32)
        for component in self.action:
            value = action[:, component.target]
            if self.representation == "delta":
                value = value + state[self.state_by_name[component.state_component].source]
            output[:, component.source] = value
        return output

    def action_mask(self, action_length: int, temporal_mask=None) -> np.ndarray:
        temporal = np.ones(action_length, dtype=np.int32) if temporal_mask is None else np.asarray(temporal_mask, dtype=np.int32)
        if temporal.shape != (action_length,):
            raise ValueError(f"temporal_mask expected shape {(action_length,)}, got {temporal.shape}")
        mask = np.zeros((action_length, self.action_dim), dtype=np.int32)
        for component in self.action:
            mask[:, component.target] = temporal[:, None]
        return mask
