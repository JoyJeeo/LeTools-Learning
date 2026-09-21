"""Typed configuration for Roban wholebody deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    topic: str = "/camera/color/image_raw/compressed"
    width: int = 640
    height: int = 480
    max_age: float = 0.5
    startup_timeout: float = 10.0


@dataclass(frozen=True)
class DdsConfig:
    domain_id: int = 0
    state_max_age: float = 0.2
    config_uri: str = ""


@dataclass(frozen=True)
class InferenceConfig:
    policy_type: str = "client"
    pretrained_path: str = ""
    device: str = "cuda"
    task_prompt: str = "robot manipulation"
    server_host: str = "localhost"
    server_port: int = 5555
    api_token: str = ""
    max_steps: int = 100000


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "async"
    action_hz: float = 50.0
    prefetch_steps: int = 20
    buffer_capacity: int = 200
    startup_timeout: float = 30.0
    action_timeout: float = 0.2
    max_hold_steps: int = 5
    blend_enabled: bool = True
    blend_overlap_steps: int = 10
    blend_ramp: str = "cosine"
    max_inference_delay_steps: int = 50


@dataclass(frozen=True)
class HeftConfig:
    hand_scale: float = 100.0
    warmup_sec: float = 0.4
    switch_cli: str = ""
    switch_library_path: str = ""
    joint_filter_enabled: bool = False
    joint_filter_cutoff_hz: float = 8.0
    joint_filter_deadband: float = 0.0
    controller: str = "heft_pico"
    fallback_controller: str = "amp"


@dataclass(frozen=True)
class RobanWholebodyDeployConfig:
    platform_type: str
    camera: CameraConfig
    dds: DdsConfig
    inference: InferenceConfig
    execution: ExecutionConfig
    heft: HeftConfig


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must be a mapping")
    return value


def load_roban_wholebody_deploy_config(
    path: str | Path,
) -> RobanWholebodyDeployConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    dds_data = dict(_section(data, "dds"))
    config_uri = str(dds_data.get("config_uri", "")).strip()
    if config_uri and not config_uri.startswith(("file://", "<")):
        config_uri = f"file://{(config_path.parent / config_uri).resolve()}"
    dds_data["config_uri"] = config_uri

    config = RobanWholebodyDeployConfig(
        platform_type=str(data.get("platform_type", "")).strip().lower(),
        camera=CameraConfig(**_section(data, "camera")),
        dds=DdsConfig(**dds_data),
        inference=InferenceConfig(**_section(data, "inference")),
        execution=ExecutionConfig(**_section(data, "execution")),
        heft=HeftConfig(**_section(data, "heft")),
    )
    _validate(config)
    return config


def _validate(config: RobanWholebodyDeployConfig) -> None:
    if config.platform_type != "roban":
        raise ValueError("Wholebody deploy requires platform_type=roban")
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ValueError("camera width and height must be positive")
    if config.camera.max_age <= 0 or config.camera.startup_timeout <= 0:
        raise ValueError("camera age and startup timeout must be positive")
    if config.dds.domain_id < 0 or config.dds.state_max_age <= 0:
        raise ValueError("dds.domain_id must be non-negative and state_max_age positive")
    if config.inference.policy_type != "client" and not config.inference.pretrained_path:
        raise ValueError("inference.pretrained_path is required for a local policy")
    if config.inference.device not in {"cpu", "cuda"}:
        raise ValueError("inference.device must be `cpu` or `cuda`")
    if not 1 <= config.inference.server_port <= 65535:
        raise ValueError("inference.server_port must be between 1 and 65535")
    if config.inference.max_steps <= 0:
        raise ValueError("inference.max_steps must be positive")
    execution = config.execution
    if execution.mode not in {"sync", "async"}:
        raise ValueError("execution.mode must be `sync` or `async`")
    if abs(execution.action_hz - 50.0) > 1e-6:
        raise ValueError("HEFT wholebody execution.action_hz must be 50")
    integer_fields = {
        "prefetch_steps": execution.prefetch_steps,
        "buffer_capacity": execution.buffer_capacity,
        "max_hold_steps": execution.max_hold_steps,
        "blend_overlap_steps": execution.blend_overlap_steps,
        "max_inference_delay_steps": execution.max_inference_delay_steps,
    }
    if any(value < 0 for value in integer_fields.values()):
        raise ValueError(f"execution counts must be non-negative: {integer_fields}")
    if execution.buffer_capacity <= 0:
        raise ValueError("execution.buffer_capacity must be positive")
    if execution.prefetch_steps >= execution.buffer_capacity:
        raise ValueError("execution.prefetch_steps must be smaller than buffer_capacity")
    if execution.blend_ramp not in {"linear", "cosine"}:
        raise ValueError("execution.blend_ramp must be linear or cosine")
    if execution.startup_timeout <= 0 or execution.action_timeout <= 0:
        raise ValueError("execution timeouts must be positive")
    if config.heft.hand_scale <= 0 or config.heft.warmup_sec < 0:
        raise ValueError("heft.hand_scale must be positive and warmup_sec non-negative")
    if config.heft.joint_filter_cutoff_hz <= 0 or config.heft.joint_filter_cutoff_hz >= config.execution.action_hz / 2:
        raise ValueError("heft.joint_filter_cutoff_hz must be between 0 and Nyquist")
    if config.heft.joint_filter_deadband < 0:
        raise ValueError("heft.joint_filter_deadband must be non-negative")
