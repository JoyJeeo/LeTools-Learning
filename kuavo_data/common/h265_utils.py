# Copyright (C) 2025-2026 LejuRobotics.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""H.265 (HEVC) rosbag video utilities — streaming direct encode path."""

from __future__ import annotations

import bisect
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import av
import cv2
import numpy as np
from lerobot.datasets.compute_stats import (
    RunningQuantileStats,
    auto_downsample_height_width,
    sample_indices,
)

logger = logging.getLogger(__name__)

_ANNEX_B_START_CODE = b"\x00\x00\x00\x01"
_ANNEX_B_START_CODE3 = b"\x00\x00\x01"
# HEVC NAL unit types
_NAL_VPS = 32
_NAL_SPS = 33
_NAL_PPS = 34
_NAL_IDR_W_RADL = 19
_NAL_IDR_N_LP = 20
_NAL_CRA = 21
_NAL_TRAIL_N = 0
_NAL_TRAIL_R = 1
_PARAM_SET_TYPES = {_NAL_VPS, _NAL_SPS, _NAL_PPS}
_IDR_TYPES = {_NAL_IDR_W_RADL, _NAL_IDR_N_LP}
_P_SLICE_TYPES = {_NAL_TRAIL_N, _NAL_TRAIL_R}
_SLICE_TYPES = _P_SLICE_TYPES | _IDR_TYPES | {_NAL_CRA}


def _split_annex_b_nals(data: bytes) -> List[bytes]:
    """Split Annex-B bitstream into NAL units (each includes its start code)."""
    if not data:
        return []
    nals: List[bytes] = []
    i = 0
    start = 0
    while i < len(data):
        if data.startswith(_ANNEX_B_START_CODE, i):
            if i > start:
                nals.append(data[start:i])
            start = i
            i += 4
            continue
        if data.startswith(_ANNEX_B_START_CODE3, i):
            if i > start:
                nals.append(data[start:i])
            start = i
            i += 3
            continue
        i += 1
    if start < len(data):
        nals.append(data[start:])
    return [nal for nal in nals if nal]


def _nal_unit_type(nal: bytes) -> int:
    payload = nal[4:] if nal.startswith(_ANNEX_B_START_CODE) else nal[3:]
    if not payload:
        return -1
    return (payload[0] & 0x7E) >> 1


def _build_access_unit(nals: Sequence[bytes], types: Sequence[int], *, include_param_sets: bool) -> bytes:
    """Assemble one access unit, optionally prefixing VPS/SPS/PPS (SEI stripped)."""
    parts: List[bytes] = []
    if include_param_sets:
        for nal, nal_type in zip(nals, types):
            if nal_type in _PARAM_SET_TYPES:
                parts.append(nal)
    for nal, nal_type in zip(nals, types):
        if nal_type in _SLICE_TYPES:
            parts.append(nal)
    return b"".join(parts)


def _sanitize_h265_messages(
    raw_messages: Sequence[Tuple[float, bytes]],
) -> Tuple[List[Tuple[float, bytes]], int]:
    """Clean rosbag H.265 messages for sequential decode.

    - Drop access units before the first IDR (dangling P/B references).
    - For IDR access units: keep VPS/SPS/PPS + slice (drop SEI).
    - For P/B access units: keep slice NALs only (strip repeated parameter sets).
    """
    parsed: List[Tuple[float, List[bytes], List[int]]] = []
    for ts, data in raw_messages:
        nals = _split_annex_b_nals(bytes(data))
        types = [_nal_unit_type(nal) for nal in nals]
        parsed.append((ts, nals, types))

    first_idr_idx = next(
        (i for i, (_, _, types) in enumerate(parsed) if any(t in _IDR_TYPES for t in types)),
        None,
    )
    if first_idr_idx is None:
        raise ValueError("No IDR frame found in H.265 stream")

    sanitized: List[Tuple[float, bytes]] = []
    for ts, nals, types in parsed[first_idr_idx:]:
        has_idr = any(t in _IDR_TYPES for t in types)
        au_bytes = _build_access_unit(nals, types, include_param_sets=has_idr)
        if au_bytes:
            sanitized.append((ts, au_bytes))

    return sanitized, first_idr_idx

# RGB camera key -> ROS camera namespace.
_CAMERA_BASE = {
    "head_cam_h": "cam_h",
    "wrist_cam_l": "cam_l",
    "wrist_cam_r": "cam_r",
}

JPEG_COLOR_TOPIC = {
    "head_cam_h": "/cam_h/color/image_raw/compressed",
    "wrist_cam_l": "/cam_l/color/image_raw/compressed",
    "wrist_cam_r": "/cam_r/color/image_raw/compressed",
    "head_cam_l": "/zedm/zed_node/left/image_rect_color/compressed",
    "head_cam_r": "/zedm/zed_node/right/image_rect_color/compressed",
}

def get_h265_topic(camera_key: str) -> str | None:
    base = _CAMERA_BASE.get(camera_key)
    if base is None:
        return None
    return f"/{base}/color/h265_stream"


def get_jpeg_topic(camera_key: str) -> str | None:
    return JPEG_COLOR_TOPIC.get(camera_key)


def detect_camera_encodings(bag_topics: set[str], camera_names: Sequence[str]) -> Dict[str, str]:
    """Return {camera_key: 'jpeg' | 'h265'} for each configured camera."""
    encodings: Dict[str, str] = {}
    for cam in camera_names:
        jpeg_topic = get_jpeg_topic(cam)
        h265_topic = get_h265_topic(cam)
        if jpeg_topic is None and h265_topic is None:
            continue
        if jpeg_topic and jpeg_topic in bag_topics:
            encodings[cam] = "jpeg"
        elif h265_topic and h265_topic in bag_topics:
            encodings[cam] = "h265"
        else:
            logger.warning(
                "Camera %s: neither JPEG (%s) nor H.265 (%s) topic found in bag",
                cam,
                jpeg_topic,
                h265_topic,
            )
    return encodings


def _read_bag_topics(bag_file: str) -> set[str]:
    import rosbag

    bag = rosbag.Bag(bag_file)
    topics = set(bag.get_type_and_topic_info().topics.keys())
    bag.close()
    return topics


def detect_camera_encodings_from_bag(bag_file: str, camera_names: Sequence[str]) -> Dict[str, str]:
    return detect_camera_encodings(_read_bag_topics(bag_file), camera_names)


def trim_main_timestamps_strict(
    main_timestamps: Sequence[float],
    camera_timestamps: Dict[str, Sequence[float]],
) -> Tuple[List[float], float, float, int]:
    """Trim main timeline to the intersection of all camera valid time windows.

    Ensures every remaining main timestamp can be matched to a decodable frame on
    every camera without clamping to first/last (strict cross-modal alignment).

    Returns:
        trimmed timestamps, valid_start, valid_end, number of dropped frames
    """
    if not main_timestamps:
        raise ValueError("main_timestamps is empty")

    sources = {cam: ts for cam, ts in camera_timestamps.items() if ts}
    if not sources:
        return list(main_timestamps), main_timestamps[0], main_timestamps[-1], 0

    valid_start = max(ts[0] for ts in sources.values())
    valid_end = min(ts[-1] for ts in sources.values())

    if valid_start > valid_end:
        raise ValueError(
            f"No overlapping camera time window: valid_start={valid_start:.6f} > "
            f"valid_end={valid_end:.6f}"
        )

    trimmed = [t for t in main_timestamps if valid_start <= t <= valid_end]
    dropped = len(main_timestamps) - len(trimmed)

    if not trimmed:
        raise ValueError(
            f"Main timeline empty after strict trim "
            f"(valid_start={valid_start:.6f}, valid_end={valid_end:.6f})"
        )

    if dropped > 0:
        logger.info(
            "Strict main timeline trim: %d -> %d frames "
            "(valid window [%.6f, %.6f], cameras=%s)",
            len(main_timestamps),
            len(trimmed),
            valid_start,
            valid_end,
            sorted(sources.keys()),
        )

    return trimmed, valid_start, valid_end, dropped


def build_camera_timestamp_sources(
    camera_names: Sequence[str],
    h265_cameras: set[str],
    h265_contexts: Dict[str, "H265CameraContext"],
    all_timestamps: Dict[str, Sequence[float]],
) -> Dict[str, Sequence[float]]:
    """Collect per-camera timestamp sequences for strict main-timeline trimming."""
    sources: Dict[str, Sequence[float]] = {}
    for cam in camera_names:
        if cam in h265_cameras and cam in h265_contexts:
            sources[cam] = h265_contexts[cam].source_timestamps
        elif cam in all_timestamps and len(all_timestamps[cam]) > 0:
            sources[cam] = all_timestamps[cam]
    return sources


def _nearest_indices(main_timestamps: Sequence[float], source_timestamps: Sequence[float]) -> List[int]:
    if not source_timestamps:
        raise ValueError("source_timestamps is empty")
    indices: List[int] = []
    for stamp in main_timestamps:
        pos = bisect.bisect_left(source_timestamps, stamp)
        if pos == 0:
            indices.append(0)
        elif pos >= len(source_timestamps):
            indices.append(len(source_timestamps) - 1)
        elif abs(source_timestamps[pos] - stamp) < abs(source_timestamps[pos - 1] - stamp):
            indices.append(pos)
        else:
            indices.append(pos - 1)
    return indices


def preload_h265_nal_data(
    bag_file: str,
    topic: str,
    output_dir: Path,
) -> Tuple[Path, List[float]]:
    """Write sanitized H.265 access units to disk and collect timestamps.

    Sanitizes the rosbag bitstream (drop pre-IDR frames, dedupe VPS/SPS/PPS) so
    PyAV/FFmpeg can decode without POC reference errors.
    """
    import rosbag

    output_dir.mkdir(parents=True, exist_ok=True)
    nal_path = output_dir / "stream.h265"

    raw_messages: List[Tuple[float, bytes]] = []
    bag = rosbag.Bag(bag_file)
    for _, msg, t in bag.read_messages(topics=[topic]):
        raw_messages.append((t.to_sec(), bytes(msg.data)))
    bag.close()

    if not raw_messages:
        raise ValueError(f"No H.265 messages on topic {topic} in {bag_file}")

    sanitized, dropped_prefix = _sanitize_h265_messages(raw_messages)
    timestamps = [ts for ts, _ in sanitized]

    with open(nal_path, "wb") as nal_file:
        for _, au_bytes in sanitized:
            nal_file.write(au_bytes)

    logger.info(
        "Preloaded H.265 topic %s: %d raw msgs -> %d sanitized AUs "
        "(dropped %d pre-IDR), %.1f MB on disk",
        topic,
        len(raw_messages),
        len(sanitized),
        dropped_prefix,
        nal_path.stat().st_size / 1024 / 1024,
    )
    return nal_path, timestamps


def _frame_to_rgb(frame: av.VideoFrame, width: int, height: int) -> np.ndarray:
    rgb = frame.to_ndarray(format="rgb24")
    if rgb.shape[1] != width or rgb.shape[0] != height:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb)


def _frame_to_bgr(frame: av.VideoFrame, width: int, height: int) -> np.ndarray:
    rgb = _frame_to_rgb(frame, width, height)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _update_video_stats_tracker(stats_tracker: RunningQuantileStats, rgb: np.ndarray) -> None:
    """Accumulate per-channel stats from an RGB frame (matches LeRobot pyav decode order)."""
    img_chw = rgb.transpose(2, 0, 1)
    img_downsampled = auto_downsample_height_width(img_chw)
    channels = img_downsampled.shape[0]
    img_for_stats = img_downsampled.transpose(1, 2, 0).reshape(-1, channels)
    stats_tracker.update(img_for_stats)


def normalize_video_stats(raw_stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert RunningQuantileStats output to LeRobot episode video stats format."""
    return {
        key: value
        if key == "count"
        else np.squeeze(value.reshape(1, -1, 1, 1) / 255.0, axis=0)
        for key, value in raw_stats.items()
    }


def save_h265_videos_direct(
    nal_path: Path,
    source_timestamps: Sequence[float],
    main_timestamps: Sequence[float],
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    *,
    crf: int = 30,
    preset: int = 12,
    g: int = 2,
) -> tuple[Path, dict[str, np.ndarray] | None]:
    """Decode H.265 sequentially and pipe selected frames to ffmpeg (libsvtav1).

    Output frame count equals len(main_timestamps). Nearest-neighbour timestamp
    mapping selects which decoded frame to emit for each main-timeline step.

    Returns:
        Path to the encoded MP4 and per-channel video stats (LeRobot format), or
        None for stats when fewer than two output frames were sampled.
    """
    needed_indices = _nearest_indices(main_timestamps, source_timestamps)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    stats_sample_set = set(sample_indices(len(main_timestamps)))
    stats_tracker = RunningQuantileStats()

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libsvtav1",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-g",
        str(g),
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    output_idx = 0
    decode_idx = 0
    last_rgb: np.ndarray | None = None
    last_bgr: np.ndarray | None = None

    def write_output_frame(rgb: np.ndarray, bgr: np.ndarray) -> None:
        nonlocal output_idx
        if output_idx in stats_sample_set:
            _update_video_stats_tracker(stats_tracker, rgb)
        proc.stdin.write(bgr.tobytes())
        output_idx += 1

    try:
        with av.open(str(nal_path)) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                try:
                    rgb = _frame_to_rgb(frame, width, height)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception as exc:
                    logger.warning("Skipping corrupted H.265 frame %d: %s", decode_idx, exc)
                    if last_rgb is None or last_bgr is None:
                        decode_idx += 1
                        continue
                    rgb, bgr = last_rgb, last_bgr

                last_rgb, last_bgr = rgb, bgr
                while output_idx < len(needed_indices) and needed_indices[output_idx] == decode_idx:
                    write_output_frame(rgb, bgr)
                decode_idx += 1
                if output_idx >= len(needed_indices):
                    break

        while output_idx < len(needed_indices):
            if last_rgb is None or last_bgr is None:
                raise RuntimeError("H.265 decode produced no frames")
            write_output_frame(last_rgb, last_bgr)
    finally:
        if proc.stdin:
            proc.stdin.close()
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (code {ret}): {stderr}")

    if output_idx != len(main_timestamps):
        raise RuntimeError(
            f"H.265 MP4 frame count mismatch: wrote {output_idx}, expected {len(main_timestamps)}"
        )

    video_stats: dict[str, np.ndarray] | None = None
    if output_idx >= 2:
        raw_stats = stats_tracker.get_statistics()
        # Match compute_episode_stats: count = number of sampled frames, not pixels.
        raw_stats["count"] = np.array([len(stats_sample_set)])
        video_stats = normalize_video_stats(raw_stats)

    logger.info(
        "Saved H.265 direct MP4: %s (%d frames @ %d fps)",
        output_mp4,
        len(main_timestamps),
        fps,
    )
    return output_mp4, video_stats


class H265CameraContext:
    """Per-camera H.265 preload state for one episode."""

    def __init__(
        self,
        camera_key: str,
        topic: str,
        nal_path: Path,
        source_timestamps: List[float],
        temp_dir: Path,
    ):
        self.camera_key = camera_key
        self.topic = topic
        self.nal_path = nal_path
        self.source_timestamps = source_timestamps
        self.temp_dir = temp_dir
        self.video_key = f"observation.images.{camera_key}"

    def cleanup(self) -> None:
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def preload_h265_cameras(
    bag_file: str,
    h265_cameras: Dict[str, str],
    temp_root: Path | None = None,
) -> Dict[str, H265CameraContext]:
    """Preload all H.265 cameras for an episode."""
    if temp_root is None:
        temp_root = Path(tempfile.mkdtemp(prefix="h265_preload_"))
    temp_root.mkdir(parents=True, exist_ok=True)

    contexts: Dict[str, H265CameraContext] = {}
    for cam, _encoding in h265_cameras.items():
        topic = get_h265_topic(cam)
        if topic is None:
            continue
        cam_dir = temp_root / cam
        nal_path, timestamps = preload_h265_nal_data(bag_file, topic, cam_dir)
        contexts[cam] = H265CameraContext(cam, topic, nal_path, timestamps, cam_dir)
    return contexts
