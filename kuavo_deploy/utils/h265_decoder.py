"""Real-time H.265 stream decoder for inference-time camera observation.

Each RGB camera gets a stateful PyAV decoder instance. CompressedImage.data
bytes (one H.265 access unit in Annex-B format) are fed per ROS callback, and
the decoded RGB frame is returned for the observation buffer.
"""

from __future__ import annotations

import logging
from typing import Optional

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RealtimeH265Decoder:
    """Stateful H.265 decoder for a single camera stream.

    Accepts one CompressedImage.data per call (one access unit in Annex-B format),
    returns the decoded RGB frame resized to target dimensions.

    The decoder maintains codec state across calls (VPS/SPS/PPS carry over),
    so it can decode P-frames that reference earlier IDR frames.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._codec_ctx: av.CodecContext = av.CodecContext.create("hevc", "r")
        self._last_frame: Optional[np.ndarray] = None
        self._initialized = False

    def decode(self, nal_data: bytes) -> Optional[np.ndarray]:
        """Decode one H.265 access unit, return RGB frame (H, W, 3) or None.

        Returns None only before the first IDR is successfully decoded.
        After that, returns last good frame on decode failure (graceful degradation).
        """
        try:
            packet = av.Packet(nal_data)
            frames = self._codec_ctx.decode(packet)
        except (av.error.InvalidDataError, av.error.ValueError):
            return self._last_frame

        if not frames:
            return self._last_frame

        frame = frames[-1]
        rgb = frame.to_ndarray(format="rgb24")
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self._last_frame = np.ascontiguousarray(rgb)
        self._initialized = True
        return self._last_frame

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        """Reset decoder state (e.g., on episode boundary)."""
        self._codec_ctx = av.CodecContext.create("hevc", "r")
        self._last_frame = None
        self._initialized = False
