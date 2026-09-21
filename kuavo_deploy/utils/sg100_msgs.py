"""Build SG100 ROS message classes from repository-local ``.msg`` files.

The generated classes use the wire type names ``kuavo_msgs/SG100HandCommand``
and ``kuavo_msgs/SG100HandState`` so they interoperate with the existing SG100
driver without installing a separate Python message package.
"""

from functools import lru_cache
from pathlib import Path

import genpy.dynamic

_MSG_DIR = Path(__file__).resolve().parents[1] / "msg"
_HEADER_TEXT = "uint32 seq\ntime stamp\nstring frame_id\n"
_SEPARATOR = "=" * 80 + "\n"


@lru_cache(maxsize=None)
def _generate_msg_class(filename: str, ros_type: str):
    msg_text = (_MSG_DIR / filename).read_text(encoding="utf-8")
    full_text = msg_text + "\n" + _SEPARATOR + "MSG: std_msgs/Header\n" + _HEADER_TEXT
    classes = genpy.dynamic.generate_dynamic(ros_type, full_text)
    return classes[ros_type]


SG100HandCommand = _generate_msg_class("SG100HandCommand.msg", "kuavo_msgs/SG100HandCommand")
SG100HandState = _generate_msg_class("SG100HandState.msg", "kuavo_msgs/SG100HandState")

__all__ = ["SG100HandCommand", "SG100HandState"]
