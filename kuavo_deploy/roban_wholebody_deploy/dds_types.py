"""CycloneDDS wire types for Roban state and HEFT commands.

Do not enable ``from __future__ import annotations`` in this module.  Older
CycloneDDS Python bindings resolve the IDL annotations while populating these
types, so ``sequence[float64]`` and ``array[...]`` must remain concrete objects.
"""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl import annotations as annotate
from cyclonedds.idl.types import (
    array,
    float64,
    int32,
    int64,
    sequence,
    uint8,
    uint32,
    uint64,
)


@dataclass
@annotate.final
@annotate.autoid("sequential")
class JointState(IdlStruct, typename="leju::msgs::JointState"):
    header_sec: int32
    header_nanosec: uint32
    q: sequence[float64]
    v: sequence[float64]
    vd: sequence[float64]
    tau: sequence[float64]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class HandState(IdlStruct, typename="leju::msgs::HandState"):
    header_sec: int32
    header_nanosec: uint32
    left_valid: bool
    right_valid: bool
    left_sample_age_ms: uint32
    right_sample_age_ms: uint32
    position: sequence[float64]
    velocity: sequence[float64]
    current: sequence[float64]
    state: sequence[uint8]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Float64Array(IdlStruct, typename="leju::msgs::Float64Array"):
    header_sec: int32
    header_nanosec: uint32
    data: sequence[float64]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class HeftReference(IdlStruct, typename="leju::msgs::HeftReference"):
    header_sec: int32
    header_nanosec: uint32
    schema_version: uint32
    robot_version: str
    seq: uint64
    calibration_epoch: uint32
    capture_timestamp_ms: int64
    valid: bool
    calibrated: bool
    root_pos: array[float64, 3]
    root_quat: array[float64, 4]
    q: sequence[float64]
    qd: sequence[float64]
