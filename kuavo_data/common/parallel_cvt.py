"""Shard rosbag conversion across processes, then merge LeRobot datasets.

Used by the Kuavo and Roban Hydra converters when ``rosbag.parallel=true``.
Each worker writes ``target_dir/_parallel_shards/shard_XX/lerobot`` with
``image_writer_processes=0``. Successful shards are merged into ``target_dir/lerobot``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SHARD_DIRNAME = "_parallel_shards"
log_print = logging.getLogger(__name__)


def parallel_options(cfg: DictConfig) -> tuple[bool, int, bool, int]:
    """Return (enabled, workers, keep_shards, image_writer_threads)."""
    enabled = bool(cfg.rosbag.get("parallel", False))
    workers = int(cfg.rosbag.get("parallel_workers", 4))
    keep_shards = bool(cfg.rosbag.get("parallel_keep_shards", False))
    writer_threads = int(cfg.rosbag.get("parallel_image_writer_threads", 8))
    if workers < 1:
        raise ValueError("rosbag.parallel_workers must be >= 1")
    if writer_threads < 1:
        raise ValueError("rosbag.parallel_image_writer_threads must be >= 1")
    return enabled, workers, keep_shards, writer_threads


def dump_resolved_cfg(cfg: DictConfig, directory: Path) -> Path:
    """Persist rosbag/dataset so spawn workers can reload without Hydra cwd."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "_resolved.yaml"
    payload = {}
    for key in ("rosbag", "dataset"):
        if key not in cfg:
            continue
        section = cfg[key]
        try:
            payload[key] = OmegaConf.to_container(section, resolve=True)
        except Exception:
            payload[key] = OmegaConf.to_container(section, resolve=False)
    OmegaConf.save(OmegaConf.create(payload), path)
    return path


def split_shards(bags: list[Path], workers: int) -> list[list[Path]]:
    workers = max(1, min(int(workers), len(bags)))
    shards: list[list[Path]] = [[] for _ in range(workers)]
    for index, bag in enumerate(bags):
        shards[index % workers].append(bag)
    return shards


def convert_shard(job: dict) -> dict:
    """Child-process entry. ``job`` must be pickleable (paths as strings)."""
    started = time.perf_counter()
    shard_id = int(job["shard_id"])
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        kind = job["kind"]
        if kind == "roban":
            _convert_roban_shard(job)
        elif kind == "kuavo":
            _convert_kuavo_shard(job)
        else:
            raise ValueError(f"Unsupported converter kind: {kind}")
        return {
            "ok": True,
            "shard_id": shard_id,
            "lerobot_dir": str(Path(job["shard_target_dir"]) / "lerobot"),
            "bags": list(job["bag_files"]),
            "elapsed_s": time.perf_counter() - started,
        }
    except Exception:
        return {
            "ok": False,
            "shard_id": shard_id,
            "lerobot_dir": job.get("shard_target_dir"),
            "bags": job.get("bag_files", []),
            "elapsed_s": time.perf_counter() - started,
            "error": traceback.format_exc(),
        }


def _writer_config(job: dict):
    from kuavo_data.CvtRosbag2Lerobot import DatasetConfig

    return DatasetConfig(
        image_writer_processes=0,
        image_writer_threads=int(job["image_writer_threads"]),
    )


def _convert_roban_shard(job: dict) -> None:
    from omegaconf import OmegaConf

    from kuavo_data.CvtRosbag2Lerobot import setup_logging
    from kuavo_data.roban_data.config import load_roban_config
    from kuavo_data.roban_data.CvtRosbag2Lerobot import (
        create_empty_roban_dataset,
        populate_roban_dataset,
    )

    setup_logging()
    cfg = OmegaConf.load(job["cfg_yaml"])
    config = load_roban_config(cfg)
    bag_files = [Path(path) for path in job["bag_files"]]
    shard_target = Path(job["shard_target_dir"])
    output_dir = shard_target / "lerobot"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset = create_empty_roban_dataset(
        repo_id=f"lerobot/{shard_target.name}",
        config=config,
        dataset_config=_writer_config(job),
        root=str(output_dir),
    )
    populate_roban_dataset(
        dataset,
        bag_files,
        config,
        config.task_description,
        chunk_size=int(job["chunk_size"]),
    )
    dataset.finalize()


def _convert_kuavo_shard(job: dict) -> None:
    from omegaconf import OmegaConf

    import kuavo_data.CvtRosbag2Lerobot as cvt
    from kuavo_data.common import kuavo_dataset as kuavo

    cvt.setup_logging()
    cfg = OmegaConf.load(job["cfg_yaml"])
    kuavo.init_parameters(cfg)
    # Workers skip Hydra main; set the same global create_empty_dataset_chunked reads.
    from kuavo_data.common.config_platform import get_arm_joint_slice, get_lower_body_joint_slice

    arm_start, arm_end = get_arm_joint_slice(kuavo.PLATFORM_TYPE)
    arm_dof = arm_end - arm_start
    arm_dof_per_side = arm_dof // 2
    if len(kuavo.DEFAULT_ARM_JOINT_NAMES) == arm_dof:
        arm_names = kuavo.DEFAULT_ARM_JOINT_NAMES
    else:
        arm_names = [
            f"{side}_arm_joint_{joint_index}"
            for side in ("left", "right")
            for joint_index in range(arm_dof_per_side)
        ]
    eef_names = (
        kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES if kuavo.USE_LEJU_CLAW
        else kuavo.DEFAULT_SG100_JOINT_NAMES if kuavo.USE_SG100
        else kuavo.DEFAULT_DEXHAND_JOINT_NAMES
    )
    eef_slices = kuavo.SLICE_CLAW if kuavo.USE_LEJU_CLAW else kuavo.SLICE_DEX
    joint_names: list[str] = []
    selected_sides = (0, 1) if kuavo.CONTROL_HAND_SIDE == "both" else ((0,) if kuavo.CONTROL_HAND_SIDE == "left" else (1,))
    for side_index in selected_sides:
        arm_name_start = side_index * arm_dof_per_side
        joint_names.extend(arm_names[arm_name_start:arm_name_start + arm_dof_per_side])
        eef_start, eef_end = eef_slices[side_index]
        joint_names.extend(eef_names[eef_start:eef_end])
    if kuavo.USE_5W_WHOLEBODY:
        lower_start, lower_end = get_lower_body_joint_slice(kuavo.PLATFORM_TYPE)
        joint_names.extend(
            f"{kuavo.DEFAULT_LOWER_BODY_JOINT_NAME_PREFIX}_{index}"
            for index in range(lower_start, lower_end)
        )
    cvt.DEFAULT_JOINT_NAMES_LIST = joint_names
    bag_files = [Path(path) for path in job["bag_files"]]
    shard_target = Path(job["shard_target_dir"])
    output_dir = shard_target / "lerobot"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset = cvt.create_empty_dataset_chunked(
        repo_id=f"lerobot/{shard_target.name}",
        robot_type=f"kuavo-{kuavo.PLATFORM_TYPE}",
        mode="video",
        dataset_config=_writer_config(job),
        root=str(output_dir),
    )
    cvt.populate_dataset_chunked(
        dataset,
        bag_files,
        task=kuavo.TASK_DESCRIPTION,
        chunk_size=int(job["chunk_size"]),
        platform_type=kuavo.PLATFORM_TYPE,
    )
    dataset.finalize()


def _move_dataset(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(source), str(dest))


def run_parallel_conversion(
    *,
    kind: str,
    cfg: DictConfig,
    bag_files: list[Path],
    target_dir: Path,
    output_dir: Path,
    chunk_size: int,
    workers: int,
    keep_shards: bool = False,
    image_writer_threads: int = 8,
) -> Path:
    if kind not in ("roban", "kuavo"):
        raise ValueError(f"Unsupported converter kind: {kind}")
    if not bag_files:
        raise FileNotFoundError("No .bag files to convert")

    shard_root = target_dir / SHARD_DIRNAME
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    cfg_yaml = dump_resolved_cfg(cfg, shard_root)
    shards = split_shards(bag_files, workers)

    log_print.info("=== Parallel rosbag conversion ===")
    log_print.info("kind=%s bags=%d workers=%d chunk_size=%d", kind, len(bag_files), len(shards), chunk_size)
    for shard_id, shard_bags in enumerate(shards):
        log_print.info("  shard %02d: %d bags", shard_id, len(shard_bags))
        for bag in shard_bags:
            log_print.info("    %s", bag.name)

    jobs = []
    for shard_id, shard_bags in enumerate(shards):
        shard_target = shard_root / f"shard_{shard_id:02d}"
        shard_target.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "kind": kind,
                "shard_id": shard_id,
                "cfg_yaml": str(cfg_yaml),
                "bag_files": [str(path) for path in shard_bags],
                "shard_target_dir": str(shard_target),
                "chunk_size": int(chunk_size),
                "image_writer_threads": int(image_writer_threads),
            }
        )

    started = time.perf_counter()
    ctx = mp.get_context("spawn")
    results: list[dict | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=ctx) as pool:
        future_to_id = {pool.submit(convert_shard, job): job["shard_id"] for job in jobs}
        for future in as_completed(future_to_id):
            result = future.result()
            shard_id = int(result["shard_id"])
            results[shard_id] = result
            status = "ok" if result["ok"] else "FAIL"
            log_print.info(
                "[shard %02d] %s  %d bags  %.1fs",
                shard_id,
                status,
                len(result.get("bags", [])),
                result["elapsed_s"],
            )
            if not result["ok"]:
                log_print.error(result.get("error", ""))

    failed = [result for result in results if result is None or not result["ok"]]
    if failed:
        raise RuntimeError(
            f"{len(failed)} shard(s) failed; datasets kept at {shard_root}"
        )

    success_dirs = [Path(result["lerobot_dir"]) for result in results if result is not None]
    log_print.info("merging shards ...")
    merge_started = time.perf_counter()
    if len(success_dirs) == 1:
        _move_dataset(success_dirs[0], output_dir)
    else:
        from kuavo_data.CvtRosbag2Lerobot import merge_lerobot_datasets

        merge_lerobot_datasets(success_dirs, output_dir)
    log_print.info("merged %s in %.1fs", output_dir, time.perf_counter() - merge_started)

    if not keep_shards and shard_root.exists():
        shutil.rmtree(shard_root)
        log_print.info("removed %s", shard_root)

    log_print.info("parallel conversion done in %.1fs  output=%s", time.perf_counter() - started, output_dir)
    return output_dir
