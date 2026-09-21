"""Convert Roban ROS1 bags to a LeRobot v3 dataset."""

import gc
import logging
import os
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

import lerobot_patches.custom_patches  # noqa: F401 - apply dataset compatibility patches
from kuavo_data.CvtRosbag2Lerobot import (
    DEFAULT_DATASET_CONFIG,
    DatasetConfig,
    ensure_clean_lerobot_output_dir,
    infer_local_repo_id,
    merge_lerobot_datasets,
    prepare_resume_target,
    resolve_lerobot_output_dir,
    resume_dataset_chunked,
    setup_logging,
)
from kuavo_data.common.parallel_cvt import parallel_options, run_parallel_conversion
from kuavo_data.roban_data.config import (
    RobanDataConfig,
    get_roban_feature_names,
    load_roban_config,
)
from kuavo_data.roban_data.roban_dataset import (
    CAMERA_KEY,
    RobanRosbagReader,
    build_wholebody_state_action,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


log_print = logging.getLogger(__name__)


def create_empty_roban_dataset(
    repo_id: str,
    config: RobanDataConfig,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
) -> LeRobotDataset:
    """Create the Roban wholebody LeRobot schema."""
    state_names, action_names = get_roban_feature_names()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": {"state_names": list(state_names)},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": {"action_names": list(action_names)},
        },
        f"observation.images.{CAMERA_KEY}": {
            "dtype": mode,
            "shape": (3, config.resize_height, config.resize_width),
            "names": ["channels", "height", "width"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.train_hz,
        robot_type="roban",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )


def populate_roban_dataset(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    config: RobanDataConfig,
    task: str,
    episodes: list[int] | None = None,
    chunk_size: int = 100,
) -> LeRobotDataset:
    if episodes is None:
        episodes = list(range(len(bag_files)))

    reader = RobanRosbagReader(config)
    build_state_action = build_wholebody_state_action
    failed_bags: list[str] = []
    process = None
    try:
        import psutil

        process = psutil.Process(os.getpid())
    except ImportError:
        pass

    def log_memory(prefix: str) -> None:
        if process is not None:
            log_print.info(
                "%s Memory: %.2f MB",
                prefix,
                process.memory_info().rss / 1024 / 1024,
            )

    for episode_index in tqdm.tqdm(episodes):
        bag_path = bag_files[episode_index]
        frames_buffer: list[dict] = []
        frame_count = 0
        log_print.warning("Processing %s", bag_path)
        log_memory("Before processing")

        def on_frame(aligned_frame: dict, frame_index: int) -> None:
            nonlocal frame_count
            try:
                state, action = build_state_action(aligned_frame)
                image_item = aligned_frame.get(CAMERA_KEY)
                if not image_item or "data" not in image_item:
                    raise ValueError(f"Aligned field `{CAMERA_KEY}` is missing")
            except ValueError as exc:
                log_print.warning(
                    "Episode %s frame %s skipped: %s",
                    episode_index,
                    frame_index,
                    exc,
                )
                return

            frames_buffer.append(
                {
                    "observation.state": torch.from_numpy(state),
                    "action": torch.from_numpy(action),
                    f"observation.images.{CAMERA_KEY}": image_item["data"],
                    "task": task,
                }
            )
            frame_count += 1

        def flush_frames() -> None:
            for frame in frames_buffer:
                dataset.add_frame(frame)
            frames_buffer.clear()
            gc.collect()

        try:
            reader.process_rosbag_chunked(
                bag_file=str(bag_path),
                frame_callback=on_frame,
                chunk_size=chunk_size,
                save_callback=flush_frames,
            )
            flush_frames()
            if frame_count == 0:
                raise ValueError("No complete aligned frames were produced")
            dataset.save_episode()
            log_print.info("Episode %s completed: %s frames", episode_index, frame_count)
        except Exception as exc:
            log_print.exception("Error processing %s: %s", bag_path, exc)
            failed_bags.append(str(bag_path))
        finally:
            frames_buffer.clear()
            gc.collect()
            log_memory("After episode")

    if failed_bags:
        raise RuntimeError("Failed to convert bag: " + ", ".join(failed_bags))
    return dataset


def _select_bags(raw_dir: Path, n: int | None) -> list[Path]:
    bag_files = [Path(path) for path in RobanRosbagReader.list_bag_files(str(raw_dir))]
    if not bag_files:
        raise FileNotFoundError(f"No .bag files found in: {raw_dir}")
    if isinstance(n, int) and n > 0:
        n = min(n, len(bag_files))
        indices = np.random.choice(len(bag_files), n, replace=False)
        bag_files = [bag_files[index] for index in indices]
    return bag_files


def port_roban_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    config: RobanDataConfig,
    *,
    root: str,
    task: str,
    n: int | None = None,
    chunk_size: int = 100,
) -> LeRobotDataset:
    dataset = create_empty_roban_dataset(repo_id, config, root=root)
    populate_roban_dataset(
        dataset,
        _select_bags(raw_dir, n),
        config,
        task,
        chunk_size=chunk_size,
    )
    dataset.finalize()
    return dataset


def resume_roban_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    config: RobanDataConfig,
    *,
    resume_root: Path,
    task: str,
    n: int | None = None,
    chunk_size: int = 100,
) -> LeRobotDataset:
    dataset = resume_dataset_chunked(repo_id=repo_id, root=str(resume_root))
    populate_roban_dataset(
        dataset,
        _select_bags(raw_dir, n),
        config,
        task,
        chunk_size=chunk_size,
    )
    dataset.finalize()
    return dataset


@hydra.main(
    config_path="../../configs/data/",
    config_name="RobanRosbag2Lerobot",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    setup_logging()
    config = load_roban_config(cfg)
    mode = str(cfg.rosbag.get("mode", "normal")).strip().lower()
    raw_dir_value = cfg.rosbag.get("rosbag_dir")
    raw_dir = Path(raw_dir_value).expanduser().resolve() if raw_dir_value else None
    target_dir = Path(cfg.rosbag.target_dir).expanduser().resolve()
    output_dir = resolve_lerobot_output_dir(target_dir)
    n = cfg.rosbag.get("num_used", None)
    chunk_size = int(cfg.rosbag.get("chunk_size", 100))

    log_print.info("Roban conversion mode: %s", mode)
    log_print.info("LeRobot output dir: %s", output_dir)
    use_parallel, parallel_workers, keep_shards, writer_threads = parallel_options(cfg)

    if mode == "normal":
        if raw_dir is None:
            raise ValueError("normal mode requires `rosbag.rosbag_dir`")
        if raw_dir == output_dir:
            raise ValueError("rosbag_dir and target_dir/lerobot cannot be the same")
        output_dir = ensure_clean_lerobot_output_dir(target_dir)
        ran_parallel = False
        if use_parallel and parallel_workers > 1:
            bag_files = _select_bags(raw_dir, n)
            if len(bag_files) > 1:
                log_print.info("Parallel conversion: workers=%s", min(parallel_workers, len(bag_files)))
                run_parallel_conversion(
                    kind="roban",
                    cfg=cfg,
                    bag_files=bag_files,
                    target_dir=target_dir,
                    output_dir=output_dir,
                    chunk_size=chunk_size,
                    workers=parallel_workers,
                    keep_shards=keep_shards,
                    image_writer_threads=writer_threads,
                )
                ran_parallel = True
            else:
                log_print.info("Falling back to sequential conversion (need >1 bag).")
        elif use_parallel:
            log_print.info("Falling back to sequential conversion (need parallel_workers>1).")
        if not ran_parallel:
            port_roban_rosbag_chunked(
                raw_dir,
                f"lerobot/{raw_dir.name}",
                config,
                root=str(output_dir),
                task=config.task_description,
                n=n,
                chunk_size=chunk_size,
            )
        return

    if mode == "resume":
        if raw_dir is None:
            raise ValueError("resume mode requires `rosbag.rosbag_dir`")
        source_value = cfg.rosbag.get("lerobot_dir_resume")
        if not source_value:
            raise ValueError("resume mode requires `rosbag.lerobot_dir_resume`")
        resume_target = prepare_resume_target(Path(source_value), output_dir)
        resume_roban_rosbag_chunked(
            raw_dir,
            infer_local_repo_id(output_dir),
            config,
            resume_root=resume_target,
            task=config.task_description,
            n=n,
            chunk_size=chunk_size,
        )
        return

    if mode == "merge":
        source_values = cfg.rosbag.get("lerobot_dir_merge", [])
        merge_lerobot_datasets([Path(value) for value in source_values], output_dir)
        return

    raise ValueError(f"Unsupported rosbag.mode: {mode}. Expected normal, resume, or merge")


if __name__ == "__main__":
    np.random.seed(42)
    main()
