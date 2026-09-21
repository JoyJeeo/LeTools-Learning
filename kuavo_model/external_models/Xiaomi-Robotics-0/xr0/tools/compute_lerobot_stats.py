#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from mibot.utils.action_layout import ActionLayout
from mibot.utils.io import ACTION_EPS


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _accumulate_episode(states, actions, layout, total, square, count, action_length):
    for frame, state in enumerate(states):
        steps = min(action_length, len(actions) - frame)
        value = layout.pack_action(state, actions[frame : frame + steps]).astype(np.float64)
        total[:steps] += value
        square[:steps] += value * value
        count[:steps] += 1


def compute_stats(dataset: LeRobotDataset, layout: ActionLayout, action_length: int):
    total = np.zeros((action_length, layout.action_dim), dtype=np.float64)
    square = np.zeros_like(total)
    count = np.zeros(action_length, dtype=np.int64)
    episode = None
    states, actions = [], []

    for index in range(len(dataset)):
        item = dataset.get_raw_item(index)
        episode_index = int(item["episode_index"])
        if episode is not None and episode_index != episode:
            _accumulate_episode(states, actions, layout, total, square, count, action_length)
            states, actions = [], []
        episode = episode_index
        states.append(_numpy(item[layout.state_key]))
        actions.append(_numpy(item[layout.action_key]))
    if states:
        _accumulate_episode(states, actions, layout, total, square, count, action_length)
    if np.any(count == 0):
        raise ValueError("Dataset episodes are shorter than action_length")

    mean = total / count[:, None]
    std = np.sqrt(np.maximum(square / count[:, None] - mean * mean, 0.0))
    std[std < ACTION_EPS] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute XR0 action statistics from a LeRobot dataset.")
    parser.add_argument("--root", required=True, help="Local LeRobot dataset root")
    parser.add_argument("--modality_config", required=True, help="Robot modality JSON")
    parser.add_argument("--repo_id", default="", help="LeRobot repo id; defaults to lerobot/<root-name>")
    parser.add_argument("--action_length", type=int, default=30)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    with open(Path(args.modality_config).expanduser(), encoding="utf-8") as file:
        layout = ActionLayout(json.load(file))
    dataset = LeRobotDataset(
        repo_id=args.repo_id or f"lerobot/{root.name}",
        root=root,
        download_videos=False,
    )
    mean, std = compute_stats(dataset, layout, args.action_length)
    payload = {
        "action_length": args.action_length,
        "modality_signature": layout.signature,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    print(output)


if __name__ == "__main__":
    main()
