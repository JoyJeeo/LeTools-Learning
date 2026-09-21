# Copyright (C) 2026 Xiaomi Corporation.
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from mmengine import Config

from mibot.models import MIMODEL
from mibot.server.runtime.server import Server
from mibot.utils.action_layout import ActionLayout
from mibot.utils.io import build_action_mask, validate_quantiles, validate_stats


def strip_prefix(state_dict, prefix):
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def resolve_model_files(model_dir):
    root = Path(model_dir).expanduser().resolve()
    if root.name.endswith(".ckpt"):
        config_path = root.parent / "config.py"
        state_path = root / "checkpoint" / "mp_rank_00_model_states.pt"
    else:
        config_path = root / "config.py"
        state_path = root / "last.ckpt" / "checkpoint" / "mp_rank_00_model_states.pt"
        if not state_path.is_file():
            candidates = list(root.glob("*.ckpt/checkpoint/mp_rank_00_model_states.pt"))
            if candidates:
                def checkpoint_step(path):
                    match = re.search(r"step=(\d+)", path.parents[1].name)
                    return int(match.group(1)) if match else -1

                state_path = max(candidates, key=checkpoint_step)

    missing = [str(path) for path in (config_path, state_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Xiaomi post-training checkpoint is incomplete: {missing}")
    return config_path, state_path


def load_model(model_dir, device, vlm_path=None):
    config_path, state_path = resolve_model_files(model_dir)
    cfg = Config.fromfile(config_path)
    if vlm_path:
        cfg.model.params.model.vlm_path = vlm_path
    model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)
    ckpt = torch.load(
        state_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )["module"]
    print(model.load_state_dict(strip_prefix(ckpt, "model."), strict=True))
    return cfg, model.eval().to(device)


def load_stats(cfg, device):
    data = cfg.data.params.train_datasets
    action_length = int(data.get("action_length", cfg.data.params.get("action_length", 30)))
    mean, std = validate_stats(data.mean, data.std, action_length)
    q01, q99 = validate_quantiles(data.q01, data.q99)
    action_mask = (
        ActionLayout(data.modality).action_mask(action_length)
        if data.get("modality")
        else build_action_mask(action_length)
    )
    return (
        torch.tensor(mean, device=device),
        torch.tensor(std, device=device),
        torch.tensor(q01, device=device),
        torch.tensor(q99, device=device),
        torch.from_numpy(action_mask).to(device),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model dir.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda:0"
    cfg, model = load_model(args.model, device)
    mean, std, q01, q99, action_mask = load_stats(cfg, device)

    try:
        server = Server(args.host, args.port, model, mean, std, q01, q99, action_mask, device)
        print(f"Starting server on {args.host}:{args.port}")
        server.run()
    except OSError as error:
        if error.errno == 98:
            print(f"Error: Port {args.port} is already in use. Please choose a different port.")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("Server interrupted")
