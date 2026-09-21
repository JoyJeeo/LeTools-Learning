# Copyright (C) 2026 Xiaomi Corporation.
import json
import os
from typing import Any

from lightning import seed_everything
from lightning.pytorch.strategies import DeepSpeedStrategy
from mmengine import Config
from omegaconf import DictConfig, OmegaConf
from transformers.utils import logging

from mibot.utils.action_layout import ActionLayout

logger = logging.get_logger(__name__)


def helper(cfg: DictConfig) -> Config:
    cfg = Config(OmegaConf.to_container(cfg, resolve=True))
    resolve_data_files(cfg)
    process_save_cfg(cfg)
    seed_everything(cfg.trainer.pop("seed", 42) + int(os.environ.get("RANK", 0)), workers=True)
    strategy_helper(cfg)
    cfg.model.params.optimizer = cfg.trainer.pop("optimizer")
    cfg.model.params.scheduler = cfg.trainer.pop("scheduler")

    return cfg


def resolve_data_files(cfg: Config) -> None:
    """Embed the robot layout and normalization statistics in the checkpoint config."""
    data = cfg.data.params.train_datasets
    layout = None
    modality_path = data.get("modality_config_path")
    if modality_path:
        with open(os.path.expanduser(modality_path), encoding="utf-8") as file:
            data.modality = json.load(file)
        layout = ActionLayout(data.modality)
        if (layout.state_dim, layout.action_dim) != (60, 60):
            raise ValueError("modality model dimensions do not match XR1 model dimensions")

    stats_path = data.get("stats_path")
    if not stats_path:
        return
    with open(os.path.expanduser(stats_path), encoding="utf-8") as file:
        stats = json.load(file)
    if stats.get("action_length") != data.get("action_length", 30):
        raise ValueError("stats action_length does not match the data config")
    if layout and stats.get("modality_signature") != layout.signature:
        raise ValueError("stats modality does not match the data config")
    for key in ("mean", "std", "q01", "q99"):
        data[key] = stats[key]


def strategy_helper(cfg: Config) -> Config:
    strategy_type: str = cfg.trainer.strategy.type
    if strategy_type == "deepspeed":
        strategy = DeepSpeedStrategy(**cfg.trainer.strategy.params)
    else:
        raise TypeError("Unsupported strategy.")
    cfg.trainer.strategy = strategy
    return cfg


def fill_num_nodes(cfg: Config) -> None:
    if cfg.trainer.num_nodes > 0:
        assert cfg.trainer.num_nodes <= int(os.environ.get("MLP_WORKER_NUM", 1)), (
            "Number of nodes exceeds available workers"
        )
    else:
        cfg.trainer.num_nodes = int(os.environ.get("MLP_WORKER_NUM", 1))


def process_save_cfg(cfg: Config) -> None:
    root_dir = os.path.join(
        cfg.trainer.default_root_dir,
        f"project_{cfg.trainer.project}",
        cfg.trainer.exp_name,
    )
    cfg.trainer.default_root_dir = root_dir

    fill_num_nodes(cfg)

    os.environ["_max_steps"] = str(cfg.trainer.max_steps)

    os.makedirs(root_dir, exist_ok=True)
    cfg.dump(os.path.join(root_dir, "config.yaml"))
    cfg.dump(os.path.join(root_dir, "config.py"))
    cfg.dump(os.path.join("./assets", "config.py"))
