# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy

from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader, DistributedSampler

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate(
            params.get("processor_path", "Qwen/Qwen3-VL-4B-Instruct")
        )
        self.num_workers = int(params.get("num_workers", 8))
        self.prefetch_factor = int(params.get("prefetch_factor", 4))
        self.train_set = None

    def setup(self, stage=None) -> None:
        if stage in (None, "fit") and self.train_set is None:
            dataset_type = str(self.params.get("type", "json"))
            if dataset_type == "json":
                dataset_cls = JsonDataset
            elif dataset_type == "lerobot":
                from mibot.data.datasets.lerobot_dataset import XR1LeRobotDataset

                dataset_cls = XR1LeRobotDataset
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            self.train_set = dataset_cls(deepcopy(self.params))

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            self.setup("fit")
        sampler = DistributedSampler(self.train_set, shuffle=True, seed=42)
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
