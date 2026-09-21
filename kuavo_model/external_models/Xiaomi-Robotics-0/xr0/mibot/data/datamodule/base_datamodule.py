# Copyright (C) 2026 Xiaomi Corporation.
from copy import deepcopy

from lightning import LightningDataModule
from mmengine import Config, DATASETS
from torch.utils.data import DataLoader

from mibot.data.collate.custom_collate import CustomCollate
from mibot.data.datasets.json_dataset import JsonDataset


@DATASETS.register_module()
class BaseDataModule(LightningDataModule):
    """Training dataloader for the official JSON or LeRobot dataset."""

    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params: Config = params
        self.batch_size: int = params.train_datasets.get("batch_size", 16)
        self.collate_fn = CustomCollate(params.get("processor_path", "Qwen/Qwen3-VL-4B-Instruct"))

    def train_dataloader(self) -> DataLoader:
        dataset_type = str(self.params.get("type", "json"))
        if dataset_type == "json":
            dataset_cls = JsonDataset
        elif dataset_type == "lerobot":
            from mibot.data.datasets.lerobot_dataset import XR0LeRobotDataset

            dataset_cls = XR0LeRobotDataset
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        train_set = dataset_cls(deepcopy(self.params))
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            prefetch_factor=8,
            collate_fn=self.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> list:
        return []
