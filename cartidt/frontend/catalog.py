"""Dataset registry (Ref: Sec. IV.A, Table II)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from cartidt.frontend.conditioning.spatial_aug import VolumeAugment, VolumeAugmentConfig
from cartidt.frontend.schema import Batch
from cartidt.frontend.sources.iwoai_split import iwoai_partition
from cartidt.frontend.sources.oai_dess import OAIGradingDataset
from cartidt.frontend.sources.ski10 import SKI10Dataset


@dataclass
class LoaderBundle:
    train: DataLoader[Batch]
    val: DataLoader[Batch]
    test: DataLoader[Batch]


def _collate(items: list[Batch]) -> Batch:
    volume = torch.stack([b.volume for b in items], dim=0)
    seg = torch.stack([b.segmentation for b in items], dim=0)
    grade = torch.stack([b.grade for b in items], dim=0)
    meta = tuple(b.meta[0] for b in items)
    return Batch(volume=volume, segmentation=seg, grade=grade, meta=meta)


def build_loaders(cfg: DictConfig) -> LoaderBundle:
    name = str(cfg.data.name).lower()
    root = Path(cfg.data.root)
    target = tuple(int(x) for x in cfg.data.target_shape)
    if len(target) != 3:
        raise ValueError("target_shape must be (D, H, W)")
    target_shape = (target[0], target[1], target[2])
    aug = VolumeAugment(VolumeAugmentConfig(**dict(cfg.data.get("augment", {}))))
    bs = int(cfg.data.batch_size)
    workers = int(cfg.data.get("num_workers", 4))

    train_ds: Dataset[Batch]
    val_ds: Dataset[Batch]
    test_ds: Dataset[Batch]
    if name == "iwoai":
        part = iwoai_partition(root, cfg.data.manifests_dir, target_shape, aug)
        train_ds = part.seg_iwoai_train
        val_ds = part.seg_iwoai_val
        test_ds = part.seg_iwoai_test
    elif name == "oai_grading":
        train_ds = OAIGradingDataset(root, cfg.data.manifest, "train", target_shape, aug)
        val_ds = OAIGradingDataset(root, cfg.data.manifest, "val", target_shape)
        test_ds = OAIGradingDataset(root, cfg.data.manifest, "test", target_shape)
    elif name == "ski10":
        train_ds = SKI10Dataset(root, cfg.data.manifests_dir + "/ski10_train.csv", target_shape)
        val_ds = SKI10Dataset(root, cfg.data.manifests_dir + "/ski10_train.csv", target_shape)
        test_ds = SKI10Dataset(root, cfg.data.manifests_dir + "/ski10_test.csv", target_shape)
    else:
        raise KeyError(f"unknown data.name: {name}")

    def _build(ds: Dataset[Batch], shuffle: bool) -> DataLoader[Batch]:
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            collate_fn=_collate,
        )

    return LoaderBundle(train=_build(train_ds, True), val=_build(val_ds, False), test=_build(test_ds, False))
