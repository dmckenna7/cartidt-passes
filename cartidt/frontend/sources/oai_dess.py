"""OAI 3D DESS volumes and WORMS grading labels (Ref: Sec. IV.A-1, Table II).

`OAIDataset` reads the 507-volume OAI-ZIB segmentation cohort.
`OAIGradingDataset` reads the 4796-subject WORMS-graded cohort with the official
3212 / 802 / 782 train/val/test split. The CSV manifest format is described in
`scripts/prepare_oai.sh`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from cartidt.frontend.conditioning.intensity_norm import zscore_per_volume
from cartidt.frontend.conditioning.spatial_aug import VolumeAugment
from cartidt.frontend.schema import Batch, SubjectMeta

_NUM_COMPARTMENTS = 6
_OAI_SEG_SUBJECTS = 507
_OAI_GRADING_TRAIN = 3212
_OAI_GRADING_VAL = 802
_OAI_GRADING_TEST = 782


@dataclass(frozen=True, slots=True)
class OAIRecord:
    subject_id: str
    timepoint: int
    volume_path: Path
    segmentation_path: Path
    grade_path: Path | None


def _read_manifest(manifest_csv: Path, root: Path) -> list[OAIRecord]:
    records: list[OAIRecord] = []
    with manifest_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            grade = row.get("grade_path") or None
            records.append(
                OAIRecord(
                    subject_id=row["subject_id"],
                    timepoint=int(row["timepoint"]),
                    volume_path=root / row["volume_path"],
                    segmentation_path=root / row["segmentation_path"],
                    grade_path=root / grade if grade else None,
                )
            )
    return records


def _to_tensor(path: Path) -> torch.Tensor:
    array = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array))


def _resample_to_target(volume: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
    while volume.ndim < 5:
        volume = volume.unsqueeze(0)
    resampled = torch.nn.functional.interpolate(volume, size=target, mode="trilinear", align_corners=False)
    return resampled.squeeze(0).squeeze(0)


class OAIDataset(Dataset[Batch]):
    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        target_shape: tuple[int, int, int] = (160, 384, 384),
        augment: VolumeAugment | None = None,
    ) -> None:
        self.root = Path(root)
        self.records = _read_manifest(Path(manifest), self.root)
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Batch:
        rec = self.records[index]
        volume = _to_tensor(rec.volume_path)
        seg = _to_tensor(rec.segmentation_path).long()
        if volume.shape[-3:] != self.target_shape:
            volume = _resample_to_target(volume, self.target_shape)
            seg = (
                _resample_to_target(seg.float().unsqueeze(0).unsqueeze(0), self.target_shape).squeeze().long()
            )
        volume = zscore_per_volume(volume).unsqueeze(0)
        if self.augment is not None:
            volume, seg = self.augment(volume, seg)
        grade = self._read_grade(rec.grade_path)
        meta = SubjectMeta(
            subject_id=rec.subject_id,
            timepoint=rec.timepoint,
            cohort="OAI-ZIB",
            voxel_spacing_mm=(0.365, 0.365, 0.7),
        )
        return Batch(
            volume=volume,
            segmentation=seg,
            grade=grade,
            meta=(meta,),
        )

    @staticmethod
    def _read_grade(path: Path | None) -> torch.Tensor:
        if path is None:
            return torch.full((_NUM_COMPARTMENTS,), -1, dtype=torch.long)
        return torch.from_numpy(np.loadtxt(path, dtype=np.int64)).long()


class OAIGradingDataset(Dataset[Batch]):
    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        split: str = "train",
        target_shape: tuple[int, int, int] = (160, 384, 384),
        augment: VolumeAugment | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split}")
        self.root = Path(root)
        self.records = self._read_split(Path(manifest), self.root, split)
        self.target_shape = target_shape
        self.augment = augment
        self._sanity_check_split(split)

    @staticmethod
    def _read_split(manifest: Path, root: Path, split: str) -> list[OAIRecord]:
        out: list[OAIRecord] = []
        with manifest.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"] != split:
                    continue
                grade = row.get("grade_path") or None
                out.append(
                    OAIRecord(
                        subject_id=row["subject_id"],
                        timepoint=int(row["timepoint"]),
                        volume_path=root / row["volume_path"],
                        segmentation_path=root / row.get("segmentation_path", ""),
                        grade_path=root / grade if grade else None,
                    )
                )
        return out

    def _sanity_check_split(self, split: str) -> None:
        expected = {"train": _OAI_GRADING_TRAIN, "val": _OAI_GRADING_VAL, "test": _OAI_GRADING_TEST}[split]
        if 0 < len(self.records) != expected:
            return

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Batch:
        rec = self.records[index]
        volume = _to_tensor(rec.volume_path)
        if volume.shape[-3:] != self.target_shape:
            volume = _resample_to_target(volume, self.target_shape)
        volume = zscore_per_volume(volume).unsqueeze(0)
        seg = torch.zeros(self.target_shape, dtype=torch.long)
        if self.augment is not None:
            volume, seg = self.augment(volume, seg)
        grade = OAIDataset._read_grade(rec.grade_path)
        meta = SubjectMeta(
            subject_id=rec.subject_id,
            timepoint=rec.timepoint,
            cohort="OAI",
            voxel_spacing_mm=(0.365, 0.365, 0.7),
        )
        return Batch(volume=volume, segmentation=seg, grade=grade, meta=(meta,))
