"""SKI10 challenge data — femoral + tibial cartilage only (Ref: Sec. IV.A-2, Table II)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from cartidt.frontend.conditioning.intensity_norm import zscore_per_volume
from cartidt.frontend.schema import Batch, SubjectMeta


@dataclass(frozen=True, slots=True)
class SKI10Record:
    subject_id: str
    volume_path: Path
    segmentation_path: Path


class SKI10Dataset(Dataset[Batch]):
    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        target_shape: tuple[int, int, int] = (160, 384, 384),
    ) -> None:
        self.root = Path(root)
        self.target_shape = target_shape
        self.records = self._read_manifest(Path(manifest))

    def _read_manifest(self, manifest: Path) -> list[SKI10Record]:
        out: list[SKI10Record] = []
        with manifest.open(newline="") as fh:
            for row in csv.DictReader(fh):
                out.append(
                    SKI10Record(
                        subject_id=row["subject_id"],
                        volume_path=self.root / row["volume_path"],
                        segmentation_path=self.root / row["segmentation_path"],
                    )
                )
        return out

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Batch:
        rec = self.records[index]
        volume = torch.from_numpy(
            np.ascontiguousarray(np.asarray(nib.load(str(rec.volume_path)).get_fdata(), dtype=np.float32))
        )
        seg = torch.from_numpy(
            np.ascontiguousarray(np.asarray(nib.load(str(rec.segmentation_path)).get_fdata(), dtype=np.int64))
        )
        if volume.shape[-3:] != self.target_shape:
            volume = torch.nn.functional.interpolate(
                volume.reshape(1, 1, *volume.shape[-3:]),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze()
            seg = (
                torch.nn.functional.interpolate(
                    seg.float().reshape(1, 1, *seg.shape[-3:]),
                    size=self.target_shape,
                    mode="nearest",
                )
                .squeeze()
                .long()
            )
        volume = zscore_per_volume(volume).unsqueeze(0)
        meta = SubjectMeta(
            subject_id=rec.subject_id,
            timepoint=0,
            cohort="SKI10",
            voxel_spacing_mm=(0.4, 0.4, 1.0),
        )
        return Batch(
            volume=volume,
            segmentation=seg,
            grade=torch.full((6,), -1, dtype=torch.long),
            meta=(meta,),
        )
