"""IWOAI 2019 Challenge partition of OAI-ZIB (Ref: Sec. IV.A-3, Table II).

88 subjects × 2 time-points → 176 volumes, split 60-14-14 ×2 train/val/test.
We honour the official IWOAI holdout test set; the 404 ZIB volumes that are
*not* in this 88-subject IWOAI set are used to fit `L_seg` while `L_EDL` is
fit on the 3212 OAI grading-train subjects (Sec IV.A ¶3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cartidt.frontend.conditioning.spatial_aug import VolumeAugment
from cartidt.frontend.sources.oai_dess import OAIDataset


@dataclass(frozen=True, slots=True)
class IWOAIPartition:
    seg_train_zib_only: OAIDataset
    seg_iwoai_train: OAIDataset
    seg_iwoai_val: OAIDataset
    seg_iwoai_test: OAIDataset


def iwoai_partition(
    root: str | Path,
    manifests_dir: str | Path,
    target_shape: tuple[int, int, int] = (160, 384, 384),
    augment: VolumeAugment | None = None,
) -> IWOAIPartition:
    manifests = Path(manifests_dir)
    return IWOAIPartition(
        seg_train_zib_only=OAIDataset(root, manifests / "zib_404_train.csv", target_shape, augment),
        seg_iwoai_train=OAIDataset(root, manifests / "iwoai_train.csv", target_shape, augment),
        seg_iwoai_val=OAIDataset(root, manifests / "iwoai_val.csv", target_shape, None),
        seg_iwoai_test=OAIDataset(root, manifests / "iwoai_test.csv", target_shape, None),
    )
