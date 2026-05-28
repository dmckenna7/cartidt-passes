from __future__ import annotations

import csv

import nibabel as nib
import numpy as np
import pytest

from cartidt.frontend.sources.oai_dess import OAIDataset


@pytest.fixture()
def tiny_manifest(tmp_path):
    vol = nib.Nifti1Image(np.zeros((8, 16, 16), dtype=np.float32), np.eye(4))
    seg = nib.Nifti1Image(np.zeros((8, 16, 16), dtype=np.int16), np.eye(4))
    vol_path = tmp_path / "vol.nii.gz"
    seg_path = tmp_path / "seg.nii.gz"
    nib.save(vol, str(vol_path))
    nib.save(seg, str(seg_path))
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["subject_id", "timepoint", "volume_path", "segmentation_path", "grade_path"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject_id": "OAI-0001",
                "timepoint": 0,
                "volume_path": "vol.nii.gz",
                "segmentation_path": "seg.nii.gz",
                "grade_path": "",
            }
        )
    return tmp_path, csv_path


def test_oai_returns_batch_with_volume_and_seg(tiny_manifest) -> None:
    root, manifest = tiny_manifest
    ds = OAIDataset(root=root, manifest=manifest, target_shape=(8, 16, 16))
    sample = ds[0]
    assert sample.volume.shape == (1, 8, 16, 16)
    assert sample.segmentation.shape == (8, 16, 16)
    assert sample.meta[0].cohort == "OAI-ZIB"
