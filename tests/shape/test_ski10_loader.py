from __future__ import annotations

import csv

import nibabel as nib
import numpy as np
import pytest

from cartidt.frontend.sources.ski10 import SKI10Dataset


@pytest.fixture()
def ski10_root(tmp_path):
    vol = nib.Nifti1Image(np.zeros((8, 16, 16), dtype=np.float32), np.eye(4))
    seg = nib.Nifti1Image(np.zeros((8, 16, 16), dtype=np.int16), np.eye(4))
    nib.save(vol, str(tmp_path / "k.nii.gz"))
    nib.save(seg, str(tmp_path / "k_seg.nii.gz"))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subject_id", "volume_path", "segmentation_path"])
        writer.writeheader()
        writer.writerow(
            {"subject_id": "SKI10-01", "volume_path": "k.nii.gz", "segmentation_path": "k_seg.nii.gz"}
        )
    return tmp_path, manifest


def test_ski10_dataset_yields_batch(ski10_root) -> None:
    root, manifest = ski10_root
    ds = SKI10Dataset(root=root, manifest=manifest, target_shape=(8, 16, 16))
    sample = ds[0]
    assert sample.volume.shape == (1, 8, 16, 16)
    assert sample.meta[0].cohort == "SKI10"
