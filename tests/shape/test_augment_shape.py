from __future__ import annotations

import torch

from cartidt.frontend.conditioning.spatial_aug import VolumeAugment, VolumeAugmentConfig


def test_augment_preserves_volume_shape() -> None:
    aug = VolumeAugment(VolumeAugmentConfig())
    volume = torch.randn(1, 16, 32, 32)
    seg = torch.zeros(16, 32, 32, dtype=torch.long)
    out_vol, out_seg = aug(volume, seg)
    assert out_vol.shape == volume.shape
    assert out_seg.shape == seg.shape


def test_augment_segmentation_stays_int() -> None:
    aug = VolumeAugment(VolumeAugmentConfig())
    out_vol, out_seg = aug(torch.randn(1, 16, 32, 32), torch.ones(16, 32, 32, dtype=torch.long))
    assert out_seg.dtype == torch.long
    assert out_vol.dtype == torch.float32
