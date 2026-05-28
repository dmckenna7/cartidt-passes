from __future__ import annotations

import torch

from cartidt.passes.segment.upernet import UPerNetDecoder


def test_upernet_outputs_target_shape() -> None:
    dec = UPerNetDecoder(in_channels=[32, 32, 32, 32], decoder_dim=32, num_classes=7)
    feats = [
        torch.randn(1, 32, 4, 32, 32),
        torch.randn(1, 32, 4, 16, 16),
        torch.randn(1, 32, 4, 8, 8),
        torch.randn(1, 32, 4, 4, 4),
    ]
    out = dec(feats, target_shape=(16, 64, 64))
    assert out.shape == (1, 7, 16, 64, 64)
