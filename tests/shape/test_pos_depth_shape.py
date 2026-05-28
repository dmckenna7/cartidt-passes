from __future__ import annotations

import torch

from cartidt.passes.encode.depth_posembed import DepthInterpolatedPosEmbed


def test_pos_depth_emits_correct_token_count() -> None:
    emb = DepthInterpolatedPosEmbed(
        embed_dim=32, patch_size=16, depth_stride=4, target_hw=(64, 64), target_depth=16
    )
    out = emb(depth=8)
    assert out.shape == (8, 16, 32)


def test_pos_depth_dtype_matches_input() -> None:
    emb = DepthInterpolatedPosEmbed(
        embed_dim=16, patch_size=16, depth_stride=2, target_hw=(32, 32), target_depth=4
    )
    out = emb(depth=4)
    assert out.dtype == torch.float32
