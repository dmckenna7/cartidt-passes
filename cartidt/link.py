"""Assembled CartiDT model and its OmegaConf builder (Ref: Sec. III, III.B)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from omegaconf import DictConfig

from cartidt.passes.encode.dinov2_weights import load_frozen_dinov2_large
from cartidt.passes.encode.vit_adapter import LoRAViTBackbone
from cartidt.passes.evidence.dirichlet_head import EvidenceHead
from cartidt.passes.segment.upernet import UPerNetDecoder
from cartidt.passes.twin.anatomy_graph import NUM_NODES
from cartidt.passes.twin.graph_sage import BiomechSAGE
from cartidt.passes.twin.region_features import compartment_masked_gap


@dataclass(frozen=True, slots=True)
class CartiDTConfig:
    num_seg_classes: int = 7
    num_grades: int = 4
    decoder_dim: int = 256
    sage_hidden: int = 256
    sage_layers: int = 3
    node_to_label: tuple[tuple[int, int], ...] = (
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 4),
        (7, 5),
        (8, 6),
        (9, 5),
        (10, 5),
        (11, 6),
        (12, 6),
        (13, 4),
    )


class CartiDT(nn.Module):
    def __init__(
        self,
        backbone: LoRAViTBackbone,
        decoder: UPerNetDecoder,
        sage: BiomechSAGE,
        evidence_head: EvidenceHead,
        weighted_adjacency: torch.Tensor,
        config: CartiDTConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.sage = sage
        self.evidence_head = evidence_head
        self.register_buffer("weighted_adjacency", weighted_adjacency, persistent=True)
        self.config = config or CartiDTConfig()
        self._node_map = {int(k): int(v) for k, v in self.config.node_to_label}
        if max(self._node_map.keys()) >= NUM_NODES:
            raise ValueError("node_to_label references node id outside graph topology")

    def forward(
        self, volume: torch.Tensor, segmentation_hint: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pyramid = self.backbone(volume)
        d, h, w = volume.shape[-3:]
        seg_logits = self.decoder(pyramid, target_shape=(d, h, w))
        seg_for_pool = segmentation_hint if segmentation_hint is not None else seg_logits.argmax(dim=1)
        node_feats = compartment_masked_gap(pyramid[0], seg_for_pool, self._node_map, NUM_NODES)
        graph_embed = self.sage(node_feats, self.weighted_adjacency)
        alpha = self.evidence_head(graph_embed)
        return seg_logits, alpha


class _TinyToyViT(nn.Module):
    def __init__(self, embed_dim: int = 32, depth: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.ModuleDict(
                            {
                                "qkv": nn.Linear(embed_dim, embed_dim * 3, bias=False),
                                "proj": nn.Linear(embed_dim, embed_dim, bias=False),
                            }
                        ),
                        "mlp": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 2),
                            nn.GELU(),
                            nn.Linear(embed_dim * 2, embed_dim),
                        ),
                        "norm1": nn.LayerNorm(embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(depth)
            ]
        )
        self.num_heads = num_heads
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            x = block["norm1"](tokens)
            qkv = block["attn"]["qkv"](x).chunk(3, dim=-1)
            attn = (qkv[0] @ qkv[1].transpose(-2, -1)) / (qkv[0].shape[-1] ** 0.5)
            attn = attn.softmax(dim=-1)
            tokens = tokens + block["attn"]["proj"](attn @ qkv[2])
            tokens = tokens + block["mlp"](block["norm2"](tokens))
        return tokens


def build_model(cfg: DictConfig) -> CartiDT:
    if bool(cfg.model.get("use_tiny_vit", False)):
        embed_dim = int(cfg.model.embed_dim)
        vit = _TinyToyViT(
            embed_dim=embed_dim, depth=int(cfg.model.tiny_depth), num_heads=int(cfg.model.tiny_num_heads)
        )
    else:
        embed_dim = int(cfg.model.embed_dim)
        vit = load_frozen_dinov2_large(weights=str(cfg.model.weights), pretrained=bool(cfg.model.pretrained))
    target_hw = tuple(int(x) for x in cfg.model.target_hw)
    if len(target_hw) != 2:
        raise ValueError("model.target_hw must be (H, W)")
    backbone = LoRAViTBackbone(
        vit=vit,
        embed_dim=embed_dim,
        patch_size=int(cfg.model.patch_size),
        depth_stride=int(cfg.model.depth_stride),
        rank=int(cfg.model.lora_rank),
        alpha=int(cfg.model.lora_alpha),
        target_hw=(target_hw[0], target_hw[1]),
        target_depth=int(cfg.model.target_depth),
        tap_layers=tuple(int(x) for x in cfg.model.tap_layers),
    )
    decoder = UPerNetDecoder(
        in_channels=[embed_dim] * len(cfg.model.tap_layers),
        decoder_dim=int(cfg.model.decoder_dim),
        num_classes=int(cfg.model.num_seg_classes),
        ppm_scales=tuple(int(x) for x in cfg.model.ppm_scales),
    )
    sage = BiomechSAGE(
        in_dim=embed_dim,
        hidden_dim=int(cfg.model.sage_hidden),
        num_layers=int(cfg.model.sage_layers),
    )
    evidence_head = EvidenceHead(
        in_dim=int(cfg.model.sage_hidden),
        num_grades=int(cfg.model.num_grades),
        hidden_dim=int(cfg.model.evidence_hidden),
    )
    weighted_adj = torch.zeros(NUM_NODES, NUM_NODES, dtype=torch.float32)
    edges_default = cfg.model.get("default_edge_weight", 1.0)
    weighted_adj.fill_(float(edges_default))
    return CartiDT(
        backbone=backbone,
        decoder=decoder,
        sage=sage,
        evidence_head=evidence_head,
        weighted_adjacency=weighted_adj,
        config=CartiDTConfig(
            num_seg_classes=int(cfg.model.num_seg_classes),
            num_grades=int(cfg.model.num_grades),
            decoder_dim=int(cfg.model.decoder_dim),
            sage_hidden=int(cfg.model.sage_hidden),
            sage_layers=int(cfg.model.sage_layers),
        ),
    )
