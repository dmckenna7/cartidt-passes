"""Random training-time volume augmentations (Ref: Sec. V.C ¶1).

Implements the six operators listed in Implementation Details:
rotation ±10°, scale 0.9-1.1, translation ±15 voxels, random elastic,
additive Gaussian σ=0.02 on intensity-normalised values, and multiplicative
gain in [0.95, 1.05]. The elastic field follows MONAI's `Rand3DElastic`
parameterisation for parity with prior community code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class VolumeAugmentConfig:
    rotation_deg: float = 10.0
    scale_lo: float = 0.9
    scale_hi: float = 1.1
    translate_voxels: int = 15
    elastic_sigma: float = 8.0
    elastic_alpha: float = 12.0
    noise_std: float = 0.02
    gain_lo: float = 0.95
    gain_hi: float = 1.05


class VolumeAugment:
    def __init__(self, cfg: VolumeAugmentConfig | None = None) -> None:
        self.cfg = cfg or VolumeAugmentConfig()

    def __call__(self, volume: torch.Tensor, segmentation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if volume.ndim != 4 or segmentation.ndim != 3:
            raise ValueError("volume must be (C, D, H, W); segmentation must be (D, H, W)")
        affine = self._sample_affine(volume.device)
        warped_vol = self._affine_warp(volume.unsqueeze(0), affine, mode="bilinear").squeeze(0)
        warped_seg = (
            self._affine_warp(segmentation.unsqueeze(0).unsqueeze(0).float(), affine, mode="nearest")
            .squeeze(0)
            .squeeze(0)
            .long()
        )
        warped_vol, warped_seg = self._elastic(warped_vol, warped_seg)
        warped_vol = self._intensity(warped_vol)
        return warped_vol, warped_seg

    def _sample_affine(self, device: torch.device) -> torch.Tensor:
        cfg = self.cfg
        angle = torch.empty(1, device=device).uniform_(-cfg.rotation_deg, cfg.rotation_deg) * torch.pi / 180.0
        scale = torch.empty(1, device=device).uniform_(cfg.scale_lo, cfg.scale_hi)
        tx = torch.empty(1, device=device).uniform_(-cfg.translate_voxels, cfg.translate_voxels)
        ty = torch.empty(1, device=device).uniform_(-cfg.translate_voxels, cfg.translate_voxels)
        tz = torch.empty(1, device=device).uniform_(-cfg.translate_voxels, cfg.translate_voxels)
        c, s = torch.cos(angle), torch.sin(angle)
        rot = torch.eye(3, device=device)
        rot[1, 1] = c.item()
        rot[1, 2] = -s.item()
        rot[2, 1] = s.item()
        rot[2, 2] = c.item()
        rot = rot * scale
        translate = torch.stack([tz, ty, tx], dim=0).reshape(3, 1) / 100.0
        affine = torch.cat([rot, translate], dim=1)
        return affine.unsqueeze(0)

    @staticmethod
    def _affine_warp(volume: torch.Tensor, affine: torch.Tensor, mode: str) -> torch.Tensor:
        grid = F.affine_grid(affine, list(volume.shape), align_corners=False)
        return F.grid_sample(volume, grid, mode=mode, padding_mode="zeros", align_corners=False)

    def _elastic(self, volume: torch.Tensor, segmentation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        d, h, w = volume.shape[-3:]
        device = volume.device
        noise = torch.randn(1, 3, d // 8, h // 8, w // 8, device=device) * cfg.elastic_sigma
        upsampled = F.interpolate(noise, size=(d, h, w), mode="trilinear", align_corners=False)
        upsampled = upsampled.squeeze(0) * (cfg.elastic_alpha / max(d, h, w))
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, d, device=device),
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij",
        )
        grid = torch.stack([xx + upsampled[2], yy + upsampled[1], zz + upsampled[0]], dim=-1).unsqueeze(0)
        warped_vol = F.grid_sample(
            volume.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
        ).squeeze(0)
        warped_seg = (
            F.grid_sample(
                segmentation.unsqueeze(0).unsqueeze(0).float(),
                grid,
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
            .long()
        )
        return warped_vol, warped_seg

    def _intensity(self, volume: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        gain = torch.empty(1, device=volume.device).uniform_(cfg.gain_lo, cfg.gain_hi)
        noisy = volume * gain + torch.randn_like(volume) * cfg.noise_std
        return noisy
