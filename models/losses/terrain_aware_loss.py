"""Terrain-aware composite loss (docs/plan §8) — the loss half of the novel
contribution. A shared L1 + SSIM base plus per-terrain edge and perceptual
terms, so each sample in a mixed-terrain batch is weighted according to its
own terrain (weights from configs/terrain_aware_loss.yaml).

The perceptual (VGG) term is only instantiated if some terrain actually
requests perceptual weight > 0 — runs that don't use it never touch the
pretrained VGG.
"""
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .edge_loss import edge_loss_per_sample
from .ssim import ssim_per_sample


class TerrainAwareLoss(nn.Module):
    def __init__(self, loss_cfg: dict, terrain_index: dict):
        super().__init__()
        self.l1_w = float(loss_cfg["base"]["l1"])
        self.ssim_w = float(loss_cfg["base"]["ssim"])

        num_terrains = len(terrain_index)
        # weight tables sized num_terrains + 1; the last row is the default
        # used for terrain_idx < 0 (unlabeled/unknown at inference)
        self.default_idx = num_terrains
        edge_w = torch.zeros(num_terrains + 1)
        perc_w = torch.zeros(num_terrains + 1)

        default = loss_cfg["default"]
        edge_w[self.default_idx] = float(default["edge"])
        perc_w[self.default_idx] = float(default["perceptual"])
        for terrain, idx in terrain_index.items():
            weights = loss_cfg["per_terrain"].get(terrain, default)
            edge_w[idx] = float(weights["edge"])
            perc_w[idx] = float(weights["perceptual"])

        self.register_buffer("edge_w", edge_w)
        self.register_buffer("perc_w", perc_w)

        self.use_perceptual = bool((perc_w > 0).any())
        self._perceptual = None   # lazily built on first use (needs VGG weights)

    def _map_idx(self, terrain_idx):
        idx = terrain_idx.clone().long()
        idx[idx < 0] = self.default_idx
        idx[idx > self.default_idx] = self.default_idx
        return idx

    def _get_perceptual(self, device):
        if self._perceptual is None:
            from .perceptual_loss import VGGPerceptualLoss
            self._perceptual = VGGPerceptualLoss().to(device)
        return self._perceptual

    def forward(self, sr, hr, terrain_idx):
        idx = self._map_idx(terrain_idx)

        l1_ps = torch.abs(sr - hr).mean(dim=[1, 2, 3])
        ssim_ps = 1.0 - ssim_per_sample(sr.clamp(0, 1), hr)
        edge_ps = edge_loss_per_sample(sr, hr)

        w_edge = self.edge_w[idx]
        w_perc = self.perc_w[idx]

        total = self.l1_w * l1_ps + self.ssim_w * ssim_ps + w_edge * edge_ps
        components = {"l1": l1_ps.mean().item(), "ssim": ssim_ps.mean().item(),
                      "edge": edge_ps.mean().item()}

        if self.use_perceptual:
            perc_ps = self._get_perceptual(sr.device).forward_per_sample(sr.clamp(0, 1), hr)
            total = total + w_perc * perc_ps
            components["perceptual"] = perc_ps.mean().item()

        return total.mean(), components


def build_loss(terrain_index: dict, cfg_path="configs/terrain_aware_loss.yaml"):
    loss_cfg = yaml.safe_load(Path(cfg_path).read_text())
    return TerrainAwareLoss(loss_cfg, terrain_index)
