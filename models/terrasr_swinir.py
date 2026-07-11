"""TerraSR-SwinIR (docs/plan §8) — the architecture half of the novel
contribution: one SwinIR conditioned on terrain via a learned terrain
embedding and FiLM (Feature-wise Linear Modulation).

A single embedding table holds one vector per terrain. Each deep-feature
block (RSTB) has its own small projection from that embedding to per-channel
(gamma, beta), which modulate the block's conv features:
    f' = (1 + gamma) * f + beta
So the network sees terrain as a first-class conditioning signal rather than
inferring it implicitly from pixels — but it stays ONE model, not seven.

Reuses the baseline SwinIR building blocks (window attention, Swin blocks)
unchanged; only the RSTB gains FiLM. Single-channel (PAN), 2x by default.
"""
import math

import torch
import torch.nn as nn

from .swinir_baseline import RSTB, SwinTransformerBlock, Upsample


class FiLMGenerator(nn.Module):
    """Maps a terrain embedding to per-channel (gamma, beta) for one RSTB."""
    def __init__(self, embed_dim_terrain, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim_terrain, feature_dim), nn.SiLU(),
            nn.Linear(feature_dim, 2 * feature_dim))
        # start as identity modulation (gamma=0 -> 1+gamma=1, beta=0) so an
        # untrained condition doesn't disrupt the pretrained-style init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.feature_dim = feature_dim

    def forward(self, terrain_emb):
        gamma_beta = self.net(terrain_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta


class ConditionedRSTB(nn.Module):
    """RSTB + FiLM: runs the Swin blocks + conv exactly as the baseline RSTB,
    then modulates the result by the terrain-derived (gamma, beta) before the
    residual add."""
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio,
                 embed_dim_terrain):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.film = FiLMGenerator(embed_dim_terrain, dim)

    def forward(self, x, x_size, terrain_emb):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        for blk in self.blocks:
            x = blk(x, x_size)
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv(x)

        gamma, beta = self.film(terrain_emb)                 # (b, c) each
        x = (1 + gamma.view(b, c, 1, 1)) * x + beta.view(b, c, 1, 1)

        x = x.flatten(2).transpose(1, 2)
        return x + shortcut


class TerraSRSwinIR(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, embed_dim=60,
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=8,
                 mlp_ratio=2.0, num_terrains=7, embed_dim_terrain=32):
        super().__init__()
        self.window_size = window_size
        # +1 row for unknown/unlabeled terrain (index == num_terrains)
        self.unknown_idx = num_terrains
        self.terrain_embedding = nn.Embedding(num_terrains + 1, embed_dim_terrain)

        self.conv_first = nn.Conv2d(num_channels, embed_dim, 3, padding=1)
        self.norm_pre = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            ConditionedRSTB(embed_dim, depths[i], num_heads[i], window_size,
                            mlp_ratio, embed_dim_terrain)
            for i in range(len(depths))])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(scale_factor, 64)
        self.conv_last = nn.Conv2d(64, num_channels, 3, padding=1)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.window_size - h % self.window_size) % self.window_size
        pw = (self.window_size - w % self.window_size) % self.window_size
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def _terrain_emb(self, terrain_idx, batch_size, device):
        if terrain_idx is None:
            terrain_idx = torch.full((batch_size,), self.unknown_idx, device=device)
        else:
            terrain_idx = terrain_idx.clone().long().to(device)
            terrain_idx[terrain_idx < 0] = self.unknown_idx
        return self.terrain_embedding(terrain_idx)

    def forward_features(self, x, terrain_emb):
        x_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_pre(x)
        for layer in self.layers:
            x = layer(x, x_size, terrain_emb)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, x.shape[2], x_size[0], x_size[1])
        return x

    def forward(self, lr, terrain_idx=None):
        x, h, w = self._pad(lr)
        terrain_emb = self._terrain_emb(terrain_idx, x.shape[0], x.device)

        feat = self.conv_first(x)
        feat = self.conv_after_body(self.forward_features(feat, terrain_emb)) + feat
        feat = self.conv_before_upsample(feat)
        out = self.conv_last(self.upsample(feat))
        scale = out.shape[2] // x.shape[2]
        return out[:, :, :h * scale, :w * scale]


def build_model(cfg=None):
    cfg = cfg or {}
    return TerraSRSwinIR(
        scale_factor=cfg.get("scale_factor", 2),
        num_channels=cfg.get("num_channels", 1),
        embed_dim=cfg.get("embed_dim", 60),
        depths=tuple(cfg.get("depths", (6, 6, 6, 6))),
        num_heads=tuple(cfg.get("num_heads", (6, 6, 6, 6))),
        window_size=cfg.get("window_size", 8),
        mlp_ratio=cfg.get("mlp_ratio", 2.0),
        num_terrains=cfg.get("num_terrains", 7),
        embed_dim_terrain=cfg.get("embed_dim_terrain", 32))
