"""SwinIR baseline (Liang et al., ICCV Workshops 2021) — the main model of
the project (the proposal's transformer-based approach).

Compact but faithful implementation of the classical-SR configuration:
shallow feature extraction (conv) -> deep feature extraction (a stack of
Residual Swin Transformer Blocks, RSTB) -> pixel-shuffle upsampling. Written
in-house rather than imported so stage 8 can inject FiLM terrain conditioning
into the RSTB internals (models/terrasr_swinir.py) without fighting a vendored
black box.

Single-channel (PAN) in/out. Input LR is expected at LR resolution; the model
upsamples internally by scale_factor.
"""
import math

import torch
import torch.nn as nn


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # relative position bias table + index
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _attn_mask(self, h, w, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, h, w, 1), device=device)
        slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for hs in slices:
            for ws in slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = window_partition(x, self.window_size).view(-1, self.window_size ** 2, c)

        attn_windows = self.attn(x_windows, mask=self._attn_mask(h, w, x.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(b, h * w, c)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: a stack of Swin blocks (alternating
    regular/shifted windows) + a conv, with a residual connection. This is the
    unit stage 8 will condition on terrain."""
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        for blk in self.blocks:
            x = blk(x, x_size)
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x + shortcut


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        layers = []
        for _ in range(int(math.log2(scale))):
            layers += [nn.Conv2d(num_feat, 4 * num_feat, 3, padding=1), nn.PixelShuffle(2)]
        super().__init__(*layers)


class SwinIR(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, embed_dim=60,
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=8,
                 mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.conv_first = nn.Conv2d(num_channels, embed_dim, 3, padding=1)
        self.norm_pre = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            RSTB(embed_dim, depths[i], num_heads[i], window_size, mlp_ratio)
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

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)      # (b, hw, c)
        x = self.norm_pre(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, x.shape[2], x_size[0], x_size[1])
        return x

    def forward(self, lr):
        x, h, w = self._pad(lr)
        feat = self.conv_first(x)
        feat = self.conv_after_body(self.forward_features(feat)) + feat
        feat = self.conv_before_upsample(feat)
        out = self.conv_last(self.upsample(feat))
        # crop padding back off (scaled)
        scale = out.shape[2] // x.shape[2]
        return out[:, :, :h * scale, :w * scale]


def build_model(cfg=None):
    cfg = cfg or {}
    return SwinIR(
        scale_factor=cfg.get("scale_factor", 2),
        num_channels=cfg.get("num_channels", 1),
        embed_dim=cfg.get("embed_dim", 60),
        depths=tuple(cfg.get("depths", (6, 6, 6, 6))),
        num_heads=tuple(cfg.get("num_heads", (6, 6, 6, 6))),
        window_size=cfg.get("window_size", 8),
        mlp_ratio=cfg.get("mlp_ratio", 2.0))
