"""SRCNN baseline (Dong et al., ECCV 2014) — the CNN baseline the proposal
benchmarks against.

Classic 3-layer design: patch extraction (9x9) -> non-linear mapping (5x5) ->
reconstruction (5x5). SRCNN operates on a bicubic-upsampled LR image, so the
LR is upscaled to HR size inside forward(). Single-channel (PAN) in/out.
"""
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.reconstruct = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, lr):
        x = F.interpolate(lr, scale_factor=self.scale_factor,
                          mode="bicubic", align_corners=False)
        x = self.features(x)
        return self.reconstruct(x)


def build_model(cfg=None):
    cfg = cfg or {}
    return SRCNN(scale_factor=cfg.get("scale_factor", 2),
                 num_channels=cfg.get("num_channels", 1))
