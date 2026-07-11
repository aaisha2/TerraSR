"""SRGAN baseline (Ledig et al., CVPR 2017) — the GAN baseline the proposal
benchmarks against.

Provides both the SRResNet generator and the SRGAN discriminator. The
generator is a residual network with sub-pixel (pixel-shuffle) upsampling;
train_baseline.py can train it either as SRResNet (pixel/L1 loss only) or as
the full SRGAN (adversarial + content loss) via --gan. Single-channel PAN.
"""
import math

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels=64, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class SRResNetGenerator(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, num_features=64, num_blocks=16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 9, padding=4), nn.PReLU())
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.body_tail = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features))

        n_ups = int(math.log2(scale_factor))
        assert 2 ** n_ups == scale_factor, "scale_factor must be a power of 2"
        self.upsample = nn.Sequential(*[UpsampleBlock(num_features, 2) for _ in range(n_ups)])
        self.tail = nn.Conv2d(num_features, num_channels, 9, padding=4)

    def forward(self, lr):
        head = self.head(lr)
        body = self.body_tail(self.body(head)) + head
        return self.tail(self.upsample(body))


class Discriminator(nn.Module):
    """SRGAN discriminator (VGG-style). Uses AdaptiveAvgPool so it accepts any
    HR patch size, not just the 96x96 in the original paper."""
    def __init__(self, num_channels=1, num_features=64):
        super().__init__()

        def conv_block(in_c, out_c, stride):
            return [nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True)]

        layers = [nn.Conv2d(num_channels, num_features, 3, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        layers += conv_block(num_features, num_features, 2)
        for mult in (2, 4, 8):
            layers += conv_block(num_features * (mult // 2), num_features * mult, 1)
            layers += conv_block(num_features * mult, num_features * mult, 2)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(num_features * 8, 1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg=None):
    cfg = cfg or {}
    return SRResNetGenerator(
        scale_factor=cfg.get("scale_factor", 2),
        num_channels=cfg.get("num_channels", 1),
        num_features=cfg.get("num_features", 64),
        num_blocks=cfg.get("num_blocks", 16))
