"""VGG perceptual/texture loss, per-sample. The term the terrain-aware loss
up-weights for texture-rich terrains (forest). Lazily instantiated: the VGG
is only built if some terrain actually asks for perceptual weight > 0, so
runs that don't use it never need the pretrained weights.

PAN is single-channel; VGG expects 3, so the input is replicated to 3
channels and normalized with ImageNet statistics.
"""
import torch
import torch.nn as nn

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_index=16):
        """layer_index: index into vgg16.features to cut at (16 ≈ relu3_3)."""
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.slice = nn.Sequential(*list(vgg.features[:layer_index])).eval()
        for p in self.slice.parameters():
            p.requires_grad_(False)

    def _prep(self, x):
        x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        mean = _IMAGENET_MEAN.to(x.device, x.dtype)
        std = _IMAGENET_STD.to(x.device, x.dtype)
        return (x - mean) / std

    def forward_per_sample(self, sr, hr):
        f_sr = self.slice(self._prep(sr))
        f_hr = self.slice(self._prep(hr))
        return torch.abs(f_sr - f_hr).mean(dim=[1, 2, 3])   # (B,)
