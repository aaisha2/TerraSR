"""Differentiable SSIM, returned per-sample so the terrain-aware loss can
weight it per patch. Single-channel (PAN)."""
import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)   # (1,1,k,k)
    return window_2d


def ssim_per_sample(x, y, window_size=11, sigma=1.5, max_val=1.0):
    """Return (B,) mean SSIM for each image in the batch. Inputs (B,1,H,W)."""
    window = _gaussian_window(window_size, sigma, x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean(dim=[1, 2, 3])   # (B,)
