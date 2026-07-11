"""Edge / gradient loss, per-sample. Penalizes differences in Sobel gradient
magnitude between SR and HR — the term the terrain-aware loss up-weights for
structurally sharp terrains (urban) and down-weights for smooth ones (water,
desert) to avoid ringing. Single-channel (PAN)."""
import torch
import torch.nn.functional as F

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = _SOBEL_X.t()


def _grad_magnitude(x):
    kx = _SOBEL_X.to(x.device, x.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(x.device, x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def edge_loss_per_sample(sr, hr):
    """Return (B,) mean |grad(SR) - grad(HR)| for each image. Inputs (B,1,H,W)."""
    diff = torch.abs(_grad_magnitude(sr) - _grad_magnitude(hr))
    return diff.mean(dim=[1, 2, 3])
