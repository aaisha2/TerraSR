"""Shared evaluation helpers (stage 9).

Loads any trained checkpoint (baseline or terrain-conditioned), runs it over a
test split, and returns per-patch PSNR/SSIM records tagged with terrain so the
per-terrain breakdown and overall metrics are just different aggregations of
the same table. Also provides a `bicubic` pseudo-model (the naive
interpolation floor the proposal benchmarks against).

Metrics use skimage.metrics (the community-standard implementations) on
[0,1] images, so the reported numbers are comparable to the SR literature.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models  # noqa: E402
from terrasr_data import TerraSRDataset, load_terrain_index  # noqa: E402

TERRAIN_MODELS = {"terrasr", "terrasr_swinir"}


def load_model_from_checkpoint(ckpt_path, device):
    # weights_only=False: our own checkpoints carry a config dict + RNG state
    # that the PyTorch 2.6+ safe loader rejects. Trusted (self-produced).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = ckpt["model_name"]
    model = models.build(name, ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, name, (name in TERRAIN_MODELS)


def make_test_loader(test_csv, terrain_config, batch_size=8, normalize="per_patch_max",
                      pan_filter="all"):
    terrain_index = load_terrain_index(terrain_config)
    ds = TerraSRDataset(test_csv, terrain_index=terrain_index, split=None,
                        normalize=normalize, augment=False, pan_filter=pan_filter)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


# --------------------------------------------------------------- LPIPS

_LPIPS = {"net": None}


def get_lpips(device):
    """Lazily build the LPIPS (AlexNet) metric. Returns None with a clear
    message if the package isn't installed, so evaluation still runs."""
    if _LPIPS["net"] == "unavailable":
        return None
    if _LPIPS["net"] is None:
        try:
            import lpips as lpips_pkg
            _LPIPS["net"] = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:
            print(f"  [LPIPS unavailable: {e}; install with `pip install lpips`]")
            _LPIPS["net"] = "unavailable"
            return None
    return _LPIPS["net"]


def lpips_per_batch(net, sr, hr, device):
    """LPIPS for a batch of single-channel images in [0,1].
    LPIPS expects 3-channel input in [-1,1], so PAN is replicated to RGB."""
    import torch as _t
    with _t.no_grad():
        a = sr.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        b = hr.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        d = net(a.to(device), b.to(device))
    return d.flatten().detach().cpu().numpy().tolist()


# --------------------------------------------------------------- speed

@torch.no_grad()
def measure_inference_speed(model, is_terrain, device, lr_size=128, batch_size=1,
                             warmup=3, runs=20):
    """Median per-image inference time (ms) and throughput (img/s).

    Uses median rather than mean so a single scheduling hiccup doesn't skew it,
    warms up first (kernel autotune / lazy init), and synchronises CUDA so the
    timing measures real compute rather than async queue submission.
    """
    import statistics
    import time

    model.eval()
    x = torch.rand(batch_size, 1, lr_size, lr_size, device=device)
    t_idx = torch.zeros(batch_size, dtype=torch.long, device=device) if is_terrain else None

    def once():
        if is_terrain:
            model(x, t_idx)
        else:
            model(x)

    for _ in range(warmup):
        once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    med = statistics.median(times)
    return {"ms_per_image": med / batch_size,
            "images_per_sec": (batch_size * 1000.0 / med) if med > 0 else float("nan"),
            "lr_size": lr_size, "batch_size": batch_size, "device": str(device)}


def _metrics(sr_np, hr_np):
    sr_np = np.clip(sr_np, 0, 1)
    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
    # window must fit the patch; 7 is safe for >=256px, guard tiny patches
    win = min(7, sr_np.shape[0], sr_np.shape[1])
    if win % 2 == 0:
        win -= 1
    ssim = structural_similarity(hr_np, sr_np, data_range=1.0, win_size=max(win, 3))
    return float(psnr), float(ssim)


@torch.no_grad()
def run_sr_over_loader(loader, device, model=None, is_terrain=False,
                        bicubic_scale=None, with_lpips=False) -> pd.DataFrame:
    """Produce a per-patch metrics table (PSNR, SSIM, optionally LPIPS).
    Provide either a `model` (+ is_terrain) or `bicubic_scale` for the
    interpolation floor."""
    net = get_lpips(device) if with_lpips else None
    records = []
    for lr, hr, terrain_idx, meta in loader:
        lr, hr = lr.to(device), hr.to(device)
        if bicubic_scale is not None:
            sr = F.interpolate(lr, scale_factor=bicubic_scale, mode="bicubic",
                               align_corners=False)
        elif is_terrain:
            sr = model(lr, terrain_idx.to(device))
        else:
            sr = model(lr)

        lp = lpips_per_batch(net, sr, hr, device) if net is not None else None

        sr_np = sr.clamp(0, 1).cpu().numpy()
        hr_np = hr.cpu().numpy()
        for i in range(sr_np.shape[0]):
            psnr, ssim = _metrics(sr_np[i, 0], hr_np[i, 0])
            rec = {
                "tile_id": meta["tile_id"][i],
                "terrain": meta["terrain_label"][i],
                "source_scene": meta["source_scene"][i],
                "psnr": psnr,
                "ssim": ssim,
            }
            if lp is not None:
                rec["lpips"] = lp[i]
            records.append(rec)
    return pd.DataFrame(records)
