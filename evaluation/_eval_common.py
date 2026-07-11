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
    ckpt = torch.load(ckpt_path, map_location=device)
    name = ckpt["model_name"]
    model = models.build(name, ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, name, (name in TERRAIN_MODELS)


def make_test_loader(test_csv, terrain_config, batch_size=8, normalize="per_patch_max"):
    terrain_index = load_terrain_index(terrain_config)
    ds = TerraSRDataset(test_csv, terrain_index=terrain_index, split=None,
                        normalize=normalize, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader


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
                        bicubic_scale=None) -> pd.DataFrame:
    """Produce a per-patch metrics table. Provide either a `model` (+ is_terrain)
    or `bicubic_scale` for the interpolation floor."""
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

        sr = sr.clamp(0, 1).cpu().numpy()
        hr = hr.cpu().numpy()
        for i in range(sr.shape[0]):
            psnr, ssim = _metrics(sr[i, 0], hr[i, 0])
            records.append({
                "tile_id": meta["tile_id"][i],
                "terrain": meta["terrain_label"][i],
                "source_scene": meta["source_scene"][i],
                "psnr": psnr,
                "ssim": ssim,
            })
    return pd.DataFrame(records)
