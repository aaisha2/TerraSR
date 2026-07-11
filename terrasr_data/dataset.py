"""Manifest-driven PyTorch Dataset for TerraSR (stages 7-9).

Reads the unified manifest produced by stage 6 (build_manifest.py /
split_train_val_test.py) -- one row per patch with hr_path, lr_path,
terrain_label, split, etc. -- and yields (lr, hr, terrain_idx, meta) tuples.

The same Dataset feeds every model: the SRCNN/SRGAN/SwinIR baselines ignore
terrain_idx, and stage 8's terrain-conditioned model uses it. So the only
thing that changes between experiments is the model, not the data path
(docs/plan §7).

Normalization: HR and LR are read at their native dtype and divided by the
SAME scale (the HR patch's own max by default) so the pair stays radiometri-
cally consistent -- stage 5 already wrote them on one scale, this just maps
that to [0,1] for the network. The scale is returned in `meta` so predictions
can be mapped back to DN if needed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

try:
    import rasterio
    _HAVE_RASTERIO = True
except ImportError:
    _HAVE_RASTERIO = False

from PIL import Image


def load_terrain_index(terrain_cfg_path="configs/terrain_classes.yaml") -> dict:
    cfg = yaml.safe_load(Path(terrain_cfg_path).read_text())
    return cfg["terrain_index"]


def _dtype_max(dtype) -> float:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return 1.0


def _read_image(path: str):
    """Return a single-band float array (H, W) in native DN, plus its dtype's
    theoretical max. Handles GeoTIFF (rasterio) and PNG/JPG (PIL)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        if not _HAVE_RASTERIO:
            raise RuntimeError("rasterio required to read GeoTIFF patches")
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            dmax = _dtype_max(ds.dtypes[0])
    else:
        img = Image.open(path)
        arr = np.asarray(img.convert("I") if img.mode.startswith("I") else img.convert("L"),
                          dtype=np.float32)
        dmax = _dtype_max(arr.dtype if arr.dtype != np.float32 else "uint8")
    return arr, dmax


class TerraSRDataset(Dataset):
    def __init__(self, manifest_csv, terrain_index: dict, split=None,
                 normalize="per_patch_max", augment=False):
        """
        manifest_csv:  a stage 6 manifest (dataset_manifest_split.csv) or a
                       per-split CSV (train.csv / val.csv / test.csv).
        split:         if given, filter rows to this split (train/val/test).
        normalize:     'per_patch_max' (HR max per sample) or 'dtype_max'.
        augment:       random 8-fold D4 flips/rotations (train only).
        """
        df = pd.read_csv(manifest_csv)
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.terrain_index = terrain_index
        self.normalize = normalize
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _norm_scale(self, hr_arr, hr_dmax) -> float:
        if self.normalize == "dtype_max":
            return hr_dmax
        return float(hr_arr.max()) or 1.0

    def _augment(self, lr, hr):
        # D4: random horizontal/vertical flip + 0/90/180/270 rotation.
        # Applied identically to LR and HR so they stay aligned.
        if np.random.rand() < 0.5:
            lr, hr = lr[:, ::-1], hr[:, ::-1]
        if np.random.rand() < 0.5:
            lr, hr = lr[::-1, :], hr[::-1, :]
        k = np.random.randint(4)
        if k:
            lr, hr = np.rot90(lr, k), np.rot90(hr, k)
        return np.ascontiguousarray(lr), np.ascontiguousarray(hr)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hr_arr, hr_dmax = _read_image(row["hr_path"])
        lr_arr, _ = _read_image(row["lr_path"])

        scale = self._norm_scale(hr_arr, hr_dmax)
        hr = np.clip(hr_arr / scale, 0.0, 1.0)
        lr = np.clip(lr_arr / scale, 0.0, 1.0)

        if self.augment:
            lr, hr = self._augment(lr, hr)

        lr_t = torch.from_numpy(lr).unsqueeze(0).float()   # (1, h, w)
        hr_t = torch.from_numpy(hr).unsqueeze(0).float()   # (1, H, W)

        terrain = row.get("terrain_label")
        terrain_idx = self.terrain_index.get(terrain, -1)

        meta = {
            "tile_id": row.get("tile_id", ""),
            "terrain_label": terrain if isinstance(terrain, str) else "",
            "source_scene": row.get("source_scene", ""),
            "norm_scale": float(scale),
        }
        return lr_t, hr_t, terrain_idx, meta
