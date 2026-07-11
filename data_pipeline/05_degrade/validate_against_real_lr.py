"""Stage 5 required sub-step (added by supervisor 2026-07-11): compare
synthetic LR output against real satellite LR imagery before locking
degradation.yaml parameters for the full dataset build.

Computes three cheap, comparable statistics per image set:
  - noise_variance:        variance of a high-pass residual (proxy for
                            sensor noise level)
  - edge_density:          mean Sobel gradient magnitude (proxy for how
                            much high-frequency detail survives blur+downsample)
  - radial_power_spectrum: azimuthally-averaged FFT power spectrum binned
                            into low/mid/high frequency thirds (proxy for
                            whether the synthetic MTF roll-off matches real
                            sensor MTF roll-off)

This does not replace visual inspection — it's a quick numeric flag for
"synthetic LR looks nothing like real LR, go re-tune degradation.yaml"
before spending compute on a full dataset build.

Usage:
    python validate_against_real_lr.py --synthetic-dir out/pairs/lr --real-dir path/to/real_lr_chips
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_gray_float(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def noise_variance(img: np.ndarray) -> float:
    smoothed = ndimage.uniform_filter(img, size=3)
    residual = img - smoothed
    return float(np.var(residual))


def edge_density(img: np.ndarray) -> float:
    gx = ndimage.sobel(img, axis=0)
    gy = ndimage.sobel(img, axis=1)
    return float(np.mean(np.hypot(gx, gy)))


def radial_power_thirds(img: np.ndarray) -> tuple:
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - cx, y - cy)
    r_max = r.max()
    low = power[r <= r_max / 3].mean()
    mid = power[(r > r_max / 3) & (r <= 2 * r_max / 3)].mean()
    high = power[r > 2 * r_max / 3].mean()
    total = low + mid + high + 1e-12
    return low / total, mid / total, high / total


def summarize(dir_path: Path) -> dict:
    paths = sorted(p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"no images found in {dir_path}")
    nv, ed, low, mid, high = [], [], [], [], []
    for p in paths:
        img = load_gray_float(p)
        nv.append(noise_variance(img))
        ed.append(edge_density(img))
        l, m, h = radial_power_thirds(img)
        low.append(l); mid.append(m); high.append(h)
    return {
        "n_images": len(paths),
        "noise_variance_mean": float(np.mean(nv)),
        "edge_density_mean": float(np.mean(ed)),
        "power_spectrum_low_mid_high": (float(np.mean(low)), float(np.mean(mid)), float(np.mean(high))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-dir", required=True, type=Path)
    ap.add_argument("--real-dir", required=True, type=Path,
                     help="Real satellite LR reference chips, comparable sensor/GSD")
    args = ap.parse_args()

    synth = summarize(args.synthetic_dir)
    real = summarize(args.real_dir)

    print(f"{'metric':<32}{'synthetic':>14}{'real':>14}{'ratio':>10}")
    for key in ("noise_variance_mean", "edge_density_mean"):
        s, r = synth[key], real[key]
        ratio = s / r if r else float("nan")
        print(f"{key:<32}{s:>14.5f}{r:>14.5f}{ratio:>10.2f}")

    print(f"\n{'power spectrum (low/mid/high)':<32}{'synthetic':>14}{'real':>14}")
    for band, s_val, r_val in zip(("low", "mid", "high"),
                                   synth["power_spectrum_low_mid_high"],
                                   real["power_spectrum_low_mid_high"]):
        print(f"  {band:<30}{s_val:>14.3f}{r_val:>14.3f}")

    print("\nRule of thumb: ratios far from 1.0 (roughly outside 0.5-2.0), or a "
          "high-frequency power share that's much lower in synthetic than real, "
          "both mean degradation.yaml needs re-tuning (usually: less blur / less "
          "aggressive downsample interpolation, or noise levels too low).")


if __name__ == "__main__":
    main()
