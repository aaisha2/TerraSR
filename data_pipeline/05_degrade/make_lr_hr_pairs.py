"""Stage 5 CLI: turn a folder of HR patches into LR/HR pairs using the
confirmed single-order degradation pipeline (configs/degradation.yaml).

Usage:
    python make_lr_hr_pairs.py --hr-dir tests/sample_images --out-dir out/pairs

Accepts .png/.jpg/.tif HR patches for now (rasterio GeoTIFF I/O gets wired in
once stage 2/3 are producing real standardized patches).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from degradation_pipeline import degrade  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_gray_float(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def save_gray_float(arr: np.ndarray, path: Path) -> None:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/degradation.yaml"), type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    rng = np.random.default_rng(cfg.get("seed"))

    hr_out = args.out_dir / "hr"
    lr_out = args.out_dir / "lr"
    hr_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)

    manifest = []
    hr_paths = sorted(p for p in args.hr_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not hr_paths:
        raise SystemExit(f"no HR images found in {args.hr_dir}")

    for hr_path in hr_paths:
        hr = load_gray_float(hr_path)
        result = degrade(hr, cfg, rng)

        stem = hr_path.stem
        hr_dst = hr_out / f"{stem}.png"
        lr_dst = lr_out / f"{stem}.png"
        save_gray_float(hr, hr_dst)
        save_gray_float(result["lr_image"], lr_dst)

        manifest.append({
            "id": stem,
            "hr_path": str(hr_dst),
            "lr_path": str(lr_dst),
            "degradation_params": result["params"],
        })
        print(f"  {stem}: HR {hr.shape} -> LR {result['lr_image'].shape} "
              f"({result['params']['downsample_method']})")

    manifest_path = args.out_dir / "degradation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} pairs -> {args.out_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
