"""Stage 1 (pseudo-PAN experiment): download USDA NAIP imagery via Microsoft
Planetary Computer.

NAIP is RGB+NIR, NOT panchromatic — it belongs to the pseudo-PAN arm
(Experiment 2) only, never to the true-PAN training set. Stage 2's
`rgb_to_pseudo_pan` mode converts it and tags it `pseudo_pan=true`.

Why Planetary Computer and not AWS: the `naip-*` S3 buckets reject anonymous
access (verified 2026-07-28 — AccessDenied on naip-visualization /
naip-analytic / naip-source), so the AWS route needs credentials and
requester-pays. Planetary Computer's STAC API searches anonymously and its
free `/api/sas/v1/sign` endpoint returns a short-lived read token with no
account — fully scriptable, which is what your requirements demand.

Verified live: STAC search returns `gsd: 0.3` items whose `image` asset is a
~2 GB COG GeoTIFF (newer NAIP is 30 cm, not 60 cm). Only the `image` asset is
fetched; `thumbnail`, `rendered_preview` and `tilejson` are skipped.

Because scenes are ~2 GB, `--windows N` (default) extracts N random crops per
scene by range-reading the COG instead of downloading it whole. Use
`--full` to pull entire scenes.

Usage:
    python download_naip_pc.py --config configs/datasets.yaml --list-only
    python download_naip_pc.py --config configs/datasets.yaml --windows 30
"""
import argparse
import sys
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import human_size, stream_download_url  # noqa: E402

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


def search(aoi, collection, limit, max_gsd, timeout=60):
    body = {"collections": [collection], "bbox": aoi["bbox"], "limit": limit}
    if aoi.get("datetime"):
        body["datetime"] = aoi["datetime"]
    r = requests.post(STAC_SEARCH, json=body, timeout=timeout)
    r.raise_for_status()
    feats = r.json().get("features", [])
    kept = []
    for f in feats:
        gsd = f["properties"].get("gsd")
        if max_gsd is not None and gsd is not None and gsd > max_gsd:
            continue
        asset = f.get("assets", {}).get("image")     # image asset only
        if not asset or "tiff" not in asset.get("type", ""):
            continue
        kept.append((f["id"], gsd, asset["href"]))
    return kept


def sign(href, timeout=60):
    r = requests.get(SIGN_URL, params={"href": href}, timeout=timeout)
    r.raise_for_status()
    return r.json()["href"]


def extract_windows(signed_url, item_id, out_dir, n_windows, window_size, seed):
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with rasterio.open(f"/vsicurl/{signed_url}") as src:
        if src.width < window_size or src.height < window_size:
            print(f"    scene smaller than window; skipped")
            return 0
        n_bands = min(3, src.count)                  # RGB only; drop NIR
        rng = np.random.default_rng(seed)
        attempts = 0
        while written < n_windows and attempts < n_windows * 12:
            attempts += 1
            col = int(rng.integers(0, src.width - window_size))
            row = int(rng.integers(0, src.height - window_size))
            dest = out_dir / f"{item_id}_w{row:06d}_{col:06d}.tif"
            if dest.exists():
                written += 1
                continue
            win = Window(col, row, window_size, window_size)
            data = src.read(list(range(1, n_bands + 1)), window=win)
            if float((data == 0).all(axis=0).mean()) > 0.05 or float(data.std()) < 4.0:
                continue
            profile = src.profile.copy()
            profile.update(height=window_size, width=window_size, count=n_bands,
                           transform=src.window_transform(win),
                           compress="deflate", tiled=True,
                           blockxsize=256, blockysize=256)
            profile.pop("nodata", None)
            tmp = dest.with_suffix(".tif.part")
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(data)
                dst.update_tags(source_scene=item_id, naip_window=f"{row},{col}",
                                band_source="naip_rgb", pseudo_pan_source="true")
            tmp.rename(dest)
            written += 1
            print(f"\r    {item_id[:40]}: {written}/{n_windows} crops", end="", flush=True)
    if written:
        print()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path("configs/datasets.yaml"), type=Path)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--windows", type=int, default=None,
                     help="crops per scene (default: value from config)")
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="download whole scenes instead of crops")
    ap.add_argument("--max-items", type=int, default=None, help="cap scenes per AOI")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["naip_pc"]
    out_dir = Path(cfg["out_dir"])
    n_windows = args.windows or cfg.get("windows_per_scene", 30)
    window_size = args.window_size or cfg.get("window_size", 1024)
    max_gsd = cfg.get("max_gsd", 0.6)

    total_items = total_crops = 0
    for aoi in cfg["aois"]:
        limit = args.max_items or aoi.get("limit", 4)
        try:
            items = search(aoi, cfg["collection"], limit, max_gsd)
        except Exception as e:
            print(f"  [skip] AOI {aoi.get('name')}: search failed: {e}")
            continue
        print(f"\n=== {aoi.get('name','aoi')} ({len(items)} item(s), gsd<={max_gsd}m) ===")
        for item_id, gsd, href in items[:limit]:
            total_items += 1
            if args.list_only:
                print(f"  {item_id}  gsd={gsd}m")
                continue
            try:
                signed = sign(href)
            except Exception as e:
                print(f"  [skip] {item_id}: signing failed: {e}")
                continue
            if args.full:
                dest = out_dir / aoi.get("name", "aoi") / f"{item_id}.tif"
                stream_download_url(signed, dest)
            else:
                total_crops += extract_windows(signed, item_id,
                                                out_dir / aoi.get("name", "aoi"),
                                                n_windows, window_size, args.seed)

    if args.list_only:
        print(f"\nwould fetch {total_items} NAIP scene(s)")
    elif args.full:
        print(f"\nfetched {total_items} NAIP scene(s) -> {out_dir}")
    else:
        print(f"\nextracted {total_crops} crops from {total_items} scene(s) -> {out_dir}")
    print("NOTE: NAIP is RGB -> pseudo-PAN (Experiment 2 only), never true PAN.")


if __name__ == "__main__":
    main()
