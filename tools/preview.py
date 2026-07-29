"""Make pipeline imagery viewable — convert 16-bit GeoTIFFs to PNGs you can
open anywhere, and build side-by-side HTML comparisons for tuning parameters.

Why this exists: stages 2-5 write 16-bit single-band GeoTIFFs. Windows Photos
can't open them, and even viewers that can usually render them near-black,
because PAN data occupies only a slice of the 16-bit range. This applies a
percentile contrast stretch (like a GIS viewer does) so you actually see the
image, and reports the real DN range so you know what you're looking at.

Two modes:

  1) Browse any stage's output as PNGs / a contact sheet
     python tools/preview.py --input data/standardized/maxar --out-dir previews/std
     python tools/preview.py --input data/patches --html previews/patches.html --limit 40

  2) HR vs LR comparison for tuning configs/degradation.yaml — shows each pair
     at matched display size, plus a zoomed centre crop where blur / noise /
     aliasing are actually visible, annotated with the degradation parameters
     that produced it.
     python tools/preview.py --pairs data/pairs --html previews/degradation.html --limit 12

Stretch is configurable so you can compare like-for-like:
  --stretch percentile (default, --percentiles 2 98) | minmax | none
Use `--stretch none` to see the raw linear mapping (what a naive viewer shows).
"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    _HAVE_RASTERIO = True
except ImportError:
    _HAVE_RASTERIO = False

RASTER_EXTS = {".tif", ".tiff"}
PIL_EXTS = {".png", ".jpg", ".jpeg"}
IMG_EXTS = RASTER_EXTS | PIL_EXTS


# ---------------------------------------------------------------- reading

def read_band(path: Path):
    """Return (float array, info dict) for a single-band image of any supported
    type. Never rescales — the raw DN values come back untouched."""
    suffix = path.suffix.lower()
    if suffix in RASTER_EXTS:
        if not _HAVE_RASTERIO:
            raise RuntimeError("rasterio is required to read GeoTIFFs")
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float64)
            info = {"dtype": str(ds.dtypes[0]), "crs": str(ds.crs),
                    "size": f"{ds.width}x{ds.height}", "tags": ds.tags()}
    else:
        img = Image.open(path)
        arr = np.asarray(img.convert("I") if img.mode.startswith("I")
                         else img.convert("L"), dtype=np.float64)
        info = {"dtype": str(np.asarray(img).dtype), "crs": "-",
                "size": f"{arr.shape[1]}x{arr.shape[0]}", "tags": {}}
    info["dn_min"] = float(arr.min())
    info["dn_max"] = float(arr.max())
    info["dn_mean"] = float(arr.mean())
    return arr, info


# ---------------------------------------------------------------- stretching

def stretch(arr: np.ndarray, mode="percentile", pcts=(2.0, 98.0)) -> np.ndarray:
    """Map raw DN to [0,1] for display.

    percentile: clip to the given percentiles then rescale — what GIS viewers
                do, and the only way faint PAN detail becomes visible.
    minmax:     rescale the full observed range.
    none:       raw linear over the dtype range (shows why the file looks black
                in a naive viewer).
    """
    a = arr.astype(np.float64)
    if mode == "none":
        hi = 65535.0 if a.max() > 255 else 255.0
        return np.clip(a / hi, 0, 1)
    if mode == "minmax":
        lo, hi = a.min(), a.max()
    else:
        valid = a[a > 0] if (a > 0).any() else a      # ignore nodata=0
        lo, hi = np.percentile(valid, pcts[0]), np.percentile(valid, pcts[1])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def to_pil(arr01: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr01 * 255, 0, 255).astype(np.uint8))


def data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def centre_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    size = min(size, h, w)
    top, left = (h - size) // 2, (w - size) // 2
    return arr[top:top + size, left:left + size]


def upscale(img: Image.Image, target: int) -> Image.Image:
    """Nearest-neighbour so individual pixels stay visible when zoomed."""
    return img.resize((target, target), Image.NEAREST)


# ---------------------------------------------------------------- discovery

def iter_images(path: Path, limit=None, recursive=True):
    if path.is_file():
        return [path]
    globber = path.rglob if recursive else path.glob
    found = sorted({p for ext in IMG_EXTS for p in globber(f"*{ext}")})
    return found[:limit] if limit else found


# ---------------------------------------------------------------- HTML

CSS = """
body{background:#12181d;color:#dfe6e2;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px;color:#7fd6cf}
.sub{color:#93a29a;font-size:13px;margin-bottom:20px}
.card{background:#161e25;border:1px solid #2b3841;border-radius:10px;padding:14px;margin-bottom:16px}
.card h2{font-size:14px;margin:0 0 10px;color:#e0a552;font-family:ui-monospace,Consolas,monospace;word-break:break-all}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}
.cell{text-align:center}
.cell img{display:block;border:1px solid #2b3841;border-radius:6px;image-rendering:pixelated;background:#000}
.cap{font-size:11px;color:#93a29a;margin-top:5px;font-family:ui-monospace,Consolas,monospace}
.meta{font-size:11.5px;color:#93a29a;font-family:ui-monospace,Consolas,monospace;margin-top:8px}
.meta b{color:#dfe6e2}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.grid .cell img{width:100%;height:auto}
.tag{display:inline-block;background:#1f6b66;color:#fff;border-radius:999px;padding:1px 9px;font-size:11px;margin-right:6px}
"""


def html_page(title, subtitle, body):
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style></head><body><h1>{title}</h1>"
            f"<div class='sub'>{subtitle}</div>{body}</body></html>")


# ---------------------------------------------------------------- mode: browse

def mode_browse(args):
    src = Path(args.input)
    paths = iter_images(src, args.limit)
    if not paths:
        raise SystemExit(f"no images found under {src}")
    print(f"found {len(paths)} image(s) under {src}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for p in paths:
        arr, info = read_band(p)
        img = to_pil(stretch(arr, args.stretch, tuple(args.percentiles)))
        if out_dir:
            img.save(out_dir / f"{p.stem}.png")
        if args.html:
            cells.append(
                f"<div class='cell'><img src='{data_uri(img)}' alt='{p.name}'>"
                f"<div class='cap'>{p.stem}<br>{info['size']} {info['dtype']}<br>"
                f"DN {info['dn_min']:.0f}-{info['dn_max']:.0f}</div></div>")

    if out_dir:
        print(f"wrote {len(paths)} PNG(s) -> {out_dir}")
    if args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        sub = (f"{len(paths)} image(s) from {src} &middot; stretch: "
               f"<b>{args.stretch}</b> {tuple(args.percentiles) if args.stretch=='percentile' else ''}")
        out.write_text(html_page("TerraSR - stage preview", sub,
                                  f"<div class='grid'>{''.join(cells)}</div>"), encoding="utf-8")
        print(f"wrote contact sheet -> {out}   (open it in any browser)")


# ---------------------------------------------------------------- mode: pairs

def load_degradation_params(pairs_dir: Path):
    mf = pairs_dir / "degradation_manifest.json"
    if not mf.exists():
        return {}
    return {r["id"]: r.get("degradation_params", {}) for r in json.loads(mf.read_text())}


def mode_pairs(args):
    pairs_dir = Path(args.pairs)
    hr_dir, lr_dir = pairs_dir / "hr", pairs_dir / "lr"
    if not hr_dir.is_dir() or not lr_dir.is_dir():
        raise SystemExit(f"expected {hr_dir} and {lr_dir} (a stage 5 --out-dir)")

    params_by_id = load_degradation_params(pairs_dir)
    hr_paths = iter_images(hr_dir, args.limit, recursive=False)
    if not hr_paths:
        raise SystemExit(f"no HR images in {hr_dir}")

    disp, zoom = args.display, args.zoom
    cards = []
    for hp in hr_paths:
        lp = next((lr_dir / f"{hp.stem}{e}" for e in IMG_EXTS
                   if (lr_dir / f"{hp.stem}{e}").exists()), None)
        if lp is None:
            print(f"  skip {hp.stem}: no matching LR")
            continue

        hr, hr_info = read_band(hp)
        lr, lr_info = read_band(lp)
        st = (args.stretch, tuple(args.percentiles))

        # full view, both rendered at the same display size
        hr_img = to_pil(stretch(hr, *st)).resize((disp, disp), Image.NEAREST)
        lr_img = to_pil(stretch(lr, *st)).resize((disp, disp), Image.NEAREST)
        # zoom: matching centre region, magnified so pixel-level effects show
        hr_zoom = upscale(to_pil(stretch(centre_crop(hr, zoom), *st)), disp)
        lr_zoom = upscale(to_pil(stretch(centre_crop(lr, zoom // 2), *st)), disp)

        p = params_by_id.get(hp.stem, {})
        tags = " ".join(f"<span class='tag'>{k}: {v}</span>" for k, v in p.items()
                        if k in ("downsample_method", "scale_factor"))
        cards.append(
            f"<div class='card'><h2>{hp.stem}</h2>{tags}"
            f"<div class='row'>"
            f"<div class='cell'><img src='{data_uri(hr_img)}' width='{disp}'>"
            f"<div class='cap'>HR ground truth<br>{hr_info['size']}</div></div>"
            f"<div class='cell'><img src='{data_uri(lr_img)}' width='{disp}'>"
            f"<div class='cap'>LR degraded<br>{lr_info['size']}</div></div>"
            f"<div class='cell'><img src='{data_uri(hr_zoom)}' width='{disp}'>"
            f"<div class='cap'>HR zoom ({zoom}px centre)</div></div>"
            f"<div class='cell'><img src='{data_uri(lr_zoom)}' width='{disp}'>"
            f"<div class='cap'>LR zoom (same area)</div></div>"
            f"</div>"
            f"<div class='meta'>HR DN <b>{hr_info['dn_min']:.0f}-{hr_info['dn_max']:.0f}</b> "
            f"(mean {hr_info['dn_mean']:.0f}) &middot; "
            f"LR DN <b>{lr_info['dn_min']:.0f}-{lr_info['dn_max']:.0f}</b> "
            f"(mean {lr_info['dn_mean']:.0f})</div></div>")

    out = Path(args.html or "previews/degradation.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = (f"{len(cards)} pair(s) from {pairs_dir} &middot; stretch: <b>{args.stretch}</b> &middot; "
           f"compare the zoom columns to judge blur / noise / aliasing, then tune "
           f"<b>configs/degradation.yaml</b>")
    out.write_text(html_page("TerraSR - HR vs LR (degradation tuning)", sub, "".join(cards)),
                    encoding="utf-8")
    print(f"wrote {len(cards)} comparison(s) -> {out}   (open it in any browser)")


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="file or directory of stage output to view")
    src.add_argument("--pairs", help="a stage 5 output dir (containing hr/ and lr/)")
    ap.add_argument("--out-dir", help="write individual PNGs here (browse mode)")
    ap.add_argument("--html", help="write a self-contained HTML page here")
    ap.add_argument("--limit", type=int, default=None, help="max images (default: all)")
    ap.add_argument("--stretch", choices=["percentile", "minmax", "none"], default="percentile")
    ap.add_argument("--percentiles", type=float, nargs=2, default=[2.0, 98.0],
                     metavar=("LO", "HI"))
    ap.add_argument("--display", type=int, default=256, help="display size px (pairs mode)")
    ap.add_argument("--zoom", type=int, default=96,
                     help="HR centre-crop size to magnify (pairs mode)")
    args = ap.parse_args()

    if args.pairs:
        mode_pairs(args)
    else:
        if not args.out_dir and not args.html:
            args.out_dir = "previews"
            print("no --out-dir/--html given; defaulting to --out-dir previews")
        mode_browse(args)


if __name__ == "__main__":
    main()
