"""Stage 2 batch driver: standardize every raw scene in a directory
(extract PAN/pseudo-PAN band -> normalize CRS/dtype) in one call, chaining
extract_pan_band.py and to_geotiff.py internally via a temp intermediate.

Used by the full-run orchestrator (run_pipeline.py): each source is
standardized with its own PAN mode (SpaceNet/Maxar = true_pan,
OpenEarthMap/NAIP = rgb_to_pseudo_pan), so the pseudo_pan tag is set
correctly per source.

Idempotent: skips scenes whose standardized output already exists.

Usage:
    python standardize_scenes.py --in-dir data/raw/spacenet --out-dir data/standardized/spacenet --mode true_pan
    python standardize_scenes.py --in-dir data/raw/openearthmap --out-dir data/standardized/openearthmap --mode rgb_to_pseudo_pan
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_pan_band import extract_pan_to_file  # noqa: E402
from to_geotiff import standardize_to_file  # noqa: E402

RASTER_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}


def iter_scenes(in_dir: Path, recursive: bool):
    globber = in_dir.rglob if recursive else in_dir.glob
    seen = set()
    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        for p in globber(ext):
            if p.resolve() not in seen:
                seen.add(p.resolve())
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["true_pan", "rgb_to_pseudo_pan"])
    ap.add_argument("--band-index", type=int, default=1)
    ap.add_argument("--recursive", action="store_true",
                     help="search sub-directories (e.g. SpaceNet AOI folders)")
    args = ap.parse_args()

    scenes = list(iter_scenes(args.in_dir, args.recursive))
    if not scenes:
        raise SystemExit(f"no rasters found under {args.in_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done, skipped, failed = 0, 0, 0
    for scene in scenes:
        out_path = args.out_dir / f"{scene.stem}.tif"
        if out_path.exists():
            skipped += 1
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "extracted.tif"
                extract_pan_to_file(scene, tmp, args.mode, args.band_index)
                info = standardize_to_file(tmp, out_path)
            done += 1
            print(f"  {scene.name} -> {out_path.name}  "
                  f"crs={info['crs']} reprojected={info['reprojected']}")
        except Exception as e:
            failed += 1
            print(f"  FAILED {scene.name}: {e}")

    print(f"\nstandardized {done}, skipped {skipped} (already done), failed {failed} "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
