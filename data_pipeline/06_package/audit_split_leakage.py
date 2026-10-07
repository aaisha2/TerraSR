"""Stage 6c: prove the train/val/test split has no same-location leakage,
instead of assuming it.

The split is meant to separate ground, not file names. This measures whether
it did, by asking the only question that matters: for every test (and val)
patch, how far away is the nearest TRAIN patch on the ground?

If that distance is smaller than a patch, the two overlap: the model was
trained on the pixels it is being tested on, and the reported PSNR/SSIM is
partly a memorisation score. The check is run on centroids with a KD-tree in
a local metric projection, so it is exact enough at these scales and fast
enough for hundreds of thousands of patches.

Also reports the leakage routes that a scene-name split cannot see:
  - locations (block cells) that appear in more than one split
  - source scenes contributing to more than one split (expected under a
    spatial split, reported for transparency)
  - groups of DIFFERENT scenes covering the same ground, which is how repeat
    CORE3D collects, Maxar pre/post pairs and the pan-vs-visual asset pair of
    one Maxar scene leak
  - identical tile geometry appearing twice (exact duplicate ground)

Exits non-zero when a violation threshold is exceeded, so it can gate a
pipeline run rather than be read and ignored.

Usage:
    python audit_split_leakage.py --manifest data/dataset/dataset_manifest_split.csv
    python audit_split_leakage.py --manifest ... --report data/dataset/split_audit.md --strict
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

KM_PER_DEG_LAT = 111.32
EXCLUDED = "excluded_buffer"


def to_metres(lat: np.ndarray, lon: np.ndarray):
    """Local equirectangular projection about the dataset centroid. Distances
    between nearby points are accurate to well under a metre at these
    separations, which is all this needs."""
    lat0 = float(np.mean(lat))
    x = lon * KM_PER_DEG_LAT * 1000 * np.cos(np.radians(lat0))
    y = lat * KM_PER_DEG_LAT * 1000
    return np.column_stack([x, y])


def nearest_cross_split_distance(a_xy: np.ndarray, b_xy: np.ndarray):
    """For each point in a, the distance to the nearest point in b."""
    if len(a_xy) == 0 or len(b_xy) == 0:
        return np.array([])
    try:
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(b_xy).query(a_xy, k=1)
        return dist
    except ImportError:
        out = np.empty(len(a_xy))
        for i, p in enumerate(a_xy):
            out[i] = np.min(np.hypot(*(b_xy - p).T))
        return out


def patch_size_m(df: pd.DataFrame) -> float | None:
    """Patch footprint width in metres, from the recorded WGS84 bounds."""
    for val in df.get("bounds_wgs84", pd.Series(dtype=object)).dropna().head(50):
        try:
            b = json.loads(val) if isinstance(val, str) else list(val)
            lat_mid = (b[1] + b[3]) / 2
            width = (b[2] - b[0]) * KM_PER_DEG_LAT * 1000 * np.cos(np.radians(lat_mid))
            height = (b[3] - b[1]) * KM_PER_DEG_LAT * 1000
            if width > 0 and height > 0:
                return float(max(width, height))
        except Exception:
            continue
    return None


def same_ground_scene_groups(df: pd.DataFrame, block_col: str) -> dict:
    """Blocks whose patches come from more than one source scene — i.e. ground
    imaged more than once. These are the pairs a scene-name split would have
    separated while leaking the location."""
    if block_col not in df.columns or "source_scene" not in df.columns:
        return {}
    groups = {}
    for block, sub in df.groupby(block_col):
        scenes = sorted(set(sub["source_scene"].dropna()))
        if len(scenes) > 1:
            groups[str(block)] = scenes
    return groups


def audit(df: pd.DataFrame, cfg: dict) -> dict:
    ratios = list(cfg["ratios"])
    buffer_m = float(cfg.get("buffer_m", 250.0))
    block_col = "split_block" if "split_block" in df.columns else None

    have_geo = ("center_lat" in df.columns and "center_lon" in df.columns
                and df["center_lat"].notna().any())
    result = {"n_patches": len(df), "have_geo": bool(have_geo),
              "splits": {s: int((df["split"] == s).sum()) for s in ratios},
              "n_excluded_buffer": int((df["split"] == EXCLUDED).sum()),
              "buffer_m": buffer_m}

    # ---- routes a scene-name split cannot see -----------------------------
    if block_col:
        per_block = (df[df["split"] != EXCLUDED]
                     .groupby(block_col)["split"].nunique())
        shared = per_block[per_block > 1]
        result["blocks_in_multiple_splits"] = int(len(shared))
        result["example_shared_blocks"] = [str(b) for b in shared.index[:10]]
        groups = same_ground_scene_groups(df[df["split"] != EXCLUDED], block_col)
        result["blocks_imaged_by_multiple_scenes"] = len(groups)
        result["example_multi_scene_blocks"] = dict(list(groups.items())[:5])
        # of those, how many straddle splits (the actual leak)
        leaking = {}
        for block, scenes in groups.items():
            sub = df[(df[block_col].astype(str) == block) & (df["split"] != EXCLUDED)]
            if sub["split"].nunique() > 1:
                leaking[block] = scenes
        result["multi_scene_blocks_straddling_splits"] = len(leaking)
        result["example_leaking_blocks"] = dict(list(leaking.items())[:5])

    if "source_scene" in df.columns:
        per_scene = (df[df["split"] != EXCLUDED]
                     .groupby("source_scene")["split"].nunique())
        result["scenes_in_multiple_splits"] = int((per_scene > 1).sum())

    # exact duplicate ground: identical rounded centroid in two splits
    if have_geo:
        key = (df["center_lat"].round(6).astype(str) + ","
               + df["center_lon"].round(6).astype(str))
        dup = df.assign(_k=key)
        dup = dup[dup["split"] != EXCLUDED].groupby("_k")["split"].nunique()
        result["identical_locations_in_multiple_splits"] = int((dup > 1).sum())

    # ---- the measurement that matters -------------------------------------
    if not have_geo:
        result["note"] = ("manifest has no patch geography, so cross-split ground "
                          "distance cannot be measured - this audit cannot clear "
                          "the split")
        return result

    geo = df[df["center_lat"].notna() & df["center_lon"].notna()]
    xy = to_metres(geo["center_lat"].to_numpy(float), geo["center_lon"].to_numpy(float))
    split_of = geo["split"].to_numpy()
    train_xy = xy[split_of == "train"]

    psize = patch_size_m(geo)
    result["patch_footprint_m"] = psize
    overlap_threshold = psize if psize else 100.0
    result["overlap_threshold_m"] = overlap_threshold

    result["nearest_train"] = {}
    for split in ("val", "test"):
        sub_xy = xy[split_of == split]
        dist = nearest_cross_split_distance(sub_xy, train_xy)
        if not len(dist):
            result["nearest_train"][split] = {"n": 0}
            continue
        result["nearest_train"][split] = {
            "n": int(len(dist)),
            "min_m": float(dist.min()),
            "p01_m": float(np.percentile(dist, 1)),
            "median_m": float(np.median(dist)),
            "n_overlapping": int((dist < overlap_threshold).sum()),
            "n_within_buffer": int((dist < buffer_m).sum()),
            "frac_overlapping": float((dist < overlap_threshold).mean()),
        }
    return result


def verdicts(a: dict, max_overlap_frac: float) -> list:
    out = []
    if not a["have_geo"]:
        out.append(("split can be audited at all", False,
                    "no patch geography in the manifest, so same-location leakage "
                    "cannot be ruled out - run backfill_patch_geo.py and re-split"))
        return out

    for split in ("val", "test"):
        s = a["nearest_train"].get(split) or {}
        if not s.get("n"):
            out.append((f"{split} has patches", False, "empty split"))
            continue
        frac = s["frac_overlapping"]
        out.append((f"{split} patches do not overlap train ground",
                    frac <= max_overlap_frac,
                    f"{s['n_overlapping']}/{s['n']} ({100*frac:.2f}%) within "
                    f"{a['overlap_threshold_m']:.0f} m of a train patch "
                    f"(allowed {100*max_overlap_frac:.2f}%); nearest "
                    f"{s['min_m']:.0f} m, median {s['median_m']:.0f} m"))

    if "identical_locations_in_multiple_splits" in a:
        n = a["identical_locations_in_multiple_splits"]
        out.append(("no identical location in two splits", n == 0,
                    f"{n} location(s) appear in more than one split"))
    if "blocks_in_multiple_splits" in a:
        n = a["blocks_in_multiple_splits"]
        out.append(("no block spans two splits", n == 0,
                    f"{n} block(s) span splits"
                    + (f", e.g. {a['example_shared_blocks'][:3]}" if n else "")))
    if "multi_scene_blocks_straddling_splits" in a:
        n = a["multi_scene_blocks_straddling_splits"]
        out.append(("repeat coverage of one location stays in one split", n == 0,
                    f"{n} location(s) imaged by several scenes straddle splits"
                    + (f", e.g. {list(a['example_leaking_blocks'])[:2]}" if n else "")))
    return out


def render(a: dict, v: list) -> str:
    L = ["# Train/val/test split leakage audit", "",
         f"{a['n_patches']} patches | splits {a['splits']} | "
         f"{a['n_excluded_buffer']} excluded as buffer", ""]
    L += ["| check | result | detail |", "|---|---|---|"]
    for name, ok, detail in v:
        L.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    L.append("")

    if a["have_geo"]:
        L += ["## Distance from each held-out patch to the nearest training patch", "",
              f"Patch footprint {a.get('patch_footprint_m') or 'unknown'} m; a "
              f"separation below that means the two patches cover overlapping "
              f"ground.", "",
              "| split | n | nearest | 1st pct | median | overlapping train | within buffer |",
              "|---|---|---|---|---|---|---|"]
        for split in ("val", "test"):
            s = a["nearest_train"].get(split) or {}
            if not s.get("n"):
                L.append(f"| {split} | 0 | - | - | - | - | - |")
                continue
            L.append(f"| {split} | {s['n']} | {s['min_m']:.0f} m | {s['p01_m']:.0f} m | "
                     f"{s['median_m']:.0f} m | {s['n_overlapping']} "
                     f"({100*s['frac_overlapping']:.2f}%) | {s['n_within_buffer']} |")
        L.append("")

    L += ["## Repeat coverage of the same ground", "",
          f"- locations imaged by more than one source scene: "
          f"{a.get('blocks_imaged_by_multiple_scenes', 'n/a')}",
          f"- of those, straddling splits: "
          f"{a.get('multi_scene_blocks_straddling_splits', 'n/a')}",
          f"- source scenes contributing to more than one split: "
          f"{a.get('scenes_in_multiple_splits', 'n/a')} "
          f"(expected under a spatial split - a scene covers many blocks)", "",
          "Repeat coverage is the leak a scene-name split cannot see: repeat "
          "CORE3D collects of one site, Maxar pre/post-event acquisitions, the "
          "`pan_analytic` and `visual` assets of a single Maxar scene, and "
          "overlapping SpaceNet strips are all the same ground under different "
          "scene names.", ""]
    if a.get("example_multi_scene_blocks"):
        L += ["Examples:", ""]
        for block, scenes in a["example_multi_scene_blocks"].items():
            L.append(f"- `{block}`: {', '.join(scenes[:4])}"
                     + (" ..." if len(scenes) > 4 else ""))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="dataset_manifest_split.csv / .parquet from stage 6b")
    ap.add_argument("--config", default=Path("configs/split.yaml"), type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--max-overlap-fraction", type=float, default=0.0,
                     help="tolerated share of held-out patches overlapping train "
                          "ground (default 0 = none allowed)")
    ap.add_argument("--strict", action="store_true",
                     help="exit non-zero if any check fails")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["split"]
    df = (pd.read_parquet(args.manifest) if args.manifest.suffix == ".parquet"
          else pd.read_csv(args.manifest))
    if "split" not in df.columns:
        raise SystemExit(f"{args.manifest} has no 'split' column - run stage 6b first")

    a = audit(df, cfg)
    v = verdicts(a, args.max_overlap_fraction)

    print(f"{'check':<52}{'result':<8}detail")
    print("-" * 110)
    for name, ok, detail in v:
        print(f"{name:<52}{'PASS' if ok else 'FAIL':<8}{detail}")
    n_failed = sum(1 for _, ok, _ in v if not ok)
    print(f"\n{len(v) - n_failed}/{len(v)} checks passed")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render(a, v), encoding="utf-8")
        args.report.with_suffix(".json").write_text(
            json.dumps({"audit": a, "verdicts": [
                {"check": n, "passed": ok, "detail": d} for n, ok, d in v]},
                indent=2, default=str), encoding="utf-8")
        print(f"report -> {args.report}")

    if n_failed and args.strict:
        raise SystemExit(f"{n_failed} leakage check(s) failed")


if __name__ == "__main__":
    main()
