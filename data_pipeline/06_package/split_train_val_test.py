"""Stage 6b: assign each patch to train/val/test by GEOGRAPHIC BLOCK — a block
of ground, not a scene name.

WHY THIS CHANGED (2026-10-07). The split used to treat `source_scene` as the
atomic unit: whole scenes went to one split. That stops *adjacent-patch*
leakage within a scene, but it does not stop SAME-LOCATION leakage, because
several of this project's sources cover the same ground under different scene
names:

  - CORE3D is ~154 scenes over 5 sites, i.e. tens of repeat WorldView collects
    of the same city. Scene-level splitting puts the same buildings, from a
    different pass, in train and in test.
  - Maxar Open Data events ship pre-event and post-event acquisitions of the
    same quadkeys.
  - `maxar_visual` (Experiment 2's pseudo-PAN) is the *same scenes* as `maxar`,
    just the `visual` asset instead of `pan_analytic` — the same ground, with
    a different file name.
  - SpaceNet strips within one AOI overlap each other at the edges.

Any of those puts near-identical ground on both sides of the split and
inflates test PSNR/SSIM. A model does not have to generalise to score well on
ground it has already memorised.

So the unit is now a geographic block: patch centroids are quantised to a grid
of `block_size_km`, whole blocks are assigned to splits, and patches within
`buffer_m` of a block assigned to a *different* split are dropped entirely
(recorded as split='excluded_buffer'). Blocks, not scenes, are what a location
is; dropping the boundary band stops near-duplicate patches straddling two
blocks from leaking across the seam. A scene may contribute to several blocks
and therefore to several splits — that is fine and intended, because those are
different ground.

Assignment within that is unchanged: a normalized-deficit greedy over blocks
(largest first, ties broken by a seeded shuffle), each going to the split with
the largest proportional remaining deficit. That converges to the configured
ratios with many blocks and still spreads a handful of blocks across splits.

`mode: scene` restores the old behaviour for datasets with no geography (the
UC Merced test fixtures have no CRS); `mode: auto` uses blocks when every row
has a centroid and falls back to scenes with a loud warning when it cannot.
Run audit_split_leakage.py afterwards to verify the result rather than trust it.

Usage:
    python split_train_val_test.py --manifest out/dataset/dataset_manifest.parquet
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXCLUDED = "excluded_buffer"
KM_PER_DEG_LAT = 111.32


def assign_blocks_to_splits(block_sizes: dict, ratios: dict, seed: int) -> dict:
    total = sum(block_sizes.values())
    targets = {split: total * frac for split, frac in ratios.items()}
    current = {split: 0 for split in ratios}

    rng = random.Random(seed)
    blocks = list(block_sizes.items())
    rng.shuffle(blocks)                                  # seeded tie-break
    blocks.sort(key=lambda kv: kv[1], reverse=True)      # largest block first

    assignment = {}
    for block, size in blocks:
        def deficit(split):
            t = targets[split]
            return (t - current[split]) / t if t > 0 else float("-inf")
        chosen = max(ratios, key=deficit)
        assignment[block] = chosen
        current[chosen] += size
    return assignment


def block_ids(df: pd.DataFrame, block_size_km: float):
    """Quantise patch centroids onto an equal-area-ish grid.

    Longitude degrees shrink with latitude, so the longitude step is widened
    by 1/cos(lat) per row; blocks are then roughly block_size_km on a side
    everywhere instead of collapsing near the poles."""
    lat = df["center_lat"].to_numpy(dtype=float)
    lon = df["center_lon"].to_numpy(dtype=float)
    lat_step = block_size_km / KM_PER_DEG_LAT
    coslat = np.clip(np.cos(np.radians(lat)), 1e-6, None)
    lon_step = lat_step / coslat
    iy = np.floor(lat / lat_step).astype(np.int64)
    ix = np.floor(lon / lon_step).astype(np.int64)
    return [f"blk_{y}_{x}" for y, x in zip(iy, ix)], lat_step, lon_step


def buffer_excluded(df: pd.DataFrame, assignment: dict, buffer_m: float,
                     block_size_km: float) -> np.ndarray:
    """Boolean mask of patches to drop: those within buffer_m of a patch
    assigned to a different split.

    Only cross-split pairs matter, and only near block seams, so this compares
    each patch against the patches of differently-assigned nearby blocks
    rather than against the whole dataset. `reach` is how many rings of
    neighbouring blocks that has to span: one ring is enough while the buffer
    is narrower than a block, more if it is not."""
    if buffer_m <= 0:
        return np.zeros(len(df), dtype=bool)
    reach = max(1, int(math.ceil(buffer_m / (block_size_km * 1000.0))))

    lat = df["center_lat"].to_numpy(dtype=float)
    lon = df["center_lon"].to_numpy(dtype=float)
    split = df["_block"].map(assignment).to_numpy()
    blocks = df["_block"].to_numpy()

    # index patches by block for neighbour lookup
    idx_by_block = {}
    for i, b in enumerate(blocks):
        idx_by_block.setdefault(b, []).append(i)

    drop = np.zeros(len(df), dtype=bool)

    def parse(b):
        _, y, x = b.split("_")
        return int(y), int(x)

    for block, members in idx_by_block.items():
        y, x = parse(block)
        my_split = assignment[block]
        neigh = []
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if dy == 0 and dx == 0:
                    continue
                nb = f"blk_{y+dy}_{x+dx}"
                if nb in idx_by_block and assignment.get(nb) != my_split:
                    neigh.extend(idx_by_block[nb])
        if not neigh:
            continue
        neigh = np.asarray(neigh)
        mem = np.asarray(members)
        # great-circle-ish distance in metres (small separations, so a local
        # flat-earth approximation is well within tolerance)
        dlat = (lat[mem][:, None] - lat[neigh][None, :]) * KM_PER_DEG_LAT * 1000
        mean_lat = np.radians((lat[mem][:, None] + lat[neigh][None, :]) / 2)
        dlon = ((lon[mem][:, None] - lon[neigh][None, :])
                * KM_PER_DEG_LAT * 1000 * np.cos(mean_lat))
        dist = np.hypot(dlat, dlon)
        close = dist.min(axis=1) <= buffer_m
        drop[mem[close]] = True
        # and the neighbours that are close to these
        close_n = dist.min(axis=0) <= buffer_m
        drop[neigh[close_n]] = True
    return drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="unified manifest from build_manifest.py (.parquet or .csv)")
    ap.add_argument("--config", default=Path("configs/split.yaml"), type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="default: alongside the input manifest")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["split"]
    ratios = cfg["ratios"]
    seed = cfg["seed"]
    mode = cfg.get("mode", "auto")
    block_km = float(cfg.get("block_size_km", 1.0))
    buffer_m = float(cfg.get("buffer_m", 250.0))

    if args.manifest.suffix == ".parquet":
        df = pd.read_parquet(args.manifest)
    else:
        df = pd.read_csv(args.manifest)

    have_geo = ("center_lat" in df.columns and "center_lon" in df.columns
                and df["center_lat"].notna().all() and df["center_lon"].notna().all())

    if mode == "spatial" and not have_geo:
        raise SystemExit(
            "split.mode=spatial but the manifest has no complete patch geography "
            "(center_lat/center_lon). Run data_pipeline/06_package/"
            "backfill_patch_geo.py first, or set split.mode=scene if these "
            "patches genuinely have no CRS.")

    use_spatial = have_geo and mode in ("auto", "spatial")

    if use_spatial:
        df["_block"], _, _ = block_ids(df, block_km)
        unit_label = f"spatial block ({block_km:g} km)"
    else:
        reason = ("split.mode=scene" if mode == "scene"
                  else "the manifest has no patch geography")
        print(f"WARNING: splitting by SOURCE SCENE because {reason}.")
        print("  A scene-name split does NOT prevent same-location leakage: repeat")
        print("  collects of one site, pre/post-event pairs, and the pan vs visual")
        print("  assets of one scene are all the same ground under different names.")
        print("  Fix by recording patch geography (stage 3 does this now; use")
        print("  backfill_patch_geo.py for an existing dataset), then re-run.")
        df["_block"] = df[cfg.get("group_by", "source_scene")]
        unit_label = f"source scene ({cfg.get('group_by', 'source_scene')})"

    block_sizes = df.groupby("_block").size().to_dict()
    n_blocks = len(block_sizes)
    if n_blocks < len(ratios):
        print(f"WARNING: only {n_blocks} {unit_label}(s) for {len(ratios)} splits — "
              f"some splits may be empty or far from the target ratios. Expected on "
              f"a tiny sample; the real dataset has many blocks per terrain.")

    assignment = assign_blocks_to_splits(block_sizes, ratios, seed)
    df["split"] = df["_block"].map(assignment)

    n_buffered = 0
    if use_spatial and buffer_m > 0:
        drop = buffer_excluded(df, assignment, buffer_m, block_km)
        df.loc[drop, "split"] = EXCLUDED
        n_buffered = int(drop.sum())

    out_dir = args.out_dir or args.manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.rename(columns={"_block": "split_block"})
    split_parquet = out_dir / "dataset_manifest_split.parquet"
    split_csv = out_dir / "dataset_manifest_split.csv"
    df_out.to_parquet(split_parquet, index=False)
    df_out.to_csv(split_csv, index=False)

    # per-split CSVs for training/eval. The buffer band is in neither.
    for split in ratios:
        df_out[df_out["split"] == split].to_csv(out_dir / f"{split}.csv", index=False)

    print(f"split unit: {unit_label} | {n_blocks} unit(s) | seed {seed}")
    if n_blocks <= 40:
        print(f"\n{'unit':<34}{'split':<8}patches")
        for block, split in sorted(assignment.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  {str(block):<32}{split:<8}{block_sizes[block]}")

    print("\npatches per split (actual vs target):")
    total = len(df_out)
    for split, frac in ratios.items():
        n = int((df_out["split"] == split).sum())
        print(f"  {split:<16}{n:>6}  ({100*n/total:5.1f}%  target {100*frac:.0f}%)")
    if n_buffered:
        print(f"  {EXCLUDED:<16}{n_buffered:>6}  ({100*n_buffered/total:5.1f}%)  "
              f"within {buffer_m:g} m of a differently-assigned block")

    # The buffer band is a real cost, and if it is large relative to a block it
    # can swallow a whole split. Say so loudly with the remedy, rather than
    # quietly handing over an empty test set.
    if n_buffered:
        frac = n_buffered / total
        empty = [s for s in ratios if int((df_out["split"] == s).sum()) == 0]
        if frac > 0.25 or empty:
            print(f"\nWARNING: the buffer band excluded {100*frac:.1f}% of patches"
                  + (f" and left {empty} empty" if empty else "") + ".")
            print(f"  buffer_m={buffer_m:g} is large relative to "
                  f"block_size_km={block_km:g}: blocks of ground smaller than the "
                  f"buffer are removed entirely.")
            print("  Remedy: raise split.block_size_km (bigger blocks, fewer seams) "
                  "or lower split.buffer_m. Do not remove the buffer - it is what "
                  "stops near-duplicate patches leaking across a block seam.")

    if use_spatial:
        # how much the old scene-level split would have leaked, stated plainly
        scenes_spanning = 0
        if "source_scene" in df_out.columns:
            per_scene = df_out[df_out["split"] != EXCLUDED].groupby("source_scene")["split"].nunique()
            scenes_spanning = int((per_scene > 1).sum())
        print(f"\n{scenes_spanning} source scene(s) contribute to more than one split. "
              f"That is expected and correct here: a scene covers many blocks of "
              f"ground, and blocks are the unit. Run audit_split_leakage.py to "
              f"confirm no two splits share a location.")

    print(f"\n-> {split_parquet}")
    print(f"-> {split_csv}")


if __name__ == "__main__":
    main()
