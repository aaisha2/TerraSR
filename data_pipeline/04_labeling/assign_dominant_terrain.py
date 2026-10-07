"""Stage 4b: turn per-patch WorldCover histograms (stage 4a) and DEM slope
statistics (stage 4a-bis) into a terrain label with an explicit confidence,
using configs/terrain_classes.yaml. Automated only -- no manual annotation,
per the confirmed labeling approach (docs/plan §1).

LABEL QUALITY (revised 2026-10-07). Three problems were fixed here; they all
produced labels that looked clean in the manifest and were not:

 1. Nodata could win the vote. Unmapped WorldCover codes (0 = no data, which
    every coastal or scene-edge patch has) were bucketed as "Unmapped" and
    entered the majority vote like a real class, so a patch that was mostly
    nodata was labelled "Unmapped" -- a string absent from terrain_index,
    which the Dataset then silently mapped to the unknown embedding. Unmapped
    pixels are now excluded from the vote and reported as their own fraction,
    and a patch with too little mapped area is left unlabelled with a reason.

 2. No purity threshold. min_purity_to_keep_label defaulted to 0.0, so a
    patch that was 34% forest and 33% urban was labelled Forest with the same
    authority as a 99% forest patch. The terrain conditioning and the
    per-terrain loss were trained on that noise. There is now a real
    threshold, plus a recorded margin over the runner-up, so a reader can see
    how decisive each label was.

 3. Mountain was unobtainable. It came only from `scene_terrain_overrides`
    (empty), so terrain index 4 had an embedding row and a loss weight that no
    sample ever used. It now comes from DEM slope/relief (stage 4a-bis).

Every label records how it was decided (`terrain_source`) and how confident
it is (`terrain_purity`, `terrain_margin`, `terrain_decisive`), so
label_quality_report.py can audit the result and the write-up can state the
label quality instead of assuming it.

NOTE ON THE TAXONOMY: Mountain is a landform while the other six are land
cover, so they are not mutually exclusive -- forested mountains are both.
`mountain.precedence` decides what wins, and the choice is recorded per patch.
This is a known limitation of the 7-class scheme, not something this script
can resolve.

Usage:
    python assign_dominant_terrain.py --manifest out/patches/patch_manifest_zonal.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

# WorldCover codes that carry no land-cover information. 0 is the product's
# nodata value; it is not a class and must never win a majority vote.
UNMAPPED_SENTINELS = {0}


def bucket_counts(histogram: dict, class_to_terrain: dict) -> tuple:
    """Returns (terrain_buckets, unmapped_count). Unmapped pixels are kept
    OUT of the buckets so they cannot win the vote, and returned separately
    so the caller can judge how much of the patch had no label at all."""
    buckets = defaultdict(int)
    unmapped = 0
    for class_code_str, count in histogram.items():
        class_code = int(class_code_str)
        terrain = class_to_terrain.get(class_code)
        if terrain is None or class_code in UNMAPPED_SENTINELS:
            unmapped += count
            continue
        buckets[terrain] += count
    return dict(buckets), unmapped


def scene_override(source_scene: str, overrides: dict):
    if not overrides:
        return None
    lowered = str(source_scene).lower()
    for substring, terrain in overrides.items():
        if substring.lower() in lowered:
            return terrain
    return None


def is_mountain(dem_stats: dict, mcfg: dict) -> bool:
    """Standard geomorphometric test, either criterion sufficient: steep
    enough on average, or enough local relief across the patch's
    neighbourhood (a cliff and a rolling hill differ on the second)."""
    if not dem_stats or not dem_stats.get("valid"):
        return False
    slope = dem_stats.get("slope_mean_deg")
    relief = dem_stats.get("relief_m")
    if slope is not None and slope >= float(mcfg.get("min_slope_mean_deg", 15.0)):
        return True
    if relief is not None and relief >= float(mcfg.get("min_relief_m", 200.0)):
        return True
    return False


def label_patch(row: dict, class_to_terrain: dict, overrides: dict,
                 min_purity: float, min_mapped_fraction: float, mcfg: dict) -> dict:
    """Decide one patch's label. Returns the fields to merge into the row."""
    out = {"terrain_label": None, "terrain_purity": 0.0, "terrain_margin": 0.0,
           "terrain_runner_up": None, "terrain_source": None,
           "terrain_unmapped_fraction": None, "terrain_decisive": False,
           "terrain_reject_reason": None}

    forced = scene_override(row.get("source_scene", ""), overrides)
    if forced is not None:
        out.update(terrain_label=forced, terrain_purity=1.0, terrain_margin=1.0,
                   terrain_source="scene_override", terrain_decisive=True)
        return out

    histogram = row.get("worldcover_histogram")
    if not histogram:
        out.update(terrain_source="missing_zonal_stats",
                   terrain_reject_reason="no WorldCover histogram")
        return out

    buckets, unmapped = bucket_counts(histogram, class_to_terrain)
    total_px = sum(buckets.values()) + unmapped
    mapped = sum(buckets.values())
    out["terrain_unmapped_fraction"] = (unmapped / total_px) if total_px else 1.0

    if not mapped or (total_px and mapped / total_px < min_mapped_fraction):
        out.update(terrain_source="worldcover",
                   terrain_reject_reason=(
                       f"only {100*(mapped/total_px if total_px else 0):.0f}% of the "
                       f"patch has a mapped land-cover class "
                       f"(need {100*min_mapped_fraction:.0f}%)"))
        return out

    ranked = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)
    dominant, dominant_count = ranked[0]
    runner_up, runner_count = ranked[1] if len(ranked) > 1 else (None, 0)
    purity = dominant_count / mapped          # over MAPPED pixels only
    margin = (dominant_count - runner_count) / mapped

    out.update(terrain_purity=purity, terrain_margin=margin,
               terrain_runner_up=runner_up, terrain_source="worldcover",
               terrain_bucket_counts=buckets)

    # Mountain is a landform, decided from the DEM, not the land-cover vote.
    mountain = is_mountain(row.get("dem_stats"), mcfg) if mcfg.get("enabled", True) else False
    precedence = mcfg.get("precedence", "always")

    if purity >= min_purity:
        out["terrain_label"] = dominant
        out["terrain_decisive"] = True
    else:
        out["terrain_reject_reason"] = (
            f"dominant class {dominant} has purity {purity:.2f} < {min_purity:.2f} "
            f"(runner-up {runner_up}, margin {margin:.2f})")

    if mountain:
        ambiguous = set(mcfg.get("overridable_labels") or [])
        if precedence == "always" or out["terrain_label"] is None \
                or (precedence == "only_if_ambiguous" and dominant in ambiguous):
            out["terrain_label"] = "Mountain"
            out["terrain_source"] = "dem_slope"
            out["terrain_decisive"] = True
            out["terrain_landcover_under_mountain"] = dominant
            out["terrain_reject_reason"] = None
        else:
            # keep the land-cover label but record that the ground is steep
            out["terrain_is_steep"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/terrain_classes.yaml"), type=Path)
    ap.add_argument("--out-manifest", type=Path, default=None,
                     help="default: <manifest_dir>/patch_manifest_labeled.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    class_to_terrain = cfg["class_to_terrain"]
    overrides = cfg.get("scene_terrain_overrides") or {}
    purity_cfg = cfg.get("purity") or {}
    min_purity = float(purity_cfg.get("min_purity_to_keep_label", 0.0))
    min_mapped = float(purity_cfg.get("min_mapped_fraction", 0.5))
    mcfg = cfg.get("mountain") or {}

    manifest = json.loads(args.manifest.read_text())

    counts_by_terrain = defaultdict(int)
    by_source = defaultdict(int)
    n_unlabeled = 0
    for row in manifest:
        row.update(label_patch(row, class_to_terrain, overrides,
                                min_purity, min_mapped, mcfg))
        if row["terrain_label"]:
            counts_by_terrain[row["terrain_label"]] += 1
            by_source[row["terrain_source"]] += 1
        else:
            n_unlabeled += 1

    out_manifest = args.out_manifest or args.manifest.parent / "patch_manifest_labeled.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))

    n_labeled = sum(counts_by_terrain.values())
    print(f"labeled {n_labeled}/{len(manifest)} patches "
          f"(purity threshold {min_purity:.2f}, min mapped fraction {min_mapped:.2f})")
    for terrain, count in sorted(counts_by_terrain.items(), key=lambda kv: -kv[1]):
        print(f"  {terrain:<20} {count}")
    print(f"  {'(unlabeled)':<20} {n_unlabeled}")
    print(f"\nby label source: {dict(by_source)}")

    missing = [t for t in cfg["terrain_index"] if t not in counts_by_terrain]
    if missing:
        print(f"\nWARNING: {len(missing)} terrain class(es) have ZERO labelled patches: "
              f"{missing}")
        print("  These still occupy rows in the stage 8 terrain embedding and weights "
              "in the terrain-aware loss, and will show as empty rows in per-terrain "
              "evaluation. Either source imagery for them or say so in the write-up - "
              "run label_quality_report.py for the full picture.")

    print(f"\nmanifest -> {out_manifest}")


if __name__ == "__main__":
    main()
