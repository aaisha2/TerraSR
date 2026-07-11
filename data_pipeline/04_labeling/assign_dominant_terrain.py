"""Stage 4b: turn per-patch WorldCover histograms into a single dominant
terrain label (+ purity), using the class mapping in
configs/terrain_classes.yaml. Automated only -- no manual annotation, per
the confirmed labeling approach (docs/plan §1).

Ambiguous/mixed patches are NOT discarded by default -- the dominant label
is kept along with its purity ratio, so a purity-filtered ablation is a
config change (`purity.min_purity_to_keep_label`), not a re-run of this
whole stage.

Mountain is not derivable from WorldCover land cover alone (it has no
landform classes) -- see the note in configs/terrain_classes.yaml. A patch
only gets labeled Mountain via `scene_terrain_overrides` (a source-scene
name match), never from the pixel histogram.

Usage:
    python assign_dominant_terrain.py --manifest out/patches/patch_manifest_zonal.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def bucket_counts(histogram: dict, class_to_terrain: dict) -> dict:
    buckets = defaultdict(int)
    unmapped = 0
    for class_code_str, count in histogram.items():
        class_code = int(class_code_str)
        terrain = class_to_terrain.get(class_code)
        if terrain is None:
            unmapped += count
            continue
        buckets[terrain] += count
    if unmapped:
        buckets["Unmapped"] += unmapped
    return dict(buckets)


def scene_override(source_scene: str, overrides: dict):
    if not overrides:
        return None
    lowered = source_scene.lower()
    for substring, terrain in overrides.items():
        if substring.lower() in lowered:
            return terrain
    return None


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
    min_purity = cfg["purity"]["min_purity_to_keep_label"]

    manifest = json.loads(args.manifest.read_text())

    counts_by_terrain = defaultdict(int)
    n_overridden = 0
    for row in manifest:
        forced = scene_override(row["source_scene"], overrides)
        if forced is not None:
            row["terrain_label"] = forced
            row["terrain_purity"] = 1.0
            row["terrain_source"] = "scene_override"
            counts_by_terrain[forced] += 1
            n_overridden += 1
            continue

        histogram = row.get("worldcover_histogram")
        if not histogram:
            row["terrain_label"] = None
            row["terrain_purity"] = 0.0
            row["terrain_source"] = "missing_zonal_stats"
            continue

        buckets = bucket_counts(histogram, class_to_terrain)
        total = sum(buckets.values())
        dominant_terrain, dominant_count = max(buckets.items(), key=lambda kv: kv[1])
        purity = dominant_count / total if total else 0.0

        row["terrain_label"] = dominant_terrain if purity >= min_purity else None
        row["terrain_purity"] = purity
        row["terrain_source"] = "worldcover"
        row["terrain_bucket_counts"] = buckets
        if row["terrain_label"]:
            counts_by_terrain[dominant_terrain] += 1

    out_manifest = args.out_manifest or args.manifest.parent / "patch_manifest_labeled.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))

    print(f"labeled {sum(counts_by_terrain.values())} patches "
          f"({n_overridden} via scene_terrain_overrides)")
    for terrain, count in sorted(counts_by_terrain.items(), key=lambda kv: -kv[1]):
        print(f"  {terrain:<20} {count}")
    print(f"\nmanifest -> {out_manifest}")


if __name__ == "__main__":
    main()
