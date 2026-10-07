"""Stage 4 quality audit: how good are the terrain labels, really?

The labels are automated (ESA WorldCover majority vote + DEM slope), so their
quality is a property of the data and the thresholds, not something to assume.
Nothing measured it before, which let three problems sit unnoticed: nodata
winning majority votes, no purity threshold at all, and a terrain class that
could never be assigned. This script reports what the labels actually look
like, and emits a stratified sample for manual verification so a human-checked
accuracy can be quoted instead of nothing.

Reports:
  - support per class, and which classes are empty or below the per-class
    floor in configs/split.yaml
  - purity and margin distributions (how decisive each label was)
  - how many patches went unlabelled, and why
  - unmapped (nodata/snow) fraction distribution
  - how labels were decided (WorldCover vote / DEM slope / scene override)
  - Mountain: how many, and what land cover sits under them
  - the WorldCover epoch vs the imagery date, per source scene, so the
    temporal mismatch is explicit
  - a purity sweep: how support per class would change at other thresholds

Also writes `label_qc_sample.csv`: N patches per class, randomly chosen,
with their paths and the label the pipeline gave them. Open those in QGIS (or
view_patch.py), mark agree/disagree, and the agreement rate is a defensible
label-accuracy figure for the write-up.

Usage:
    python label_quality_report.py \
        --manifest out/patches/patch_manifest_labeled.json \
        --report out/patches/label_quality.md
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

PURITY_SWEEP = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "n/a"


def describe(values, fmt="{:.3f}"):
    if not len(values):
        return "no data"
    v = np.asarray(values, dtype=float)
    return (f"min {fmt.format(v.min())}, p25 {fmt.format(np.percentile(v, 25))}, "
            f"median {fmt.format(np.median(v))}, p75 {fmt.format(np.percentile(v, 75))}, "
            f"max {fmt.format(v.max())}")


def imagery_year(row: dict):
    """Best-effort acquisition year for a patch, from whatever the source
    tagged. Returns None when the source recorded no date."""
    for key in ("acquisition_date", "datetime", "TIFFTAG_DATETIME", "date"):
        val = row.get(key) or (row.get("tags") or {}).get(key)
        if val:
            digits = "".join(ch for ch in str(val) if ch.isdigit())
            if len(digits) >= 4 and digits[:4].isdigit():
                year = int(digits[:4])
                if 1990 <= year <= 2100:
                    return year
    return None


def analyse(manifest: list, cfg: dict, split_cfg: dict) -> dict:
    terrain_index = cfg["terrain_index"]
    floor = (split_cfg.get("stats") or {}).get("per_class_floor")
    epoch = (cfg.get("worldcover_source") or {}).get("epoch_year")

    labelled = [r for r in manifest if r.get("terrain_label")]
    unlabelled = [r for r in manifest if not r.get("terrain_label")]

    support = Counter(r["terrain_label"] for r in labelled)
    sources = Counter(r.get("terrain_source") or "unknown" for r in labelled)
    reasons = Counter(r.get("terrain_reject_reason") or
                      (r.get("terrain_source") or "unknown") for r in unlabelled)

    purity = [r.get("terrain_purity") or 0.0 for r in labelled]
    margin = [r.get("terrain_margin") or 0.0 for r in labelled]
    unmapped = [r["terrain_unmapped_fraction"] for r in manifest
                if r.get("terrain_unmapped_fraction") is not None]

    purity_by_class = defaultdict(list)
    for r in labelled:
        purity_by_class[r["terrain_label"]].append(r.get("terrain_purity") or 0.0)

    # purity sweep over the WorldCover-voted patches (DEM/override labels do
    # not depend on purity, so they are reported separately and unchanged)
    voted = [r for r in manifest if r.get("terrain_source") == "worldcover"
             and r.get("terrain_purity")]
    sweep = {}
    for thr in PURITY_SWEEP:
        kept = Counter()
        for r in voted:
            if (r.get("terrain_purity") or 0) >= thr:
                kept[r["terrain_label"] or r.get("terrain_runner_up")] += 1
        sweep[thr] = {"total": sum(kept.values()), "per_class": dict(kept)}

    mountains = [r for r in labelled if r["terrain_label"] == "Mountain"]
    under_mountain = Counter(r.get("terrain_landcover_under_mountain") or "unknown"
                             for r in mountains)
    steep_not_mountain = sum(1 for r in manifest if r.get("terrain_is_steep"))

    dem_valid = sum(1 for r in manifest
                    if (r.get("dem_stats") or {}).get("valid"))
    dem_missing = sum(1 for r in manifest if r.get("dem_stats") is not None
                      and not (r.get("dem_stats") or {}).get("valid"))
    dem_absent = sum(1 for r in manifest if r.get("dem_stats") is None)

    years = {}
    for r in manifest:
        y = imagery_year(r)
        scene = r.get("source_scene", "?")
        years.setdefault(scene, y)
    dated = {s: y for s, y in years.items() if y}

    return {
        "n_patches": len(manifest),
        "n_labelled": len(labelled),
        "n_unlabelled": len(unlabelled),
        "support": dict(support),
        "empty_classes": [t for t in terrain_index if t not in support],
        "below_floor": ({t: support.get(t, 0) for t in terrain_index
                         if support.get(t, 0) < floor} if floor else {}),
        "per_class_floor": floor,
        "label_sources": dict(sources),
        "reject_reasons": dict(reasons.most_common(10)),
        "purity": purity, "margin": margin, "unmapped_fraction": unmapped,
        "purity_by_class": {k: (len(v), float(np.median(v)))
                             for k, v in sorted(purity_by_class.items())},
        "purity_sweep": sweep,
        "n_mountain": len(mountains),
        "under_mountain": dict(under_mountain),
        "steep_but_not_mountain": steep_not_mountain,
        "dem": {"valid": dem_valid, "no_coverage": dem_missing, "not_run": dem_absent},
        "worldcover_epoch": epoch,
        "n_scenes": len(years),
        "n_scenes_dated": len(dated),
        "temporal_gaps": ({s: abs(epoch - y) for s, y in dated.items()}
                           if epoch else {}),
        "thresholds": {"min_purity": (cfg.get("purity") or {}).get("min_purity_to_keep_label"),
                        "min_mapped_fraction": (cfg.get("purity") or {}).get("min_mapped_fraction"),
                        "mountain": cfg.get("mountain") or {}},
    }


def write_qc_sample(manifest: list, out_csv: Path, per_class: int, seed: int) -> int:
    by_class = defaultdict(list)
    for r in manifest:
        if r.get("terrain_label"):
            by_class[r["terrain_label"]].append(r)

    rng = np.random.default_rng(seed)
    rows = []
    for terrain, items in sorted(by_class.items()):
        idx = rng.choice(len(items), size=min(per_class, len(items)), replace=False)
        for i in idx:
            r = items[int(i)]
            rows.append({
                "tile_id": r.get("tile_id"),
                "patch_path": r.get("patch_path"),
                "assigned_label": r.get("terrain_label"),
                "terrain_source": r.get("terrain_source"),
                "purity": round(r.get("terrain_purity") or 0.0, 3),
                "runner_up": r.get("terrain_runner_up"),
                "slope_mean_deg": round((r.get("dem_stats") or {}).get("slope_mean_deg") or 0.0, 1),
                "human_verdict": "",        # fill in: agree / <correct label>
                "notes": "",
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                            ["tile_id", "patch_path", "assigned_label", "human_verdict"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def render(a: dict, qc_csv: Path | None, n_qc: int) -> str:
    L = ["# Stage 4 terrain label quality", "",
         f"{a['n_labelled']}/{a['n_patches']} patches carry a terrain label "
         f"({pct(a['n_labelled'], a['n_patches'])}); "
         f"{a['n_unlabelled']} are unlabelled and train through the unknown "
         f"embedding.", "",
         f"Thresholds in force: purity >= {a['thresholds']['min_purity']}, "
         f"mapped fraction >= {a['thresholds']['min_mapped_fraction']}.", ""]

    L += ["## Support per class", "", "| terrain | patches | share | median purity |",
          "|---|---|---|---|"]
    for terrain, (n, med) in a["purity_by_class"].items():
        L.append(f"| {terrain} | {n} | {pct(n, a['n_labelled'])} | {med:.3f} |")
    L.append("")
    if a["empty_classes"]:
        L += [f"**{len(a['empty_classes'])} class(es) have no patches at all: "
              f"{', '.join(a['empty_classes'])}.** They still occupy rows in the "
              f"terrain embedding and weights in the terrain-aware loss, and appear "
              f"as empty rows in per-terrain evaluation. Any claim of "
              f"\"{len(a['purity_by_class'])+len(a['empty_classes'])} terrain types\" "
              f"must be qualified accordingly.", ""]
    if a["below_floor"]:
        L += [f"Below the per-class floor of {a['per_class_floor']} patches "
              f"(configs/split.yaml): " +
              ", ".join(f"{k} ({v})" for k, v in a["below_floor"].items()), ""]

    L += ["## How decisive the labels are", "",
          f"- purity (dominant class share of mapped pixels): {describe(a['purity'])}",
          f"- margin over runner-up: {describe(a['margin'])}",
          f"- unmapped fraction (nodata + unmapped classes): "
          f"{describe(a['unmapped_fraction'])}",
          f"- decided by: {a['label_sources']}", ""]

    if a["n_unlabelled"]:
        L += ["### Why patches went unlabelled", ""]
        for reason, n in a["reject_reasons"].items():
            L.append(f"- {n}: {reason}")
        L.append("")

    L += ["## Purity threshold sweep", "",
          "How many WorldCover-voted patches survive at each threshold - the "
          "label-quality/dataset-size trade-off, in numbers.", "",
          "| min purity | patches kept |", "|---|---|"]
    for thr, info in a["purity_sweep"].items():
        L.append(f"| {thr:.2f} | {info['total']} |")
    L.append("")

    L += ["## Mountain (DEM-derived)", "",
          f"- patches labelled Mountain: {a['n_mountain']}",
          f"- land cover underneath them: {a['under_mountain'] or 'n/a'}",
          f"- steep patches that kept their land-cover label "
          f"(precedence={a['thresholds']['mountain'].get('precedence')}): "
          f"{a['steep_but_not_mountain']}",
          f"- DEM coverage: {a['dem']}", ""]
    if a["dem"]["not_run"] == a["n_patches"]:
        L += ["**DEM slope statistics were never computed** (run stage 4a-bis, "
              "`dem_slope_stats.py`), so Mountain cannot be assigned at all.", ""]

    L += ["## Temporal mismatch", ""]
    if a["worldcover_epoch"]:
        L.append(f"Labels come from ESA WorldCover {a['worldcover_epoch']}. "
                 f"{a['n_scenes_dated']}/{a['n_scenes']} source scenes record an "
                 f"acquisition date.")
        gaps = a["temporal_gaps"]
        if gaps:
            vals = list(gaps.values())
            L.append(f"Gap between imagery and label epoch: "
                     f"{describe(vals, '{:.0f}')} years.")
        else:
            L.append("No scene recorded a usable acquisition date, so the gap "
                     "cannot be quantified from the manifest. The known source "
                     "epochs are SpaceNet 2015-2017, CORE3D 2014-2016 and Maxar "
                     "Open Data at the event date - i.e. 4-7 years before the "
                     "label epoch. Land cover that changed in between (urban "
                     "fringe, cropland rotation) is labelled as 2021, not as the "
                     "patch shows it.")
    L.append("")

    if qc_csv and n_qc:
        L += ["## Manual verification sample", "",
              f"`{qc_csv.name}` holds {n_qc} patches, stratified across classes, "
              f"with an empty `human_verdict` column. Inspect them (QGIS, or "
              f"`python tools/view_patch.py <path>`), write `agree` or the correct "
              f"label, and the agreement rate is a human-verified label accuracy "
              f"to quote. Without it, label accuracy is unmeasured - the numbers "
              f"above describe how decisive the automated vote was, which is not "
              f"the same thing as whether it was right.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="stage 4b output (patch_manifest_labeled.json)")
    ap.add_argument("--config", default=Path("configs/terrain_classes.yaml"), type=Path)
    ap.add_argument("--split-config", default=Path("configs/split.yaml"), type=Path)
    ap.add_argument("--report", type=Path, default=None,
                     help="markdown report path (a .json sidecar is written too)")
    ap.add_argument("--qc-sample", type=int, default=20,
                     help="patches per class in the manual-verification CSV (0 = none)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    split_cfg = yaml.safe_load(args.split_config.read_text()) if args.split_config.exists() else {}
    manifest = json.loads(args.manifest.read_text())

    a = analyse(manifest, cfg, split_cfg)

    print(f"labels: {a['n_labelled']}/{a['n_patches']} "
          f"({pct(a['n_labelled'], a['n_patches'])}), "
          f"{a['n_unlabelled']} unlabelled")
    print(f"support: {a['support']}")
    if a["empty_classes"]:
        print(f"EMPTY CLASSES: {a['empty_classes']}")
    print(f"purity:  {describe(a['purity'])}")
    print(f"sources: {a['label_sources']}")
    print(f"DEM:     {a['dem']}")

    qc_csv, n_qc = None, 0
    if args.qc_sample and a["n_labelled"]:
        qc_csv = (args.report.parent if args.report else args.manifest.parent) / "label_qc_sample.csv"
        n_qc = write_qc_sample(manifest, qc_csv, args.qc_sample, args.seed)
        print(f"QC sample: {n_qc} patches -> {qc_csv}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render(a, qc_csv, n_qc), encoding="utf-8")
        json_path = args.report.with_suffix(".json")
        payload = {k: v for k, v in a.items()
                   if k not in ("purity", "margin", "unmapped_fraction")}
        payload["distributions"] = {
            k: {"n": len(a[k]),
                "median": float(np.median(a[k])) if len(a[k]) else None}
            for k in ("purity", "margin", "unmapped_fraction")}
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"report -> {args.report}\n       -> {json_path}")


if __name__ == "__main__":
    main()
