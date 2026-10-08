"""Stage 5 validation — does the degradation we actually apply match the
sensor model configs/degradation.yaml claims to implement?

The degradation parameters were originally tuned by eye, and nothing checked
them, so two errors went unnoticed for a long time: the Gaussian blur ranges
were far stronger than the WorldView-class MTF they cited, and the MTF kernel
family had a unit bug that made it ~2x weaker than configured. This script is
the check that would have caught both, and it needs no external reference
data.

Three groups of checks:

  1. KERNEL MTF (config only, no dataset needed)
     Samples kernels straight from degradation.yaml and MEASURES, by FFT, the
     MTF each one leaves at the LR grid's Nyquist frequency. Compares the
     distribution against validation.expected_nyquist_mtf_range and its
     median against validation.expected_median_nyquist_mtf_range (the
     published sensor band). Also confirms the three kernel families agree
     with each other, which is what the unit bug broke.

  2. RECORDED PARAMETERS (needs a stage 5 manifest)
     Summarises what was actually applied across the built dataset: blur
     family mix, realized MTF, noise SNR, downsample methods, JPEG use. This
     only works because degrade() now logs every sampled value per patch.

  3. MEASURED PAIRS (needs the written LR/HR files)
     For a random sample of pairs, re-derives from the images themselves:
       - PSNR of the generated LR against an IDEAL reference LR (pure area
         downsample of the same HR patch). This is the clearest single number
         for "how much of this degradation is the resolution change, and how
         much is synthetic blur and noise piled on top".
       - the noise std left on LR (high-pass residual), vs the same statistic
         on the ideal reference, so added noise is separated from scene texture.
       - the high-frequency share of the radial power spectrum, LR vs ideal.

Usage:
    # config-only (fast, no dataset required)
    python data_pipeline/05_degrade/validate_degradation.py

    # full audit of a built dataset, non-zero exit if a check fails
    python data_pipeline/05_degrade/validate_degradation.py \
        --manifest data/pairs/degradation_manifest.json --strict \
        --report data/pairs/degradation_validation.md
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from degradation_pipeline import ideal_reference_lr  # noqa: E402
from kernels import measured_nyquist_mtf, sample_kernel  # noqa: E402
import patch_io as pio  # noqa: E402

N_KERNEL_SAMPLES = 2000
N_PAIR_SAMPLES = 200


# --------------------------------------------------------------------------
# small image statistics
# --------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse))


def noise_sigma(img: np.ndarray) -> float:
    """Std of a high-pass residual — a noise proxy that is mostly insensitive
    to scene content at these patch sizes."""
    from scipy import ndimage
    return float(np.std(img - ndimage.uniform_filter(img, size=3)))


def high_freq_share(img: np.ndarray) -> float:
    """Fraction of radial power above 2/3 of the Nyquist radius."""
    power = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - w // 2, y - h // 2)
    r_max = r.max()
    total = power.sum() + 1e-12
    return float(power[r > 2 * r_max / 3].sum() / total)


def read_float(path: str) -> np.ndarray:
    """Read a pair image to [0,1] float, 16-bit GeoTIFF or 8-bit PNG alike.
    (The old validator used PIL .convert('L'), which cannot read the 16-bit
    GeoTIFFs this pipeline actually produces.)"""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        arr, _, _ = pio.read_geotiff(path)
        # read the peak off the NATIVE dtype, before casting to float — else
        # every integer raster silently stays in raw DN
        peak = pio.dtype_max(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else 1.0
        return arr.astype(np.float64) / (peak if peak else 1.0)
    return pio.read_png_gray_float(path)


def pair_to_unit(hr: np.ndarray, lr: np.ndarray):
    """Put an HR/LR pair on one common [0,1] scale (the HR patch max), the
    same convention stage 5 and the Dataset use."""
    scale = float(hr.max()) or 1.0
    return np.clip(hr / scale, 0, 1), np.clip(lr / scale, 0, 1)


# --------------------------------------------------------------------------
# check 1 — kernel MTF straight from the config
# --------------------------------------------------------------------------
def check_kernel_mtf(cfg: dict, n=N_KERNEL_SAMPLES) -> dict:
    scale = cfg["scale_factor"]
    rng = np.random.default_rng(0)
    by_family, measured = {}, []
    for _ in range(n):
        kernel, params = sample_kernel(cfg["blur"], rng, scale)
        m = measured_nyquist_mtf(kernel, scale)
        measured.append(m)
        by_family.setdefault(params["blur_kernel"], []).append(
            (params["effective_nyquist_mtf"], m))

    measured = np.array(measured)
    families = {}
    for fam, vals in by_family.items():
        pred = np.array([p for p, _ in vals])
        meas = np.array([m for _, m in vals])
        families[fam] = {
            "n": len(vals),
            "measured_mtf_min": float(meas.min()),
            "measured_mtf_median": float(np.median(meas)),
            "measured_mtf_max": float(meas.max()),
            # predicted vs measured must agree, or the analytic sigma <-> MTF
            # conversion in kernels.py is wrong again
            "max_abs_pred_minus_measured": float(np.abs(pred - meas).max()),
        }
    return {
        "scale_factor": scale,
        "n_samples": int(n),
        "mtf_min": float(measured.min()),
        "mtf_p05": float(np.percentile(measured, 5)),
        "mtf_median": float(np.median(measured)),
        "mtf_p95": float(np.percentile(measured, 95)),
        "mtf_max": float(measured.max()),
        "by_family": families,
    }


# --------------------------------------------------------------------------
# check 2 — what the built dataset actually recorded
# --------------------------------------------------------------------------
def check_recorded_params(rows: list) -> dict:
    def collect(key):
        return np.array([r["degradation_params"][key] for r in rows
                         if r.get("degradation_params", {}).get(key) is not None],
                        dtype=np.float64)

    families = Counter(r.get("degradation_params", {}).get("blur_kernel", "unrecorded")
                       for r in rows)
    methods = Counter(r.get("degradation_params", {}).get("downsample_method", "unrecorded")
                      for r in rows)
    jpeg = Counter("on" if r.get("degradation_params", {}).get("jpeg_quality")
                   else "off" for r in rows)

    out = {"n_pairs": len(rows), "blur_families": dict(families),
           "downsample_methods": dict(methods), "jpeg": dict(jpeg)}

    mtf = collect("effective_nyquist_mtf")
    if mtf.size:
        out["realized_mtf"] = {"min": float(mtf.min()),
                                "median": float(np.median(mtf)),
                                "max": float(mtf.max())}
    snr = collect("noise_snr_at_full_signal")
    snr = snr[np.isfinite(snr)]
    if snr.size:
        out["noise_snr_at_full_signal"] = {"min": float(snr.min()),
                                            "median": float(np.median(snr)),
                                            "max": float(snr.max())}
    if families.get("unrecorded"):
        out["warning"] = (f"{families['unrecorded']} pair(s) carry no blur parameters - "
                          f"they were generated before stage 5 logged them. Re-run "
                          f"stage 5 to regenerate them against the current config.")

    # Which config actually produced this dataset? Stage 5 stamps each pair
    # with a fingerprint of the output-affecting config, so a dataset built
    # before a config change is identifiable instead of merely suspicious.
    prints = Counter(r.get("config_fingerprint") or "unstamped" for r in rows)
    out["config_fingerprints"] = dict(prints)
    return out


# --------------------------------------------------------------------------
# check 3 — measured from the written pairs
# --------------------------------------------------------------------------
def check_measured_pairs(rows: list, scale: int, n=N_PAIR_SAMPLES, seed=0) -> dict:
    usable = [r for r in rows
              if r and Path(r["hr_path"]).exists() and Path(r["lr_path"]).exists()]
    if not usable:
        return {"n_sampled": 0,
                "note": "no readable LR/HR pairs on disk — nothing measured"}

    rng = np.random.default_rng(seed)
    sample = [usable[i] for i in rng.choice(len(usable), size=min(n, len(usable)),
                                             replace=False)]

    vs_ideal, lr_sigma, ideal_sigma, lr_hf, ideal_hf, failed = [], [], [], [], [], 0
    for row in sample:
        try:
            hr, lr = pair_to_unit(read_float(row["hr_path"]), read_float(row["lr_path"]))
            ideal = ideal_reference_lr(hr, scale)
            if ideal.shape != lr.shape:          # odd-sized patch, skip
                failed += 1
                continue
            vs_ideal.append(psnr(lr, ideal))
            lr_sigma.append(noise_sigma(lr))
            ideal_sigma.append(noise_sigma(ideal))
            lr_hf.append(high_freq_share(lr))
            ideal_hf.append(high_freq_share(ideal))
        except Exception:
            failed += 1

    if not vs_ideal:
        return {"n_sampled": 0, "n_failed": failed,
                "note": "every sampled pair failed to read"}

    def stats(v):
        v = np.array(v)
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max()), "mean": float(v.mean())}

    return {
        "n_sampled": len(vs_ideal),
        "n_failed": failed,
        "lr_vs_ideal_psnr_db": stats(vs_ideal),
        "lr_noise_sigma": stats(lr_sigma),
        "ideal_lr_noise_sigma": stats(ideal_sigma),
        # how much of the LR's high-frequency content the extra blur removed
        # relative to an ideal resample of the same scene
        "high_freq_share_lr": stats(lr_hf),
        "high_freq_share_ideal_lr": stats(ideal_hf),
        "high_freq_retained_vs_ideal": float(np.mean(lr_hf) / (np.mean(ideal_hf) + 1e-12)),
    }


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------
def evaluate(cfg: dict, kernel: dict, measured: dict | None,
              recorded: dict | None = None, expected_fingerprint: str = None) -> list:
    """Returns a list of (name, passed, detail) verdicts."""
    v = cfg.get("validation") or {}
    out = []

    # Check this FIRST: if the dataset was not built by the config being
    # validated, every measurement below describes a different dataset than
    # the one the config describes, and that is the finding.
    if recorded and expected_fingerprint:
        prints = recorded.get("config_fingerprints") or {}
        matching = prints.get(expected_fingerprint, 0)
        total = recorded.get("n_pairs", 0)
        others = {k: n for k, n in prints.items() if k != expected_fingerprint}
        out.append(("dataset was built by this config", not others,
                    (f"all {total} pair(s) carry the current fingerprint "
                     f"{expected_fingerprint}")
                    if not others else
                    (f"{total - matching}/{total} pair(s) were built with a "
                     f"different degradation config ({others}); re-run stage 5 "
                     f"- it regenerates pairs whose fingerprint does not match")))

    lo, hi = v.get("expected_nyquist_mtf_range", [0.0, 1.0])
    ok = lo <= kernel["mtf_p05"] and kernel["mtf_p95"] <= hi
    out.append(("kernel MTF range", ok,
                f"5th-95th pct {kernel['mtf_p05']:.3f}-{kernel['mtf_p95']:.3f} "
                f"(allowed {lo}-{hi}); full span "
                f"{kernel['mtf_min']:.3f}-{kernel['mtf_max']:.3f}"))

    mlo, mhi = v.get("expected_median_nyquist_mtf_range", [0.0, 1.0])
    med = kernel["mtf_median"]
    out.append(("median MTF in sensor band", mlo <= med <= mhi,
                f"median {med:.3f} (sensor band {mlo}-{mhi})"))

    worst = max(f["max_abs_pred_minus_measured"] for f in kernel["by_family"].values())
    out.append(("kernel families self-consistent", worst < 0.05,
                f"max |predicted - measured| MTF across families {worst:.4f} "
                f"(must be < 0.05; a large value means the sigma <-> MTF "
                f"conversion in kernels.py is wrong)"))

    fam_medians = {k: f["measured_mtf_median"] for k, f in kernel["by_family"].items()}
    if len(fam_medians) > 1:
        spread = max(fam_medians.values()) - min(fam_medians.values())
        out.append(("kernel families agree with each other", spread < 0.20,
                    "median MTF per family " +
                    ", ".join(f"{k}={val:.3f}" for k, val in sorted(fam_medians.items()))
                    + f" (spread {spread:.3f}, must be < 0.20)"))

    if measured and measured.get("n_sampled"):
        floor = v.get("min_lr_vs_ideal_psnr_db")
        if floor is not None:
            got = measured["lr_vs_ideal_psnr_db"]["median"]
            out.append(("LR vs ideal-LR PSNR", got >= floor,
                        f"median {got:.2f} dB (floor {floor} dB) - below the floor "
                        f"means synthetic blur/noise dominates the resolution change"))
        cap = v.get("max_lr_noise_sigma")
        if cap is not None:
            got = measured["lr_noise_sigma"]["median"]
            ref = measured["ideal_lr_noise_sigma"]["median"]
            out.append(("LR noise level", got <= cap,
                        f"median high-pass sigma {got:.4f} (cap {cap}); "
                        f"ideal reference LR of the same scenes {ref:.4f}"))
        # Scene-independent form of the same check: how much noise did we add
        # relative to the scene's own high-frequency content? Travels across
        # datasets, where the absolute cap above does not.
        ratio_cap = v.get("max_lr_noise_sigma_ratio_vs_ideal")
        if ratio_cap is not None:
            got = measured["lr_noise_sigma"]["median"]
            ref = measured["ideal_lr_noise_sigma"]["median"]
            ratio = got / ref if ref else float("inf")
            out.append(("LR noise vs scene texture", ratio <= ratio_cap,
                        f"LR high-pass sigma is {ratio:.2f}x the ideal reference LR's "
                        f"(cap {ratio_cap}x) - a large ratio means the LR is dominated "
                        f"by synthetic noise, not by the scene"))
    return out


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def render_markdown(cfg_path, kernel, recorded, measured, verdicts) -> str:
    L = ["# Stage 5 degradation validation", "",
         f"Config: `{cfg_path}`", ""]

    L += ["## Verdicts", "", "| check | result | detail |", "|---|---|---|"]
    for name, ok, detail in verdicts:
        L.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    L.append("")

    L += ["## 1. Kernel MTF at the LR Nyquist frequency (measured by FFT)", "",
          f"{kernel['n_samples']} kernels sampled from the config, scale factor "
          f"{kernel['scale_factor']}.", "",
          f"- min {kernel['mtf_min']:.3f}, 5th {kernel['mtf_p05']:.3f}, "
          f"median {kernel['mtf_median']:.3f}, 95th {kernel['mtf_p95']:.3f}, "
          f"max {kernel['mtf_max']:.3f}", "",
          "| family | n | measured MTF min / median / max | max |pred-meas| |",
          "|---|---|---|---|"]
    for fam, f in sorted(kernel["by_family"].items()):
        L.append(f"| {fam} | {f['n']} | {f['measured_mtf_min']:.3f} / "
                 f"{f['measured_mtf_median']:.3f} / {f['measured_mtf_max']:.3f} | "
                 f"{f['max_abs_pred_minus_measured']:.4f} |")
    L.append("")

    if recorded:
        L += ["## 2. Parameters recorded for the built dataset", "",
              f"- pairs: {recorded['n_pairs']}",
              f"- blur families: {recorded['blur_families']}",
              f"- downsample methods: {recorded['downsample_methods']}",
              f"- JPEG: {recorded['jpeg']}"]
        if "realized_mtf" in recorded:
            r = recorded["realized_mtf"]
            L.append(f"- realized MTF: min {r['min']:.3f}, median {r['median']:.3f}, "
                     f"max {r['max']:.3f}")
        if "noise_snr_at_full_signal" in recorded:
            s = recorded["noise_snr_at_full_signal"]
            L.append(f"- noise SNR at full signal: min {s['min']:.1f}, "
                     f"median {s['median']:.1f}, max {s['max']:.1f}")
        if "warning" in recorded:
            L.append(f"- **{recorded['warning']}**")
        L.append("")

    if measured and measured.get("n_sampled"):
        m = measured
        L += ["## 3. Measured from the written LR/HR pairs", "",
              f"{m['n_sampled']} pairs sampled"
              + (f" ({m['n_failed']} unreadable/skipped)" if m.get("n_failed") else ""),
              "",
              "| statistic | min | median | max |", "|---|---|---|---|"]
        for key, label in (("lr_vs_ideal_psnr_db", "PSNR: LR vs ideal area-downsample LR (dB)"),
                            ("lr_noise_sigma", "LR high-pass noise sigma"),
                            ("ideal_lr_noise_sigma", "ideal LR high-pass noise sigma"),
                            ("high_freq_share_lr", "LR high-frequency power share"),
                            ("high_freq_share_ideal_lr", "ideal LR high-frequency power share")):
            s = m[key]
            L.append(f"| {label} | {s['min']:.4f} | {s['median']:.4f} | {s['max']:.4f} |")
        L += ["",
              f"High-frequency content retained relative to an ideal resample: "
              f"{100*m['high_freq_retained_vs_ideal']:.1f}%.", ""]
    elif measured:
        L += ["## 3. Measured from the written LR/HR pairs", "",
              measured.get("note", "not run"), ""]

    L += ["## How to read this", "",
          "The LR/HR pairs define the task. If the degradation is stronger than the",
          "sensor model it claims, the models are being trained to undo synthetic",
          "blur rather than to super-resolve, and every PSNR gain over bicubic is",
          "inflated by that. These checks exist to keep the task honest, and to",
          "catch a repeat of the two bugs that motivated them: Gaussian blur ranges",
          "far outside the cited MTF band, and an MTF kernel family off by a factor",
          "of the scale factor through a pixel-grid unit error.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path("configs/degradation.yaml"), type=Path)
    ap.add_argument("--manifest", type=Path, default=None,
                     help="stage 5 degradation_manifest.json (enables checks 2 and 3)")
    ap.add_argument("--report", type=Path, default=None,
                     help="write a markdown report here (plus a .json sidecar)")
    ap.add_argument("--pair-samples", type=int, default=N_PAIR_SAMPLES)
    ap.add_argument("--strict", action="store_true",
                     help="exit non-zero if any check fails")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    kernel = check_kernel_mtf(cfg)

    recorded = measured = None
    if args.manifest:
        rows = [r for r in json.loads(args.manifest.read_text()) if r]
        recorded = check_recorded_params(rows)
        measured = check_measured_pairs(rows, cfg["scale_factor"], n=args.pair_samples)

    # the fingerprint of the config we are validating against, so a dataset
    # built by a different one is reported as such rather than just failing
    sys.path.insert(0, str(Path(__file__).parent))
    from make_lr_hr_pairs import degradation_fingerprint  # noqa: E402
    expected = degradation_fingerprint(cfg)
    verdicts = evaluate(cfg, kernel, measured, recorded, expected)

    print(f"{'check':<42}{'result':<8}detail")
    print("-" * 100)
    for name, ok, detail in verdicts:
        print(f"{name:<42}{'PASS' if ok else 'FAIL':<8}{detail}")
    n_failed = sum(1 for _, ok, _ in verdicts if not ok)
    print(f"\n{len(verdicts) - n_failed}/{len(verdicts)} checks passed")

    if recorded:
        print(f"\nrecorded over {recorded['n_pairs']} pairs: "
              f"blur {recorded['blur_families']}, jpeg {recorded['jpeg']}")
        print(f"  config fingerprints in the dataset: "
              f"{recorded.get('config_fingerprints')}   "
              f"(current config: {expected})")
        if "warning" in recorded:
            print(f"  WARNING: {recorded['warning']}")
    if measured and measured.get("n_sampled"):
        print(f"measured on {measured['n_sampled']} pairs: LR vs ideal-LR PSNR median "
              f"{measured['lr_vs_ideal_psnr_db']['median']:.2f} dB, LR noise sigma "
              f"{measured['lr_noise_sigma']['median']:.4f}, high-frequency retained "
              f"{100*measured['high_freq_retained_vs_ideal']:.1f}%")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(args.config, kernel, recorded,
                                                measured, verdicts), encoding="utf-8")
        payload = {"config": str(args.config), "kernel_mtf": kernel,
                   "recorded_params": recorded, "measured_pairs": measured,
                   "verdicts": [{"check": n, "passed": ok, "detail": d}
                                 for n, ok, d in verdicts]}
        json_path = args.report.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}\n        -> {json_path}")

    if n_failed and args.strict:
        raise SystemExit(f"{n_failed} degradation validation check(s) failed")


if __name__ == "__main__":
    main()
