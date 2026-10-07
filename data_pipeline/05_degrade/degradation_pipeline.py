"""Single-order degradation pipeline: blur -> downsample -> noise -> jpeg.

Confirmed with supervisor 2026-07-11: one randomized pass is sufficient for
a fixed, known 2x SR factor on a controlled sensor set (SpaceNet/Maxar) —
no need for Real-ESRGAN's second-order (double-pass) degradation, which
targets blind unconstrained real-world SR at aggressive scale factors.

Operates on single-channel (PAN / pseudo-PAN) float images in [0, 1].
"""
import cv2
import numpy as np

from kernels import sample_kernel

_CV2_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
}


def apply_blur(img: np.ndarray, cfg: dict, rng: np.random.Generator,
                scale_factor: int):
    kernel, params = sample_kernel(cfg["blur"], rng, scale_factor)
    out = cv2.filter2D(img, ddepth=-1, kernel=kernel.astype(np.float32),
                        borderType=cv2.BORDER_REFLECT)
    return out, params


def apply_downsample(img: np.ndarray, scale_factor: int, cfg: dict,
                      rng: np.random.Generator):
    ds_cfg = cfg["downsample"]
    method = str(rng.choice(ds_cfg["method_choices"], p=ds_cfg["method_probs"]))
    h, w = img.shape[:2]
    new_h, new_w = h // scale_factor, w // scale_factor
    out = cv2.resize(img, (new_w, new_h), interpolation=_CV2_INTERP[method])
    return out, {"downsample_method": method}


def apply_noise(img: np.ndarray, cfg: dict, rng: np.random.Generator):
    noise_cfg = cfg["noise"]
    plo, phi = noise_cfg["poisson_scale_range"]
    glo, ghi = noise_cfg["gaussian_sigma_range"]
    poisson_scale = float(rng.uniform(plo, phi))
    gaussian_sigma = float(rng.uniform(glo, ghi))

    out = img.copy()
    if poisson_scale > 0:
        # signal-dependent shot noise: scale up, sample Poisson, scale back.
        # poisson_scale is 1/peak, so shot SNR at full signal = 1/sqrt(scale).
        peak = 1.0 / max(poisson_scale, 1e-12)
        out = rng.poisson(np.clip(out, 0, 1) * peak).astype(np.float64) / peak
    if gaussian_sigma > 0:
        out = out + rng.normal(0, gaussian_sigma, size=out.shape)
    params = {
        "noise_poisson_scale": poisson_scale,
        "noise_gaussian_sigma": gaussian_sigma,
        # the physically meaningful summary: combined SNR at full signal
        "noise_snr_at_full_signal": float(
            1.0 / np.sqrt(poisson_scale + gaussian_sigma ** 2))
        if (poisson_scale > 0 or gaussian_sigma > 0) else float("inf"),
    }
    return np.clip(out, 0, 1), params


def apply_jpeg(img: np.ndarray, cfg: dict, rng: np.random.Generator):
    comp_cfg = cfg["compression"]
    if not comp_cfg.get("enabled", True):
        return img, {"jpeg_quality": None}
    qlo, qhi = comp_cfg["quality_range"]
    quality = int(rng.integers(qlo, qhi + 1))
    # NOTE: this necessarily quantises to 8 bits and back, discarding the
    # 16-bit depth of the PAN source. Off by default — see the comment on
    # `compression` in configs/degradation.yaml.
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img, {"jpeg_quality": None}
    decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR)
    return decoded.astype(np.float64) / 255.0, {"jpeg_quality": quality,
                                                 "jpeg_quantised_to_8bit": True}


def ideal_reference_lr(hr_image: np.ndarray, scale_factor: int) -> np.ndarray:
    """The LR you would get from the resolution change ALONE: an area
    downsample, no blur, no noise, no compression.

    validate_degradation.py measures the generated LR against this to put a
    number on how much of the degradation is the 2x resolution change and how
    much is synthetic blur/noise piled on top."""
    h, w = hr_image.shape[:2]
    return cv2.resize(hr_image, (w // scale_factor, h // scale_factor),
                       interpolation=cv2.INTER_AREA)


def degrade(hr_image: np.ndarray, cfg: dict, rng: np.random.Generator | None = None) -> dict:
    """Run the full single-order pipeline on one HR patch.

    hr_image: float array in [0, 1], shape (H, W) for PAN/pseudo-PAN.
    Returns dict with the LR image and EVERY sampled parameter — the blur
    family and its widths, the realized MTF at the LR Nyquist frequency, the
    noise levels and SNR, the downsample method and the JPEG quality. Stage 5
    writes these to its manifest, which is both the per-patch reproducibility
    log the plan requires and the input validate_degradation.py audits.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed"))

    scale_factor = cfg["scale_factor"]
    params = {"scale_factor": scale_factor}

    blurred, blur_params = apply_blur(hr_image, cfg, rng, scale_factor)
    downsampled, ds_params = apply_downsample(blurred, scale_factor, cfg, rng)
    noised, noise_params = apply_noise(downsampled, cfg, rng)
    compressed, jpeg_params = apply_jpeg(noised, cfg, rng)

    params.update(blur_params)
    params.update(ds_params)
    params.update(noise_params)
    params.update(jpeg_params)

    return {"lr_image": compressed, "params": params}
