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


def apply_blur(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    kernel = sample_kernel(cfg["blur"], rng)
    return cv2.filter2D(img, ddepth=-1, kernel=kernel.astype(np.float32),
                         borderType=cv2.BORDER_REFLECT)


def apply_downsample(img: np.ndarray, scale_factor: int, cfg: dict,
                      rng: np.random.Generator) -> np.ndarray:
    ds_cfg = cfg["downsample"]
    method = rng.choice(ds_cfg["method_choices"], p=ds_cfg["method_probs"])
    h, w = img.shape[:2]
    new_h, new_w = h // scale_factor, w // scale_factor
    return cv2.resize(img, (new_w, new_h), interpolation=_CV2_INTERP[method]), method


def apply_noise(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    noise_cfg = cfg["noise"]
    plo, phi = noise_cfg["poisson_scale_range"]
    glo, ghi = noise_cfg["gaussian_sigma_range"]
    poisson_scale = rng.uniform(plo, phi)
    gaussian_sigma = rng.uniform(glo, ghi)

    out = img.copy()
    if poisson_scale > 0:
        # signal-dependent shot noise: scale up, sample Poisson, scale back
        peak = 1.0 / max(poisson_scale, 1e-6)
        out = rng.poisson(np.clip(out, 0, 1) * peak).astype(np.float64) / peak
    if gaussian_sigma > 0:
        out = out + rng.normal(0, gaussian_sigma, size=out.shape)
    return np.clip(out, 0, 1)


def apply_jpeg(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    comp_cfg = cfg["compression"]
    if not comp_cfg.get("enabled", True):
        return img
    qlo, qhi = comp_cfg["quality_range"]
    quality = int(rng.integers(qlo, qhi + 1))
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if img.ndim == 2 else cv2.IMREAD_COLOR)
    return decoded.astype(np.float64) / 255.0


def degrade(hr_image: np.ndarray, cfg: dict, rng: np.random.Generator | None = None) -> dict:
    """Run the full single-order pipeline on one HR patch.

    hr_image: float array in [0, 1], shape (H, W) for PAN/pseudo-PAN.
    Returns dict with the LR image and the sampled parameters (for the
    per-patch reproducibility log required by the plan).
    """
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed"))

    scale_factor = cfg["scale_factor"]

    blurred = apply_blur(hr_image, cfg, rng)
    downsampled, ds_method = apply_downsample(blurred, scale_factor, cfg, rng)
    noised = apply_noise(downsampled, cfg, rng)
    compressed = apply_jpeg(noised, cfg, rng)

    return {
        "lr_image": compressed,
        "params": {
            "scale_factor": scale_factor,
            "downsample_method": ds_method,
        },
    }
