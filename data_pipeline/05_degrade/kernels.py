"""Blur kernels for the confirmed degradation pipeline (configs/degradation.yaml).

Three kernel families, randomly chosen per patch:
  - isotropic_gaussian:  standard symmetric Gaussian PSF
  - anisotropic_gaussian: rotated, elongated Gaussian (motion/along-track blur)
  - mtf_elliptical:      Gaussian PSF sized from a target Nyquist MTF value,
                         elongated to approximate a satellite sensor's optical
                         + detector MTF (WorldView-class, per Zhu et al. 2020
                         and the MTF-filter degradation literature)
"""
import numpy as np


def _grid(kernel_size: int):
    r = kernel_size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    return x, y


def isotropic_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    x, y = _grid(kernel_size)
    k = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def anisotropic_gaussian_kernel(kernel_size: int, sigma_x: float, sigma_y: float,
                                 angle_deg: float) -> np.ndarray:
    x, y = _grid(kernel_size)
    theta = np.deg2rad(angle_deg)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    k = np.exp(-(xr ** 2 / (2 * sigma_x ** 2) + yr ** 2 / (2 * sigma_y ** 2)))
    return k / k.sum()


def sigma_from_nyquist_mtf(nyquist_mtf: float) -> float:
    """Invert the Gaussian MTF formula to get the sigma (in pixels of the
    *downsampled* grid) that produces a given MTF value at Nyquist frequency
    (f = 0.5 cycles/pixel). MTF(f) = exp(-2 * (pi * sigma * f)^2).
    """
    nyquist_mtf = np.clip(nyquist_mtf, 1e-3, 0.999)
    f = 0.5
    return np.sqrt(-np.log(nyquist_mtf) / (2 * (np.pi * f) ** 2))


def mtf_elliptical_kernel(kernel_size: int, nyquist_mtf: float, ellipticity: float,
                           angle_deg: float) -> np.ndarray:
    """Approximate a satellite sensor PSF: a Gaussian whose width is derived
    from a target Nyquist-frequency MTF value, elongated by `ellipticity`
    along one axis and rotated to a random along-track angle."""
    base_sigma = sigma_from_nyquist_mtf(nyquist_mtf)
    sigma_x = base_sigma
    sigma_y = base_sigma * ellipticity
    return anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, angle_deg)


def sample_kernel(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Sample one blur kernel per the randomized choices in degradation.yaml."""
    choice = rng.choice(cfg["kernel_choices"], p=cfg["kernel_probs"])
    ksize = cfg["kernel_size"]

    if choice == "isotropic_gaussian":
        lo, hi = cfg["isotropic_sigma_range"]
        sigma = rng.uniform(lo, hi)
        return isotropic_gaussian_kernel(ksize, sigma)

    if choice == "anisotropic_gaussian":
        lo, hi = cfg["anisotropic_sigma_range"]
        sigma_x = rng.uniform(lo, hi)
        sigma_y = rng.uniform(lo, hi)
        alo, ahi = cfg["anisotropic_angle_range"]
        angle = rng.uniform(alo, ahi)
        return anisotropic_gaussian_kernel(ksize, sigma_x, sigma_y, angle)

    if choice == "mtf_elliptical":
        mlo, mhi = cfg["mtf_nyquist_range"]
        elo, ehi = cfg["mtf_ellipticity_range"]
        nyquist_mtf = rng.uniform(mlo, mhi)
        ellipticity = rng.uniform(elo, ehi)
        angle = rng.uniform(0, 180)
        return mtf_elliptical_kernel(ksize, nyquist_mtf, ellipticity, angle)

    raise ValueError(f"unknown kernel choice: {choice}")
