"""Blur kernels for the confirmed degradation pipeline (configs/degradation.yaml).

Three kernel families, randomly chosen per patch:
  - isotropic_gaussian:  standard symmetric Gaussian PSF
  - anisotropic_gaussian: rotated, elongated Gaussian (motion/along-track blur)
  - mtf_elliptical:      Gaussian PSF sized from a target Nyquist MTF value,
                         elongated to approximate a satellite sensor's optical
                         + detector MTF (WorldView-class, per Zhu et al. 2020
                         and the MTF-filter degradation literature)

UNITS — the one thing to get right here. The blur is applied on the HR grid,
*before* downsampling, but a sensor MTF is specified at the Nyquist frequency
of the grid it samples onto, i.e. the LR grid. One LR pixel spans
`scale_factor` HR pixels, so a sigma derived from an MTF target is in LR
pixels and must be multiplied by scale_factor before it is used as a kernel
width in HR space. Getting this wrong is a silent factor-of-`scale_factor`
error in the degradation strength (it was: the MTF family used the LR sigma
directly in HR space, so it blurred ~2x too little, while the plain Gaussian
ranges were tuned by eye ~2x too much — see git history / docs).

Every sampler returns (kernel, params) so stage 5 can log exactly what was
applied to each patch; validate_degradation.py audits those logs.
"""
import numpy as np

# Spatial frequency of the sampling grid's Nyquist limit, in cycles/pixel.
NYQUIST_CYCLES_PER_PIXEL = 0.5


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
    """Gaussian sigma, IN PIXELS OF THE SAMPLING (LR) GRID, that produces a
    given MTF at that grid's Nyquist frequency (f = 0.5 cycles/pixel):
        MTF(f) = exp(-2 * (pi * sigma * f)^2)
    Multiply by scale_factor to use it as an HR-space kernel width."""
    nyquist_mtf = np.clip(nyquist_mtf, 1e-3, 0.999)
    f = NYQUIST_CYCLES_PER_PIXEL
    return float(np.sqrt(-np.log(nyquist_mtf) / (2 * (np.pi * f) ** 2)))


def nyquist_mtf_from_sigma(sigma_hr: float, scale_factor: int) -> float:
    """Inverse of sigma_from_nyquist_mtf: the MTF an HR-space Gaussian of
    width `sigma_hr` leaves at the LR grid's Nyquist frequency. Used to state
    the degradation strength of the plain Gaussian families in the same
    sensor-MTF terms the config targets."""
    f_hr = NYQUIST_CYCLES_PER_PIXEL / scale_factor      # LR Nyquist in HR cycles/px
    return float(np.exp(-2 * (np.pi * sigma_hr * f_hr) ** 2))


def mtf_elliptical_kernel(kernel_size: int, nyquist_mtf: float, ellipticity: float,
                           angle_deg: float, scale_factor: int) -> np.ndarray:
    """Approximate a satellite sensor PSF: a Gaussian whose width comes from a
    target MTF at the LR grid's Nyquist frequency, elongated by `ellipticity`
    and rotated to a random along-track angle.

    The elongation is split as (1/sqrt(e), sqrt(e)) about the base sigma, so
    the *geometric mean* width — and therefore the average MTF over
    orientation — stays on the configured target instead of drifting weaker
    as ellipticity grows."""
    base_sigma = scale_factor * sigma_from_nyquist_mtf(nyquist_mtf)
    spread = np.sqrt(max(ellipticity, 1e-6))
    return anisotropic_gaussian_kernel(kernel_size, base_sigma / spread,
                                        base_sigma * spread, angle_deg)


def sample_kernel(cfg: dict, rng: np.random.Generator, scale_factor: int):
    """Sample one blur kernel per the randomized choices in degradation.yaml.

    Returns (kernel, params). `params` records the family and every sampled
    value, plus `effective_nyquist_mtf`: the realized MTF at the LR Nyquist
    frequency, which is the one number comparable across all three families
    and against published sensor MTF."""
    choice = str(rng.choice(cfg["kernel_choices"], p=cfg["kernel_probs"]))
    ksize = cfg["kernel_size"]

    if choice == "isotropic_gaussian":
        lo, hi = cfg["isotropic_sigma_range"]
        sigma = float(rng.uniform(lo, hi))
        kernel = isotropic_gaussian_kernel(ksize, sigma)
        params = {"blur_kernel": choice, "blur_sigma_hr": sigma,
                  "effective_nyquist_mtf": nyquist_mtf_from_sigma(sigma, scale_factor)}
        return kernel, params

    if choice == "anisotropic_gaussian":
        lo, hi = cfg["anisotropic_sigma_range"]
        sigma_x = float(rng.uniform(lo, hi))
        sigma_y = float(rng.uniform(lo, hi))
        alo, ahi = cfg["anisotropic_angle_range"]
        angle = float(rng.uniform(alo, ahi))
        kernel = anisotropic_gaussian_kernel(ksize, sigma_x, sigma_y, angle)
        params = {"blur_kernel": choice, "blur_sigma_x_hr": sigma_x,
                  "blur_sigma_y_hr": sigma_y, "blur_angle_deg": angle,
                  # geometric mean width -> orientation-averaged MTF
                  "effective_nyquist_mtf": nyquist_mtf_from_sigma(
                      float(np.sqrt(sigma_x * sigma_y)), scale_factor)}
        return kernel, params

    if choice == "mtf_elliptical":
        mlo, mhi = cfg["mtf_nyquist_range"]
        elo, ehi = cfg["mtf_ellipticity_range"]
        nyquist_mtf = float(rng.uniform(mlo, mhi))
        ellipticity = float(rng.uniform(elo, ehi))
        angle = float(rng.uniform(0, 180))
        kernel = mtf_elliptical_kernel(ksize, nyquist_mtf, ellipticity, angle,
                                        scale_factor)
        params = {"blur_kernel": choice, "blur_target_nyquist_mtf": nyquist_mtf,
                  "blur_ellipticity": ellipticity, "blur_angle_deg": angle,
                  "blur_sigma_hr": scale_factor * sigma_from_nyquist_mtf(nyquist_mtf),
                  "effective_nyquist_mtf": nyquist_mtf}
        return kernel, params

    raise ValueError(f"unknown kernel choice: {choice}")


def measured_nyquist_mtf(kernel: np.ndarray, scale_factor: int) -> float:
    """Measure, rather than predict, the MTF a kernel leaves at the LR Nyquist
    frequency: the orientation-average of |FFT(kernel)| on the ring
    f = 0.5/scale_factor cycles per HR pixel.

    This makes no Gaussian assumption, so validate_degradation.py can check
    the realized degradation of any kernel — including the elliptical ones,
    where the predicted value is an analytic average."""
    k = np.asarray(kernel, dtype=np.float64)
    k = k / k.sum()
    n = 256                                    # zero-pad for frequency resolution
    pad = np.zeros((n, n))
    h, w = k.shape
    pad[:h, :w] = k
    mtf = np.abs(np.fft.fft2(pad))             # DC at [0,0], normalized by sum=1
    freqs = np.fft.fftfreq(n)                  # cycles per HR pixel

    f_target = NYQUIST_CYCLES_PER_PIXEL / scale_factor
    fy, fx = np.meshgrid(freqs, freqs, indexing="ij")
    radius = np.hypot(fx, fy)
    tol = 1.0 / n                              # one frequency bin either side
    ring = np.abs(radius - f_target) <= tol
    if not ring.any():                         # pathological n/scale combination
        return float("nan")
    return float(mtf[ring].mean())
