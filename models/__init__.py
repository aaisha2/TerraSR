"""Model registry — maps a config `model.name` to its build_model factory.

Keeps train_baseline.py model-agnostic: switching baselines is a config
change, not a code change (docs/plan §7). Stage 8's terrain-conditioned model
registers here too once built.
"""
import importlib

_REGISTRY = {
    "srcnn": "models.srcnn_baseline",
    "srgan": "models.srgan_baseline",      # SRResNet generator (+ optional GAN)
    "srresnet": "models.srgan_baseline",
    "swinir": "models.swinir_baseline",
}


def build(name: str, cfg=None):
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'; known: {sorted(_REGISTRY)}")
    module = importlib.import_module(_REGISTRY[key])
    return module.build_model(cfg)
