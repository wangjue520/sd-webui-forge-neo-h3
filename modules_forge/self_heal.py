"""
Self-healing for numerical overflow (black / garbled "ghost" images)

- sampling: every denoising step is checked for NaN / Inf; on failure the image is re-sampled with fp32 attention,
  then with a freshly reloaded model, instead of silently zeroing the latent (which produced black images)
- VAE: a non-finite decode is retried in fp32 (Forge already falls back to tiled decoding on OOM)
Each intervention is logged and recorded in the generation parameters ("Self-heal").
"""

import logging

import torch

from backend.logging import setup_logger

logger = logging.getLogger("self_heal")
setup_logger(logger)


class NonFiniteError(Exception):
    def __init__(self, where: str):
        super().__init__(f"NaN / Inf in {where}")
        self.where = where


def enabled() -> bool:
    from modules.shared import opts

    return bool(getattr(opts, "forge_self_heal", True))


def is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all())


def check(x: torch.Tensor, where: str):
    """raise NonFiniteError if x holds NaN / Inf (no-op when self-heal is disabled)"""
    if enabled() and x is not None and not is_finite(x):
        raise NonFiniteError(where)


def _note(p, message: str):
    logger.warning(f"[Self-heal] {message}")
    if p is not None:
        notes = p.extra_generation_params.get("Self-heal", "")
        p.extra_generation_params["Self-heal"] = f"{notes}; {message}" if notes else message


# region Sampling


def sample(p, sample_fn, reset_rng=None):
    """
    run `sample_fn()`; on NaN / Inf retry with
      1. fp32 attention (the usual fp16 overflow)
      2. fp32 attention + a freshly reloaded model (weights corrupted / drifted during the session)
    `reset_rng()` recreates the noise generator so a retry uses the same seed
    """
    from backend import args as backend_args

    if not enabled():
        return sample_fn()

    attempts = [None, "fp32 attention", "reloaded model + fp32 attention"]
    original_upcast = backend_args.args.force_upcast_attention
    try:
        for i, remedy in enumerate(attempts):
            try:
                if remedy is not None:
                    backend_args.args.force_upcast_attention = True
                    if remedy.startswith("reloaded"):
                        _reload_model(p)
                    if reset_rng is not None:
                        reset_rng()
                return sample_fn()
            except NonFiniteError as e:
                if i == len(attempts) - 1:
                    _note(p, f"{e} persisted after all retries")
                    raise RuntimeError("Sampling kept producing NaN / Inf even with fp32 attention and a reloaded model. Try another sampler / lower CFG, or disable SageAttention (--disable-sage).") from e
                _note(p, f"{e}; retrying with {attempts[i + 1]}")
    finally:
        backend_args.args.force_upcast_attention = original_upcast


def _reload_model(p):
    from backend import memory_management
    from modules import sd_models

    sd_models.model_data.forge_hash = ""  # force forge_model_reload to rebuild the model from disk
    memory_management.unload_all_models()
    memory_management.soft_empty_cache(force=True)
    sd_models.forge_model_reload()
    p.sd_model = sd_models.model_data.sd_model
    p.sd_model.forge_objects = p.sd_model.forge_objects_after_applying_lora.shallow_copy()


# region VAE


def decode(p, model, batch: torch.Tensor, decode_fn) -> torch.Tensor:
    """`decode_fn(model, batch)`; if the result is not finite, retry with the VAE in fp32"""
    out = decode_fn(model, batch)
    if not enabled() or is_finite(out):
        return out

    vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    if vae is None or not hasattr(vae, "vae_dtype"):
        _note(p, "NaN / Inf in VAE decode (no fp32 fallback available)")
        return out

    original_dtype = vae.vae_dtype
    try:
        vae.vae_dtype = torch.float32
        vae.first_stage_model.to(torch.float32)
        _note(p, f"NaN / Inf in VAE decode ({original_dtype}); retried in fp32")
        out = decode_fn(model, batch)
        if not is_finite(out):
            _note(p, "fp32 VAE decode still not finite (the latent itself is bad); replacing NaN / Inf")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    finally:
        vae.vae_dtype = original_dtype
        vae.first_stage_model.to(original_dtype)
    return out
