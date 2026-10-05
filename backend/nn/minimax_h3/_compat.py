"""Thin ComfyUI -> Forge shims used by the MiniMax-H3 modules (ported from ComfyUI's comfy/ldm/minimax)"""

import contextlib

import torch

from backend import memory_management
from backend.attention import attention_function
from backend.operations import main_stream_worker, weights_manual_cast
from backend.operations_gguf import dequantize_tensor
from backend.quant_ops import QuantizedTensor, ck  # noqa: F401
from backend.utils import pad_to_patch_size  # noqa: F401


def cast_to(t: torch.Tensor, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    if t is None:
        return None
    if getattr(t, "gguf_cls", None) is not None:
        t = dequantize_tensor(t)
    elif isinstance(t, QuantizedTensor):
        t = t.dequantize()
    return memory_management.cast_to_device(t, device or t.device, dtype or t.dtype)


def cast_to_input(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return cast_to(t, x.dtype, x.device)


def _is_gguf(layer) -> bool:
    return getattr(getattr(layer, "weight", None), "gguf_cls", None) is not None


@contextlib.contextmanager
def cast_bias_weight(layer, x: torch.Tensor, dtype: torch.dtype = None):
    """yields (weight, bias) of ``layer`` cast to the device (and dtype) of ``x``"""
    if layer is None:
        yield None, None
        return
    kwargs = {"weight_fn": dequantize_tensor, "skip_bias_dtype": True} if _is_gguf(layer) else {}
    if dtype is not None:
        weight, bias, signal = weights_manual_cast(layer, None, dtype=dtype, device=x.device, **kwargs)
    else:
        weight, bias, signal = weights_manual_cast(layer, x, **kwargs)
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    with main_stream_worker(weight, bias, signal):
        yield weight, bias


def rms_norm(x: torch.Tensor, weight: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    if weight is None:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps)
    return torch.nn.functional.rms_norm(x, weight.shape, weight=cast_to(weight, x.dtype, x.device), eps=eps)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def linear_input_act(linear, x: torch.Tensor, input_act: str, act_weight: torch.Tensor = None, act_eps: float = 0.0, residual: torch.Tensor = None, residual_scale: torch.Tensor = None) -> torch.Tensor:
    """``linear(act(x))`` (optionally ``residual + residual_scale * linear(act(x))``); eager version of ComfyUI's fused op"""
    if input_act == "rms_norm":
        x = rms_norm(x, act_weight, act_eps)
    elif input_act == "swiglu":
        x = swiglu(x)
    elif input_act == "gelu_tanh":
        x = torch.nn.functional.gelu(x, approximate="tanh")
    out = linear(x)
    if residual is None:
        return out
    return torch.addcmul(residual, out, residual_scale)


def optimized_attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
    return attention_function(q, k, v, heads, mask=mask, skip_reshape=skip_reshape)


class Ops:
    """resolves ``operations.Linear`` etc. lazily, so the ops patched in by ``using_forge_operations`` are used"""

    def __getattr__(self, name):
        return getattr(torch.nn, name)


ops = Ops()
