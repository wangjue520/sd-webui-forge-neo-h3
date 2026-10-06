"""
MiniMax-H3 (33B joint audio-video DiT)
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py
https://github.com/huggingface/diffusers/tree/main/src/diffusers/modular_pipelines/minimax_h3

The checkpoint is guidance-distilled: one forward per step, no unconditional branch.
Text-to-Video / FirstLastFrame-to-Video use the FL2VA checkpoint; Reference-to-Video uses the Ref2VA checkpoint.
"""

import json
import logging
import math
import os
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F
from transformers.modeling_utils import no_init_weights

from backend import memory_management, utils
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.logging import setup_logger
from backend.operations import using_forge_operations
from backend.patcher.base import ModelPatcher
from backend.state_dict import convert_quantization, detect_quantization, load_state_dict, state_dict_prefix_replace

logger = logging.getLogger("minimax_h3")
setup_logger(logger)

TOKENIZER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "huggingface", "krea", "Krea-2-Raw", "tokenizer")

FPS = 24
AUDIO_LATENT_FPS = 40
AUDIO_SAMPLE_RATE = 32000
CANVAS_MULTIPLE = 32
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0
REF_IMAGE_SHORT_EDGE = 2048


# region Helpers


def align_frame_count(n: int) -> int:
    """snap up to 17k + 5"""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def snap(v: int, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(v / multiple)) * multiple)


def time_shift(base: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * base / (1 + (shift - 1) * base)


def resize_image(image: torch.Tensor, width: int, height: int, crop: bool) -> torch.Tensor:
    """[H, W, C] in [0, 1] -> [H', W', C]; stretch, or aspect-preserving cover-crop"""
    x = image[..., :3].movedim(-1, 0).unsqueeze(0).float()
    if crop:
        h, w = x.shape[-2:]
        scale = max(width / w, height / h)
        nh, nw = max(height, round(h * scale)), max(width, round(w * scale))
        x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)
        top, left = (nh - height) // 2, (nw - width) // 2
        x = x[..., top : top + height, left : left + width]
    else:
        x = F.interpolate(x, size=(height, width), mode="bicubic", antialias=True, align_corners=False)
    return x.clamp(0.0, 1.0)[0].movedim(0, -1)


def peek_keys(path: str) -> set[str]:
    path = str(path)
    if path.lower().endswith(".gguf"):
        from modules_forge.packages import gguf

        return {str(t.name) for t in gguf.GGUFReader(path).tensors}
    if path.lower().endswith((".safetensors", ".sft")):
        import safetensors

        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return set(f.keys())
    return set()


def _strip(keys: set[str], prefixes: tuple[str, ...]) -> set[str]:
    out = set()
    for k in keys:
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
                break
        out.add(k)
    return out


DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")


def is_minimax_h3(path: str) -> bool:
    try:
        keys = _strip(peek_keys(path), DIT_PREFIXES)
    except Exception:
        return False
    return "video_patch_proj.weight" in keys and "audio_patch_proj.weight" in keys


def classify_module(path: str) -> str | None:
    keys = peek_keys(path)
    if any(k.endswith("decoder.transformer_blocks.0.scale1") for k in keys):
        return "vae"
    if any(k.endswith("pre_block.attn.zero_k_bias") for k in keys):
        return "audio_vae"
    if any(("layers.49." in k or "blk.49." in k) for k in keys):
        return "text_encoder"
    if any(k.startswith(("v.blk.", "visual.blocks.", "model.visual.blocks.")) for k in keys):
        return "vision"  # mmproj of a GGUF text encoder
    return None


def _load_sd(path: str) -> tuple[dict, dict]:
    sd, metadata = utils.load_torch_file(path, return_metadata=True)
    _materialize(sd)
    sd, metadata = convert_quantization(sd, metadata)
    return sd, metadata or {}


def _materialize(sd: dict):
    """
    copy memory-mapped weights into process RAM: H3's files total >30 GB, so Windows evicts their file-cache pages
    between generations and every reload of the text encoder / DiT re-reads the disk at ~1 GB/s (20+ s per generation)
    instead of copying RAM -> VRAM at ~9 GB/s
    """
    import psutil

    # only GGUF (numpy memmap); safetensors / mixed-precision files are left to Forge, and only with RAM to spare
    tensors = [v for v in sd.values() if isinstance(v, torch.Tensor) and v.device.type == "cpu" and getattr(v, "gguf_cls", None) is not None]
    size = sum(v.numel() * v.element_size() for v in tensors)
    available = psutil.virtual_memory().available
    if size == 0:
        return
    if available < size * 1.5 + 8 * 2**30:
        logger.warning(f"not enough free RAM to cache {size / 2**30:.1f} GB of GGUF weights ({available / 2**30:.1f} GB free); reloads will read the disk")
        return
    for v in tensors:
        v.data = v.data.clone()


def _storage(sd: dict, default: torch.dtype, is_unet: bool = False):
    quant_config = detect_quantization(sd, is_unet=is_unet)
    sd_dtype = utils.weight_dtype(sd)
    if quant_config is not None:
        return sd_dtype, quant_config
    if sd_dtype in (torch.float8_e4m3fn, torch.float8_e5m2, "gguf"):
        if sd_dtype == "gguf":
            utils.beautiful_print_gguf_state_dict_statics(sd)
        return sd_dtype, None
    return default, None


def _build(model_fn: Callable, sd: dict, storage_dtype, quant_config, compute_dtype: torch.dtype, device: torch.device, name: str):
    sd_dtype = utils.weight_dtype(sd)
    logger.info(f"[{name}] storage: {storage_dtype if quant_config is None else 'MixedPrecision'} | compute: {compute_dtype}")
    with no_init_weights():
        if storage_dtype == "gguf":
            with using_forge_operations(device=device, dtype=compute_dtype, manual_cast_enabled=True, extra_dtype="gguf"):
                model = model_fn()
        else:
            dtype = torch.bfloat16 if quant_config is not None else storage_dtype
            # always cast to the input dtype: the H3 modules mix fp32 islands with the compute dtype
            with using_forge_operations(device=device, dtype=dtype, manual_cast_enabled=True, sd_dtype=sd_dtype, extra_dtype=quant_config):
                model = model_fn()
    _fix_shapes(model, sd)
    load_state_dict(model, sd, log_name=name)
    return model


def _fix_shapes(model: torch.nn.Module, sd: dict):
    """GGUF stores at most 4 dims (e.g. the Qwen3-VL Conv3d patch embedding [1152, 3, 2, 16, 16] -> [3456, 2, 16, 16])"""
    from backend.loader_gguf import dequantize

    expected = {k: v.shape for k, v in model.state_dict().items()}
    for k, v in list(sd.items()):
        shape = expected.get(k, None)
        if shape is None or tuple(v.shape) == tuple(shape) or math.prod(v.shape) != math.prod(shape):
            continue
        t = dequantize(v, torch.bfloat16) if getattr(v, "gguf_cls", None) is not None else v.detach().as_subclass(torch.Tensor)
        sd[k] = t.to(torch.bfloat16).reshape(shape).clone()
        logger.info(f"reshaped {k}: {tuple(v.shape)} -> {tuple(shape)}")


# region Loading


def load_transformer(path: str):
    from backend.nn.minimax_h3.model import MiniMaxH3Model

    sd, metadata = _load_sd(path)
    for p in DIT_PREFIXES:
        if any(k.startswith(p) for k in sd):
            sd = state_dict_prefix_replace(sd, {p: ""})
            break

    config = {
        "num_layers": len({k.split(".")[1] for k in sd if k.startswith("blocks.")}),
        "token_refiner_num_layers": len({k.split(".")[2] for k in sd if k.startswith("token_refiner.blocks.")}),
        "hidden_size": sd["video_patch_proj.weight"].shape[0],
        "attention_head_dim": sd["blocks.0.attn.q_norm.weight"].shape[0],
        "text_dim": sd["condition_proj.weight"].shape[1],
        "rope_inv_freq_len": sd["rope.inv_freq"].shape[0],
        "gate_compress": "blocks.0.attn.to_gate_compress.weight" in sd,
        "pdd_heads": max(1, sd["final_layer.audio_out.weight"].shape[0] // 32),
    }
    config["num_attention_heads"] = sd["blocks.0.attn.qkv_proj.weight"].shape[0] // (3 * config["attention_head_dim"])
    config["ffn_hidden_size"] = sd["blocks.0.mlp.fc1.weight"].shape[0] // 2
    if "adaln_t_table" in sd:
        config["adaln_curve_grid"], config["time_embed_dim"] = sd["adaln_t_table"].shape
    else:
        config["timestep_input_dim"] = sd["time_embedder.proj_in.weight"].shape[1]
        config["time_embed_hidden_size"] = sd["time_embedder.proj_in.weight"].shape[0]
        config["time_embed_dim"] = sd["time_embedder.proj_out.weight"].shape[0]
    if "config" in metadata:
        config.update(json.loads(metadata["config"]).get("transformer", {}))
    logger.info(f"[Transformer] {config}")

    load_device = memory_management.get_torch_device()
    params = utils.calculate_parameters(sd)
    default = memory_management.unet_dtype(device=load_device, model_params=params, supported_dtypes=[torch.bfloat16, torch.float32], weight_dtype=utils.weight_dtype(sd))
    from backend.args import dynamic_args

    if dynamic_args.forge_unet_storage_dtype is not None:
        default = dynamic_args.forge_unet_storage_dtype
    storage_dtype, quant_config = _storage(sd, default, is_unet=True)
    compute_dtype = memory_management.inference_cast(weight_dtype=storage_dtype, inference_device=load_device, supported_dtypes=[torch.bfloat16, torch.float32])
    if storage_dtype == "gguf" or quant_config is not None:
        compute_dtype = torch.bfloat16 if memory_management.should_use_bf16(load_device) else torch.float32

    from backend.nn.minimax_h3._compat import ops

    dynamic_args.ops = None
    model = _build(lambda: MiniMaxH3Model(**config, dtype=None, operations=ops), sd, storage_dtype, quant_config, compute_dtype, memory_management.cpu, "Transformer")
    model.computation_dtype = compute_dtype
    model.storage_dtype = storage_dtype
    model.quantized = storage_dtype == "gguf" or quant_config is not None
    return model


def load_text_encoder(paths: list[str]):
    from backend.loader_gguf import gguf_remapping
    from backend.nn.minimax_h3.text_encoder import MiniMaxQwen3VL

    sd = {}
    for path in paths:
        _sd, _ = _load_sd(path)
        if path.lower().endswith(".gguf"):
            _sd = gguf_remapping(_sd)
        sd.update(_sd)
    sd = state_dict_prefix_replace(sd, {"model.language_model.": "model.", "model.visual.": "visual.", "text_encoders.qwen3vl_32b.transformer.": ""})
    sd = {k: v for k, v in sd.items() if not k.startswith(("lm_head.", "model.norm."))}

    te_dtype = memory_management.text_encoder_dtype()
    storage_dtype, quant_config = _storage(sd, te_dtype)
    model = _build(lambda: MiniMaxQwen3VL(), sd, storage_dtype, quant_config, te_dtype, memory_management.cpu, "TextEncoder")
    model.has_vision = any(k.startswith("visual.blocks.") for k in sd) and any(k.startswith("visual.deepstack_merger_list.") for k in sd)
    model.embed_cache = _embedding_table(sd.get("model.embed_tokens.weight", None))
    if not model.has_vision:
        logger.warning("[TextEncoder] Qwen3-VL vision tower not found; images will only condition the video latents (not the prompt)")
    return model


def _embedding_table(weight) -> torch.Tensor | None:
    """bf16 copy of the 151936 x 5120 token table on the CPU: Forge's GGUF Embedding would dequantize the whole table on the GPU on every call"""
    if weight is None:
        return None
    try:
        if getattr(weight, "gguf_cls", None) is not None:
            from backend.loader_gguf import dequantize

            return dequantize(weight, torch.bfloat16).cpu()
        if weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return weight.detach().to(torch.bfloat16).cpu()
    except Exception as e:
        logger.warning(f"[TextEncoder] embedding cache unavailable: {e}")
    return None


def load_vae(path: str):
    from backend.nn.minimax_h3.vae import MiniMaxH3VideoVAE

    sd, _ = _load_sd(path)
    num_layers = sum(k.startswith("decoder.transformer_blocks.") and k.endswith(".scale1") for k in sd)
    device = memory_management.vae_device()
    dtype = memory_management.vae_dtype(device=device, allowed_dtypes=[torch.float16, torch.float32])
    _, quant_config = _storage(sd, dtype)
    operations = None
    if quant_config is not None:
        from backend.operations_mixed_precision import mixed_precision_ops

        operations = mixed_precision_ops(quant_config=quant_config, compute_dtype=dtype)
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=dtype, manual_cast_enabled=True, sd_dtype=utils.weight_dtype(sd), operations=operations):
            model = MiniMaxH3VideoVAE(num_layers=num_layers, **({"operations": operations} if operations is not None else {}))
    load_state_dict(model, sd, log_name="VAE")
    model.dtype = dtype
    return model


def load_audio_vae(path: str):
    from backend.nn.minimax_h3.audio_vae import MiniMaxH3AudioVAE

    sd, _ = _load_sd(path)
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=torch.float32, manual_cast_enabled=True):
            model = MiniMaxH3AudioVAE()
    load_state_dict(model, sd, log_name="AudioVAE")
    return model


def load_minimax_h3(path: str, additional_state_dicts: list[str] = None) -> "MiniMaxH3":
    modules = {"text_encoder": [], "vision": [], "vae": [], "audio_vae": []}
    for p in additional_state_dicts or []:
        if (kind := classify_module(p)) is None:
            logger.warning(f'Ignoring unrecognized module "{os.path.basename(p)}"')
            continue
        modules[kind].append(p)

    if not modules["text_encoder"]:
        raise ValueError("MiniMax-H3 requires the Qwen3-VL-32B text encoder (select it in VAE / Text Encoder)")
    if not modules["vae"]:
        raise ValueError("MiniMax-H3 requires the video VAE (select it in VAE / Text Encoder)")

    transformer = load_transformer(path)
    memory_management.soft_empty_cache()
    text_encoder = load_text_encoder(modules["text_encoder"] + modules["vision"])
    memory_management.soft_empty_cache()
    vae = load_vae(modules["vae"][0])
    audio_vae = load_audio_vae(modules["audio_vae"][0]) if modules["audio_vae"] else None

    return MiniMaxH3(transformer, text_encoder, vae, audio_vae, is_ref2va="ref" in os.path.basename(path).lower())


# region VRAM


_BASE_RESERVED_VRAM: int = None


def _external_vram(device: torch.device) -> int:
    """VRAM used by other processes (another WebUI, games...); torch.cuda.mem_get_info misses it on Windows (WDDM)"""
    if getattr(memory_management, "PYNVML_IS_AVAILABLE", False) or device.type != "cuda":
        return 0  # Forge already accounts for it (--pynvml)
    import subprocess

    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(index)], capture_output=True, text=True, timeout=10)
        used = int(out.stdout.strip().splitlines()[0]) * 2**20
    except Exception:
        return 0
    return max(0, used - torch.cuda.memory_reserved(device))


def load_gpu(patcher: ModelPatcher, inference_memory: float = 0):
    """load_model_gpu, reserving the VRAM other processes use so that Forge offloads (streams) instead of
    oversubscribing into Windows' shared GPU memory, which makes the 17 GB text encoder and the DiT 10-100x slower"""
    global _BASE_RESERVED_VRAM
    # H3 runs its huge components one after another (TE 17 GB -> DiT 11 GB -> VAE 5 GB); Forge's policy partially
    # loads the next one instead of evicting the previous, which ends up oversubscribing VRAM; so evict explicitly
    if not any(m.model is patcher for m in memory_management.current_loaded_models):
        memory_management.unload_all_models()
        memory_management.soft_empty_cache()
    if _BASE_RESERVED_VRAM is None:
        _BASE_RESERVED_VRAM = memory_management.EXTRA_RESERVED_VRAM
    external = _external_vram(patcher.load_device)
    memory_management.EXTRA_RESERVED_VRAM = _BASE_RESERVED_VRAM + external
    if external > 512 * 2**20:
        logger.info(f"{external / 2**30:.1f} GiB VRAM is used by other processes; reserving it")
    try:
        # keep `inference_memory` free for activations; Forge streams the rest of the weights if they do not fit
        memory_management.load_models_gpu([patcher], memory_required=inference_memory)
    finally:
        memory_management.EXTRA_RESERVED_VRAM = _BASE_RESERVED_VRAM


class _StageTimer:
    def __init__(self):
        import time

        self.time = time.perf_counter
        self.last = self.start = self.time()
        self.stages = []

    def __call__(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = self.time()
        self.stages.append((name, now - self.last))
        self.last = now

    def summary(self) -> str:
        return " | ".join(f"{n} {t:.1f}s" for n, t in self.stages if t >= 0.05) + f" | total {self.last - self.start:.1f}s"


# region Engine


class _RuntimeLoRA:
    """forward hook: out + scale * B(A x); A / B are cached on the input's device and dtype"""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, scale: float):
        self.A, self.B, self.scale = A, B, scale
        self.cache = None

    def __call__(self, module, inputs, output):
        x = inputs[0]
        if self.cache is None or self.cache[0].device != x.device or self.cache[0].dtype != x.dtype:
            self.cache = (self.A.to(device=x.device, dtype=x.dtype), self.B.to(device=x.device, dtype=x.dtype))
        A, B = self.cache
        return output + F.linear(F.linear(x, A), B).mul_(self.scale).to(output.dtype)


class MiniMaxH3(ForgeDiffusionEngine):
    matched_guesses = []

    def __init__(self, transformer, text_encoder, vae, audio_vae=None, is_ref2va: bool = False):
        # does not go through huggingface_guess; set up the attributes the WebUI reads
        self.model_config = SimpleNamespace(model_type=SimpleNamespace(name="FLOW"), inpaint_model=lambda: False, ztsnr=False, huggingface_repo="MiniMaxAI/MiniMax-H3")
        self.is_inpaint = False
        self.current_lora_hash = str([])
        self.tiling_enabled = False
        self.use_distilled_cfg_scale = False
        self.use_shift = True
        self.is_sd1 = self.is_sdxl = self.is_wan = False
        self.is_minimax_h3 = True
        self.is_ref2va = is_ref2va
        self.ini_latent = None
        self.ref_latents = []

        load_device = memory_management.get_torch_device()
        self.transformer = ModelPatcher(transformer, load_device=load_device, offload_device=memory_management.unet_offload_device())
        self.transformer_original = self.transformer
        self.lora_hash = str([])
        self.lora_hooks = []
        self.lora_adapters: list[_RuntimeLoRA] = []
        self.text_encoder = ModelPatcher(text_encoder, load_device=memory_management.text_encoder_device(), offload_device=memory_management.text_encoder_offload_device())
        self.vae = ModelPatcher(vae, load_device=memory_management.vae_device(), offload_device=memory_management.vae_offload_device())
        self.audio_vae = None if audio_vae is None else ModelPatcher(audio_vae, load_device=memory_management.vae_device(), offload_device=memory_management.vae_offload_device())

        from transformers import Qwen2TokenizerFast

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(TOKENIZER)

        vae_stub = SimpleNamespace(upscale_ratio=16, latent_channels=24, first_stage_model=vae, patcher=self.vae)
        clip_stub = SimpleNamespace(patcher=self.text_encoder, cond_stage_model=text_encoder, tokenizer=self.tokenizer)
        self.forge_objects = ForgeObjects(unet=self.transformer, clip=clip_stub, vae=vae_stub, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.shift = SHIFT_VIDEO

    def set_shift(self, shift: float):
        self.shift = float(shift)

    # region LoRA

    def _lora_key_map(self) -> dict[str, str]:
        key_map = {}
        for k in self.transformer_original.model.state_dict().keys():
            if not k.endswith(".weight"):
                continue
            base = k[: -len(".weight")]
            for prefix in ("diffusion_model.", "model.diffusion_model.", "transformer.", ""):
                key_map[prefix + base] = k
            key_map["lora_unet_" + base.replace(".", "_")] = k
        return key_map

    def set_loras(self, loras: list[tuple[str, float]]):
        """loras: [(path, strength)]; ComfyUI-format MiniMax-H3 LoRAs (e.g. lightx2v Turbo *_comfyui_*.safetensors)"""
        from modules_forge.packages.comfy.lora import load_lora

        lora_hash = str([(os.path.abspath(p), float(w)) for p, w in loras])
        if lora_hash == self.lora_hash:
            return
        self.lora_hash = lora_hash

        patcher = self.transformer_original.clone() if loras else self.transformer_original
        dit = self.transformer_original.model
        online = getattr(dit, "quantized", False)
        key_map = self._lora_key_map()

        for handle in self.lora_hooks:
            handle.remove()
        self.lora_hooks.clear()
        self.lora_adapters.clear()

        for path, strength in loras:
            lora_sd = utils.load_torch_file(path)
            if online and self._add_runtime_lora(dit, lora_sd, key_map, float(strength), path):
                del lora_sd
                continue
            patches, unmatched = load_lora(lora_sd, key_map)
            if unmatched:
                logger.info(f"LoRA \"{os.path.basename(path)}\": {len(unmatched)} unmatched keys")
            loaded = patcher.add_patches(patches, float(strength), filename=path, online_mode=online)
            if len(loaded) == 0:
                logger.warning(f'LoRA "{os.path.basename(path)}" matched no MiniMax-H3 weights (use the ComfyUI-format file)')
            else:
                logger.info(f'Loaded LoRA "{os.path.basename(path)}" ({len(loaded)} weights, strength {strength}, online: {online})')
            del lora_sd

        self.transformer = patcher
        self.forge_objects.unet = patcher
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def _add_runtime_lora(self, dit: torch.nn.Module, lora_sd: dict, key_map: dict, strength: float, path: str) -> bool:
        """
        quantized (GGUF / int8 / nvfp4) DiT: apply plain LoRAs as a low-rank side path, out += s * B(A x),
        instead of re-merging B @ A into every dequantized weight on every forward (Forge's online patching)
        """
        pairs = {}
        for k in lora_sd:
            for down, up in ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight")):
                if k.endswith(down) and k[: -len(down)] + up in lora_sd:
                    pairs[k[: -len(down)]] = (lora_sd[k], lora_sd[k[: -len(down)] + up], lora_sd.get(k[: -len(down)] + ".alpha", None))
        if not pairs or len(pairs) * 2 < sum(1 for k in lora_sd if k.endswith(".weight")):
            return False  # not a plain LoRA (LoHa / LoKr / diff ...): use Forge's patching

        matched = 0
        for base, (A, B, alpha) in pairs.items():
            if (target := key_map.get(base, None)) is None:
                continue
            module = dit.get_submodule(target[: -len(".weight")])
            rank = A.shape[0]
            scale = strength * ((float(alpha) / rank) if alpha is not None else 1.0)
            adapter = _RuntimeLoRA(A, B, scale)
            self.lora_adapters.append(adapter)
            self.lora_hooks.append(module.register_forward_hook(adapter))
            matched += 1

        if matched == 0:
            logger.warning(f'LoRA "{os.path.basename(path)}" matched no MiniMax-H3 weights (use the ComfyUI-format file)')
        else:
            logger.info(f'Loaded LoRA "{os.path.basename(path)}" ({matched} layers, strength {strength}, runtime low-rank)')
        return True

    def get_learned_conditioning(self, prompt: list[str]):
        raise NotImplementedError("MiniMax-H3 runs through modules_forge.minimax_h3")

    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        count = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return count, max(7000, count)

    # region Conditioning

    def _tokens(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"] if text else []

    @torch.inference_mode()
    def encode_prompt(self, prompt: str, pictures: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """pictures: [H, W, C] images shown to Qwen3-VL as <Picture i>"""
        from backend.nn.minimax_h3.text_encoder import VISION_END, VISION_START

        te = self.text_encoder.model
        entries = []
        if te.has_vision:
            for i, img in enumerate(pictures):
                entries += self._tokens(f"<Picture {i + 1}>: ")
                entries += [VISION_START, {"data": img.unsqueeze(0).float(), "video_block": False}, VISION_END]
        entries += self._tokens(prompt)

        load_gpu(self.text_encoder, inference_memory=1.5 * 2**30)  # bf16, ~1 GB transient: fits 24 GB entirely
        device = self.text_encoder.load_device
        hidden, tags = te.encode(entries, device)
        if not torch.isfinite(hidden).all():
            logger.warning("[Self-heal] NaN / Inf in the text encoder output; re-encoding in fp32")
            hidden, tags = te.encode(entries, device, dtype=torch.float32)
            if not torch.isfinite(hidden).all():
                raise RuntimeError("MiniMax-H3 text encoder produced NaN / Inf even in fp32 (corrupted text encoder file?)")
        return hidden.cpu(), tags

    @torch.inference_mode()
    def vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """[H, W, C] in [0, 1] -> normalized latent [1, 24, 1, H/16, W/16]"""
        load_gpu(self.vae)
        model = self.vae.model
        x = image.movedim(-1, 0).unsqueeze(0).unsqueeze(2).mul(2.0).sub(1.0)
        x = x.to(device=self.vae.load_device, dtype=model.dtype)
        return model.encode(x).float().cpu()

    @torch.inference_mode()
    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        """normalized latent -> [F, H, W, C] float in [0, 1] (cpu)"""
        # measured: ~9 GB of activations decoding a 22-frame chunk in 256 px tiles (batches up to 4 tiles when VRAM allows)
        load_gpu(self.vae, inference_memory=9 * 2**30)
        model = self.vae.model
        out = model.decode(z.to(device=self.vae.load_device, dtype=model.dtype))
        if not torch.isfinite(out).all() and model.dtype != torch.float32:
            logger.warning(f"[Self-heal] NaN / Inf in the VAE decode ({model.dtype}); retrying in fp32")
            original = model.dtype
            try:
                model.to(torch.float32)
                model.dtype = torch.float32
                out = model.decode(z.to(device=self.vae.load_device, dtype=torch.float32))
            finally:
                model.to(original)
                model.dtype = original
        return out[0].float().nan_to_num(0.5).movedim(0, -1).cpu()

    @torch.inference_mode()
    def audio_decode(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.audio_vae is None:
            return None
        load_gpu(self.audio_vae)
        return self.audio_vae.model.decode(z.to(device=self.audio_vae.load_device, dtype=torch.float32)).float().cpu()

    # region Generation

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        length: int,
        steps: int,
        seed: int,
        first_frame: torch.Tensor = None,
        last_frame: torch.Tensor = None,
        references: list[torch.Tensor] = None,
        callback: Callable[[int, int], bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        returns (frames [F, H, W, C] in [0, 1], waveform [2, L] at 32kHz or None)
        callback(step, total) -> True to interrupt
        """
        width, height = snap(width), snap(height)
        frame_count = align_frame_count(length)
        latent_t = video_latent_t(frame_count)
        audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
        references = references or []

        keyframes, refs, pictures = [], [], []

        if references:
            if not self.is_ref2va:
                logger.warning("Reference images are meant for the Ref2VA checkpoint")
            logger.info(f"[MiniMax-H3] Reference-to-Video ({len(references)} reference(s))")
            for img in references:
                h, w = img.shape[0], img.shape[1]
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
                tw, th = snap(w * scale), snap(h * scale)
                resized = resize_image(img, tw, th, crop=False)
                pictures.append(resized)
                refs.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "ref_audio_t": 0, "latent": None, "_image": resized})
            # keyframes on top of references (ComfyUI's MiniMaxH3AddGuide): latent-only guides, not shown to Qwen
            if first_frame is not None:
                keyframes.append({"resolved_frame_index": 0, "_image": resize_image(first_frame, width, height, crop=True)})
            if last_frame is not None:
                keyframes.append({"resolved_frame_index": frame_count - 1, "_image": resize_image(last_frame, width, height, crop=True)})
        else:
            if first_frame is not None:
                img = resize_image(first_frame, width, height, crop=False)
                pictures.append(img)
                keyframes.append({"resolved_frame_index": 0, "_image": img})
            if last_frame is not None:
                img = resize_image(last_frame, width, height, crop=True)
                pictures.append(img)
                keyframes.append({"resolved_frame_index": frame_count - 1, "_image": img})
            mode = {(False, False): "Text-to-Video", (True, False): "Image-to-Video", (False, True): "LastFrame-to-Video", (True, True): "FirstLastFrame-to-Video"}
            logger.info(f"[MiniMax-H3] {mode[(first_frame is not None, last_frame is not None)]}")

        logger.info(f"[MiniMax-H3] {width}x{height} | {frame_count} frames ({frame_count / FPS:.2f}s) | {steps} steps | shift {self.shift}")

        timer = _StageTimer()
        text_states, tags = self.encode_prompt(prompt, pictures)
        timer("text encoder")

        for item in keyframes + refs:
            item["latent"] = self.vae_encode(item.pop("_image"))

        payload = {
            "text_token_tags": tags,
            "seed": int(seed),
            "cond_video_latents": [k["latent"] for k in keyframes] + [r["latent"] for r in refs],
            "cond_audio_latents": [],
        }
        if keyframes:
            payload["keyframes"] = keyframes
        if refs:
            payload["refs"] = refs

        generator = torch.Generator("cpu").manual_seed(int(seed))
        x_video = torch.randn((1, 24, latent_t, height // 16, width // 16), generator=generator, dtype=torch.float32)
        x_audio = torch.randn((1, 32, 2, audio_t), generator=generator, dtype=torch.float32)

        # activations of the packed sequence (hidden 5376, qkv 3x, mlp 2x14336) + a dequantized weight in flight
        tokens = latent_t * (height // 32) * (width // 32) + audio_t * 2 + text_states.shape[1] + sum(int(k["latent"].shape[2]) * (height // 32) * (width // 32) for k in keyframes) + sum(int(r["latent"].shape[2] * r["latent"].shape[3] * r["latent"].shape[4] // 4) for r in refs)
        timer("vae encode")
        load_gpu(self.transformer, inference_memory=tokens * (5376 * 2 * 8 + 28672 * 2 * 2) + 2**30)
        timer("load transformer")
        device = self.transformer.load_device
        dit = self.transformer.model
        dtype = dit.computation_dtype

        context = dit.preprocess_text_embeds(text_states.to(device=device, dtype=dtype))
        x_video, x_audio = x_video.to(device), x_audio.to(device)

        shift_v, shift_a = self.shift, SHIFT_AUDIO
        base = torch.linspace(1.0, 0.0, int(steps) + 1, dtype=torch.float32)
        sigmas_v = time_shift(base, shift_v)
        sigmas_a = time_shift(base, shift_a)
        transformer_options = {"sample_sigmas": sigmas_v, "minimax_h3_sigma_shift_video": shift_v, "minimax_h3_sigma_shift_audio": shift_a}

        for i in range(int(steps)):
            if callback is not None and callback(i, int(steps)):
                break
            sv, sv_next = sigmas_v[i], sigmas_v[i + 1]
            sa, sa_next = sigmas_a[i], sigmas_a[i + 1]
            timestep = (sv * 1000.0).view(1).to(device)
            out_v, out_a = dit([x_video, x_audio], timestep, context, transformer_options=transformer_options, minimax_payload=payload)
            # Euler on the rectified flow: x0 = x - sigma * v
            if not (torch.isfinite(out_v).all() and torch.isfinite(out_a).all()):
                raise RuntimeError(f"MiniMax-H3 produced NaN / Inf at step {i + 1} (try fewer / more steps, another shift, or a less aggressive quantization)")
            x_video = x_video + (sv_next - sv).item() * out_v.float()
            x_audio = x_audio + (sa_next - sa).item() * out_a.float()

        if callback is not None:
            callback(int(steps), int(steps))

        del context
        timer(f"{int(steps)} steps")
        for adapter in self.lora_adapters:
            adapter.cache = None  # the 1.8 GB Turbo LoRA must not stay in VRAM next to the text encoder / VAE
        frames = self.vae_decode(x_video)[:frame_count]
        timer("vae decode")
        audio = self.audio_decode(x_audio)
        timer("audio decode")
        logger.info(f"[MiniMax-H3] {timer.summary()}")
        return frames, (audio[0] if audio is not None else None)
