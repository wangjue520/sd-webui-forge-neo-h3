"""
MiniMax-H3 conditioner: Qwen3-VL-32B truncated to 50 layers
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/text_encoders/minimax.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.nn.llm import qwen_vl
from backend.nn.llm.llama import BaseLlama, Llama2_, Qwen3VL_32BConfig
from backend.nn.llm.qwen35 import QWEN3VL_VISION, Qwen3VLVisionModel

VISION_START = 151652
VISION_END = 151653
PAD = 151643
QWEN_IMAGE_MEAN = [0.5, 0.5, 0.5]
QWEN_IMAGE_STD = [0.5, 0.5, 0.5]

QWEN3VL_32B_VISION = {**QWEN3VL_VISION, "hidden_size": 1152, "intermediate_size": 4304, "depth": 27, "deepstack_visual_indexes": [8, 16, 24]}


def process_video_block(frames: torch.Tensor, patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=3136, max_pixels=12845056):
    """[2, H, W, C] frame pair -> (flatten_patches, grid_thw) with grid_t=1"""
    t, height, width, _ = frames.shape
    imgs = frames.permute(0, 3, 1, 2)
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    imgs = F.interpolate(imgs, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = torch.tensor(QWEN_IMAGE_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(QWEN_IMAGE_STD, device=imgs.device).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    patches = imgs.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.stack([torch.tensor([1, grid_h, grid_w], device=frames.device, dtype=torch.long)])
    return flatten, grid_thw


class MiniMaxQwen3VL(BaseLlama, nn.Module):
    def __init__(self, config_dict: dict = None):
        super().__init__()
        config = Qwen3VL_32BConfig()
        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config)
        self.visual = Qwen3VLVisionModel({**QWEN3VL_32B_VISION, "out_hidden_size": config.hidden_size})

    def encode_vision(self, data: torch.Tensor, video_block: bool, device: torch.device):
        if video_block:
            flatten, grid = process_video_block(data)
        else:
            flatten, grid = qwen_vl.process_qwen2vl_images(data, patch_size=16, image_mean=QWEN_IMAGE_MEAN, image_std=QWEN_IMAGE_STD)
        merged, deepstack = self.visual(flatten.to(device, dtype=torch.float32), grid)
        return merged, grid, deepstack

    @torch.inference_mode()
    def encode(self, entries: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        entries: list of token ids (int) and vision dicts {"data": [T, H, W, C] in [0, 1], "video_block": bool}
        returns: (hidden state after layer 50 [1, S, 5120], token tags [S])
        """
        if len(entries) == 0:
            entries = [PAD]

        embed_tokens = self.model.embed_tokens
        embed_cache = getattr(self, "embed_cache", None)
        # bf16 activations: half the transient memory of fp32 (the 32B conditioner barely fits a 24 GB card)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        pieces, embeds_info = [], []
        pending: list[int] = []
        index = 0

        def flush():
            nonlocal pending, index
            if pending:
                if embed_cache is not None:
                    ids = torch.tensor([pending], dtype=torch.long)
                    pieces.append(torch.nn.functional.embedding(ids, embed_cache).to(device=device, dtype=dtype))
                else:
                    ids = torch.tensor([pending], device=device, dtype=torch.long)
                    pieces.append(embed_tokens(ids).to(dtype=dtype))
                index += len(pending)
                pending = []

        for e in entries:
            if isinstance(e, int):
                pending.append(e)
                continue
            flush()
            merged, grid, deepstack = self.encode_vision(e["data"], e.get("video_block", False), device)
            merged = merged.view(1, -1, merged.shape[-1]).to(device=device, dtype=dtype)
            embeds_info.append({"type": "image", "index": index, "size": merged.shape[1], "extra": {"grid": grid, "deepstack": deepstack}})
            pieces.append(merged)
            index += merged.shape[1]
        flush()

        embeds = torch.cat(pieces, dim=1)
        seq = embeds.shape[1]

        position_ids, visual_pos_masks, deepstack = None, None, None
        if embeds_info:
            position_ids = qwen_vl.qwen2vl_mrope_position_ids(embeds_info, seq, device)
            visual_pos_masks = torch.zeros((1, seq), dtype=torch.bool, device=device)
            for e in embeds_info:
                visual_pos_masks[0, e["index"] : e["index"] + e["size"]] = True
                ds = e["extra"]["deepstack"]
                deepstack = list(ds) if deepstack is None else [torch.cat([deepstack[i], ds[i]], dim=0) for i in range(len(ds))]

        # whole vision block is VIDEO(0), including the flanking <|vision_start|>/<|vision_end|> tokens
        tags = torch.ones(seq, dtype=torch.long)
        for e in embeds_info:
            tags[max(0, e["index"] - 1) : e["index"] + e["size"] + 1] = 0

        out = self.model(None, embeds=embeds, dtype=dtype, position_ids=position_ids, deepstack_embeds=deepstack, visual_pos_masks=visual_pos_masks)
        return out[0].float(), tags
