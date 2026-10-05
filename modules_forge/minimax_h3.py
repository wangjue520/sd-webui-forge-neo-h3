"""
WebUI glue for MiniMax-H3
- txt2img -> Text-to-Video ; img2img (img2video) -> Image-to-Video ; Reference Images -> Reference-to-Video
- Last Frame / Long Video (segment chaining) / Turbo LoRA / Video Upscale (options from the "MiniMax-H3 Video" panel)
"""

import os
import tempfile
import wave

import numpy as np
import torch
from PIL import Image

reference_images: list[str] = []
"""file paths from the "Reference Images" picker next to the Checkpoint selection"""

ui_panels: list = []
"""the "MiniMax-H3 Video" accordions (txt2img / img2img); only visible for the h3 UI Preset"""


def set_reference_images(files: list[str] | None):
    global reference_images
    reference_images = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL -> [H, W, 3] float in [0, 1]"""
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)


def _save_wav(waveform: torch.Tensor, sample_rate: int) -> str:
    """[2, L] in [-1, 1] -> temporary 16-bit stereo wav"""
    data = (waveform.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16).T.contiguous().numpy()).tobytes()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as f:
        f.setnchannels(int(waveform.shape[0]))
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(data)
    return path


def _find_lora(name: str) -> str | None:
    from modules import shared

    for folder in [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])]:
        if not folder or not os.path.isdir(folder):
            continue
        for f in shared.walk_files(folder, allowed_extensions=[".safetensors"]):
            if os.path.splitext(os.path.basename(f))[0] == name:
                return f
    return None


def _parse_loras(prompt: str) -> tuple[str, list[tuple[str, float]]]:
    """strips <lora:name:weight> from the prompt"""
    from modules import extra_networks

    prompts, data = extra_networks.parse_prompts([prompt])
    loras = []
    for params in data.get("lora", []):
        name = params.items[0]
        weight = float(params.items[1]) if len(params.items) > 1 else 1.0
        if (path := _find_lora(name)) is None:
            print(f'[MiniMax-H3] LoRA "{name}" not found')
            continue
        loras.append((path, weight))
    return prompts[0], loras


def _upscale(frames: list[np.ndarray], upscaler_name: str, scale: float) -> list[np.ndarray]:
    from modules import shared
    from modules.shared import state

    if upscaler_name in (None, "None") or scale <= 1.0:
        return frames
    upscaler = next((x for x in shared.sd_upscalers if x.name == upscaler_name), None)
    if upscaler is None:
        print(f'[MiniMax-H3] upscaler "{upscaler_name}" not found')
        return frames

    h, w = frames[0].shape[:2]
    tw, th = int(round(w * scale / 2)) * 2, int(round(h * scale / 2)) * 2  # even size for yuv420p
    print(f"[MiniMax-H3] upscaling {len(frames)} frames with {upscaler_name}: {w}x{h} -> {tw}x{th}")
    state.textinfo = f"Upscaling ({upscaler_name})"
    out = []
    for i, f in enumerate(frames):
        if state.interrupted:
            return frames
        img = upscaler.scaler.upscale(Image.fromarray(f), scale, upscaler.data_path)
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.Resampling.LANCZOS)
        out.append(np.asarray(img.convert("RGB")))
        state.sampling_step, state.sampling_steps = i + 1, len(frames)
    return out


def process_images(p) -> "Processed":
    from backend.diffusion_engine.minimax_h3 import AUDIO_SAMPLE_RATE, FPS
    from modules import devices, images, sd_vae, shared
    from modules.processing import Processed, StableDiffusionProcessingImg2Img, create_infotext, get_fixed_seed
    from modules.shared import opts, state

    model = shared.sd_model
    options: dict = getattr(p, "h3_options", None) or {}

    # the "Frames" slider is the batch size for video presets
    length = int(p.batch_size)
    p.batch_size = 1
    p.do_not_save_grid = True

    seed = get_fixed_seed(p.seed)
    p.sd_model_name = model.sd_checkpoint_info.name_for_extra
    p.sd_model_hash = model.sd_model_hash
    p.sd_vae_name = sd_vae.get_loaded_vae_name()
    p.sd_vae_hash = sd_vae.get_loaded_vae_hash()
    p.fill_fields_from_opts()
    p.setup_prompts()
    p.all_seeds = [int(seed) + x for x in range(len(p.all_prompts))]
    p.all_subseeds = [int(get_fixed_seed(p.subseed)) + x for x in range(len(p.all_prompts))]

    # region Turbo / LoRA

    steps = int(p.steps)
    shift = float(getattr(p, "distilled_cfg_scale", None) or 12.0)
    loras: list[tuple[str, float]] = []
    if options.get("turbo"):
        if options.get("turbo_lora"):
            loras.append((options["turbo_lora"], options["turbo_strength"]))
            p.extra_generation_params["Turbo LoRA"] = os.path.basename(options["turbo_lora"])
        else:
            print("[MiniMax-H3] Turbo is enabled but no Turbo LoRA is selected")
        steps, shift = int(options["turbo_steps"]), float(options["turbo_shift"])

    prompts, prompt_loras = [], []
    for prompt in p.all_prompts:
        prompt, _loras = _parse_loras(prompt)
        prompts.append(prompt)
        prompt_loras = _loras or prompt_loras
    loras += prompt_loras
    model.set_loras(loras)
    model.set_shift(shift)

    p.steps = steps
    p.extra_generation_params["Shift"] = shift
    p.extra_generation_params["Frames"] = length

    # region Conditioning images

    first_frame = None
    if isinstance(p, StableDiffusionProcessingImg2Img) and getattr(p, "init_images", None):
        first_frame = _to_tensor(p.init_images[0])
    last_frame = _to_tensor(options["last_frame"]) if options.get("last_frame") is not None else None

    keyframes = []
    for path in options.get("keyframes", []):
        try:
            keyframes.append(_to_tensor(Image.open(path)))
        except Exception as e:
            print(f"[MiniMax-H3] failed to read keyframe {path}: {e}")

    references = []
    for path in reference_images:
        try:
            references.append(_to_tensor(Image.open(path)))
        except Exception as e:
            print(f"[MiniMax-H3] failed to read reference image {path}: {e}")
    if references:
        p.extra_generation_params["Reference Images"] = len(references)

    # region Segments
    # keyframes given: segment i goes from keyframe i (or the previous segment's last frame) to keyframe i + 1
    # otherwise: each segment continues from the previous segment's last frame
    # the optional Last Frame is the target of the final segment

    if keyframes and first_frame is None:
        first_frame, keyframes = keyframes[0], keyframes[1:]
    num_segments = max(int(options.get("segments", 1)), len(keyframes), 1)
    segment_prompts: list[str] = options.get("segment_prompts", [])
    if num_segments > 1:
        p.extra_generation_params["Segments"] = num_segments

    if state.job_count == -1:
        state.job_count = p.n_iter * num_segments

    output_images, infotexts = [], []
    video_path = None
    samples_per_frame = AUDIO_SAMPLE_RATE / FPS

    def callback(step: int, total: int) -> bool:
        state.sampling_steps = total
        state.sampling_step = step
        return bool(state.interrupted or state.skipped or state.stopping_generation)

    for n in range(p.n_iter):
        if state.interrupted or state.stopping_generation:
            break

        p.iteration = n
        p.prompts = p.all_prompts[n : n + 1]
        p.negative_prompts = p.all_negative_prompts[n : n + 1]
        p.seeds = p.all_seeds[n : n + 1]
        p.subseeds = p.all_subseeds[n : n + 1]

        video: list[np.ndarray] = []
        audio_parts: list[torch.Tensor] = []
        start = first_frame

        for i in range(num_segments):
            if state.interrupted or state.stopping_generation:
                break
            state.job = f"Segment {i + 1} / {num_segments}" + (f" (Batch {n + 1} / {p.n_iter})" if p.n_iter > 1 else "")

            if i < len(keyframes):
                end = keyframes[i]
            elif i == num_segments - 1:
                end = last_frame
            else:
                end = None

            if segment_prompts:
                prompt, _ = _parse_loras(segment_prompts[min(i, len(segment_prompts) - 1)])
            else:
                prompt = prompts[n]

            frames, audio = model.generate(
                prompt=prompt,
                width=p.width,
                height=p.height,
                length=length,
                steps=steps,
                seed=p.seeds[0] + i,
                first_frame=start,
                last_frame=end,
                references=references,
                callback=callback,
            )

            # the first frame of a continuation repeats the previous segment's last frame
            skip = 1 if i > 0 else 0
            start = frames[-1].clone()
            frames = frames[skip:]
            video.extend(f for f in frames.mul(255.0).round().clamp(0, 255).to(torch.uint8).numpy())
            if audio is not None:
                a, b = round(skip * samples_per_frame), round((skip + len(frames)) * samples_per_frame)
                audio_parts.append(audio[:, a:b])
            state.nextjob()

        if not video:
            break

        upscaler, upscale_by = options.get("upscaler"), float(options.get("upscale_by", 1.0))
        if upscaler not in (None, "None") and upscale_by > 1.0:
            video = _upscale(video, upscaler, upscale_by)
            p.extra_generation_params["Video Upscale"] = f"{upscaler} x{upscale_by}"

        info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, iteration=n, position_in_batch=0)
        audio = torch.cat(audio_parts, dim=-1) if audio_parts else None
        audio_path = _save_wav(audio, AUDIO_SAMPLE_RATE) if audio is not None else None
        try:
            video_path = images.save_video(p, video, fps=FPS, info=info, audio_copy=audio_path)
        finally:
            if audio_path is not None and os.path.isfile(audio_path):
                os.remove(audio_path)
        print(f"[MiniMax-H3] saved {len(video)} frames ({len(video) / FPS:.2f}s) to {video_path}")

        preview = Image.fromarray(video[0])
        if opts.enable_pnginfo:
            preview.info["parameters"] = info
        output_images.append(preview)
        infotexts.append(info)

    devices.torch_gc()

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0] if infotexts else "",
        subseed=p.all_subseeds[0],
        infotexts=infotexts,
    )
    res.video_path = video_path
    return res
