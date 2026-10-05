"""WebUI glue for MiniMax-H3: txt2img -> Text-to-Video, img2img -> Image-to-Video, Reference Images -> Reference-to-Video"""

import os
import tempfile
import wave

import numpy as np
import torch
from PIL import Image

reference_images: list[str] = []
"""file paths from the "Reference Images" picker next to the Checkpoint selection"""


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


def process_images(p) -> "Processed":
    from backend.diffusion_engine.minimax_h3 import AUDIO_SAMPLE_RATE, FPS
    from modules import devices, images, sd_vae, shared
    from modules.processing import Processed, StableDiffusionProcessingImg2Img, create_infotext, get_fixed_seed
    from modules.shared import opts, state

    model = shared.sd_model

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

    shift = float(getattr(p, "distilled_cfg_scale", None) or 12.0)
    model.set_shift(shift)
    p.extra_generation_params["Shift"] = shift
    p.extra_generation_params["Frames"] = length

    first_frame = None
    if isinstance(p, StableDiffusionProcessingImg2Img) and getattr(p, "init_images", None):
        first_frame = _to_tensor(p.init_images[0])

    references = []
    for path in reference_images:
        try:
            references.append(_to_tensor(Image.open(path)))
        except Exception as e:
            print(f"[MiniMax-H3] failed to read reference image {path}: {e}")
    if references:
        p.extra_generation_params["Reference Images"] = len(references)

    if state.job_count == -1:
        state.job_count = p.n_iter

    output_images, infotexts = [], []
    video_path = None

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
        if p.n_iter > 1:
            state.job = f"Batch {n + 1} out of {p.n_iter}"

        frames, audio = model.generate(
            prompt=p.prompts[0],
            width=p.width,
            height=p.height,
            length=length,
            steps=p.steps,
            seed=p.seeds[0],
            first_frame=first_frame,
            references=references,
            callback=callback,
        )

        frames = [f for f in frames.mul(255.0).round().clamp(0, 255).to(torch.uint8).numpy()]
        info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, iteration=n, position_in_batch=0)

        audio_path = _save_wav(audio, AUDIO_SAMPLE_RATE) if audio is not None else None
        try:
            video_path = images.save_video(p, frames, fps=FPS, info=info, audio_copy=audio_path)
        finally:
            if audio_path is not None and os.path.isfile(audio_path):
                os.remove(audio_path)

        preview = Image.fromarray(frames[0])
        if opts.enable_pnginfo:
            preview.info["parameters"] = info
        output_images.append(preview)
        infotexts.append(info)
        state.nextjob()

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
