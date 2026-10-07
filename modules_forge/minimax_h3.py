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

reference_images: dict[str, list[str]] = {"txt2img": [], "img2img": []}
"""file paths from the "参考图 / Reference" tab (next to Generation / Lora), per WebUI tab"""

ui_panels: list = []
"""components only visible for the h3 UI Preset (Reference tabs, Segments / Duration, MiniMax-H3 Video panels)"""


reference_videos: dict[str, list[str]] = {"txt2img": [], "img2img": []}
reference_audios: dict[str, list[str]] = {"txt2img": [], "img2img": []}


def _paths(files: list | None) -> list[str]:
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]


def set_reference_images(tabname: str, files: list | None):
    reference_images[tabname] = _paths(files)


# region ControlNet preprocessing

# label shown in the UI -> Forge preprocessor name (modules_forge.shared.supported_preprocessors)
CONTROL_PREPROCESSORS = {
    "姿态 Pose (DWPose)": "dw_openpose_full",
    "姿态 Pose (OpenPose)": "openpose_full",
    "深度 Depth (Depth Anything V2)": "depth_anything_v2",
    "深度 Depth (MiDaS)": "depth_midas",
    "线稿 Canny": "canny",
    "线稿 Lineart (写实)": "lineart_realistic",
    "线稿 Lineart (动漫)": "lineart_anime",
    "软边缘 SoftEdge (PiDiNet)": "softedge_pidinet",
    "涂鸦 Scribble (PiDiNet)": "scribble_pidinet",
    "无 / 已处理好的控制视频": None,
}


def preprocess_video(frames: torch.Tensor, label: str, resolution: int) -> torch.Tensor:
    """[T, H, W, 3] in [0, 1] -> control frames of the same size, extracted frame by frame with a Forge preprocessor"""
    import cv2

    from backend import memory_management
    from modules.shared import state
    from modules_forge.shared import supported_preprocessors

    name = CONTROL_PREPROCESSORS.get(label, label)
    if name is None:
        return frames
    pre = supported_preprocessors.get(name, None)
    if pre is None:
        raise ValueError(f'ControlNet preprocessor "{name}" is not available')

    s1 = pre.slider_1.value if getattr(pre.slider_1, "visible", False) else None
    s2 = pre.slider_2.value if getattr(pre.slider_2, "visible", False) else None
    # legacy preprocessors unload their model after every call: keep it for the whole clip
    unload = getattr(pre, "unload_function", None)
    if unload is not None:
        pre.unload_function = None
    out = []
    h, w = frames.shape[1:3]
    try:
        state.textinfo = f"ControlNet preprocessing ({name})"
        for i, f in enumerate((frames.numpy() * 255).round().astype(np.uint8)):
            if state.interrupted:
                raise RuntimeError("interrupted")
            r = pre(f, resolution, slider_1=s1, slider_2=s2)
            r = np.asarray(r)
            if r.ndim == 2:
                r = np.repeat(r[..., None], 3, axis=-1)
            if r.shape[:2] != (h, w):
                r = cv2.resize(r[..., :3], (w, h), interpolation=cv2.INTER_LINEAR)
            out.append(r[..., :3])
            state.sampling_step, state.sampling_steps = i + 1, len(frames)
    finally:
        if unload is not None:
            pre.unload_function = unload
            unload()
        memory_management.soft_empty_cache()
    print(f"[MiniMax-H3] ControlNet preprocessing: {len(out)} frames with {name}")
    return torch.from_numpy(np.stack(out).astype(np.float32) / 255.0)


# region Media (ffmpeg)


def load_video(path: str, fps: int = 24, max_frames: int = 15 * 24 + 5) -> torch.Tensor:
    """video file -> [T, H, W, 3] float in [0, 1], resampled to `fps`"""
    import json
    import subprocess

    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path], capture_output=True, text=True, check=True).stdout)
    w, h = int(info["streams"][0]["width"]), int(info["streams"][0]["height"])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-r", str(fps), "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    frames = frames[: (frames.size // (w * h * 3)) * w * h * 3].reshape(-1, h, w, 3)
    return torch.from_numpy(frames.astype(np.float32) / 255.0)


def load_audio(path: str, sample_rate: int = 32000, max_seconds: float = 30.0) -> torch.Tensor | None:
    """audio file (or a video's soundtrack) -> [2, L] float in [-1, 1] at `sample_rate`; None if there is no audio stream"""
    import subprocess

    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-t", str(max_seconds), "-ac", "2", "-ar", str(sample_rate), "-f", "f32le", "-"], capture_output=True)
    if out.returncode != 0 or len(out.stdout) < 8 * 800:
        return None
    pcm = np.frombuffer(out.stdout, dtype=np.float32)
    return torch.from_numpy(pcm[: pcm.size // 2 * 2].reshape(-1, 2).T.copy())


def create_reference_tab(tabname: str):
    """a tab in the Generation / Textual Inversion / Checkpoints / Lora row"""
    import gradio as gr

    from modules import shared

    with gr.Tab("参考图 / Reference", id=f"{tabname}_h3_references", elem_id=f"{tabname}_h3_references_tab", visible=shared.opts.forge_preset == "h3") as tab:
        gr.Markdown("参考图生视频（Reference-to-Video）：上传 1~9 张参考图，在提示词中可用 Picture 1、Picture 2… 指代。需使用 Ref2VA 模型；留空则为普通文生 / 图生视频。")
        files = gr.File(label="Reference Images", file_count="multiple", file_types=["image"], type="filepath", elem_id=f"{tabname}_h3_references")
        gallery = gr.Gallery(label="Preview", columns=6, height=240, interactive=False, visible=False, elem_id=f"{tabname}_h3_references_preview")

        def on_change(f):
            set_reference_images(tabname, f)
            return gr.update(value=reference_images[tabname], visible=bool(reference_images[tabname]))

        files.change(on_change, inputs=[files], outputs=[gallery], queue=False, show_progress=False)

        gr.Markdown("参考视频（自带声音会一起参考）与参考音频：提示词中用 Video 1、Audio 1… 指代；参考视频会按目标画面大小缩放，并截到与生成时长相同。")
        with gr.Row():
            videos = gr.File(label="Reference Videos (≤3)", file_count="multiple", file_types=["video"], type="filepath", elem_id=f"{tabname}_h3_reference_videos")
            audios = gr.File(label="Reference Audios (≤3)", file_count="multiple", file_types=["audio"], type="filepath", elem_id=f"{tabname}_h3_reference_audios")

        def on_videos(f):
            reference_videos[tabname] = _paths(f)[:3]

        def on_audios(f):
            reference_audios[tabname] = _paths(f)[:3]

        videos.change(on_videos, inputs=[videos], queue=False, show_progress=False)
        audios.change(on_audios, inputs=[audios], queue=False, show_progress=False)
    ui_panels.append(tab)


def preset_targets() -> list:
    """components updated on UI Preset change (see main_entry.on_preset_change)"""
    from modules_forge import main_entry

    targets = list(ui_panels)
    if (modes := getattr(main_entry, "ui_img2img_modes", None)) is not None:
        tabs, items, selected = modes
        targets += [tabs, *items, selected]
    return targets


def preset_updates(preset: str) -> list:
    import gradio as gr

    from modules_forge import main_entry

    h3 = preset == "h3"
    updates = [gr.update(visible=h3) for _ in ui_panels]
    if (modes := getattr(main_entry, "ui_img2img_modes", None)) is not None:
        _, items, _ = modes
        updates.append(gr.update(selected="h3_i2v" if h3 else "img2img"))
        updates += [gr.update(visible=not h3) for _ in items[:-2]]
        updates += [gr.update(visible=h3) for _ in items[-2:]]
        updates.append(gr.update(value=6 if h3 else 0))
    return updates


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

    # length from "Video Duration" (seconds @ 24 fps); the Batch Size slider is hidden for h3
    length = round(float(options["duration"]) * FPS) if options.get("duration") else int(p.batch_size)
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
    if options.get("duration"):
        p.extra_generation_params["Duration"] = options["duration"]

    # region Conditioning images

    first_frame = None
    if isinstance(p, StableDiffusionProcessingImg2Img) and getattr(p, "init_images", None):
        first_frame = _to_tensor(p.init_images[0])
    # "首尾帧图生视频" tab
    last_frame = _to_tensor(p.h3_last_frame) if getattr(p, "h3_last_frame", None) is not None else None

    keyframes = []
    for path in options.get("keyframes", []):
        try:
            keyframes.append(_to_tensor(Image.open(path)))
        except Exception as e:
            print(f"[MiniMax-H3] failed to read keyframe {path}: {e}")

    references = []
    for path in reference_images["img2img" if isinstance(p, StableDiffusionProcessingImg2Img) else "txt2img"]:
        try:
            references.append(_to_tensor(Image.open(path)))
        except Exception as e:
            print(f"[MiniMax-H3] failed to read reference image {path}: {e}")
    if references:
        p.extra_generation_params["Reference Images"] = len(references)

    tab = "img2img" if isinstance(p, StableDiffusionProcessingImg2Img) else "txt2img"
    ref_videos, ref_audios = [], []
    for path in reference_videos[tab]:
        try:
            ref_videos.append({"frames": load_video(path), "audio": load_audio(path, AUDIO_SAMPLE_RATE)})
        except Exception as e:
            print(f"[MiniMax-H3] failed to read reference video {path}: {e}")
    for path in reference_audios[tab]:
        if (waveform := load_audio(path, AUDIO_SAMPLE_RATE)) is not None:
            ref_audios.append(waveform)
        else:
            print(f"[MiniMax-H3] failed to read reference audio {path}")
    guides = []
    for path, seconds in options.get("guides", []):
        try:
            guides.append((round(float(seconds) * FPS), _to_tensor(Image.open(path))))
        except Exception as e:
            print(f"[MiniMax-H3] failed to read frame guide {path}: {e}")
    if guides:
        p.extra_generation_params["Frame Guides"] = ", ".join(f"{s}s" for _, s in options["guides"])
    control, control_preview = None, None
    if (c := options.get("control")) and c.get("model"):
        control = {"model": c["model"], "strength": c["strength"], "start": c["start"], "end": c["end"], "video": None, "mask": None, "source": None}
        if c.get("video"):
            from backend.diffusion_engine.minimax_h3 import align_frame_count

            video = load_video(c["video"])[: align_frame_count(length)]  # only the frames the generation uses
            if c.get("preprocessor") and CONTROL_PREPROCESSORS.get(c["preprocessor"], c["preprocessor"]) is not None:
                video = preprocess_video(video, c["preprocessor"], resolution=min(p.width, p.height))
                control_preview = [f for f in (video.numpy() * 255).round().astype(np.uint8)]
                p.extra_generation_params["Control Preprocessor"] = CONTROL_PREPROCESSORS.get(c["preprocessor"], c["preprocessor"])
            control["video"] = video
        if c.get("mask"):
            control["mask"] = load_video(c["mask"]).mean(dim=-1)  # white = regenerate
            if c.get("source"):
                control["source"] = load_video(c["source"])
        p.extra_generation_params["Fun ControlNet"] = f'{os.path.basename(c["model"])} x{c["strength"]} ({c["start"]:.2f}-{c["end"]:.2f})'
    if ref_videos:
        p.extra_generation_params["Reference Videos"] = len(ref_videos)
    if ref_audios:
        p.extra_generation_params["Reference Audios"] = len(ref_audios)

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

    def _preview(image):
        shared.state.assign_current_image(image)

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
                sampler=p.sampler_name,
                preview=_preview if opts.live_previews_enable else None,
                ref_videos=ref_videos,
                ref_audios=ref_audios,
                guides=guides if i == 0 else None,  # guide times refer to the first segment
                control=control if num_segments == 1 else None,
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
        if control_preview is not None and n == 0:
            # the extracted control video (pose / depth / lines ...), to check the preprocessing
            control_path = images.save_video(p, control_preview, fps=FPS, basename="control")
            print(f"[MiniMax-H3] control video saved to {control_path}")

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
