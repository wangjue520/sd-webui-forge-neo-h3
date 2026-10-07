"""MiniMax-H3 options (only shown for the h3 UI Preset)
- under Width / Height: Segments (long video by chaining first / last frames) and Video Duration
- "MiniMax-H3 Video" panel: Long Video keyframes & prompts / Turbo LoRA / Video Upscale
"""

import os

import gradio as gr

from modules import scripts, shared
from modules_forge import minimax_h3


def _is_h3() -> bool:
    return shared.opts.forge_preset == "h3"


def _options(p) -> dict:
    if getattr(p, "h3_options", None) is None:
        p.h3_options = {}
    return p.h3_options


def list_loras() -> list[str]:
    files = []
    for folder in [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])]:
        if folder and os.path.isdir(folder):
            files.extend(shared.walk_files(folder, allowed_extensions=[".safetensors"]))
    h3 = sorted({os.path.basename(f) for f in files if "h3" in os.path.basename(f).lower()})
    return h3 or ["None"]


def find_lora(name: str) -> str | None:
    for folder in [shared.cmd_opts.lora_dir, *getattr(shared.cmd_opts, "lora_dirs", [])]:
        if not folder or not os.path.isdir(folder):
            continue
        for f in shared.walk_files(folder, allowed_extensions=[".safetensors"]):
            if os.path.basename(f) == name:
                return f
    return None


def _controlnet_dirs() -> list[str]:
    dirs = [os.path.join(os.path.dirname(shared.cmd_opts.lora_dir or ""), "model_patches"), os.path.join("models", "ControlNet")]
    dirs += list(getattr(shared.cmd_opts, "controlnet_dirs", None) or [])
    for d in list(getattr(shared.cmd_opts, "lora_dirs", None) or []) + list(getattr(shared.cmd_opts, "text_encoder_dirs", None) or []):
        dirs.append(os.path.join(os.path.dirname(os.path.normpath(d)), "model_patches"))
    return [d for d in dict.fromkeys(dirs) if d and os.path.isdir(d)]


def list_controlnets() -> list[str]:
    found = {}
    for d in _controlnet_dirs():
        for f in shared.walk_files(d, allowed_extensions=[".safetensors", ".gguf"]):
            name = os.path.basename(f).lower()
            if "h3" in name and "control" in name:
                found[os.path.basename(f)] = f
    return ["None"] + sorted(found)


def find_controlnet(name: str) -> str | None:
    for d in _controlnet_dirs():
        for f in shared.walk_files(d, allowed_extensions=[".safetensors", ".gguf"]):
            if os.path.basename(f) == name:
                return f
    return None


def list_upscalers() -> list[str]:
    return ["None"] + [x.name for x in shared.sd_upscalers if x.name != "None"]


class MiniMaxH3Length(scripts.Script):
    section = "dimensions"
    create_group = False

    def title(self):
        return "MiniMax-H3 Length"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Row(visible=_is_h3(), elem_id=self.elem_id("row")) as row:
            segments = gr.Slider(label="分段数 / Segments", info="首尾帧串联长视频：每段以上一段的末帧为首帧", minimum=1, maximum=20, step=1, value=1, elem_id=self.elem_id("segments"))
            duration = gr.Slider(label="视频时长 / Duration (s)", info="每段时长，24 fps（自动对齐到 17k+5 帧）", minimum=1.0, maximum=15.0, step=0.5, value=5.0, elem_id=self.elem_id("duration"))
        minimax_h3.ui_panels.append(row)
        return [segments, duration]

    def before_process(self, p, segments, duration, *args, **kwargs):
        options = _options(p)
        options["segments"] = int(segments)
        options["duration"] = float(duration)


class MiniMaxH3Video(scripts.Script):
    sorting_priority = 1

    def title(self):
        return "MiniMax-H3 Video"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title(), visible=_is_h3(), elem_id=self.elem_id("panel")) as panel:
            with gr.Tab("长视频关键帧 / Keyframes"):
                gr.Markdown("配合「分段数」使用：第 1 张关键帧作为开头（img2video 有首帧时以首帧开头），之后第 i 段以第 i+1 张关键帧为尾帧；首尾帧模式的尾帧作为最后一段的尾帧。")
                keyframes = gr.File(label="Keyframes (按文件名排序)", file_count="multiple", file_types=["image"], type="filepath", height=96)
                segment_prompts = gr.Textbox(label="Segment Prompts (每行一段，留空则用主提示词)", lines=3)

            with gr.Tab("中间帧引导 / Frame Guides"):
                gr.Markdown("在任意时间点放置画面锚点（ComfyUI 的 AddGuide）：上传图片（按文件名排序），并填写对应的秒数（逗号分隔，可用负数表示从结尾倒数）。可与首帧 / 尾帧 / 参考图同时使用。")
                guide_files = gr.File(label="Guide Images", file_count="multiple", file_types=["image"], type="filepath", height=96)
                guide_times = gr.Textbox(label="Times (seconds)", placeholder="例如: 1.0, 2.5, -0.5", lines=1)

            with gr.Tab("ControlNet 控制"):
                gr.Markdown("Fun ControlNet-Union：上传普通视频，选择预处理（姿态 / 深度 / 线稿…）自动逐帧提取控制信号；或用遮罩做视频局部重绘（遮罩白色 = 重新生成，源视频提供其余部分）。提取出的控制视频会另存一份（文件名带 control）便于检查。模型放在 MOD\\model_patches。")
                with gr.Row():
                    cn_model = gr.Dropdown(label="ControlNet Model", choices=list_controlnets(), value="None")
                    cn_refresh = gr.Button("🔄", scale=0, min_width=40)
                cn_pre = gr.Dropdown(label="预处理 / Preprocessor", choices=list(minimax_h3.CONTROL_PREPROCESSORS), value="姿态 Pose (DWPose)")
                with gr.Row():
                    cn_video = gr.File(label="Control Video（普通视频即可）", file_types=["video"], type="filepath", height=96)
                    cn_mask = gr.File(label="Inpaint Mask (video or image)", file_types=["video", "image"], type="filepath", height=96)
                    cn_source = gr.File(label="Source Video (for inpaint)", file_types=["video"], type="filepath", height=96)
                with gr.Row():
                    cn_strength = gr.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
                    cn_start = gr.Slider(label="Start", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
                    cn_end = gr.Slider(label="End", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                cn_refresh.click(lambda: gr.update(choices=list_controlnets()), outputs=[cn_model], queue=False, show_progress=False)

            with gr.Tab("Turbo 加速"):
                gr.Markdown("lightx2v Turbo LoRA（请使用 *_comfyui_* 格式文件，放在 models/Lora）。推荐 8 步 / Shift 6。")
                with gr.Row():
                    turbo = gr.Checkbox(label="Enable Turbo", value=False)
                    turbo_lora = gr.Dropdown(label="Turbo LoRA", choices=list_loras(), value=None)
                    refresh = gr.Button("🔄", scale=0, min_width=40)
                with gr.Row():
                    turbo_strength = gr.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
                    turbo_steps = gr.Slider(label="Turbo Steps", minimum=2, maximum=20, step=1, value=8)
                    turbo_shift = gr.Slider(label="Turbo Shift", minimum=1.0, maximum=24.0, step=0.5, value=6.0)
                refresh.click(lambda: gr.update(choices=list_loras()), outputs=[turbo_lora], queue=False, show_progress=False)

            with gr.Tab("视频放大 / Upscale"):
                with gr.Row():
                    upscaler = gr.Dropdown(label="Upscaler", choices=list_upscalers(), value="None")
                    upscale_by = gr.Slider(label="Scale", minimum=1.0, maximum=4.0, step=0.25, value=2.0)

        minimax_h3.ui_panels.append(panel)
        return [keyframes, segment_prompts, turbo, turbo_lora, turbo_strength, turbo_steps, turbo_shift, upscaler, upscale_by, guide_files, guide_times, cn_model, cn_video, cn_mask, cn_source, cn_strength, cn_start, cn_end, cn_pre]

    def before_process(self, p, keyframes, segment_prompts, turbo, turbo_lora, turbo_strength, turbo_steps, turbo_shift, upscaler, upscale_by, guide_files=None, guide_times="", cn_model="None", cn_video=None, cn_mask=None, cn_source=None, cn_strength=1.0, cn_start=0.0, cn_end=1.0, cn_pre=None, *args, **kwargs):
        if cn_model not in (None, "None") and (cn_video or cn_mask):
            _options(p)["control"] = {"model": find_controlnet(cn_model), "video": cn_video, "mask": cn_mask, "source": cn_source, "strength": float(cn_strength), "start": float(cn_start), "end": float(cn_end), "preprocessor": cn_pre}
        files = sorted(guide_files or [])
        times = [t.strip() for t in (guide_times or "").replace("，", ",").split(",") if t.strip()]
        if files and len(times) != len(files):
            raise ValueError(f"Frame Guides: {len(files)} image(s) but {len(times)} time(s)")
        _options(p).update(
            {
                "guides": list(zip(files, [float(t) for t in times])),
                "keyframes": sorted(keyframes or []),
                "segment_prompts": [x.strip() for x in (segment_prompts or "").splitlines() if x.strip()],
                "turbo": bool(turbo),
                "turbo_lora": find_lora(turbo_lora) if turbo and turbo_lora not in (None, "None") else None,
                "turbo_strength": float(turbo_strength),
                "turbo_steps": int(turbo_steps),
                "turbo_shift": float(turbo_shift),
                "upscaler": upscaler,
                "upscale_by": float(upscale_by),
            }
        )
