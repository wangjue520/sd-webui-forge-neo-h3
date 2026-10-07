<h1 align="center">Stable Diffusion WebUI Forge Neo - H3</h1>

<p align="center"><b>在 Forge Neo 里运行 MiniMax-H3：文生 / 图生 / 首尾帧 / 参考生视频，原生立体声音频</b><br>
<sup>Run <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax-H3</a> video generation (with native stereo audio) inside Stable Diffusion WebUI Forge Neo</sup></p>

<p align="center">中文 | <a href="#english">English</a> | <a href="README_Forge_Neo.md">Forge Neo 原版说明 / original README</a></p>

---

本仓库基于 [Haoming02/sd-webui-forge-classic](https://github.com/Haoming02/sd-webui-forge-classic)（Forge **Neo** 分支），在 `minimax-h3` 分支上加入了 **MiniMax-H3**（33B 音视频联合生成模型）的完整支持，并针对消费级显卡（12GB 起）做了显存 / 内存优化。Forge Neo 原有的全部功能保持不变，切换 UI Preset 即可在出图和出视频之间切换。

## ✨ 功能

### 生成模式
| 功能 | 入口 | 模型 |
|---|---|---|
| 文生视频 | txt2img | FL2VA |
| 图生视频（首帧） | img2video → 图生视频 | FL2VA |
| 首尾帧生视频（也可只给尾帧） | img2video → 首尾帧图生视频 | FL2VA |
| 参考图生视频（1–9 张） | 参考图 / Reference 标签 | Ref2VA |
| 参考视频（含原声）/ 参考音频 | 参考图 / Reference 标签 | Ref2VA |
| 中间帧引导（任意时间点的画面锚点） | MiniMax-H3 Video → 中间帧引导 | 均可 |
| ControlNet（上传普通视频，自动预处理） | MiniMax-H3 Video → ControlNet 控制 | FL2VA |
| 人物替换 / 动作迁移（角色图 + 舞蹈视频 → 角色跳同样的舞） | 参考图标签放角色图 + ControlNet 控制放视频（DWPose） | Ref2VA |
| 视频局部重绘（遮罩 + 源视频） | MiniMax-H3 Video → ControlNet 控制 | FL2VA |
| 长视频（多段无缝续写、分段提示词、关键帧） | 分段数 + MiniMax-H3 Video → 长视频关键帧 | FL2VA |

所有模式都会同时生成 **32 kHz 立体声音频**，直接封装进 mp4。

### 更多
- **长视频无缝续写**：每段把上一段最后 39 帧（含音频）作为冻结的开头继续生成，动作和声音自然衔接，并按重渲染的 39 帧自动校色；模型自行切镜头时自动检测并换种子重生成（思路来自 [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes) 的 H3 Extension）
- **ControlNet 预处理**：DWPose / OpenPose 姿态、Depth Anything V2 / MiDaS 深度、Canny / Lineart / 动漫线稿、软边缘、涂鸦，逐帧自动提取，并另存提取结果便于检查
- **采样器**：Res Multistep（默认）/ Euler / DPM++ 2M，视频与音频各自按自己的噪声调度（shift 12 / 3）
- **Turbo 加速**：lightx2v Turbo LoRA（8 步），也支持在提示词中用 `<lora:名字:权重>` 加载 H3 LoRA
- **实时预览**：生成过程中预览开头 / 中间 / 结尾三帧
- **视频放大**：生成后用 ESRGAN / Lanczos 等逐帧放大
- **界面**：选择 `h3` 预设后，img2img 变为 img2video，并自动隐藏对视频无效的控件（Hires、Refiner、CFG、反向提示词、重绘幅度等），切回其他预设自动恢复
- **溢出自愈**（出图和出视频都有效）：逐步检测 NaN / Inf，自动以 fp32 注意力、重新加载模型、fp32 VAE 重试，避免黑图 / 鬼图；显存不足时自动降级重试
- **显存 / 内存管理**：各大组件轮流使用显卡、扣除其他程序占用的显存、显存上限保护，64GB 内存可稳定运行

## 📦 模型

| 类型 | 文件 | 放到 |
|---|---|---|
| 主模型（文生 / 图生 / 首尾帧 / ControlNet） | [`minimax_h3_fl2va_pruned-Q4_K.gguf`](https://huggingface.co/unsloth/MiniMax-H3-GGUF) | `models/Stable-diffusion` |
| 主模型（参考生视频） | [`minimax_h3_ref2va_pruned-Q4_K.gguf`](https://huggingface.co/unsloth/MiniMax-H3-GGUF) | `models/Stable-diffusion` |
| 文本编码器（Qwen3-VL-32B） | [`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`](https://huggingface.co/unsloth/MiniMax-H3-GGUF) | `models/text_encoder` |
| 视频 VAE | [`minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/VAE` |
| 音频 VAE | [`minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) | `models/VAE` |
| Turbo LoRA（可选） | [`minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16` / `ref2v`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | `models/Lora` |
| ControlNet（可选） | [`minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/model_patches) | `models/model_patches` |

- 其他量化（Q2–Q8）同样可用；LoRA 请使用 **ComfyUI 格式**（文件名带 `comfyui`）
- 官方 int8 / bf16 safetensors 也能加载，但需要更多内存（int8 一套约 45GB 内存）

**配置参考**：GGUF Q4 一套约 33GB 内存；12GB 显卡可运行（显存不足的部分自动从内存流式加载），24GB 显卡更流畅。

## 🚀 使用

1. 安装方式与 Forge Neo 相同，见 [原版说明 / Installation](README_Forge_Neo.md#installation)，克隆时使用本仓库的 `minimax-h3` 分支
2. 顶部 **UI Preset** 选 `h3`
3. **Checkpoint** 选 FL2VA（参考生视频选 Ref2VA）
4. **VAE / Text Encoder** 同时选上：文本编码器 + 视频 VAE + 音频 VAE
5. 设置宽高（默认 960×576）、**视频时长**、**分段数**，点 Generate
6. 推荐：MiniMax-H3 Video → Turbo 加速 → 勾选并选择对应的 Turbo LoRA（8 步 / Shift 6）

参考速度（RTX 3090，GGUF Q4，Turbo 8 步）：640×352、1.6 秒约 30 秒；448×576、2.3 秒约 1 分钟。

## ⚠️ 已知限制
- Schedule Type 对 H3 无效（H3 使用自己的噪声调度）
- 中间帧引导与 Turbo LoRA 同时使用时过渡容易生硬，建议关闭 Turbo
- 长视频每段仍可能自行转场 / 转镜头，用「分段提示词」描述每段内容可以约束；续写段比首段多生成 39 帧，每段约慢 25%
- 视频局部重绘需要遮罩完整覆盖要替换的主体
- 实时预览为潜空间线性近似
- 目前主要在 RTX 3090 上以 2–3 秒、≤960×576 测试

## 🙏 致谢与许可
- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic) by **Haoming02**，[Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) by **lllyasviel**，[Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) by **AUTOMATIC1111**
- 长视频续写的接续 / 校色方法参考 [ComfyUI-MAINodes](https://github.com/matlowai/ComfyUI-MAINodes)（matlowai）
- [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) by **MiniMax**；模型实现移植自 [ComfyUI](https://github.com/Comfy-Org/ComfyUI) 与 [diffusers](https://github.com/huggingface/diffusers)
- GGUF 量化：[unsloth](https://huggingface.co/unsloth/MiniMax-H3-GGUF)；Turbo LoRA：[lightx2v](https://huggingface.co/lightx2v/Minimax-h3-Turbo)；ControlNet：[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3)
- 代码沿用原项目的 **AGPL-3.0** 许可（见 [LICENSE](LICENSE)）。MiniMax-H3 模型权重适用 **MiniMax H3 Community License**（含地区与商用限制），使用前请自行确认

---

<a id="english"></a>
## English

This repository is a fork of [Forge **Neo**](https://github.com/Haoming02/sd-webui-forge-classic) that adds full **MiniMax-H3** (33B joint audio-video model) support on the `minimax-h3` branch, optimized for consumer GPUs (12 GB and up). Everything in Forge Neo keeps working; switch the **UI Preset** to `h3` for video.

**Features**
- Text-to-video, image-to-video, first/last-frame-to-video, reference-to-video (images, videos with soundtrack, audio), frame guides at any time, long videos (each segment continues the last 39 frames + audio of the previous one, with colour matching and automatic shot-cut retry), all with native **32 kHz stereo audio**
- Character replacement / motion transfer: Ref2VA + a character image + DWPose ControlNet from an ordinary dance video
- Fun ControlNet-Union from an **ordinary video** with automatic per-frame preprocessing (DWPose / OpenPose / depth / canny / lineart ...), and masked video inpainting
- Samplers Res Multistep / Euler / DPM++ 2M on separate video and audio schedules; Turbo LoRA (8 steps); live preview; per-frame upscaling
- A video-focused UI for the `h3` preset (img2img becomes img2video; irrelevant controls are hidden)
- NaN / Inf and out-of-memory **self-healing** (also for image generation), and VRAM / RAM management that runs the GGUF Q4 set on 64 GB RAM

**Models**: see the table above (GGUF from unsloth, VAEs / ControlNet from Comfy-Org, Turbo LoRA from lightx2v; use ComfyUI-format LoRAs).
**Usage**: install like Forge Neo ([Installation](README_Forge_Neo.md#installation)) using the `minimax-h3` branch, select the `h3` preset, the FL2VA (or Ref2VA) checkpoint, and the text encoder + video VAE + audio VAE, then generate.

**License**: AGPL-3.0 as the upstream project. MiniMax-H3 weights are under the MiniMax H3 Community License (regional / commercial restrictions apply).
