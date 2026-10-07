// MiniMax-H3 (UI Preset: h3): hide the controls that do nothing for video generation
// (Hires. fix / Refiner / CFG / negative prompt / scheduler / img2img resize & denoising / image ControlNet ...)
// the elements with stable ids are hidden by style.css (body.forge-preset-h3), the rest by label here

function h3PresetChanged(preset) {
    const h3 = preset === "h3";
    document.body.classList.toggle("forge-preset-h3", h3);

    const labels = ["Soft inpainting"];
    for (const tab of ["tab_txt2img", "tab_img2img"]) {
        const root = gradioApp().getElementById(tab);
        if (!root) continue;
        for (const label of labels) {
            for (const el of root.querySelectorAll("span, button, label")) {
                if ((el.textContent || "").trim() !== label) continue;
                const block = el.closest(".gradio-accordion, .input-accordion, .block");
                if (block) block.classList.toggle("h3-hidden", h3);
            }
        }
    }
    return [];
}
