def register(options_templates, options_section, OptionInfo):
    from modules.ui_components import FormColorPicker

    options_templates.update(
        options_section(
            (None, "Forge Hidden Options"),
            {
                "VERSION_UID": OptionInfo(None, "internal version for breaking-changes"),
                "forge_preset": OptionInfo("sd"),
                "forge_additional_modules": OptionInfo([]),
                "forge_unet_storage_dtype": OptionInfo("Automatic"),
            },
        )
    )
    options_templates.update(
        options_section(
            ("forge_self_heal", "Self-Heal", "sd"),
            {
                "forge_self_heal": OptionInfo(True, "Self-heal NaN / Inf (black or garbled images)").info("check every sampling step; on overflow retry with fp32 attention, then with a reloaded model; retry the VAE decode in fp32. Recorded as \"Self-heal\" in the generation parameters"),
            },
        )
    )
    options_templates.update(
        options_section(
            ("ui_forgecanvas", "Forge Canvas", "ui"),
            {
                "forge_canvas_height": OptionInfo(512, "Canvas Height").info("in pixels").needs_reload_ui(),
                "forge_canvas_toolbar_always": OptionInfo(False, "Always Visible Toolbar").info("disabled: toolbar only appears when hovering the canvas").needs_reload_ui(),
                "forge_canvas_consistent_brush": OptionInfo(False, "Fixed Brush Size").info("disabled: the brush size is <b>pixel-space</b>, the brush stays small when zoomed out ; enabled: the brush size is <b>canvas-space</b>, the brush stays big when zoomed in").needs_reload_ui(),
                "forge_canvas_plain": OptionInfo(False, "Plain Background").info("disabled: checkerboard pattern ; enabled: solid color").needs_reload_ui(),
                "forge_canvas_plain_color": OptionInfo("#808080", "Solid Color for Plain Background", FormColorPicker, {}).needs_reload_ui(),
            },
        )
    )
