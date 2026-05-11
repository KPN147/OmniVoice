#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
from typing import Any, Dict
import tempfile
import soundfile as sf
import pandas as pd

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from omnivoice.cli.db import OmniVoiceDB


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate
    db = OmniVoiceDB()

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Please upload a reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = audio[0].squeeze(0).numpy()  # (T,)
        waveform = (waveform * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.7,
                1.3,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi Next-gen Kaldi team.
"""
        )

        def get_voice_clone_choices():
            if not db.initialized:
                return []
            df = db.get_voice_clones()
            if df.empty:
                return []
            return df["Name"].tolist()

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Trang chủ (Tạo Voice)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_clone_dropdown = gr.Dropdown(
                            label="Choose Voice Clone / Chọn Voice Clone",
                            choices=get_voice_clone_choices(),
                            interactive=True,
                        )
                        vc_refresh_clones_btn = gr.Button("Refresh Voice Clones / Tải lại danh sách")
                        
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _refresh_choices():
                    return gr.update(choices=get_voice_clone_choices())
                
                vc_refresh_clones_btn.click(fn=_refresh_choices, inputs=[], outputs=[vc_clone_dropdown])

                def _clone_fn(
                    text, lang, clone_name, ns, gs, dn, sp, du, pp, po
                ):
                    if not db.initialized:
                        return None, "Database not initialized. Check credentials."
                    if not clone_name:
                        return None, "Please select a Voice Clone."
                    if not text or not text.strip():
                        return None, "Please enter text."
                    
                    df = db.get_voice_clones()
                    clone_row = df[df["Name"] == clone_name]
                    if clone_row.empty:
                        return None, "Selected Voice Clone not found in DB."
                    
                    audio_drive_id = clone_row.iloc[0]["Ref Audio Drive ID"]
                    ref_text = clone_row.iloc[0]["Ref Text"]
                    
                    history_id = db.add_history(text, clone_name, "Processing")
                    
                    # Download ref audio to temp
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
                        ref_audio_path = tmp_ref.name
                    
                    success = db.download_audio(audio_drive_id, ref_audio_path)
                    if not success:
                        db.update_history_status(history_id, "Failed (Cannot download ref audio)")
                        return None, "Failed to download reference audio from Drive."
                    
                    res_audio, res_status = _gen(
                        text,
                        lang,
                        ref_audio_path,
                        None,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )
                    
                    if res_audio:
                        # res_audio is (sampling_rate, waveform)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                            sf.write(tmp_out.name, res_audio[1], res_audio[0])
                            db.update_history_status(history_id, "Success", tmp_out.name)
                    else:
                        db.update_history_status(history_id, f"Failed ({res_status})")
                        
                    return res_audio, res_status

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_clone_dropdown,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # Trang quản lý voice clone
            # ==============================================================
            with gr.TabItem("Quản lý Voice Clone"):
                gr.Markdown("### Create and Manage Voice Clones")
                with gr.Row():
                    with gr.Column(scale=1):
                        new_clone_name = gr.Textbox(label="Clone Name / Tên Voice")
                        new_clone_audio = gr.Audio(label="Reference Audio / File ghi âm", type="filepath")
                        new_clone_text = gr.Textbox(label="Reference Text / Văn bản mẫu (Optional)", lines=2)
                        add_clone_btn = gr.Button("Add Voice Clone / Thêm Voice", variant="primary")
                        add_clone_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=2):
                        clones_df_ui = gr.Dataframe(label="Existing Voice Clones")
                        refresh_clones_btn = gr.Button("Refresh List")
                        
                        delete_clone_id = gr.Textbox(label="Enter ID to delete / Nhập ID để xóa")
                        delete_clone_btn = gr.Button("Delete Voice Clone", variant="stop")
                        delete_clone_status = gr.Textbox(label="Delete Status", interactive=False)

                def load_clones_df():
                    if not db.initialized:
                        return pd.DataFrame()
                    return db.get_voice_clones()

                def handle_add_clone(name, audio_path, text):
                    if not db.initialized:
                        return "DB not initialized.", load_clones_df()
                    if not name or not audio_path:
                        return "Name and Audio are required.", load_clones_df()
                    db.add_voice_clone(name, audio_path, text)
                    return "Success", load_clones_df()
                
                def handle_delete_clone(clone_id):
                    if not db.initialized:
                        return "DB not initialized.", load_clones_df()
                    if db.delete_voice_clone(clone_id):
                        return "Deleted successfully", load_clones_df()
                    return "ID not found or failed to delete", load_clones_df()

                add_clone_btn.click(
                    handle_add_clone,
                    inputs=[new_clone_name, new_clone_audio, new_clone_text],
                    outputs=[add_clone_status, clones_df_ui]
                )
                refresh_clones_btn.click(fn=load_clones_df, inputs=[], outputs=[clones_df_ui])
                delete_clone_btn.click(
                    handle_delete_clone,
                    inputs=[delete_clone_id],
                    outputs=[delete_clone_status, clones_df_ui]
                )
                demo.load(fn=load_clones_df, inputs=[], outputs=[clones_df_ui])

            # ==============================================================
            # Trang quản lý lịch sử tạo voice
            # ==============================================================
            with gr.TabItem("Quản lý Lịch sử"):
                gr.Markdown("### Generation History / Lịch sử tạo voice")
                history_df_ui = gr.Dataframe(label="History Records")
                refresh_history_btn = gr.Button("Refresh History")
                
                with gr.Row():
                    history_action_id = gr.Textbox(label="History ID for Action")
                with gr.Row():
                    history_download_btn = gr.Button("Download Generated Audio", variant="primary")
                    history_delete_btn = gr.Button("Delete Record", variant="stop")
                    history_retry_btn = gr.Button("Regenerate", variant="secondary")
                
                history_action_status = gr.Textbox(label="Action Status", interactive=False)
                history_audio_output = gr.Audio(label="Downloaded Audio", type="filepath")

                def load_history_df():
                    if not db.initialized:
                        return pd.DataFrame()
                    return db.get_history()

                def handle_download_history(hist_id):
                    if not db.initialized:
                        return "DB not initialized.", None
                    df = db.get_history()
                    row = df[df["ID"] == hist_id]
                    if row.empty:
                        return "History ID not found.", None
                    audio_id = row.iloc[0].get("Output Audio Drive ID")
                    if not audio_id:
                        return "No audio generated for this record.", None
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                        output_path = tmp_out.name
                    success = db.download_audio(audio_id, output_path)
                    if success:
                        return "Downloaded successfully.", output_path
                    return "Failed to download.", None

                def handle_delete_history(hist_id):
                    if not db.initialized:
                        return "DB not initialized.", load_history_df()
                    if db.delete_history(hist_id):
                        return "Deleted successfully", load_history_df()
                    return "ID not found or failed to delete", load_history_df()

                def handle_regenerate_history(hist_id):
                    if not db.initialized:
                        return "DB not initialized.", load_history_df()
                    df = db.get_history()
                    row = df[df["ID"] == hist_id]
                    if row.empty:
                        return "History ID not found.", load_history_df()
                    
                    text = row.iloc[0]["Text"]
                    clone_name = row.iloc[0]["Voice Clone Used"]
                    
                    res_audio, res_status = _clone_fn(
                        text, "Auto", clone_name, 32, 2.0, True, 1.0, None, True, True
                    )
                    if res_audio:
                        return "Regenerated successfully", load_history_df()
                    else:
                        return f"Regeneration failed: {res_status}", load_history_df()

                refresh_history_btn.click(fn=load_history_df, inputs=[], outputs=[history_df_ui])
                history_download_btn.click(
                    handle_download_history,
                    inputs=[history_action_id],
                    outputs=[history_action_status, history_audio_output]
                )
                history_delete_btn.click(
                    handle_delete_history,
                    inputs=[history_action_id],
                    outputs=[history_action_status, history_df_ui]
                )
                history_retry_btn.click(
                    handle_regenerate_history,
                    inputs=[history_action_id],
                    outputs=[history_action_status, history_df_ui]
                )
                demo.load(fn=load_history_df, inputs=[], outputs=[history_df_ui])

            # ==============================================================
            # Voice Design (Original)
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=True,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
