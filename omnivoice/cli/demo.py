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
    "Gender ": ["Male ", "Female "],
    "Age ": [
        "Child ",
        "Teenager ",
        "Young Adult ",
        "Middle-aged ",
        "Elderly ",
    ],
    "Pitch ": [
        "Very Low Pitch ",
        "Low Pitch ",
        "Moderate Pitch ",
        "High Pitch ",
        "Very High Pitch ",
    ],
    "Style ": ["Whisper "],
    "English Accent ": [
        "American Accent",
        "Australian Accent ",
        "British Accent ",
        "Chinese Accent ",
        "Canadian Accent ",
        "Indian Accent ",
        "Korean Accent ",
        "Portuguese Accent ",
        "Russian Accent ",
        "Japanese Accent ",
    ],
    "Chinese Dialect ": [
        "Henan Dialect ",
        "Shaanxi Dialect ",
        "Sichuan Dialect ",
        "Guizhou Dialect ",
        "Yunnan Dialect",
        "Guilin Dialect ",
        "Jinan Dialect ",
        "Shijiazhuang Dialect ",
        "Gansu Dialect ",
        "Ningxia Dialect ",
        "Qingdao Dialect ",
        "Northeast Dialect ",
    ],
}

_ATTR_INFO = {
    "English Accent ": "Only effective for English speech.",
    "Chinese Dialect ": "Only effective for Chinese speech.",
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

    # Cache VoiceClonePrompt theo tên clone để tránh chạy lại ASR + tokenize
    # mỗi lần nhấn Generate với cùng một giọng nói mẫu.
    _voice_prompt_cache: dict = {}

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
            num_step=int(num_step or 16),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
            audio_chunk_duration=12.0,   # chunk ngắn hơn → mỗi forward pass nhẹ hơn
            audio_chunk_threshold=20.0,  # chia chunk sớm hơn → tránh sequence dài
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
            # Dùng cache key kết hợp path + ref_text để tránh tạo lại
            cache_key = (ref_audio, ref_text)
            if cache_key not in _voice_prompt_cache:
                _voice_prompt_cache[cache_key] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            kw["voice_clone_prompt"] = _voice_prompt_cache[cache_key]

        if mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

        try:
            import time
            t0 = time.time()
            audio = model.generate(**kw)
            t1 = time.time()
        except Exception as e:
            return None, f"Lỗi: {type(e).__name__}: {e}"

        import torchaudio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio[0], sampling_rate)
        return tmp_file.name, f"Thành công! Thời gian xử lý model: {t1 - t0:.1f} giây."

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
    def _lang_dropdown(label="Ngôn ngữ (Tuỳ chọn)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Để Auto để tự động phát hiện ngôn ngữ.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Cài đặt sinh giọng (Tuỳ chọn)", open=False):
            sp = gr.Slider(
                0.7,
                1.3,
                value=1.0,
                step=0.05,
                label="Tốc độ",
                info="1.0 = bình thường. >1 nhanh hơn, <1 chậm hơn. Sẽ bị bỏ qua nếu thiết lập Thời lượng.",
            )
            du = gr.Number(
                value=None,
                label="Thời lượng (giây)",
                info=(
                    "Để trống để sử dụng Tốc độ. "
                    "Đặt thời lượng cố định để ghi đè Tốc độ."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Số bước (Inference Steps)",
                info="Mặc định: 32. Có thể giảm xuống 16 để gen nhanh hơn (chất lượng giảm nhẹ).",
            )
            dn = gr.Checkbox(
                label="Khử nhiễu",
                value=True,
                info="Mặc định: bật. Bỏ chọn để tắt khử nhiễu.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Mức độ hướng dẫn (CFG)",
                info="Mặc định: 2.0.",
            )
            pp = gr.Checkbox(
                label="Tiền xử lý mẫu giọng",
                value=True,
                info="áp dụng loại bỏ khoảng lặng và cắt tỉa cho âm thanh mẫu, thêm dấu câu ở cuối văn bản mẫu (nếu chưa có)",
            )
            po = gr.Checkbox(
                label="Hậu xử lý đầu ra",
                value=True,
                info="Loại bỏ các khoảng lặng dài khỏi âm thanh được sinh ra.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """UI Demo"""
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
                            label="Văn bản cần tổng hợp",
                            lines=4,
                            placeholder="Nhập văn bản bạn muốn tổng hợp...",
                        )
                        vc_clone_dropdown = gr.Dropdown(
                            label="Chọn Voice Clone",
                            choices=get_voice_clone_choices(),
                            interactive=True,
                        )
                        vc_refresh_clones_btn = gr.Button("Tải lại danh sách Voice Clone")
                        
                        vc_lang = _lang_dropdown("Ngôn ngữ (Tuỳ chọn)")
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        with gr.Row():
                            vc_add_btn = gr.Button("Thêm vào DS chờ", variant="secondary")
                            vc_start_btn = gr.Button("Bắt đầu xử lý", variant="primary")
                            vc_clear_btn = gr.Button("Xóa DS", variant="stop")
                    with gr.Column(scale=1):
                        vc_queue_state = gr.State([])
                        vc_queue_df = gr.Dataframe(
                            headers=["STT", "Văn bản (rút gọn)", "Voice Clone", "Trạng thái"],
                            datatype=["number", "str", "str", "str"],
                            label="Danh sách chờ xử lý",
                            interactive=False,
                        )
                        vc_audio = gr.Audio(
                            label="Âm thanh đầu ra (File mới nhất)",
                            type="filepath",
                        )
                        vc_status = gr.Textbox(label="Trạng thái ", lines=2)

                def _refresh_choices():
                    return gr.update(choices=get_voice_clone_choices())
                
                vc_refresh_clones_btn.click(fn=_refresh_choices, inputs=[], outputs=[vc_clone_dropdown])
                
                def get_queue_df(q_state):
                    rows = []
                    for i, q in enumerate(q_state):
                        short_text = q["text"][:30] + "..." if len(q["text"]) > 30 else q["text"]
                        rows.append([i+1, short_text, q["clone_name"], q["status"]])
                    if not rows:
                        return pd.DataFrame(columns=["STT", "Văn bản (rút gọn)", "Voice Clone", "Trạng thái"])
                    return pd.DataFrame(rows, columns=["STT", "Văn bản (rút gọn)", "Voice Clone", "Trạng thái"])

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
                    ref_text = clone_row.iloc[0]["Ref Text"] or None
                    
                    history_id = db.add_history(text, clone_name, "Processing")

                    # --- Cache VoiceClonePrompt theo tên clone ---
                    # Lần đầu: download file âm thanh mẫu và tạo prompt (chạy ASR nếu cần).
                    # Các lần sau: dùng lại prompt đã cache, bỏ qua download + ASR.
                    if clone_name not in _voice_prompt_cache:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
                            ref_audio_path = tmp_ref.name
                        success = db.download_audio(audio_drive_id, ref_audio_path)
                        if not success:
                            db.update_history_status(history_id, "Failed (Cannot download ref audio)")
                            return None, "Failed to download reference audio from Drive."
                        try:
                            _voice_prompt_cache[clone_name] = model.create_voice_clone_prompt(
                                ref_audio=ref_audio_path,
                                ref_text=ref_text,
                            )
                        except Exception as e:
                            db.update_history_status(history_id, f"Failed (create_voice_clone_prompt: {e})")
                            return None, f"Error building voice prompt: {e}"

                    voice_prompt = _voice_prompt_cache[clone_name]

                    # --- Gen audio dùng cached voice prompt ---
                    gen_config = OmniVoiceGenerationConfig(
                        num_step=int(ns or 16),
                        guidance_scale=float(gs) if gs is not None else 2.0,
                        denoise=bool(dn) if dn is not None else True,
                        preprocess_prompt=bool(pp),
                        postprocess_output=bool(po),
                        audio_chunk_duration=12.0,
                        audio_chunk_threshold=20.0,
                    )
                    lang = lang if (lang and lang != "Auto") else None
                    kw: Dict[str, Any] = dict(
                        text=text.strip(),
                        language=lang,
                        generation_config=gen_config,
                        voice_clone_prompt=voice_prompt,
                    )
                    if sp is not None and float(sp) != 1.0:
                        kw["speed"] = float(sp)
                    if du is not None and float(du) > 0:
                        kw["duration"] = float(du)

                    try:
                        import time
                        t0 = time.time()
                        import torchaudio as _ta
                        audio = model.generate(**kw)
                        t1 = time.time()
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
                            _ta.save(tmp_out.name, audio[0], sampling_rate)
                        res_audio = tmp_out.name
                        res_status = f"Thành công! Thời gian xử lý model: {t1 - t0:.1f} giây."
                    except Exception as e:
                        res_audio = None
                        res_status = f"Lỗi: {type(e).__name__}: {e}"
                    
                    if res_audio:
                        db.update_history_status(history_id, "Success", res_audio)
                    else:
                        db.update_history_status(history_id, f"Failed ({res_status})")
                        
                    return res_audio, res_status

                def _add_to_queue(text, lang, clone_name, q_state):
                    if not clone_name:
                        return q_state, get_queue_df(q_state), "Vui lòng chọn Voice Clone."
                    if not text or not text.strip():
                        return q_state, get_queue_df(q_state), "Vui lòng nhập văn bản."
                    q_state.append({
                        "text": text,
                        "lang": lang,
                        "clone_name": clone_name,
                        "status": "Chờ xử lý"
                    })
                    return q_state, get_queue_df(q_state), f"Đã thêm vào danh sách (tổng {len(q_state)} task)."

                vc_add_btn.click(
                    _add_to_queue,
                    inputs=[vc_text, vc_lang, vc_clone_dropdown, vc_queue_state],
                    outputs=[vc_queue_state, vc_queue_df, vc_status]
                )

                def _clear_queue():
                    return [], get_queue_df([]), "Đã xóa danh sách."
                
                vc_clear_btn.click(
                    _clear_queue,
                    inputs=[],
                    outputs=[vc_queue_state, vc_queue_df, vc_status]
                )

                def _process_queue(q_state, ns, gs, dn, sp, du, pp, po):
                    if not q_state:
                        yield q_state, get_queue_df(q_state), None, "Danh sách rỗng."
                        return
                    
                    last_audio = None
                    total_tasks = len(q_state)
                    
                    for i, task in enumerate(q_state):
                        if task["status"] in ["Chờ xử lý", "Lỗi"]:
                            task["status"] = "Đang xử lý..."
                            yield q_state, get_queue_df(q_state), last_audio, f"Đang xử lý task {i+1}/{total_tasks}..."
                            
                            res_audio, res_status = _clone_fn(
                                task["text"], task["lang"], task["clone_name"], ns, gs, dn, sp, du, pp, po
                            )
                            
                            if res_audio:
                                task["status"] = "Xong"
                                last_audio = res_audio
                            else:
                                task["status"] = "Lỗi"
                            
                            yield q_state, get_queue_df(q_state), last_audio, f"Hoàn thành task {i+1}/{total_tasks}: {res_status}"

                vc_start_btn.click(
                    _process_queue,
                    inputs=[vc_queue_state, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                    outputs=[vc_queue_state, vc_queue_df, vc_audio, vc_status]
                )

            # ==============================================================
            # Trang quản lý voice clone
            # ==============================================================
            with gr.TabItem("Quản lý Voice Clone"):
                gr.Markdown("### Tạo và Quản lý Voice Clones")
                with gr.Row():
                    with gr.Column(scale=1):
                        new_clone_name = gr.Textbox(label="Tên Voice")
                        new_clone_audio = gr.Audio(label="File ghi âm mẫu", type="filepath")
                        new_clone_text = gr.Textbox(label="Văn bản mẫu (Tuỳ chọn)", lines=2)
                        add_clone_btn = gr.Button("Thêm Voice Clone", variant="primary")
                        add_clone_status = gr.Textbox(label="Trạng thái", interactive=False)
                    
                    with gr.Column(scale=2):
                        clones_df_ui = gr.Dataframe(label="Các Voice Clones hiện có")
                        refresh_clones_btn = gr.Button("Làm mới danh sách")
                        
                        delete_clone_id = gr.Textbox(label="Nhập ID để xóa")
                        delete_clone_btn = gr.Button("Xóa Voice Clone", variant="stop")
                        delete_clone_status = gr.Textbox(label="Trạng thái xóa", interactive=False)

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

                def on_select_clone(evt: gr.SelectData, df):
                    try:
                        if isinstance(df, pd.DataFrame) and not df.empty and evt.index[0] < len(df):
                            return str(df.iloc[evt.index[0]]["ID"])
                    except Exception:
                        pass
                    return ""

                clones_df_ui.select(
                    on_select_clone,
                    inputs=[clones_df_ui],
                    outputs=[delete_clone_id]
                )

                demo.load(fn=load_clones_df, inputs=[], outputs=[clones_df_ui])

            # ==============================================================
            # Trang quản lý lịch sử tạo voice
            # ==============================================================
            with gr.TabItem("Quản lý Lịch sử"):
                gr.Markdown("### Lịch sử tạo voice")
                history_df_ui = gr.Dataframe(label="Bản ghi lịch sử")
                refresh_history_btn = gr.Button("Làm mới lịch sử")
                
                with gr.Row():
                    history_action_id = gr.Textbox(label="ID lịch sử để thao tác")
                with gr.Row():
                    history_download_btn = gr.Button("Tải xuống âm thanh đã tạo", variant="primary")
                    history_delete_btn = gr.Button("Xóa bản ghi", variant="stop")
                    history_retry_btn = gr.Button("Tạo lại", variant="secondary")
                
                history_action_status = gr.Textbox(label="Trạng thái thao tác", interactive=False)
                history_audio_output = gr.Audio(label="Âm thanh đã tải xuống", type="filepath")

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
                    
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
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

                def on_select_history(evt: gr.SelectData, df):
                    try:
                        if isinstance(df, pd.DataFrame) and not df.empty and evt.index[0] < len(df):
                            return str(df.iloc[evt.index[0]]["ID"])
                    except Exception:
                        pass
                    return ""

                history_df_ui.select(
                    on_select_history,
                    inputs=[history_df_ui],
                    outputs=[history_action_id]
                )

                demo.load(fn=load_history_df, inputs=[], outputs=[history_df_ui])

            # ==============================================================
            # Voice Design (Original)
            # ==============================================================
            with gr.TabItem("Thiết kế giọng nói (Voice Design)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Văn bản cần tổng hợp ",
                            lines=4,
                            placeholder="Nhập văn bản bạn muốn tổng hợp...",
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
                        vd_btn = gr.Button("Tạo ", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Âm thanh đầu ra",
                            type="filepath",
                        )
                        vd_status = gr.Textbox(label="Trạng thái ", lines=2)

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

    # --- torch.compile: tăng tốc LLM backbone ~20-40% sau lần đầu warm-up ---
    # Chỉ áp dụng trên CUDA vì MPS/CPU chưa hỗ trợ tốt.
    if device.startswith("cuda"):
        try:
            logging.info("Compiling LLM backbone with torch.compile (mode=reduce-overhead)...")
            model.llm = torch.compile(model.llm, mode="reduce-overhead")
            logging.info("torch.compile applied successfully.")
        except Exception as e:
            logging.warning(f"torch.compile failed (skipping): {e}")

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
