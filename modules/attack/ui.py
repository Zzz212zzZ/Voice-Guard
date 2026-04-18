"""Attack tab — voice cloning with OpenVoice v2.

Demo-optimized UI:
  - Default reference audio is demo_speaker0.mp3 (the sample used in the
    live presentation), pre-populated in the upload widget.
  - Three preset scripts (scam / neutral / CEO fraud) quick-fill the text
    box with one click, so the presenter can show multiple attack framings
    without typing on stage.
  - Result card emphasizes the cloned audio + similarity score with
    consistent colors (teal = reference/real, red = cloned/fake) matching
    the slide deck.
  - Prominent "Send to Shield →" hand-off, labelled for the audience.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from modules.attack.clone import VoiceCloner


# ----------------------------------------------------------------------
# Color palette — keep in sync with slides/build_pptx.js
# ----------------------------------------------------------------------
C_REAL    = "#2E8B8B"   # teal — reference / real
C_FAKE    = "#E53E3E"   # red  — cloned / fake
C_GRID    = "#E2E8F0"
C_INK     = "#0F172A"
C_MUTED   = "#64748B"


# ----------------------------------------------------------------------
# Plot helpers (themed + tighter than matplotlib defaults)
# ----------------------------------------------------------------------
def _apply_style(ax, title, color):
    ax.set_title(title, fontsize=12, fontweight="bold", color=C_INK, pad=6)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _plot_waveform(audio_path, color, title):
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(5, 1.8), dpi=110)
    t = np.linspace(0, len(y) / sr, len(y))
    ax.fill_between(t, y, -y, color=color, alpha=0.75, linewidth=0)
    ax.set_xlabel("Time (s)", fontsize=9, color=C_MUTED)
    _apply_style(ax, title, color)
    fig.tight_layout()
    return fig


def _plot_spectrogram(audio_path, color, title):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(5, 2.6), dpi=110)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    _apply_style(ax, title, color)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors=C_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(C_GRID)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Demo script presets — 1-click quick-fill
# ----------------------------------------------------------------------
DEMO_SCRIPTS = {
    "🚨 Family emergency (scam)":
        "I have an emergency! Please help me right now!",
    "💼 CEO fraud (scam)":
        "This is urgent. Please transfer fifty thousand dollars to "
        "account number two four five six immediately.",
    "🦊 Neutral test (pangram)":
        "The quick brown fox jumps over the lazy dog.",
}


def build_attack_tab(cloner: VoiceCloner):
    """Build the Attack tab UI. Must be called inside a gr.Tab context.

    Returns:
        (output_audio, send_to_shield_btn) for cross-tab wiring.
    """

    # ----- Header -------------------------------------------------
    gr.Markdown(
        "### 🗡️  Clone a voice from a short reference clip\n"
        "Upload or select reference audio · pick a script · click **Clone & Speak**. "
        "Then send the result to the **Shield** tab for detection.",
        elem_classes=["vg-section-header"],
    )

    # ----- Main row: inputs (left) + result (right) --------------
    with gr.Row(equal_height=False):

        # ============== LEFT: inputs ==============
        with gr.Column(scale=5):
            gr.Markdown("#### 1 · Reference voice")
            reference_audio = gr.Audio(
                label="Target voice to clone  (3–30 s; default: demo_speaker0)",
                type="filepath",
                value=cloner.default_reference,
            )

            gr.Markdown("#### 2 · What should it say?")
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Type the text you want the cloned voice to say...",
                lines=3,
                max_lines=6,
                value=DEMO_SCRIPTS["🚨 Family emergency (scam)"],
            )

            with gr.Row():
                preset_btns = [
                    gr.Button(label, size="sm", variant="secondary")
                    for label in DEMO_SCRIPTS
                ]

            gr.Markdown("#### 3 · Generate")
            with gr.Row():
                with gr.Column(scale=3):
                    speed_slider = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Speaking speed",
                    )
                with gr.Column(scale=2):
                    clone_button = gr.Button(
                        "🎙️  Clone & Speak",
                        variant="primary",
                        size="lg",
                    )

        # ============== RIGHT: results ==============
        with gr.Column(scale=5):
            gr.Markdown("#### Result")
            status_text = gr.Textbox(
                label="Status",
                value="Ready — click Clone & Speak. Generation takes ~15 s on CPU.",
                interactive=False,
            )
            output_audio = gr.Audio(
                label="🔊 Cloned audio",
                type="filepath",
                interactive=False,
            )
            similarity_text = gr.Textbox(
                label="Similarity to reference (cosine, 0–100%)",
                interactive=False,
            )

            gr.Markdown(
                "↓ Once the clone is ready, hand it off to the detector.",
                elem_classes=["vg-muted-hint"],
            )
            send_to_shield_btn = gr.Button(
                "🛡️  Send to Shield  →",
                variant="primary",
                size="lg",
            )

    # ----- Side-by-side comparison ---------------------------------
    gr.Markdown("### Side-by-side · reference vs. cloned")
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"<div style='color:{C_REAL};font-weight:700;font-size:15px'>"
                f"● Reference  (real human)</div>"
            )
            ref_waveform = gr.Plot(label="Waveform")
            ref_spectrogram = gr.Plot(label="Mel spectrogram")
        with gr.Column():
            gr.Markdown(
                f"<div style='color:{C_FAKE};font-weight:700;font-size:15px'>"
                f"● Cloned  (AI-generated)</div>"
            )
            clone_waveform = gr.Plot(label="Waveform")
            clone_spectrogram = gr.Plot(label="Mel spectrogram")

    # ----- Demo tips (collapsed) -----------------------------------
    with gr.Accordion("📋  Presenter notes (click to expand)", open=False):
        gr.Markdown(
            "- **Reference audio** defaults to `demo_speaker0.mp3` — pre-loaded for the live demo.\n"
            "- **Preset buttons** fill the text box instantly; click one on stage to show a variant "
            "(scam vs. neutral) without typing.\n"
            "- **Clone & Speak** takes ~15 s on CPU (~3 s on GPU). The status field shows a hint "
            "so the audience knows to wait.\n"
            "- **Similarity score** ≥ 80 % is considered a recognizable match.\n"
            "- **Send to Shield →** passes the cloned `.wav` into the detector tab; expect "
            "a `FAKE` verdict at ~99.97 % confidence."
        )

    # ----- Wiring --------------------------------------------------
    # Preset buttons: capture the label by default-arg closure so each
    # button writes its own script into the text box.
    for btn, label in zip(preset_btns, DEMO_SCRIPTS):
        btn.click(fn=lambda l=label: DEMO_SCRIPTS[l], outputs=[text_input])

    def on_clone(text, ref_audio, speed):
        try:
            if not text or len(text.strip()) < 2:
                return ("⚠️  Please enter at least 2 characters.", None, "",
                        None, None, None, None)
            ref = ref_audio if ref_audio else None
            output_path, sr = cloner.clone(
                text=text, reference_audio=ref, speed=speed,
            )
            ref_path = ref if ref else cloner.default_reference

            sim = cloner.compute_similarity(ref_path, output_path)
            sim_str = f"{sim * 100:.1f}%"

            ref_wf = _plot_waveform(ref_path, C_REAL, "Reference waveform")
            clone_wf = _plot_waveform(output_path, C_FAKE, "Cloned waveform")
            ref_sp = _plot_spectrogram(ref_path, C_REAL, "Reference spectrogram")
            clone_sp = _plot_spectrogram(output_path, C_FAKE, "Cloned spectrogram")

            return (f"✅  Cloned. Similarity {sim_str}. Ready to hand off to Shield.",
                    output_path, sim_str,
                    ref_wf, ref_sp, clone_wf, clone_sp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f"❌  Error: {e}", None, "",
                    None, None, None, None)

    clone_button.click(
        fn=on_clone,
        inputs=[text_input, reference_audio, speed_slider],
        outputs=[status_text, output_audio, similarity_text,
                 ref_waveform, ref_spectrogram, clone_waveform, clone_spectrogram],
    )

    return output_audio, send_to_shield_btn
