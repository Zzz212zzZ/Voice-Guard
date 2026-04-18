import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from modules.attack.clone import VoiceCloner


def _plot_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(np.linspace(0, len(y) / sr, len(y)), y, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    fig.tight_layout()
    return fig


def _plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(5, 3))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def build_attack_tab(cloner: VoiceCloner):
    """Build the Attack tab UI. Must be called inside a gr.Tab context.

    Returns:
        (output_audio, send_to_shield_btn) for cross-tab wiring.
    """

    gr.Markdown(
        "Upload a reference voice (5-30 seconds), type text, "
        "and generate a cloned voice speaking your text."
    )

    with gr.Row():
        with gr.Column(scale=1):
            reference_audio = gr.Audio(
                label="Reference Audio (target voice to clone)",
                type="filepath",
                value=cloner.default_reference,
            )
            text_input = gr.Textbox(
                label="Text to Speak",
                placeholder="Type the text you want the cloned voice to say...",
                lines=3,
                max_lines=6,
                value="I have an emergency! Please help me right now!",
            )
            speed_slider = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                label="Speed",
            )
            clone_button = gr.Button("Clone & Speak", variant="primary")

        with gr.Column(scale=1):
            status_text = gr.Textbox(label="Status", interactive=False)
            output_audio = gr.Audio(label="Cloned Audio", type="filepath")
            similarity_text = gr.Textbox(label="Similarity Score", interactive=False)

    # Side-by-side comparison
    gr.Markdown("### Side-by-Side Comparison")
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Original (Reference)**")
            ref_waveform = gr.Plot(label="Waveform")
            ref_spectrogram = gr.Plot(label="Spectrogram")
        with gr.Column():
            gr.Markdown("**Cloned**")
            clone_waveform = gr.Plot(label="Waveform")
            clone_spectrogram = gr.Plot(label="Spectrogram")

    send_to_shield_btn = gr.Button("Send to Shield →", variant="secondary")

    def on_clone(text, ref_audio, speed):
        try:
            if not text or len(text.strip()) < 2:
                return ("Error: Please enter at least 2 characters.", None, "",
                        None, None, None, None)
            ref = ref_audio if ref_audio else None
            output_path, sr = cloner.clone(
                text=text, reference_audio=ref, speed=speed,
            )
            ref_path = ref if ref else cloner.default_reference

            # Compute similarity
            sim = cloner.compute_similarity(ref_path, output_path)
            sim_str = f"{sim * 100:.1f}%"

            # Generate plots
            ref_wf = _plot_waveform(ref_path)
            clone_wf = _plot_waveform(output_path)
            ref_sp = _plot_spectrogram(ref_path)
            clone_sp = _plot_spectrogram(output_path)

            return (f"Voice cloned successfully. Similarity: {sim_str}",
                    output_path, sim_str,
                    ref_wf, ref_sp, clone_wf, clone_sp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f"Error: {e}", None, "",
                    None, None, None, None)

    clone_button.click(
        fn=on_clone,
        inputs=[text_input, reference_audio, speed_slider],
        outputs=[status_text, output_audio, similarity_text,
                 ref_waveform, ref_spectrogram, clone_waveform, clone_spectrogram],
    )

    return output_audio, send_to_shield_btn
