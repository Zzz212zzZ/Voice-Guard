import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from modules.shield.detect import DeepfakeDetector


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


def _plot_attention_heatmap(audio_path, attention_combined):
    """Overlay AASIST attention on mel spectrogram to show where the model focuses."""
    from scipy.ndimage import zoom

    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(5, 3))
    # Base spectrogram
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, alpha=0.7)

    # Resize attention map to match spectrogram dimensions
    att = attention_combined
    # Normalize to [0, 1]
    att = att - att.min()
    if att.max() > 0:
        att = att / att.max()

    # Resize attention to spectrogram shape (n_mels, time_frames)
    zoom_factors = (S_dB.shape[0] / att.shape[0], S_dB.shape[1] / att.shape[1])
    att_resized = zoom(att, zoom_factors, order=1)

    # Overlay attention as semi-transparent heatmap
    extent = [0, S_dB.shape[1], 0, S_dB.shape[0]]
    im = ax.imshow(
        att_resized, aspect="auto", origin="lower", extent=extent,
        cmap="hot", alpha=0.5, interpolation="bilinear",
    )
    ax.set_title("Model Attention Heatmap")
    fig.colorbar(im, ax=ax, label="Attention Intensity")
    fig.tight_layout()
    return fig


def _plot_pitch_contour(f0, voiced_flag, sr, hop_length):
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    fig, ax = plt.subplots(figsize=(5, 3))
    f0_plot = f0.copy().astype(float)
    f0_plot[~voiced_flag] = np.nan
    ax.plot(times, f0_plot, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Pitch Contour (F0)")
    ax.set_ylim(50, 500)
    fig.tight_layout()
    return fig


def _plot_energy(rms, sr, hop_length):
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times, rms, color="tab:green", linewidth=1.0)
    ax.fill_between(times, rms, alpha=0.3, color="tab:green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title("Energy Envelope")
    fig.tight_layout()
    return fig


def _plot_voice_quality(prosody):
    """Horizontal bar chart for voice quality features with values labeled."""
    features = ["Jitter", "Shimmer", "F0 CV", "Energy CV"]
    values = [
        prosody["jitter"],
        prosody["shimmer"],
        prosody["f0_cv"],
        prosody["energy_cv"],
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
    bars = ax.barh(features, values, color=colors, height=0.6)

    # Label each bar with its value
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10,
        )

    ax.set_xlabel("Value")
    ax.set_title("Voice Quality Features")
    ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def _plot_spectral_features(spectral):
    """Bar chart for spectral features (normalized to make them comparable)."""
    features = ["Centroid", "Bandwidth", "Flatness\n(x10000)", "Rolloff"]
    values = [
        spectral["spectral_centroid"],
        spectral["spectral_bandwidth"],
        spectral["spectral_flatness"] * 10000,  # scale up for visibility
        spectral["spectral_rolloff"],
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    bars = ax.bar(features, values, color=colors, width=0.6)

    for bar, val, orig_key in zip(bars, values, ["spectral_centroid", "spectral_bandwidth", "spectral_flatness", "spectral_rolloff"]):
        orig = spectral[orig_key]
        if orig_key == "spectral_flatness":
            label = f"{orig:.4f}"
        else:
            label = f"{orig:.0f} Hz"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            label, ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Value")
    ax.set_title("Spectral Features")
    fig.tight_layout()
    return fig


def build_shield_tab(detector: DeepfakeDetector):
    """Build the Shield tab UI. Must be called inside a gr.Tab context.

    Returns:
        audio_input component (for "Send to Shield" wiring from Attack tab).
    """

    gr.Markdown(
        "Upload an audio file (or receive from Attack tab) to analyze "
        "whether it is real or AI-generated."
    )

    # --- Input + Verdict ---
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Audio to Analyze", type="filepath")
            analyze_button = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            verdict_md = gr.Markdown(value="*Upload audio and click Analyze*")
            spoof_prob_num = gr.Number(
                label="W2V-AASIST Spoof Probability", interactive=False,
            )

    # --- Feature charts ---
    with gr.Row():
        voice_quality_plot = gr.Plot(label="Voice Quality")
        spectral_plot = gr.Plot(label="Spectral Features")

    # --- Detailed values (collapsed) ---
    with gr.Accordion("Raw Feature Values", open=False):
        voice_quality_md = gr.Markdown(value="")
        spectral_md = gr.Markdown(value="")

    # --- Visualizations ---
    gr.Markdown("### Visualizations")
    with gr.Row():
        spectrogram_plot = gr.Plot(label="Spectrogram")
        attention_plot = gr.Plot(label="Attention Heatmap")
    with gr.Row():
        pitch_plot = gr.Plot(label="Pitch Contour")
        energy_plot = gr.Plot(label="Energy")

    def on_analyze(audio_path):
        if not audio_path:
            empty = "*Please upload an audio file first.*"
            return empty, None, None, None, "", "", None, None, None, None

        try:
            result = detector.full_analyze(audio_path)

            # Verdict (W2V-AASIST only)
            v = result["verdict"]
            c = result["confidence"]
            if v == "FAKE":
                verdict_html = f"### Verdict: **FAKE** — Confidence: {c}%"
            elif v == "SUSPICIOUS":
                verdict_html = f"### Verdict: **SUSPICIOUS** — Confidence: {c}%"
            else:
                verdict_html = f"### Verdict: **REAL** — Confidence: {c}%"

            p = result["prosody"]
            s = result["spectral"]

            # Feature charts
            vq_fig = _plot_voice_quality(p)
            sp_fig = _plot_spectral_features(s)

            # Raw value tables (in accordion)
            vq_table = (
                "| Feature | Value | Description |\n"
                "|---------|-------|-------------|\n"
                f"| Jitter | {p['jitter']:.6f} | Pitch period perturbation |\n"
                f"| Shimmer | {p['shimmer']:.6f} | Amplitude perturbation |\n"
                f"| F0 Mean | {p['f0_mean']:.1f} Hz | Average pitch |\n"
                f"| F0 Std | {p['f0_std']:.1f} Hz | Pitch variability |\n"
                f"| F0 CV | {p['f0_cv']:.4f} | Normalized pitch variation |\n"
                f"| Energy CV | {p['energy_cv']:.4f} | Rhythm variability |"
            )
            sp_table = (
                "| Feature | Value | Description |\n"
                "|---------|-------|-------------|\n"
                f"| Centroid | {s['spectral_centroid']:.1f} Hz | Center of mass of spectrum |\n"
                f"| Bandwidth | {s['spectral_bandwidth']:.1f} Hz | Spectral spread |\n"
                f"| Flatness | {s['spectral_flatness']:.6f} | Spectrum uniformity |\n"
                f"| Rolloff | {s['spectral_rolloff']:.1f} Hz | 85% energy cutoff |\n"
                f"| Sample Rate | {s['sample_rate']} Hz | Native sample rate |"
            )

            # Time-domain plots
            spec_fig = _plot_spectrogram(audio_path)
            att_fig = _plot_attention_heatmap(
                audio_path, result["attention"]["combined"],
            )
            pd = result["prosody"]["plot_data"]
            pitch_fig = _plot_pitch_contour(
                pd["f0"], pd["voiced_flag"], pd["sr"], pd["hop_length"],
            )
            energy_fig = _plot_energy(
                pd["rms"], pd["sr"], pd["hop_length"],
            )

            return (verdict_html, result["spoof_prob"],
                    vq_fig, sp_fig,
                    vq_table, sp_table,
                    spec_fig, att_fig, pitch_fig, energy_fig)

        except Exception as e:
            import traceback
            traceback.print_exc()
            err = f"### Error: {e}"
            return err, None, None, None, "", "", None, None, None, None

    analyze_button.click(
        fn=on_analyze,
        inputs=[audio_input],
        outputs=[
            verdict_md, spoof_prob_num,
            voice_quality_plot, spectral_plot,
            voice_quality_md, spectral_md,
            spectrogram_plot, attention_plot, pitch_plot, energy_plot,
        ],
    )

    return audio_input
