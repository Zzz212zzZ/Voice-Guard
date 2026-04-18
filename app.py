import argparse
import os
import sys

# Ensure the OpenVoice package is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OPENVOICE_ROOT = os.path.join(PROJECT_ROOT, "OpenVoice")
SSL_ROOT = os.path.join(PROJECT_ROOT, "SSL_Anti-spoofing")
CHECKPOINTS_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")
sys.path.insert(0, OPENVOICE_ROOT)

import gradio as gr

from modules.attack.clone import VoiceCloner
from modules.attack.ui import build_attack_tab
from modules.shield.detect import DeepfakeDetector
from modules.shield.ui import build_shield_tab


# ----------------------------------------------------------------------
# Theme — consistent with slides/build_pptx.js
# ----------------------------------------------------------------------
# Palette:  Navy #1E2761 (primary)  ·  Teal #2E8B8B (safe)  ·
#           Red #E53E3E (fake)  ·  Orange #DD6B20 (accent)
VG_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EBEEF7", c100="#CCD4E8", c200="#9AA7D1",
        c300="#687AB9", c400="#3A4E8F", c500="#1E2761",
        c600="#192152", c700="#151B42", c800="#101533", c900="#0B0F24",
        c950="#070A1A",
    ),
    secondary_hue=gr.themes.Color(
        c50="#ECF5F5", c100="#C8E4E4", c200="#A1D1D1",
        c300="#7ABFBE", c400="#54ADAB", c500="#2E8B8B",
        c600="#267373", c700="#1E5B5B", c800="#164344", c900="#0E2B2C",
        c950="#081718",
    ),
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    text_size=gr.themes.sizes.text_md,
    spacing_size=gr.themes.sizes.spacing_md,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#F8FAFC",
    body_text_color="#0F172A",
    button_primary_background_fill="#1E2761",
    button_primary_background_fill_hover="#2C3E80",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#E2E8F0",
    button_secondary_background_fill_hover="#CBD5E1",
    button_secondary_text_color="#1E2761",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E8F0",
    block_title_text_color="#1E2761",
    block_title_text_weight="600",
    block_label_text_color="#64748B",
    input_background_fill="#FFFFFF",
    input_border_color="#CBD5E1",
    input_border_color_focus="#1E2761",
)


VG_CSS = """
/* App shell */
.gradio-container { max-width: 1400px !important; }
footer { display: none !important; }

/* Global headline */
#vg-app-header {
    background: linear-gradient(135deg, #1E2761 0%, #2C3E80 100%);
    color: #FFFFFF;
    padding: 22px 28px 20px;
    border-radius: 6px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
}
#vg-app-header h1 {
    font-family: 'Cambria', Georgia, serif;
    font-weight: 700;
    font-size: 30px;
    letter-spacing: -0.01em;
    margin: 0 0 6px 0;
    color: #FFFFFF;
}
#vg-app-header p {
    color: #CADCFC;
    font-size: 14px;
    line-height: 1.5;
    margin: 0;
    font-style: italic;
}
#vg-app-header .vg-tagline {
    display: inline-block;
    color: #CADCFC;
    font-size: 11px;
    letter-spacing: 3px;
    margin-bottom: 4px;
    text-transform: uppercase;
}
#vg-app-header .vg-red-bar {
    height: 3px;
    background: #E53E3E;
    width: 48px;
    margin-bottom: 10px;
    border-radius: 2px;
}

/* Section header markdown */
.vg-section-header {
    color: #1E2761;
    padding-bottom: 4px;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 10px;
}

.vg-muted-hint { color: #64748B; font-size: 12px; font-style: italic; }

/* Tabs */
.tab-nav button.selected {
    color: #1E2761 !important;
    border-bottom: 3px solid #1E2761 !important;
    font-weight: 600 !important;
}

/* Large primary buttons (Clone, Send to Shield) */
button.lg.primary {
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 14px 20px !important;
}

/* Similarity / status Textbox values, make numbers feel important */
textarea[readonly], input[readonly] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    color: #0F172A !important;
}
"""


def build_app(cloner: VoiceCloner, detector: "DeepfakeDetector | None"):
    with gr.Blocks(
        title="VoiceGuard · Clone + Detect",
        theme=VG_THEME,
        css=VG_CSS,
        analytics_enabled=False,
    ) as app:

        # ---- Branded header ----
        gr.HTML(
            """
            <div id="vg-app-header">
                <span class="vg-tagline">14-795 · VoiceGuard · Apr 2026</span>
                <div class="vg-red-bar"></div>
                <h1>VoiceGuard &nbsp;·&nbsp; Voice-Cloning Attack &amp; Detection</h1>
                <p>Clone a voice from a short reference clip — then catch it with
                a wav2vec&nbsp;2.0 + AASIST detector. All in one app.</p>
            </div>
            """
        )

        with gr.Tab("🗡️  Attack  ·  clone a voice"):
            attack_output, send_btn = build_attack_tab(cloner)

        with gr.Tab("🛡️  Shield  ·  detect deepfake"):
            if detector:
                shield_input = build_shield_tab(detector)
                send_btn.click(
                    fn=lambda x: x,
                    inputs=[attack_output],
                    outputs=[shield_input],
                )
            else:
                gr.Markdown(
                    "⚠️ **Shield tab unavailable** — model checkpoints not found.\n\n"
                    "Download these files to `checkpoints/`:\n"
                    "- `xlsr2_300m.pt` (wav2vec 2.0 XLSR)\n"
                    "- `best_SSL_model_DF.pth` (W2V-AASIST weights)"
                )

    return app


def main():
    parser = argparse.ArgumentParser(description="VoiceGuard")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda:0)")
    args = parser.parse_args()

    cloner = VoiceCloner(openvoice_root=OPENVOICE_ROOT, device=args.device)

    # Only init detector if checkpoints exist
    detector = None
    ckpt_file = os.path.join(CHECKPOINTS_ROOT, "best_SSL_model_DF.pth")
    if os.path.isfile(ckpt_file):
        detector = DeepfakeDetector(
            ssl_root=SSL_ROOT,
            checkpoints_root=CHECKPOINTS_ROOT,
            device=args.device,
        )

    app = build_app(cloner, detector)
    app.queue()
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
