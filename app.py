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


# Minimal polish — Soft theme + small CSS tweaks. Nothing heavy.
VG_CSS = """
.gradio-container { max-width: 1280px !important; }
footer { display: none !important; }
"""


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

    with gr.Blocks(title="VoiceGuard", theme=gr.themes.Soft(), css=VG_CSS) as app:
        gr.Markdown("# VoiceGuard: Voice Cloning Attack & Detection")

        with gr.Tab("Attack"):
            attack_output, send_btn = build_attack_tab(cloner)

        with gr.Tab("Shield"):
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

    app.queue()
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
