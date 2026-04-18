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


# Icon polish only — two SVG icons inlined as CSS background images.
# Mic (white) on the primary "Clone & Speak" button; shield (dark) on
# the secondary "Send to Shield" button. No layout or color changes.
_MIC_WHITE = (
    "url(\"data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
    "fill='none' stroke='white' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round'%3E"
    "%3Cpath d='M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z'/%3E"
    "%3Cpath d='M19 10v2a7 7 0 0 1-14 0v-2'/%3E"
    "%3Cline x1='12' x2='12' y1='19' y2='22'/%3E"
    "%3C/svg%3E\")"
)
_SHIELD_DARK = (
    "url(\"data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
    "fill='none' stroke='%231f2937' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round'%3E"
    "%3Cpath d='M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 "
    "4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 "
    "1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z'/%3E"
    "%3Cpath d='m9 12 2 2 4-4'/%3E"
    "%3C/svg%3E\")"
)
VG_CSS = f"""
#vg-btn-clone::before,
#vg-btn-send-shield::before {{
    content: "";
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 8px;
    vertical-align: -3px;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}}
#vg-btn-clone::before       {{ background-image: {_MIC_WHITE}; }}
#vg-btn-send-shield::before {{ background-image: {_SHIELD_DARK}; }}
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

    with gr.Blocks(title="VoiceGuard", css=VG_CSS) as app:
        gr.Markdown("# VoiceGuard: Voice Cloning Attack & Detection")

        with gr.Tab("Attack"):
            attack_output, send_btn = build_attack_tab(cloner)

        with gr.Tab("Shield"):
            if detector:
                shield_input = build_shield_tab(detector)
                # Wire "Send to Shield" button
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
