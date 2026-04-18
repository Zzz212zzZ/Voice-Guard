import gradio as gr

from modules.attack.clone import VoiceCloner


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

    send_to_shield_btn = gr.Button("Send to Shield →", variant="secondary")

    def on_clone(text, ref_audio, speed):
        try:
            if not text or len(text.strip()) < 2:
                return ("Error: Please enter at least 2 characters.", None, "")
            ref = ref_audio if ref_audio else None
            output_path, sr = cloner.clone(
                text=text, reference_audio=ref, speed=speed,
            )
            ref_path = ref if ref else cloner.default_reference

            sim = cloner.compute_similarity(ref_path, output_path)
            sim_str = f"{sim * 100:.1f}%"

            return (f"Voice cloned successfully. Similarity: {sim_str}",
                    output_path, sim_str)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (f"Error: {e}", None, "")

    clone_button.click(
        fn=on_clone,
        inputs=[text_input, reference_audio, speed_slider],
        outputs=[status_text, output_audio, similarity_text],
    )

    return output_audio, send_to_shield_btn
