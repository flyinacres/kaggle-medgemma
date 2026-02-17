# ui/layout.py
#
# Defines the full Gradio UI: all tabs, components, and event wiring.
# Call build_ui() to get back the configured gr.Blocks instance, which
# app.py then launches.
#
# Component references that span tabs (e.g. output_text referenced in
# the User Notes pull button) are kept in local scope within build_ui()
# so all event wiring can access them without globals.

import gradio as gr
from ui.animation_handler import invoke_llm_with_animation, handle_follow_up
from ui.js_strings import LOAD_QUILL_JS, PRINT_JS, QUILL_CSS_HTML
from ui.helpers import prepare_notes_html, process_audio


def build_ui() -> gr.Blocks:
    """
    Constructs and returns the configured Gradio Blocks application.

    All tabs, components, and event handlers are defined here. The
    returned demo object is launched by app.py.
    """

    with gr.Blocks(title="MedSumma") as demo:

        # Holds the full conversation history across follow-up questions.
        # Passed into and out of handle_follow_up on every turn.
        conversation_state = gr.State(value=None)

        with gr.Tabs(selected=0) as tabs:

            # --- Tab 0: Welcome ---
            with gr.TabItem("Welcome", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Image(
                            value="assets/hero-image.jpg",
                            elem_id="hero_image",
                            show_label=False,
                            interactive=False,
                            width=225
                        )
                    with gr.Column(scale=4):
                        gr.Markdown(
                            """
                            # Welcome to the Medical Document Summarizer
                            ### Your AI-powered assistant for understanding complex medical information.
                            This tool is designed to help patients and caregivers by summarizing medical notes, lab results, and reports into clear, easy-to-understand language.

                            **To begin, click the 'Inputs' tab above. Once you have entered all relevent information select the Explain button**
                            ---
                            ### **Disclaimer**
                            > **This is not medical advice.** Use this tool to help understand your medical information, but always consult with your medical professionals when you have questions and concerns.
                            """
                        )

            # --- Tab 1: Inputs ---
            with gr.TabItem("Inputs", id=1):
                with gr.Column():
                    main_text_input = gr.Textbox(
                        lines=15,
                        label="Medical Notes",
                        placeholder="Paste or type medical text here...",
                        elem_id="main_textbox_id"
                    )

                    with gr.Row(equal_height=True):
                        mic_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record & Transcribe Audio"
                        )
                        image_upload = gr.Image(
                            sources=["upload", "webcam"],
                            type="filepath",
                            label="Upload or Take Photo"
                        )

                    with gr.Row():
                        gr.Column(scale=1)  # Spacer to align button under image
                        with gr.Column(scale=1):
                            explain_btn = gr.Button(
                                "Explain",
                                variant="primary",
                                size="lg",
                                elem_id="explain_button_id"
                            )

            # --- Tab 2: Explanations ---
            with gr.TabItem("Explanations", id=2):
                with gr.Column():

                    # Shown during LLM processing; hidden once results arrive
                    animation_output = gr.Textbox(
                        lines=8,
                        label="Processing...",
                        interactive=False,
                        visible=False,
                        elem_id="animation_output_id"
                    )

                    with gr.Group():
                        with gr.Column(elem_id="scroll_container"):
                            output_text = gr.Markdown(
                                label="Simplified Explanation",
                                elem_id="summary_output_textbox"
                            )

                    # Hidden until the first summary is generated
                    with gr.Group(visible=False) as follow_up_ui:
                        chatbot = gr.Chatbot(label="Follow-up Questions", height=200)
                        with gr.Row():
                            follow_up_txt = gr.Textbox(
                                placeholder="Ask a question about the summary...",
                                show_label=False,
                                scale=4,
                                container=False
                            )
                            follow_up_btn = gr.Button("Ask", scale=1, variant="primary")

                    with gr.Row():
                        gr.Column(scale=1)
                        with gr.Column(scale=1):
                            notes_btn = gr.Button(
                                "Create User Notes",
                                variant="primary",
                                size="lg",
                                elem_id="notes_button_id"
                            )

            # --- Tab 3: User Notes ---
            with gr.TabItem("User Notes", id=3) as notes_tab:

                with gr.Row():
                    pull_btn = gr.Button(
                        "Pull Data from Explanations",
                        elem_id="pull_data_btn",
                        size="lg",
                        variant="primary",
                        scale=0
                    )
                    with gr.Column(scale=1):
                        pass  # Absorbs remaining row width

                # The div that Quill mounts into
                gr.HTML('<div id="quill-editor" style="height: 400px; background: white;"></div>')

                # Quill theme overrides to match Gradio's CSS variable system
                gr.HTML(QUILL_CSS_HTML)

                # Hidden textbox that bridges Quill (JS) and Python event handlers.
                # Quill writes its HTML here; Python reads it via Gradio events.
                hidden_input = gr.Textbox(value="", elem_id="hidden_text", visible=True)

                with gr.Row():
                    with gr.Column(scale=1):
                        pass  # Absorbs remaining row width
                    print_btn = gr.Button("Print or Save", variant="primary", scale=0)

        # --- Event Wiring ---

        # Initialize Quill when the User Notes tab is selected
        notes_tab.select(fn=None, inputs=None, outputs=None, js=LOAD_QUILL_JS)

        # Transcribe audio and append to the medical notes textbox
        mic_input.stop_recording(
            fn=process_audio,
            inputs=[mic_input, main_text_input],
            outputs=[main_text_input]
        )

        # Run the LLM summarization with loading animation, then switch to Explanations tab
        explain_btn.click(
            fn=invoke_llm_with_animation,
            inputs=[main_text_input, image_upload],
            outputs=[
                animation_output,
                output_text,
                tabs,
                conversation_state,
                follow_up_ui,
                chatbot
            ]
        )

        # Navigate to User Notes tab
        notes_btn.click(
            fn=lambda: gr.Tabs(selected=3),
            outputs=tabs
        )

        # Pull summary and conversation into the Quill editor.
        # Step 1: Python formats and writes HTML to the hidden bridge textbox.
        # Step 2: JS confirms overwrite if notes exist, then pastes into Quill.
        pull_btn.click(
            fn=prepare_notes_html,
            inputs=[output_text, chatbot],
            outputs=hidden_input
        ).then(
            fn=None,
            js="""() => {
                const currentContent = window.quill.getText().trim();
                if (currentContent.length > 0) {
                    if (!confirm("This will overwrite your current notes. Continue?")) {
                        return;
                    }
                }
                window.quill.root.innerHTML = document.querySelector('#hidden_text textarea').value;
            }"""
        )

        # Follow-up question: both button click and Enter key submit
        follow_up_btn.click(
            fn=handle_follow_up,
            inputs=[follow_up_txt, conversation_state, image_upload],
            outputs=[chatbot, conversation_state, follow_up_txt]
        )
        follow_up_txt.submit(
            fn=handle_follow_up,
            inputs=[follow_up_txt, conversation_state, image_upload],
            outputs=[chatbot, conversation_state, follow_up_txt]
        )

        # Open print/save dialog using Quill content
        print_btn.click(fn=None, inputs=None, outputs=None, js=PRINT_JS)

    return demo
