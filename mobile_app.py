# mobile_app.py

import gradio as gr
import time
from services.asr.engine import transcribe_audio
from animation_handler import invoke_llm_with_animation, handle_follow_up

def prepare_notes_html(summary, history):
    safe_summary = str(summary) if summary else "No summary available."
    html_content = f"<h2>Summary</h2><p>{safe_summary}</p>"
    
    if not history:
        return html_content

    html_content += "<h2>Follow-up Conversation</h2>"
    
    for i, message in enumerate(history):
        try:
            # Access the 'content' list, then the first dict, then the 'text' key
            role = message.get("role", "Unknown").capitalize()
            content_list = message.get("content", [])
            
            if content_list and isinstance(content_list, list):
                text_content = content_list[0].get("text", "")
                html_content += f"<p><strong>{role}:</strong> {text_content}</p>"
            else:
                # Fallback for unexpected structure
                html_content += f"<p><strong>{role}:</strong> {str(message)}</p>"
                
        except Exception as e:
            print(f"Error at index {i}: {str(e)}")
            continue

    return html_content

def process_audio(audio_file, existing_text):
    """
    Placeholder for a speech-to-text model.
    Appends the transcribed text to the existing text in the main input box.
    """
    if audio_file is None:
        return existing_text
    time.sleep(2)
    transcribed_text = transcribe_audio(audio_file)
    return f"{existing_text}{transcribed_text}".strip()

head = """
<link href="https://cdn.quilljs.com/1.3.6/quill.snow.css" rel="stylesheet">
<script src="https://cdn.quilljs.com/1.3.6/quill.js"></script>
"""

# The JS function we want to run
load_quill_js = """
async () => {
    if (typeof Quill !== 'undefined') {
        // Check if the editor container already has the Quill class
        let quill_container = document.querySelector('#quill-editor');
        if (quill_container && !quill_container.classList.contains('ql-container')) {
            // Assign the instance to window.quill
            window.quill = new Quill('#quill-editor', {
                modules: {
                    toolbar: [
                        [{ 'header': [1, 2, 3, false] }],
                        ['bold', 'italic', 'underline', 'strike'],
                        [{ 'color': [] }, { 'background': [] }],
                        [{ 'list': 'ordered'}, { 'list': 'bullet' }, { 'list': 'check'}],
                        ['blockquote', 'code-block'],
                        ['link', 'image'],
                        ['clean']
                    ]
                },
                theme: 'snow'
            });
            }

        const syncGradio = () => {
            const html = window.quill.root.innerHTML;
            // Target the textarea specifically within the elem_id
            const container = document.querySelector('#hidden_text');
            const gradioTextbox = container ? container.querySelector('textarea') : null;
            
            if (gradioTextbox) {
                gradioTextbox.value = html;
                gradioTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                gradioTextbox.dispatchEvent(new Event('change', { bubbles: true }));
            }
        };

        // 1. Sync immediately upon initialization
        syncGradio();

        // 2. Sync on every edit
        window.quill.on('text-change', syncGradio);

        // Listener for Python -> Quill updates
        const container = document.querySelector('#hidden_text');
        const gradioTextbox = container ? container.querySelector('textarea') : null;

        if (gradioTextbox) {
            // Monitor for changes coming FROM Python/Gradio
            const updateQuillFromGradio = () => {
                const newValue = gradioTextbox.value;
                // Only update if the content is actually different to avoid loops
                if (window.quill && newValue !== window.quill.root.innerHTML) {
                    // This triggers the Quill parser correctly
                    window.quill.clipboard.dangerouslyPasteHTML(newValue);
                }
            };

            // Use 'input' event which Gradio often triggers for value updates
            gradioTextbox.addEventListener('input', updateQuillFromGradio);
            // Backup for standard change events
            gradioTextbox.addEventListener('change', updateQuillFromGradio);
        }

    } else {
        alert("Quill library not loaded yet.");
    }
}
"""

# JavaScript to grab Quill content and open a print window
print_js = """
() => {
    if (!window.quill) {
        alert('Editor not initialized.');
        return;
    }
    const data = window.quill.root.innerHTML;
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
        <html>
            <head>
                <title>MedSumma Notes</title>
                <style>
                    body { font-family: sans-serif; padding: 20px; line-height: 1.6; }
                    h2 { color: #2c3e50; border-bottom: 1px solid #eee; }
                    p { margin-bottom: 10px; }
                </style>
            </head>
            <body>${data}</body>
        </html>
    `);
    printWindow.document.close();
    // Wait for content to load, then print
    printWindow.onload = function() {
        printWindow.print();
    };
}
"""


def save_content(rich_text):
    # This receives the raw HTML from the editor
    return f"Saved HTML length: {len(rich_text)} characters."


# --- Gradio UI Definition ---
with gr.Blocks(title="MedSumma", head=head) as demo:
    # This invisible component will hold our session state.
    conversation_state = gr.State(value=None)

    with gr.Tabs(selected=0) as tabs:
        
        # --- Tab 0: Welcome / Landing Page ---
        with gr.TabItem("Welcome", id=0):
            # 5. Hero Image
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Image(
                        value="assets/hero-image.jpg",
                        elem_id="hero_image", # Assign an ID for the CSS to target
                        show_label=False,
                        interactive=False,
                        width=225
                    )
                with gr.Column(scale=4):
                    gr.Markdown(
                        """
                        # Welcome to MedSumma
                        ### Your AI-powered assistant for understanding complex medical information.
                        This tool is designed to help patients and caregivers by summarizing medical notes, lab results, and reports into clear, easy-to-understand language.

                        **To begin, click the 'Inputs' tab above.**
                        ---
                        ### **Disclaimer**
                        > **This is not medical advice.** Use this tool to help understand your medical information, but always consult with your medical professionals when you have questions and concerns.
                        """
                    )

        # --- Tab 1: Input Page ---
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
                    # 1 & 2. Sized image upload with webcam
                    image_upload = gr.Image(
                        sources=["upload", "webcam"],
                        type="filepath",
                        label="Upload or Take Photo"
                    )
                with gr.Row():
                    gr.Column(scale=1)  # Empty spacer matching audio width
                    with gr.Column(scale=1):  # Matches image width
                        explain_btn = gr.Button("Explain", variant="primary", size="lg", elem_id="explain_button_id")
        
        # --- Tab 2: Output Page ---
        with gr.TabItem("Explanations", id=2):
            with gr.Column():
                animation_output = gr.Textbox(
                    lines=8,
                    label="Processing...",
                    interactive=False,
                    visible=False,
                    elem_id="animation_output_id"
                )

                # This entire group is hidden by default and appears after the first summary.
                with gr.Group(visible=False) as follow_up_ui:
                    with gr.Column(elem_id="scroll_container"):
                        output_text = gr.Markdown(
                            label="Simplified Explanation",
                            elem_id="summary_output_textbox"
                        )

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
                        notes_btn = gr.Button("Create User Notes", variant="primary", size="lg", elem_id="notes_button_id")


        # --- Tab 3: User Notes ---
        with gr.TabItem("User Notes", id=3) as notes_tab:
            with gr.Row():
                pull_btn = gr.Button("Pull Data from Explanations", elem_id="pull_data_btn", size="lg", variant="primary", scale=0)
                with gr.Column(scale=1): # This container absorbs the extra width
                        pass

            # The target div
            gr.HTML('<div id="quill-editor" style="height: 200px; background: white;"></div>')
            gr.HTML("""
                <style>
                    /* Use Gradio's internal variables for true theme responsiveness */
                    .ql-container.ql-snow, .ql-editor {
                        /* This variable automatically changes between light/dark in Gradio */
                        background-color: var(--input-background-fill) !important;
                        color: var(--body-text-color) !important;
                        border: 1px solid var(--border-color-primary) !important;
                    }

                    /* Force all child elements (b, i, p, h) to use the same dynamic color */
                    .ql-editor * {
                        color: inherit !important;
                    }

                    /* Fix for bullets and list numbers */
                    .ql-editor li::before {
                        color: var(--body-text-color) !important;
                    }

                    /* Ensure the toolbar doesn't look like a 'dark hole' in light mode */
                    .ql-toolbar.ql-snow {
                        background-color: var(--block-background-fill, #f9f9f9) !important;
                        border-color: var(--border-color-primary) !important;
                    }

                    /* Keep your Enlarge Checkboxes logic */
                    .ql-editor li[data-list="checked"]::before, 
                    .ql-editor li[data-list="unchecked"]::before {
                        transform: scale(2.0);
                        margin-right: 15px;
                        cursor: pointer;
                    }

                    /* 5. Placeholder */
                    .ql-editor.ql-blank::before {
                        color: var(--body-text-color-subdued, #666) !important;
                        font-style: normal;
                        opacity: 0.6;
                    }
                    
                    /* 6. Section Spacing (Adds breathing room between headers) */
                    .ql-editor h2, .ql-editor h3 {
                        margin-top: 1.5em !important;
                        margin-bottom: 0.5em !important;
                    }


                    /* Enlarge Checkboxes */
                    .ql-editor li[data-list="checked"]::before, 
                    .ql-editor li[data-list="unchecked"]::before {
                        transform: scale(2.0); /* Makes them 50% larger */
                        margin-right: 15px;    /* Prevents overlap with text */
                        cursor: pointer;
                    }

                    /* Ensure the placeholder text is visible but faint */
                    .ql-editor.ql-blank::before {
                            color: #666 !important;
                            font-style: normal;
                        }
                </style>
                """)

            # The "Bridge" component (Hidden from user)
            hidden_input = gr.Textbox(value="", elem_id="hidden_text", visible=True)

            # Row of final action buttons
            with gr.Row():
                with gr.Column(scale=1): # This container absorbs the extra width
                        pass
                print_btn = gr.Button("Print or Save", 
                    variant="primary", scale=0 )

    # Load the quill rte when the notes tab is selected
    notes_tab.select(fn=None, inputs=None, outputs=None, js=load_quill_js)

    # Event Wire-up
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
    # --- Component Logic ---
    mic_input.stop_recording(
        fn=process_audio,
        inputs=[mic_input, main_text_input],
        outputs=[main_text_input]
    )

    explain_btn.click(
        fn=invoke_llm_with_animation,
        inputs=[main_text_input, image_upload],
        outputs=[
            animation_output,
            output_text,
            tabs,
            conversation_state, # 4. Update the state
            follow_up_ui      # 5. Make the follow-up UI visible
        ],
    )

    notes_btn.click(
        fn=lambda: gr.Tabs(selected=3), 
        outputs=tabs
    )
    # Click event for the follow-up button.
    follow_up_btn.click(
        fn=handle_follow_up,
        inputs=[follow_up_txt, conversation_state, image_upload],
        outputs=[
            chatbot,              # 1. Update the chatbot display
            conversation_state,   # 2. Pass the updated state back to itself
            follow_up_txt         # 3. Clear the user's input textbox
        ]
    )

    # 2. The Enter Key (Submit)
    follow_up_txt.submit(
        fn=handle_follow_up,
        inputs=[follow_up_txt, conversation_state, image_upload],
        outputs=[
            chatbot,              # 1. Update the chatbot display
            conversation_state,   # 2. Pass the updated state back to itself
            follow_up_txt         # 3. Clear the user's input textbox
        ]
    )

    print_btn.click(fn=None, inputs=None, outputs=None, js=print_js)

   


with open("styles.css", "r") as f:
    custom_css = f.read()
demo.launch(theme=gr.themes.Glass(primary_hue="blue"), css=custom_css, share=False)