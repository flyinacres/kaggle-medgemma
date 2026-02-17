import gradio as gr

head = """
<link href="https://cdn.quilljs.com/1.3.6/quill.snow.css" rel="stylesheet">
<script src="https://cdn.quilljs.com/1.3.6/quill.js"></script>
"""

# The JS function we want to run
load_quill_js = """
async () => {
    if (typeof Quill !== 'undefined') {
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

        // Test this inside your load_quill_js to see how it renders
        const testMedicalHTML = `<h2>Summary Report</h2>
<blockquote><b>CRITICAL:</b> Patient shows acute worsening of kidney function (GFR 41 -> 35).</blockquote>

<h3>📌 Key Takeaways</h3>
<ul>
    <li><b>Worsening CKD:</b> Creatinine rose from 1.8 to 2.3.</li>
    <li><b>Dizziness:</b> Likely <i>Orthostatic Hypotension</i>.</li>
    <li><b>Herb Review:</b> Stop all Chinese supplements immediately.</li>
</ul>

<h3>💊 Medications</h3>
<ul>
    <li><b>TriCor</b> (Cholesterol)
        <ul>
            <li><i>Dosage:</i> 145mg Daily</li>
            <li><i>Status:</i> Continue</li>
        </ul>
    </li>
    <li><b>Flomax</b> (BPH)
        <ul>
            <li><i>Status:</i> Review effectiveness; symptoms persist.</li>
        </ul>
    </li>
</ul>

<h3>📖 Terms Explained</h3>
<p><b>GFR:</b> Glomerular Filtration Rate. Measures kidney efficiency.</p>
<p><b>Orthostatic Hypotension:</b> Drop in BP upon standing.</p>

<hr>
<h3>❓ Questions for Doctor</h3>
<ol>
    <li>Should we adjust the TriCor dosage given the GFR drop?</li>
    <li>Is a referral to a Nephrologist necessary now?</li>
</ol>`;
        window.quill.clipboard.dangerouslyPasteHTML(testMedicalHTML);
    } else {
        alert("Quill library not loaded yet.");
    }
}
"""


def save_content(rich_text):
    # This receives the raw HTML from the editor
    return f"Saved HTML length: {len(rich_text)} characters."

with gr.Blocks(head=head) as demo:
    gr.Markdown("### Manual Editor Load")
    
    # The target div
    gr.HTML('<div id="quill-editor" style="height: 200px; background: white;"></div>')
    gr.HTML("""
<style>
    /* Force white background and black text inside the editor */
    .ql-container.ql-snow, .ql-editor {
        background-color: white !important;
        color: black !important;
    }

    /* Make the toolbar icons visible on dark backgrounds */
    .ql-toolbar.ql-snow {
        background-color: #f0f0f0 !important;
        border-color: #ccc !important;
    }

    /* 1. Set the baseline for the entire editor area */
    .ql-editor {
        color: #FF0000 !important;
        /* This ensures that any text not inside a tag inherits black */
    }

    /* 2. Force all children to inherit the parent's color 
       UNLESS they have an inline style (which is what Quill 
       uses when a user manually picks a color) */
    .ql-editor * {
        color: inherit;
    }

    /* 3. Handle the '::before' pseudo-elements (bullets and checkboxes) */
    .ql-editor li::before {
        color: #000000 !important;
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

    # The Trigger Button
    btn = gr.Button("Initialize Editor")
    
    # This runs the JS directly in the browser
    btn.click(None, None, None, js=load_quill_js)

    # The "Bridge" component (Hidden from user)
    hidden_input = gr.Textbox(value="", elem_id="hidden_text", visible=True)
    
    # Output to show it works
    out = gr.Textbox(label="Python side received:")
    
    btn = gr.Button("Submit Rich Text")
    btn.click(save_content, inputs=hidden_input, outputs=out)

demo.launch()