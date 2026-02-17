# ui/js_strings.py
#
# All JavaScript and HTML string constants injected into the Gradio interface.
# Keeping these here prevents the layout file from being cluttered with large
# multi-line strings that are hard to read alongside Python logic.

# Injected into the Gradio <head> to load the Quill rich-text editor library
# and its default Snow theme stylesheet from the CDN.
HEAD_HTML = """
<link href="https://cdn.quilljs.com/1.3.6/quill.snow.css" rel="stylesheet">
<script src="https://cdn.quilljs.com/1.3.6/quill.js"></script>
"""

# Runs when the User Notes tab is selected.
# Initializes the Quill editor on first load (guards against double-init),
# then sets up two-way sync between the Quill instance and the hidden Gradio
# textbox that acts as the bridge between JS and Python.
LOAD_QUILL_JS = """
async () => {
    if (typeof Quill !== 'undefined') {
        // Guard: only initialize if Quill hasn't already taken over the container
        let quill_container = document.querySelector('#quill-editor');
        if (quill_container && !quill_container.classList.contains('ql-container')) {
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

        // Pushes Quill HTML content into the hidden Gradio textarea so Python
        // can read it via the standard Gradio event system.
        const syncGradio = () => {
            const html = window.quill.root.innerHTML;
            const container = document.querySelector('#hidden_text');
            const gradioTextbox = container ? container.querySelector('textarea') : null;
            if (gradioTextbox) {
                gradioTextbox.value = html;
                gradioTextbox.dispatchEvent(new Event('input', { bubbles: true }));
                gradioTextbox.dispatchEvent(new Event('change', { bubbles: true }));
            }
        };

        // Sync immediately on init, then on every subsequent edit
        syncGradio();
        window.quill.on('text-change', syncGradio);

        // Pulls content from the hidden Gradio textarea into Quill when Python
        // updates it (e.g. when the user clicks "Pull Data from Explanations").
        const container = document.querySelector('#hidden_text');
        const gradioTextbox = container ? container.querySelector('textarea') : null;
        if (gradioTextbox) {
            const updateQuillFromGradio = () => {
                const newValue = gradioTextbox.value;
                // Only update if content differs to avoid infinite sync loops
                if (window.quill && newValue !== window.quill.root.innerHTML) {
                    window.quill.clipboard.dangerouslyPasteHTML(newValue);
                }
            };
            gradioTextbox.addEventListener('input', updateQuillFromGradio);
            gradioTextbox.addEventListener('change', updateQuillFromGradio);
        }

    } else {
        alert("Quill library not loaded yet.");
    }
}
"""

# Runs when the user clicks "Print or Save".
# Reads Quill HTML content, opens a blank tab with minimal print styling,
# and triggers the browser's native print/save-as-PDF dialog.
PRINT_JS = """
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
    // Wait for content to fully load before triggering print
    printWindow.onload = function() {
        printWindow.print();
    };
}
"""

# Injected as a gr.HTML block inside the User Notes tab.
# These styles theme the Quill editor to match Gradio's CSS variable system
# so it respects the active light/dark theme automatically.
QUILL_CSS_HTML = """
<style>
    /* Editor body and container use Gradio theme variables for light/dark support */
    .ql-container.ql-snow, .ql-editor {
        background-color: var(--input-background-fill) !important;
        color: var(--body-text-color) !important;
        border: 1px solid var(--border-color-primary) !important;
    }

    /* Force all inline elements to inherit the dynamic text color */
    .ql-editor * {
        color: inherit !important;
    }

    /* List numbers and bullet markers */
    .ql-editor li::before {
        color: var(--body-text-color) !important;
    }

    /* Toolbar background and border */
    .ql-toolbar.ql-snow {
        background-color: var(--block-background-fill, #f9f9f9) !important;
        border-color: var(--border-color-primary) !important;
    }

    /* Toolbar icon colors */
    .ql-toolbar.ql-snow .ql-stroke {
        stroke: var(--body-text-color) !important;
    }
    .ql-toolbar.ql-snow .ql-fill {
        fill: var(--body-text-color) !important;
    }
    .ql-toolbar.ql-snow .ql-picker-label {
        color: var(--body-text-color) !important;
    }

    /* Active and hover states for toolbar buttons */
    .ql-toolbar.ql-snow button:hover,
    .ql-toolbar.ql-snow button.ql-active {
        background-color: var(--button-secondary-background-fill-hover) !important;
    }
    .ql-toolbar.ql-snow button:hover .ql-stroke,
    .ql-toolbar.ql-snow button.ql-active .ql-stroke {
        stroke: var(--primary-600) !important;
    }
    .ql-toolbar.ql-snow button:hover .ql-fill,
    .ql-toolbar.ql-snow button.ql-active .ql-fill {
        fill: var(--primary-600) !important;
    }

    /* Enlarge checkboxes for usability and add spacing before text */
    .ql-editor li[data-list="checked"]::before,
    .ql-editor li[data-list="unchecked"]::before {
        transform: scale(2.0);
        margin-right: 15px;
        cursor: pointer;
    }

    /* Placeholder text: visible but subdued */
    .ql-editor.ql-blank::before {
        color: var(--body-text-color-subdued, #666) !important;
        font-style: normal;
        opacity: 0.6;
    }

    /* Breathing room above section headers */
    .ql-editor h2, .ql-editor h3 {
        margin-top: 1.5em !important;
        margin-bottom: 0.5em !important;
    }
</style>
"""
