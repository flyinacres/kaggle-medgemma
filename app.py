# app.py
#
# Entry point for MedSumma. Builds the Gradio UI and launches the server.
#
# Usage:
#   python app.py
#
# Configuration is handled via environment variables (see README):
#   INFRA_ABILITY  - hardware profile: LOW | HIGH | APPLE_SILICON
#   MODEL_MODE     - input modality: TEXT | VLM

import gradio as gr
from ui.layout import build_ui
from ui.js_strings import HEAD_HTML

# Load custom CSS from the ui/ folder. The css= parameter in demo.launch()
# requires the file contents as a string, not a file path.
with open("ui/styles.css", "r") as f:
    custom_css = f.read()

demo = build_ui()

demo.launch(
    theme=gr.themes.Glass(primary_hue="blue"), 
    head=HEAD_HTML,
    css=custom_css,
    share=False
)
