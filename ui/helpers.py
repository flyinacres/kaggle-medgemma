# ui/helpers.py
#
# Helper functions that support the Gradio UI layer.
# These sit between raw Gradio events and the core business logic —
# they format data for display or pre-process UI inputs before passing
# them further down the stack.

import time
from services.asr.engine import transcribe_audio


def prepare_notes_html(summary: str, history: list) -> str:
    """
    Converts the Explanations tab content into an HTML string suitable
    for loading into the Quill rich-text editor on the User Notes tab.

    Formats the summary as an <h2> section, followed by each message in
    the follow-up conversation history as labeled paragraphs.

    Args:
        summary:  The markdown/text summary from the Explanations tab.
        history:  The chatbot message history. Each entry is a dict with
                  'role' and 'content' keys, where content is a list of
                  typed dicts (e.g. [{"type": "text", "text": "..."}]).

    Returns:
        An HTML string ready to be pasted into Quill.
    """
    safe_summary = str(summary) if summary else "No summary available."
    html_content = f"<h2>Summary</h2><p>{safe_summary}</p>"

    if not history:
        return html_content

    html_content += "<h2>Follow-up Conversation</h2>"

    for i, message in enumerate(history):
        try:
            role = message.get("role", "Unknown").capitalize()
            content_list = message.get("content", [])

            if content_list and isinstance(content_list, list):
                text_content = content_list[0].get("text", "")
                html_content += f"<p><strong>{role}:</strong> {text_content}</p>"
            else:
                # Fallback for unexpected message structure
                html_content += f"<p><strong>{role}:</strong> {str(message)}</p>"

        except Exception as e:
            print(f"prepare_notes_html: error processing message at index {i}: {e}")
            continue

    return html_content


def process_audio(audio_file: str, existing_text: str) -> str:
    """
    Transcribes a recorded audio file and appends the result to any
    text already in the medical notes input box.

    Called by the mic_input.stop_recording event. The brief sleep gives
    the audio component time to finalize the file before transcription.

    Args:
        audio_file:    File path to the recorded audio, or None if no
                       recording was made.
        existing_text: Current contents of the medical notes textbox.

    Returns:
        The combined text (existing + transcribed), stripped of leading
        and trailing whitespace. Returns existing_text unchanged if no
        audio file was provided.
    """
    if audio_file is None:
        return existing_text

    time.sleep(2)
    transcribed_text = transcribe_audio(audio_file)
    return f"{existing_text}{transcribed_text}".strip()
