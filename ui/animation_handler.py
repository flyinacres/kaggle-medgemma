# ui/animation_handler.py
#
# Handles the animated loading state and follow-up question logic for the
# Gradio UI. These are generator/handler functions wired directly to Gradio
# events in ui/layout.py.
#
# The LLM call is run in a background thread via BackgroundTaskRunner so the
# UI can yield animation frames while waiting for the model to respond.

import gradio as gr
import time
from core.task_runner import BackgroundTaskRunner
from core.core_logic import get_llm_summary, get_follow_up_answer


# ASCII art frames cycled during LLM processing to show the user something
# is happening. The robot cycles through front poses, side poses, and
# upside-down poses to give the impression of break-dancing.
# Intermediate frames have been added at section transitions and within the
# front section to smooth out abrupt position changes.
ROBOT_FRAMES = [
    # --- Front poses ---
    # Arms symmetrical, neutral
"""
(●_●)
 ╱|╲
 |
 ╱ ╲
""",
    # Arms spread wider
"""
(●_●)
╱ | ╲
 |
╱  ╲
""",
    # Left arm drops
"""
(●_●)
╲ | ╲
 |
╱  ╲
""",
    # Transition: arms angling up-right
"""
(●_●)
╲ | ╱
 |
╱  ╲
""",
    # Right arm raises
"""
(●_●)
╱ | ╱
 |
╱  ╲
""",
    # Transition: arms crossing, weight shifting
"""
(●o●)
╱ | ╲
 ╱|
╱  ╲
""",
    # Arms crossed, foot forward
"""
(●o●)
╲ | ╱
 ╱|
   ╲
""",
    # Arms crossed, feet even
"""
(●o●)
╲ | ╱
 |
╱  ╲
""",
    # Transition: left arm tucking
"""
(●_●)
╲ | ╱
 |╲
╱  ╲
""",
    # Left arm tucked fully
"""
(●_●)
 ╲|
  |╲
╱  ╲
""",
    # Transition: arms resetting
"""
(●o●)
╲ | ╲
 |
╱  ╲
""",
    # Right arm sweeps up
"""
(●_●)
  |╱
 ╱|
╱  ╲
""",
    # --- Transition: front to side ---
    # Head begins turning
"""
(●_◑)
 ╱|╲
 |
╱  ╲
""",
    # Head three-quarter turn
"""
(●_◐)
  |╲
 |
╱ ╲
""",
    # --- Side poses: robot slides across ---
"""
(●‿)
 |╲
 |
╱ ╲
""",
"""
 (●‿)
  |╲
  |
 | ╲
""",
"""
  (●‿)
   |╲
   |
  ╱ ╲
""",
"""
  (●_)
  ╱|╲
   |
  ╱ ╲
""",
"""
  (●o●)
  ╲ | ╱
   |
  ╱  ╲
""",
"""
  (_●)
  ╱|╲
  |
  ╱ ╲
""",
"""
  (‿●)
  ╱|
   |
  ╱ ╲
""",
"""
 (‿●)
 ╱|
  |
 ╱ |
""",
"""
(‿●)
╱|
 |
╱ ╲
""",
    # --- Transition: side to upside-down ---
    # Starting to tip forward
"""
(●_●)
  ╱|
 ╱ |
╱  ╲
""",
    # Mid-tumble
"""
 (●●)
  |╱
╱ |
  ╲
""",
    # --- Upside-down poses ---
"""

 ● ╱__| 
  ● ╲  ╲
 
""",
"""

    
 ● ╱__╱ 
 ● |  ╲
""",
"""
 ╱ ╲
 ╱|╲
(●‾●)

""",
"""
 ╱ ╲
 |
 ╱|╲
(●‾●)
""",
"""
╱  ╲
 |
╱ | ╲
(●‾●)
""",
"""
 ╱  ╱
 |
╱ | ╲
(●‾●)
""",
"""
╲  ╲
 |
╱ | ╲
(●‾●)
""",
"""
 ╱ ╲
 ╱|╲
(●‾●)

""",
"""

 ╲__╱ ● 
 |  | ●

""",
"""

       
╲__| ●
 ╱  ╲ ●
""",
    # Back to upright start
"""
(●_●)
 ╱|╲
 |
 ╱ ╲
""",
]

# Displayed alongside the animation while the model is running
PLACEHOLDER_TEXT = "Consulting the AI engine... please wait.\nThis may take a number of minutes!"


def invoke_llm_with_animation(text, image_path):
    """
    Gradio generator that runs LLM summarization in a background thread
    and yields UI updates for each animation frame while waiting.

    Yields 6-tuples matching the outputs defined in layout.py:
        (animation_output, output_text, tabs, conversation_state, follow_up_ui, chatbot)

    On completion, hides the animation, shows the summary, switches to the
    Explanations tab, saves conversation state, and reveals the follow-up UI.
    On error, displays the exception message in the output area.

    Args:
        text:       Medical text entered by the user.
        image_path: Optional path to an uploaded image, or None.
    """
    task = BackgroundTaskRunner(get_llm_summary, text, image_path)
    task.start()

    # Cycle through animation frames at ~2.5fps while the model runs
    frame_index = 0
    while task.is_running():
        time.sleep(0.4)
        frame_index = (frame_index + 1) % len(ROBOT_FRAMES)
        animation_text = f"{PLACEHOLDER_TEXT}\n\n{ROBOT_FRAMES[frame_index]}"
        yield (
            gr.update(value=animation_text, visible=True),
            gr.update(visible=False),
            gr.update(selected=2),
            gr.update(),
            gr.update(visible=False),
            gr.update()
        )

    # Task complete: retrieve result and update the UI to its final state
    try:
        final_text = task.get_result()

        # Initial conversation state stored in gr.State for follow-up questions
        new_state = {
            "original_text": text,
            "summary": final_text,
            "history": []  # List of (question, answer) tuples, appended on each follow-up
        }

        yield (
            gr.update(visible=False),                   # Hide animation
            gr.update(value=final_text, visible=True),  # Show summary
            gr.update(selected=2),                      # Stay on Explanations tab
            new_state,                                  # Save state
            gr.update(visible=True),                    # Reveal follow-up UI
            gr.update(value=[])
        )

    except Exception as e:
        error_message = (
            f"An unexpected error occurred: {e}. "
            "Please change your information and try again."
        )
        yield (
            gr.update(visible=False),
            gr.update(value=error_message, visible=True),
            gr.update(selected=2),
            None,
            gr.update(visible=False),
            gr.update(value=[])
        )


def handle_follow_up(new_question, current_state, image_path):
    """
    Handles a single follow-up question turn.

    Calls get_follow_up_answer() with the full conversation context, appends
    the result to history, and returns data formatted for the Gradio Chatbot
    component (list of role/content dicts).

    Args:
        new_question:   The user's question string.
        current_state:  The gr.State dict containing original_text, summary,
                        and history.
        image_path:     Optional image path passed through for VLM context.

    Returns:
        Tuple of (chatbot_display_data, updated_state, empty_string).
        The empty string clears the follow-up input textbox.
    """
    if not new_question or not new_question.strip():
        return current_state["history"], current_state, ""

    answer = get_follow_up_answer(
        original_text=current_state["original_text"],
        summary=current_state["summary"],
        history=current_state["history"],
        new_question=new_question,
        image_path=image_path
    )

    current_state["history"].append((new_question, answer))

    # Gradio Chatbot requires a flat list of role/content dicts
    chatbot_display_data = []
    for q, a in current_state["history"]:
        chatbot_display_data.append({"role": "user", "content": q})
        chatbot_display_data.append({"role": "assistant", "content": a})

    return chatbot_display_data, current_state, ""