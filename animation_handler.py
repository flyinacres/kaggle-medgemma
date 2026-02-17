# animation_handler.py

import gradio as gr
import time
from core.task_runner import BackgroundTaskRunner
from core.core_logic import get_llm_summary, get_follow_up_answer

# --- Animation Constants ---
ROBOT_FRAMES = [
    # Front poses
    """
(●_●)
 ╱|╲
 |
 ╱ ╲
""",
    """
(●_●)
╱ | ╲
 |
╱  ╲
""",
    """
(●_●)
╲ | ╲
 |
╱  ╲
""",
    """
(●_●)
╱ | ╱
 |
╱  ╲
""",
    """
(●o●)
╲ | ╱
 ╱|
   ╲
""",
    """
(●o●)
╲ | ╱
 |
╱  ╲
""",
    """
(●_●)
 ╲|
  |╲
╱  ╲
""",
    """
(●o●)
╲ | ╲
 |
╱  ╲
""",
    """
(●_●)
  |╱
 ╱|
╱  ╲
""",
    # Side poses
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
    # Upside down poses
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
]
PLACEHOLDER_TEXT = "Consulting the AI engine... please wait.\nThis may take a number of minutes!"

def invoke_llm_with_animation(text, image_path):
    """
    The main generator for the Gradio UI.
    It uses BackgroundTaskRunner to run the core logic and yields UI updates.
    """
    # 1. Instantiate and start the background task.
    # The UI handler doesn't know *how* it runs, only that it can be started.
    task = BackgroundTaskRunner(get_llm_summary, text, image_path)
    task.start()

    # 2. Run the animation loop while the task is running.
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
            gr.update(visible=False)
            )

    # 3. Once the task is finished, get the result and yield the final UI state.
    try:
        # The UI handler gets the final result or an exception.
        final_text = task.get_result()
        # Create the initial state object to be saved
        new_state = {
            "original_text": text,
            "summary": final_text,
            "history": [] # History is a list of (question, answer) tuples
        }
        # Final yield shows the result, saves state, and makes the follow-up UI visible
        yield (
            gr.update(visible=False),
            gr.update(value=final_text, visible=True),
            gr.update(selected=2),
            new_state,  # Update the gr.State component
            gr.update(visible=True) # Make the follow-up UI group visible
        )
    except Exception as e:
        # If the task failed, display the error in the UI.
        error_message = f"An unexpected error occurred: {e}. Please change your information and try again"
        yield (
            gr.update(visible=False), 
            gr.update(value=error_message, visible=True), 
            gr.update(selected=2), 
            None, 
            gr.update(visible=False))


def handle_follow_up(new_question, current_state, image_path):
    """
    Handles the logic for follow-up questions.
    """
    if not new_question or not new_question.strip():
        return current_state["history"], current_state, ""

    # Call the core logic to get a grounded answer
    answer = get_follow_up_answer(
        original_text=current_state["original_text"],
        summary=current_state["summary"],
        history=current_state["history"],
        new_question=new_question,
        image_path=image_path
    )

    # Update the history and the state object
    current_state["history"].append((new_question, answer))
    # Convert the internal history (list of tuples) into the required
    # List of Dictionaries format.
    chatbot_display_data = []
    for q, a in current_state["history"]:
        chatbot_display_data.append({'role': 'user', 'content': q})
        chatbot_display_data.append({'role': 'assistant', 'content': a})
    
    # Return updates for the UI: the chatbot, the state, and clear the textbox
    return chatbot_display_data, current_state, ""