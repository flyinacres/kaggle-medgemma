# --- Setup: Required imports and data ---
import gradio as gr
import time

import random


# Define the frames for the robot animation.
# Each multi-line string in the list is a single frame.
robot_frames = [
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
(●o●)
╲ | ╱
 ╱|
   ╲
""",
    """
(●_●)
 ╲|╱
  |╲
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
(●_)
╱|╲
 |
╱ ╲
""",
    # Upside down poses
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
]
# --- Core Logic ---
start_idx = random.randint(0, len(robot_frames) - 1)

def mock_long_process_with_animation():
    """
    Simulates a long-running task while displaying an animation.
    It yields animation frames to the output component while running.
    """
    # Animation loop: repeats the sequence of frames 3 times.
    for _ in range(3):
        for frame in robot_frames:
            # 'yield' sends an update to the Gradio interface immediately.
            # This is the key to creating the animation effect.
            yield frame
            time.sleep(0.2)  # Controls the speed of the animation.

    # Simulate the final part of the long-running task.
    time.sleep(1)

    # 'return' sends the final output and concludes the function.
    return "Task Complete!"

# --- Gradio Interface Definition ---

with gr.Blocks() as demo:
    gr.Markdown("# Gradio Animation Demo")
    gr.Markdown("Click the button to start a mock process and see a Unicode animation.")

    with gr.Row():
        start_button = gr.Button("Start Process")
        # Use a Textbox to display the animation.
        # `lines=4` ensures it's tall enough for the 4x4 art.
        output_display = gr.Textbox(label="Status", lines=5, interactive=False)

    # Connect the button's 'click' event to the animation function.
    # The function's output (from 'yield' and 'return') will be sent to 'output_display'.
    start_button.click(
        fn=mock_long_process_with_animation,
        inputs=None,
        outputs=output_display
    )

# --- Application Launch ---

# To run this, save it as a Python file (e.g., app.py) and run 'python app.py'
# from your terminal. Then, open the provided URL in your browser.
if __name__ == "__main__":
    demo.launch()