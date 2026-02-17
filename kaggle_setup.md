Kaggle Notebook Setup Guide (Refactored Architecture)
This document provides step-by-step instructions for running the refactored MedGemma service project within a Kaggle Notebook.
Overview
The workflow involves four main steps:
Configure the Environment: Set up Kaggle secrets, attach the prompt dataset, and enable GPU/internet access.
Install Dependencies: Run a pip install cell to get the required libraries.
Set Up Code Cells: Create cells for the combined configuration, the combined service engines, and the final execution.
Run Sequentially: Execute the cells in order.
Step 1: Environment Configuration
These actions are identical to your previous setup.
1.1. Set Up Secrets:
HUGGING_FACE_HUB_TOKEN: Your Hugging Face API token.
INFRA_ABILITY: HIGH (or LOW for smaller GPUs).
MODEL_MODE: TEXT.
1.2. Upload and Attach Prompt Dataset:
Upload your prompts folder as a Kaggle Dataset.
Attach the dataset to your notebook.
1.3. Configure Notebook Settings:
Set Accelerator to a GPU (e.g., "GPU T4 x2").
Ensure Internet is ON.
Step 2: Code Setup in Notebook Cells
Create the following four cells in your notebook. You must run them in order.
Cell 1: Install Dependencies
Action: This is a new, required first step. Paste the following commands into the first cell to install the project's dependencies.
code
Python
# Cell 1: Dependencies
%pip install accelerate bitsandbytes huggingface_hub safetensors tokenizers soundfile torchaudio gradio python-dotenv --quiet
%pip install git+https://github.com/huggingface/transformers.git --quiet
Cell 2: Combined Configuration
Action: In this cell, you will paste the contents of all three configuration files one after another. This single cell defines the entire application's configuration.
Copy the entire contents of app_config.py and paste it into the cell.
CRITICAL: In the code you just pasted, find the line BASE_DIR: Path = ... and update it to match your Kaggle dataset's path. For example:
code
Python
# Find this line from app_config.py and edit it:
BASE_DIR: Path = Path("/kaggle/input/medgemma-prompts") # <-- Change "medgemma-prompts" to your dataset name
Next, copy the entire contents of services/llm/config.py and append it to the bottom of the same cell.
Finally, copy the entire contents of services/asr/config.py and append it to the bottom of the same cell.
The final cell will be large, containing the code from all three files sequentially.
Cell 3: Combined Service Engines
Action: Similar to the previous step, combine the engine files here. This cell will initialize and load the models.
Copy the entire contents of services/llm/engine.py and paste it into this cell.
Copy the entire contents of services/asr/engine.py and append it to the bottom of the same cell.
This cell will take several minutes to run as it downloads and initializes both the MedGemma and MedASR models.
Cell 4: Main Execution Block
Action: Copy the code below into your fourth cell. This cell demonstrates the full workflow, including a simulated audio transcription.
code
Python
# Cell 4: Main Execution Block

# This cell calls the functions and variables defined in the previous cells.

# 1. Define your input text for the model
medical_text = """
Patient presents with a 2-day history of sharp, stabbing pain in the lower right quadrant.
Physical examination reveals rebound tenderness and guarding.
WBC count is elevated at 15,000.
Ultrasound shows a non-compressible, dilated appendix.
"""

# 2. Simulate the output from the ASR service
# In a real app, you would call transcribe_audio(path_to_file) here.
# For this test, we will just use a sample transcribed text.
transcribed_notes = "Doctor's follow-up note: patient is afebrile and pain is responding to initial treatment. Plan to monitor."

# 3. Combine the text sources
combined_text = f"{medical_text}\n\n{transcribed_notes}"

# 4. Call the generation function
print("\n--- Calling generate_summary with combined text ---")
print(f"Input to LLM:\n{combined_text}")
summary = generate_summary(combined_text)

# 5. Print the final result
print("\n--- Final Generated Summary ---")
print(summary)