# Kaggle Notebook Setup Guide

This guide shows how to run the Medical Document Summarizer in a Kaggle notebook for testing and demonstration. The Kaggle version runs in batch mode without the Gradio UI and focuses on the core LLM summarization functionality.

## Key Differences from Local Deployment

Due to Kaggle's architecture and limitations:

1. **Flattened code structure**: Python packages are collapsed into sequential cells (no `services.llm.engine` imports)
2. **No subdirectories in datasets**: Prompt files must be uploaded to the dataset root, not a `prompts/` folder
3. **No UI**: Gradio interface is removed; execution happens via code cells
4. **Simplified dependencies**: Only core packages needed for LLM inference
5. **Testing stubs removed**: Any `if __name__ == "__main__"` blocks must be stripped out

## Setup Steps

### 1. Create and Upload the Prompts Dataset

**Prepare your files:**

- `json_prompt.txt` - System prompt for initial summarization
- `conversational_prompt.txt` - System prompt for follow-up questions

**Upload to Kaggle:**

1. Go to Kaggle.com → Your Profile → Datasets → New Dataset
2. Upload both `.txt` files **directly to the root** (not in a subdirectory)
3. Name the dataset (e.g., `medgemma-laurie-prompts`)
4. Set visibility to Private
5. Create the dataset

### 2. Configure Notebook Settings

Create a new Kaggle notebook with:

- **Accelerator**: GPU T4 x2 (or any GPU)
- **Internet**: ON (required to download models)
- **Attached Dataset**: Your prompts dataset from step 1

### 3. Set Up Secrets

Go to **Add-ons → Secrets** and add:

- **Name**: `HUGGING_FACE_HUB_TOKEN`
- **Value**: Your HuggingFace token (get from https://huggingface.co/settings/tokens)

Optional secrets for configuration:

- **Name**: `INFRA_ABILITY`, **Value**: `HIGH` (for full precision) or `LOW` (for 4-bit quantization, default)
- **Name**: `MODEL_MODE`, **Value**: `TEXT` (default) or `VLM` (for vision support)

### 4. Notebook Code Structure

Create the following cells in order. **Critical**: Update the dataset path in Cell 2.

---

#### Cell 1: Install Dependencies

```python
%pip install -q bitsandbytes transformers accelerate json5
```

**Note**: This is much simpler than local installation. No Gradio, no ASR, no audio libraries.

---

#### Cell 2: Configuration

Copy the **entire contents** of the following files into this single cell:

1. `core/app_config.py`
2. `services/llm/config.py`

**CRITICAL MODIFICATION**: In the `app_config.py` section, find this line:

```python
BASE_DIR: Path = Path(__file__).parent.parent.resolve()
```

And replace it with your Kaggle dataset path:

```python
BASE_DIR: Path = Path("/kaggle/input/YOUR-DATASET-NAME-HERE")
```

Example:

```python
BASE_DIR: Path = Path("/kaggle/input/medgemma-laurie-prompts")
```

**Also modify these lines** to point directly to files in the dataset root:

```python
PROMPT_FILE_PATH: Path = BASE_DIR / "json_prompt.txt"
CONVERSATIONAL_PROMPT_FILE_PATH: Path = BASE_DIR / "conversational_prompt.txt"
```

The cell should output:

```
✅ Running in Kaggle environment.
✅ Kaggle secret 'HUGGING_FACE_HUB_TOKEN' loaded into environment.
✅ App config loaded. Mode: 'LOW'
✅ LLM Service Mode: 'TEXT' using model 'google/medgemma-1.5-4b-it'
```

---

#### Cell 3: JSON Parsing Functions

Copy the **entire contents** of `core/parse_json.py`.

**IMPORTANT**: Remove the testing stub at the bottom (the `if __name__ == "__main__":` block and everything after it).

---

#### Cell 4: LLM Engine

Copy the **entire contents** of `services/llm/engine.py`.

**IMPORTANT MODIFICATIONS**:

1. **Remove all import statements** at the top that reference local modules:

```python
   # DELETE these lines:
   from core.app_config import active_infra_config
   from services.llm.config import active_model_config, MODEL_MODE
```

These variables are already defined in the global namespace from Cell 2.

2. Keep only the standard library and transformers imports:

```python
   from typing import Optional, Dict, Any, Union, List
   import torch
   from transformers import (
       AutoProcessor,
       AutoModelForCausalLM,
       BitsAndBytesConfig,
       PreTrainedModel,
       PreTrainedTokenizer,
   )
```

This cell will not produce output (it just defines functions).

---

#### Cell 5: Core Logic Wrapper

Copy the **entire contents** of `core/core_logic.py`.

**IMPORTANT MODIFICATIONS**:

1. **Remove all local module imports** at the top
2. Ensure the function calls reference the global namespace

This cell defines the `get_llm_summary()` function that orchestrates the summarization.

---

#### Cell 6: Test Execution

```python
# Example medical text
medical_text = """
The patient is a 68-year-old gentleman with a history of coronary artery disease,
hypertension, diabetes and stage III CKD with a creatinine of 1.8 in May 2006
corresponding with the GFR of 40-41 mL/min. The patient had blood work done at
Dr. XYZ's office on June 01, 2006, which revealed an elevation in his creatinine
up to 2.3. He was asked to come in to see a nephrologist for further evaluation.
"""

# Generate summary
print("\n--- Calling get_llm_summary ---")
summary = get_llm_summary(medical_text, None)

# Display result
print("\n--- Final Generated Summary ---")
print(summary)
```

**Expected behavior**:

- First run: Downloads model (~8.6GB), takes 3-5 minutes
- Subsequent runs: Uses cached model, much faster
- Output: Structured JSON summary or formatted text depending on your prompt

---

## Execution Notes

1. **Run cells sequentially** (1 → 2 → 3 → 4 → 5 → 6)
2. **First run will be slow** due to model download
3. **Check cell outputs** to ensure each step completes successfully
4. **Model caching**: Once downloaded, the model persists for the session

## Testing VLM Mode

To test vision-language capabilities (image + text input):

1. Set `MODEL_MODE` secret to `VLM` before starting
2. Upload a test medical image to a Kaggle dataset and attach it
3. Modify Cell 6 to include image path:

```python
image_path = "/kaggle/input/your-image-dataset/xray.jpg"
medical_text = "What does this X-ray show?"

summary = get_llm_summary(medical_text, image_path)
print(summary)
```

## Troubleshooting

**"Prompt file not found"**

- Ensure prompts are in dataset root (not subdirectory)
- Check dataset name matches the path in `BASE_DIR`
- Verify dataset is attached to notebook

**"Failed to get Kaggle secret"**

- Add `HUGGING_FACE_HUB_TOKEN` in Secrets panel
- Restart the kernel after adding secrets

**"Out of memory"**

- Set `INFRA_ABILITY` to `LOW` (enables 4-bit quantization)
- Use a GPU with more VRAM (P100 or A100)

**"Module not found"**

- Ensure you removed all local imports (`from services.llm...`, `from core...`)
- Everything should reference the global namespace

## Limitations

- **No interactive UI**: This is batch processing only
- **No ASR support**: Voice transcription not available in Kaggle version
- **Session-based**: Model cache and data don't persist across sessions
- **No package structure**: Code is flattened into notebook cells

---

For the full interactive experience with Gradio UI, ASR support, and rich text editing, see the main README for local deployment instructions.
