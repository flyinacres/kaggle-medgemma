# [Your Project Name: e.g., MedGemma Patient Summarizer]

**[Brief, one-sentence description of the project. e.g., A tool to synthesize complex medical notes into patient-friendly summaries using Google's HAI-DEF models.]**

This project leverages state-of-the-art models from the Hugging Face ecosystem to provide a simple interface for medical information synthesis. It is built with a modular, service-oriented architecture to allow for easy extension and maintenance.

## Features

*   **Core Text Synthesis:** Utilizes the `google/medgemma-1.5-4b-it` model for summarizing and simplifying medical text.
*   **Voice-to-Text Transcription:** Integrates `google/MedASR` for accurate medical dictation, allowing for hands-free input.
*   **Multimodal Image Comprehension:** Integrates `google/medgemma-1.5-4b-it` in its visual mode to answer questions about medical images (e.g., X-rays, CT scans).
*   **Resource-Aware Configuration:** Automatically adapts model precision (quantization) and resource allocation based on the detected hardware environment (Kaggle, High-end GPU, Apple Silicon, etc.).
*   **Modular Architecture:** Services for text generation (LLM) and speech recognition (ASR) are isolated in their own modules, making the system easy to test, maintain, and extend.

## Project Architecture

The project is organized into a service-oriented structure to separate concerns:

```
├── mobile_app.py              # Main Gradio/UI application
├── app_config.py              # Global configuration (environment, hardware)
├── requirements.txt           # Project dependencies
├── services/
│   ├── llm/
│   │   ├── config.py          # LLM-specific model configuration
│   │   └── engine.py          # LLM pipeline initialization and logic
│   └── asr/
│       ├── config.py          # ASR-specific model configuration
│       └── engine.py          # ASR pipeline initialization and logic
└── prompts/
    └── refined_prompt.txt     # System prompt for the LLM
```

## Setup and Installation

### 1. Prerequisites

*   Python 3.11
*   An NVIDIA GPU with CUDA support (for GPU acceleration), an Apple Silicon Mac, or a Kaggle Notebook.
*   Git

### 2. Clone the Repository

```bash
git clone [Your GitHub Repository URL]
cd [your-project-directory]
```

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For venv
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# For Conda
conda create -n medgemma_env python=3.11
conda activate medgemma_env
```

### 4a. Install Pytorch

First, install PyTorch according to your specific hardware configuration. This ensures you get the correct version for your GPU's CUDA toolkit or CPU.
Visit the official PyTorch "Get Started" page to find the correct command for your system:
https://pytorch.org/get-started/locally/
For example, a common command for a recent NVIDIA GPU would be:

```Bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```

### 4b. Install Project Dependencies

Once PyTorch is installed, install the remaining libraries from the requirements.txt file.
code


The required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 5. Configure Secrets

The application requires a Hugging Face Hub token to download the models.

*   **Local Development:**
    1.  Create a file named `.env` in the project root.
    2.  Add your Hugging Face token to it:
        ```
        HUGGING_FACE_HUB_TOKEN="hf_YourTokenHere"
        ```

*   **Kaggle Environment:**
    1.  Go to "Add-ons" -> "Secrets" in your Kaggle notebook.
    2.  Add a new secret with the name `HUGGING_FACE_HUB_TOKEN` and your token as the value.

## Configuration

The application's behavior can be controlled via environment variables.

*   `INFRA_ABILITY`: Controls the hardware profile.
    *   `LOW` (Default): Uses 4-bit quantization for lower VRAM/RAM usage. Ideal for consumer GPUs and constrained environments.
    *   `HIGH`: Runs models in `bfloat16` for maximum precision. Requires a high-VRAM GPU.
    *   `APPLE_SILICON`: Optimizes settings for Macs with M-series chips (`mps` device).
*   `MODEL_MODE`: Selects the LLM's operational mode.
    *   `TEXT` (Default): For text-only generation and summarization.
    *   `VLM`: Enables multimodal mode. The model can process both an image and a text prompt to perform visual question answering.
## Usage

To run the main application, execute the following command from the root directory:

```bash
python mobile_app.py
```

This will launch the Gradio web interface, which you can access in your browser on your computer or phone.

### Using the Multimodal (VLM) Mode

To enable image comprehension, you must set the `MODEL_MODE` environment variable before launching the application.

```bash
export MODEL_MODE=VLM
python mobile_app.py

When running in VLM mode, the model expects two inputs: an image file and a text prompt.
VLM Prompt Structure
The model requires a precise prompt template to understand multimodal inputs. The prompt must be a single string containing special tokens that the model is trained to recognize:
code
Code
<|image|>\n<|user|>\n{your_prompt_text_here}<|endofchunk|>
<|image|>: A literal token that acts as a placeholder for the image.
<|user|>: A token indicating the start of the user's question or instruction.
<|endofchunk|>: A token signaling the end of the user's input.
The engine.py file automatically constructs this prompt string from the text you provide in the UI. For example, if you input "Describe this X-ray", the engine sends the model the full <|image|>\n<|user|>\nDescribe this X-ray<|endofchunk|> string along with the image.
code


## License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.