# Medical Document Summarizer

**A patient-centered tool that translates complex medical documentation into clear, understandable summaries using Google's MedGemma model.**

This project addresses the critical gap between the medical information patients receive and their ability to comprehend it. Built for the HAI-DEF Kaggle hackathon, it leverages state-of-the-art medical language models to help patients understand doctors' notes, test results, discharge instructions, and specialist reports without providing medical advice or diagnosis.

## The Problem

Research shows that only 12% of American adults have the health literacy needed to fully understand what their doctors tell them. Patients forget 40-80% of medical information immediately after consultations, and 78% of emergency department patients leave with deficient comprehension of their discharge instructions—often without realizing it. This comprehension gap costs the US healthcare system $106-238 billion annually and is independently associated with a 75% increase in mortality risk among elderly patients.

## The Solution

This application provides three input modes (text entry, voice recording with medical transcription, and image upload) that feed into MedGemma's medical language understanding. The output is a structured, patient-friendly summary with key takeaways, medication explanations, medical term definitions, and suggested follow-up questions. A built-in chat interface allows patients to ask clarifying questions, all grounded in the original medical documentation.

The system runs locally on consumer hardware (preserving patient privacy) and is optimized for mobile viewing, as patients often need to reference medical information away from their computers.

## Sources

Health Literacy:

12% proficient health literacy: 2003 National Assessment of Adult Literacy (NAAL) - NCES Publication 2006-483; HHS/AHRQ 2008 Issue Brief

Comprehension Failures:

40-80% forgotten immediately: Kessels, Journal of the Royal Society of Medicine, 2003; AAFP, 2018
78% ED patients deficient comprehension: Engel et al., Annals of Emergency Medicine, 2009

Economic Impact:

$106-238 billion annually: Vernon et al., University of Connecticut/National Patient Safety Foundation, 2007

Mortality:

75% increased mortality risk (HR 1.75): Sudore et al., Journal of General Internal Medicine, 2006

Privacy Concerns:

92% consider privacy a right: AMA/Savvy Cooperative survey, AMA 2022
14% trust tech companies: Rock Health 2023 consumer survey

## Features

- **Medical Text Summarization**: Converts dense medical jargon into plain-language summaries using `google/medgemma-1.5-4b-it`
- **Voice-to-Text Transcription**: Accurate medical dictation via `google/MedASR` with medical terminology support
- **Image Comprehension**: Visual question-answering for medical images (X-rays, lab results, etc.) using MedGemma's vision-language capabilities
- **Multi-turn Chat**: Follow-up questions grounded in the provided medical documentation
- **Privacy-Preserving**: Runs entirely on local hardware—no cloud transmission of medical data
- **Mobile-Optimized UI**: Responsive Gradio interface designed for smartphone viewing
- **Rich Text Editor**: Quill-based note-taking with formatting, checklists, and highlighting
- **Print/PDF Export**: Save summaries for offline reference

## Project Architecture

```
├── app.py                    # Main Gradio application (mobile-optimized)
├── requirements.txt          # Python dependencies
├── .env                      # Secrets (HuggingFace token) - not committed
├── kaggle_setup.md           # Kaggle-specific deployment instructions
├── LICENSE.md                # MIT license. Note that components used in this projet may have other restrictions
├── README.md                 # This readme file
├── assets/
│   └── hero-image.jpg        # UI imagery
├── core/
│   ├── app_config.py         # Environment detection, hardware config, paths
│   ├── core_logic.py         # High-level request orchestration
│   ├── parse_json.py         # Defensive JSON extraction from LLM output
│   └── task_runner.py        # Background task execution (for animation)
├── prompts/
│   ├── json_prompt.txt       # System prompt for initial summarization
│   ├── conversational_prompt.txt  # System prompt for follow-up questions
│   └── [other experimental prompts]
├── services/
│   ├── llm/
│   │   ├── config.py         # MedGemma model configuration
│   │   └── engine.py         # Model initialization and generation
│   └── asr/
│       └── engine.py         # Speech-to-text pipeline
└── tests/
    └── [experimental test applications - not maintained unit tests]
├── ui/
│   ├── animation_handler.py  # Loading animation while LLM processes
│   ├── helpers.py            # Helpers for UI functionality
│   ├── js_strings.py         # JavaScript strings used in UI
│   └── layout.py             # Fundamental Gradio layout
    └── styles.css            # CSS for Gradio UI customization
```

## User Flow

1. **Welcome Tab**: Introduction, disclaimer, and usage guidance
2. **Inputs Tab**: Enter medical text manually, record audio (transcribed via MedASR), or upload an image
3. **Explanations Tab**: View structured summary with key takeaways, medications, term definitions, and suggested questions
4. **Follow-up Questions**: Multi-turn chat to ask clarifying questions about the medical information
5. **User Notes Tab**: Edit summary in rich text editor (Quill), add personal notes, create checklists
6. **Print/Save**: Export to PDF or print directly

## Setup and Installation

### Prerequisites

- **Python 3.11** (specific version required for dependency compatibility)
- **Hardware**: NVIDIA GPU with CUDA support, Apple Silicon Mac (M1/M2/M3), or Kaggle environment
- **Git**
- **Hugging Face Account** with access token

You will not be able to use the required models from Hugging Face if you do not accept the
model license and create a proper token and set it in your environment.
All links subject to change...
https://developers.google.com/health-ai-developer-foundations/medgemma/model-card-v1
https://huggingface.co/settings/tokens

### 1. Clone the Repository

```bash
git clone [Your GitHub Repository URL]
cd [your-project-directory]
```

### 2. Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv venv
venv\Scripts\activate.bat

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch

Install PyTorch for your specific hardware **before** installing other dependencies. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct command.
Consult requirements.txt to see general and specific requirements

**Example for NVIDIA GPU (CUDA 12.1):**

```bash
 pip install torch==2.9.1+cu126 torchvision==0.24.1+cu126 torchaudio==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

**Example for Apple Silicon (MPS):**

```bash
pip3 install torch torchvision torchaudio
```

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Hugging Face Token

**Local Development:**

Create a `.env` file in the project root:

```
HUGGING_FACE_HUB_TOKEN="hf_YourTokenHere"
```

**Kaggle Environment:**

See `kaggle_setup.md` for Kaggle-specific configuration.

## Configuration

The application adapts to hardware capabilities via environment variables.

### Hardware Profile (`INFRA_ABILITY`)

Controls quantization and resource allocation:

- **`LOW`** (default): 4-bit quantization for consumer GPUs (8-11GB VRAM). Ideal for NVIDIA 1080Ti, RTX 3060, etc.
- **`HIGH`**: Full precision (bfloat16) for high-end GPUs (24GB+ VRAM)
- **`APPLE_SILICON`**: Optimized for Mac M1/M2/M3 chips (uses MPS backend)

**Windows (PowerShell):**

```powershell
$env:INFRA_ABILITY="LOW"
python app.py
```

**Windows (CMD):**

```cmd
set INFRA_ABILITY=LOW
python app.py
```

**macOS/Linux:**

```bash
export INFRA_ABILITY=LOW
python app.py
```

### Model Mode (`MODEL_MODE`)

Selects text-only or vision-language capabilities:

- **`TEXT`** (default): Text summarization and chat only
- **`VLM`**: Enables image upload and visual question-answering

**Windows (PowerShell):**

```powershell
$env:MODEL_MODE="VLM"
python app.py
```

**Windows (CMD):**

```cmd
set MODEL_MODE=VLM
python app.py
```

**macOS/Linux:**

```bash
export MODEL_MODE=VLM
python app.py
```

## Usage

### Basic Usage (Text Mode)

```bash
python app.py
```

Access the web interface at `http://localhost:7860` (or the URL shown in the terminal). The interface is mobile-responsive—you can access it from your phone by using your computer's local IP address.

To access the web interface remotely change the last few lines in app.py. Where it says share=False, change this
to share=True. The URL required to access the application will show in the terminal. THIS WILL CHANGE with each
run of the application.

### With Vision-Language Support

```bash
# Windows PowerShell
$env:MODEL_MODE="VLM"
python app.py

# macOS/Linux
export MODEL_MODE=VLM
python app.py
```

When running in VLM mode, you can upload medical images (X-rays, lab results, etc.) along with a text prompt asking questions about the image.

### Kaggle Deployment

See `kaggle_setup.md` for deploying in Kaggle notebooks. Note that Kaggle deployment does not include the Gradio UI—it's designed for batch processing and experimentation.

### Usage Notes

- This is a reasonably featured prototype, but is NOT a production application, and should not be used as such.
- The time it takes to generate output is entirely dependent on the hardware upon which it runs.
- First run will download ~8.6GB of model files and may take 5-10 minutes depending on connection speed. Subsequent runs use cached models.
- VLM mode (image+text) uses significantly more memory and is slower than text-only mode.
- It is NOT a multiuser application at this point in time. It can work on one query at a time.
- It can support multiple queries in a row.
- It has been tested on Windows 11 with 1080Ti, Mac M1 Max 64GB, and Kaggle. It should work on other systems of equivalent or greater capability
- This application has token limits so cannot process arbitrarily long text. Particularly in LOW INFRA_ABILITY, input should be restricted to a paragraph or two. You can modify the token limits in the code, but you may hit GPU, memory, or time constraints if you raise them too high. When the model hits the token limit it generates output, but it is invalid JSON so does not show results on screen.

## Technical Implementation Highlights

### Privacy-First Architecture

All processing happens locally. Medical documents never leave the user's device, addressing the critical concern that 92% of patients consider health data privacy a fundamental right and only 14% trust tech companies with health information.

### Resource-Aware Deployment

- **4-bit quantization** (BitsAndBytes) enables deployment on consumer GPUs
- **Lazy model loading** minimizes startup time and memory footprint
- **Tested across environments**: Windows 11 (1080Ti), Mac M1 Max, Kaggle T4

### Structured Output Generation

Rather than parsing unreliable free-text output, the system prompts MedGemma to generate JSON-structured summaries. A defensive parsing layer (`parse_json.py`) handles common LLM errors like trailing commas, type mismatches, and preamble text using the `json5` library.

### Multimodal Architecture

A unified `AutoProcessor` handles both text-only and vision-language inputs through the same inference engine, controlled by the `MODEL_MODE` configuration.

## Known Limitations

- **Not a medical device**: This tool summarizes information patients already have—it does not diagnose or provide medical advice
- **Model size constraints**: Using a 4B parameter model balances capability with edge deployment feasibility
- **LLM output variability**: Defensive parsing and prompt engineering mitigate but don't eliminate occasional formatting inconsistencies

## License

This project is licensed under the MIT License.

## Acknowledgments

Built for the HAI-DEF Kaggle Hackathon using Google's MedGemma and MedASR models. Inspired by the real-world challenge of understanding complex medical information during a family member's hospitalization.

---

**Disclaimer**: This application is a comprehension aid, not a medical tool. It does not provide medical advice, diagnosis, or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.
