"""
Initializes and provides the core ASR transcription function.

This engine bypasses the Hugging Face `pipeline` for inference. Instead, it
replicates the direct inference steps from the official MedASR guide. This is
necessary to ensure the model's custom CTC decoder, located in the processor,
is used correctly to prevent output artifacts like <epsilon> tokens.

This engine uses a robust audio loading strategy. It first uses the `pydub`
library to convert the input audio file into a standardized WAV format. This
prevents errors from browser-specific audio formats (like MP4/AAC from Safari)
before passing the clean data to torchaudio and the model.
"""

import librosa

from transformers import AutoProcessor, AutoModelForCTC
from typing import Optional

from core.app_config import active_infra_config
from services.asr.config import (
    active_asr_config,
    asr_loader_kwargs,
    asr_model_loader_kwargs
)

# --- Global variables to hold the loaded model and processor ---
processor: Optional[AutoProcessor] = None
model: Optional[AutoModelForCTC] = None
target_device: str = "cpu" # Default device


def initialize_asr_model():
    """
    Loads the ASR model and processor into the global variables.
    This function should only be called once.
    """
    global processor, model, target_device  # Declare modification of global variables

    # --- Prevent re-initialization ---
    if model is not None and processor is not None:
        print("✅ ASR model already initialized.")
        return

    print("🚀 Initializing ASR model and processor for the first time...")
    print(f"   - ASR Model: {active_asr_config['model_id']}")
    print(f"   - ASR Quantization: {'Enabled' if 'quantization_config' in asr_model_loader_kwargs else 'Disabled'}")

    try:
        # 1. Load the processor.
        print("   - Loading ASR processor...")
        processor = AutoProcessor.from_pretrained(**asr_loader_kwargs)

        # 2. Load the model.
        print("   - Loading ASR model...")
        model = AutoModelForCTC.from_pretrained(**asr_model_loader_kwargs)

        # 3. Determine the target device and move the model to it.
        if "device_map" in asr_model_loader_kwargs:
            target_device = model.device
            print(f"   - Model device automatically set to: {target_device}")
        else:
            target_device = active_infra_config.get("device_map", "cpu")
            model.to(target_device)
            print(f"   - Model manually moved to device: {target_device}")

        print("✅ ASR model and processor initialized successfully.")

    except Exception as e:
        print(f"🔥 CRITICAL: ASR initialization failed. Transcription not functional. Error: {e}")
        # Reset globals on failure to allow retry
        processor = None
        model = None

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes an audio file using librosa for loading and direct model inference.
    initialized on the first call.
    """

    # --- Lazy Loading Trigger ---
    # If the model isn't loaded, call the initialization function.
    if model is None or processor is None:
        initialize_asr_model()

    if not model or not processor:
        return "Sorry, the transcription model is not available."
    if not audio_file_path:
        return ""

    try:
        print("⏳ Transcribing audio...")

        # 1. Load and resample audio using librosa.
        target_sample_rate = processor.feature_extractor.sampling_rate
        waveform, _ = librosa.load(audio_file_path, sr=target_sample_rate, mono=True)

        from transformers import pipeline

        model_id = "google/medasr"
        pipe = pipeline("automatic-speech-recognition", model=model_id)
        result = pipe(waveform, chunk_length_s=20, stride_length_s=2)
        # the chunk length is how long in seconds MedASR segments audio and the stride length is the overlap between chunks.

        print("✅ Transcription complete.")
        return result['text'].replace('<epsilon>', '').replace('</s>', '').strip()      
    except Exception as e:
        print("🔥 An error occurred during audio transcription: {e}")
        return f"Sorry, an error occurred during transcription: {e}"