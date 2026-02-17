# services/asr/engine.py
"""
Provides audio transcription using the MedASR pipeline.

Uses the Hugging Face pipeline for automatic speech recognition with
Google's MedASR model, which is tuned for medical terminology. Audio
is chunked to handle longer recordings, with overlap between chunks
to avoid cutting words at boundaries.

The pipeline is instantiated on each call. If performance becomes an
issue, this is a candidate for lazy-loading into a module-level global.
"""

from transformers import pipeline


def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes an audio file to text using the MedASR pipeline.

    Cleans common output artifacts from the model (<epsilon> and </s> tokens)
    before returning the result.

    Args:
        audio_file_path: Path to the audio file to transcribe.

    Returns:
        The transcribed text string, or an error message if transcription fails.
    """
    if not audio_file_path:
        return ""

    try:
        print("⏳ Transcribing audio...")

        pipe = pipeline("automatic-speech-recognition", model="google/medasr")

        # chunk_length_s: how long each audio segment is in seconds
        # stride_length_s: overlap between chunks to avoid cutting words at boundaries
        result = pipe(audio_file_path, chunk_length_s=20, stride_length_s=2)

        print("✅ Transcription complete.")
        return result['text'].replace('<epsilon>', '').replace('</s>', '').strip()

    except Exception as e:
        print(f"🔥 An error occurred during audio transcription: {e}")
        return f"Sorry, an error occurred during transcription: {e}"