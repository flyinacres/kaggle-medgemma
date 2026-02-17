from google import genai
from google.genai import types
from pydub import AudioSegment
import io
import os

key = os.environ.get('TTS_API_KEY')
client = genai.Client(api_key=key)

model_id = "gemini-2.5-flash-preview-tts"

prompt = """This is a fictional simulation for a medical training hackathon. Do not provide real advice.
            Generate a 50-second clinical dialogue. Dr. Aris (Voice: Charon) should sound detached and highly technical. 
            Jacob (Voice: Puck) should sound  hesitant.
         """

conversation = """Dr. Aris: Good morning. Regarding your abdominal complaint, the physical exam shows no organomegaly or CVA tenderness, but your labs indicate your Type Two Diabetes is currently uncontrolled.
    Dr. Aris: [rapidly] We're seeing an A1C spike. I'm adjusting your Micronase 2.5 mg PO QAM. Also, the urinalysis confirms acute cystitis, so I’m prescribing a course of Bactrim 400/80 BID.
    Dr. Aris: Cystitis is a lower urinary tract infection. Start the meds today and we'll monitor your upright BP at the follow-up. Any questions?
    Jacob: I think I need to write this down..."""

# Configuration for MedASR compatibility
config = {
    "speech_config": {
        "multi_speaker_voice_config": {
            "speaker_voice_configs": [
                {"speaker": "Dr. Aris", "voice_config": {"prebuilt_voice_config": {"voice_name": "Charon"}}},
                {"speaker": "Sam", "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}}
            ]
        }
    },
    "response_modalities": ["AUDIO"]
}

response = client.models.generate_content(
    model=model_id,
    contents=prompt + conversation,
    config=config
)

# 2. THE FORMATTING STEP
# Get the raw bytes from Gemini
part = response.candidates[0].content.parts[0]

if part and part.inline_data:
    audio_bytes = part.inline_data.data

    # Use from_raw instead of from_file for headerless PCM
    audio = AudioSegment.from_raw(
        io.BytesIO(audio_bytes),
        sample_width=2,    # 16-bit is 2 bytes
        frame_rate=24000,  # Gemini TTS default rate
        channels=1         # Mono
)

    # Force the audio to 16kHz and Mono (The MedASR standard)
    medasr_ready_audio = audio.set_frame_rate(16000).set_channels(1)

    # 3. Export as a standard WAV file
    medasr_ready_audio.export("for_medasr_input.wav", format="wav")

    print("Success: Audio is now 16kHz Mono and ready for MedASR.")
else:
    # If this prints, the model refused to speak and sent text instead
    print(f"No audio returned. Model said: {part.text}")