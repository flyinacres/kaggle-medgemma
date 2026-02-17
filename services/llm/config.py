# services/llm/config.py
"""LLM-specific model configurations."""
from typing import Dict, Any
from core.app_config import PROMPT_FILE_PATH, CONVERSATIONAL_PROMPT_FILE_PATH, MODEL_MODE, active_infra_config

# --- Prompt Loading ---
try:
    system_prompt_text = PROMPT_FILE_PATH.read_text()
except FileNotFoundError:
    print(f"🔥 CRITICAL: Prompt file not found at '{PROMPT_FILE_PATH}'. Using fallback.")
    system_prompt_text = "You are a helpful medical assistant."

try:
    conversational_prompt_text = CONVERSATIONAL_PROMPT_FILE_PATH.read_text()
except FileNotFoundError:
    print(f"🔥 CRITICAL: Prompt file not found at '{CONVERSATIONAL_PROMPT_FILE_PATH}'. Using fallback.")
    conversational_prompt_text = "You are a helpful medical assistant."

# --- LLM Model Definitions ---
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "TEXT": {
        "model_id": "google/medgemma-1.5-4b-it",
        "pipeline_task": "text-generation",
        "system_prompt": system_prompt_text,
        "conversational_prompt": conversational_prompt_text
    },
    # --- MODIFIED: VLM configuration now uses MedGemma ---
    "VLM": {
        "model_id": "google/medgemma-1.5-4b-it", # Unified model for both text and vision
        "pipeline_task": "image-text-to-text",        # Correct pipeline for multimodal input
        "system_prompt": system_prompt_text,
        "conversational_prompt": conversational_prompt_text
    }
}
active_model_config: Dict[str, Any] = MODEL_CONFIGS.get(MODEL_MODE, MODEL_CONFIGS["TEXT"])
print(f"✅ LLM Service Mode: '{MODEL_MODE}' using model '{active_model_config['model_id']}'")


# --- Pipeline & Generation Arguments ---
GENERATION_KEYS: set = {"max_new_tokens", "do_sample", "temperature", "top_p"}


llm_model_init_kwargs: Dict[str, Any] = {}
if active_infra_config.get("llm_quantization_config"):
    llm_model_init_kwargs["quantization_config"] = active_infra_config["llm_quantization_config"]

generation_kwargs: Dict[str, Any] = {
    key: active_infra_config[key] for key in GENERATION_KEYS if key in active_infra_config
}
# For image-to-text, we want the full generated text, not just the completion.
# For text-generation with chat format, this should be False. We handle this in the engine.
if MODEL_MODE == "VLM":
    generation_kwargs['return_full_text'] = True
else:
    generation_kwargs['return_full_text'] = False