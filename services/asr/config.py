# services/asr/config.py
"""ASR-specific model configurations."""
from typing import Dict, Any
from core.app_config import active_infra_config

# --- ASR Model Definitions ---
ASR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MED_ASR": {
        "model_id": "google/MedASR",
        "pipeline_task": "automatic-speech-recognition",
    }
}
active_asr_config: Dict[str, Any] = ASR_CONFIGS["MED_ASR"]

# --- Loader & Pipeline Arguments ---
asr_loader_kwargs: Dict[str, Any] = {
    "pretrained_model_name_or_path": active_asr_config["model_id"],
    "trust_remote_code": True,
}

asr_model_loader_kwargs: Dict[str, Any] = {**asr_loader_kwargs}
if active_infra_config.get("asr_quantization_config"):
    asr_model_loader_kwargs["quantization_config"] = active_infra_config.get("asr_quantization_config")
    asr_model_loader_kwargs["device_map"] = "auto"