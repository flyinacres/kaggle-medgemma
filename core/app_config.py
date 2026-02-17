# core/app_config.py
"""
Handles application-wide configuration: environment detection, secrets, and hardware profiles.
This module is the single source of truth for the runtime environment.
"""

import os
import torch
from pathlib import Path
from transformers import BitsAndBytesConfig
from typing import Dict, Any

# --- Environment Detection & Constants ---
IS_KAGGLE_ENV: bool = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
# !!! May need to edit these paths on Kaggle to point to your prompts
BASE_DIR: Path = Path("/kaggle/input/medgemma-laurie-prompts") if IS_KAGGLE_ENV else Path(__file__).parent.parent.resolve()
PROMPT_FILE_PATH: Path = BASE_DIR / "prompts" / "json_prompt.txt"
CONVERSATIONAL_PROMPT_FILE_PATH: Path = BASE_DIR / "prompts" / "conversational_prompt.txt"

# --- Secret & Environment Variable Loading ---
if IS_KAGGLE_ENV:
    print("✅ Running in Kaggle environment.")
    try:
        from kaggle_secrets import UserSecretsClient # type: ignore
        user_secrets = UserSecretsClient()
        os.environ["HUGGING_FACE_HUB_TOKEN"] = user_secrets.get_secret("HUGGING_FACE_HUB_TOKEN")
        print("✅ Kaggle secret 'HUGGING_FACE_HUB_TOKEN' loaded into environment.")
    except (ImportError, Exception) as e:
        print(f"🔥 Failed to get Kaggle secret. Model loading may fail. Error: {e}")
else:
    print("✅ Running in a local environment.")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Local .env file processed.")
    except ImportError:
        print("⚠️ `python-dotenv` not found. Relying on manually set environment variables.")

# --- Core Infrastructure Configuration ---
INFRA_ABILITY: str = os.environ.get("INFRA_ABILITY", "LOW").upper()
# MODEL_MODE "VLM" for text and image
MODEL_MODE: str = os.environ.get("MODEL_MODE", "TEXT").upper() # Used by LLM service

# Defines hardware and performance configurations for all potential services.
INFRA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "LOW": {
        "llm_init_args": {
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
            "torch_dtype": torch.bfloat16,
            "device_map": "auto"
        },
        "llm_gen_args": {
            "max_new_tokens": 1536,
            "do_sample": False
        },
        "asr_init_args": {
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
            "torch_dtype": torch.float16
        }
    },
    "HIGH": {
        "llm_init_args": {
            "quantization_config": None,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto"
        },
        "llm_gen_args": {
            "max_new_tokens": 2048,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9
        },
        "asr_init_args": {
            "quantization_config": None,
            "torch_dtype": torch.float16
        }
    },
    "APPLE_SILICON": {
        "llm_init_args": {
            "quantization_config": None,
            "torch_dtype": torch.float32,
            "device_map": "mps"
        },
        "llm_gen_args": {
            "max_new_tokens": 3096,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9
        },
        "asr_init_args": {
            "quantization_config": None,
            "torch_dtype": torch.float32
        }
    }
}

active_infra_config: Dict[str, Any] = INFRA_CONFIGS.get(INFRA_ABILITY, INFRA_CONFIGS["LOW"])
print(f"✅ App config loaded. Mode: '{INFRA_ABILITY}'")