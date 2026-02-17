# services/llm/engine.py
"""Initializes and provides the core LLM text-generation and image-comprehension function."""
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from core.app_config import active_infra_config
from services.llm.config import (
    active_model_config,
    MODEL_MODE,
)

# ==============================================================================
# SETUP: Define the Generator and its Initialization Function
# ==============================================================================
_initialized = False
_model = None
_processor = None

# This dictionary holds the loaded model and tokenizer
GeneratorObjects = Dict[str, Union[PreTrainedModel, PreTrainedTokenizer]]

def initialize_generator(
    model_id: str,
    torch_dtype: torch.dtype,
    quantization_config: BitsAndBytesConfig = None,
    device_map: str = "auto",
    **kwargs: Any # Catches unused arguments like 'task'
) -> GeneratorObjects:
    """
    Loads and initializes the model and tokenizer, returning them in a dictionary.
    """

    global _initialized
    global _model
    global _processor
    if _initialized:
        return {"model": _model, "processor": _processor}

    print("🚀 Initializing model and processor for direct generation...")
    print(f"   - Model: {model_id}")
    print(f"   - DType: {torch_dtype}")
    print(f"   - Quantization: {'Enabled' if quantization_config else 'Disabled'}")

    try:
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=device_map,
        )
        # Load a processor which can handle both text and images.
        _processor = AutoProcessor.from_pretrained(model_id)
        
        # Many models like Llama don't have a pad token, so we use the EOS token.
        if _processor.tokenizer.pad_token is None:
            _processor.tokenizer.pad_token = _processor.tokenizer.eos_token
        return {"model": _model, "processor": _processor}

    except Exception as e:
        print(f"🔥 CRITICAL: Model initialization failed. Error: {e}")
        return {}


def generate_text(
    generator: GeneratorObjects,
    messages: List[Dict[str, str]],
    image_input: Optional[str],
    **kwargs: Any
) -> str:
    """
    A stateless text generation function using the model.generate() method.
    This function replaces the call to the `pipe()` object.
    """
    if not generator:
        print("🔥 Generation failed: model and tokenizer not available.")
        return "Error: Generator not initialized."

    model = generator["model"]
    _processor = generator["processor"]
    
    print("Prepare inputs with the processor template")
    # --- Step 1: Prepare inputs using the processor ---
    # Mode: Use the processor's internal tokenizer to apply the chat template
    prompt_text = _processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if image_input:
        # VLM Mode: Load image and process text/image together
        inputs = _processor(text=prompt_text, images=image_input, return_tensors="pt").to(model.device)
    else:
        inputs = _processor.tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True).to(model.device)

    print("Generate the model outputs...")
    # 2. Call model.generate()
    outputs = model.generate(**inputs, **kwargs)

    # 3. Decode only the newly generated tokens, not the original prompt
    # --- Step 3: Decode ---
    input_ids_len = inputs["input_ids"].shape[1]
    newly_generated_ids = outputs[0, input_ids_len:]
    generated_text = _processor.decode(newly_generated_ids, skip_special_tokens=True)

    return generated_text


def generate_summary(prompt_key: str, medical_text: str, image_input: Optional[str] = None) -> str:
    """
    Generates a summary from medical text or provides comprehension for an image.
    - In 'TEXT' mode, it uses `medical_text`.
    - In 'VLM' mode, it uses both `image_input` (as a file path) and `medical_text` (as a prompt).
    """
    # --- Logic is split based on MODEL_MODE ---
    if MODEL_MODE == "VLM":
        if not image_input:
            return "Image input is required for VLM mode."
        if not medical_text or not medical_text.strip():
            return "A text prompt is required for VLM mode."

        try:
            llm_generator = initialize_generator(
                model_id=active_model_config["model_id"],
                **active_infra_config["llm_init_args"]
            )
            # The 'image-text-to-text' pipeline requires a structured chat format.
            # The user's message content is a list containing dicts for the image and text.
            messages = [
                {
                    "role": "system",
                    # The content is a LIST containing a DICT
                    "content": [{"type": "text", "text": active_model_config[prompt_key]}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_input},
                        {"type": "text", "text": medical_text}
                    ]
                }
            ]

            print("⏳ Applying chat template and generating text...")
            output = generate_text(
                generator=llm_generator,
                messages=messages,
                image_input=image_input, # Pass the image path/URL here
                **active_infra_config["llm_gen_args"]
            )
            print("✅ VLM generation complete.")

            # The output structure for image-to-text is slightly different
            return output.strip()
        except FileNotFoundError:
            print(f"🔥 Error: Image file not found at '{image_input}'")
            return f"Error: Image file not found at '{image_input}'"
        except Exception as e:
            print(f"🔥 An error occurred during VLM model inference: {e}")
            return "Sorry, an error occurred while processing your request."

    else: # Handles 'TEXT' mode
        if not medical_text or not medical_text.strip():
            return "Please enter a medical text to process."

        messages = [
            {"role": "system", "content": active_model_config[prompt_key]},
            {"role": "user", "content": medical_text},
        ]
        try:
            print("⏳ Generating text...")

            llm_generator = initialize_generator(
                model_id=active_model_config["model_id"],
                **active_infra_config["llm_init_args"]
            )

            result = generate_text(
                generator=llm_generator,
                messages=messages,
                image_input=None,
                **active_infra_config["llm_gen_args"]
            )

            print("✅ Text generation complete.")
            return result
        except Exception as e:
            print(f"🔥 An error occurred during text model inference: {e}")
            return "Sorry, an error occurred while processing your request."