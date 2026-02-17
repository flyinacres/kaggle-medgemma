# services/llm/engine.py
"""Initializes and provides the core LLM text-generation and image-comprehension function."""
from typing import Optional, Dict, Any, Union, List
import torch

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers import (
    DynamicCache
)

from services.llm.kvcache import initialize_kv_cache

from core.app_config import active_infra_config
from services.llm.config import (
    active_model_config,
    MODEL_MODE,
)

def build_message(mode: str, prompt_key: str, medical_text: str, image_input: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build the message for the LLM based upon the prompt type (prompt_key)
    and the mode that the system is operating in, the user text, and
    possibly an image (in VLM mode)

    If this is called to generate a message when a KV cache is in place, 
    set prompt_key = None
    """
    prompt_str = None
    message_list = []

    # Add the prompt information, if a key is specified
    # When using the KV Cache this is not used (or it will duplicate the prompt)
    if prompt_key is not None:
        prompt_str = active_model_config[prompt_key]

        # The 'image-text-to-text' (VLM) pipeline requires a structured chat format.
        # The user's message content is a list containing dicts for the image and text.
        vlm_system_prompt = [
                    {
                        "role": "system",
                        # The content is a LIST containing a DICT
                        "content": [{"type": "text", "text": prompt_str}]
                    }
                ]
        text_system_prompt = [
                    {"role": "system", "content": active_model_config[prompt_key]}
                ]
        if mode == "VLM":
            prompt_dict = vlm_system_prompt
        else:        
            prompt_dict = text_system_prompt

        message_list.extend(prompt_dict)

    # Handle the message format.  This is always needed
    vlm_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_input},
                        {"type": "text", "text": medical_text}
                    ]
                }
            ]

    text_message = [
                {"role": "user", "content": medical_text}
            ]
    
    # Pick the prompt and message based upon the mode
    if mode == "VLM":
        message_dict = vlm_message
    elif mode == "TEXT":
        message_dict = text_message
    else:
        # Should never get here...
        print(f"An invalid mode {mode} specified, cannot construct prompt!")
        raise ValueError("Invalid LLM mode specified")

    # construct the final list, containing necessary components
    message_list.extend(message_dict)

    return message_list

        


# ==============================================================================
# SETUP: Define the Generator and its Initialization Function
# ==============================================================================
_initialized = False
_model = None
_processor = None

# This dictionary holds the loaded model and tokenizer
GeneratorObjects = Dict[str, Union[PreTrainedModel, PreTrainedTokenizer, DynamicCache]]

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
            # Also update the model's config to know this
            _processor._id = _processor.tokenizer.eos_token_id

        print("Initialize kv cache")
        kv_cache, cache_length = initialize_kv_cache(_model, _processor)

        return {"model": _model, "processor": _processor, "KV_cache": kv_cache, "Cache_Length": cache_length}

    except Exception as e:
        print(f"🔥 CRITICAL: Model initialization error. Error: {e}")
        return {}


def generate_text(
    generator: GeneratorObjects,
    message_str: str,
    image_input: None,
    **kwargs: Any
) -> str:
    """
    A stateless text generation function using the model.generate() method.
    This function replaces the call to the `pipe()` object.
    """
    if not generator:
        print("🔥 Generation failed: model and tokenizer not available.")
        return "Error: Generator not initialized."

    print("in generate text")
    model = generator["model"]
    processor = generator["processor"]
    kv_cache = generator["KV_cache"]
    cache_length = generator["Cache_Length"]

    # --- Step 1: Prepare inputs  ---
    # This is really just the user text NOT the prompt text for the caching case I am testing
    prompt_text = message_str

    if image_input:
        # VLM Mode: Load image and process text/image together
        inputs = processor(text=prompt_text, images=image_input, return_tensors="pt").to(model.device)
    else:
        inputs = processor.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    print(f"Cache generation (inference) full input_ids: {inputs['input_ids']}")

    # Calculations to help with caching
    past_len = kv_cache.get_seq_length()  # Use the cache's reported length
    cache_position = torch.arange(past_len, past_len + inputs['input_ids'].shape[1], device=model.device)

    # Attention mask for entire sequence: cache + new tokens
    full_attention = torch.ones(
        (1, cache_length + inputs['input_ids'].shape[1]), 
        # CAUTION -- this is likely based upon config!!!
        # TODO--figure out how to fix this
        dtype=torch.float32, 
        device=model.device
    )

    # Interesting to track generation time...
    import time

    print("before step 2")
    # 2. Call model.generate()
    start_time = time.perf_counter()
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=full_attention, 
        cache_position=cache_position,
        past_key_values=kv_cache, 
        **kwargs)
    end_time = time.perf_counter()

    generation_time = end_time - start_time
    output_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens generated: {output_tokens}")
    print(f"Tokens/sec: {output_tokens/generation_time:.1f}")
    
    # 3. Decode only the newly generated tokens, not the original prompt
    # Length of what was actually passed to generate (user portion only)
    user_input_len = inputs["input_ids"].shape[1] - cache_length
    newly_generated_ids = outputs[0, user_input_len:]
    generated_text = processor.decode(newly_generated_ids, skip_special_tokens=True)
    return generated_text

def generate_summary(prompt_key: str, medical_text: str, image_input: Optional[str] = None) -> str:
    """
    Generates a summary from medical text or provides comprehension for an image.
    - In 'TEXT' mode, it uses `medical_text`.
    - In 'VLM' mode, it uses both `image_input` (as a file path) and `medical_text` (as a prompt).
    """
    if not medical_text or not medical_text.strip():
        return "No text was provided to analyze"

    try:
        print("⏳ Initializing Generator...")
        llm_generator = initialize_generator(
            model_id=active_model_config["model_id"],
            **active_infra_config["llm_init_args"]
        )

        print(f"Building messages for prompt key {prompt_key} in mode {MODEL_MODE}")
        actual_prompt_key = prompt_key
        # If this is for an initial message, as opposed to follow up chat, and the KV cache exists, set the key to None
        if prompt_key == "system_prompt" and llm_generator["KV_cache"] is not None:
            actual_prompt_key = None
            print("Setting the actual prompt key to none, as KV cache is present and this is an initial query")
            #print(f"Should clear the actual prompt key, but I want to see how it runs with the ye olde prompt style {prompt_key}")
        
            # Since this is the caching path, build the string manually
            # CAUTION: Only supporting text mode at this point
            message_string = medical_text + "<end_of_turn>\n"
        else: 
            # This is the case where I am building the message string for without caching
            messages = build_message(MODEL_MODE, actual_prompt_key, medical_text, image_input)

        print("⏳ Generating results...")
        result = generate_text(
            generator=llm_generator,
            message_str = message_string,
            image_input=image_input,
            **active_infra_config["llm_gen_args"]
        )

        print("✅ Generation complete.")

        if MODEL_MODE == "VLM":
            # The output structure for image-to-text is slightly different
            return result[0]['generated_text'][2]['content'].strip()
        else:
            return result

    except FileNotFoundError:
        print(f"🔥 Error: Image file not found at '{image_input}'")
        return f"Error: Image file not found at '{image_input}'"

    except Exception as e:
        print(f"🔥 An error occurred during text model inference: {e}")
        return "Sorry, an error occurred while processing your request."

