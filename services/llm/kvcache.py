import torch
import os

from transformers import (
    DynamicCache
)
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
from services.llm.config import (
    active_model_config
)

cache_path = "./medgemma-1.5-4b-it_system_prompt.pt"

def initialize_kv_cache(model, processor) -> tuple[DynamicCache, int]:
    """
    Loads a pre-computed KV cache from disk if it exists.
    If not, it generates the cache, saves it to disk, and returns it.

    Args:
        model: The loaded Hugging Face causal language model.
        processor: The loaded processor/tokenizer.

    Returns:
        A past_key_values tuple (the KV cache) on the same device as the model.
    """
    # Check if the cache file exists on disk
    if os.path.exists(cache_path):
        print(f"Loading existing KV cache from {cache_path}")
        # Load the cache, which was saved on CPU
        loaded = torch.load(cache_path)
        reconstructed_layers = []

        for layer_info in loaded['layer_info']:
            if layer_info['type'] == 'DynamicLayer':
                layer = DynamicLayer()
            elif layer_info['type'] == 'DynamicSlidingWindowLayer':
                layer = DynamicSlidingWindowLayer(layer_info['sliding_window'])
            
            layer.keys = layer_info['keys'].to(model.device)
            layer.values = layer_info['values'].to(model.device)
            layer.is_initialized = layer_info['is_initialized']
            if layer_info['cumulative_length'] is not None:
                layer.cumulative_length = layer_info['cumulative_length']
            reconstructed_layers.append(layer)

        new_cache = DynamicCache()
        new_cache.layer_class_to_replicate = loaded['layer_class_to_replicate']
        new_cache.offloading = loaded['offloading']
        cache_length = loaded['cache_length']
        new_cache.layers = reconstructed_layers
        print(f"Loaded cache attrs: {vars(new_cache)}")
        return new_cache, cache_length

    system_prompt_text = active_model_config["system_prompt"]

    # --- Generation Logic (if cache file does not exist) ---
    print("KV cache not found. Generating a new one...")
    
    # 1. Format the system prompt manually--using the chat template makes conflicts with caching

    prompt_message_portion = f"<bos><start_of_turn>user\n{system_prompt_text}\n"

    # 2. Tokenize the formatted prompt
    inputs = processor.tokenizer(prompt_message_portion, return_tensors="pt").to(model.device)

    # 3. Perform a single forward pass to generate the cache
    # Use torch.no_grad() for efficiency as we are not training
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    new_kv_cache = outputs.past_key_values
    print("KV cache generated successfully.")

    cache_length = inputs['input_ids'].shape[1]  # This is your slice point
    
    print(f"Cache creation input_ids: {inputs['input_ids']}")
    print(f"Cache length: {cache_length}")

    # 4. Save the cache to disk for future runs
    cache_data = {
        'cache_length': cache_length,  # Used to slice the user and system portions of the tokenized stream apart...
        'layer_class_to_replicate': type(new_kv_cache.layer_class_to_replicate).__name__ if new_kv_cache.layer_class_to_replicate else None,
        'offloading': new_kv_cache.offloading,
        'layer_info': [
            {
                'type': type(layer).__name__,
                'sliding_window': layer.sliding_window if hasattr(layer, 'sliding_window') else None,
                'keys': layer.keys.cpu(),
                'values': layer.values.cpu(),
                'is_initialized': layer.is_initialized,
                'cumulative_length': layer.cumulative_length if hasattr(layer, 'cumulative_length') else None
            }
            for layer in new_kv_cache.layers
        ]
    }
    torch.save(cache_data, cache_path)
    print(f"KV cache saved to {cache_path}")

    # 5. Return the generated cache (which is already on the correct device)
    return new_kv_cache, cache_length