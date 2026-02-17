# --- Setup: Imports and Dummy Objects (for demonstration) ---
# In your actual code, you would already have your model and processor loaded.
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Function Definition ---
def initialize_kv_cache(model, processor, system_prompt_text: str, cache_path: str):
    """
    Loads a pre-computed KV cache from disk if it exists.
    If not, it generates the cache, saves it to disk, and returns it.

    Args:
        model: The loaded Hugging Face causal language model.
        processor: The loaded processor/tokenizer.
        system_prompt_text: The static system prompt content.
        cache_path: The file path to save/load the cache file.

    Returns:
        A past_key_values tuple (the KV cache) on the same device as the model.
    """
    # Check if the cache file exists on disk
    if os.path.exists(cache_path):
        print(f"Loading existing KV cache from {cache_path}")
        # Load the cache, which was saved on CPU
        kv_cache_cpu = torch.load(cache_path)
        # Move the loaded cache to the model's current device
        kv_cache = tuple(
            (key.to(model.device), value.to(model.device)) for key, value in kv_cache_cpu
        )
        return kv_cache

    # --- Generation Logic (if cache file does not exist) ---
    print(f"KV cache not found. Generating a new one...")
    
    # 1. Format the system prompt using the chat template
    # IMPORTANT: add_generation_prompt must be False for pre-filling context.
    system_message = [{"role": "system", "content": system_prompt_text}]
    prompt_text = processor.tokenizer.apply_chat_template(
        system_message,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 2. Tokenize the formatted prompt
    inputs = processor.tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # 3. Perform a single forward pass to generate the cache
    # Use torch.no_grad() for efficiency as we are not training
    with torch.no_grad():
        outputs = model(**inputs)
    
    new_kv_cache = outputs.past_key_values
    print("KV cache generated successfully.")

    # 4. Save the cache to disk for future runs
    # It is best practice to move tensors to the CPU before saving
    kv_cache_cpu = tuple(
        (key.cpu(), value.cpu()) for key, value in new_kv_cache
    )
    torch.save(kv_cache_cpu, cache_path)
    print(f"KV cache saved to {cache_path}")

    # 5. Return the generated cache (which is already on the correct device)
    return new_kv_cache


# --- Execution Example ---
if __name__ == '__main__':
    # This block demonstrates how you would use the function
    MODEL_NAME = "google/gemma-2b" # Using a small model for the example
    SYSTEM_PROMPT = "You are a helpful medical assistant providing concise and accurate information."
    CACHE_PATH = "./gemma-2b_system_prompt.pt"
    
    # 1. Load your model and processor as usual
    # Using bfloat16 for memory efficiency if available
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda")
    processor = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Initialize (load or generate) the KV cache
    # This is the single line you'd add to your app's startup routine
    kv_cache = initialize_kv_cache(model, processor, SYSTEM_PROMPT, CACHE_PATH)
    
    # 3. Verify the cache is loaded
    print(f"\nKV Cache loaded. Type: {type(kv_cache)}")
    print(f"Number of layers in cache: {len(kv_cache)}")
    # Each element of the tuple is a (key, value) tensor pair for a decoder layer
    # Shape is typically [batch_size, num_heads, sequence_length, head_dim]
    print(f"Shape of key tensor for the first layer: {kv_cache[0][0].shape}")
    print(f"Device of key tensor for the first layer: {kv_cache[0][0].device}")