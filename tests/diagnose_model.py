# diagnose_model.py (final check for NaN/inf)

# --- SETUP ---
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
MODEL_ID = "google/medgemma-1.5-4b-it"

# --- EXECUTION ---

# 1. Verify PyTorch CUDA
print("--- 1. Verifying PyTorch CUDA ---")
if not torch.cuda.is_available():
    print("❌ ERROR: PyTorch cannot see CUDA.")
    exit()
print(f"✅ CUDA is available. Device: {torch.cuda.get_device_name(0)}")
print("-" * 30)

# 2. Load components
print("\n--- 2. Loading Tokenizer and Model ---")
print("\n--- 2. Loading Tokenizer and Model (WITHOUT QUANTIZATION) ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, # Use the model's native precision
        device_map="auto",
    )
    print("✅ Components loaded in bfloat16.")
    print("-" * 30)
except Exception as e:
    print(f"❌ ERROR: Failed to load components. Error: {e}")
    exit()

# 3. Apply vocab resize fix
print("\n--- 3. Checking and Fixing Vocabulary Size ---")
tokenizer_vocab_size = len(tokenizer)
model_vocab_size = model.get_input_embeddings().weight.shape[0]
if tokenizer_vocab_size != model_vocab_size:
    print("Mismatch detected. Applying fix...")
    model.resize_token_embeddings(tokenizer_vocab_size)
    print(f"✅ Model resized to {len(tokenizer)}")
else:
    print("✅ Vocabulary sizes match.")
print("-" * 30)


# 4. FINAL TEST: Manual forward pass and logit inspection
print("\n--- 4. Inspecting Model Output for Numerical Stability ---")
try:
    # Prepare inputs
    prompt = "This is a test."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Perform a single forward pass to get the raw logits
    # We use torch.no_grad() as we are not training
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print("✅ Forward pass completed.")

    # Check for NaN and Inf values in the output logits
    has_nan = torch.isnan(logits).any()
    has_inf = torch.isinf(logits).any()

    print(f"Are there any NaN values in the logits? -> {has_nan.item()}")
    print(f"Are there any Inf values in the logits? -> {has_inf.item()}")

    if has_nan or has_inf:
        print("\n❌ CRITICAL ERROR: Numerical instability detected.")
        print("The model is producing invalid values (NaN/Inf), which causes the CUDA crash.")
        print("This confirms an incompatibility between the model's quantization and your GPU's compute capabilities.")
    else:
        print("\n✅ Logits appear stable. The cause of the crash may be something else.")

except Exception as e:
    print("\n❌ ERROR: The manual forward pass itself failed.")
    raise e