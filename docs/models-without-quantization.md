# Model Selection Without Quantization

## Models that Fit on M4 Mac WITHOUT Quantization

### Best Options (in float16):

1. **Qwen/Qwen2.5-1.5B-Instruct** (1.5B parameters)
   - ~3GB in float16
   - Plenty of headroom
   - Still very capable (2024 model)
   - Better than GPT-2 by far

2. **microsoft/Phi-2** (2.7B parameters)
   - ~5.4GB in float16
   - Should fit comfortably
   - Good instruction following

3. **microsoft/Phi-1_5** (1.3B parameters)
   - ~2.6GB in float16
   - Definitely fits
   - Surprisingly capable

4. **stabilityai/stablelm-2-1_6b** (1.6B parameters)
   - ~3.2GB in float16
   - Comfortable fit
   - Good for comparisons

### Why These Are Still Major Upgrades from GPT-2:

Even Qwen2.5-1.5B (the smallest) has:
- Instruction tuning
- Translation capability
- 2024 training data
- Better architecture
- 10x more parameters than GPT-2

### Implementation Without Quantization:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Pick model that fits comfortably
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Only 3GB!

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Still saves memory vs float32
    device_map="mps"
)

# Verify you can access embeddings
embeddings = model.get_input_embeddings()  # Should work fine
print(f"Embedding matrix shape: {embeddings.weight.shape}")
```

## Memory Check Script:

```python
import torch
import psutil

def check_memory_usage():
    # System memory
    mem = psutil.virtual_memory()
    print(f"Available RAM: {mem.available / 1024**3:.1f} GB")
    
    # MPS memory if available
    if torch.backends.mps.is_available():
        # Note: MPS doesn't have direct memory query like CUDA
        print("MPS is available")
    
    # Model memory estimate
    param_count = sum(p.numel() for p in model.parameters())
    memory_bytes = param_count * 2  # float16
    print(f"Model size: {memory_bytes / 1024**3:.1f} GB")

check_memory_usage()
```

## Recommendation:

Start with **Qwen2.5-1.5B-Instruct**:
- Definitely fits without quantization
- Modern architecture and training
- Has translation capability (unlike GPT-2)
- Clean float16 weights for your vector operations

Then if that works well, try Phi-2 (2.7B) which should also fit.

No quantization needed!
