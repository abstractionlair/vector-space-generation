# Testing Larger Models for Concept Token Generation

## Model Recommendations for M4 Mac

### Primary Choice: Qwen2.5-3B-Instruct
- Latest architecture and training (2024)
- Excellent performance per parameter
- Should fit in memory with float16

### Installation and Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"

# Load with memory optimization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="mps",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
```

### Alternative: Test Multiple Models

```python
models_to_test = [
    "microsoft/phi-2",           # 2.7B - baseline
    "microsoft/Phi-3-mini-4k-instruct",  # 3.8B - newer Phi
    "Qwen/Qwen2.5-3B-Instruct",  # 3B - latest architecture
    "stabilityai/stablelm-2-1_6b-chat",  # 1.6B - if memory tight
]
```

### Key Advantages for Concept Tokens

Qwen2.5 and Phi-3 have:
1. **Better instruction following** - might understand "translate these vectors"
2. **More recent training** - exposed to more diverse tasks
3. **Improved architectures** - potentially better at handling unusual inputs
4. **Multilingual training** - actual translation capability

### Memory Management

If you hit memory limits:

```python
# Option 1: Use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatically balance between MPS and CPU
    offload_folder="offload",
    trust_remote_code=True
)

# Option 2: Clear cache aggressively
import gc
torch.mps.empty_cache()
gc.collect()

# Option 3: Use 8-bit quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

## Testing Protocol

1. Start with Qwen2.5-3B-Instruct
2. Test translation of concept tokens
3. Compare with Phi-2 if memory allows
4. Document differences in understanding

The hypothesis: newer, instruction-tuned models might show emergent understanding of the concept token translation task.
