# Qwen2.5-1.5B Implementation Specification

## Objective
Test whether modern instruction-tuned models (Qwen2.5-1.5B) can better interpret concept tokens than GPT-2, without needing quantization.

## Implementation Requirements

### 1. Create `src/qwen_concept_tokens.py`
Based on existing `concept_tokens.py` but adapted for Qwen2.5-1.5B.

### Key Changes Needed:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenConceptGenerator:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="mps"
        ).eval()
        
        # Verify embeddings access
        self.embeddings = self.model.get_input_embeddings()
        print(f"Embedding matrix shape: {self.embeddings.weight.shape}")
```

### 2. Adapt Translation Prompts for Instruction Model

Since Qwen2.5 is instruction-tuned, try:

```python
def translate_vectors_instruction(self, vector_sequence):
    """Use Qwen's instruction format"""
    
    # Qwen uses special tokens for instructions
    system_prompt = "You are a helpful assistant that can interpret encoded messages."
    
    # Try multiple formats:
    # Format 1: Direct instruction
    instruction = "Decode the following embedded vectors into natural text:"
    
    # Format 2: Completion style but with better model
    prefix = "The meaning of "
    suffix = " in plain English is:"
    
    # Test both approaches
```

### 3. Use Qwen's Chat Template (if needed)

```python
# Qwen might expect specific formatting
messages = [
    {"role": "system", "content": "You are a decoder."},
    {"role": "user", "content": "Interpret these vectors: [vectors]"}
]

# Apply chat template
text = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

### 4. Testing Protocol

Run the same tests as GPT-2:
1. Basic prompts: "The capital of France is", "2 plus 2 equals", "Once upon a time"
2. Different N values: 2, 3, 4 vectors
3. Compare all translation methods:
   - Completion style
   - Instruction style  
   - Nearest token
   - Direct feeding (no translation prompt)

### 5. Add Comparison Script

Create `experiments/compare_gpt2_vs_qwen.py`:
- Run both models on same prompts
- Compare their concept token distributions
- Compare translation quality
- Check if Qwen shows any emergent understanding

## Memory Management

```python
import gc
import torch

# After each test
torch.mps.empty_cache()
gc.collect()

# Monitor memory
import psutil
mem = psutil.virtual_memory()
print(f"Available RAM: {mem.available / 1024**3:.1f} GB")
```

## Expected Outcomes

### If successful (Qwen understands concept tokens better):
- Translations are more coherent
- Model shows understanding of probability mixtures
- Evidence for scale/training helping

### If unsuccessful (same problems as GPT-2):
- Still gets repetitive/garbage output
- Suggests need for:
  - Even larger models
  - Explicit training on this task
  - Or fundamental architectural limitation

## Git Setup

Before running, ensure everything is committed:

```bash
git add -A
git commit -m "Add Qwen2.5-1.5B implementation"
git push
```

This way results can be shared via GitHub if needed.
