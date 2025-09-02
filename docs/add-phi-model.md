# Add Phi Model Support

## Objective
Test whether instruction-tuned models can better interpret concept tokens due to:
1. Better instruction following capabilities
2. Potential emergent understanding at larger scale
3. More robust handling of unusual inputs

## Implementation Requirements

### 1. Create new file: `src/phi_concept_tokens.py`
Based on `concept_tokens.py` but with:
- Support for Phi-2 or Phi-1.5 model
- Adjusted translation prompts for instruction model
- Memory-efficient loading (use `torch_dtype=torch.float16`)

### 2. Key Changes:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class PhiConceptGenerator:
    def __init__(self, model_name="microsoft/phi-1_5"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # For memory efficiency
            device_map="mps",
            trust_remote_code=True
        )
```

### 3. Translation Prompts:
Try both completion and instruction styles:
- Instruction: "Translate the following embedded vectors to text: [vectors]\nText:"
- System-style: "You are a decoder. Convert these embeddings to text: [vectors]"
- Direct: Just feed vectors after the prompt without translation framing

### 4. Memory Management:
- Use float16 for model weights
- Clear cache between runs: `torch.mps.empty_cache()`
- Consider 4-bit quantization if needed (using bitsandbytes)

## Testing Protocol

1. Run same test prompts as GPT-2
2. Compare:
   - Can Phi models interpret concept tokens better?
   - Do instruction-tuned models show different behavior?
   - Is there a scale threshold where this starts working?

## Hypothesis
Instruction-tuned models might:
- Better understand the meta-task of "translating" embedded vectors
- Have more robust attention that handles continuous inputs
- Show emergent capability to interpret concept tokens

Even if Phi models fail, this would suggest we need:
- Even larger models (GPT-3.5+)
- Specific training on this task
- Different approach entirely
