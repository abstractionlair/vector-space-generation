# Vector-Space Generation

Experimental implementation testing whether language models can generate in continuous vector space before discretizing to tokens.

## Project Status
- Testing concept token generation with multiple models
- Current: Implementing Qwen2.5-1.5B comparison

## Repository Structure
```
vector-space-generation/
├── src/
│   ├── baseline.py           # Standard GPT-2 generation
│   ├── concept_tokens.py     # GPT-2 concept token generation  
│   └── qwen_concept_tokens.py # Qwen2.5 implementation (in progress)
├── experiments/
│   ├── test_translation_capability.py
│   └── compare_gpt2_vs_qwen.py (planned)
├── docs/
│   ├── implementation-spec.md
│   ├── project-instructions.md
│   └── *.md (various specifications)
└── results/
    └── (experimental outputs)
```

## Current Findings
- GPT-2 cannot interpret concept tokens as input
- Concept tokens do capture semantic mixtures (e.g., 55% "capital", 20% "city")
- Testing whether modern models show emergent understanding

## Running Experiments
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python src/qwen_concept_tokens.py
```
