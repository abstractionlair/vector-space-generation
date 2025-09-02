# NOTE

Below is Claude's attempt to write up what we've done. I don't think it pulled together the info from mlultiple documents, conversations, and experiments all that well.

# Concept Tokens and the Limits of Continuous Generation in Language Models

## Abstract

We investigated whether language models can effectively generate and interpret text through continuous vector representations ("concept tokens") without discretizing to tokens at each step. Building on prior work in continuous reasoning (COCONUT, Soft Thinking), we tested whether deferring discretization preserves meaningful information that improves generation. Our experiments with GPT-2 and Qwen2.5-1.5B reveal that while models can generate meaningful probability-weighted embedding mixtures, they cannot interpret these as input—suggesting current models have a "write-only" continuous semantic space. Additionally, we discovered that instruction-tuned models exhibit mechanical dependencies on specific token IDs rather than semantic understanding of instruction markers, revealing brittleness in current instruction-following approaches.

## 1. Introduction

Standard autoregressive language generation forces discretization at every step: the model produces a probability distribution over tokens, samples one, and feeds only that token's embedding forward. This process discards the full distribution's information, potentially limiting the model's ability to maintain uncertainty or explore multiple semantic paths simultaneously.

Recent work has explored alternatives. COCONUT (Zhang et al., 2024) demonstrates that models can perform abstract reasoning in continuous latent space before generating output. Soft Thinking introduced probability-weighted token embeddings ("concept tokens") that preserve distributional information without requiring training. Building on these approaches, we investigate:

1. Can models generate coherent sequences using concept tokens?
2. Can models interpret concept tokens as meaningful input?
3. What information do concept tokens actually preserve?
4. How does instruction-tuning affect continuous input processing?

## 2. Related Work

### 2.1 Continuous Representations in LLMs

**COCONUT** (Chain of Continuous Thought) modifies transformers to enable "thinking" in latent space by feeding hidden states directly back as inputs. This requires specialized training with curriculum learning and shows breadth-first-like exploration benefits for reasoning tasks. Our approach differs by using training-free concept tokens and exploring interleaved continuous-discrete generation.

**Soft Thinking** uses probability-weighted mixtures of token embeddings without additional training, introducing stability mechanisms (entropy gating, distribution sharpening) to prevent off-manifold drift. We extend this with systematic analysis of what these mixtures represent and why translation fails.

**CoT2** provides theoretical grounding for continuous reasoning, demonstrating potential advantages for parallel exploration and backtracking. This motivates our investigation of whether similar benefits emerge at the generation level.

### 2.2 Interpretability and Continuous Spaces

The logit lens (nostalgebraist, 2020) and tuned lens work show that intermediate model states can be decoded to vocabulary distributions, suggesting continuous representations maintain semantic meaning throughout layers. Anthropic's work on superposition and sparse autoencoders reveals that models learn thousands of interpretable directions in activation space, supporting the hypothesis that continuous semantic operations are possible.

## 3. Methods

### 3.1 Concept Token Generation

Following Soft Thinking, we generate concept tokens as probability-weighted embedding mixtures:

```
e_t = Σ_i p_i * E[i]
```

Where p_i is the probability of token i from the model's distribution, and E[i] is its embedding. We apply:
- Top-k filtering (k=50)
- Distribution sharpening (p^α, α=1.5)
- Norm clipping to embedding range
- Entropy-based early stopping

### 3.2 Translation Approaches

We tested multiple methods to convert concept tokens back to text:

1. **Instruction-style**: "Please translate the following: [vectors]"
2. **Completion-style**: "The meaning of [vectors] in plain English is"
3. **Direct feeding**: Input vectors without framing
4. **Nearest-token decoding**: Map each vector to its closest discrete token

### 3.3 Models and Implementation

- **GPT-2 (124M)**: Baseline completion model without instruction tuning
- **Qwen2.5-1.5B-Instruct**: Modern instruction-tuned model with translation capabilities

All experiments used PyTorch with MPS acceleration on Apple Silicon.

## 4. Results

### 4.1 Concept Token Generation Succeeds

Models successfully generate meaningful concept tokens representing probability distributions over continuations:

| Prompt | Vector 1 | Vector 2 |
|--------|----------|----------|
| "The capital of France is" | 77% "Paris", 8% "located" | 58% ".", 36% "," |
| "2 plus 2 equals" | 65% " ", 30% "what" | 91% "4", 3% "3" |

These represent superpositions of possible continuations rather than a "richer" semantic language.

### 4.2 Translation/Interpretation Fails

Models cannot interpret concept tokens as meaningful input:

| Model | Translation Capability | Concept Token Interpretation |
|-------|----------------------|------------------------------|
| GPT-2 | Cannot translate Spanish | Produces repetitive garbage |
| Qwen | Can translate Spanish | Produces empty output (instruction) or nonsense (completion) |

The one interesting exception: Qwen with direct feeding produced "ариж" (attempting "Париж"/Paris in Cyrillic), suggesting partial semantic understanding but in an unexpected language space.

### 4.3 Instruction-Tuning Creates Mechanical Dependencies

Critical discovery: Qwen's instruction-following depends on seeing specific token IDs (151644 for `<|im_start|>`), not their embeddings:

| Input Method | Format | Output |
|--------------|--------|--------|
| Token IDs | Instruction | ✓ Normal |
| Embeddings | Instruction | ✗ Empty |
| Embeddings | Plain text | ✓ Works |

This reveals that instruction-tuning creates dual-track processing:
- **Semantic track**: Understanding via embeddings
- **Mechanical track**: Control flow via token ID detection

### 4.4 Information Preservation Analysis

Concept tokens preserve probability distributions, not new semantic content:
- Maintain uncertainty (77% Paris + 8% located vs. 100% Paris)
- Defer but don't avoid discretization
- Represent superpositions of discrete options

## 5. Technical Insights

### 5.1 The Asymmetry Problem

Models exhibit asymmetric capabilities with continuous representations:
- **Writing**: Can generate meaningful concept tokens ✓
- **Reading**: Cannot interpret concept tokens as input ✗

This suggests the continuous semantic space is "write-only" in current architectures.

### 5.2 Instruction-Tuning Brittleness

The discovery that instruction models require exact token IDs (not just their semantic embeddings) reveals:
1. Instruction-tuning doesn't create true semantic understanding of instructions
2. Models have hardcoded triggers rather than flexible comprehension
3. Current instruction-following is more mechanical than previously understood

### 5.3 Multilingual Semantic Space

The Cyrillic output ("ариж" for Paris) suggests concept tokens may navigate a language-agnostic semantic space, though models lack robust mechanisms to interpret these representations.

## 6. Discussion

### 6.1 Implications for Continuous Generation

Our results suggest that at the output layer, deferring discretization doesn't provide clear benefits—you still must eventually choose discrete tokens. The preserved "information" is simply the uncertainty about which token to emit, not a richer representation.

### 6.2 Potential for Internal Reasoning

While output-layer concept tokens show limited utility, the approach may be valuable for internal reasoning (as COCONUT demonstrates). Maintaining superpositions during chain-of-thought could allow exploring multiple reasoning paths simultaneously before committing to one.

### 6.3 Comparison with Prior Work

Our findings align with but extend previous research:
- Confirms Soft Thinking's feasibility of training-free concept tokens
- Validates COCONUT's finding that models struggle with continuous inputs without training
- Provides new evidence about instruction-tuning's mechanical nature

## 7. Limitations and Future Work

### 7.1 Limitations
- Limited to smaller models due to hardware constraints
- Translation approaches may not optimally decode continuous representations
- Concept tokens tested only at input/output interfaces, not mid-layers

### 7.2 Future Directions
1. Test with larger models (GPT-3.5+) via API adaptations
2. Explore mid-layer continuous representations with tuned lens decoders
3. Fine-tune models specifically for concept token interpretation
4. Investigate continuous reasoning for multi-path exploration

## 8. Conclusion

We demonstrated that while language models can generate meaningful probability-weighted embedding mixtures (concept tokens), they cannot interpret these as input—revealing an asymmetric "write-only" continuous semantic space. The concept tokens represent superpositions of discrete possibilities rather than a richer continuous language. Additionally, we discovered that instruction-tuned models depend on mechanical token ID recognition rather than semantic understanding of instructions. These findings suggest that true bidirectional continuous communication with language models will require architectural changes or specialized training, while also revealing unexpected brittleness in current instruction-following mechanisms.

## Acknowledgments

We thank the developers of the transformers library, and the authors of COCONUT, Soft Thinking, and related work for establishing the foundations this research builds upon.

## References

- Zhang et al. (2024). "COCONUT: Chain of Continuous Thought"
- [Soft Thinking paper] (2024). "Training-Free Continuous Generation"
- [CoT2 paper] (2024). "Continuous Chain of Thought"
- nostalgebraist (2020). "The Logit Lens"
- Anthropic (2024). "Sparse Autoencoders and Superposition"

## Appendix: Key Code Snippets

### Concept Token Generation
```python
def generate_concept_token(logits, temperature=0.8):
    probs = F.softmax(logits / temperature, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=50)
    top_k_probs = top_k_probs ** 1.5  # Sharpening
    top_k_probs = top_k_probs / top_k_probs.sum()
    
    concept_embedding = sum(
        p * embeddings.weight[idx] 
        for p, idx in zip(top_k_probs, top_k_indices)
    )
    return concept_embedding
```

### Discovery of Instruction-Tuning Brittleness
```python
# This works (generates output)
output = model.generate(input_ids=ids, max_new_tokens=50)

# This fails with instruction format (empty output)
embeds = model.get_input_embeddings()(ids)
output = model.generate(inputs_embeds=embeds, max_new_tokens=50)

# But works with plain text format
plain_embeds = model.get_input_embeddings()(plain_ids)
output = model.generate(inputs_embeds=plain_embeds, max_new_tokens=50)
```