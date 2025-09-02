# Vector-Space Generation Implementation Specification v2

## Core Hypothesis
Language models understand continuous vector space as their native language. We test whether allowing models to "speak" in this language (via concept tokens or raw states) before translating to discrete tokens reveals different properties than forced discretization at each step.

## Novel Contributions
1. **Interleaved generation pattern**: Alternate between continuous and discrete modes throughout generation
2. **Explicit translation mechanism**: "Please translate the following:" treats vector sequences as messages
3. **Systematic N-limit study**: How many continuous steps before quality degrades?
4. **Continuation after translation**: Resume continuous generation from discrete checkpoints

## Technical Approach

### Method 1: Concept Tokens (Primary)
Following Soft Thinking, create probability-weighted mixtures of token embeddings:
```python
logits = model(sequence)
probs = softmax(logits / temperature)
probs = sharpen(probs, alpha=1.5)  # p^alpha / sum(p^alpha)
probs = top_k(probs, k=50)
concept_token = sum(probs[i] * token_embeddings[i] for i in top_k)
concept_token = clip_norm(concept_token)  # Stay in embedding range
```

### Method 2: Raw Hidden States (Experimental)
Use pre-vocabulary-projection hidden states directly:
```python
hidden = model.get_hidden_states(sequence)[-1]  # Last layer
# Optional: Apply linear aligner to project to embedding space
if use_aligner:
    hidden = aligner(hidden)  # Learned via least squares offline
```

### Interleaved Generation Protocol
```
1. Generate up to N concept tokens (with entropy-based early stop)
2. Translate via: Embed("Please translate the following: ") + vector_sequence
3. Let model generate discrete tokens as translation
4. Continue from translated checkpoint with N more concept tokens
5. Repeat until done
```

## Experimental Design

### Primary Experiments
All without additional context initially:

1. **Baseline**: Standard token-by-token generation
2. **Continuous-only**: Generate N vectors, translate once at end
3. **Interleaved**: Alternate between N vectors and translation
4. **Raw states**: Same as above but with hidden states (expect instability)

### Key Parameters
- N ∈ {2, 3, 4, 8} (upper limit, can stop early on entropy)
- Temperature: 0.8 for generation
- Top-k: 50 tokens
- Sharpening α: 1.5
- Entropy threshold: 1.0 bits (stop if below for 2 consecutive steps)

### Secondary Experiments
After primary results:
1. **Self-containment test**: Add original context, measure translation improvement
2. **Position sensitivity**: Test with/without position alignment
3. **Aligner for raw states**: Simple linear projection
4. **Alternative translations**: Prompt-free mechanical decoding

## Success Metrics

### Functionality (Pass/Fail)
- Translations are readable and coherent
- No catastrophic repetition or drift
- Entropy stays reasonable (not collapsing to near-zero)

### Quantitative
- **Information preserved**: KL divergence at each step
- **Semantic coherence**: BLEU/perplexity on translations
- **Diversity**: Type-token ratio, distinct n-grams
- **Consistency**: Variance across multiple runs
- **Distance from manifold**: Cosine to nearest token embedding

### Qualitative
- Does interleaving help or hurt coherence?
- What types of content benefit from continuous generation?
- Where do the methods diverge in interesting ways?

## Test Prompts

### Simple Completions
- "The capital of France is"
- "2 plus 2 equals"
- "Once upon a time"

### Reasoning Tasks
- "If it's raining, I need an umbrella. It's raining, so"
- "Alice is taller than Bob. Bob is taller than Carol. Therefore"

### Creative Generation
- "In a world where gravity worked backwards"
- "The recipe for happiness includes"

## Implementation Order

### Week 1: Foundation
1. Set up GPT-2 small with MPS support
2. Implement concept token generation
3. Test basic N=2 generation
4. Implement "Please translate" mechanism

### Week 2: Core Experiments
1. Run primary experiments (concept tokens)
2. Implement interleaving
3. Test different N values
4. Analyze entropy and drift patterns

### Week 3: Extensions
1. Try raw hidden states (with small N)
2. Optional: implement aligner
3. Run self-containment tests
4. Statistical analysis

### Week 4: Analysis
1. Compare all methods systematically
2. Identify failure modes
3. Document interesting divergences
4. Write up findings

## Technical Details

### Model: GPT-2 small (124M)
- Hidden dimension: 768
- Vocabulary: 50,257 tokens
- Layers: 12

### RoPE Considerations
- Concept tokens: Position-agnostic, no special handling needed
- Raw states: Position-sensitive, may need alignment
- For translation: Use fresh positions 0...N-1 (no context replay initially)

### Entropy Monitoring
```python
def should_stop(probs, threshold=1.0, history=[]):
    entropy = -sum(p * log(p) for p in probs if p > 0)
    history.append(entropy)
    if len(history) >= 2 and all(h < threshold for h in history[-2:]):
        return True  # Stop if entropy low for 2 steps
    return False
```

## Expected Outcomes

### Likely Successes
- Concept tokens with N=2-3 should work
- Interleaving might prevent drift
- Some tasks may show reduced variance

### Likely Failures
- Raw states without aligner will probably collapse
- Large N (>4) may drift off-manifold
- Translation might lose nuance

### Interesting Questions
- Does periodic discretization act as error correction?
- What information is preserved in continuous vs discrete?
- Is there an optimal rhythm for interleaving?

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Semantic drift | Entropy monitoring, small N, norm clipping |
| Translation failure | Fallback to nearest-token decoding |
| Position misalignment | Start with position-free experiments |
| Raw state collapse | Keep as ablation, try aligner |

## Dependencies
- PyTorch 2.0+ with MPS support
- Transformers 4.30+
- NumPy, Matplotlib
- 16GB RAM (for GPT-2 small)

## What This Tests
We're testing whether the forced discretization at each step loses information that could be preserved by staying in continuous space. This isn't about proving continuous is "better" - it's about characterizing the differences and understanding what each approach preserves or loses.

## Change Log
- 2024-01-20 v1: Initial specification
- 2024-01-20 v2: Incorporated feedback on RoPE, positioning, and recent work