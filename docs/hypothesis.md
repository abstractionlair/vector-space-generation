# Project Hypothesis and Goals

## Core Hypothesis

### The Continuous Language Hypothesis
Language models have learned to understand a continuous "language" where every point in their embedding space represents a valid semantic unit - a "word" in their native language of thought. This emerges naturally from training, even though models are only ever shown discrete tokens.

### Why This Would Emerge
Training pressure incentivizes models to:
1. **Learn smooth representations**: Similar concepts should have similar vectors (for generalization)
2. **Interpolate between known points**: The space between "hot" and "cold" should represent "warm"
3. **Compose meanings continuously**: Vector arithmetic should approximate semantic composition
4. **Maximize information density**: Use the full space, not just discrete points

This is analogous to how interpretability research finds that models learn abstract concepts, directions in activation space that correspond to features like "truthfulness" or "formality." The model learns these not because they were explicitly trained, but because they're efficient representations for minimizing loss.

### The Discretization Problem
Standard generation forces a "translation" at every step:
- Vector state → probability distribution → sample token → lookup embedding → next vector state

This is like forcing someone to translate their thoughts to English after every word, even when thinking internally. The information lost at each discretization includes:
- The full probability distribution (collapsed to single token)
- Precise position in semantic space (quantized to nearest vocabulary item)
- Potential meanings "between" tokens (forced to choose one)

### What We're Testing
If models truly understand this continuous language, they should be able to:
1. Generate coherent sequences of vectors without discretization
2. Translate these vector sequences back to natural language when asked
3. Potentially express concepts that don't map cleanly to single tokens

### The Key Insight
This isn't about models learning a *new* capability. It's about recognizing and utilizing a capability they already have. Just as models learn features and concepts without explicit training (as shown by interpretability research), they've likely learned to navigate continuous semantic space as a natural consequence of:

1. **Embedding structure**: The initial embedding layer creates a continuous space
2. **Gradient-based learning**: Updates create smooth paths through the space
3. **Attention mechanisms**: Computing weighted averages creates intermediate points
4. **Residual connections**: Adding vectors creates new valid points in the space

The model has been navigating this continuous space all along internally - we're just testing whether it can do so without being forced to discretize at each step.

### Refinements to the Hypothesis

#### Layer-Specific Languages (Weaker Version)
The vector "language" might differ across layers, with each layer having its own semantic space. However, adjacent layers must maintain compatibility to communicate. Evidence from interpretability research (e.g., "On the Biology of Large Language Models") shows that:
- Early layers are tightly coupled to surface features of the input language
- Middle layers work with more abstract concepts
- Final layers converge back toward output language representations

This suggests that even if each layer has its own "dialect," the first and last layers share enough commonality to enable our proposed vector generation and translation approach.

#### Bounded Continuous Regions (Partial Space Version)
Rather than the entire vector space being meaningful, models may have learned to use:
1. **The convex hull** of token embeddings - filling in gaps between discrete tokens
2. **A fuzzy boundary region** extending somewhat beyond token embeddings
3. **Continuous submanifolds** within the high-dimensional space

This is still a richer space than just the discrete token points. The coupling of early and late layers to the target language would create natural boundaries - the model learns to work within regions that can ultimately be decoded back to language.

#### Supporting Evidence
The fact that early and late layers are most tightly coupled to the input/output language actually *supports* our approach:
- Vector generation starts from embedded tokens (early layer compatible)
- Vector sequences maintain some relationship to the token "cloud"
- Translation back to text leverages the late layers' language coupling
- Middle layers can explore more abstract spaces while maintaining chain of compatibility

### Training Considerations
If vector-space generation doesn't work with pretrained models, this might not indicate the hypothesis is wrong, but rather that models need exposure to this mode during training. A potential path forward would be reinforcement learning post-training:
- Start with N=1 (standard generation)
- Gradually increase N during training
- Reward successful translation of N-vector sequences
- Build up the model's ability to maintain coherence across longer vector chains

This would explicitly teach the model to use its latent continuous language capability, rather than hoping it emerges from standard training alone. The fact that models aren't explicitly trained for vector-to-vector generation might be the only barrier, not a fundamental limitation of the architecture.

### Future Extensions: Vector-Space Reasoning
If basic vector generation proves successful, a natural extension would be allowing Chain-of-Thought (CoT) reasoning entirely in vector space:
- Models could perform multi-step reasoning without discretizing intermediate thoughts
- This might preserve more nuanced logical connections between steps
- Could potentially enable forms of reasoning that don't map cleanly to natural language
- The "thinking" process would be truly in the model's native representation

This would test whether reasoning benefits from staying in continuous space, where logical operations might be more naturally represented as vector transformations rather than discrete symbol manipulation.

### Relationship to Recent Work
Recent work has explored similar territory:
- **COCONUT**: Uses hidden state feedback for abstract reasoning (requires training)
- **Soft Thinking**: Uses probability-weighted token embeddings ("concept tokens") without training
- **CoT2**: Theoretical framework for continuous reasoning

Our approach builds directly on Soft Thinking's concept token method but introduces:
1. **Interleaved generation**: Alternate between continuous (up to N steps) and discrete modes throughout generation
2. **Partial translation and continuation**: Periodically translate concept tokens to discrete tokens, then resume continuous generation
3. **Explicit translation mechanism**: "Please translate the following:" framing treats vector sequences as messages
4. **Systematic N variation**: Testing how checkpoint frequency affects generation quality
5. **Information preservation analysis**: Measuring what's retained through continuous vs. discrete steps

The key novelty is the alternating pattern: generate continuously → translate → continue generating continuously. This hasn't been explored and could reveal how models balance continuous and discrete representations.

### Implementation Variations to Test
We'll test two approaches for generating continuous representations:

1. **Concept Tokens** (following Soft Thinking): Probability-weighted mixtures of token embeddings
   - Stays within the convex hull of learned embeddings
   - Proven stable without training

2. **Raw Hidden States**: Direct use of pre-vocabulary-projection vectors
   - More direct test of the "native language" hypothesis
   - May be stabilized by keeping N small
   - Could reveal whether concept tokens are necessary or just one solution

Both approaches will use our interleaved generation pattern (generate → translate → continue). Comparing them will help identify whether stability comes from:
- Staying within the embedding manifold (concept tokens)
- Limiting continuous steps (small N)
- Or both

## Research Questions
1. What information is preserved when generating in vector space that is lost in token generation?
2. How does this information preservation affect observable properties of generated text?
3. Is there an optimal number of vector steps (N) before discretization?
4. Can models effectively translate their own vector sequences back to natural language without additional context?
5. Does interleaving continuous and discrete generation provide benefits over single-mode generation?

## What We're Testing
We remain agnostic about specific effects. Information preservation might affect:
- **Coherence**: Maintaining consistent themes or logic
- **Creativity**: Exploring regions "between" standard tokens
- **Accuracy**: Better preservation of factual or computational information
- **Consistency**: Less variance across multiple generations
- **Error correction**: Whether periodic discretization prevents drift
- **Other emergent properties**: Unknown effects we'll discover

## Experimental Approach
Test generation with N vectors before discretization, where N is an upper limit (model can stop early based on entropy). Three modes:
1. **Continuous-only**: Generate all vectors, translate once at end
2. **Interleaved**: Alternate between N vectors and translation, continue generating
3. **Comparison**: Both concept tokens and raw hidden states

Translation uses: `Embed("Please translate the following: ") + vector_sequence` without providing original context, testing whether vector sequences are self-contained messages.

## Success Metrics
### Metric 0: Basic Functionality
- Does the output make any sense at all?
- Can we successfully translate vector sequences to text?

### Quantitative Metrics
- Information-theoretic measures (KL divergence, entropy)
- Distance from standard token embeddings over time
- Task-specific accuracy (arithmetic, logic, factual Q&A)
- Generation diversity metrics

### Qualitative Analysis
- What kinds of differences emerge between methods?
- Are there consistent patterns in where methods diverge?

## Key Insight
We're not claiming this will definitively improve generation. We're testing whether the model's "native language of thought" (continuous vectors) has different properties than forced token discretization, and characterizing what those differences are.