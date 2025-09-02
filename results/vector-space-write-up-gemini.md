Experimental Write-Up: Probing Continuous Vector Generation in Language Models
Abstract
This report details an empirical investigation into the "Continuous Language Hypothesis," which posits that language models operate in a continuous vector space and that forcing discretization into tokens at each generation step results in information loss. We tested whether pretrained models could generate sequences of continuous vectors ("concept tokens") and subsequently translate them back into coherent text. Using GPT-2 and the instruction-tuned Qwen2.5-1.5B, we found that while models can generate semantically meaningful concept vectors that represent a superposition of potential next tokens, they largely fail to interpret these vectors when fed back as input. Our key finding is that this failure is not necessarily a conceptual limitation but a technical one, particularly in instruction-tuned models, which appear to rely on specific token IDs as control triggers, a mechanism that is bypassed when using direct vector inputs (inputs_embeds). This reveals a "write-only" nature of the model's continuous semantic space and highlights the brittleness of current instruction-tuning paradigms.

1. Introduction and Hypothesis
Standard autoregressive language models generate text token-by-token, a process that involves mapping a high-dimensional internal state to a probability distribution over a discrete vocabulary, sampling a single token, and then re-embedding it to continue the process. This "discretization problem" potentially discards a significant amount of information at each step, such as the full probability distribution and nuanced semantic positions "between" tokens.

Our core hypothesis was that language models have learned a continuous "language of thought" where every point in their embedding space is a valid semantic unit. We aimed to test this by allowing a model to generate a sequence of N vectors directly in this continuous space before attempting to translate the entire sequence back to natural language.

This research is informed by prior work like Soft Thinking, which uses probability-weighted token embeddings ("concept tokens") to improve reasoning, and COCONUT, which uses hidden state feedback for a continuous chain of thought. Our novelty lies in testing an interleaved generation pattern (generate continuously → translate → continue) and systematically investigating the translation mechanism itself as a test of the model's bidirectional understanding of its own representations.

2. Experimental Setup & Methodology
Models Used
To isolate the effects of model capability and training style, we used two distinct models on a local M4 MacBook Pro:

GPT-2 (124M): A foundational, completion-style model with no instruction-tuning.

Qwen2.5-1.5B-Instruct: A modern, more capable instruction-tuned model.

Concept Token Generation
Following the method from "Soft Thinking," we generated "concept tokens" not from raw hidden states, but as a probability-weighted mixture of token embeddings from the model's vocabulary. This ensures the vectors remain within the convex hull of the learned embedding manifold, a more stable approach for pretrained models.

Translation Mechanisms Tested
The core of our experiment was testing how a model would interpret a sequence of its own concept tokens. We tried several methods:

Instructional Prompt: Please translate the following: [vector_sequence]

Completion Prompt: The meaning of [vector_sequence] in plain English is

Nearest-Token Decoding: A diagnostic method where each concept vector is mapped to the single closest token in the embedding space to see what it "represents."

3. Results
Phase 1: Experiments with GPT-2
Baseline Failure: Initial tests confirmed that GPT-2 is a poor baseline for this task, as it lacks fundamental translation capabilities even with regular text (e.g., it cannot translate simple Spanish words).

Translation Output: When translating concept tokens, GPT-2 produced incoherent, nonsensical text with both instructional and completion-style prompts.

Meaningful Vectors: Despite the translation failure, nearest-token decoding was revealing. For the prompt "The capital of France is," the generated concept vectors were closest to the tokens "the" and "capital." This showed that the vectors themselves were capturing relevant semantic information. 

Phase 2: Experiments with Qwen2.5-1.5B
Switching to a more capable, instruction-tuned model produced a different, more insightful failure mode.

Instructional Prompt Failure: When using the instructional chat template with inputs_embeds, the model produced no output at all. This was true even when feeding it perfect, non-superpositional embeddings of real tokens, not just concept tokens. 

Completion Prompt Partial Success: The completion-style prompt generated some thematically related, but still incoherent, output. For example, after generating concept vectors for "The capital of France is," the translation mentioned "Paris" and "government." This was a significant improvement over GPT-2. 

Nearest-Token Success: Nearest-token decoding was again highly successful, showing that the concept vectors for our test prompts correctly corresponded to "Paris .", " 4", and ", there". This confirmed the "writer" side of the model was working perfectly. 

4. Analysis & Discussion
The results lead to several key insights that refine the initial hypothesis.

The "Writer vs. Reader" Asymmetry
The experiments consistently show that models can write into continuous vector space (generating meaningful superpositions) but cannot read from it (interpreting those vectors as input). The nearest-token decoding proves the generated vectors are semantically valid, yet the model fails to translate them coherently. This suggests the continuous representation is a "write-only" internal state.

The Instruction-Tuning Brittleness
The most significant finding is why the instruction-tuned Qwen model produced empty output. Further testing revealed that instruction-tuned models are trained to respond to specific special token IDs (e.g., <|im_start|>, <|im_end|>) as mechanical control triggers. When we provide input via inputs_embeds, the model never sees these integer IDs, so its instruction-following behavior is never activated. This is not a semantic failure to understand, but a mechanical failure to trigger a trained pattern. This reveals a fundamental brittleness in how current instruction-tuning works. 

What Concept Tokens Really Are
Our results suggest that concept tokens are not a "richer language" but rather a superposition of discrete options. For the prompt "The capital of France is," the concept vector represented a probability distribution like 

77% "Paris" + 8% "located" + ....  This is not a new word with a blended meaning, but rather the preservation of uncertainty over the existing vocabulary. This aligns with the findings of COCONUT, suggesting that the main benefit of continuous representations is for internal reasoning (exploring multiple paths) rather than for generating final output.

5. Conclusion & Future Work
Our experiment, while failing to produce a model that can "speak" continuously, yielded valuable insights:

Asymmetric Capability: Pretrained models can generate meaningful continuous vector representations but cannot interpret them when fed back as input.

Instruction-Tuning is Brittle: Modern instruction-tuned models may rely on hardcoded token ID triggers, making them fail when these are bypassed, even with semantically identical vector inputs.

Concept Tokens Preserve Uncertainty: Continuous vectors act as superpositions of discrete tokens, a mechanism more suited for internal Chain-of-Thought reasoning than for final output generation.

Future work should focus on leveraging these "write-only" representations where they are most powerful: in the intermediate steps of a reasoning process. Furthermore, research into fine-tuning models to explicitly "read" concept tokens could potentially unlock a truly bidirectional continuous language capability.