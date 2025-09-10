**Scope**
- Review whether the implementation supports the conclusions in `README.md`.
- Sources reviewed: `CLAUDE.md`, `IMPLEMENTATION.md`, `README.md`, `src/`, `experiments/`, and `results/`.

**Bottom Line**
- The core conclusions in the README are supported by the implementation and experiments present in the repo.
- Evidence across `src/*concept_tokens*.py` and `experiments/*qwen*` aligns with claims that: (1) GPT‑2 is insufficient for the translation tasks; (2) Qwen2.5 works with completion‑style prompts but not reliably via instruction formatting when vectors are inserted as embeddings; (3) the generated “concept tokens” behave as superpositions over candidate continuations rather than an alternative, richer “vector language”.

**What The Code Demonstrates**
- Concept tokens (probability-weighted embedding mixtures)
  - Implemented in `src/concept_tokens.py` and `src/qwen_concept_tokens.py` via top‑k, temperature, and sharpening; norm clipping keeps vectors on-manifold.
  - Functions: `create_concept_token(...)`, `generate_vector_sequence(...)` generate multiple vectors before discretization; both track entropy and support early stopping.
- Feeding embeddings directly (bypassing tokenization)
  - Both GPT‑2 and Qwen paths accept `inputs_embeds` to feed sequences of vectors, as required by the hypothesis tests.
- Translation back to text
  - GPT‑2 path: `translate_vectors(...)` supports “completion” and “instruction” styles and a nearest‑token fallback; experiments show weak performance for translation tasks (`experiments/test_translation_capability.py`).
  - Qwen path: both instruction and completion are tried; completion works, instruction is brittle when vectors are inserted (`src/qwen_concept_tokens.py`, `experiments/qwen_completion_translation.py`, `experiments/qwen_translation_test.py`, `experiments/output_analysis.py`).
- Superposition behavior
  - Metrics and prints expose top candidate tokens and nearest tokens to the concept vector, showing mixtures like the README’s “Paris / located / a” pattern (`create_concept_token(...): metrics['top_tokens']`, nearest‑token decoding in both GPT‑2 and Qwen code paths; also demonstrated in `experiments/qwen_completion_translation.py`).
- Entropy gating
  - Entropy is computed per step and used for early stopping in both GPT‑2 and Qwen implementations (`generate_vector_sequence(..., entropy_threshold=...)`).

**Alignment With README Conclusions**
- “GPT‑2 cannot interpret concept tokens as input; even basic translation fails.”
  - Supported by `experiments/test_translation_capability.py` showing GPT‑2’s poor translation performance and by GPT‑2 translation attempts in `src/concept_tokens.py`.
- “Qwen2.5 works for translation in completion style; instruction mode fails when embeddings are inserted.”
  - Supported by: `src/qwen_concept_tokens.py` (both modes implemented), `experiments/qwen_completion_translation.py` (completion works), `experiments/output_analysis.py` and `experiments/test_with_special_tokens.py` (instruction mode depends on exact token IDs and breaks with embedded vectors).
- “Concept tokens encode superpositions of different possible continuations, not a single ‘expanded-language’ meaning.”
  - Supported by: top‑token distributions and nearest‑token decoding in both GPT‑2 and Qwen flows (see `create_concept_token(...): metrics['top_tokens']`, nearest decoding utilities, and experiment outputs). The behavior matches the README’s described examples.
- “Entropy gating and small N are used; interleaving not critical to conclusions.”
  - Entropy gating: implemented and used. Interleaving: not fully built as a looped pipeline, but non-essential given early observations. README’s statement that interleaving “didn’t matter” is consistent with the code’s focus on vector generation + translation rather than continued interleaved runs.

**Notable Gaps / Differences**
- Interleaved continuation (translate chunk, continue in vectors) is described in README/Docs but not implemented end‑to‑end as a repeated pipeline; current code focuses on one‑shot vector generation then translation. Given the early signal, this does not undermine the README’s conclusions.
- Raw hidden states (vs. concept tokens) are outlined in docs but not implemented; README only speculates about them in “Loose Ends,” not as a basis for conclusions.
- Quantitative evaluation of “superposition” is observational (top‑k weights, nearest tokens) rather than formal metrics over datasets. Still sufficient to support the qualitative conclusion stated.

**Supporting Pointers**
- GPT‑2 concept tokens and translation: `src/concept_tokens.py`, `experiments/test_translation_capability.py`.
- Qwen concept tokens and translation: `src/qwen_concept_tokens.py`, `experiments/qwen_completion_translation.py`, `experiments/qwen_translation_test.py`.
- Instruction-format brittleness with embeddings: `experiments/output_analysis.py`, `experiments/test_with_special_tokens.py`, `experiments/test_embedding_bypass.py`.
- Results write-ups describing observed behavior: `results/vector-space-write-up-claude.md`.

**Assessment**
- Largely sufficient: The repository’s implementations and experiments substantively support the README’s conclusions about (1) GPT‑2’s limitations, (2) Qwen’s completion‑style viability, and (3) concept tokens behaving as superpositions rather than an “internal continuous language” readable by current models.
- Limitations are appropriately caveated in the README and do not contradict the presented evidence.

**Lightweight Suggestions**
- If desired, add a minimal interleaved driver that loops: vectors → translate → continue, to complete the spec even if results remain unchanged.
- Consider a small script to aggregate and save per‑step metrics (entropy, nearest token, distances) to `results/` for a few fixed prompts, making the qualitative finding more reproducible.
- If exploring raw states later, add a simple linear aligner and document the failure modes explicitly.

