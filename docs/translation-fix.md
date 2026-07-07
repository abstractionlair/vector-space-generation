# Translation Fix Specification

## Problem
The current translation method produces repetitive garbage when trying to interpret concept tokens. The issue may be the "instruction-style" prompt with a completion model.

## Proposed Solution
Modify the `translate_vectors` method in `concept_tokens.py` to use completion-style prompts that GPT-2 would recognize from training.

## Implementation Requirements

1. **Change the translation prompt format** from:
   - Current: "Please translate the following: [vectors]"
   - To: "The meaning of [vectors] in plain English is"
   - Alternative: "The translation of [vectors] to English is"

2. **Use model.generate() properly**:
   - Use `max_new_tokens` instead of `max_length` to control generation length
   - Add `repetition_penalty=1.2` to reduce repetitive output
   - Ensure proper stopping with EOS token

3. **Test both prompt formats** to see which works better

4. **Key changes needed**:
   ```python
   # Structure: prefix + vectors + suffix
   prefix = "The meaning of "
   suffix = " in plain English is"
   
   # Then concatenate: prefix_embeddings + vectors + suffix_embeddings
   # And use model.generate() with proper parameters
   ```

5. **Also implement a fallback**: nearest-token decoding
   - For each concept token, find the nearest discrete token in embedding space
   - Decode those tokens to see what the concept tokens are "close to"
   - This gives us a baseline understanding even if translation fails

## Expected Behavior
- Initial output might be as good/bad as baseline GPT-2
- May degrade into repetition if model can't maintain coherence with concept tokens
- This would be informative about partial vs. complete failure

## Test After Implementation
Run the same test prompts and compare:
1. Original translation method (instruction style)
2. New completion-style translation
3. Nearest-token decoding
4. Baseline GPT-2 for reference
