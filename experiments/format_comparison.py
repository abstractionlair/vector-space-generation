#!/usr/bin/env python3
"""
Show the exact difference between normal and instruction format.
"""

from transformers import AutoTokenizer

def show_format_differences():
    """Display the exact differences between formats."""
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    print("="*60)
    print("FORMAT COMPARISON: Normal vs Instruction")
    print("="*60)
    
    # The actual content we want to generate from
    content = "The capital of France is"
    
    print(f"\nOriginal content: '{content}'")
    print("="*60)
    
    # 1. NORMAL FORMAT (just the text)
    print("\n1. NORMAL FORMAT (plain text)")
    print("-" * 40)
    normal_text = content
    normal_tokens = tokenizer(normal_text, return_tensors="pt")
    
    print(f"Text: '{normal_text}'")
    print(f"Token count: {normal_tokens['input_ids'].shape[1]}")
    print(f"Token IDs: {normal_tokens['input_ids'][0].tolist()}")
    print(f"Decoded: '{tokenizer.decode(normal_tokens['input_ids'][0])}'")
    
    # 2. INSTRUCTION FORMAT (wrapped in special tokens)
    print("\n2. INSTRUCTION FORMAT (chat template)")
    print("-" * 40)
    
    messages = [
        {"role": "user", "content": content}
    ]
    
    instruction_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"Formatted text:\n'''{instruction_text}'''")
    
    instruction_tokens = tokenizer(instruction_text, return_tensors="pt", add_special_tokens=False)
    print(f"\nToken count: {instruction_tokens['input_ids'].shape[1]}")
    print(f"Token IDs: {instruction_tokens['input_ids'][0].tolist()[:50]}...")  # First 50 tokens
    
    # Show the special tokens
    print("\n" + "="*60)
    print("SPECIAL TOKENS IN INSTRUCTION FORMAT")
    print("="*60)
    
    # Identify special tokens
    special_token_positions = []
    for i, token_id in enumerate(instruction_tokens['input_ids'][0].tolist()):
        token_str = tokenizer.decode([token_id])
        if token_id > 150000:  # Qwen's special tokens are high IDs
            special_token_positions.append((i, token_id, token_str))
    
    print(f"Found {len(special_token_positions)} special tokens:")
    for pos, tid, tstr in special_token_positions:
        print(f"  Position {pos}: ID={tid} → '{tstr}'")
    
    # 3. Show the structure
    print("\n" + "="*60)
    print("INSTRUCTION FORMAT STRUCTURE")
    print("="*60)
    
    print("""
The instruction format wraps content in a specific structure:

<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
[YOUR CONTENT HERE]<|im_end|>
<|im_start|>assistant
[MODEL GENERATES FROM HERE]

Key special tokens:
- <|im_start|> (ID: 151644) - Marks role beginning
- <|im_end|>   (ID: 151645) - Marks role ending
- These are SPECIAL tokens, not regular text
""")
    
    # 4. Why this matters
    print("="*60)
    print("WHY THIS MATTERS FOR EMBEDDINGS")
    print("="*60)
    
    print("""
NORMAL FORMAT:
- Input: "The capital of France is"
- Model sees: Regular tokens → Generates continuation
- With embeddings: Still works! Model just continues from embeddings

INSTRUCTION FORMAT:
- Input: <|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n
- Model sees: Special pattern → Enters "assistant mode"
- With embeddings: BREAKS! Model doesn't recognize the special tokens
  when they're passed as embeddings instead of token IDs

The model is trained to:
1. Look for EXACT token ID sequences (151644, ..., 151645)
2. When it sees this pattern, it "knows" to generate a response
3. When we use embeddings, even OF those tokens, it doesn't trigger
   the pattern recognition

It's like the difference between:
- Typing a password (normal format)
- Typing a password in a specific dialog box that checks for exact
  keystrokes, not just the resulting text (instruction format)
""")
    
    # 5. Demonstration with actual generation
    print("\n" + "="*60)
    print("DEMONSTRATION: How each format generates")
    print("="*60)
    
    print("""
NORMAL: "The capital of France is" →
  Model continues: "Paris, the city of lights..."
  (Simple continuation of the text)

INSTRUCTION: "<|im_start|>user\\nThe capital of France is<|im_end|>\\n<|im_start|>assistant\\n" →
  Model responds: "The capital of France is Paris."
  (Complete, helpful response in assistant voice)

WITH EMBEDDINGS:
- Normal format → Still generates something
- Instruction format → Empty output (pattern broken)
""")

if __name__ == "__main__":
    show_format_differences()