#!/usr/bin/env python3
"""
Test whether GPT-2 can handle translation-like tasks at all.
This helps us understand if our translation approach is reasonable.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def test_translation_capability():
    """Test GPT-2's ability to handle translation-style completion prompts."""
    
    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    print("="*60)
    print("GPT-2 TRANSLATION CAPABILITY TEST")
    print("="*60)
    
    # Test various translation-style prompts
    test_prompts = [
        # Direct translation attempts
        ("Spanish to English translation: 'Hola' means", "hello"),
        ("The Spanish word 'gato' in English is", "cat"),
        ("'Bonjour' is French for", "hello/good morning"),
        
        # Completion-style translation format (like our concept tokens)
        ("The meaning of 'amor' in plain English is", "love"),
        ("The translation of 'merci' to English is", "thank you"),
        
        # Code-like translations
        ("Python code: print('hello') outputs", "hello"),
        ("The JavaScript alert('Hi') displays", "Hi"),
        
        # Symbol/emoji translation
        ("The emoji 😊 means", "happy/smile"),
        ("The symbol $ represents", "dollar/money"),
        
        # Nonsense "translation" (similar to concept tokens)
        ("The meaning of 'blurgh' in plain English is", "unknown"),
        ("The translation of 'xyzabc' to English is", "unknown"),
    ]
    
    print("\nTesting translation-style prompts:\n")
    
    for prompt, expected in test_prompts:
        print(f"Prompt: {prompt}")
        print(f"Expected: {expected}")
        
        # Generate completion
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"Generated: {generated}")
        print("-" * 40)
    
    print("\n" + "="*60)
    print("TESTING CONCEPT TOKEN STYLE PROMPTS")
    print("="*60)
    
    # Test prompts that mirror our concept token translation format
    concept_style_prompts = [
        "The meaning of [UNKNOWN] in plain English is",
        "The translation of [VECTOR] to English is",
        "The meaning of ### in plain English is",
        "The translation of @@@ to English is",
    ]
    
    print("\nTesting prompts similar to concept token translation:\n")
    
    for prompt in concept_style_prompts:
        print(f"Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Try multiple samples to see variation
        print("Generated samples:")
        for i in range(3):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            print(f"  {i+1}: {generated}")
        print("-" * 40)
    
    print("\n" + "="*60)
    print("TESTING FEW-SHOT EXAMPLES")
    print("="*60)
    
    # Try few-shot prompting to teach the pattern
    few_shot_prompt = """Examples of translations:
The meaning of 'casa' in plain English is house.
The meaning of 'libro' in plain English is book.
The meaning of 'agua' in plain English is water.
The meaning of 'sol' in plain English is"""
    
    print(f"Few-shot prompt:\n{few_shot_prompt}")
    
    inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f"\nGenerated completion: {generated}")
    print("(Expected: 'sun')")
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("""
1. GPT-2 has limited translation capability (not trained for it)
2. It may recognize some common words but not reliably translate
3. For concept tokens, we're asking it to 'translate' continuous vectors
4. The incoherent output we see is expected - GPT-2 doesn't know how
   to interpret arbitrary vectors as 'foreign language'
5. The completion format works, but GPT-2 treats unknown tokens as
   requiring creative completion rather than translation
""")


if __name__ == "__main__":
    test_translation_capability()