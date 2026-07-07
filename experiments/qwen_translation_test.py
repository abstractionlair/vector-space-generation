#!/usr/bin/env python3
"""
Test if Qwen can handle translation tasks in completion style (no instruction format).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_qwen_translation():
    """Test Qwen's translation capability without instruction format."""
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None
    ).eval()
    
    print("="*60)
    print("QWEN TRANSLATION CAPABILITY TEST (Completion Style)")
    print("="*60)
    
    # Test various translation prompts WITHOUT instruction format
    test_prompts = [
        # Direct translation attempts
        "Spanish to English: 'Hola' means",
        "The Spanish word 'gato' in English is",
        "'Bonjour' is French for",
        "The word 'amor' translates to",
        
        # Completion-style translation (like our concept tokens)
        "The meaning of 'casa' in plain English is",
        "The translation of 'merci' to English is",
        
        # Few-shot to help it understand
        "Spanish to English: 'si' means yes, 'no' means no, 'gracias' means",
        
        # Code translation (models often know this better)
        "Python code: print('hello') outputs",
        "In JavaScript, console.log('test') prints",
        
        # Math "translation"
        "In mathematics, 2+2 equals",
        "The square root of 16 is",
    ]
    
    print("\nTesting translation-style prompts (NO instruction format):\n")
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        
        # Tokenize WITHOUT instruction format
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2
            )
        
        # Decode only the generated part
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Generated: {generated}")
        print("-" * 40)
    
    print("\n" + "="*60)
    print("FEW-SHOT TRANSLATION TEST")
    print("="*60)
    
    # Try a more elaborate few-shot example
    few_shot_prompt = """Translate these Spanish words to English:
'casa' means house
'libro' means book  
'agua' means water
'sol' means sun
'luna' means"""
    
    print(f"Few-shot prompt:\n{few_shot_prompt}\n")
    
    inputs = tokenizer(few_shot_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.3,  # Lower temperature for more focused answer
            do_sample=True
        )
    
    generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"Generated: {generated}")
    print(f"(Expected: 'moon')")
    
    print("\n" + "="*60)
    print("CONCEPT-STYLE PROMPTS")  
    print("="*60)
    
    # Test prompts similar to what we'd use for concept tokens
    concept_style_prompts = [
        "The meaning of [UNKNOWN] in plain English is",
        "The translation of [VECTOR] to English is",
        "[CONCEPT] translates to",
        "Decode this: [X] means",
    ]
    
    print("Testing prompts we might use with concept tokens:\n")
    
    for prompt in concept_style_prompts:
        print(f"Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate 3 samples to see variation
        for i in range(2):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50
                )
            
            generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            print(f"  Sample {i+1}: {generated}")
        print("-" * 40)
    
    print("\n" + "="*60)
    print("COMPARISON WITH GPT-2")
    print("="*60)
    print("""
Based on these results, we can see if Qwen is better at translation than GPT-2.
If Qwen can translate basic words correctly, we could use completion-style
prompts for our concept token experiments instead of instruction format.
""")


if __name__ == "__main__":
    test_qwen_translation()