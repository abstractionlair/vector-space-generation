#!/usr/bin/env python3
"""
Test if we're properly bypassing encoding and whether Qwen can process embedded inputs at all.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_embedding_bypass():
    """Test different ways of passing embeddings to ensure we're doing it right."""
    
    # Load Qwen
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None
    ).eval()
    
    embeddings = model.get_input_embeddings()
    
    print("="*60)
    print("TEST 1: Normal text generation (baseline)")
    print("="*60)
    
    # First, normal generation to verify model works
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.7
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Normal generation: {generated}\n")
    
    print("="*60)
    print("TEST 2: Pass same text via embeddings")
    print("="*60)
    
    # Now try passing the SAME text but via embeddings
    input_ids = inputs["input_ids"]
    input_embeds = embeddings(input_ids)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            temperature=0.7
        )
    
    # Skip the input length when decoding
    generated_ids = outputs[0, input_embeds.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Via embeddings: {generated}\n")
    
    print("="*60)
    print("TEST 3: Mix real tokens with noise")
    print("="*60)
    
    # Create a sequence: real tokens + random noise + real tokens
    part1 = "The answer is"
    part2 = " obviously"
    
    part1_ids = tokenizer(part1, return_tensors="pt")["input_ids"].to(device)
    part2_ids = tokenizer(part2, return_tensors="pt")["input_ids"].to(device)
    
    part1_embeds = embeddings(part1_ids)
    part2_embeds = embeddings(part2_ids)
    
    # Create random noise in embedding space
    noise = torch.randn(1, 2, embeddings.weight.shape[1], device=device, dtype=part1_embeds.dtype)
    # Scale noise to similar magnitude as real embeddings
    noise = noise * embeddings.weight.std()
    
    # Concatenate: real + noise + real
    mixed_embeds = torch.cat([part1_embeds, noise, part2_embeds], dim=1)
    attention_mask = torch.ones(1, mixed_embeds.shape[1], device=device)
    
    print(f"Input structure: '{part1}' + [2 noise vectors] + '{part2}'")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=mixed_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            temperature=0.7
        )
    
    generated_ids = outputs[0, mixed_embeds.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: {generated}\n")
    
    print("="*60)
    print("TEST 4: Instruction format with embedded vectors")
    print("="*60)
    
    # Try using the instruction format but with embedded vectors
    messages = [
        {"role": "user", "content": "Complete this: The sky is"}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # First, normal generation with this prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.7
        )
    
    generated = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"Normal instruction: {generated}")
    
    # Now via embeddings
    input_embeds = embeddings(inputs["input_ids"])
    attention_mask = torch.ones_like(inputs["input_ids"])
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            temperature=0.7
        )
    
    generated_ids = outputs[0, input_embeds.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Instruction via embeddings: {generated}\n")
    
    print("="*60)
    print("TEST 5: What happens with ONLY noise vectors?")
    print("="*60)
    
    # Just noise, no real tokens
    pure_noise = torch.randn(1, 5, embeddings.weight.shape[1], device=device, dtype=torch.float16)
    pure_noise = pure_noise * embeddings.weight.std()
    attention_mask = torch.ones(1, 5, device=device)
    
    print("Passing 5 pure noise vectors...")
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs_embeds=pure_noise,
                attention_mask=attention_mask,
                max_new_tokens=10,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            
            generated_ids = outputs[0, 5:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"Generated from pure noise: '{generated}'")
        except Exception as e:
            print(f"Error with pure noise: {e}")
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("""
If Test 2 works but our concept tokens don't, the issue is likely:
1. The concept tokens are too far from the embedding manifold
2. The sharpening/weighting creates invalid representations
3. Dimension mismatch or dtype issues

If Test 3 shows the model handles noise poorly, that explains our issue.
If Test 5 produces empty output like ours, we've found the problem.
""")

if __name__ == "__main__":
    test_embedding_bypass()