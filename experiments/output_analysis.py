#!/usr/bin/env python3
"""
Systematic analysis of when Qwen produces output vs empty responses.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def analyze_output_patterns():
    """Test various conditions to understand output patterns."""
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    ).eval()
    
    embeddings = model.get_input_embeddings()
    
    print("="*60)
    print("SYSTEMATIC ANALYSIS: When do we get output?")
    print("="*60)
    
    results = []
    
    # Test 1: Normal token IDs
    print("\n1. NORMAL TOKEN IDS")
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    results.append(("Normal tokens", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 2: Same text via embeddings (no instruction format)
    print("\n2. EMBEDDINGS (no instruction format)")
    input_embeds = embeddings(inputs['input_ids'])
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10
        )
    result = tokenizer.decode(outputs[0][input_embeds.shape[1]:], skip_special_tokens=True)
    results.append(("Embeddings (plain)", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 3: Instruction format with token IDs
    print("\n3. INSTRUCTION FORMAT WITH TOKEN IDS")
    messages = [{"role": "user", "content": "Say hello"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    results.append(("Instruction tokens", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 4: Instruction format via embeddings
    print("\n4. INSTRUCTION FORMAT VIA EMBEDDINGS")
    input_embeds = embeddings(inputs['input_ids'])
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10
        )
    result = tokenizer.decode(outputs[0][input_embeds.shape[1]:], skip_special_tokens=True)
    results.append(("Instruction embeds", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 5: Just the assistant tag via embeddings
    print("\n5. JUST ASSISTANT TAG VIA EMBEDDINGS")
    assistant_text = "<|im_start|>assistant\n"
    assistant_ids = tokenizer(assistant_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    assistant_embeds = embeddings(assistant_ids)
    attention_mask = torch.ones_like(assistant_ids)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=assistant_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10
        )
    result = tokenizer.decode(outputs[0][assistant_embeds.shape[1]:], skip_special_tokens=True)
    results.append(("Assistant tag only", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 6: Random noise embeddings
    print("\n6. RANDOM NOISE EMBEDDINGS")
    noise = torch.randn(1, 5, embeddings.weight.shape[1], device=device, dtype=torch.float16)
    noise = noise * embeddings.weight.std()
    attention_mask = torch.ones(1, 5, device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=noise,
            attention_mask=attention_mask,
            max_new_tokens=10
        )
    result = tokenizer.decode(outputs[0][5:], skip_special_tokens=True)
    results.append(("Pure noise", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 7: Mix of real embeddings and noise
    print("\n7. MIX: REAL EMBEDDINGS + NOISE")
    real_text = "The answer is"
    real_ids = tokenizer(real_text, return_tensors="pt")["input_ids"].to(device)
    real_embeds = embeddings(real_ids)
    noise = torch.randn(1, 2, embeddings.weight.shape[1], device=device, dtype=torch.float16)
    noise = noise * embeddings.weight.std()
    
    mixed = torch.cat([real_embeds, noise], dim=1)
    attention_mask = torch.ones(1, mixed.shape[1], device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=mixed,
            attention_mask=attention_mask,
            max_new_tokens=10
        )
    result = tokenizer.decode(outputs[0][mixed.shape[1]:], skip_special_tokens=True)
    results.append(("Real + noise", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 8: Concept tokens (probability-weighted embeddings)
    print("\n8. CONCEPT TOKENS (our approach)")
    prompt_ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"].to(device)
    prompt_embeds = embeddings(prompt_ids)
    
    with torch.no_grad():
        outputs = model(inputs_embeds=prompt_embeds, return_dict=True)
        logits = outputs.logits[:, -1, :]
    
    # Create concept token
    probs = torch.softmax(logits / 0.8, dim=-1)
    top_k = 50
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
    top_probs = torch.pow(top_probs, 1.5)
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    top_embeddings = embeddings(top_indices)
    weighted_embeddings = top_probs.unsqueeze(-1) * top_embeddings
    concept_token = weighted_embeddings.sum(dim=1)
    
    # Add concept to prompt
    with_concept = torch.cat([prompt_embeds, concept_token.unsqueeze(1)], dim=1)
    attention_mask = torch.ones(1, with_concept.shape[1], device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=with_concept,
            attention_mask=attention_mask,
            max_new_tokens=10
        )
    result = tokenizer.decode(outputs[0][with_concept.shape[1]:], skip_special_tokens=True)
    results.append(("Concept tokens", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    # Test 9: Concept tokens with instruction format
    print("\n9. CONCEPT TOKENS + INSTRUCTION FORMAT")
    messages = [{"role": "user", "content": "Complete: The capital of France is"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    formatted_embeds = embeddings(formatted_ids)
    
    # Add concept token
    with_instruction = torch.cat([formatted_embeds, concept_token.unsqueeze(1)], dim=1)
    attention_mask = torch.ones(1, with_instruction.shape[1], device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=with_instruction,
            attention_mask=attention_mask,
            max_new_tokens=10
        )
    result = tokenizer.decode(outputs[0][with_instruction.shape[1]:], skip_special_tokens=True)
    results.append(("Concept + instruction", "✓" if result else "✗", result[:20]))
    print(f"   Output: {result[:30]}")
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Test Case':<25} {'Output?':<10} {'Sample':<20}")
    print("-" * 55)
    for test, has_output, sample in results:
        print(f"{test:<25} {has_output:<10} {sample:<20}")
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print("""
Pattern observed:
1. Token IDs → Always works
2. Embeddings without instruction format → Usually works
3. Embeddings WITH instruction format → Usually FAILS
4. Special tokens via embeddings → FAILS
5. Noise/concept tokens → Sometimes works without instruction format

The instruction format appears to require exact token sequences.
When we use embeddings, we break the pattern recognition that
triggers the model's instruction-following behavior.
""")

if __name__ == "__main__":
    analyze_output_patterns()