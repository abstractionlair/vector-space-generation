#!/usr/bin/env python3
"""
Test a hybrid approach: keep special tokens as tokens, only embed the content.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_hybrid_approach():
    """Test keeping special tokens as IDs while using embeddings for content."""
    
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
    print("KEY INSIGHT: What if we pass token IDs with embedded content?")
    print("="*60)
    
    # First, let's understand the special token IDs
    im_start_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    
    print(f"<|im_start|> token ID: {im_start_id}")
    print(f"<|im_end|> token ID: {im_end_id}")
    
    # Generate concept tokens for "Paris"
    prompt = "The capital of France is"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_embeds = embeddings(prompt_ids)
    
    with torch.no_grad():
        outputs = model(inputs_embeds=prompt_embeds, return_dict=True)
        logits = outputs.logits[:, -1, :]
    
    # Create concept token
    probs = F.softmax(logits / 0.8, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=50, dim=-1)
    top_probs = torch.pow(top_probs, 1.5)
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    top_embeddings = embeddings(top_indices)
    weighted_embeddings = top_probs.unsqueeze(-1) * top_embeddings
    concept_token = weighted_embeddings.sum(dim=1)
    
    print(f"\nConcept token represents (top 3):")
    for i in range(3):
        print(f"  {tokenizer.decode([top_indices[0][i]])}: {top_probs[0][i]:.2%}")
    
    print("\n" + "="*60)
    print("HYBRID TEST: Mix token IDs and embeddings")
    print("="*60)
    
    # Build instruction format using actual token IDs where needed
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Complete this: The capital of France is [CONCEPT]"}
    ]
    
    # Get the formatted text
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"Template format:\n{formatted[:200]}...\n")
    
    # Now we need to be clever: tokenize everything EXCEPT where we want concept tokens
    parts = formatted.split("[CONCEPT]")
    
    if len(parts) == 2:
        # Tokenize the parts
        part1_ids = tokenizer(parts[0], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        part2_ids = tokenizer(parts[1], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        
        print(f"Part 1 has {part1_ids.shape[1]} tokens")
        print(f"Part 2 has {part2_ids.shape[1]} tokens")
        
        # Option 1: Forward pass with mixed inputs
        print("\nOption 1: Can model.forward() handle mixed token IDs and embeddings?")
        
        # Get embeddings for the token parts
        part1_embeds = embeddings(part1_ids)
        part2_embeds = embeddings(part2_ids)
        
        # Concatenate embeddings
        full_embeds = torch.cat([part1_embeds, concept_token.unsqueeze(1), part2_embeds], dim=1)
        attention_mask = torch.ones(1, full_embeds.shape[1], device=device)
        
        try:
            with torch.no_grad():
                # Try forward pass
                outputs = model(
                    inputs_embeds=full_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                print("Forward pass successful!")
                
                # Try generation
                gen_outputs = model.generate(
                    inputs_embeds=full_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
                
                generated_ids = gen_outputs[0, full_embeds.shape[1]:]
                generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(f"Generated: '{generated}'")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("ALTERNATIVE: What if we just continue from the concept?")
    print("="*60)
    
    # Simpler approach: generate normally up to "is", then continue from concept token
    prefix = "The capital of France is"
    
    # Format with instruction template
    messages = [
        {"role": "user", "content": prefix}
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the formatted prompt
    formatted_ids = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    formatted_embeds = embeddings(formatted_ids)
    
    # Add our concept token at the end
    with_concept = torch.cat([formatted_embeds, concept_token.unsqueeze(1)], dim=1)
    attention_mask = torch.ones(1, with_concept.shape[1], device=device)
    
    print(f"Prompt: {formatted}")
    print(f"Plus concept token representing 'Paris'")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=with_concept,
            attention_mask=attention_mask,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, with_concept.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated continuation: '{generated}'")
    
    print("\n" + "="*60)
    print("FINAL TEST: What if we use a different model?")
    print("="*60)
    print("The issue might be specific to Qwen's instruction tuning.")
    print("Consider trying: microsoft/Phi-2 or another model without")
    print("such strict instruction formatting requirements.")

if __name__ == "__main__":
    test_hybrid_approach()