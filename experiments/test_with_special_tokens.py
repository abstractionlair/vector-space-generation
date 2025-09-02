#!/usr/bin/env python3
"""
Test if adding special tokens properly makes instruction format work with concept tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

def create_concept_token(model, embeddings, logits, temperature=0.8, top_k=50):
    """Create a concept token from logits."""
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Get probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Apply top-k filtering
    top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
    
    # Sharpen probabilities
    top_probs = torch.pow(top_probs, 1.5)
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    
    # Get embeddings for top tokens
    top_embeddings = embeddings(top_indices)
    
    # Weight by probabilities
    weighted_embeddings = top_probs.unsqueeze(-1) * top_embeddings
    concept_token = weighted_embeddings.sum(dim=1)
    
    return concept_token, top_indices[0][:5], top_probs[0][:5]

def test_with_special_tokens():
    """Test concept tokens with proper special token handling."""
    
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
    print("TEST 1: Generate concept tokens from a prompt")
    print("="*60)
    
    # Start with a simple prompt
    prompt = "The capital of France is"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_embeds = embeddings(prompt_ids)
    
    # Generate 2 concept tokens
    concept_tokens = []
    current_embeds = prompt_embeds
    
    for i in range(2):
        with torch.no_grad():
            outputs = model(inputs_embeds=current_embeds, return_dict=True)
            logits = outputs.logits[:, -1, :]
        
        concept_token, top_ids, top_probs = create_concept_token(
            model, embeddings, logits
        )
        concept_tokens.append(concept_token)
        
        # Show what we generated
        print(f"Concept token {i+1}:")
        for tid, prob in zip(top_ids, top_probs):
            token_str = tokenizer.decode([tid])
            print(f"  {token_str}: {prob:.2%}")
        
        # Append for next iteration
        current_embeds = torch.cat([current_embeds, concept_token.unsqueeze(1)], dim=1)
    
    print("\n" + "="*60)
    print("TEST 2: Try different ways to add instruction format")
    print("="*60)
    
    # Method 1: Build instruction format with concept tokens in user message
    print("\nMethod 1: Concept tokens as part of user message")
    
    # Build the instruction prompt structure
    system_msg = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    user_start = "<|im_start|>user\nPlease complete: The capital of France is "
    user_end = "<|im_end|>\n"
    assistant_start = "<|im_start|>assistant\n"
    
    # Tokenize each part
    system_ids = tokenizer(system_msg, return_tensors="pt")["input_ids"].to(device)
    user_start_ids = tokenizer(user_start, return_tensors="pt")["input_ids"].to(device)
    user_end_ids = tokenizer(user_end, return_tensors="pt")["input_ids"].to(device)
    assistant_ids = tokenizer(assistant_start, return_tensors="pt")["input_ids"].to(device)
    
    # Get embeddings
    system_embeds = embeddings(system_ids)
    user_start_embeds = embeddings(user_start_ids)
    user_end_embeds = embeddings(user_end_ids)
    assistant_embeds = embeddings(assistant_ids)
    
    # Stack concept tokens
    concept_stack = torch.stack(concept_tokens, dim=1)
    
    # Build full sequence: system + user_start + [concepts] + user_end + assistant
    full_sequence = torch.cat([
        system_embeds,
        user_start_embeds,
        concept_stack,  # Our concept tokens go here
        user_end_embeds,
        assistant_embeds
    ], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones(1, full_sequence.shape[1], device=device)
    
    print(f"Sequence length: {full_sequence.shape[1]}")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=full_sequence,
            attention_mask=attention_mask,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, full_sequence.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: '{generated}'")
    
    # Method 2: Put concept tokens in assistant response
    print("\nMethod 2: Concept tokens as start of assistant response")
    
    user_msg = "<|im_start|>user\nThe capital of France is<|im_end|>\n"
    user_ids = tokenizer(user_msg, return_tensors="pt")["input_ids"].to(device)
    user_embeds = embeddings(user_ids)
    
    # Build: system + user + assistant_start + [concepts]
    full_sequence2 = torch.cat([
        system_embeds,
        user_embeds,
        assistant_embeds,
        concept_stack  # Concepts as part of assistant's response
    ], dim=1)
    
    attention_mask2 = torch.ones(1, full_sequence2.shape[1], device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=full_sequence2,
            attention_mask=attention_mask2,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, full_sequence2.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: '{generated}'")
    
    # Method 3: Simplest - just add the assistant tag after concepts
    print("\nMethod 3: Simple format - prompt + concepts + assistant tag")
    
    # Just use: prompt + concepts + "<|im_start|>assistant\n"
    simple_sequence = torch.cat([
        prompt_embeds,
        concept_stack,
        assistant_embeds
    ], dim=1)
    
    attention_mask3 = torch.ones(1, simple_sequence.shape[1], device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=simple_sequence,
            attention_mask=attention_mask3,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, simple_sequence.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated: '{generated}'")
    
    print("\n" + "="*60)
    print("TEST 3: Check if special tokens alone trigger generation")
    print("="*60)
    
    # Just the assistant start token
    just_assistant = assistant_embeds
    attention_mask4 = torch.ones(1, just_assistant.shape[1], device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=just_assistant,
            attention_mask=attention_mask4,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, just_assistant.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Just assistant tag generates: '{generated}'")
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)

if __name__ == "__main__":
    test_with_special_tokens()