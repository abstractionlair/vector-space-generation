#!/usr/bin/env python3
"""
Debug why instruction format produces empty output with embeddings.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_instruction_embeds():
    """Figure out why instruction format fails with embeddings."""
    
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
    print("INVESTIGATION: Why does instruction format fail?")
    print("="*60)
    
    # Create a simple instruction prompt
    messages = [
        {"role": "user", "content": "Say hello"}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"Prompt text: {repr(prompt_text)}\n")
    
    # Tokenize it
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Token IDs shape: {inputs['input_ids'].shape}")
    print(f"First few tokens: {inputs['input_ids'][0][:10].tolist()}")
    
    # Check for special tokens
    special_tokens = []
    for i, token_id in enumerate(inputs['input_ids'][0].tolist()):
        token = tokenizer.decode([token_id])
        if token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
            special_tokens.append((i, token_id, token))
        # Check for other special tokens
        if token_id >= tokenizer.vocab_size - 1000:  # Often special tokens are at the end
            special_tokens.append((i, token_id, token))
    
    if special_tokens:
        print(f"\nSpecial tokens found:")
        for pos, tid, token in special_tokens[:5]:
            print(f"  Position {pos}: ID={tid}, Token='{token}'")
    
    print("\n" + "="*60)
    print("TEST: Generate with token IDs vs embeddings")
    print("="*60)
    
    # 1. Normal generation
    print("1. With token IDs:")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.7,
            do_sample=True
        )
    generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    print(f"   Generated: '{generated}'")
    
    # 2. With embeddings
    print("\n2. With embeddings:")
    input_embeds = embeddings(inputs['input_ids'])
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, input_embeds.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"   Generated: '{generated}'")
    
    # 3. Try without special tokens
    print("\n3. Remove special tokens and try:")
    
    # Find content without special formatting
    simple_prompt = "Say hello"
    simple_inputs = tokenizer(simple_prompt, return_tensors="pt")
    simple_inputs = {k: v.to(device) for k, v in simple_inputs.items()}
    
    simple_embeds = embeddings(simple_inputs['input_ids'])
    simple_attention = torch.ones_like(simple_inputs['input_ids'])
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=simple_embeds,
            attention_mask=simple_attention,
            max_new_tokens=10,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0, simple_embeds.shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"   Generated: '{generated}'")
    
    print("\n" + "="*60)
    print("TEST: What if we add BOS token embedding?")
    print("="*60)
    
    if tokenizer.bos_token_id:
        bos_embeds = embeddings(torch.tensor([[tokenizer.bos_token_id]], device=device))
        
        # Prepend BOS to our simple prompt
        with_bos = torch.cat([bos_embeds, simple_embeds], dim=1)
        attention_with_bos = torch.ones(1, with_bos.shape[1], device=device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=with_bos,
                attention_mask=attention_with_bos,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0, with_bos.shape[1]:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"With BOS: '{generated}'")
    
    print("\n" + "="*60)
    print("HYPOTHESIS")
    print("="*60)
    print("""
The instruction-tuned model likely relies on special tokens or
positional patterns that get disrupted when using inputs_embeds.
The model may be trained to only respond after seeing specific
token patterns that our embedded approach doesn't preserve.
""")

if __name__ == "__main__":
    debug_instruction_embeds()