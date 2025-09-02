#!/usr/bin/env python3
"""
Test concept token translation using Qwen with completion-style prompts only.
No instruction format - just pure completion.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class QwenCompletionTranslator:
    """Qwen-based concept token generator using only completion-style prompts."""
    
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device != "cpu" else None
        ).eval()
        
        self.embeddings = self.model.get_input_embeddings()
        print(f"Model loaded. Embedding dim: {self.embeddings.weight.shape[1]}\n")
    
    def create_concept_token(self, prompt_text, temperature=0.8, top_k=50):
        """Generate a concept token from a prompt."""
        # Tokenize and embed prompt
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.device)
        prompt_embeds = self.embeddings(prompt_ids)
        
        # Get next token prediction
        with torch.no_grad():
            outputs = self.model(inputs_embeds=prompt_embeds, return_dict=True)
            logits = outputs.logits[:, -1, :]
        
        # Create concept token
        probs = F.softmax(logits / temperature, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        
        # Sharpen probabilities
        top_probs = torch.pow(top_probs, 1.5)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        
        # Weight embeddings
        top_embeddings = self.embeddings(top_indices)
        weighted_embeddings = top_probs.unsqueeze(-1) * top_embeddings
        concept_token = weighted_embeddings.sum(dim=1)
        
        # Get top tokens for display
        top_tokens = [(self.tokenizer.decode([idx]), prob.item()) 
                      for idx, prob in zip(top_indices[0][:5], top_probs[0][:5])]
        
        return concept_token, top_tokens
    
    def translate_with_completion(self, concept_vectors, translation_prompt="The meaning of", suffix=" in plain English is"):
        """Translate concept vectors using completion-style prompt."""
        # Build the translation prompt
        prefix_ids = self.tokenizer(translation_prompt, return_tensors="pt")["input_ids"].to(self.device)
        suffix_ids = self.tokenizer(suffix, return_tensors="pt")["input_ids"].to(self.device)
        
        prefix_embeds = self.embeddings(prefix_ids)
        suffix_embeds = self.embeddings(suffix_ids)
        
        # Stack concept vectors
        if len(concept_vectors) > 0:
            vectors_tensor = torch.stack(concept_vectors, dim=1)
        else:
            return "[No vectors to translate]"
        
        # Concatenate: prefix + vectors + suffix
        full_sequence = torch.cat([prefix_embeds, vectors_tensor, suffix_embeds], dim=1)
        attention_mask = torch.ones(1, full_sequence.shape[1], dtype=torch.long, device=self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=full_sequence,
                attention_mask=attention_mask,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0, full_sequence.shape[1]:]
        translation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return translation
    
    def nearest_tokens(self, concept_vectors):
        """Find nearest discrete tokens for concept vectors."""
        nearest = []
        for vector in concept_vectors:
            distances = torch.cdist(
                vector.unsqueeze(0).unsqueeze(0),
                self.embeddings.weight.unsqueeze(0),
                p=2
            )
            nearest_idx = distances.argmin(dim=-1).item()
            nearest_token = self.tokenizer.decode([nearest_idx])
            nearest.append(nearest_token)
        return " ".join(nearest)


def main():
    """Test concept token translation with completion-style prompts."""
    
    translator = QwenCompletionTranslator()
    
    print("="*60)
    print("CONCEPT TOKEN TRANSLATION - COMPLETION STYLE")
    print("="*60)
    
    test_cases = [
        "The capital of France is",
        "2 plus 2 equals",
        "The opposite of hot is",
        "Water freezes at",
        "Python is a programming"
    ]
    
    for prompt in test_cases:
        print(f"\n{'='*60}")
        print(f"PROMPT: '{prompt}'")
        print("-"*60)
        
        # Generate 2 concept tokens
        concept_vectors = []
        current_prompt = prompt
        
        print("\nGenerating concept tokens:")
        for i in range(2):
            concept, top_tokens = translator.create_concept_token(current_prompt)
            concept_vectors.append(concept)
            
            print(f"\nConcept {i+1} top weights:")
            for token, prob in top_tokens[:3]:
                print(f"  '{token}': {prob:.1%}")
            
            # Update prompt for next token
            current_prompt = current_prompt + " [CONCEPT]"
        
        print("\n" + "-"*40)
        print("TRANSLATIONS:")
        print("-"*40)
        
        # Method 1: Standard translation prompt
        translation1 = translator.translate_with_completion(
            concept_vectors,
            "The meaning of",
            " in plain English is"
        )
        print(f"\n1. 'The meaning of [vectors] in plain English is'")
        print(f"   → {translation1}")
        
        # Method 2: Direct translation style
        translation2 = translator.translate_with_completion(
            concept_vectors,
            "Translating",
            " gives us"
        )
        print(f"\n2. 'Translating [vectors] gives us'")
        print(f"   → {translation2}")
        
        # Method 3: Completion style
        translation3 = translator.translate_with_completion(
            concept_vectors,
            "This means",
            " which is"
        )
        print(f"\n3. 'This means [vectors] which is'")
        print(f"   → {translation3}")
        
        # Method 4: Simple decoding prompt
        translation4 = translator.translate_with_completion(
            concept_vectors,
            "Decoding:",
            " →"
        )
        print(f"\n4. 'Decoding: [vectors] →'")
        print(f"   → {translation4}")
        
        # Show nearest tokens for reference
        nearest = translator.nearest_tokens(concept_vectors)
        print(f"\n5. Nearest tokens (reference): {nearest}")
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT VECTOR COUNTS")
    print("="*60)
    
    prompt = "Once upon a time"
    
    for n_vectors in [1, 2, 3, 4]:
        print(f"\nWith {n_vectors} concept vector(s):")
        
        # Generate N concept vectors
        concept_vectors = []
        current_prompt = prompt
        
        for i in range(n_vectors):
            concept, _ = translator.create_concept_token(current_prompt)
            concept_vectors.append(concept)
            current_prompt = current_prompt + " [CONCEPT]"
        
        # Translate
        translation = translator.translate_with_completion(
            concept_vectors,
            "The meaning of",
            " in plain English is"
        )
        print(f"  Translation: {translation}")
        
        # Nearest tokens
        nearest = translator.nearest_tokens(concept_vectors)
        print(f"  Nearest: {nearest}")
    
    print("\n" + "="*60)
    print("COMPARISON WITH GPT-2")
    print("="*60)
    print("""
Key differences from GPT-2 results:
1. Qwen has actual translation capability
2. Using completion-style prompts avoids instruction format issues
3. We can now see if the model interprets concept tokens differently
""")


if __name__ == "__main__":
    main()