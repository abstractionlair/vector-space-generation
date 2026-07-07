#!/usr/bin/env python3
"""
Baseline GPT-2 generation script for comparison.
Standard token-by-token generation using the transformers library.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, List, Dict
import numpy as np


class BaselineGenerator:
    """Standard GPT-2 token-by-token generator."""
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """Initialize the generator with GPT-2 model.
        
        Args:
            model_name: Name of the GPT-2 model variant
            device: Device to run on ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Using device: {device}")
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Set pad token to eos token (GPT-2 doesn't have a pad token by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model info
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        print(f"Model loaded: {model_name}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Hidden dimension: {self.hidden_size}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_sequences: int = 1,
        return_metrics: bool = False
    ) -> Dict:
        """Generate text using standard token-by-token approach.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            num_sequences: Number of sequences to generate
            return_metrics: Whether to return generation metrics
            
        Returns:
            Dictionary with generated text and optional metrics
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Track metrics if requested
        metrics = {
            "entropy_per_step": [],
            "perplexity_per_step": [],
            "token_probabilities": []
        } if return_metrics else None
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=return_metrics,
                output_scores=return_metrics
            )
        
        # Process outputs
        if return_metrics and hasattr(outputs, 'scores'):
            # Calculate metrics for each generation step
            for scores in outputs.scores:
                probs = torch.softmax(scores / temperature, dim=-1)
                
                # Entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                metrics["entropy_per_step"].append(entropy.mean().item())
                
                # Perplexity
                perplexity = torch.exp(entropy)
                metrics["perplexity_per_step"].append(perplexity.mean().item())
                
                # Top token probabilities
                top_probs, top_indices = torch.topk(probs[0], k=5)
                top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
                metrics["token_probabilities"].append(list(zip(top_tokens, top_probs.tolist())))
        
        # Decode generated sequences
        generated_ids = outputs if not return_metrics else outputs.sequences
        generated_texts = []
        for seq in generated_ids:
            # Remove the prompt from the generated text
            generated_only = seq[len(input_ids[0]):]
            text = self.tokenizer.decode(generated_only, skip_special_tokens=True)
            generated_texts.append(text)
        
        result = {
            "prompt": prompt,
            "generated": generated_texts[0] if num_sequences == 1 else generated_texts,
            "full_text": prompt + generated_texts[0] if num_sequences == 1 else [prompt + g for g in generated_texts]
        }
        
        if return_metrics:
            result["metrics"] = metrics
        
        return result
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of a given text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()


def main():
    """Test the baseline generator with example prompts."""
    
    # Initialize generator
    generator = BaselineGenerator(model_name="gpt2")
    
    # Test prompts from the specification
    test_prompts = [
        "The capital of France is",
        "2 plus 2 equals",
        "Once upon a time",
        "If it's raining, I need an umbrella. It's raining, so",
        "In a world where gravity worked backwards"
    ]
    
    print("\n" + "="*60)
    print("BASELINE GPT-2 GENERATION TEST")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        # Generate with metrics
        result = generator.generate(
            prompt,
            max_length=50,
            temperature=0.8,
            return_metrics=True
        )
        
        print(f"Generated: {result['generated']}")
        
        if "metrics" in result:
            avg_entropy = np.mean(result["metrics"]["entropy_per_step"])
            print(f"Average entropy: {avg_entropy:.2f}")
            
            # Show top tokens for first few steps
            print("Top tokens (first 3 steps):")
            for i, step_probs in enumerate(result["metrics"]["token_probabilities"][:3]):
                tokens_str = ", ".join([f"'{t}' ({p:.2f})" for t, p in step_probs[:3]])
                print(f"  Step {i+1}: {tokens_str}")
    
    print("\n" + "="*60)
    print("BASELINE TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()