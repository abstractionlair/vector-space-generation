#!/usr/bin/env python3
"""
Concept token generation for vector-space language modeling.
Implements probability-weighted mixtures of token embeddings that remain in continuous space.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, List, Dict, Tuple
import numpy as np


class ConceptTokenGenerator:
    """Generator that uses concept tokens (weighted embedding mixtures) for multi-step generation."""
    
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
        
        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Access the embedding layer (tied embeddings in GPT-2)
        self.embeddings = self.model.transformer.wte  # Word token embeddings
        self.position_embeddings = self.model.transformer.wpe  # Position embeddings
        
        # Model info
        self.vocab_size = self.model.config.vocab_size
        self.hidden_size = self.model.config.hidden_size
        self.n_layers = self.model.config.n_layer
        print(f"Model loaded: {model_name}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Hidden dimension: {self.hidden_size}")
        print(f"Number of layers: {self.n_layers}")
    
    def create_concept_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
        sharpening_alpha: float = 1.5,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Create a concept token from logits using probability-weighted embedding mixture.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            sharpening_alpha: Sharpening exponent for probabilities
            return_metrics: Whether to return generation metrics
            
        Returns:
            Concept token embedding and optional metrics
        """
        # Apply temperature
        scaled_logits = logits / temperature
        
        # Get probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Apply top-k filtering
        if top_k > 0:
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
            
            # Sharpen probabilities (p^alpha / sum(p^alpha))
            if sharpening_alpha != 1.0:
                top_probs = torch.pow(top_probs, sharpening_alpha)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            
            # Create weighted mixture of embeddings
            # Get embeddings for top tokens
            top_embeddings = self.embeddings(top_indices)  # [batch_size, top_k, hidden_size]
            
            # Weight by probabilities
            weighted_embeddings = top_probs.unsqueeze(-1) * top_embeddings  # [batch_size, top_k, hidden_size]
            concept_token = weighted_embeddings.sum(dim=1)  # [batch_size, hidden_size]
        else:
            # Use all tokens (not recommended for efficiency)
            all_embeddings = self.embeddings.weight.unsqueeze(0).expand(probs.size(0), -1, -1)
            concept_token = (probs.unsqueeze(-1) * all_embeddings).sum(dim=1)
        
        # Clip norm to stay within reasonable embedding range
        # Calculate typical embedding norm from the embedding matrix
        typical_norm = self.embeddings.weight.norm(dim=-1).mean().item()
        max_norm = typical_norm * 1.5  # Allow some flexibility
        concept_token = F.normalize(concept_token, dim=-1) * min(concept_token.norm(dim=-1, keepdim=True), 
                                                                  torch.tensor(max_norm).to(self.device))
        
        metrics = None
        if return_metrics:
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            # Find nearest token embedding
            distances = torch.cdist(concept_token.unsqueeze(1), self.embeddings.weight.unsqueeze(0), p=2)
            nearest_distance, nearest_token_idx = distances.min(dim=-1)
            nearest_token = self.tokenizer.decode([nearest_token_idx[0].item()])
            
            metrics = {
                "entropy": entropy.mean().item(),
                "top_tokens": [(self.tokenizer.decode([idx]), prob.item()) 
                              for idx, prob in zip(top_indices[0][:5], top_probs[0][:5])],
                "nearest_token": nearest_token,
                "distance_from_manifold": nearest_distance[0].item(),
                "concept_norm": concept_token.norm(dim=-1).mean().item()
            }
        
        return concept_token, metrics
    
    def generate_vector_sequence(
        self,
        prompt_embeddings: torch.Tensor,
        n_vectors: int = 2,
        temperature: float = 0.8,
        top_k: int = 50,
        sharpening_alpha: float = 1.5,
        entropy_threshold: float = 1.0,
        return_all_metrics: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[List[Dict]]]:
        """Generate a sequence of concept tokens without discretization.
        
        Args:
            prompt_embeddings: Initial prompt embeddings [batch_size, seq_len, hidden_size]
            n_vectors: Number of vectors to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            sharpening_alpha: Sharpening exponent
            entropy_threshold: Stop if entropy falls below this for 2 consecutive steps
            return_all_metrics: Whether to return metrics for each step
            
        Returns:
            List of generated concept tokens and optional metrics
        """
        generated_vectors = []
        all_metrics = [] if return_all_metrics else None
        entropy_history = []
        
        # Current sequence (will grow as we add concept tokens)
        current_sequence = prompt_embeddings
        
        for step in range(n_vectors):
            # Get position embeddings for the current sequence length
            seq_len = current_sequence.size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            position_embeds = self.position_embeddings(position_ids)
            
            # Add position embeddings
            inputs_embeds = current_sequence + position_embeds
            
            # Forward pass through the transformer
            outputs = self.model.transformer(
                inputs_embeds=inputs_embeds,
                return_dict=True
            )
            
            # Get logits from the last hidden state
            hidden_states = outputs.last_hidden_state
            logits = self.model.lm_head(hidden_states[:, -1, :])  # Only last position
            
            # Create concept token
            concept_token, metrics = self.create_concept_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                sharpening_alpha=sharpening_alpha,
                return_metrics=return_all_metrics
            )
            
            # Check entropy stopping condition
            if metrics:
                entropy_history.append(metrics["entropy"])
                if len(entropy_history) >= 2 and all(e < entropy_threshold for e in entropy_history[-2:]):
                    print(f"Stopping early at step {step+1} due to low entropy")
                    if return_all_metrics:
                        metrics["early_stop"] = True
                    break
            
            # Add to generated sequence
            generated_vectors.append(concept_token)
            if return_all_metrics:
                all_metrics.append(metrics)
            
            # Append to current sequence for next iteration
            concept_token_expanded = concept_token.unsqueeze(1)  # [batch_size, 1, hidden_size]
            current_sequence = torch.cat([current_sequence, concept_token_expanded], dim=1)
        
        return generated_vectors, all_metrics
    
    def translate_vectors(
        self,
        vector_sequence: List[torch.Tensor],
        max_new_tokens: int = 30,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        prompt_style: str = "completion"
    ) -> str:
        """Translate a sequence of concept tokens back to natural language.
        
        Args:
            vector_sequence: List of concept token embeddings
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for translation generation
            repetition_penalty: Penalty for repetition (1.0 = no penalty)
            prompt_style: "completion" or "instruction" style prompt
            
        Returns:
            Translated text
        """
        # Stack vector sequence
        vectors_tensor = torch.stack(vector_sequence, dim=1)  # [1, n_vectors, hidden_size]
        
        # Create translation prompt based on style
        if prompt_style == "completion":
            # Completion-style: "The meaning of [vectors] in plain English is"
            prefix = "The meaning of "
            suffix = " in plain English is"
            
            # Tokenize and embed prefix and suffix
            prefix_tokens = self.tokenizer(prefix, return_tensors="pt").to(self.device)
            suffix_tokens = self.tokenizer(suffix, return_tensors="pt").to(self.device)
            
            prefix_embeddings = self.embeddings(prefix_tokens["input_ids"])
            suffix_embeddings = self.embeddings(suffix_tokens["input_ids"])
            
            # Concatenate: prefix + vectors + suffix
            full_sequence = torch.cat([prefix_embeddings, vectors_tensor, suffix_embeddings], dim=1)
        else:
            # Instruction-style (original): "Please translate the following: [vectors]"
            translation_prompt = "Please translate the following: "
            prompt_tokens = self.tokenizer(translation_prompt, return_tensors="pt").to(self.device)
            prompt_embeddings = self.embeddings(prompt_tokens["input_ids"])
            
            # Concatenate prompt and vector sequence
            full_sequence = torch.cat([prompt_embeddings, vectors_tensor], dim=1)
        
        # Add position embeddings
        seq_len = full_sequence.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        inputs_embeds = full_sequence + position_embeds
        
        # Use model.generate() for better generation control
        with torch.no_grad():
            # Get attention mask (all ones for our embeddings)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)
            
            # Generate using the model's generate method
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract only the generated tokens (skip the input sequence)
            generated_ids = outputs[0, seq_len:]
        
        # Decode generated tokens
        translation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return translation
    
    def nearest_token_decoding(
        self,
        vector_sequence: List[torch.Tensor]
    ) -> str:
        """Decode concept tokens by finding nearest discrete tokens in embedding space.
        
        Args:
            vector_sequence: List of concept token embeddings
            
        Returns:
            Decoded text from nearest tokens
        """
        nearest_tokens = []
        
        for vector in vector_sequence:
            # Calculate distances to all token embeddings
            distances = torch.cdist(
                vector.unsqueeze(0).unsqueeze(0),  # [1, 1, hidden_size]
                self.embeddings.weight.unsqueeze(0),  # [1, vocab_size, hidden_size]
                p=2
            )
            
            # Find nearest token
            nearest_idx = distances.argmin(dim=-1).item()
            nearest_token = self.tokenizer.decode([nearest_idx])
            nearest_tokens.append(nearest_token)
        
        # Join tokens with spaces (simple approach)
        decoded_text = " ".join(nearest_tokens)
        return decoded_text
    
    def generate(
        self,
        prompt: str,
        n_vectors: int = 2,
        max_translation_length: int = 30,
        temperature: float = 0.8,
        top_k: int = 50,
        sharpening_alpha: float = 1.5,
        translation_method: str = "completion",
        return_metrics: bool = False
    ) -> Dict:
        """Complete generation pipeline: prompt → vectors → translation.
        
        Args:
            prompt: Input text prompt
            n_vectors: Number of concept tokens to generate
            max_translation_length: Maximum length for translation
            temperature: Sampling temperature
            top_k: Top-k filtering
            sharpening_alpha: Sharpening exponent
            translation_method: "completion", "instruction", or "nearest"
            return_metrics: Whether to return generation metrics
            
        Returns:
            Dictionary with prompt, generated vectors info, and translation
        """
        # Tokenize and embed prompt
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_embeddings = self.embeddings(prompt_tokens["input_ids"])
        
        # Generate vector sequence
        vectors, metrics = self.generate_vector_sequence(
            prompt_embeddings,
            n_vectors=n_vectors,
            temperature=temperature,
            top_k=top_k,
            sharpening_alpha=sharpening_alpha,
            return_all_metrics=return_metrics
        )
        
        # Translate to text using specified method
        if translation_method == "nearest":
            translation = self.nearest_token_decoding(vectors)
        else:
            translation = self.translate_vectors(
                vectors, 
                max_new_tokens=max_translation_length,
                prompt_style=translation_method
            )
        
        result = {
            "prompt": prompt,
            "n_vectors_generated": len(vectors),
            "translation_method": translation_method,
            "translation": translation,
            "full_text": prompt + " " + translation
        }
        
        if return_metrics:
            result["metrics"] = {
                "per_vector": metrics,
                "average_entropy": np.mean([m["entropy"] for m in metrics]),
                "average_distance_from_manifold": np.mean([m["distance_from_manifold"] for m in metrics])
            }
        
        return result


def main():
    """Test concept token generation with different translation approaches."""
    
    # Initialize generator
    generator = ConceptTokenGenerator(model_name="gpt2")
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "2 plus 2 equals",
        "Once upon a time"
    ]
    
    print("\n" + "="*60)
    print("CONCEPT TOKEN GENERATION - TRANSLATION COMPARISON")
    print("="*60)
    
    # Test first prompt with all translation methods
    prompt = test_prompts[0]
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    
    # Test different translation methods with N=2 vectors
    translation_methods = ["completion", "instruction", "nearest"]
    
    for method in translation_methods:
        print(f"\nTranslation method: {method}")
        result = generator.generate(
            prompt,
            n_vectors=2,
            temperature=0.8,
            translation_method=method,
            return_metrics=True
        )
        
        print(f"  Generated {result['n_vectors_generated']} concept tokens")
        print(f"  Translation: '{result['translation']}'")
        
        if "metrics" in result and method == "completion":  # Show detailed metrics for one method
            print(f"  Average entropy: {result['metrics']['average_entropy']:.2f}")
            print(f"  Average distance from manifold: {result['metrics']['average_distance_from_manifold']:.2f}")
    
    print("\n" + "="*60)
    print("TESTING ALL PROMPTS WITH COMPLETION METHOD")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        # Test with completion method (new default)
        result = generator.generate(
            prompt,
            n_vectors=2,
            temperature=0.8,
            translation_method="completion",
            return_metrics=True
        )
        
        print(f"Completion translation: {result['translation']}")
        
        # Also show nearest token decoding for reference
        result_nearest = generator.generate(
            prompt,
            n_vectors=2,
            temperature=0.8,
            translation_method="nearest",
            return_metrics=False
        )
        print(f"Nearest tokens: {result_nearest['translation']}")
        
        if "metrics" in result:
            # Show what vectors were actually generated
            for i, vector_metrics in enumerate(result['metrics']['per_vector']):
                top_tokens_str = ", ".join([f"'{t}' ({p:.2f})" for t, p in vector_metrics['top_tokens'][:3]])
                print(f"  Vector {i+1} top weights: {top_tokens_str}")
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT N VALUES")
    print("="*60)
    
    prompt = "In a world where gravity worked backwards"
    for n in [2, 3, 4]:
        print(f"\nN={n} vectors:")
        result = generator.generate(
            prompt, 
            n_vectors=n, 
            temperature=0.8,
            translation_method="completion"
        )
        print(f"  Completion: {result['translation']}")
        
        result_nearest = generator.generate(
            prompt,
            n_vectors=n,
            temperature=0.8,
            translation_method="nearest"
        )
        print(f"  Nearest: {result_nearest['translation']}")
    
    print("\n" + "="*60)
    print("CONCEPT TOKEN TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()