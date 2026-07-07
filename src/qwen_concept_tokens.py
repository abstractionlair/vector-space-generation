#!/usr/bin/env python3
"""
Concept token generation using Qwen2.5-1.5B-Instruct.
Testing whether modern instruction-tuned models better understand concept tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Tuple
import numpy as np
import gc
import psutil


class QwenConceptGenerator:
    """Concept token generator using Qwen2.5-1.5B-Instruct."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize with Qwen2.5-1.5B-Instruct model.
        
        Args:
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
        
        # Check available memory
        mem = psutil.virtual_memory()
        print(f"Available RAM: {mem.available / 1024**3:.1f} GB")
        
        # Model name
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model in float16 for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device != "cpu" else None
        ).eval()
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        # Get embeddings layer
        self.embeddings = self.model.get_input_embeddings()
        
        # Model info
        self.vocab_size = self.embeddings.weight.shape[0]
        self.hidden_size = self.embeddings.weight.shape[1]
        print(f"Model loaded successfully")
        print(f"Embedding matrix shape: {self.embeddings.weight.shape}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Hidden dimension: {self.hidden_size}")
        
        # Check memory usage
        param_count = sum(p.numel() for p in self.model.parameters())
        memory_bytes = param_count * 2  # float16
        print(f"Model size in memory: {memory_bytes / 1024**3:.1f} GB")
    
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
            
            # Sharpen probabilities
            if sharpening_alpha != 1.0:
                top_probs = torch.pow(top_probs, sharpening_alpha)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            
            # Get embeddings for top tokens
            top_embeddings = self.embeddings(top_indices)
            
            # Weight by probabilities
            weighted_embeddings = top_probs.unsqueeze(-1) * top_embeddings
            concept_token = weighted_embeddings.sum(dim=1)
        else:
            all_embeddings = self.embeddings.weight.unsqueeze(0).expand(probs.size(0), -1, -1)
            concept_token = (probs.unsqueeze(-1) * all_embeddings).sum(dim=1)
        
        # Clip norm to stay within reasonable range
        typical_norm = self.embeddings.weight.norm(dim=-1).mean().item()
        max_norm = typical_norm * 1.5
        current_norm = concept_token.norm(dim=-1, keepdim=True)
        if current_norm > max_norm:
            concept_token = F.normalize(concept_token, dim=-1) * max_norm
        
        metrics = None
        if return_metrics:
            # Calculate entropy (handle numerical issues)
            # Filter out very small probabilities to avoid log(0)
            probs_filtered = probs.clone()
            probs_filtered[probs_filtered < 1e-10] = 1e-10
            entropy = -torch.sum(probs_filtered * torch.log(probs_filtered), dim=-1)
            
            # Find nearest token
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
        """Generate a sequence of concept tokens without discretization."""
        generated_vectors = []
        all_metrics = [] if return_all_metrics else None
        entropy_history = []
        
        # Current sequence
        current_embeddings = prompt_embeddings
        
        for step in range(n_vectors):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=current_embeddings,
                    return_dict=True
                )
            
            # Get logits from last position
            logits = outputs.logits[:, -1, :]
            
            # Create concept token
            concept_token, metrics = self.create_concept_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                sharpening_alpha=sharpening_alpha,
                return_metrics=return_all_metrics
            )
            
            # Check entropy stopping
            if metrics:
                entropy_history.append(metrics["entropy"])
                if len(entropy_history) >= 2 and all(e < entropy_threshold for e in entropy_history[-2:]):
                    print(f"Stopping early at step {step+1} due to low entropy")
                    if return_all_metrics:
                        metrics["early_stop"] = True
                    break
            
            # Add to sequence
            generated_vectors.append(concept_token)
            if return_all_metrics:
                all_metrics.append(metrics)
            
            # Append for next iteration
            concept_token_expanded = concept_token.unsqueeze(1)
            current_embeddings = torch.cat([current_embeddings, concept_token_expanded], dim=1)
        
        return generated_vectors, all_metrics
    
    def translate_vectors_completion(
        self,
        vector_sequence: List[torch.Tensor],
        max_new_tokens: int = 30,
        temperature: float = 0.8
    ) -> str:
        """Translate vectors using completion-style prompt."""
        # Stack vectors
        vectors_tensor = torch.stack(vector_sequence, dim=1)
        
        # Completion style prompt
        prefix = "The meaning of "
        suffix = " in plain English is:"
        
        # Tokenize and embed
        prefix_ids = self.tokenizer(prefix, return_tensors="pt")["input_ids"].to(self.device)
        suffix_ids = self.tokenizer(suffix, return_tensors="pt")["input_ids"].to(self.device)
        
        prefix_embeddings = self.embeddings(prefix_ids)
        suffix_embeddings = self.embeddings(suffix_ids)
        
        # Concatenate: prefix + vectors + suffix
        full_sequence = torch.cat([prefix_embeddings, vectors_tensor, suffix_embeddings], dim=1)
        
        # Create attention mask (all 1s for all positions we want to attend to)
        seq_len = full_sequence.size(1)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=full_sequence,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Extract generated tokens (skip input)
        generated_ids = outputs[0, seq_len:]
        translation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return translation
    
    def translate_vectors_instruction(
        self,
        vector_sequence: List[torch.Tensor],
        max_new_tokens: int = 30,
        temperature: float = 0.8
    ) -> str:
        """Translate vectors using Qwen's instruction format."""
        # Stack vectors
        vectors_tensor = torch.stack(vector_sequence, dim=1)
        
        # Use Qwen's chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can decode embedded messages."},
            {"role": "user", "content": "Decode the following embedded representation into natural text: [EMBEDDED_VECTORS]"}
        ]
        
        # Apply chat template to get the text format
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Find where to insert vectors
        # We'll replace [EMBEDDED_VECTORS] with actual vectors
        parts = prompt_text.split("[EMBEDDED_VECTORS]")
        
        if len(parts) == 2:
            # Tokenize parts
            part1_ids = self.tokenizer(parts[0], return_tensors="pt")["input_ids"].to(self.device)
            part2_ids = self.tokenizer(parts[1], return_tensors="pt")["input_ids"].to(self.device)
            
            part1_embeddings = self.embeddings(part1_ids)
            part2_embeddings = self.embeddings(part2_ids)
            
            # Concatenate with vectors in middle
            full_sequence = torch.cat([part1_embeddings, vectors_tensor, part2_embeddings], dim=1)
        else:
            # Fallback if template doesn't have placeholder
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.device)
            prompt_embeddings = self.embeddings(prompt_ids)
            full_sequence = torch.cat([prompt_embeddings, vectors_tensor], dim=1)
        
        # Create attention mask
        seq_len = full_sequence.size(1)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=full_sequence,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Extract generated tokens
        generated_ids = outputs[0, seq_len:]
        translation = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return translation
    
    def nearest_token_decoding(
        self,
        vector_sequence: List[torch.Tensor]
    ) -> str:
        """Decode concept tokens by finding nearest discrete tokens."""
        nearest_tokens = []
        
        for vector in vector_sequence:
            # Find nearest token
            distances = torch.cdist(
                vector.unsqueeze(0).unsqueeze(0),
                self.embeddings.weight.unsqueeze(0),
                p=2
            )
            nearest_idx = distances.argmin(dim=-1).item()
            nearest_token = self.tokenizer.decode([nearest_idx])
            nearest_tokens.append(nearest_token)
        
        return " ".join(nearest_tokens)
    
    def generate(
        self,
        prompt: str,
        n_vectors: int = 2,
        max_translation_length: int = 30,
        temperature: float = 0.8,
        top_k: int = 50,
        sharpening_alpha: float = 1.5,
        translation_method: str = "instruction",
        return_metrics: bool = False
    ) -> Dict:
        """Complete generation pipeline."""
        # Tokenize and embed prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        prompt_embeddings = self.embeddings(prompt_ids)
        
        # Generate vector sequence
        vectors, metrics = self.generate_vector_sequence(
            prompt_embeddings,
            n_vectors=n_vectors,
            temperature=temperature,
            top_k=top_k,
            sharpening_alpha=sharpening_alpha,
            return_all_metrics=return_metrics
        )
        
        # Translate using specified method
        if translation_method == "nearest":
            translation = self.nearest_token_decoding(vectors)
        elif translation_method == "completion":
            translation = self.translate_vectors_completion(vectors, max_new_tokens=max_translation_length)
        elif translation_method == "instruction":
            translation = self.translate_vectors_instruction(vectors, max_new_tokens=max_translation_length)
        else:
            translation = "[Unknown translation method]"
        
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
                "average_entropy": np.mean([m["entropy"] for m in metrics]) if metrics else 0,
                "average_distance_from_manifold": np.mean([m["distance_from_manifold"] for m in metrics]) if metrics else 0
            }
        
        return result


def main():
    """Test Qwen concept token generation."""
    
    print("\n" + "="*60)
    print("QWEN2.5-1.5B CONCEPT TOKEN GENERATION TEST")
    print("="*60)
    
    # Initialize generator
    generator = QwenConceptGenerator()
    
    # Clean up memory after loading
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    gc.collect()
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "2 plus 2 equals",
        "Once upon a time"
    ]
    
    print("\n" + "="*60)
    print("COMPARING TRANSLATION METHODS")
    print("="*60)
    
    # Test first prompt with all methods
    prompt = test_prompts[0]
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    
    for method in ["instruction", "completion", "nearest"]:
        print(f"\nMethod: {method}")
        result = generator.generate(
            prompt,
            n_vectors=2,
            temperature=0.8,
            translation_method=method,
            return_metrics=True
        )
        print(f"  Translation: '{result['translation']}'")
        if method == "instruction" and "metrics" in result:
            print(f"  Avg entropy: {result['metrics']['average_entropy']:.2f}")
            print(f"  Avg distance: {result['metrics']['average_distance_from_manifold']:.2f}")
    
    print("\n" + "="*60)
    print("TESTING ALL PROMPTS WITH INSTRUCTION METHOD")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        # Instruction method (best for Qwen)
        result = generator.generate(
            prompt,
            n_vectors=2,
            temperature=0.8,
            translation_method="instruction",
            return_metrics=True
        )
        print(f"Instruction translation: {result['translation']}")
        
        # Nearest tokens for reference
        result_nearest = generator.generate(
            prompt,
            n_vectors=2,
            temperature=0.8,
            translation_method="nearest",
            return_metrics=False
        )
        print(f"Nearest tokens: {result_nearest['translation']}")
        
        if "metrics" in result:
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
            translation_method="instruction"
        )
        print(f"  Instruction: {result['translation']}")
        
        result_nearest = generator.generate(
            prompt,
            n_vectors=n,
            temperature=0.8,
            translation_method="nearest"
        )
        print(f"  Nearest: {result_nearest['translation']}")
    
    # Clean up
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    gc.collect()
    
    print("\n" + "="*60)
    print("QWEN TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()