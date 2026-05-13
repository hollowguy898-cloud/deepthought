"""
Fast weight memory for Deep Thought.

Implements Hebbian-style fast weights for rapid adaptation
without corrupting long-term knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from deep_thought.config import MetaLearningConfig


class FastWeightMemory(nn.Module):
    """
    Fast weight memory for rapid local adaptation.
    
    Uses Hebbian learning rule:
    ΔW_fast = λ * W_fast + η * h_t * h_t^T
    
    Creates temporary associations that decay over time,
    allowing rapid adaptation without permanent corruption.
    """
    
    def __init__(self, config: MetaLearningConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.fast_weight_dim = config.fast_weight_dim
        
        # Fast weight matrix
        self.fast_weights = nn.Parameter(
            torch.zeros(latent_dim, self.fast_weight_dim),
            requires_grad=False
        )
        
        # Learning rate
        self.fast_lr = config.fast_weight_lr
        
        # Decay rate
        self.decay = config.fast_weight_decay
        
        # Projection for fast weight computation (fast_weight_dim -> latent_dim)
        self.proj = nn.Linear(self.fast_weight_dim, latent_dim, bias=False)
        
        # Projection for Hebbian update (latent_dim -> fast_weight_dim)
        self.hebbian_proj = nn.Linear(latent_dim, self.fast_weight_dim, bias=False)
    
    def forward(
        self,
        h_t: torch.Tensor,
        update: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply fast weights and optionally update them.
        
        Args:
            h_t: Current hidden state
            update: Whether to update fast weights
            
        Returns:
            h_fast: Fast-weight modified hidden state
            fast_contribution: Contribution from fast weights
        """
        # Compute fast weight contribution
        fast_contribution = torch.matmul(h_t, self.fast_weights)
        
        # Project back to latent dim
        h_fast = self.proj(fast_contribution)
        
        # Update fast weights using Hebbian rule
        if update:
            with torch.no_grad():
                # Hebbian update: ΔW = η * proj(h) ⊗ h
                # Shape: [fast_weight_dim] ⊗ [latent_dim] = [latent_dim, fast_weight_dim]
                projected = self.hebbian_proj(h_t)  # [B, fast_weight_dim]
                hebbian_update = self.fast_lr * torch.matmul(
                    h_t.T,
                    projected
                ) / h_t.size(0)
                
                # Apply update with decay
                self.fast_weights.data = (
                    self.decay * self.fast_weights.data +
                    (1 - self.decay) * hebbian_update
                )
        
        return h_fast, fast_contribution
    
    def reset(self):
        """Reset fast weights to zero."""
        with torch.no_grad():
            self.fast_weights.zero_()
    
    def get_norm(self) -> float:
        """Get current norm of fast weights."""
        return self.fast_weights.norm().item()
    
    def constrain_norm(self, max_norm: float = 10.0):
        """Constrain fast weight norm to prevent explosion."""
        current_norm = self.fast_weights.norm()
        if current_norm > max_norm:
            with torch.no_grad():
                self.fast_weights.data = (
                    self.fast_weights.data * max_norm / current_norm
                )
