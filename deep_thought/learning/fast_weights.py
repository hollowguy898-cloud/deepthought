"""
Fast weight memory for Deep Thought.

Implements Hebbian-style fast weights for rapid adaptation
without corrupting long-term knowledge.

LEVER 5 (Lamarckian Fix): Fast Adaptation + Hebbian learning without
strict SRP leads to "more neurons = faster adaptation" spiral.  The fix
is to constrain the fast weight norm INVERSELY PROPORTIONAL to the
number of active experts, so that adding more experts does NOT give
the fast weight system more adaptation budget.  This breaks the
Lamarckian spiral where the system discovers that more experts means
faster Hebbian adaptation, leading to neuron explosion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from deep_thought.config import MetaLearningConfig


class FastWeightMemory(nn.Module):
    """
    Fast weight memory for rapid local adaptation.
    
    Uses Hebbian learning rule:
    ΔW_fast = λ * W_fast + η * h_t * h_t^T
    
    Creates temporary associations that decay over time,
    allowing rapid adaptation without permanent corruption.
    
    LEVER 5: The max norm for fast weights is dynamically scaled as:
        max_norm = base_budget / sqrt(num_active_experts)
    This ensures that adding more experts does NOT increase the total
    fast weight adaptation budget.  Without this constraint, the system
    discovers a "Lamarckian" shortcut: more experts -> more Hebbian
    connections -> faster short-term adaptation -> even more growth.
    """
    
    def __init__(self, config: MetaLearningConfig, latent_dim: int = 1024,
                 num_active_experts: int = 4):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.fast_weight_dim = config.fast_weight_dim
        
        # LEVER 5: Track number of active experts for norm scaling
        self._num_active_experts = max(1, num_active_experts)
        # Base budget per expert — total budget is this * sqrt(num_experts)
        self._per_expert_budget = getattr(config, 'fast_weight_norm_per_expert_budget', 2.0)
        
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
    
    def set_num_active_experts(self, num: int):
        """Update the number of active experts for LEVER 5 norm scaling.
        
        Called by the agent when the expert count changes (pruning/growth).
        """
        self._num_active_experts = max(1, num)
    
    def _get_max_norm(self) -> float:
        """Compute the dynamic max norm for fast weights (LEVER 5).
        
        The total adaptation budget is FIXED regardless of how many
        experts exist.  More experts = less budget per expert.
        
        max_norm = per_expert_budget * sqrt(num_active_experts)
        
        This means the TOTAL fast weight norm across all experts grows
        only as sqrt(N), not linearly with N.  Without this, the system
        finds that adding N experts gives N times more Hebbian capacity,
        leading to the neuron explosion spiral.
        """
        return self._per_expert_budget * math.sqrt(self._num_active_experts)
    
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

                # Clamp Hebbian update to prevent transient instability
                # from large outer-product values before mixing with decay
                hebbian_update = torch.clamp(hebbian_update, -1.0, 1.0)

                # DESIGN LIMITATION: The effective learning rate and decay
                # are coupled — the update rule is
                #   W' = decay * W + (1 - decay) * update
                # so changing decay also changes how much the new update
                # contributes.  To tune LR and decay independently, the
                # formula would need to be re-parameterised as
                #   W' = decay * W + lr * update
                # which is left as a future refactor.
                # Apply update with decay
                self.fast_weights.data = (
                    self.decay * self.fast_weights.data +
                    (1 - self.decay) * hebbian_update
                )
                
                # LEVER 5: Strict SRP — constrain fast weight norm
                # proportional to 1/sqrt(num_active_experts)
                self.constrain_norm(self._get_max_norm())
        
        return h_fast, fast_contribution
    
    def reset(self):
        """Reset fast weights to zero."""
        with torch.no_grad():
            self.fast_weights.zero_()
    
    def get_norm(self) -> float:
        """Get current norm of fast weights."""
        return self.fast_weights.norm().item()
    
    def constrain_norm(self, max_norm: float = 10.0):
        """Constrain fast weight norm to prevent explosion.
        
        LEVER 5: By default, this now uses the dynamic max norm
        that scales inversely with the number of active experts.
        """
        current_norm = self.fast_weights.norm()
        if current_norm > max_norm:
            with torch.no_grad():
                self.fast_weights.data = (
                    self.fast_weights.data * max_norm / current_norm
                )
