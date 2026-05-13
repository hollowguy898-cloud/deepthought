"""
Sparse Router module for Deep Thought.

Implements top-k expert selection with load balancing, entropy regularization,
and noise-augmented gating for stable sparse activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

from deep_thought.config import RouterConfig


class NoisyTopKRouter(nn.Module):
    """
    Router with noisy top-k gating for load balancing.
    
    Uses additive noise during training to encourage expert diversity
    and prevent routing collapse.
    """
    
    def __init__(self, config: RouterConfig, latent_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.active_experts = config.active_experts
        self.hidden_dim = config.hidden_dim
        self.noise_epsilon = config.noise_epsilon
        
        # Router network - input is concatenated h_t, x_t, m_t (each latent_dim)
        self._latent_dim = latent_dim
        if latent_dim is not None:
            self.router = nn.Sequential(
                nn.Linear(latent_dim * 3, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.num_experts),
            )
        else:
            # Will be initialized lazily
            self.router = None
        
        # Load balancing loss tracking
        self.register_buffer("expert_usage", torch.zeros(self.num_experts))
        self.usage_ema = 0.99
    
    def _ensure_router(self, h_t: torch.Tensor):
        """Lazily initialize router if needed."""
        if self.router is None:
            latent_dim = h_t.size(-1)
            self._latent_dim = latent_dim
            self.router = nn.Sequential(
                nn.Linear(latent_dim * 3, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.num_experts),
            )
            # Move to same device as input
            self.router = self.router.to(h_t.device)
    
    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route to top-k experts.
        
        Args:
            h_t: Hidden state
            x_t: Encoded observation
            m_t: Memory read
            training: Whether in training mode
            
        Returns:
            gates: Gate values for selected experts
            selected_indices: Indices of selected experts
            info: Routing information
        """
        # Concatenate inputs
        combined = torch.cat([h_t, x_t, m_t], dim=-1)

        # Ensure router is initialized
        self._ensure_router(h_t)

        # Compute router logits
        logits = self.router(combined)
        
        # Add noise during training
        if training:
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits = logits + noise
        
        # Softmax for probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            probs,
            self.active_experts,
            dim=-1
        )
        
        # Normalize gates
        gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Update expert usage statistics
        if training:
            with torch.no_grad():
                batch_size = logits.size(0)
                for idx in top_k_indices.view(-1).unique():
                    self.expert_usage[idx] = self.expert_usage[idx] * self.usage_ema + \
                                           (1 - self.usage_ema) / batch_size
        
        # Compute routing entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        info = {
            "logits": logits,
            "probs": probs,
            "entropy": entropy,
            "expert_usage": self.expert_usage.clone(),
        }
        
        return gates, top_k_indices, info
    
    def load_balance_loss(self) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Encourages uniform expert usage to prevent collapse.
        """
        # Ideal uniform distribution
        target = torch.ones_like(self.expert_usage) / self.num_experts
        
        # KL divergence
        loss = F.kl_div(
            self.expert_usage.log(),
            target,
            reduction="batchmean"
        )
        
        return self.config.load_balance_loss_coef * loss
    
    def entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularization loss.
        
        Keeps routing entropy in healthy range.
        """
        if entropy < self.config.min_entropy:
            return self.config.entropy_coef * (self.config.min_entropy - entropy)
        elif entropy > self.config.max_entropy:
            return self.config.entropy_coef * (entropy - self.config.max_entropy)
        return torch.tensor(0.0, device=entropy.device)


class AdaptiveRouter(nn.Module):
    """
    Adaptive router that adjusts based on context and prediction error.
    
    Modifies routing distribution based on:
    - Context embedding
    - Prediction error signals
    - Meta-router controller
    """
    
    def __init__(self, config: RouterConfig, context_dim: int = 256, latent_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.context_dim = context_dim
        
        # Base router
        self.base_router = NoisyTopKRouter(config, latent_dim=latent_dim)
        
        # Context encoder - will be lazily initialized if latent_dim not known
        self._latent_dim = latent_dim
        if latent_dim is not None:
            self.context_encoder = nn.Linear(latent_dim * 3, self.context_dim)
        else:
            self.context_encoder = None
        
        # Adapter - will be lazily initialized
        self.adapter = None
        self._adapter_input_dim = None
    
    def _ensure_modules(self, h_t: torch.Tensor, x_t: torch.Tensor, m_t: torch.Tensor):
        """Lazily initialize context encoder and adapter."""
        combined = torch.cat([h_t, x_t, m_t], dim=-1)
        
        if self.context_encoder is None:
            input_dim = combined.size(-1)
            self.context_encoder = nn.Linear(input_dim, self.context_dim).to(combined.device)
        
        if self.adapter is None:
            # adapter_input = context + prediction_error (1 dim)
            self._adapter_input_dim = self.context_dim + 1
            self.adapter = nn.Sequential(
                nn.Linear(self._adapter_input_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.hidden_dim, self.config.num_experts),
            ).to(combined.device)
    
    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prediction_error: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route with adaptive modification.
        
        Args:
            h_t: Hidden state
            x_t: Encoded observation
            m_t: Memory read
            context: Context embedding
            prediction_error: Prediction error signal
            training: Whether in training mode
            
        Returns:
            gates: Gate values for selected experts
            selected_indices: Indices of selected experts
            info: Routing information
        """
        # Base routing
        gates, indices, base_info = self.base_router(
            h_t, x_t, m_t, training
        )
        
        # Ensure adapter modules are initialized
        self._ensure_modules(h_t, x_t, m_t)
        
        # Encode context if not provided
        if context is None:
            combined = torch.cat([h_t, x_t, m_t], dim=-1)
            context = self.context_encoder(combined)
        
        # Adaptation based on context and error
        if prediction_error is not None:
            # Normalize error
            if prediction_error.dim() == 0:
                # Scalar tensor
                error_norm = prediction_error.unsqueeze(0).unsqueeze(0)
                if error_norm.size(0) != context.size(0):
                    error_norm = error_norm.expand(context.size(0), 1)
            else:
                error_norm = (prediction_error - prediction_error.mean()) / \
                            (prediction_error.std() + 1e-8)
                if error_norm.dim() == 1:
                    error_norm = error_norm.unsqueeze(-1)
                elif error_norm.dim() == 0:
                    error_norm = error_norm.unsqueeze(0).unsqueeze(0)
                    error_norm = error_norm.expand(context.size(0), 1)
            adapter_input = torch.cat([context, error_norm], dim=-1)
        else:
            # Use zeros for prediction_error placeholder
            adapter_input = torch.cat([context, torch.zeros(context.size(0), 1, device=context.device)], dim=-1)
        
        adaptation = self.adapter(adapter_input)
        
        # Modify routing logits
        modified_logits = base_info["logits"] + adaptation
        
        # Re-select with modified logits
        modified_probs = F.softmax(modified_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(
            modified_probs,
            self.config.active_experts,
            dim=-1
        )
        
        # Normalize gates
        gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Update info (create new dict to avoid mutating base_info)
        info = {**base_info}
        info["modified_logits"] = modified_logits
        info["adaptation"] = adaptation
        
        return gates, top_k_indices, info


class SparseRouter(nn.Module):
    """
    Main sparse router for Deep Thought.
    
    Combines base routing with adaptive modification based on
    context and error signals.
    """
    
    def __init__(self, config: RouterConfig, use_adaptive: bool = True, 
                 latent_dim: Optional[int] = None, context_dim: int = 256):
        super().__init__()
        self.config = config
        self.use_adaptive = use_adaptive
        
        if use_adaptive:
            self.router = AdaptiveRouter(config, context_dim=context_dim, latent_dim=latent_dim)
        else:
            self.router = NoisyTopKRouter(config, latent_dim=latent_dim)
    
    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prediction_error: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Route to experts."""
        if self.use_adaptive:
            return self.router(
                h_t, x_t, m_t, context, prediction_error, training
            )
        else:
            return self.router(h_t, x_t, m_t, training)
    
    def compute_losses(self, info: dict) -> dict:
        """Compute routing losses."""
        losses = {}

        # Load balance loss
        if self.use_adaptive:
            losses["load_balance"] = self.router.base_router.load_balance_loss()
        else:
            losses["load_balance"] = self.router.load_balance_loss()

        # Entropy loss
        if "entropy" in info:
            if self.use_adaptive:
                losses["entropy"] = self.router.base_router.entropy_loss(
                    info["entropy"]
                )
            else:
                losses["entropy"] = self.router.entropy_loss(
                    info["entropy"]
                )

        return losses
    
    def get_expert_usage(self) -> torch.Tensor:
        """Get current expert usage statistics."""
        if self.use_adaptive:
            return self.router.base_router.expert_usage.clone()
        else:
            return self.router.expert_usage.clone()
