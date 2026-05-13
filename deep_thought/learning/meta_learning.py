"""
Meta-learning layer for Deep Thought.

Implements gradient-based meta-learning and context-based adaptation
for rapid task inference and fast adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from copy import deepcopy

from deep_thought.config import MetaLearningConfig


class ContextEncoder(nn.Module):
    """
    Encodes trajectory history into task context embedding.
    
    Infers "what kind of environment am I in?" from experience.
    """
    
    def __init__(self, config: MetaLearningConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.context_dim = config.context_dim
        self.latent_dim = latent_dim
        
        # Context encoder (processes trajectory history)
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.context_dim),
        )
        
        # GRU for temporal aggregation
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=self.context_dim,
            batch_first=True
        )
    
    def forward(
        self,
        trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode trajectory into context.
        
        Args:
            trajectory: Sequence of latent states [B, T, D]
            
        Returns:
            context: Context embedding [B, context_dim]
        """
        # Use GRU to aggregate temporal information
        _, h_n = self.gru(trajectory)
        context = h_n.squeeze(0)
        
        return context


class MetaLearningLayer(nn.Module):
    """
    Meta-learning layer for fast adaptation.
    
    Combines:
    - Gradient-based meta-learning (MAML-style)
    - Context-based adaptation
    - Fast weight memory
    """
    
    def __init__(self, config: MetaLearningConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        # Context encoder
        self.context_encoder = ContextEncoder(config, latent_dim)
        
        # Fast weights
        if config.use_fast_weights:
            from deep_thought.learning.fast_weights import FastWeightMemory
            self.fast_weights = FastWeightMemory(config, latent_dim)
        else:
            self.fast_weights = None
        
        # Adaptation network (modifies parameters based on context)
        self.adaptation_net = nn.Sequential(
            nn.Linear(config.context_dim + latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        
        # Inner learning rate (learned)
        self.inner_lr = nn.Parameter(torch.tensor(config.inner_lr))
        
        # Trajectory buffer for context encoding
        self.trajectory_buffer: list = []
        self.buffer_size = 100
    
    def update_context(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Update context from recent trajectory.
        
        Args:
            latent: Current latent state
            
        Returns:
            context: Current context embedding
        """
        # Add to buffer
        self.trajectory_buffer.append(latent.detach())
        
        # Keep buffer bounded
        if len(self.trajectory_buffer) > self.buffer_size:
            self.trajectory_buffer.pop(0)
        
        # Encode trajectory if we have enough history
        if len(self.trajectory_buffer) >= 10:
            trajectory = torch.stack(self.trajectory_buffer[-10:], dim=1)
            context = self.context_encoder(trajectory)
        else:
            # Default context
            device = latent.device
            context = torch.zeros(1, self.config.context_dim, device=device)
        
        return context
    
    def adapt(
        self,
        h_t: torch.Tensor,
        context: torch.Tensor,
        gradient: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Adapt hidden state based on context and gradients.
        
        Args:
            h_t: Current hidden state
            context: Context embedding
            gradient: Optional gradient for meta-update
            
        Returns:
            h_adapted: Adapted hidden state
            info: Adaptation information
        """
        info = {}
        
        # Fast weight adaptation
        if self.fast_weights is not None:
            h_fast, fast_contrib = self.fast_weights(h_t, update=True)
            info["fast_contribution"] = fast_contrib
            info["fast_weight_norm"] = self.fast_weights.get_norm()
        else:
            h_fast = torch.zeros_like(h_t)
            info["fast_contribution"] = torch.zeros_like(h_t)
            info["fast_weight_norm"] = 0.0
        
        # Context-based adaptation
        adapt_input = torch.cat([context, h_t], dim=-1)
        adaptation = self.adaptation_net(adapt_input)
        
        # Gradient-based adaptation (MAML-style)
        if gradient is not None:
            # Simpler version: apply gradient step
            h_adapted = h_t - self.inner_lr * gradient
        else:
            h_adapted = h_t
        
        # Combine adaptations
        h_adapted = h_adapted + adaptation + h_fast
        
        info["adaptation"] = adaptation
        info["context"] = context
        
        return h_adapted, info
    
    def meta_update(
        self,
        model: nn.Module,
        support_loss: torch.Tensor,
        query_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform meta-learning update (MAML-style).
        
        Args:
            model: Model to meta-update
            support_loss: Loss on support set
            query_loss: Loss on query set
            
        Returns:
            meta_loss: Meta-learning loss
        """
        # Compute gradients on support set
        support_gradients = torch.autograd.grad(
            support_loss,
            model.parameters(),
            create_graph=True,
            allow_unused=True
        )
        
        # Create adapted model (conceptual)
        # In practice, this would require more complex implementation
        # For now, use query loss as meta objective
        
        meta_loss = query_loss
        
        return meta_loss
    
    def reset_context(self):
        """Reset trajectory buffer."""
        self.trajectory_buffer = []
    
    def reset_fast_weights(self):
        """Reset fast weights."""
        if self.fast_weights is not None:
            self.fast_weights.reset()
