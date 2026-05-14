"""
Working memory module for Deep Thought.

Implements fast, volatile short-term memory for immediate reasoning
and short-horizon planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from deep_thought.config import MemoryConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5 + self.eps)
        return self.weight * x / (norm + self.eps)


class WorkingMemory(nn.Module):
    """
    Working memory for immediate reasoning.
    
    Maintains short-term state that is constantly overwritten.
    Used for:
    - Immediate reasoning
    - Short-horizon planning
    - "What am I doing right now"
    
    Implemented as a gated recurrent state.
    """
    
    def __init__(self, config: MemoryConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.memory_size = config.working_memory_size
        
        # GRU-based working memory
        self.gru = nn.GRU(
            input_size=latent_dim * 2,  # x_t + memory_read
            hidden_size=latent_dim,
            batch_first=True
        )
        
        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.Sigmoid(),
        )
        
        # Residual connection coefficient
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Normalization
        self.norm = RMSNorm(latent_dim)
    
    def forward(
        self,
        h_prev: torch.Tensor,
        x_t: torch.Tensor,
        memory_read: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update working memory.
        
        Args:
            h_prev: Previous hidden state
            x_t: Current encoded observation
            memory_read: Read from episodic/semantic memory
            
        Returns:
            h_t: Updated hidden state
            delta_h: Change in hidden state
        """
        # Concatenate inputs
        gru_input = torch.cat([x_t, memory_read], dim=-1).unsqueeze(1)
        
        # GRU update
        gru_out, h_t = self.gru(gru_input, h_prev.unsqueeze(0))
        h_t = h_t.squeeze(0)
        
        # Compute update gate
        gate_input = torch.cat([h_prev, x_t, memory_read], dim=-1)
        update = self.update_gate(gate_input)
        
        # Residual update
        delta_h = self.alpha * (h_t - h_prev) * update
        
        # Apply update
        h_t = h_prev + delta_h
        
        # Normalize
        h_t = self.norm(h_t)
        
        return h_t, delta_h
    
    def reset(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Reset working memory to zeros."""
        return torch.zeros(batch_size, self.latent_dim, device=device)
