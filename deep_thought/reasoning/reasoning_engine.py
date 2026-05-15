"""
Reasoning Engine for Deep Thought.

Implements multi-step deliberation before action selection:
1. Chain-of-Thought (CoT) Reasoning: Multiple internal "thinking" steps
   that refine the agent's hidden state before producing a final action.
2. Self-Consistency Check: Run multiple reasoning paths and select the
   most consistent outcome.
3. Counterfactual Reasoning: Use the world model to simulate "what if"
   scenarios.
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ReasoningEngine(nn.Module):
    """
    Multi-step reasoning engine for deliberative action selection.
    
    Implements:
    - Chain-of-Thought reasoning: refine hidden state over multiple steps
    - Self-consistency: majority vote across multiple reasoning paths
    - Counterfactual simulation: "what if" via world model rollouts
    
    Architecture:
    - thought_gru: GRU that refines h_tilde over N reasoning steps
    - consistency_head: predicts consistency score for each reasoning path
    - value_refiner: refines the value estimate after reasoning
    """
    
    def __init__(self, config, latent_dim, num_experts):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        # Thought refinement GRU
        self.thought_gru = nn.GRUCell(latent_dim, latent_dim)
        
        # Consistency scoring
        self.consistency_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.SiLU(),
            nn.Linear(latent_dim // 4, 1),
        )
        
        # Value refinement
        self.value_refiner = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, 1),
        )
        
        # Step counter
        self._reasoning_step = 0
    
    def forward(self, h_tilde, x_t, world_model=None, action_dim=None, training=True):
        """
        Run multi-step reasoning.
        
        Args:
            h_tilde: Current hidden state after expert processing
            x_t: Current latent observation
            world_model: Optional world model for counterfactual rollouts
            action_dim: Action dimension for counterfactual simulation
            training: Whether in training mode
            
        Returns:
            refined_h: Refined hidden state
            reasoning_info: Dict with reasoning metadata
        """
        batch_size = h_tilde.size(0)
        device = h_tilde.device
        num_steps = self.config.num_reasoning_steps if training else max(1, self.config.num_reasoning_steps // 2)
        
        # Chain-of-Thought reasoning
        h_current = h_tilde
        consistency_scores = []
        refined_states = []
        
        for step in range(num_steps):
            # Refine hidden state through thought GRU
            h_current = self.thought_gru(h_current, h_current)
            
            # Score consistency of this reasoning step
            consistency = torch.sigmoid(self.consistency_head(h_current))
            consistency_scores.append(consistency)
            refined_states.append(h_current)
        
        # Stack all refined states: (num_steps, batch, latent_dim)
        refined_stack = torch.stack(refined_states, dim=0)
        consistency_stack = torch.stack(consistency_scores, dim=0).squeeze(-1)  # (num_steps, batch)
        
        # Self-consistency: weight refined states by consistency scores
        # Use softmax over consistency to get weights
        consistency_weights = F.softmax(consistency_stack, dim=0)  # (num_steps, batch)
        consistency_weights = consistency_weights.unsqueeze(-1)  # (num_steps, batch, 1)
        
        # Weighted sum of refined states
        refined_h = (refined_stack * consistency_weights).sum(dim=0)  # (batch, latent_dim)
        
        # Counterfactual reasoning (if world model available)
        counterfactual_info = {}
        if world_model is not None and self.config.use_counterfactual and action_dim is not None:
            num_cf = self.config.num_counterfactual_actions
            # Generate candidate actions
            # For discrete action spaces: generate one-hot encoded actions
            # For continuous action spaces: generate random continuous actions
            cf_actions = torch.zeros(batch_size, num_cf, action_dim, device=device)
            
            # Simulate each candidate action (use torch.no_grad to prevent
            # gradient flow through counterfactual simulations which would
            # destabilize training)
            cf_values = []
            with torch.no_grad():
                for i in range(num_cf):
                    # Create one-hot action for discrete space
                    action_idx = i % action_dim  # Cycle through actions
                    cf_actions[:, i, action_idx] = 1.0
                    z_next, r_pred, _ = world_model(x_t, cf_actions[:, i])
                    cf_values.append(r_pred.unsqueeze(-1) if r_pred is not None else 
                                    torch.zeros(batch_size, 1, device=device))
            
            if cf_values:
                cf_value_stack = torch.cat(cf_values, dim=-1)  # (batch, num_cf)
                best_cf_idx = cf_value_stack.argmax(dim=-1)  # (batch,)
                counterfactual_info["best_action_idx"] = best_cf_idx
                counterfactual_info["cf_values"] = cf_value_stack
        
        # Value refinement
        value_input = torch.cat([refined_h, h_tilde], dim=-1)
        refined_value = self.value_refiner(value_input)
        
        reasoning_info = {
            "consistency_scores": consistency_stack,
            "mean_consistency": consistency_stack.mean().item(),
            "num_reasoning_steps": num_steps,
            "counterfactual_info": counterfactual_info,
            "refined_value": refined_value,
        }
        
        self._reasoning_step += 1
        
        return refined_h, reasoning_info
    
    def reset(self):
        """Reset reasoning state."""
        self._reasoning_step = 0
