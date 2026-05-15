"""
Reasoning Engine for Deep Thought.

Implements multi-step deliberation before action selection:
1. Chain-of-Thought (CoT) Reasoning: Multiple internal "thinking" steps
   that refine the agent's hidden state before producing a final action.
2. Self-Consistency Check: Run multiple reasoning paths and select the
   most consistent outcome.
3. Counterfactual Reasoning: Use the world model to simulate "what if"
   scenarios.

CRITICAL FIX: The reasoning engine was a "decorative" module — it ran
but its outputs never trained meaningfully. The consistency_head and
value_refiner received no useful gradient signal because:
  - consistency_scores had no target to predict
  - refined_value had no loss connecting it to actual returns

Now:
  - consistency_head is trained to predict the absolute advantage (how
    confident the agent should be about its current direction)
  - value_refiner is trained to predict actual returns (adding an
    auxiliary value prediction loss)
  - The thought_gru receives gradient from the main policy/value loss
    because refined_h flows directly into the policy and value heads
  - Counterfactual rollouts now provide a gradient signal to the
    world model (not wrapped in no_grad)
  - thought_classifier predicts which reasoning steps are productive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ReasoningEngine(nn.Module):
    """
    Multi-step reasoning engine for deliberative action selection.

    Architecture:
    - thought_gru: GRU that refines h_tilde over N reasoning steps
    - consistency_head: predicts |advantage| for each reasoning step
    - value_refiner: refines the value estimate after reasoning
    - thought_classifier: predicts which reasoning steps are productive

    All sub-modules receive gradient signal from auxiliary losses so
    they learn meaningful representations rather than remaining dead weights.
    """

    def __init__(self, config, latent_dim, num_experts):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim

        # Thought refinement GRU
        self.thought_gru = nn.GRUCell(latent_dim, latent_dim)

        # Consistency scoring - predicts |advantage| (how confident to be)
        self.consistency_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.SiLU(),
            nn.Linear(latent_dim // 4, 1),
        )

        # Value refinement - predicts actual returns
        self.value_refiner = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, 1),
        )

        # Thought quality classifier - predicts which steps improve the state
        self.thought_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.SiLU(),
            nn.Linear(latent_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Step counter
        self._reasoning_step = 0

    def forward(self, h_tilde, x_t, world_model=None, action_dim=None, training=True):
        """
        Run multi-step reasoning.

        The key design: refined_h flows into the policy/value heads, so
        the thought_gru receives gradient from the main PPO loss. The
        consistency_head and value_refiner receive gradient from their
        auxiliary losses (computed in ppo.py).

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
        thought_quality = []

        for step in range(num_steps):
            # Refine hidden state through thought GRU
            h_current = self.thought_gru(h_current, h_current)

            # Score consistency of this reasoning step
            consistency = torch.sigmoid(self.consistency_head(h_current))
            consistency_scores.append(consistency)

            # Score thought quality (is this step improving things?)
            quality = self.thought_classifier(h_current)
            thought_quality.append(quality)

            refined_states.append(h_current)

        # Stack all refined states: (num_steps, batch, latent_dim)
        refined_stack = torch.stack(refined_states, dim=0)
        consistency_stack = torch.stack(consistency_scores, dim=0).squeeze(-1)  # (num_steps, batch)
        quality_stack = torch.stack(thought_quality, dim=0).squeeze(-1)  # (num_steps, batch)

        # Self-consistency: weight refined states by quality scores
        # Use softmax over quality to get weights (more productive steps get more weight)
        quality_weights = F.softmax(quality_stack, dim=0)  # (num_steps, batch)
        quality_weights = quality_weights.unsqueeze(-1)  # (num_steps, batch, 1)

        # Weighted sum of refined states
        refined_h = (refined_stack * quality_weights).sum(dim=0)  # (batch, latent_dim)

        # Counterfactual reasoning (if world model available)
        counterfactual_info = {}
        if world_model is not None and self.config.use_counterfactual and action_dim is not None:
            num_cf = self.config.num_counterfactual_actions
            cf_actions = torch.zeros(batch_size, num_cf, action_dim, device=device)

            cf_values = []
            # IMPORTANT: Do NOT wrap in torch.no_grad(). The world model
            # needs gradient signal from counterfactual rollouts. We use
            # detach only on the input x_t to prevent double-counting
            # gradients, but the world model forward pass itself is
            # differentiable so its weights receive gradients.
            x_t_detached = x_t.detach()  # Prevent double-counting
            for i in range(num_cf):
                action_idx = i % max(1, action_dim)
                cf_actions[:, i, action_idx] = 1.0
                z_next, r_pred, _ = world_model(x_t_detached, cf_actions[:, i])
                cf_values.append(r_pred.unsqueeze(-1) if r_pred is not None else
                                torch.zeros(batch_size, 1, device=device))

            if cf_values:
                cf_value_stack = torch.cat(cf_values, dim=-1)  # (batch, num_cf)
                best_cf_idx = cf_value_stack.argmax(dim=-1)  # (batch,)
                counterfactual_info["best_action_idx"] = best_cf_idx
                counterfactual_info["cf_values"] = cf_value_stack

                # Auxiliary loss: world model should predict that the best
                # action has higher reward than the average action
                best_values = cf_value_stack.max(dim=-1)[0]
                mean_values = cf_value_stack.mean(dim=-1)
                counterfactual_info["cf_ranking_loss"] = F.relu(mean_values - best_values + 0.1).mean()

        # Value refinement - predict actual returns
        value_input = torch.cat([refined_h, h_tilde], dim=-1)
        refined_value = self.value_refiner(value_input)

        reasoning_info = {
            "consistency_scores": consistency_stack,
            "thought_quality": quality_stack,
            "mean_consistency": consistency_stack.mean().item(),
            "mean_quality": quality_stack.mean().item(),
            "num_reasoning_steps": num_steps,
            "counterfactual_info": counterfactual_info,
            "refined_value": refined_value,
        }

        self._reasoning_step += 1

        return refined_h, reasoning_info

    def compute_auxiliary_losses(self, reasoning_info, advantages=None, returns=None):
        """
        Compute auxiliary losses for the reasoning engine.

        These losses ensure the reasoning engine's sub-modules receive
        gradient signal and learn meaningful representations.

        Args:
            reasoning_info: Dict from forward()
            advantages: GAE advantages (for consistency target)
            returns: Discounted returns (for value target)

        Returns:
            Dict of auxiliary losses
        """
        losses = {}

        # Consistency loss: predict absolute advantage magnitude
        if advantages is not None and 'consistency_scores' in reasoning_info:
            consistency_pred = reasoning_info['consistency_scores'][-1]  # Last step's prediction
            consistency_target = advantages.detach().abs().clamp(0, 1)
            if consistency_pred.dim() > consistency_target.dim():
                consistency_pred = consistency_pred.squeeze(-1)
            losses['consistency_loss'] = F.mse_loss(
                torch.sigmoid(consistency_pred),
                consistency_target
            )

        # Thought quality loss: later steps should be higher quality
        # if reasoning is productive (advantages are high)
        if advantages is not None and 'thought_quality' in reasoning_info:
            quality_scores = reasoning_info['thought_quality']
            has_advantage = (advantages.detach().abs() > 0.1).float()
            if quality_scores.size(0) > 1:
                quality_diff = quality_scores[1:] - quality_scores[:-1]
                quality_loss = F.mse_loss(quality_diff.mean(dim=0), has_advantage)
                losses['quality_loss'] = quality_loss

        # Value refinement loss
        if returns is not None and 'refined_value' in reasoning_info:
            rv = reasoning_info['refined_value'].squeeze()
            if rv.dim() == 0:
                rv = rv.unsqueeze(0)
            rt = returns.detach()
            if rt.dim() == 0:
                rt = rt.unsqueeze(0)
            if rv.size(0) == rt.size(0):
                losses['value_refinement_loss'] = F.mse_loss(rv, rt)

        # Counterfactual ranking loss
        if 'counterfactual_info' in reasoning_info:
            cf_info = reasoning_info['counterfactual_info']
            if 'cf_ranking_loss' in cf_info:
                losses['cf_ranking_loss'] = cf_info['cf_ranking_loss']

        return losses

    def reset(self):
        """Reset reasoning state."""
        self._reasoning_step = 0
