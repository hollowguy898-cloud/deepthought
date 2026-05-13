"""
Loss functions for Deep Thought training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01
) -> Dict[str, torch.Tensor]:
    """
    Compute PPO loss components.
    
    Args:
        log_probs: Current policy log probabilities
        old_log_probs: Old policy log probabilities
        advantages: Advantage estimates
        values: Value predictions
        returns: Return targets
        clip_eps: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        
    Returns:
        Dictionary of loss components
    """
    # Policy loss
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss - ensure same shape
    if values.dim() != returns.dim():
        if values.dim() > returns.dim():
            values = values.squeeze(-1)
        else:
            returns = returns.unsqueeze(-1)
    value_loss = F.mse_loss(values, returns)
    
    # Entropy bonus - use proper entropy computation
    # For discrete: entropy = -sum(p * log(p))
    # For continuous with log_probs: use -log_probs as proxy
    # Clamp log_probs for numerical stability
    clamped_log_probs = torch.clamp(log_probs, min=-20, max=20)
    probs = torch.exp(clamped_log_probs)
    entropy = -(probs * clamped_log_probs).sum(dim=-1).mean()
    # Clamp entropy to prevent extreme values
    entropy = torch.clamp(entropy, min=-10, max=10)
    entropy_loss = -entropy_coef * entropy
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss + entropy_loss
    
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "total_loss": total_loss,
        "entropy": entropy,
    }


def compute_world_model_loss(
    z_pred: torch.Tensor,
    z_next: torch.Tensor,
    reward_pred: Optional[torch.Tensor],
    reward: Optional[torch.Tensor],
    done_pred: Optional[torch.Tensor],
    done: Optional[torch.Tensor],
    obs_recon: Optional[torch.Tensor],
    observation: Optional[torch.Tensor],
    state_coef: float = 1.0,
    reward_coef: float = 1.0,
    done_coef: float = 1.0,
    recon_coef: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Compute world model loss components.
    
    Args:
        z_pred: Predicted next latent
        z_next: Actual next latent
        reward_pred: Predicted reward
        reward: Actual reward
        done_pred: Predicted done probability
        done: Actual done
        obs_recon: Reconstructed observation
        observation: Actual observation
        state_coef: State prediction coefficient
        reward_coef: Reward prediction coefficient
        done_coef: Done prediction coefficient
        recon_coef: Reconstruction coefficient
        
    Returns:
        Dictionary of loss components
    """
    losses = {}
    
    # State prediction loss
    state_loss = F.mse_loss(z_pred, z_next)
    losses["state_loss"] = state_coef * state_loss
    
    # Reward prediction loss
    if reward_pred is not None and reward is not None:
        reward_loss = F.mse_loss(reward_pred, reward)
        losses["reward_loss"] = reward_coef * reward_loss
    
    # Done prediction loss
    if done_pred is not None and done is not None:
        done_loss = F.binary_cross_entropy(done_pred, done.float())
        losses["done_loss"] = done_coef * done_loss
    
    # Observation reconstruction loss
    if obs_recon is not None and observation is not None:
        recon_loss = F.mse_loss(obs_recon, observation)
        losses["recon_loss"] = recon_coef * recon_loss
    
    # Total loss
    total_loss = sum(losses.values())
    losses["total_loss"] = total_loss
    
    return losses


def compute_compute_penalty(
    expert_costs: Dict[int, float],
    gates: torch.Tensor,
    selected_indices: torch.Tensor,
    coef: float = 0.001
) -> torch.Tensor:
    """
    Compute compute penalty for sparse activation.
    
    Args:
        expert_costs: Cost per expert
        gates: Gate values
        selected_indices: Selected expert indices
        coef: Penalty coefficient
        
    Returns:
        compute_loss: Compute penalty loss (tensor)
    """
    total_cost = torch.tensor(0.0, device=gates.device)
    
    for i in range(selected_indices.size(-1)):
        expert_idx = selected_indices[:, i]
        gate = gates[:, i]
        
        for exp_id in expert_idx.unique():
            exp_id_val = exp_id.item()
            if exp_id_val in expert_costs:
                cost = expert_costs[exp_id_val]
                mask = (expert_idx == exp_id).float()
                total_cost = total_cost + (gate * mask * cost).sum()
    
    compute_loss = coef * total_cost / gates.size(0)
    
    return compute_loss


def compute_orthogonality_loss(
    subspaces: Tuple[torch.Tensor, ...],
    coef: float = 0.01
) -> torch.Tensor:
    """
    Compute orthogonality loss for factorized latents.
    
    Args:
        subspaces: Tuple of subspace vectors
        coef: Loss coefficient
        
    Returns:
        orth_loss: Orthogonality loss
    """
    num = len(subspaces)
    orth_loss = 0.0
    
    for i in range(num):
        for j in range(i + 1, num):
            corr = torch.abs(
                (subspaces[i] * subspaces[j]).sum(dim=-1).mean()
            )
            orth_loss += corr
    
    orth_loss = coef * orth_loss / (num * (num - 1) / 2)
    
    return orth_loss


def compute_sparsity_loss(
    sparse_z: torch.Tensor,
    coef: float = 0.01
) -> torch.Tensor:
    """
    Compute sparsity loss for sparse coding.
    
    Args:
        sparse_z: Sparse activations
        coef: Loss coefficient
        
    Returns:
        sparse_loss: Sparsity loss
    """
    sparse_loss = coef * sparse_z.abs().sum(dim=-1).mean()
    return sparse_loss


def compute_anchor_loss(
    expert_output: torch.Tensor,
    prototype: torch.Tensor,
    coef: float = 1.0
) -> torch.Tensor:
    """
    Compute anchor loss for expert fidelity.
    
    Args:
        expert_output: Expert output
        prototype: Feature prototype
        coef: Loss coefficient
        
    Returns:
        anchor_loss: Anchor loss
    """
    anchor_loss = coef * F.mse_loss(expert_output, prototype)
    return anchor_loss
