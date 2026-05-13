"""
World model module for Deep Thought.

Implements latent dynamics model for predicting next states, rewards,
and termination signals. Enables internal planning and imagination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from deep_thought.config import WorldModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5 + self.eps)
        return self.weight * x / (norm + self.eps)


class WorldModel(nn.Module):
    """
    Latent dynamics world model.
    
    Predicts:
    - Next latent state
    - Reward
    - Done probability
    
    Enables imagination-based planning and better credit assignment.
    """
    
    def __init__(self, config: WorldModelConfig, action_dim: int):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.action_dim = action_dim
        
        # Dynamics network (z_t, a_t) -> z_{t+1}
        self.dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + action_dim, config.hidden_dim),
            nn.SiLU(),
            RMSNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            RMSNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        
        # Reward predictor
        if config.predict_reward:
            self.reward_head = nn.Sequential(
                nn.Linear(config.latent_dim + action_dim, config.hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
        
        # Done predictor
        if config.predict_done:
            self.done_head = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
        
        # Observation decoder (for reconstruction loss) - initialized lazily
        self.observation_decoder = None
        self._obs_dim = None
    
    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict next state, reward, and done.
        
        Args:
            z_t: Current latent state
            action: Current action
            
        Returns:
            z_next: Predicted next latent state
            reward_pred: Predicted reward (if enabled)
            done_pred: Predicted done probability (if enabled)
        """
        # Concatenate state and action
        za = torch.cat([z_t, action], dim=-1)
        
        # Predict next state
        z_next = self.dynamics(za)
        
        # Predict reward
        reward_pred = None
        if self.config.predict_reward:
            reward_pred = self.reward_head(za).squeeze(-1)
        
        # Predict done
        done_pred = None
        if self.config.predict_done:
            done_pred = self.done_head(z_t).squeeze(-1)
            done_pred = torch.sigmoid(done_pred)
        
        return z_next, reward_pred, done_pred
    
    def decode_observation(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation space."""
        if self.observation_decoder is None:
            # Cannot decode without observation_dim
            return z
        return self.observation_decoder(z)

    def set_observation_dim(self, obs_dim: int):
        """Set observation dimension and initialize decoder."""
        if self._obs_dim == obs_dim and self.observation_decoder is not None:
            return
        self._obs_dim = obs_dim
        device = next(self.parameters()).device
        self.observation_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, obs_dim),
        ).to(device)
    
    def imagine_rollout(
        self,
        z_0: torch.Tensor,
        policy_fn,
        horizon: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagine a trajectory rollout.
        
        Args:
            z_0: Initial latent state
            policy_fn: Function that takes latent and returns action
            horizon: Rollout length
            
        Returns:
            z_seq: Sequence of latent states
            r_seq: Sequence of predicted rewards
            d_seq: Sequence of done probabilities
        """
        batch_size = z_0.size(0)
        device = z_0.device
        
        z_seq = [z_0]
        r_seq = []
        d_seq = []
        
        z_t = z_0
        done = torch.zeros(batch_size, device=device)
        
        for t in range(horizon):
            # Get action from policy
            action = policy_fn(z_t)
            
            # Predict next state
            z_next, r_pred, d_pred = self.forward(z_t, action)
            
            # Mask done states
            z_next = z_next * (1 - done).unsqueeze(-1)
            
            # Store
            z_seq.append(z_next)
            if r_pred is not None:
                r_seq.append(r_pred * (1 - done))
            if d_pred is not None:
                d_seq.append(d_pred)
            
            # Update
            z_t = z_next
            done = done + d_pred if d_pred is not None else done
            done = done.clamp(max=1.0)
        
        z_seq = torch.stack(z_seq, dim=1)  # [B, T+1, D]
        r_seq = torch.stack(r_seq, dim=1) if r_seq else None
        d_seq = torch.stack(d_seq, dim=1) if d_seq else None
        
        return z_seq, r_seq, d_seq
    
    def compute_loss(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        observation: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute world model loss.
        
        Args:
            z_t: Current latent state
            action: Current action
            z_next: Actual next latent state
            reward: Actual reward
            done: Actual done
            observation: Actual observation (for reconstruction)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Predict
        z_pred, r_pred, d_pred = self.forward(z_t, action)
        
        # Next state loss
        state_loss = F.mse_loss(z_pred, z_next)
        losses["state"] = state_loss
        
        # Reward loss
        if r_pred is not None and reward is not None:
            reward_loss = F.mse_loss(r_pred, reward)
            losses["reward"] = reward_loss
        
        # Done loss
        if d_pred is not None and done is not None:
            done_loss = F.binary_cross_entropy(d_pred, done.float())
            losses["done"] = done_loss
        
        # Observation reconstruction loss
        if observation is not None:
            obs_recon = self.decode_observation(z_next)
            recon_loss = F.mse_loss(obs_recon, observation)
            losses["reconstruction"] = recon_loss
        
        return losses
    
    def get_prediction_error(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction error for adaptation signals.
        
        Args:
            z_t: Current latent state
            action: Current action
            z_next: Actual next latent state
            
        Returns:
            Prediction error
        """
        z_pred, _, _ = self.forward(z_t, action)
        error = F.mse_loss(z_pred, z_next, reduction='none').mean(dim=-1)
        return error


class EnsembleWorldModel(nn.Module):
    """
    Ensemble of world models for uncertainty estimation.
    
    Uses multiple world models to estimate epistemic uncertainty
    for exploration and adaptation.
    """
    
    def __init__(self, config: WorldModelConfig, action_dim: int, num_models: int = 5):
        super().__init__()
        self.num_models = num_models
        
        self.models = nn.ModuleList([
            WorldModel(config, action_dim) for _ in range(num_models)
        ])
    
    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        return_ensemble: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with optional ensemble output.
        
        Args:
            z_t: Current latent state
            action: Current action
            return_ensemble: Whether to return all model outputs
            
        Returns:
            z_next: Predicted next latent state (mean if ensemble)
            reward_pred: Predicted reward
            done_pred: Predicted done probability
        """
        if return_ensemble:
            z_nexts = []
            r_preds = []
            d_preds = []
            
            for model in self.models:
                z_n, r_p, d_p = model(z_t, action)
                z_nexts.append(z_n)
                r_preds.append(r_p)
                d_preds.append(d_p)
            
            z_next = torch.stack(z_nexts, dim=0).mean(dim=0)
            r_pred = torch.stack(r_preds, dim=0).mean(dim=0) if r_preds[0] else None
            d_pred = torch.stack(d_preds, dim=0).mean(dim=0) if d_preds[0] else None
            
            return z_next, r_pred, d_pred, (z_nexts, r_preds, d_preds)
        else:
            # Use first model for efficiency
            return self.models[0](z_t, action)
    
    def get_uncertainty(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute epistemic uncertainty from ensemble disagreement.
        
        Args:
            z_t: Current latent state
            action: Current action
            z_next: Actual next latent state
            
        Returns:
            Uncertainty estimate
        """
        z_preds = []
        for model in self.models:
            z_p, _, _ = model(z_t, action)
            z_preds.append(z_p)
        
        z_preds = torch.stack(z_preds, dim=0)  # [N, B, D]
        
        # Variance across ensemble
        uncertainty = z_preds.var(dim=0).mean(dim=-1)  # [B]
        
        return uncertainty
    
    def compute_loss(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        z_next: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor
    ) -> dict:
        """Compute loss averaged over ensemble."""
        total_losses = {}
        
        for model in self.models:
            losses = model.compute_loss(z_t, action, z_next, reward, done)
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = []
                total_losses[key].append(value)
        
        # Average losses
        avg_losses = {
            key: torch.stack(values).mean()
            for key, values in total_losses.items()
        }
        
        return avg_losses
