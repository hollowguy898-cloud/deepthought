"""
PPO trainer for Deep Thought.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import deque

from deep_thought.config import TrainingConfig


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    """
    
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.latents = []
        self.memory_reads = []
    
    def add(
        self,
        observation,
        action,
        reward,
        done,
        log_prob,
        value,
        latent,
        memory_read
    ):
        """Add a timestep to buffer."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(log_prob)
        # Store value as scalar for consistent advantage computation
        if isinstance(value, torch.Tensor):
            self.values.append(value.item())
        else:
            self.values.append(float(value))
        self.latents.append(latent)
        self.memory_reads.append(memory_read)
    
    def clear(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.latents = []
        self.memory_reads = []
    
    def get_batch(self) -> Dict:
        """Get batch as dictionary."""
        obs = torch.stack(self.observations)
        actions = torch.stack(self.actions)
        latents = torch.stack(self.latents)
        memory_reads = torch.stack(self.memory_reads)

        # Squeeze out extra dimensions from single-step unsqueeze(0) during rollout
        # obs: (T, 1, ...) -> (T, ...), actions: (T, 1) -> (T,)
        if obs.dim() == 3 and obs.size(1) == 1:
            obs = obs.squeeze(1)
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions = actions.squeeze(-1)
        if latents.dim() == 3 and latents.size(1) == 1:
            latents = latents.squeeze(1)
        if memory_reads.dim() == 3 and memory_reads.size(1) == 1:
            memory_reads = memory_reads.squeeze(1)

        return {
            "observations": obs,
            "actions": actions,
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
            "log_probs": torch.stack(self.log_probs),
            "values": torch.tensor(self.values, dtype=torch.float32),
            "latents": latents,
            "memory_reads": memory_reads,
        }
    
    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO trainer for Deep Thought.
    
    Implements Proximal Policy Optimization with:
    - GAE advantage estimation
    - Value clipping
    - Entropy regularization
    - Multiple epochs per update
    """
    
    def __init__(self, config: TrainingConfig, model: nn.Module,
                 action_space: str = "discrete", action_dim: int = 2):
        self.config = config
        self.model = model
        self.action_space = action_space
        self.action_dim = action_dim
        
        # Rollout buffer
        self.buffer = RolloutBuffer(capacity=config.batch_size)
        
        # GAE parameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        
        # Training parameters
        self.clip_eps = config.clip_eps
        self.value_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.ppo_epochs = config.ppo_epochs
        self.target_kl = config.target_kl
        self.max_grad_norm = config.max_grad_norm
    
    def collect_rollout(
        self,
        env,
        observation,
        h_t,
        m_t,
        device
    ) -> Tuple[float, int, bool, torch.Tensor]:
        """
        Collect a rollout of experience.
        
        Args:
            env: Environment
            observation: Current observation (tensor on device)
            h_t: Hidden state
            m_t: Memory read
            device: Torch device
            
        Returns:
            total_reward: Total reward in rollout
            steps: Number of steps in this rollout
            episode_done: Whether the episode ended
            last_observation: The last observation (after the last env.step)
        """
        total_reward = 0.0
        steps = 0
        episode_done = False
        
        for _ in range(self.config.rollout_length):
            # Get action from model
            with torch.no_grad():
                latent, encoder_info = self.model.encoder(observation)
                
                # Get policy and value
                policy_output = self.model.policy_head(latent)
                value_output = self.model.critic_head(latent)
                
                if self.action_space == "discrete":
                    action_probs = F.softmax(policy_output, dim=-1)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                else:
                    # Continuous action space
                    mean = policy_output[:, :self.action_dim]
                    log_std = policy_output[:, self.action_dim:]
                    log_std = torch.clamp(log_std, -20, 2)
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                value = value_output.squeeze(-1)
            
            # Step environment
            action_np = action.cpu().numpy()
            if action_np.ndim == 0:
                action_np = action_np.item()
            else:
                action_np = action_np[0]
            
            next_observation, reward, done, truncated, info = env.step(action_np)
            done = done or truncated
            
            # Store in buffer
            self.buffer.add(
                observation,
                action,
                reward,
                done,
                log_prob,
                value,
                latent,
                m_t
            )
            
            total_reward += reward
            steps += 1
            
            # Move next observation to device
            observation = torch.tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            if done:
                episode_done = True
                break
        
        return total_reward, steps, episode_done, observation
    
    def compute_advantages(self, bootstrap_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            bootstrap_value: Value estimate for the state after the last
                transition in the buffer. Used for bootstrapping when the
                rollout did not end with a terminal state.
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        rewards = self.buffer.rewards
        values = self.buffer.values  # Now floats, not tensors
        dones = self.buffer.dones
        
        advantages = []
        returns = []
        
        # Bootstrap value
        advantage = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = bootstrap_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            advantages.insert(0, advantage)
            
            returns.insert(0, advantage + values[t])
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(
        self,
        optimizer: torch.optim.Optimizer,
        bootstrap_value: float = 0.0
    ) -> Dict[str, float]:
        """
        Update model using PPO.
        
        Args:
            optimizer: Optimizer
            bootstrap_value: Value estimate for bootstrapping when the
                last transition in the buffer is not terminal.
            
        Returns:
            metrics: Training metrics
        """
        if len(self.buffer) == 0:
            return {"total_loss": 0.0}
        
        # Compute advantages and returns with bootstrap
        advantages, returns = self.compute_advantages(bootstrap_value=bootstrap_value)
        
        # Get batch
        batch = self.buffer.get_batch()
        
        metrics = {}
        
        # Multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle batch
            indices = torch.randperm(len(batch["observations"]))
            
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_kl = 0.0
            num_updates = 0
            
            # Mini-batch updates
            mini_batch_size = min(64, len(indices))
            for start in range(0, len(indices), mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                # Get mini-batch
                mb_obs = batch["observations"][mb_indices]
                mb_actions = batch["actions"][mb_indices]
                mb_old_log_probs = batch["log_probs"][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_latents = batch["latents"][mb_indices]
                
                # Forward pass through FULL agent pipeline for gradient flow.
                # We use a stateless forward pass that goes through encoder →
                # router → experts → policy/value heads, but WITHOUT the
                # agent's persistent GRU hidden state (which is batch_size=1
                # from rollout collection and incompatible with mini-batches).
                # This ensures router, experts, and all learnable components
                # receive gradients during the PPO update.

                # Step 1: Encode observations (gradient flows through encoder)
                latent, _ = self.model.encoder(mb_obs)

                # Step 2: Route through router and experts using latent
                mb_batch_size = mb_obs.size(0)
                mb_device = mb_obs.device
                mb_h_t = torch.zeros(mb_batch_size, self.model.config.encoder.latent_dim, device=mb_device)
                mb_m_t = torch.zeros(mb_batch_size, self.model.config.encoder.latent_dim, device=mb_device)
                mb_context = None
                if self.model.meta_learning is not None:
                    mb_context = torch.zeros(mb_batch_size, self.model.config.meta_learning.context_dim, device=mb_device)

                # Router (gradient flows through router weights)
                gates, selected_indices, router_info = self.model.router(
                    mb_h_t, latent, mb_m_t, mb_context,
                    prediction_error=None, training=True, detach_gates=False
                )

                # Experts (gradient flows through expert weights)
                delta_h, compute_costs = self.model.expert_bank(
                    mb_h_t, selected_indices, gates
                )
                h_tilde = mb_h_t + delta_h

                # Hierarchical expert society (if enabled)
                if self.model.hierarchical is not None:
                    hierarchical_output, _ = self.model.hierarchical(
                        h_tilde, latent, context=mb_context
                    )
                    h_tilde = h_tilde + hierarchical_output

                # Meta-learning adaptation (if enabled)
                if self.model.meta_learning is not None and mb_context is not None:
                    h_adapted, _ = self.model.meta_learning.adapt(
                        h_tilde, mb_context, gradient=None
                    )
                    h_tilde = h_adapted

                # Policy and value heads (gradient flows through heads)
                policy_output = self.model.policy_head(h_tilde)
                value_output = self.model.critic_head(h_tilde)
                
                # Compute new log probs
                if self.action_space == "discrete":
                    action_probs = F.softmax(policy_output, dim=-1)
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(mb_actions)
                else:
                    mean = policy_output[:, :self.action_dim]
                    log_std = policy_output[:, self.action_dim:]
                    log_std = torch.clamp(log_std, -20, 2)
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                
                value = value_output.squeeze(-1)
                
                # Compute PPO loss with proper entropy from the distribution
                entropy = dist.entropy()
                if self.action_space == "continuous":
                    entropy = entropy.sum(dim=-1)
                entropy_mean = entropy.mean()
                
                from deep_thought.optimization.losses import compute_ppo_loss
                loss_dict = compute_ppo_loss(
                    new_log_probs,
                    mb_old_log_probs,
                    mb_advantages,
                    value,
                    mb_returns,
                    self.clip_eps,
                    self.value_coef,
                    self.entropy_coef,
                    entropy_mean=entropy_mean
                )
                
                # Compute KL divergence
                kl = (mb_old_log_probs - new_log_probs).mean()
                
                total_policy_loss += loss_dict["policy_loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                total_entropy += loss_dict["entropy"].item()
                total_kl += kl.item()
                num_updates += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                optimizer.step()
                
                # Early stopping if KL is too high
                if kl > self.target_kl * 1.5:
                    break
            
            # Average metrics
            if num_updates > 0:
                metrics[f"epoch_{epoch}_policy_loss"] = total_policy_loss / num_updates
                metrics[f"epoch_{epoch}_value_loss"] = total_value_loss / num_updates
                metrics[f"epoch_{epoch}_entropy"] = total_entropy / num_updates
                metrics[f"epoch_{epoch}_kl"] = total_kl / num_updates
        
        # Add aggregate metrics for easy access
        if any("policy_loss" in k for k in metrics):
            policy_losses = [v for k, v in metrics.items() if "policy_loss" in k]
            metrics["policy_loss"] = sum(policy_losses) / len(policy_losses)
        if any("value_loss" in k for k in metrics):
            value_losses = [v for k, v in metrics.items() if "value_loss" in k]
            metrics["value_loss"] = sum(value_losses) / len(value_losses)
        metrics["total_loss"] = metrics.get("policy_loss", 0.0) + metrics.get("value_loss", 0.0)
        
        # Clear buffer
        self.buffer.clear()
        
        return metrics
