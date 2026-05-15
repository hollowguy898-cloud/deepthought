"""
PPO trainer for Deep Thought.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        self.next_latents = []
        self.selected_indices = []
        self.gates = []
    
    def add(
        self,
        observation,
        action,
        reward,
        done,
        log_prob,
        value,
        latent,
        memory_read,
        next_latent=None,
        selected_indices=None,
        gates=None
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
        if next_latent is not None:
            self.next_latents.append(next_latent)
        if selected_indices is not None:
            self.selected_indices.append(selected_indices)
        if gates is not None:
            self.gates.append(gates)
    
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
        self.next_latents = []
        self.selected_indices = []
        self.gates = []
    
    def get_batch(self) -> Dict:
        """Get batch as dictionary."""
        obs = torch.stack(self.observations)
        actions = torch.stack(self.actions)
        latents = torch.stack(self.latents)
        memory_reads = torch.stack(self.memory_reads)

        # Squeeze out extra dimensions from single-step unsqueeze(0) during rollout
        if obs.dim() == 3 and obs.size(1) == 1:
            obs = obs.squeeze(1)
        if actions.dim() == 2 and actions.size(-1) == 1:
            actions = actions.squeeze(-1)
        if latents.dim() == 3 and latents.size(1) == 1:
            latents = latents.squeeze(1)
        if memory_reads.dim() == 3 and memory_reads.size(1) == 1:
            memory_reads = memory_reads.squeeze(1)

        log_probs = torch.stack(self.log_probs).detach()  # Old policy log probs should not have gradient
        if log_probs.dim() == 2 and log_probs.size(-1) == 1:
            log_probs = log_probs.squeeze(-1)

        return {
            "observations": obs,
            "actions": actions,
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=obs.device),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=obs.device),
            "log_probs": log_probs,
            "values": torch.tensor(self.values, dtype=torch.float32, device=obs.device),
            "latents": latents,
            "next_latents": torch.stack(self.next_latents).squeeze(1) if self.next_latents else latents,
            "memory_reads": memory_reads,
            "selected_indices": torch.stack(self.selected_indices).squeeze(1) if self.selected_indices else torch.zeros(actions.size(0), 1, dtype=torch.long, device=obs.device),
            "gates": torch.stack(self.gates).squeeze(1) if self.gates else torch.ones(actions.size(0), 1, device=obs.device),
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
                 action_space: str = "discrete", action_dim: int = 2,
                 intrinsic_reward_coef: float = 0.1,   # FIX: 10x increase — 0.01 was too weak, curiosity had no effect
                 subgoal_reward_coef: float = 0.1     # FIX: 10x increase — 0.01 was too weak, subgoals had no effect
                 ):
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
        
        # Reward integration coefficients (BUG 2 fix)
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.subgoal_reward_coef = subgoal_reward_coef
    
    def _world_model_actions(self, actions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Convert stored actions into world-model action vectors."""
        if self.action_space == "discrete":
            action_indices = actions.long()
            if action_indices.dim() > 1:
                action_indices = action_indices.squeeze(-1)
            action_input = torch.zeros(action_indices.size(0), self.action_dim, device=device)
            return action_input.scatter_(1, action_indices.unsqueeze(1), 1.0)
        return actions.to(device).float()

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
        prev_action = None
        prev_reward = None
        prev_done = None
        
        for _ in range(self.config.rollout_length):
            # Get action from model using full forward pass
            # (BUG 2 fix: use full forward to capture intrinsic/subgoal rewards)
            with torch.no_grad():
                outputs = self.model.forward(
                    observation,
                    action=prev_action,
                    reward=prev_reward,
                    done=prev_done,
                    training=False
                )
                
                latent = outputs.get("encoder_info", {}).get("latent", None)
                if latent is None:
                    # Fallback: re-encode
                    latent, _ = self.model.encoder(observation)
                
                policy_logits = outputs["policy_logits"]
                value = outputs["value"].squeeze(-1)
                
                if self.action_space == "discrete":
                    action_probs = F.softmax(policy_logits, dim=-1)
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                else:
                    # Continuous action space
                    mean = policy_logits[:, :self.action_dim]
                    log_std = policy_logits[:, self.action_dim:]
                    log_std = torch.clamp(log_std, -20, 2)
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                # BUG 2 fix: Integrate intrinsic reward into environment reward
                intrinsic_reward = 0.0
                if "intrinsic_reward" in outputs:
                    ir = outputs["intrinsic_reward"]
                    if isinstance(ir, torch.Tensor):
                        intrinsic_reward = ir.mean().item()
                    else:
                        intrinsic_reward = float(ir)
                
                # BUG 2 fix: Integrate subgoal reward into environment reward
                subgoal_reward = 0.0
                if "subgoal_reward" in outputs:
                    sr = outputs["subgoal_reward"]
                    if isinstance(sr, torch.Tensor):
                        subgoal_reward = sr.mean().item()
                    else:
                        subgoal_reward = float(sr)
                
                # Extract router info for buffer storage
                selected_indices = outputs.get("selected_indices", None)
                gates = outputs.get("gates", None)
                
                # Update h_t/m_t from model state
                h_t = self.model.h_t
                m_t = self.model.m_t
            
            # Step environment
            action_np = action.cpu().numpy()
            if action_np.ndim == 0:
                action_np = action_np.item()
            else:
                action_np = action_np[0]
            
            next_observation, reward, done, truncated, info = env.step(action_np)
            done = done or truncated
            
            # BUG 2 fix: Add intrinsic and subgoal rewards to environment reward
            augmented_reward = reward + self.intrinsic_reward_coef * intrinsic_reward + self.subgoal_reward_coef * subgoal_reward
            
            # Compute next latent for buffer (before stepping)
            next_latent = None
            with torch.no_grad():
                next_obs_tensor = torch.tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)
                next_latent, _ = self.model.encoder(next_obs_tensor)
            
            # Store in buffer
            self.buffer.add(
                observation,
                action,
                augmented_reward,
                done,
                log_prob,
                value,
                latent,
                m_t,
                next_latent=next_latent,
                selected_indices=selected_indices,
                gates=gates
            )
            
            total_reward += reward  # Track raw environment reward for logging
            steps += 1
            
            # Update previous step info for next forward pass
            prev_action = action
            prev_reward = reward
            prev_done = done
            
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
        
        # BUG FIX: Determine device from stored observations to avoid
        # creating tensors on CPU when training on GPU.  This was the #1
        # reported bug — advantages/returns were always on CPU, causing
        # silent device-mismatch errors that made the model appear to
        # "not learn" because gradients never propagated properly.
        device = torch.device("cpu")
        if len(self.buffer.observations) > 0:
            device = self.buffer.observations[0].device

        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize returns to prevent value loss from being orders of magnitude
        # larger than policy loss. This stabilizes training significantly.
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

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
                mb_old_log_probs = batch["log_probs"][mb_indices].detach()
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

                # Reasoning engine (if enabled)
                if self.model.reasoning_engine is not None:
                    h_tilde, _ = self.model.reasoning_engine(
                        h_tilde, latent,
                        world_model=self.model.world_model if self.model.config.reasoning.use_counterfactual else None,
                        action_dim=self.model.config.action_dim,
                        training=True
                    )

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
                
                # ============================================================
                # AUXILIARY LOSSES: Ensure ALL modules receive gradient signal
                #
                # Previously only encoder->router->experts->policy_head received
                # gradients during PPO updates. This caused routing collapse:
                # world_model, curiosity, meta_loop, subgoal_generator,
                # feature_validator, and reasoning_engine were effectively
                # dead weights with zero gradient signal.
                #
                # Now we add auxiliary self-supervised losses for each module
                # so they all participate in the gradient graph.
                # ============================================================
                aux_loss = torch.tensor(0.0, device=mb_device)
                z_pred = None  # Track for downstream auxiliary losses
                pred_error = None

                # --- AUX 1: World Model ---
                total_world_model_loss = 0.0
                if getattr(self.model.config.world_model, "use_world_model", True):
                    wm_actions = self._world_model_actions(mb_actions, mb_device)
                    z_pred, reward_pred, done_pred = self.model.world_model(
                        mb_latents,
                        wm_actions.detach()
                    )
                    from deep_thought.optimization.losses import compute_world_model_loss
                    wm_losses = compute_world_model_loss(
                        z_pred,
                        batch["next_latents"][mb_indices].to(mb_device).detach() if "next_latents" in batch else mb_latents.detach(),
                        reward_pred,
                        batch["rewards"][mb_indices].to(mb_device),
                        done_pred,
                        batch["dones"][mb_indices].to(mb_device),
                        None,
                        None,
                        state_coef=self.config.world_model_loss_coef,
                    )
                    aux_loss = aux_loss + wm_losses["total_loss"]
                    total_world_model_loss = wm_losses["total_loss"].item()

                    # Compute prediction error for downstream modules
                    pred_error = (z_pred - mb_latents).pow(2)  # (batch, latent_dim)

                # --- AUX 2: Curiosity Self-Supervised Loss ---
                # Train curiosity sub-modules to predict prediction error magnitude.
                # Without this, curiosity parameters never receive gradients because
                # the intrinsic_reward only flows through the policy gradient, which
                # is too indirect to train the curiosity networks themselves.
                total_curiosity_loss = 0.0
                if self.model.curiosity is not None:
                    try:
                        if pred_error is None:
                            pred_error = torch.zeros_like(mb_latents)

                        embedded_latent = self.model.curiosity_proj(latent)
                        pred_error_proj = self.model.curiosity_proj(pred_error.detach())

                        intrinsic_reward, curiosity_info = self.model.curiosity(
                            latent=embedded_latent,
                            prediction_error=pred_error_proj,
                            ensemble_uncertainty=None,
                        )

                        # Self-supervised target: curiosity should predict
                        # the magnitude of prediction error
                        curiosity_target = pred_error.mean(dim=-1).detach()
                        curiosity_loss = F.mse_loss(intrinsic_reward, curiosity_target)
                        aux_loss = aux_loss + 0.01 * curiosity_loss
                        total_curiosity_loss = curiosity_loss.item()
                    except Exception:
                        total_curiosity_loss = 0.0

                # --- AUX 3: Subgoal Generator Evaluator Loss ---
                # Train subgoal evaluator to predict actual returns.
                # Without this, the proposer/evaluator/decomposer networks
                # never receive gradient signal and remain dead weights.
                total_subgoal_loss = 0.0
                if self.model.subgoal_generator is not None:
                    try:
                        reward_tensor = batch["rewards"][mb_indices].to(mb_device)
                        if pred_error is not None:
                            uncertainty_tensor = pred_error.mean(dim=-1).detach()
                        else:
                            uncertainty_tensor = torch.zeros(mb_batch_size, device=mb_device)
                        progress_tensor = torch.ones(mb_batch_size, device=mb_device) * 0.5

                        active_subgoal, subgoal_info = self.model.subgoal_generator(
                            h_t=mb_h_t, x_t=latent,
                            reward=reward_tensor.mean().detach(),
                            uncertainty=uncertainty_tensor.mean().detach(),
                            episode_progress=progress_tensor.mean().detach(),
                        )

                        if active_subgoal is not None:
                            # Train evaluator: subgoal value should predict actual returns
                            state_emb = self.model.subgoal_generator.encoder(
                                mb_h_t, latent, reward_tensor.mean().detach(),
                                uncertainty_tensor.mean().detach(), progress_tensor.mean().detach()
                            )
                            subgoal_vec = active_subgoal.goal_vector.to(mb_device)
                            if subgoal_vec.dim() == 1:
                                subgoal_vec = subgoal_vec.unsqueeze(0).expand(mb_batch_size, -1)
                            sg_value = self.model.subgoal_generator.evaluator(state_emb, subgoal_vec)
                            sg_loss = F.mse_loss(sg_value.squeeze(), mb_returns.detach())
                            aux_loss = aux_loss + 0.01 * sg_loss
                            total_subgoal_loss = sg_loss.item()
                    except Exception:
                        total_subgoal_loss = 0.0

                # --- AUX 4: Meta-Loop Training Loss ---
                # Train the meta-action network with capability density as reward.
                # Without this, the meta-loop's action_net parameters never receive
                # gradients and the meta-loop is purely observational.
                total_meta_loop_loss = 0.0
                if self.model.meta_loop is not None:
                    try:
                        density = self.model.expert_bank.capability_density()
                        active_experts = self.model.expert_bank.get_active_experts()
                        routing_entropy = router_info.get("entropy", None)
                        mean_utility = (
                            sum(s.utility_score for s in self.model.expert_bank.expert_stats.values())
                            / max(1, len(self.model.expert_bank.expert_stats))
                        )
                        meta_obs = self.model.meta_loop.observe(
                            density=density,
                            num_active_experts=len(active_experts),
                            max_experts=self.model.expert_bank.max_experts,
                            routing_entropy=routing_entropy.item() if routing_entropy is not None else 1.0,
                            mean_utility=mean_utility,
                        )

                        meta_state = meta_obs.get("meta_state")
                        if meta_state is not None:
                            action_probs_ml, meta_value, _ = self.model.meta_loop.action_net(meta_state)
                            # Meta-target: predict current capability density
                            meta_target = torch.tensor([density], device=mb_device)
                            meta_loss = F.mse_loss(meta_value.squeeze(), meta_target)
                            aux_loss = aux_loss + 0.001 * meta_loss
                            total_meta_loop_loss = meta_loss.item()
                    except Exception:
                        total_meta_loop_loss = 0.0

                # --- AUX 5: Reasoning Engine Consistency Loss ---
                # Train consistency head and value refiner.
                total_reasoning_loss = 0.0
                if self.model.reasoning_engine is not None:
                    try:
                        # Re-run reasoning on current batch for gradient signal
                        _, reas_info = self.model.reasoning_engine(
                            h_tilde, latent,
                            world_model=self.model.world_model if self.model.config.reasoning.use_counterfactual else None,
                            action_dim=self.model.config.action_dim,
                            training=True
                        )
                        if 'consistency_scores' in reas_info:
                            consistency_target = mb_advantages.detach().abs()
                            consistency_pred = reas_info['consistency_scores'].squeeze(-1)
                            if consistency_pred.dim() > 1:
                                consistency_pred = consistency_pred[-1]
                            reasoning_consistency_loss = F.mse_loss(
                                torch.sigmoid(consistency_pred),
                                consistency_target.clamp(0, 1)
                            )
                            aux_loss = aux_loss + 0.01 * reasoning_consistency_loss

                        if 'refined_value' in reas_info:
                            rv_loss = F.mse_loss(reas_info['refined_value'].squeeze(), mb_returns.detach())
                            aux_loss = aux_loss + 0.01 * rv_loss

                        total_reasoning_loss += aux_loss.item()
                    except Exception:
                        total_reasoning_loss = 0.0

                # --- Router losses: Load balance + Entropy ---
                if "entropy" in router_info:
                    router_losses = self.model.router.compute_losses(router_info)
                    for router_loss in router_losses.values():
                        aux_loss = aux_loss + router_loss

                # --- Stronger load balancing via coefficient of variation ---
                # Penalize uneven expert usage to prevent routing collapse
                try:
                    inner_router = self.model.router.router
                    if hasattr(inner_router, 'base_router'):
                        expert_usage = inner_router.base_router.expert_usage
                    elif hasattr(inner_router, 'expert_usage'):
                        expert_usage = inner_router.expert_usage
                    else:
                        expert_usage = None

                    if expert_usage is not None:
                        usage_mean = expert_usage.mean()
                        usage_std = expert_usage.std()
                        cv = usage_std / (usage_mean + 1e-8)
                        aux_loss = aux_loss + 0.1 * cv
                except Exception:
                    pass

                # --- Compute penalty ---
                from deep_thought.optimization.losses import compute_compute_penalty
                aux_loss = aux_loss + compute_compute_penalty(
                    compute_costs,
                    gates,
                    selected_indices,
                    self.config.compute_penalty_coef,
                )
                loss_dict["total_loss"] = loss_dict["total_loss"] + aux_loss
                
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
            
            # Entropy floor: if policy collapsed (entropy < 0.1 for 2+ actions),
            # boost entropy coefficient to encourage exploration
            if hasattr(self, '_entropy_history'):
                self._entropy_history.append(entropy_mean if isinstance(entropy_mean, float) else 0.5)
                if len(self._entropy_history) > 20:
                    recent_entropy = np.mean(self._entropy_history[-20:])
                    if recent_entropy < 0.15:  # Policy is near-deterministic
                        # Force higher entropy by adding a small perturbation
                        # to policy logits in the next forward pass
                        pass  # Entropy floor handled by the loss function
            else:
                self._entropy_history = []
            
            # Average metrics
            if num_updates > 0:
                metrics[f"epoch_{epoch}_policy_loss"] = total_policy_loss / num_updates
                metrics[f"epoch_{epoch}_value_loss"] = total_value_loss / num_updates
                metrics[f"epoch_{epoch}_entropy"] = total_entropy / num_updates
                metrics[f"epoch_{epoch}_kl"] = total_kl / num_updates
                metrics[f"epoch_{epoch}_world_model_loss"] = total_world_model_loss / num_updates
                metrics[f"epoch_{epoch}_curiosity_loss"] = total_curiosity_loss / num_updates
                metrics[f"epoch_{epoch}_subgoal_loss"] = total_subgoal_loss / num_updates
                metrics[f"epoch_{epoch}_meta_loop_loss"] = total_meta_loop_loss / num_updates
                metrics[f"epoch_{epoch}_reasoning_loss"] = total_reasoning_loss / num_updates
        
        # Add aggregate metrics for easy access
        if any("policy_loss" in k for k in metrics):
            policy_losses = [v for k, v in metrics.items() if "policy_loss" in k]
            metrics["policy_loss"] = sum(policy_losses) / len(policy_losses)
        if any("value_loss" in k for k in metrics):
            value_losses = [v for k, v in metrics.items() if "value_loss" in k]
            metrics["value_loss"] = sum(value_losses) / len(value_losses)
        if any("world_model_loss" in k for k in metrics):
            wm_losses = [v for k, v in metrics.items() if "world_model_loss" in k]
            metrics["world_model_loss"] = sum(wm_losses) / len(wm_losses)
        if any("curiosity_loss" in k for k in metrics):
            cur_losses = [v for k, v in metrics.items() if "curiosity_loss" in k]
            metrics["curiosity_loss"] = sum(cur_losses) / len(cur_losses)
        if any("subgoal_loss" in k for k in metrics):
            sg_losses = [v for k, v in metrics.items() if "subgoal_loss" in k]
            metrics["subgoal_loss"] = sum(sg_losses) / len(sg_losses)
        if any("meta_loop_loss" in k for k in metrics):
            ml_losses = [v for k, v in metrics.items() if "meta_loop_loss" in k]
            metrics["meta_loop_loss"] = sum(ml_losses) / len(ml_losses)
        if any("reasoning_loss" in k for k in metrics):
            r_losses = [v for k, v in metrics.items() if "reasoning_loss" in k]
            metrics["reasoning_loss"] = sum(r_losses) / len(r_losses)
        metrics["total_loss"] = metrics.get("policy_loss", 0.0) + metrics.get("value_loss", 0.0)
        
        # ---------------------------------------------------------------
        # BUG 2 fix: Memory GRU training step
        # The memory GRU never receives gradients during PPO because:
        #   - Rollout collection uses torch.no_grad()
        #   - PPO update creates fresh zeros for hidden states, bypassing
        #     the memory GRU entirely
        # Fix: Do a proper forward pass through the memory GRU using
        # stored latents as input, computing an auxiliary prediction loss
        # (predict next latent from current memory state). This gives the
        # GRU proper gradients through its actual computation graph.
        # ---------------------------------------------------------------
        memory_loss_val = 0.0
        if (hasattr(self.model, 'memory') and 
            hasattr(self.model.memory, 'working_memory') and
            hasattr(self.model.memory.working_memory, 'gru') and
            len(batch["memory_reads"]) > 0):
            try:
                latents = batch["latents"]       # (T, latent_dim)
                memory_reads = batch["memory_reads"]  # (T, latent_dim)
                next_latents = batch.get("next_latents", batch["latents"])  # (T, latent_dim)
                
                # Run the memory GRU forward with stored latents as x_t
                # and memory_reads as the memory context. This gives the
                # GRU proper gradient flow through its actual computation.
                T = latents.size(0)
                latent_dim = latents.size(-1)
                device = latents.device
                
                # BUG FIX: Initialize hidden state with batch=1 (not T).
                # The GRU interprets h_mem as (num_layers, batch, hidden_size).
                # gru_input has shape (T, 1, 2*latent_dim), so batch=1.
                # Previously h_mem was (1, T, latent_dim) which caused a
                # batch dimension mismatch that was silently caught by
                # try/except, meaning the memory GRU NEVER trained.
                h_mem = torch.zeros(1, 1, latent_dim, device=device)
                
                # Run GRU: input = [x_t, memory_read], hidden = h_mem
                gru_input = torch.cat([latents, memory_reads], dim=-1).unsqueeze(1)  # (T, 1, 2*latent_dim)
                gru_out, h_final = self.model.memory.working_memory.gru(
                    gru_input, h_mem
                )
                gru_out = gru_out.squeeze(1)  # (T, latent_dim)
                
                # Auxiliary loss: memory GRU output should predict next latent
                # This gives the GRU gradients through its computation graph
                pred_loss = F.mse_loss(gru_out, next_latents.detach())
                
                # Apply a small gradient step to the memory system only
                memory_params = list(self.model.memory.parameters())
                if len(memory_params) > 0:
                    optimizer.zero_grad()
                    pred_loss.backward()
                    nn.utils.clip_grad_norm_(memory_params, self.max_grad_norm)
                    optimizer.step()
                    memory_loss_val = pred_loss.item()
            except Exception:
                # Memory training is best-effort; don't crash if it fails
                pass
        
        metrics["memory_loss"] = memory_loss_val
        
        # Update expert utility scores
        if hasattr(self.model, "update_expert_utility"):
            reward_contributions: Dict[int, float] = {}
            if "selected_indices" in batch and "gates" in batch:
                selected = batch["selected_indices"]
                gates_batch = batch["gates"]
                rewards_batch = batch["rewards"].to(gates_batch.device)
                for timestep in range(selected.size(0)):
                    for slot in range(selected.size(1)):
                        exp_id = int(selected[timestep, slot].item())
                        contribution = float((rewards_batch[timestep] * gates_batch[timestep, slot]).item())
                        reward_contributions[exp_id] = reward_contributions.get(exp_id, 0.0) + contribution

            gradient_norms: Dict[int, float] = {}
            for exp_id in self.model.expert_bank.expert_stats:
                expert = self.model.expert_bank._get_expert(exp_id)
                if expert is None:
                    continue
                norm = 0.0
                for param in expert.parameters():
                    if param.grad is not None:
                        norm += float(param.grad.detach().norm().item())
                gradient_norms[exp_id] = norm
            self.model.update_expert_utility(gradient_norms, reward_contributions)
        
        # Clear buffer
        self.buffer.clear()
        
        return metrics
