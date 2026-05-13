"""
Main Deep Thought agent class.

Integrates all components into a unified RL agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

from deep_thought.config import DeepThoughtConfig
from deep_thought.architecture.encoder import Encoder
from deep_thought.architecture.router import SparseRouter
from deep_thought.architecture.experts import ExpertBank, ExpertConfig
from deep_thought.architecture.world_model import WorldModel
from deep_thought.architecture.memory.memory_system import MemorySystem
from deep_thought.architecture.planning.temporal_planning import TemporalPlanningLayer
from deep_thought.learning.feature_validation import FeatureValidationEngine
from deep_thought.learning.meta_learning import MetaLearningLayer
from deep_thought.learning.expert_compiler import ExpertCompiler
from deep_thought.stability.srp import SelfRegressionPrevention


class DeepThoughtAgent(nn.Module):
    """
    Deep Thought: Adaptive Sparse Cognitive Network for RL.
    
    Integrates:
    - Sparse encoder with factorized latents
    - Sparse router with top-k expert selection
    - Expert bank with lifecycle management
    - World model for imagination
    - Multi-scale memory system
    - Feature validation engine
    - Expert compiler
    - Temporal planning layer
    - Meta-learning with fast weights
    - Self-regression prevention
    """
    
    def __init__(self, config: DeepThoughtConfig):
        super().__init__()
        self.config = config
        
        # Set observation and action dims in encoder config
        config.encoder.observation_dim = config.observation_dim
        config.world_model.action_dim = config.action_dim
        
        # Core architecture
        self.encoder = Encoder(config.encoder)
        self.router = SparseRouter(config.router, use_adaptive=True)
        self.expert_bank = ExpertBank(config.expert, config.router.num_experts)
        self.world_model = WorldModel(config.world_model, config.action_dim)
        
        # Memory system
        self.memory = MemorySystem(config.memory, config.encoder.latent_dim)
        
        # Planning layer
        if config.planning.use_tcpl:
            self.planning = TemporalPlanningLayer(
                config.planning,
                config.router.num_experts
            )
        else:
            self.planning = None
        
        # Learning systems
        if config.feature_validation.use_fve:
            self.feature_validator = FeatureValidationEngine(
                config.feature_validation,
                config.encoder.latent_dim
            )
        else:
            self.feature_validator = None
        
        if config.meta_learning.use_meta_learning:
            self.meta_learning = MetaLearningLayer(
                config.meta_learning,
                config.encoder.latent_dim
            )
        else:
            self.meta_learning = None
        
        if config.expert_compiler.use_fec:
            self.expert_compiler = ExpertCompiler(
                config.expert_compiler,
                config.expert,
                config.encoder.latent_dim
            )
        else:
            self.expert_compiler = None
        
        # Stability system
        if config.srp.use_srp:
            self.srp = SelfRegressionPrevention(config.srp)
        else:
            self.srp = None
        
        # Policy and value heads
        self.policy_head = nn.Linear(config.encoder.latent_dim, config.num_actions)
        self.critic_head = nn.Linear(config.encoder.latent_dim, 1)
        
        # Initialize hidden state
        self.h_t = None
        self.m_t = None
        self.context = None
        
        # Training state
        self.step = 0
    
    def reset(self, batch_size: int = 1):
        """Reset agent state."""
        device = next(self.parameters()).device
        self.h_t = self.memory.reset_working_memory(batch_size, device)
        self.m_t = torch.zeros(batch_size, self.config.encoder.latent_dim, device=device)
        self.context = torch.zeros(batch_size, self.config.meta_learning.context_dim, device=device)
        
        if self.planning is not None:
            self.planning.reset_plan()
        
        if self.meta_learning is not None:
            self.meta_learning.reset_context()
    
    def forward(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = None,
        training: bool = True
    ) -> Dict:
        """
        Forward pass through Deep Thought.
        
        Args:
            observation: Current observation
            action: Previous action (for memory write)
            reward: Previous reward (for memory write)
            done: Previous done (for memory write)
            training: Whether in training mode
            
        Returns:
            outputs: Dictionary with all outputs
        """
        outputs = {}
        
        # Encode observation
        x_t, encoder_info = self.encoder(observation)
        outputs["encoder_info"] = encoder_info
        
        # Update memory
        if self.h_t is None:
            self.reset(observation.size(0))
        
        h_t, memory_info = self.memory(
            self.h_t,
            x_t,
            observation,
            action if action is not None else torch.zeros_like(observation[:, :self.config.action_dim]),
            reward if reward is not None else 0.0,
            done if done is not None else False,
            write=training
        )
        self.h_t = h_t
        outputs["memory_info"] = memory_info
        
        # Get prediction error for adaptation
        if action is not None and self.world_model is not None:
            with torch.no_grad():
                z_next_pred, _, _ = self.world_model(x_t, action)
                prediction_error = F.mse_loss(z_next_pred, x_t)
        else:
            prediction_error = torch.tensor(0.0)
        
        # Update context
        if self.meta_learning is not None:
            self.context = self.meta_learning.update_context(x_t)
        
        # Route to experts
        gates, selected_indices, router_info = self.router(
            self.h_t,
            x_t,
            self.m_t,
            self.context,
            prediction_error,
            training=training
        )
        outputs["router_info"] = router_info
        outputs["gates"] = gates
        outputs["selected_indices"] = selected_indices
        
        # Apply experts
        delta_h, compute_costs = self.expert_bank(
            self.h_t,
            selected_indices,
            gates
        )
        h_tilde = self.h_t + delta_h
        outputs["delta_h"] = delta_h
        outputs["compute_costs"] = compute_costs
        
        # Meta-learning adaptation
        if self.meta_learning is not None:
            h_adapted, meta_info = self.meta_learning.adapt(
                h_tilde,
                self.context,
                gradient=None  # Would compute from loss
            )
            h_tilde = h_adapted
            outputs["meta_info"] = meta_info
        
        # Planning
        if self.planning is not None:
            if self.step % self.config.planning.replan_interval == 0:
                plans = self.planning.hierarchical_planning(
                    h_tilde,
                    x_t,
                    self.m_t
                )
                outputs["plans"] = plans
            else:
                self.planning.advance_step()
        
        # Policy and value
        policy_logits = self.policy_head(h_tilde)
        value = self.critic_head(h_tilde)
        
        outputs["policy_logits"] = policy_logits
        outputs["value"] = value
        
        # World model prediction
        if self.world_model is not None and action is not None:
            z_next, r_pred, d_pred = self.world_model(x_t, action)
            outputs["world_model"] = {
                "z_next": z_next,
                "r_pred": r_pred,
                "d_pred": d_pred,
            }
        
        # Feature extraction
        if self.feature_validator is not None and training:
            feature_ids = self.feature_validator.extract_features(x_t)
            outputs["feature_ids"] = feature_ids
        
        # Update step
        self.step += 1
        
        return outputs
    
    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Select action.
        
        Args:
            observation: Current observation
            deterministic: Whether to act deterministically
            
        Returns:
            action: Selected action
            value: Value estimate
            info: Additional information
        """
        with torch.no_grad():
            outputs = self.forward(observation, training=False)
            
            policy_logits = outputs["policy_logits"]
            value = outputs["value"]
            
            if self.config.action_space == "discrete":
                action_probs = F.softmax(policy_logits, dim=-1)
                if deterministic:
                    action = action_probs.argmax(dim=-1)
                else:
                    action = torch.distributions.Categorical(action_probs).sample()
            else:
                # Continuous
                mean = policy_logits[:, :self.config.action_dim]
                log_std = policy_logits[:, self.config.action_dim:]
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                if deterministic:
                    action = mean
                else:
                    action = dist.sample()
        
        return action, value, outputs
    
    def compute_loss(
        self,
        batch: Dict,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for training.
        
        Args:
            batch: Batch of data
            advantages: Advantage estimates
            returns: Return targets
            
        Returns:
            losses: Dictionary of losses
        """
        losses = {}
        
        # Forward pass
        outputs = self.forward(batch["observations"], training=True)
        
        # PPO loss
        from deep_thought.optimization.losses import compute_ppo_loss
        if self.config.action_space == "discrete":
            policy_logits = outputs["policy_logits"]
            action_probs = F.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(batch["actions"])
        else:
            mean = outputs["policy_logits"][:, :self.config.action_dim]
            log_std = outputs["policy_logits"][:, self.config.action_dim:]
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(batch["actions"]).sum(dim=-1)
        
        ppo_losses = compute_ppo_loss(
            log_probs,
            batch["log_probs"],
            advantages,
            outputs["value"].squeeze(-1),
            returns,
            self.config.training.clip_eps,
            self.config.training.value_loss_coef,
            self.config.training.entropy_coef
        )
        losses.update(ppo_losses)
        
        # World model loss
        if "world_model" in outputs:
            wm_outputs = outputs["world_model"]
            from deep_thought.optimization.losses import compute_world_model_loss
            wm_losses = compute_world_model_loss(
                wm_outputs["z_next"],
                batch["latents"],
                wm_outputs["r_pred"],
                batch["rewards"],
                wm_outputs["d_pred"],
                batch["dones"],
                None,  # obs_recon
                None,  # observation
                state_coef=self.config.training.world_model_loss_coef
            )
            losses.update(wm_losses)
        
        # Compute penalty
        compute_loss = 0.0
        if "compute_costs" in outputs:
            from deep_thought.optimization.losses import compute_compute_penalty
            compute_loss = compute_compute_penalty(
                outputs["compute_costs"],
                outputs["gates"],
                outputs["selected_indices"],
                self.config.training.compute_penalty_coef
            )
        losses["compute_loss"] = compute_loss
        
        # Encoder losses
        if "encoder_info" in outputs:
            encoder_losses = self.encoder.compute_losses(outputs["encoder_info"])
            losses.update(encoder_losses)
        
        # Router losses
        if "router_info" in outputs:
            router_losses = self.router.compute_losses(outputs["router_info"])
            losses.update(router_losses)
        
        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        
        return losses
    
    def update_expert_utility(
        self,
        gradient_norms: Dict[int, float],
        reward_contributions: Dict[int, float]
    ):
        """Update expert utility scores."""
        self.expert_bank.update_utility(gradient_norms, reward_contributions)
    
    def prune_experts(self):
        """Prune low-utility experts."""
        if self.srp is not None:
            signals = self.srp.get_stats()
            if signals["architecture_gate"]["allow_pruning"]:
                self.expert_bank.mark_dormant(self.config.training.dormant_threshold)
                self.expert_bank.mark_dead(
                    self.config.training.delete_threshold,
                    self.config.training.delete_confirmation_steps
                )
                self.expert_bank.prune_dead_experts()
    
    def grow_experts(self):
        """Grow new experts if needed."""
        if self.srp is not None:
            signals = self.srp.get_stats()
            if signals["architecture_gate"]["allow_growth"]:
                # Check if growth is needed (e.g., stagnation)
                # For now, grow if under max
                if len(self.expert_bank.experts) < self.config.expert_compiler.max_experts:
                    self.expert_bank.grow_expert()
    
    def consolidate_memory(self):
        """Consolidate episodic memory into semantic memory."""
        self.memory.consolidate()
        self.memory.age_memories()
        self.memory.decay_semantic()
        self.memory.prune_semantic()
    
    def validate_features(self):
        """Validate features and promote to experts."""
        if self.feature_validator is not None:
            for feature_id in list(self.feature_validator.features.keys()):
                self.feature_validator.validate_feature(feature_id)
            
            promoted = self.feature_validator.promote_features()
            
            # Compile promoted features into experts
            if self.expert_compiler is not None:
                for feature_id in promoted:
                    feature = self.feature_validator.features[feature_id]
                    self.expert_compiler.create_candidate(feature)
    
    def update_srp(
        self,
        reward: float,
        loss: float,
        routing_entropy: Optional[float] = None
    ):
        """Update self-regression prevention system."""
        if self.srp is not None:
            expert_utilities = {
                i: stats.utility_score
                for i, stats in self.expert_bank.expert_stats.items()
            }
            
            signals = self.srp.update(
                reward,
                loss,
                expert_utilities,
                routing_entropy
            )
            
            # Rollback if needed
            if signals["should_rollback"]:
                self.srp.rollback(self)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            "step": self.step,
            "num_experts": len(self.expert_bank.experts),
            "active_experts": len(self.expert_bank.get_active_experts()),
            "dormant_experts": len(self.expert_bank.get_dormant_experts()),
            "memory_stats": self.memory.get_memory_stats(),
        }
        
        if self.feature_validator is not None:
            stats["feature_stats"] = self.feature_validator.get_feature_stats()
        
        if self.expert_compiler is not None:
            stats["expert_compiler_stats"] = self.expert_compiler.get_candidate_stats()
        
        if self.srp is not None:
            stats["srp_stats"] = self.srp.get_stats()
        
        return stats
