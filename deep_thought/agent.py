"""
Main Deep Thought agent class.

Integrates all components into a unified RL agent, governed by the
7 architectural governance principles:
1. Single dominant objective (RL primary, others as constraints)
2. Hard time-scale separation (fast/medium/slow/very-slow)
3. Capacity ledger for growth/pruning
4. Decoupled routing (slow router + fast deterministic gating)
5. Asymmetric memory read/write with firewalls
6. Non-interference rule (propose -> evaluate -> accept)
7. Shared signal space normalization (expected return impact)
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
from deep_thought.curiosity.intrinsic_motivation import IntrinsicMotivationSystem
from deep_thought.hierarchical.expert_society import HierarchicalExpertSociety
from deep_thought.opponent_modeling.opponent_model import OpponentModelingSystem
from deep_thought.compute_economy.compute_market import ComputeMarket
from deep_thought.subgoals.subgoal_generator import SubgoalGenerator
from deep_thought.architecture.attention_maps import AttentionProbabilityMap
from deep_thought.governance.governor import Governor, GovernorConfig
from deep_thought.governance.timescale_controller import TimescaleConfig, TimescaleTier
from deep_thought.governance.capacity_ledger import CapacityLedgerConfig
from deep_thought.governance.proposal_bus import Proposal, ProposalType
from deep_thought.stability.meta_loop import MetaLoopController, MetaLoopConfig
from deep_thought.learning.formal_verification import FormalVerificationLayer, FormalVerificationConfig
from deep_thought.learning.shadow_evolution import ShadowEvolutionEngine, ShadowEvolutionConfig
from deep_thought.learning.dynamic_hyperparams import DynamicHyperparamController, DynamicHyperparamsConfig
from deep_thought.reasoning.reasoning_engine import ReasoningEngine


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
    - Architectural governance layer (7 principles)
    - Stable self-improvement: Meta-Loop on Capability Density
    - Stable self-improvement: Formal Verification Layer
    - Stable self-improvement: Shadow Evolution
    - Stable self-improvement: Dynamic Hyperparameter Adaptation
    - Reasoning Engine: Multi-step deliberation before action selection
    """

    def __init__(self, config: DeepThoughtConfig):
        super().__init__()
        self.config = config

        # Set observation and action dims in encoder config
        config.encoder.observation_dim = config.observation_dim
        config.world_model.action_dim = config.action_dim
        # Ensure world model latent_dim matches encoder latent_dim
        config.world_model.latent_dim = config.encoder.latent_dim

        # ----------------------------------------------------------------
        # Governance Layer (Fixes 1-7)
        # ----------------------------------------------------------------
        if config.governance.use_governor:
            gov = config.governance
            gov_config = GovernorConfig(
                timescale_config=TimescaleConfig(
                    medium_interval=gov.medium_interval,
                    slow_interval=gov.slow_interval,
                    very_slow_interval=gov.very_slow_interval,
                ),
                ledger_config=CapacityLedgerConfig(
                    max_total_parameters=gov.max_total_parameters,
                    max_experts=gov.max_experts,
                    min_experts=gov.min_experts,
                    pruning_confirmation_window=gov.pruning_confirmation_window,
                    growth_marginal_threshold=gov.growth_marginal_threshold,
                    pruning_utility_threshold=gov.pruning_utility_threshold,
                    redundancy_threshold=gov.redundancy_threshold,
                ),
                sparsity_constraint_coef=gov.sparsity_constraint_coef,
                entropy_constraint_coef=gov.entropy_constraint_coef,
                load_balance_constraint_coef=gov.load_balance_constraint_coef,
                world_model_constraint_coef=gov.world_model_constraint_coef,
                compute_penalty_constraint_coef=gov.compute_penalty_constraint_coef,
                memory_coherence_constraint_coef=gov.memory_coherence_constraint_coef,
                capability_density_coef=getattr(gov, 'capability_density_coef', 0.01),
                memory_read_filter_threshold=gov.memory_read_filter_threshold,
                memory_influence_on_pruning=gov.memory_influence_on_pruning,
                memory_influence_on_growth=gov.memory_influence_on_growth,
            )
            self.governor = Governor(gov_config)
        else:
            self.governor = None

        # Core architecture
        self.encoder = Encoder(config.encoder)
        # Router uses meta_learning context_dim if available
        router_context_dim = config.meta_learning.context_dim if config.meta_learning.use_meta_learning else 256
        self.router = SparseRouter(
            config.router,
            use_adaptive=True,
            latent_dim=config.encoder.latent_dim,
            context_dim=router_context_dim
        )
        # LEVER 3: Pass expert_hard_cap as max_experts to ExpertBank
        expert_hard_cap = getattr(config.training, 'expert_hard_cap',
                                  getattr(config.governance, 'expert_hard_cap', 256))
        self.expert_bank = ExpertBank(
            config.expert, config.router.num_experts, config.encoder.latent_dim,
            max_experts=expert_hard_cap
        )
        self.world_model = WorldModel(config.world_model, config.action_dim)

        # Set observation dim on world model for decoder
        if config.observation_dim is not None:
            self.world_model.set_observation_dim(config.observation_dim)

        # Register all experts with the capacity ledger (Fix 3)
        if self.governor is not None:
            for exp_id in range(config.router.num_experts):
                expert = self.expert_bank._get_expert(exp_id)
                if expert is not None:
                    n_params = sum(p.numel() for p in expert.parameters())
                    self.governor.ledger.register_expert(exp_id, n_params)

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

        # Curiosity / Intrinsic Motivation
        if config.curiosity.use_curiosity:
            self.curiosity = IntrinsicMotivationSystem(config.curiosity)
            self.curiosity_proj = nn.Linear(config.encoder.latent_dim, config.curiosity.state_embedding_dim)
        else:
            self.curiosity = None

        # Hierarchical Expert Society
        if config.hierarchical.use_hierarchy:
            self.hierarchical = HierarchicalExpertSociety(config.hierarchical, config.encoder.latent_dim)
        else:
            self.hierarchical = None

        # Opponent Modeling
        if config.opponent_modeling.use_opponent_modeling:
            self.opponent_modeling = OpponentModelingSystem(config.opponent_modeling, config.encoder.latent_dim)
        else:
            self.opponent_modeling = None

        # Compute Economy
        if config.compute_economy.use_compute_market:
            self.compute_market = ComputeMarket(config.compute_economy, config.router.num_experts, config.encoder.latent_dim)
        else:
            self.compute_market = None

        # Subgoal Generator
        if config.subgoal.use_subgoals:
            self.subgoal_generator = SubgoalGenerator(config.subgoal, config.encoder.latent_dim)
        else:
            self.subgoal_generator = None

        # Attention Probability Maps
        if config.attention_maps.use_attention_maps:
            self.attention_maps = AttentionProbabilityMap(config.attention_maps, config.encoder.latent_dim)
        else:
            self.attention_maps = None

        # -----------------------------------------------------------
        # Stable Self-Improvement Components
        # -----------------------------------------------------------

        # Component 1: Meta-Loop on Capability Density
        if config.meta_loop.use_meta_loop:
            self.meta_loop = MetaLoopController(
                MetaLoopConfig(
                    use_meta_loop=config.meta_loop.use_meta_loop,
                    density_reward_coef=config.meta_loop.density_reward_coef,
                    density_regression_threshold=config.meta_loop.density_regression_threshold,
                    meta_lr=config.meta_loop.meta_lr,
                    meta_action_dim=config.meta_loop.meta_action_dim,
                    history_length=config.meta_loop.history_length,
                    min_density_improvement=config.meta_loop.min_density_improvement,
                    density_ema_decay=config.meta_loop.density_ema_decay,
                ),
                state_dim=32,
            )
        else:
            self.meta_loop = None

        # Component 2: Formal Verification Layer
        if config.formal_verification.use_formal_verification:
            self.formal_verification = FormalVerificationLayer(
                FormalVerificationConfig(
                    use_formal_verification=config.formal_verification.use_formal_verification,
                    kl_epsilon=config.formal_verification.kl_epsilon,
                    kl_check_interval=config.formal_verification.kl_check_interval,
                    max_output_norm=config.formal_verification.max_output_norm,
                    min_capability_density=config.formal_verification.min_capability_density,
                    gradient_explosion_threshold=config.formal_verification.gradient_explosion_threshold,
                    verification_tier=config.formal_verification.verification_tier,
                    constraint_violation_cooldown=config.formal_verification.constraint_violation_cooldown,
                ),
                num_experts=config.router.num_experts,
            )
        else:
            self.formal_verification = None

        # Component 3: Shadow Evolution
        if config.shadow_evolution.use_shadow_evolution:
            self.shadow_evolution = ShadowEvolutionEngine(
                ShadowEvolutionConfig(
                    use_shadow_evolution=config.shadow_evolution.use_shadow_evolution,
                    max_shadow_experts=config.shadow_evolution.max_shadow_experts,
                    mutation_rate=config.shadow_evolution.mutation_rate,
                    mutation_strength=config.shadow_evolution.mutation_strength,
                    tournament_size=config.shadow_evolution.tournament_size,
                    validation_window=config.shadow_evolution.validation_window,
                    swap_threshold=config.shadow_evolution.swap_threshold,
                    evolution_interval=config.shadow_evolution.evolution_interval,
                    max_mutations_per_cycle=config.shadow_evolution.max_mutations_per_cycle,
                    archive_size=config.shadow_evolution.archive_size,
                ),
            )
        else:
            self.shadow_evolution = None

        # Component 4: Dynamic Hyperparameter Adaptation
        if config.dynamic_hyperparams.use_dynamic_hyperparams:
            self.dynamic_hyperparams = DynamicHyperparamController(
                DynamicHyperparamsConfig(
                    use_dynamic_hyperparams=config.dynamic_hyperparams.use_dynamic_hyperparams,
                    volatility_window=config.dynamic_hyperparams.volatility_window,
                    volatility_ema_decay=config.dynamic_hyperparams.volatility_ema_decay,
                    lr_min=config.dynamic_hyperparams.lr_min,
                    lr_max=config.dynamic_hyperparams.lr_max,
                    lr_adjustment_rate=config.dynamic_hyperparams.lr_adjustment_rate,
                    pruning_threshold_min=config.dynamic_hyperparams.pruning_threshold_min,
                    pruning_threshold_max=config.dynamic_hyperparams.pruning_threshold_max,
                    pruning_threshold_adjustment_rate=config.dynamic_hyperparams.pruning_threshold_adjustment_rate,
                    warmup_trigger_threshold=config.dynamic_hyperparams.warmup_trigger_threshold,
                    warmup_duration=config.dynamic_hyperparams.warmup_duration,
                    warmup_lr_multiplier=config.dynamic_hyperparams.warmup_lr_multiplier,
                    warmup_freeze_architecture=config.dynamic_hyperparams.warmup_freeze_architecture,
                    curvature_window=config.dynamic_hyperparams.curvature_window,
                    meta_controller_hidden_dim=config.dynamic_hyperparams.meta_controller_hidden_dim,
                ),
            )
        else:
            self.dynamic_hyperparams = None

        # Reasoning Engine
        if config.reasoning.use_reasoning:
            self.reasoning_engine = ReasoningEngine(
                config.reasoning,
                config.encoder.latent_dim,
                config.router.num_experts
            )
        else:
            self.reasoning_engine = None

        # Policy and value heads
        # For continuous action space, policy head outputs mean + log_std (2 * action_dim)
        if config.action_space == "continuous" and config.action_dim is not None:
            policy_dim = config.action_dim * 2
        else:
            policy_dim = config.num_actions if config.num_actions is not None else config.action_dim or 2
        self.policy_head = nn.Linear(config.encoder.latent_dim, policy_dim)
        self.critic_head = nn.Linear(config.encoder.latent_dim, 1)

        # Initialize hidden state
        self.h_t = None
        self.m_t = None
        self.context = None
        self._prev_z_pred = None  # Previous world model prediction for error computation

        # Training state
        self.step = 0

    def reset(self, batch_size: int = 1):
        """Reset agent state for a new episode."""
        device = next(self.parameters()).device
        self.h_t = self.memory.reset_working_memory(batch_size, device)
        self.m_t = torch.zeros(batch_size, self.config.encoder.latent_dim, device=device)
        self._prev_z_pred = None  # Reset prediction tracking

        if self.meta_learning is not None:
            self.context = torch.zeros(batch_size, self.config.meta_learning.context_dim, device=device)
        else:
            self.context = None

        if self.planning is not None:
            self.planning.reset_plan()

        if self.meta_learning is not None:
            self.meta_learning.reset_context()

        if self.curiosity is not None:
            self.curiosity.reset_visit_counts()

        if self.subgoal_generator is not None:
            self.subgoal_generator._active_subgoals = []
            self.subgoal_generator._completed_subgoals = []
            self.subgoal_generator._step_count = 0
            self.subgoal_generator._next_subgoal_id = 0
            self.subgoal_generator._current_state_embedding = None

        if self.attention_maps is not None:
            self.attention_maps.reset()

        if self.reasoning_engine is not None:
            self.reasoning_engine.reset()

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

        # Tick the governor (Fix 2: time-scale separation)
        if self.governor is not None:
            self.governor.tick(self.step)

        # Encode observation
        x_t, encoder_info = self.encoder(observation)
        outputs["encoder_info"] = encoder_info

        # Update memory
        if self.h_t is None:
            self.reset(observation.size(0))

        # Detach hidden states to prevent graph leaks across steps
        self.h_t = self.h_t.detach()
        self.m_t = self.m_t.detach() if self.m_t is not None else self.m_t
        if self.context is not None:
            self.context = self.context.detach()

        # Safe default action tensor
        if action is not None:
            # Convert action to proper format for internal use
            if self.config.action_space == "discrete":
                # One-hot encode discrete actions
                if action.dim() == 0:
                    action = action.unsqueeze(0)
                batch_size = action.size(0)
                action_input = torch.zeros(batch_size, self.config.action_dim, device=observation.device)
                action_input.scatter_(1, action.unsqueeze(1), 1.0)
            else:
                # Continuous actions - ensure 2D
                if action.dim() == 1:
                    action_input = action.unsqueeze(0)
                else:
                    action_input = action
        else:
            action_input = torch.zeros(observation.size(0), self.config.action_dim, device=observation.device)

        # Fix 5: Asymmetric memory - cheap writes, expensive filtered reads
        write_approved = True
        if self.governor is not None and training:
            # Cheap write: almost always approved
            importance = abs(reward) if reward is not None else 0.0
            write_approved = self.governor.approve_memory_write(importance)

        h_t, memory_info = self.memory(
            self.h_t,
            x_t,
            observation,
            action_input,
            reward if reward is not None else 0.0,
            done if done is not None else False,
            write=(training and write_approved)
        )
        self.h_t = h_t
        # Update m_t with the actual memory read so routing uses real memory
        # BUG 1 fix: detach to prevent graph leaks across steps
        self.m_t = memory_info["memory_read"].detach()
        outputs["memory_info"] = memory_info

        # Get prediction error for adaptation
        # Compare previous world model prediction with current latent
        if hasattr(self, '_prev_z_pred') and self._prev_z_pred is not None and action is not None:
            with torch.no_grad():
                elementwise_pred_error = (self._prev_z_pred - x_t.detach()).pow(2)  # (batch, latent_dim)
                prediction_error = elementwise_pred_error.mean()  # scalar for backward compat
        else:
            elementwise_pred_error = torch.zeros_like(x_t)
            prediction_error = torch.tensor(0.0, device=observation.device)

        # Store current prediction for next step comparison.
        # BUG FIX: The world model was previously computed inside
        # torch.no_grad(), which meant the world model NEVER received
        # gradient signal during the main forward pass. Now we compute
        # the prediction WITH gradient flow so the world model learns
        # from the main forward pass too. We detach only the stored
        # _prev_z_pred to prevent graph leaks across steps.
        # Also cache the full output to avoid redundant computation below.
        if action is not None and self.world_model is not None:
            z_pred_current, r_pred_current, d_pred_current = self.world_model(x_t, action_input)
            self._prev_z_pred = z_pred_current.detach()
            # Cache for reuse in the "World model prediction" section below
            self._wm_cache = (z_pred_current, r_pred_current, d_pred_current)
        else:
            self._prev_z_pred = None
            self._wm_cache = None

        # --- Attention Probability Maps ---
        # Weight the latent by attention before routing
        routing_x = x_t  # default: use raw x_t for routing
        if self.attention_maps is not None:
            weighted_latent, attention_info = self.attention_maps(
                latent=x_t,
                prediction_error=elementwise_pred_error,
                uncertainty=None,
                novelty=None,
            )
            routing_x = weighted_latent  # use attention-weighted latent for routing
            outputs["attention_info"] = attention_info

        # --- Curiosity / Intrinsic Motivation ---
        if self.curiosity is not None:
            # Project x_t to curiosity embedding dim
            embedded_latent = self.curiosity_proj(x_t)  # (batch, state_embedding_dim)
            # Project elementwise prediction error to curiosity embedding dim
            pred_error_proj = self.curiosity_proj(elementwise_pred_error)  # (batch, state_embedding_dim)
            intrinsic_reward, curiosity_info = self.curiosity(
                latent=embedded_latent,
                prediction_error=pred_error_proj,
                ensemble_uncertainty=None,
            )
            outputs["intrinsic_reward"] = intrinsic_reward
            outputs["curiosity_info"] = curiosity_info
            # Update visit counts during training
            if training:
                self.curiosity.update_visit_counts(embedded_latent)

        # Update context
        if self.meta_learning is not None and self.context is not None:
            # BUG 1 fix: detach x_t to prevent graph leaks across steps
            self.context = self.meta_learning.update_context(x_t.detach())

        # --- Opponent Modeling ---
        opponent_context = None
        if self.opponent_modeling is not None:
            # Use mean across batch for opponent modeling to avoid
            # batch shape mismatch with internal GRU hidden states
            batch_size = observation.size(0)
            opponent_obs_input = x_t.mean(dim=0, keepdim=True)  # (1, latent_dim)
            opponent_obs_input = opponent_obs_input.unsqueeze(1)  # (1, 1, latent_dim)
            opponent_context, opponent_info = self.opponent_modeling(opponent_obs=opponent_obs_input)
            outputs["opponent_info"] = opponent_info
            # Ensure opponent_context has correct batch dimension
            if opponent_context.dim() == 1:
                opponent_context = opponent_context.unsqueeze(0).expand(batch_size, -1)
            elif opponent_context.size(0) == 1 and batch_size > 1:
                opponent_context = opponent_context.expand(batch_size, -1)

        # Fix 4: Decoupled routing - slow router policy + fast deterministic gating
        # The router forward pass is always fast (deterministic top-k).
        # Router WEIGHT updates happen only at MEDIUM timescale, controlled by governor.
        gates, selected_indices, router_info = self.router(
            self.h_t,
            routing_x,
            self.m_t,
            self.context,
            prediction_error,
            training=training
        )
        outputs["router_info"] = router_info
        outputs["gates"] = gates
        outputs["selected_indices"] = selected_indices

        # --- Compute Economy ---
        compute_allocations = None
        if self.compute_market is not None:
            batch_size = observation.size(0)
            num_experts = self.config.router.num_experts
            # Build expert_utilities tensor (batch, num_experts)
            utility_tensor = torch.zeros(batch_size, num_experts, device=observation.device)
            for i, stats in self.expert_bank.expert_stats.items():
                utility_tensor[:, i] = stats.utility_score

            # Build routing_gates tensor (batch, num_experts) from sparse gates
            routing_gates_full = torch.zeros(batch_size, num_experts, device=observation.device)
            for k in range(selected_indices.size(1)):
                routing_gates_full.scatter_(1, selected_indices[:, k:k+1], gates[:, k:k+1])

            compute_allocations, market_info = self.compute_market(
                expert_utilities=utility_tensor,
                routing_gates=routing_gates_full,
                context=self.h_t,
            )
            outputs["market_info"] = market_info

        # Apply experts
        delta_h, compute_costs = self.expert_bank(
            self.h_t,
            selected_indices,
            gates
        )
        outputs["compute_costs"] = compute_costs

        # Fix 3: Update capacity ledger with activation data
        if self.governor is not None:
            active_set = set()
            for k in range(selected_indices.size(1)):
                for idx in selected_indices[:, k].unique().tolist():
                    active_set.add(idx)
            for exp_id in self.governor.ledger._entries:
                self.governor.ledger.update_activation(exp_id, exp_id in active_set)

        # Apply compute allocations to expert outputs if compute market is enabled
        if self.compute_market is not None and compute_allocations is not None:
            # BUG 3 fix: expand compute_allocations from (num_experts,) to
            # (batch_size, num_experts) before gathering with selected_indices
            # which is shape (batch_size, active_experts)
            batch_size = observation.size(0)
            if compute_allocations.dim() == 1:
                compute_allocations = compute_allocations.unsqueeze(0).expand(batch_size, -1)
            selected_allocs = compute_allocations.gather(1, selected_indices)  # (batch, active_experts)
            alloc_scale = selected_allocs.mean(dim=-1, keepdim=True)  # (batch, 1)
            mean_alloc = compute_allocations.mean() + 1e-8
            alloc_scale = alloc_scale / mean_alloc
            delta_h = delta_h * alloc_scale

        h_tilde = self.h_t + delta_h
        outputs["delta_h"] = delta_h

        # --- Hierarchical Expert Society ---
        if self.hierarchical is not None:
            hierarchical_output, hierarchical_routing_info = self.hierarchical(
                h_tilde, x_t, context=self.context
            )
            h_tilde = h_tilde + hierarchical_output  # residual connection
            outputs["hierarchical_info"] = {
                "routing_info": hierarchical_routing_info,
            }

        # --- Opponent Modeling residual ---
        if self.opponent_modeling is not None and opponent_context is not None:
            # Add small residual from opponent context
            h_tilde = h_tilde + 0.1 * opponent_context

        # Meta-learning adaptation
        if self.meta_learning is not None and self.context is not None:
            h_adapted, meta_info = self.meta_learning.adapt(
                h_tilde,
                self.context,
                gradient=None  # Would compute from loss
            )
            h_tilde = h_adapted
            outputs["meta_info"] = meta_info

        # --- Reasoning Engine ---
        if self.reasoning_engine is not None:
            refined_h, reasoning_info = self.reasoning_engine(
                h_tilde, x_t,
                world_model=self.world_model if self.config.reasoning.use_counterfactual else None,
                action_dim=self.config.action_dim,
                training=training
            )
            h_tilde = refined_h
            outputs["reasoning_info"] = reasoning_info

        # --- Subgoal Generator ---
        if self.subgoal_generator is not None:
            # Convert reward to tensor, handle None
            reward_tensor = torch.tensor(
                reward if reward is not None else 0.0,
                device=observation.device
            )
            # Estimate uncertainty from prediction error
            uncertainty_tensor = prediction_error.detach().clone()
            # Compute episode progress from step count
            progress_tensor = torch.tensor(
                min(1.0, self.step / max(1, self.config.training.rollout_length)),
                device=observation.device
            )
            active_subgoal, subgoal_info = self.subgoal_generator(
                h_t=self.h_t,
                x_t=x_t,
                reward=reward_tensor,
                uncertainty=uncertainty_tensor,
                episode_progress=progress_tensor,
            )
            outputs["subgoal_info"] = subgoal_info
            if "subgoal_reward" in subgoal_info:
                outputs["subgoal_reward"] = subgoal_info["subgoal_reward"]

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

        # World model prediction — reuse the computation from earlier if available.
        # BUG FIX: Previously, the world model was computed TWICE with identical inputs:
        #   1. At line ~473 for prediction error tracking (result immediately detached)
        #   2. Here for the outputs dict (result used in losses)
        # The first call created an orphaned gradient graph that wasted compute.
        # Now we store the result from the first call and reuse it here.
        if self.world_model is not None and action is not None:
            if hasattr(self, '_wm_cache') and self._wm_cache is not None:
                # Reuse cached result from prediction error computation
                z_next, r_pred, d_pred = self._wm_cache
                outputs["world_model"] = {
                    "z_next": z_next,
                    "r_pred": r_pred,
                    "d_pred": d_pred,
                }
                self._wm_cache = None  # Clear cache
            else:
                # Fallback: compute if cache was cleared
                z_next, r_pred, d_pred = self.world_model(x_t, action_input)
                outputs["world_model"] = {
                    "z_next": z_next,
                    "r_pred": r_pred,
                    "d_pred": d_pred,
                }

        # Feature extraction
        if self.feature_validator is not None and training:
            feature_ids = self.feature_validator.extract_features(x_t)
            outputs["feature_ids"] = feature_ids

        # -------------------------------------------------------
        # Stable Self-Improvement: Meta-Loop Observation
        # -------------------------------------------------------
        if self.meta_loop is not None and training:
            density = self.expert_bank.capability_density()
            active_experts = self.expert_bank.get_active_experts()
            routing_entropy = outputs.get("router_info", {}).get("entropy", None)
            mean_utility = (
                sum(s.utility_score for s in self.expert_bank.expert_stats.values())
                / max(1, len(self.expert_bank.expert_stats))
            )
            meta_obs = self.meta_loop.observe(
                density=density,
                num_active_experts=len(active_experts),
                max_experts=self.expert_bank.max_experts,
                routing_entropy=routing_entropy.item() if routing_entropy is not None else 1.0,
                mean_utility=mean_utility,
            )
            outputs["meta_loop_obs"] = meta_obs

            # If meta-loop detects regression, freeze architecture
            if meta_obs.get("is_regressing", False) and self.governor is not None:
                self.governor.freeze_structural_changes()

        # -------------------------------------------------------
        # Stable Self-Improvement: Formal Verification
        # -------------------------------------------------------
        if self.formal_verification is not None and training:
            # Update stable baseline at slow timescale
            if "probs" in outputs.get("router_info", {}):
                probs = outputs["router_info"]["probs"]
                if self.step % self.config.formal_verification.kl_check_interval == 0:
                    self.formal_verification.update_stable_baseline(probs)

        # -------------------------------------------------------
        # Stable Self-Improvement: Dynamic Hyperparams Recording
        # -------------------------------------------------------
        if self.dynamic_hyperparams is not None and training:
            # Record gradient norm proxy (use delta_h norm)
            grad_norm = 0.0
            if "delta_h" in outputs:
                grad_norm = outputs["delta_h"].detach().norm().item()
            loss_proxy = prediction_error.item() if isinstance(prediction_error, torch.Tensor) else float(prediction_error)
            self.dynamic_hyperparams.record(
                grad_norm=grad_norm,
                loss=loss_proxy,
                prediction_error=loss_proxy,
            )

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
                action_dim = self.config.action_dim
                mean = policy_logits[:, :action_dim]
                log_std = policy_logits[:, action_dim:]
                # Clamp log_std for numerical stability
                log_std = torch.clamp(log_std, -20, 2)
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

        Fix 1: The RL objective is PRIMARY. All auxiliary losses are
        CONSTRAINT regularizers whose coefficients are governed.

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

        # PPO loss (PRIMARY objective)
        from deep_thought.optimization.losses import compute_ppo_loss
        if self.config.action_space == "discrete":
            policy_logits = outputs["policy_logits"]
            action_probs = F.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(batch["actions"])
        else:
            action_dim = self.config.action_dim
            mean = outputs["policy_logits"][:, :action_dim]
            log_std = outputs["policy_logits"][:, action_dim:]
            log_std = torch.clamp(log_std, -20, 2)
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

        # RL loss is the primary objective
        rl_loss = ppo_losses["total_loss"]

        # World model loss (CONSTRAINT)
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

        # Compute penalty (CONSTRAINT)
        compute_loss = torch.tensor(0.0, device=batch["observations"].device)
        if "compute_costs" in outputs:
            from deep_thought.optimization.losses import compute_compute_penalty
            compute_loss = compute_compute_penalty(
                outputs["compute_costs"],
                outputs["gates"],
                outputs["selected_indices"],
                self.config.training.compute_penalty_coef
            )
        losses["compute_loss"] = compute_loss

        # LEVER 5: Capability density reward
        # Capability Density = mean_expert_utility / total_param_count
        # This is a REWARD (negative loss) — the system is incentivized to
        # achieve the same performance with fewer parameters.  This directly
        # combats the neuron explosion bug where "more neurons = lower error"
        # was the equilibrium the system found.
        capability_density = self.expert_bank.capability_density()
        density_coef = getattr(self.config.training, 'capability_density_coef', 0.01)
        # Negative because we want to MAXIMIZE density (minimize the loss)
        losses["capability_density_loss"] = torch.tensor(
            -density_coef * capability_density,
            device=batch["observations"].device
        )

        # Encoder losses
        if "encoder_info" in outputs:
            encoder_losses = self.encoder.compute_losses(outputs["encoder_info"])
            losses.update(encoder_losses)

        # Router losses
        if "router_info" in outputs:
            router_losses = self.router.compute_losses(outputs["router_info"])
            losses.update(router_losses)

        # Fix 1: Governed total loss — RL is primary, others are constraints
        if self.governor is not None:
            # Build auxiliary losses dict for governance
            auxiliary_losses = {}
            for key, val in losses.items():
                if isinstance(val, torch.Tensor) and key not in (
                    "total_loss", "policy_loss", "value_loss", "entropy_loss"
                ):
                    auxiliary_losses[key] = val

            total_loss, constraint_weights = self.governor.compute_governed_loss(
                rl_loss, auxiliary_losses
            )
            losses["total_loss"] = total_loss
            losses["governance_constraint_weights"] = constraint_weights
        else:
            # Legacy: un-governed total loss (no constraint hierarchy)
            device = batch["observations"].device
            total_loss = torch.tensor(0.0, device=device)
            for loss_val in losses.values():
                if isinstance(loss_val, torch.Tensor):
                    total_loss = total_loss + loss_val
            losses["total_loss"] = total_loss

        return losses

    def update_expert_utility(
        self,
        gradient_norms: Dict[int, float],
        reward_contributions: Dict[int, float]
    ):
        """Update expert utility scores and capacity ledger."""
        self.expert_bank.update_utility(gradient_norms, reward_contributions)

        # Fix 3 + Fix 7: Update capacity ledger with normalized utility
        if self.governor is not None:
            for exp_id, stats in self.expert_bank.expert_stats.items():
                self.governor.ledger.update_utility(exp_id, stats.utility_score)

    def prune_experts(self):
        """
        Prune low-utility experts through governance.

        Fix 2: Only allowed at SLOW timescale.
        Fix 3: Must pass capacity ledger check.
        Fix 6: Goes through proposal bus.
        """
        # Check governance (Fix 2: timescale, Fix 3: capacity)
        if self.governor is not None:
            if not self.governor.is_operation_allowed("expert_pruning"):
                return

            # Fix 6: Submit pruning proposal
            self.expert_bank.mark_dormant(self.config.training.dormant_threshold)
            self.expert_bank.mark_dead(
                self.config.training.delete_threshold,
                self.config.training.delete_confirmation_steps
            )

            # Check each dead expert against the capacity ledger
            dead_experts = [
                exp_id for exp_id, stats in self.expert_bank.expert_stats.items()
                if stats.state == self.expert_bank.ExpertState.DEAD
            ]

            approved_to_prune = []
            for exp_id in dead_experts:
                approved, reason = self.governor.evaluate_pruning_proposal(exp_id)
                if approved:
                    approved_to_prune.append(exp_id)
                else:
                    # Submit as proposal for future evaluation
                    self.governor.submit_proposal(Proposal(
                        proposal_type=ProposalType.PRUNE_EXPERT,
                        source="expert_bank",
                        payload={"expert_id": exp_id},
                        predicted_impact=self.governor.ledger._entries.get(
                            exp_id
                        ).marginal_contribution if exp_id in self.governor.ledger._entries else 0.0,
                        created_step=self.step,
                    ))

            # Only prune approved experts
            for exp_id in approved_to_prune:
                self.expert_bank.expert_stats[exp_id].state = self.expert_bank.ExpertState.DEAD
                self.governor.ledger.remove_expert(exp_id)

            self.expert_bank.prune_dead_experts()
            self.governor.mark_operation_done("expert_pruning")
        else:
            # Legacy path (no governance)
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
        """
        Grow new experts through governance.

        Fix 2: Only allowed at SLOW timescale.
        Fix 3: Must pass capacity ledger (growth must "buy out" capacity).
        Fix 6: Goes through proposal bus.
        Stable SI: Must pass Formal Verification and Meta-Loop density check.
        """
        # Check governance
        if self.governor is not None:
            if not self.governor.is_operation_allowed("expert_growth"):
                return

            # Stable SI: Meta-loop density check
            if self.meta_loop is not None:
                if self.meta_loop.should_freeze_architecture():
                    return  # Meta-loop detected density regression

            # Fix 3: Check capacity ledger
            predicted_marginal = 0.5  # Default estimate
            if not self.governor.evaluate_growth_proposal(-1, predicted_marginal):
                return  # Growth denied by capacity ledger

            # Stable SI: Formal verification of the growth proposal
            if self.formal_verification is not None:
                current_density = self.expert_bank.capability_density()
                # Conservative: predict density will drop slightly due to new params
                predicted_density = current_density * 0.95
                approved, details = self.formal_verification.verify_change(
                    change_type="growth",
                    current_capability_density=current_density,
                )
                if not approved:
                    return  # Growth denied by formal verification

            # Fix 6: Submit growth proposal
            self.governor.submit_proposal(Proposal(
                proposal_type=ProposalType.GROW_EXPERT,
                source="expert_bank",
                payload={"predicted_marginal": predicted_marginal},
                predicted_impact=predicted_marginal,
                created_step=self.step,
            ))

            # Evaluate pending proposals
            approved = self.governor.evaluate_proposals()
            for proposal in approved:
                if proposal.proposal_type == ProposalType.GROW_EXPERT:
                    new_id = self.expert_bank.grow_expert()
                    # Register with capacity ledger
                    expert = self.expert_bank._get_expert(new_id)
                    if expert is not None:
                        n_params = sum(p.numel() for p in expert.parameters())
                        self.governor.ledger.register_expert(new_id, n_params)
                    self.governor.proposal_bus.mark_executed(proposal.proposal_id)

            self.governor.mark_operation_done("expert_growth")
        else:
            # Legacy path (no governance)
            if self.srp is not None:
                signals = self.srp.get_stats()
                if signals["architecture_gate"]["allow_growth"]:
                    if len(self.expert_bank) < self.config.expert_compiler.max_experts:
                        self.expert_bank.grow_expert()

    def consolidate_memory(self):
        """
        Consolidate episodic memory into semantic memory.

        Fix 2: Only allowed at MEDIUM timescale.
        """
        if self.governor is not None:
            if not self.governor.is_operation_allowed("memory_consolidation"):
                return

        self.memory.consolidate()
        self.memory.age_memories()
        self.memory.decay_semantic()
        self.memory.prune_semantic()

        # Decay curiosity during consolidation
        if self.curiosity is not None:
            self.curiosity.decay_curiosity()

        if self.governor is not None:
            self.governor.mark_operation_done("memory_consolidation")

    def validate_features(self):
        """
        Validate features and promote to experts.

        Fix 2: Only allowed at SLOW timescale.
        """
        if self.governor is not None:
            if not self.governor.is_operation_allowed("feature_validation"):
                return

        if self.feature_validator is not None:
            for feature_id in list(self.feature_validator.features.keys()):
                self.feature_validator.validate_feature(feature_id)

            promoted = self.feature_validator.promote_features()

            # Compile promoted features into experts
            if self.expert_compiler is not None:
                for feature_id in promoted:
                    if feature_id in self.feature_validator.features:
                        feature = self.feature_validator.features[feature_id]
                        self.expert_compiler.create_candidate(feature)

        if self.governor is not None:
            self.governor.mark_operation_done("feature_validation")

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

            # Fix 1+6: On regression, freeze structural changes via governor
            if signals["should_rollback"]:
                self.srp.rollback(self)
                if self.governor is not None:
                    self.governor.freeze_structural_changes()

            # Update governor regression state
            if self.governor is not None:
                self.governor.update_regression_state(reward, loss)
                # Update return sensitivity for signals (Fix 7)
                self.governor.update_return_sensitivity("reward", reward, reward)
                self.governor.update_return_sensitivity("loss", reward, loss)
                if routing_entropy is not None:
                    self.governor.update_return_sensitivity(
                        "routing_entropy", reward, routing_entropy
                    )

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        stats = {
            "step": self.step,
            "num_experts": len(self.expert_bank),
            "active_experts": len(self.expert_bank.get_active_experts()),
            "dormant_experts": len(self.expert_bank.get_dormant_experts()),
            "capability_density": self.expert_bank.capability_density(),
            "dormant_cached_experts": len(self.expert_bank.dormant_cache),
            "memory_stats": self.memory.get_memory_stats(),
        }

        if self.feature_validator is not None:
            stats["feature_stats"] = self.feature_validator.get_feature_stats()

        if self.expert_compiler is not None:
            stats["expert_compiler_stats"] = self.expert_compiler.get_candidate_stats()

        if self.srp is not None:
            stats["srp_stats"] = self.srp.get_stats()

        if self.curiosity is not None:
            stats["curiosity_stats"] = self.curiosity.get_curiosity_stats()

        if self.hierarchical is not None:
            stats["hierarchical_stats"] = {
                "num_tiers": self.hierarchical.config.num_tiers,
            }

        if self.opponent_modeling is not None:
            stats["opponent_modeling_stats"] = {
                "max_opponents": self.opponent_modeling.config.max_opponents,
                "mean_risk": self.opponent_modeling.risk_ema.mean().item(),
                "mean_deception": self.opponent_modeling.deception_scores.mean().item(),
                "total_interactions": self.opponent_modeling.interaction_counts.sum().item(),
            }

        if self.compute_market is not None:
            stats["compute_market_stats"] = self.compute_market.get_market_stats()

        if self.subgoal_generator is not None:
            stats["subgoal_stats"] = {
                "active_subgoals": len(self.subgoal_generator._active_subgoals),
                "completed_subgoals": len(self.subgoal_generator._completed_subgoals),
                "step_count": self.subgoal_generator._step_count,
            }

        if self.attention_maps is not None:
            stats["attention_maps_stats"] = {
                "compute_allocation": self.attention_maps.get_compute_allocation(),
            }

        if self.governor is not None:
            stats["governance_stats"] = self.governor.get_stats()

        # Stable Self-Improvement stats
        if self.meta_loop is not None:
            stats["meta_loop_stats"] = self.meta_loop.get_stats()

        if self.formal_verification is not None:
            stats["formal_verification_stats"] = self.formal_verification.get_stats()

        if self.shadow_evolution is not None:
            stats["shadow_evolution_stats"] = self.shadow_evolution.get_stats()

        if self.dynamic_hyperparams is not None:
            stats["dynamic_hyperparams_stats"] = self.dynamic_hyperparams.get_stats()

        return stats
