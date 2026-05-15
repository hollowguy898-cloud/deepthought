"""Configuration system for Deep Thought."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class EncoderConfig:
    """Encoder configuration."""
    latent_dim: int = 1024
    hidden_dim: int = 2048
    use_conv: bool = False
    image_size: Optional[int] = None
    num_layers: int = 2
    activation: str = "silu"
    use_layer_norm: bool = True
    observation_dim: Optional[int] = None


@dataclass
class RouterConfig:
    """Router configuration."""
    num_experts: int = 128
    active_experts: int = 4
    hidden_dim: int = 1024
    noise_epsilon: float = 0.1
    load_balance_loss_coef: float = 0.1  # FIX: 10x increase to prevent routing collapse
    entropy_coef: float = 0.1  # FIX: Higher to prevent routing collapse and ensure diverse expert usage
    min_entropy: float = 0.5
    max_entropy: float = 2.5


@dataclass
class ExpertConfig:
    """Expert configuration."""
    hidden_dim: int = 4096
    num_layers: int = 2
    activation: str = "swiglu"
    use_residual: bool = True
    dropout: float = 0.0


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    use_working_memory: bool = True
    use_episodic_memory: bool = True
    use_semantic_memory: bool = True
    use_procedural_memory: bool = True
    
    # Working memory
    working_memory_size: int = 1024
    
    # Episodic memory
    episodic_capacity: int = 10000
    episodic_key_dim: int = 256
    episodic_value_dim: int = 1024
    importance_threshold: float = 0.1  # Was 0.5 -- too high, no memories stored
    
    # Semantic memory
    semantic_capacity: int = 5000
    semantic_dim: int = 512
    consolidation_rate: float = 0.01
    
    # Procedural memory (expert bank)
    procedural_decay: float = 0.999


@dataclass
class WorldModelConfig:
    """World model configuration."""
    use_world_model: bool = True
    latent_dim: int = 1024
    hidden_dim: int = 2048
    action_dim: Optional[int] = None
    predict_reward: bool = True
    predict_done: bool = True
    prediction_horizon: int = 5


@dataclass
class FeatureValidationConfig:
    """Feature Validation Engine configuration."""
    use_fve: bool = True
    feature_buffer_size: int = 1000
    validation_window: int = 100
    temporal_consistency_threshold: float = 0.7
    causal_test_interval: int = 500
    promotion_threshold: float = 0.8
    competition_strength: float = 1.0
    noise_robustness_threshold: float = 0.1


@dataclass
class ExpertCompilerConfig:
    """Feature -> Expert Compiler configuration."""
    use_fec: bool = True
    quarantine_steps: int = 1000
    anchor_loss_coef: float = 1.0
    specialization_epochs: int = 10
    split_variance_threshold: float = 0.5
    merge_distance_threshold: float = 0.1
    max_experts: int = 256


@dataclass
class PlanningConfig:
    """Temporal Cognition & Planning Layer configuration."""
    use_tcpl: bool = True
    planning_horizon: int = 10
    micro_horizon: int = 1
    tactical_horizon: int = 5
    strategic_horizon: int = 20
    plan_memory_size: int = 1000
    replan_interval: int = 5
    correction_threshold: float = 0.3
    simulation_rollouts: int = 4


@dataclass
class MetaLearningConfig:
    """Meta-learning configuration."""
    use_meta_learning: bool = True
    use_fast_weights: bool = True
    fast_weight_dim: int = 512
    fast_weight_lr: float = 0.1
    fast_weight_decay: float = 0.99
    context_dim: int = 256
    adaptation_steps: int = 5
    inner_lr: float = 0.01


@dataclass
class SRPConfig:
    """Self-Regression Prevention configuration."""
    use_srp: bool = True
    performance_window: int = 1000
    drift_threshold: float = 0.15  # Was 0.05 -- too sensitive, triggered regression during normal training
    stability_check_interval: int = 100
    max_fast_weight_norm: float = 10.0
    fast_weight_decay: float = 0.95
    routing_entropy_min: float = 0.5
    routing_entropy_max: float = 3.0
    rollback_on_regression: bool = True
    checkpoint_interval: int = 10000


@dataclass
class CuriosityConfig:
    """Curiosity and intrinsic motivation configuration."""
    use_curiosity: bool = True
    prediction_error_coef: float = 0.1
    novelty_coef: float = 0.05
    uncertainty_coef: float = 0.05
    info_gain_coef: float = 0.02
    curiosity_decay: float = 0.999
    visit_count_hash_size: int = 10000
    min_curiosity: float = 0.001
    max_intrinsic_reward: float = 1.0
    state_embedding_dim: int = 64


@dataclass
class SubgoalConfig:
    """Self-generated subgoal system configuration."""
    use_subgoals: bool = True
    goal_embedding_dim: int = 256
    max_active_subgoals: int = 5
    subgoal_proposal_interval: int = 50
    completion_threshold: float = 0.8
    decomposition_depth: int = 3
    subgoal_reward_coef: float = 0.1
    subgoal_types: list = field(default_factory=lambda: ["explore", "secure", "conserve", "reduce_uncertainty", "probe"])


@dataclass
class OpponentModelingConfig:
    """Opponent and social modeling configuration.

    Controls the opponent modeling system that tracks opponent tendencies,
    deception patterns, risk preferences, and habitual movement patterns.

    Attributes:
        use_opponent_modeling: Whether to enable opponent modeling.
            If False, the system returns zero opponent contexts.
        opponent_latent_dim: Dimensionality of the opponent latent space
            produced by the OpponentEncoder.
        tendency_dim: Dimensionality of the tendency embedding produced
            by the TendencyTracker.
        max_opponents: Maximum number of opponents that can be tracked
            simultaneously.
        strategy_horizon: Number of future steps the StrategyPredictor
            auto-regressively rolls out.
        deception_threshold: Threshold above which an opponent is
            considered deceptive by the DeceptionDetector.
        risk_estimation_ema: Exponential moving average decay for the
            RiskProfileEstimator.  Higher values make risk estimates
            more stable; lower values make them more reactive.
        tendency_update_rate: EMA update rate for the tendency vectors.
            Controls how quickly the stored tendency adapts to new
            observations.
        num_tendency_types: Number of discrete tendency categories
            (e.g., aggressive, defensive, deceptive, exploratory,
            conservative, chaotic, cooperative, opportunistic).
    """
    use_opponent_modeling: bool = True
    opponent_latent_dim: int = 256
    tendency_dim: int = 64
    max_opponents: int = 8
    strategy_horizon: int = 10
    deception_threshold: float = 0.5
    risk_estimation_ema: float = 0.95
    tendency_update_rate: float = 0.01
    num_tendency_types: int = 8  # aggressive, defensive, deceptive, etc.


@dataclass
class HierarchicalConfig:
    """Hierarchical expert society configuration."""
    use_hierarchy: bool = True
    num_tiers: int = 4
    reflex_experts: int = 32
    tactical_experts: int = 16
    strategic_experts: int = 8
    meta_experts: int = 4
    reflex_hidden_dim: int = 256
    tactical_hidden_dim: int = 512
    strategic_hidden_dim: int = 512
    meta_hidden_dim: int = 256
    compute_budget_total: float = 1.0
    budget_allocation_lr: float = 0.01


@dataclass
class ComputeEconomyConfig:
    """Dynamic compute economy configuration.

    Controls the competitive market where experts bid for compute
    resources.  The market allocates compute to experts that can
    make the best use of it, while enforcing a global energy budget
    that prevents runaway computation.

    Attributes:
        use_compute_market: Whether to enable the compute market.
            If False, compute is allocated equally across experts.
        total_energy_budget: Maximum compute energy capacity.  The
            total compute spent by all experts in a single step
            cannot exceed this value.
        energy_recharge_rate: How much energy is restored per step.
            A rate of 1.0 means the budget fully recharges each step;
            lower rates enforce stronger compute conservation.
        min_bid_price: Floor price for compute bids.  Ensures every
            bid has a non-trivial cost, preventing zero-cost compute
            grabs.
        auction_type: Auction mechanism to use.  ``"sealed_bid"``
            is a first-price auction; ``"vickrey"`` is a second-price
            auction that encourages truthful bidding.
        credit_ema: Exponential moving average decay for expert
            credit updates.  A value close to 1.0 makes credit
            change slowly (stable); closer to 0.0 makes it reactive.
        bidding_hidden_dim: Width of the hidden layer in each
            expert's bidding MLP.
        market_temperature: Softmax temperature for the differentiable
            soft auction.  Higher temperatures produce more uniform
            allocations; lower temperatures concentrate compute on
            the highest-priority experts.
    """
    use_compute_market: bool = True
    total_energy_budget: float = 100.0
    energy_recharge_rate: float = 1.0
    min_bid_price: float = 0.01
    auction_type: str = "sealed_bid"  # sealed_bid, vickrey
    credit_ema: float = 0.99
    bidding_hidden_dim: int = 128
    market_temperature: float = 1.0


@dataclass
class AttentionMapsConfig:
    """Attention-Based Probability Maps configuration.

    Controls the attention-driven probability map system that dynamically
    allocates compute based on confidence, uncertainty, and temporal
    evolution signals.

    Attributes:
        use_attention_maps: Whether to enable attention probability maps.
            If False, the latent representation is passed through without
            attention weighting.
        num_heads: Number of attention heads for multi-head content
            attention.  Must evenly divide *latent_dim*.
        confidence_decay: Exponential moving average decay for the
            confidence tracker.  Values closer to 1.0 make confidence
            estimates more stable; closer to 0.0 makes them reactive.
        uncertainty_threshold: Threshold above which a region is
            classified as high-uncertainty and given additional compute.
        evolution_hidden_dim: Hidden size of the GRU cell in the
            temporal evolution module.
        min_attention: Floor value for attention weights.  Prevents
            any dimension from receiving exactly zero compute.
    """
    use_attention_maps: bool = True
    num_heads: int = 8
    confidence_decay: float = 0.99
    uncertainty_threshold: float = 0.5
    evolution_hidden_dim: int = 256
    min_attention: float = 0.01


@dataclass
class MetaLearningRulesConfig:
    """Meta-Learning of Learning Rules configuration.

    Controls the learned optimiser that replaces hand-designed learning-rate
    schedules with an LSTM-based network that produces per-parameter-group
    hyperparameters (lr, momentum, weight_decay).

    Attributes:
        use_meta_optimizer: Whether to enable the meta-optimiser.  If False,
            a standard optimiser (e.g. Adam) should be used instead.
        hidden_dim: Hidden size of the LSTM in the UpdateRuleNetwork.
        num_lstm_layers: Number of stacked LSTM layers.
        max_learning_rate: Upper bound on the learning rate output.
        min_learning_rate: Lower bound on the learning rate output.
        max_weight_decay: Upper bound on the weight-decay output.
        meta_lr: Learning rate for the meta-optimiser itself (Adam over the
            rule-network parameters).
        regularization_coef: Penalty coefficient that discourages the rule
            network from producing extremely high learning rates.
        statistics_decay: EMA decay factor for gradient-statistic tracking.
            Closer to 1.0 → smoother / longer memory.
    """
    use_meta_optimizer: bool = True
    hidden_dim: int = 128
    num_lstm_layers: int = 2
    max_learning_rate: float = 0.1
    min_learning_rate: float = 1e-6
    max_weight_decay: float = 0.1
    meta_lr: float = 0.001
    regularization_coef: float = 0.01
    statistics_decay: float = 0.99


@dataclass
class MetaLoopConfig:
    """Capability Density Meta-Loop configuration.

    Controls the meta-RL loop that treats Capability Density as the
    primary reward signal for architectural changes.

    Attributes:
        use_meta_loop: Whether to enable the meta-loop.
        density_reward_coef: Coefficient for the density reward.
        density_regression_threshold: Fraction drop from max to trigger regression.
        meta_lr: Learning rate for the meta-optimizer.
        meta_action_dim: Dimensionality of meta-action embedding.
        history_length: Number of density observations for trend detection.
        min_density_improvement: Minimum relative improvement to approve change.
        density_ema_decay: EMA decay for density tracking.
    """
    use_meta_loop: bool = True
    density_reward_coef: float = 0.1
    density_regression_threshold: float = 0.15
    meta_lr: float = 1e-5
    meta_action_dim: int = 64
    history_length: int = 200
    min_density_improvement: float = 0.01
    density_ema_decay: float = 0.999


@dataclass
class FormalVerificationConfig:
    """Formal Verification Layer configuration.

    Controls the verification layer that enforces logic-based constraints
    and entropy regulation before architectural changes.

    Attributes:
        use_formal_verification: Whether to enable the formal verification layer.
        kl_epsilon: Maximum KL divergence from stable baseline.
        kl_check_interval: How often to check KL divergence.
        max_output_norm: Maximum expert output norm.
        min_capability_density: Minimum density below which growth is blocked.
        gradient_explosion_threshold: Maximum gradient norm.
        verification_tier: Default verification tier (syntactic/semantic/causal).
        constraint_violation_cooldown: Steps to wait after violation.
    """
    use_formal_verification: bool = True
    kl_epsilon: float = 0.1
    kl_check_interval: int = 100
    max_output_norm: float = 100.0
    min_capability_density: float = 0.001
    gradient_explosion_threshold: float = 10.0
    verification_tier: str = "semantic"
    constraint_violation_cooldown: int = 500


@dataclass
class ShadowEvolutionConfig:
    """Shadow Evolution configuration.

    Controls the evolutionary search system that mutates dormant experts
    in the background and swaps them in when they outperform active ones.

    Attributes:
        use_shadow_evolution: Whether to enable shadow evolution.
        max_shadow_experts: Maximum shadow experts to maintain.
        mutation_rate: Probability of mutating each weight.
        mutation_strength: Std dev of Gaussian noise for weight mutations.
        tournament_size: Number of experts compared in tournament selection.
        validation_window: Number of validation samples per evaluation.
        swap_threshold: Minimum improvement ratio for swap approval.
        evolution_interval: Steps between evolution cycles.
        max_mutations_per_cycle: Maximum mutations per cycle.
        archive_size: Number of best shadows to archive.
    """
    use_shadow_evolution: bool = True
    max_shadow_experts: int = 16
    mutation_rate: float = 0.1
    mutation_strength: float = 0.01
    tournament_size: int = 4
    validation_window: int = 100
    swap_threshold: float = 0.1
    evolution_interval: int = 1000
    max_mutations_per_cycle: int = 4
    archive_size: int = 5


@dataclass
class DynamicHyperparamsConfig:
    """Dynamic Hyperparameter Adaptation configuration.

    Controls the system that dynamically adjusts learning rates, pruning
    thresholds, and other hyperparameters based on task volatility.

    Attributes:
        use_dynamic_hyperparams: Whether to enable dynamic hyperparams.
        volatility_window: Number of observations for volatility estimation.
        volatility_ema_decay: EMA decay for volatility tracking.
        lr_min: Minimum allowed learning rate.
        lr_max: Maximum allowed learning rate.
        lr_adjustment_rate: How quickly the meta-controller adjusts LR.
        pruning_threshold_min: Minimum pruning utility threshold.
        pruning_threshold_max: Maximum pruning utility threshold.
        pruning_threshold_adjustment_rate: How quickly pruning threshold adapts.
        warmup_trigger_threshold: Prediction error increase to trigger warmup.
        warmup_duration: Steps the re-warmup phase lasts.
        warmup_lr_multiplier: LR multiplier during warmup.
        warmup_freeze_architecture: Freeze architecture during warmup.
        curvature_window: Observations for loss curvature estimation.
        meta_controller_hidden_dim: Hidden dim of the meta-controller network.
    """
    use_dynamic_hyperparams: bool = True
    volatility_window: int = 100
    volatility_ema_decay: float = 0.99
    lr_min: float = 1e-6
    lr_max: float = 1e-2
    lr_adjustment_rate: float = 0.01
    pruning_threshold_min: float = 0.02
    pruning_threshold_max: float = 0.30
    pruning_threshold_adjustment_rate: float = 0.001
    warmup_trigger_threshold: float = 0.5
    warmup_duration: int = 1000
    warmup_lr_multiplier: float = 3.0
    warmup_freeze_architecture: bool = True
    curvature_window: int = 50
    meta_controller_hidden_dim: int = 64


@dataclass
class GovernanceConfig:
    """Architectural Governance configuration.

    Controls the 7 governance principles that impose hierarchy,
    separation of authority, and timing discipline on the system.

    Attributes:
        use_governor: Whether to enable the governance layer.
            If False, the system operates without governance constraints
            (legacy mode, not recommended).
        medium_interval: Step interval for MEDIUM-tier operations
            (routing temperature, memory consolidation, etc.).
        slow_interval: Step interval for SLOW-tier operations
            (pruning, growth, architecture changes).
        very_slow_interval: Step interval for VERY_SLOW-tier operations
            (world model updates, representation restructuring).
        max_total_parameters: Maximum total parameter budget across all experts.
        max_experts: Maximum number of experts allowed.
        min_experts: Minimum number of experts (cannot prune below this).
        pruning_confirmation_window: Steps to confirm a pruning decision.
        growth_marginal_threshold: Minimum predicted marginal contribution
            required for growth approval.
        sparsity_constraint_coef: Constraint coefficient for sparsity loss.
        entropy_constraint_coef: Constraint coefficient for entropy loss.
        load_balance_constraint_coef: Constraint coefficient for load balance.
        world_model_constraint_coef: Constraint coefficient for world model.
        compute_penalty_constraint_coef: Constraint coefficient for compute.
        memory_read_filter_threshold: Minimum relevance for memory reads.
        memory_influence_on_pruning: Whether memory can influence pruning
            (default False — Fix 5).
        memory_influence_on_growth: Whether memory can influence growth
            (default False — Fix 5).
    """
    use_governor: bool = True
    # Timescale separation
    medium_interval: int = 100
    slow_interval: int = 10_000
    very_slow_interval: int = 1_000_000
    # Capacity ledger
    max_total_parameters: int = 100_000_000
    max_experts: int = 256
    min_experts: int = 4
    pruning_confirmation_window: int = 10_000
    growth_marginal_threshold: float = 0.1
    # LEVER 2: Pruning triggers sooner with higher utility threshold
    pruning_utility_threshold: float = 0.08  # Was 0.05 -- prune low-value experts
    # LEVER 2: Detect redundancy sooner
    redundancy_threshold: float = 0.85      # Was 0.9 -- detect redundancy sooner
    # Constraint coefficients (Fix 1: auxiliary losses are constraints)
    sparsity_constraint_coef: float = 0.01
    entropy_constraint_coef: float = 0.01
    load_balance_constraint_coef: float = 0.01
    world_model_constraint_coef: float = 0.5
    # LEVER 1: Tightened compute penalty to prevent neuron explosion.
    # The old value (0.001) was far too permissive -- experts learned that
    # adding parameters was an easy shortcut to lower error, causing bloat.
    compute_penalty_constraint_coef: float = 0.05  # Was 0.001 -- 50x increase
    memory_coherence_constraint_coef: float = 0.01
    # LEVER 3: Expert hard cap (overrides max_experts for growth)
    expert_hard_cap: int = 64
    # LEVER 5: Capability density reward coefficient
    capability_density_coef: float = 0.01
    # Asymmetric memory (Fix 5)
    memory_read_filter_threshold: float = 0.3
    memory_influence_on_pruning: bool = False
    memory_influence_on_growth: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    algorithm: str = "ppo"  # ppo, sac, impala
    learning_rate: float = 3e-4
    batch_size: int = 256
    rollout_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_loss_coef: float = 0.25  # Was 0.5 -- reduced to prevent value loss dominating
    entropy_coef: float = 0.1  # FIX: Higher to prevent routing collapse and ensure diverse expert usage
    max_grad_norm: float = 0.5
    target_kl: float = 0.05  # Was 0.01 -- too strict, prevented learning
    ppo_epochs: int = 4
    
    # World model loss coefficients
    world_model_loss_coef: float = 0.5
    # LEVER 1: Tightened compute penalty in training config too.
    compute_penalty_coef: float = 0.05  # Was 0.001 -- matches governance coef
    
    # Pruning and growth
    # LEVER 2: Aggressive synaptic pruning thresholds.
    # Old values were far too lenient -- the model could accumulate
    # millions of dead parameters before cleanup.  The system should
    # be "impatient": if a neuron isn't contributing significantly,
    # it should be deleted quickly.
    prune_interval: int = 5000           # Was 10000 -- prune twice as often
    growth_interval: int = 10000         # Was 5000 -- grow half as often
    utility_ema_alpha: float = 0.99
    dormant_threshold: float = 0.25      # Was 0.15 -- mark dormant sooner
    delete_threshold: float = 0.10       # Was 0.05 -- delete sooner
    dormant_confirmation_steps: int = 10000   # Was 100000 -- 10x faster
    delete_confirmation_steps: int = 50000    # Was 1000000 -- 20x faster
    
    # NEURON EXPLOSION FIX: Hard expert cap budget
    # Maximum total experts that can ever exist. Growth beyond this is
    # absolutely forbidden regardless of predicted marginal contribution.
    expert_hard_cap: int = 64
    
    # NEURON EXPLOSION FIX: Capability density reward coefficient
    # Capability Density = performance / parameter_count
    # The system should REWARD achieving the same performance with fewer params.
    capability_density_coef: float = 0.01
    
    # NEURON EXPLOSION FIX: Dormancy offloading
    # After this many steps dormant, compress expert weights to save memory.
    dormancy_offload_steps: int = 20000
    
    # NEURON EXPLOSION FIX: Fast weight SRP constraint
    # Fast weight norm is constrained proportional to 1/sqrt(num_active_experts)
    # This prevents "more neurons = faster adaptation" spiral (Lamarckian problem).
    fast_weight_norm_per_expert_budget: float = 2.0


@dataclass
class ReasoningConfig:
    """Reasoning Engine configuration."""
    use_reasoning: bool = True
    num_reasoning_steps: int = 3
    use_counterfactual: bool = True
    num_counterfactual_actions: int = 4
    consistency_coef: float = 0.1
    reasoning_value_coef: float = 0.5


@dataclass
class MechanicDiscoveryConfig:
    """Mechanic Discovery Engine (MDE) configuration."""
    use_mde: bool = True
    observation_dim: int = 64
    action_dim: int = 4
    context_dim: int = 32
    stability_threshold: float = 0.85
    min_context_span: int = 3
    window_size: int = 200
    validation_interval: int = 100
    contradiction_threshold: float = 0.3
    max_mechanics: int = 100
    tag_prefix: str = "Mechanic"
    routing_hint_strength: float = 0.1
    expert_affinity_decay: float = 0.99


@dataclass
class AutonomousSpecializationConfig:
    """Autonomous Expert Specialization configuration."""
    use_autonomous_specialization: bool = True
    growth_confidence_threshold: float = 0.8
    growth_context_span_min: int = 3
    contradiction_prune_threshold: float = 0.2
    specialization_noise_scale: float = 0.005
    max_specialists_per_mechanic: int = 3


@dataclass
class StabilityInTheDarkConfig:
    """Stability in the Dark configuration."""
    use_stability_in_the_dark: bool = True
    density_rollback_gate: float = 0.1
    scientific_method_proof_threshold: float = 0.05
    rollback_cooldown: int = 500
    dissection_checkpoint_interval: int = 1000


@dataclass
class DeepThoughtConfig:
    """Main configuration for Deep Thought."""
    # Architecture
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    
    # Advanced systems
    feature_validation: FeatureValidationConfig = field(default_factory=FeatureValidationConfig)
    expert_compiler: ExpertCompilerConfig = field(default_factory=ExpertCompilerConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    srp: SRPConfig = field(default_factory=SRPConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    subgoal: SubgoalConfig = field(default_factory=SubgoalConfig)
    opponent_modeling: OpponentModelingConfig = field(default_factory=OpponentModelingConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    compute_economy: ComputeEconomyConfig = field(default_factory=ComputeEconomyConfig)
    attention_maps: AttentionMapsConfig = field(default_factory=AttentionMapsConfig)
    meta_learning_rules: MetaLearningRulesConfig = field(default_factory=MetaLearningRulesConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    
    # Stable self-improvement components
    meta_loop: MetaLoopConfig = field(default_factory=MetaLoopConfig)
    formal_verification: FormalVerificationConfig = field(default_factory=FormalVerificationConfig)
    shadow_evolution: ShadowEvolutionConfig = field(default_factory=ShadowEvolutionConfig)
    dynamic_hyperparams: DynamicHyperparamsConfig = field(default_factory=DynamicHyperparamsConfig)
    
    # Reasoning engine
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)

    # Black Box components
    mechanic_discovery: MechanicDiscoveryConfig = field(default_factory=MechanicDiscoveryConfig)
    autonomous_specialization: AutonomousSpecializationConfig = field(default_factory=AutonomousSpecializationConfig)
    stability_in_the_dark: StabilityInTheDarkConfig = field(default_factory=StabilityInTheDarkConfig)
    
    # Training
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Environment
    observation_dim: Optional[int] = None
    action_dim: Optional[int] = None
    action_space: str = "discrete"  # discrete, continuous
    num_actions: Optional[int] = None
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 5000
    save_interval: int = 10000
    log_dir: str = "./logs"
    
    @classmethod
    def from_yaml(cls, path: str) -> "DeepThoughtConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return cls()
        
        # Handle nested dataclasses properly
        nested_configs = {
            "encoder": EncoderConfig,
            "router": RouterConfig,
            "expert": ExpertConfig,
            "memory": MemoryConfig,
            "world_model": WorldModelConfig,
            "feature_validation": FeatureValidationConfig,
            "expert_compiler": ExpertCompilerConfig,
            "planning": PlanningConfig,
            "meta_learning": MetaLearningConfig,
            "srp": SRPConfig,
            "training": TrainingConfig,
            "curiosity": CuriosityConfig,
            "subgoal": SubgoalConfig,
            "opponent_modeling": OpponentModelingConfig,
            "hierarchical": HierarchicalConfig,
            "compute_economy": ComputeEconomyConfig,
            "attention_maps": AttentionMapsConfig,
            "meta_learning_rules": MetaLearningRulesConfig,
            "governance": GovernanceConfig,
            "meta_loop": MetaLoopConfig,
            "formal_verification": FormalVerificationConfig,
            "shadow_evolution": ShadowEvolutionConfig,
            "dynamic_hyperparams": DynamicHyperparamsConfig,
            "reasoning": ReasoningConfig,
            "mechanic_discovery": MechanicDiscoveryConfig,
            "autonomous_specialization": AutonomousSpecializationConfig,
            "stability_in_the_dark": StabilityInTheDarkConfig,
        }
        
        config = cls()
        for key, value in data.items():
            if key in nested_configs and isinstance(value, dict):
                # Filter out keys that don't belong to the nested config
                valid_keys = {f.name for f in nested_configs[key].__dataclass_fields__.values()}
                filtered_value = {k: v for k, v in value.items() if k in valid_keys}
                nested_obj = nested_configs[key](**filtered_value)
                setattr(config, key, nested_obj)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        # Convert nested dataclasses to dicts for YAML serialization
        from dataclasses import asdict
        data = asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
