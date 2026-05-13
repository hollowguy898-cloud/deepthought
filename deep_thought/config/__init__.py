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
    load_balance_loss_coef: float = 0.01
    entropy_coef: float = 0.01
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
    importance_threshold: float = 0.5
    
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
    drift_threshold: float = 0.05
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
class TrainingConfig:
    """Training configuration."""
    algorithm: str = "ppo"  # ppo, sac, impala
    learning_rate: float = 3e-4
    batch_size: int = 256
    rollout_length: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    ppo_epochs: int = 4
    
    # World model loss coefficients
    world_model_loss_coef: float = 0.5
    compute_penalty_coef: float = 0.001
    
    # Pruning and growth
    prune_interval: int = 10000
    growth_interval: int = 5000
    utility_ema_alpha: float = 0.99
    dormant_threshold: float = 0.15
    delete_threshold: float = 0.05
    dormant_confirmation_steps: int = 100000
    delete_confirmation_steps: int = 1000000


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
