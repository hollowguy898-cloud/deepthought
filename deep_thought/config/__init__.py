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
