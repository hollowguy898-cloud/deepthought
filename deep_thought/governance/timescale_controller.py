"""
Fix 2: Hard Time-Scale Separation.

Enforces that different subsystems operate at different frequencies:
  - FAST (every step): RL policy, forward pass, fast weights (small magnitude only)
  - MEDIUM (every N steps): routing temperature, memory consolidation,
    expert utilization stats
  - SLOW (every episode / million steps): pruning, expert growth,
    architecture changes
  - VERY_SLOW (training phase boundary): world model updates,
    representation restructuring

No subsystem may operate outside its assigned tier. The controller
provides a single `tick(step)` method that returns which actions
are permitted at the current step.
"""

from dataclasses import dataclass, field
from typing import Dict, Set
from enum import Enum


class TimescaleTier(Enum):
    """Time-scale tiers for subsystem operations."""
    FAST = "fast"           # Every step
    MEDIUM = "medium"       # Every N steps (default 100)
    SLOW = "slow"           # Every M steps (default 10_000)
    VERY_SLOW = "very_slow" # Every P steps (default 1_000_000)


@dataclass
class TimescaleConfig:
    """Configuration for time-scale separation."""
    # Step intervals for each tier
    medium_interval: int = 100
    slow_interval: int = 10_000
    very_slow_interval: int = 1_000_000

    # Which operations belong to which tier
    fast_operations: Set[str] = field(default_factory=lambda: {
        "rl_policy_update",
        "forward_pass",
        "fast_weight_update",
        "encoder_update",
        "critic_update",
        "intrinsic_reward_compute",
        "attention_map_update",
    })

    medium_operations: Set[str] = field(default_factory=lambda: {
        "routing_temperature_update",
        "memory_consolidation",
        "expert_utility_update",
        "compute_market_clearing",
        "subgoal_proposal",
        "opponent_model_update",
        "curiosity_decay",
    })

    slow_operations: Set[str] = field(default_factory=lambda: {
        "expert_pruning",
        "expert_growth",
        "feature_validation",
        "expert_split",
        "expert_merge",
        "capacity_rebalance",
        "routing_structure_update",
    })

    very_slow_operations: Set[str] = field(default_factory=lambda: {
        "world_model_update",
        "representation_restructuring",
        "meta_optimizer_update",
        "architecture_reconfiguration",
        "semantic_memory_restructure",
    })


class TimescaleController:
    """
    Enforces hard time-scale separation between subsystems.

    Usage:
        controller = TimescaleController(config)
        if controller.is_allowed("expert_pruning", step=50000):
            agent.prune_experts()

    The controller is the sole authority on WHEN operations may occur.
    No subsystem may bypass it.
    """

    def __init__(self, config: TimescaleConfig = None):
        self.config = config or TimescaleConfig()
        self._operation_tier: Dict[str, TimescaleTier] = {}

        # Build operation -> tier mapping
        for op in self.config.fast_operations:
            self._operation_tier[op] = TimescaleTier.FAST
        for op in self.config.medium_operations:
            self._operation_tier[op] = TimescaleTier.MEDIUM
        for op in self.config.slow_operations:
            self._operation_tier[op] = TimescaleTier.SLOW
        for op in self.config.very_slow_operations:
            self._operation_tier[op] = TimescaleTier.VERY_SLOW

        # Track last execution step for each operation
        self._last_executed: Dict[str, int] = {}

    def is_allowed(self, operation: str, step: int) -> bool:
        """
        Check whether an operation is allowed at the current step.

        Args:
            operation: Name of the operation to check.
            step: Current global training step.

        Returns:
            True if the operation is permitted at this step.
        """
        tier = self._operation_tier.get(operation)
        if tier is None:
            # Unknown operations default to FAST (safest tier)
            return True

        if tier == TimescaleTier.FAST:
            return True

        last = self._last_executed.get(operation, 0)  # Default: start of training

        if tier == TimescaleTier.MEDIUM:
            return (step - last) >= self.config.medium_interval
        elif tier == TimescaleTier.SLOW:
            return (step - last) >= self.config.slow_interval
        elif tier == TimescaleTier.VERY_SLOW:
            return (step - last) >= self.config.very_slow_interval

        return False

    def mark_executed(self, operation: str, step: int):
        """
        Record that an operation was executed at this step.

        Must be called after the operation completes successfully.
        """
        self._last_executed[operation] = step

    def get_allowed_operations(self, step: int) -> Dict[TimescaleTier, Set[str]]:
        """
        Return all operations allowed at this step, grouped by tier.
        """
        allowed = {tier: set() for tier in TimescaleTier}
        for op, tier in self._operation_tier.items():
            if self.is_allowed(op, step):
                allowed[tier].add(op)
        return allowed

    def get_tier(self, operation: str) -> TimescaleTier:
        """Get the assigned tier for an operation."""
        return self._operation_tier.get(operation, TimescaleTier.FAST)

    def register_operation(self, operation: str, tier: TimescaleTier):
        """
        Register a custom operation at a specific tier.

        This allows extension without modifying the config defaults.
        """
        self._operation_tier[operation] = tier

    def enforce_fast_weight_magnitude(self, fast_weight_norm: float,
                                       max_magnitude: float = 0.1) -> bool:
        """
        FAST-tier fast weights must have small magnitude.

        Returns True if the magnitude is within acceptable bounds.
        This is a structural constraint, not a hyperparameter.
        """
        return fast_weight_norm <= max_magnitude
