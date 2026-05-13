"""Meta-learning of learning rules for Deep Thought.

This package implements learned optimisation rules: instead of hand-crafted
learning-rate schedules, the system evolves the learning process itself.
"""

from deep_thought.meta_learning_rules.meta_optimizer import (
    GradientStatistics,
    MetaLearningRule,
    MetaOptimizer,
    UpdateRuleNetwork,
)

__all__ = [
    "GradientStatistics",
    "MetaLearningRule",
    "MetaOptimizer",
    "UpdateRuleNetwork",
]
