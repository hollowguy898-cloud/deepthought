"""Learning systems for Deep Thought."""

from deep_thought.learning.feature_validation import FeatureValidationEngine
from deep_thought.learning.meta_learning import MetaLearningLayer
from deep_thought.learning.fast_weights import FastWeightMemory
from deep_thought.learning.formal_verification import (
    FormalVerificationConfig,
    FormalVerificationLayer,
    EntropyRegulator,
)
from deep_thought.learning.shadow_evolution import (
    ShadowEvolutionConfig,
    ShadowEvolutionEngine,
    ShadowMutator,
)
from deep_thought.learning.dynamic_hyperparams import (
    DynamicHyperparamsConfig,
    DynamicHyperparamController,
    MetaController,
    VolatilityDetector,
)

__all__ = [
    "FeatureValidationEngine",
    "MetaLearningLayer",
    "FastWeightMemory",
    "FormalVerificationConfig",
    "FormalVerificationLayer",
    "EntropyRegulator",
    "ShadowEvolutionConfig",
    "ShadowEvolutionEngine",
    "ShadowMutator",
    "DynamicHyperparamsConfig",
    "DynamicHyperparamController",
    "MetaController",
    "VolatilityDetector",
]
