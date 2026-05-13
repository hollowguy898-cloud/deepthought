"""
Deep Thought: Adaptive Sparse Cognitive Network for Reinforcement Learning

A novel RL architecture combining sparse neural networks, mixture-of-experts routing,
synaptic pruning, neuroplasticity, and fast adaptive memory systems.
"""

__version__ = "0.1.0"
__author__ = "Deep Thought Contributors"

from deep_thought.agent import DeepThoughtAgent
from deep_thought.config import DeepThoughtConfig

__all__ = [
    "DeepThoughtAgent",
    "DeepThoughtConfig",
    "__version__",
]
