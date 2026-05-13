"""
Performance monitoring for Deep Thought.

Tracks performance metrics, detects drift, and provides
signals for stability systems.
"""

import torch
from typing import Dict, List, Optional
from collections import deque


class PerformanceMonitor:
    """
    Monitors performance metrics for stability detection.
    
    Tracks:
    - Performance EMA
    - Performance drift
    - Multi-timescale performance
    - Expert health
    - Routing entropy
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Performance tracking
        self.rewards = deque(maxlen=window_size)
        self.losses = deque(maxlen=window_size)
        
        # EMA
        self.reward_ema = 0.0
        self.loss_ema = 0.0
        self.ema_alpha = 0.99
        
        # Drift tracking
        self.drift_history = deque(maxlen=100)
        
        # Expert health
        self.expert_utilities: Dict[int, float] = {}
        self.expert_variances: Dict[int, float] = {}
        
        # Routing entropy
        self.routing_entropies = deque(maxlen=window_size)
    
    def update(
        self,
        reward: float,
        loss: float,
        expert_utilities: Optional[Dict[int, float]] = None,
        routing_entropy: Optional[float] = None
    ):
        """
        Update monitor with new metrics.
        
        Args:
            reward: Current reward
            loss: Current loss
            expert_utilities: Expert utility scores
            routing_entropy: Routing entropy
        """
        # Update EMAs
        self.reward_ema = self.ema_alpha * self.reward_ema + (1 - self.ema_alpha) * reward
        self.loss_ema = self.ema_alpha * self.loss_ema + (1 - self.ema_alpha) * loss
        
        # Store history
        self.rewards.append(reward)
        self.losses.append(loss)
        
        if routing_entropy is not None:
            self.routing_entropies.append(routing_entropy)
        
        if expert_utilities is not None:
            self.expert_utilities = expert_utilities
    
    def compute_drift(self) -> float:
        """
        Compute performance drift.
        
        Returns:
            drift: Performance drift (negative = regression)
        """
        if len(self.rewards) < 2:
            return 0.0
        
        # Compare recent to historical
        recent = list(self.rewards)[-100:]
        historical = list(self.rewards)[:-100] if len(self.rewards) > 100 else list(self.rewards)
        
        if len(historical) == 0:
            return 0.0
        
        recent_mean = sum(recent) / len(recent)
        historical_mean = sum(historical) / len(historical)
        
        drift = recent_mean - historical_mean
        self.drift_history.append(drift)
        
        return drift
    
    def check_regression(self, threshold: float = 0.05) -> bool:
        """
        Check if performance is regressing.
        
        Args:
            threshold: Drift threshold
            
        Returns:
            is_regressing: Whether performance is regressing
        """
        drift = self.compute_drift()
        return drift < -threshold
    
    def get_short_term_performance(self) -> float:
        """Get short-term (recent) performance."""
        if len(self.rewards) == 0:
            return 0.0
        return sum(list(self.rewards)[-100:]) / min(100, len(self.rewards))
    
    def get_long_term_performance(self) -> float:
        """Get long-term (historical) performance."""
        if len(self.rewards) == 0:
            return 0.0
        return sum(self.rewards) / len(self.rewards)
    
    def get_routing_entropy_stats(self) -> Dict[str, float]:
        """Get routing entropy statistics."""
        if len(self.routing_entropies) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        entropies = list(self.routing_entropies)
        return {
            "mean": sum(entropies) / len(entropies),
            "std": (sum((e - sum(entropies)/len(entropies))**2 for e in entropies) / len(entropies))**0.5,
            "min": min(entropies),
            "max": max(entropies),
        }
    
    def get_expert_health(self) -> Dict[int, float]:
        """Get expert health scores."""
        return self.expert_utilities.copy()
    
    def get_stats(self) -> Dict:
        """Get all monitoring statistics."""
        return {
            "reward_ema": self.reward_ema,
            "loss_ema": self.loss_ema,
            "drift": self.compute_drift(),
            "is_regressing": self.check_regression(),
            "short_term_perf": self.get_short_term_performance(),
            "long_term_perf": self.get_long_term_performance(),
            "routing_entropy": self.get_routing_entropy_stats(),
            "expert_health": self.get_expert_health(),
        }
