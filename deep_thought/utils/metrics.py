"""
Metrics tracking utilities for Deep Thought.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class MetricsTracker:
    """
    Track and aggregate training metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def update(self, metrics: Dict[str, float]):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.window_size)
            self.metrics[key].append(value)
    
    def add_episode(self, reward: float, length: int):
        """
        Add episode statistics.
        
        Args:
            reward: Total episode reward
            length: Episode length
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get mean value of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        return np.mean(self.metrics[name])
    
    def get_episode_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Get episode statistics.
        
        Args:
            window: Window size for averaging
            
        Returns:
            stats: Dictionary of episode statistics
        """
        if len(self.episode_rewards) == 0:
            return {"mean_reward": 0.0, "mean_length": 0.0}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        
        return {
            "mean_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "max_reward": np.max(recent_rewards),
            "min_reward": np.min(recent_rewards),
            "mean_length": np.mean(recent_lengths),
        }
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all current metric means."""
        result = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                result[key] = np.mean(values)
        return result
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
