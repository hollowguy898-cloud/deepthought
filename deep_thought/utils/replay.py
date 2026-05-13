"""
Replay buffer utilities for Deep Thought.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import random


class ReplayBuffer:
    """
    Simple replay buffer for experience storage.
    """
    
    def __init__(self, capacity: int, observation_shape: Tuple):
        self.capacity = capacity
        self.observation_shape = observation_shape
        
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        
        self.size = 0
        self.ptr = 0
    
    def add(
        self,
        observation: np.ndarray,
        action: float,
        reward: float,
        done: bool,
        next_observation: np.ndarray
    ):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.next_observations[self.ptr] = next_observation
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = random.sample(range(self.size), batch_size)
        
        return {
            "observations": torch.tensor(self.observations[indices], dtype=torch.float32),
            "actions": torch.tensor(self.actions[indices], dtype=torch.float32),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32),
            "next_observations": torch.tensor(self.next_observations[indices], dtype=torch.float32),
        }
    
    def __len__(self):
        return self.size
