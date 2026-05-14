"""
Plan memory for Deep Thought.

Stores successful and failed plans for reuse and learning.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from dataclasses import dataclass

from deep_thought.config import PlanningConfig


@dataclass
class StoredPlan:
    """A stored plan with expert sequence."""
    expert_sequence: List[int]
    duration: List[int]
    reward: float
    success: bool
    context_hash: int
    usage_count: int = 0


class PlanMemory(nn.Module):
    """
    Memory for storing and retrieving plans.
    
    Stores:
    - Successful plans
    - Failed plans
    - Partially successful trajectories
    
    Enables plan reuse and learning from experience.
    """
    
    def __init__(self, config: PlanningConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        self.capacity = config.plan_memory_size
        self.plans: List[StoredPlan] = []
    
    def store_plan(
        self,
        expert_sequence: List[int],
        duration: List[int],
        reward: float,
        success: bool,
        context_hash: int
    ):
        """
        Store a plan in memory.
        
        Args:
            expert_sequence: Sequence of expert IDs
            duration: Duration for each expert
            reward: Total reward
            success: Whether plan succeeded
            context_hash: Hash of context for similarity
        """
        plan = StoredPlan(
            expert_sequence=expert_sequence,
            duration=duration,
            reward=reward,
            success=success,
            context_hash=context_hash,
        )
        
        plan.usage_count += 1
        self.plans.append(plan)
        
        # Evict if over capacity
        if len(self.plans) > self.capacity:
            self._evict()
    
    def retrieve_plan(
        self,
        context_hash: int,
        k: int = 5
    ) -> List[StoredPlan]:
        """
        Retrieve similar plans.
        
        Args:
            context_hash: Current context hash
            k: Number of plans to retrieve
            
        Returns:
            Retrieved plans
        """
        # Find plans with similar context.
        # WARNING: Integer hash proximity is NOT semantic similarity — two
        # contexts that are semantically unrelated can have numerically
        # close hashes due to hash collisions.  This filter should be
        # replaced with an embedding-based similarity measure for
        # production use.
        similar = [
            p for p in self.plans
            if abs(p.context_hash - context_hash) < 100
        ]
        
        # Increment usage_count for retrieved plans
        for p in similar[:k]:
            p.usage_count += 1

        # Sort by reward and success
        similar.sort(
            key=lambda p: (p.success, p.reward),
            reverse=True
        )
        
        return similar[:k]
    
    def get_successful_plans(self) -> List[StoredPlan]:
        """Get all successful plans."""
        return [p for p in self.plans if p.success]
    
    def _evict(self):
        """Evict worst plans."""
        # Sort by reward and usage
        self.plans.sort(
            key=lambda p: (p.reward, p.usage_count)
        )
        
        # Remove worst
        self.plans = self.plans[self.capacity:]
    
    def get_size(self) -> int:
        """Get current number of plans."""
        return len(self.plans)
