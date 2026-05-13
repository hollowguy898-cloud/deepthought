"""
Self-Regression Prevention (SRP) for Deep Thought.

Prevents performance collapse through multi-layer safeguards:
- Performance drift monitoring
- Multi-timescale validation
- Architecture change gating
- Rollback system
- Expert health tracking
- Routing collapse detection
- Memory contamination control
- Fast-weight drift control
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import os

from deep_thought.config import SRPConfig
from deep_thought.stability.monitoring import PerformanceMonitor


class SelfRegressionPrevention(nn.Module):
    """
    Self-Regression Prevention system.
    
    Acts as an immune system for the model, preventing
    self-inflicted damage through adaptation, pruning,
    and memory updates.
    
    Inherits from nn.Module so its state is properly
    saved/loaded with model checkpoints.
    """
    
    def __init__(self, config: SRPConfig):
        super().__init__()
        self.config = config
        self.monitor = PerformanceMonitor(config.performance_window)
        
        # Checkpoint storage
        self.checkpoints: Dict[int, Dict] = {}
        self.best_checkpoint_id = None
        self.best_performance = float('-inf')
        self.checkpoint_counter = 0
        
        # Architecture change gate
        self.allow_pruning = True
        self.allow_growth = True
        self.allow_routing_changes = True
        
        # Expert health tracking
        self.expert_health: Dict[int, float] = {}
        
        # Memory contamination tracking
        self.memory_usefulness: Dict[int, float] = {}
        
        # LEVER 5: Neuron explosion detection
        # Track expert count over time to detect parameter bloat
        self._expert_count_history = []
        self._neuron_explosion_detected = False
        self._expert_count_growth_threshold = 1.5  # Flag if count grows >50% without reward improvement
    
    def update(
        self,
        reward: float,
        loss: float,
        expert_utilities: Optional[Dict[int, float]] = None,
        routing_entropy: Optional[float] = None
    ) -> Dict:
        """
        Update SRP with new metrics.
        
        Args:
            reward: Current reward
            loss: Current loss
            expert_utilities: Expert utility scores
            routing_entropy: Routing entropy
            
        Returns:
            signals: Dictionary of control signals
        """
        # Update monitor
        self.monitor.update(reward, loss, expert_utilities, routing_entropy)
        
        # Check for regression
        is_regressing = self.monitor.check_regression(self.config.drift_threshold)
        
        # LEVER 5: Detect neuron explosion (parameter bloat without reward improvement)
        if expert_utilities is not None:
            self._check_neuron_explosion(len(expert_utilities), reward)
        
        # Update architecture gate
        self._update_architecture_gate(is_regressing)
        
        # Check routing collapse
        routing_ok = self._check_routing_health()
        
        # Generate control signals
        signals = {
            "is_regressing": is_regressing,
            "allow_pruning": self.allow_pruning,
            "allow_growth": self.allow_growth and not self._neuron_explosion_detected,
            "allow_routing_changes": self.allow_routing_changes,
            "routing_ok": routing_ok,
            "should_rollback": is_regressing and self.config.rollback_on_regression,
            "neuron_explosion_detected": self._neuron_explosion_detected,
        }
        
        return signals
    
    def _check_neuron_explosion(self, current_expert_count: int, current_reward: float):
        """LEVER 5: Detect neuron explosion pattern.
        
        Neuron explosion = expert count growing significantly while
        reward is NOT improving proportionally.  This catches the
        "more neurons = faster adaptation" spiral early.
        """
        self._expert_count_history.append((current_expert_count, current_reward))
        
        # Keep only recent history
        if len(self._expert_count_history) > 100:
            self._expert_count_history = self._expert_count_history[-100:]
        
        if len(self._expert_count_history) < 20:
            self._neuron_explosion_detected = False
            return
        
        # Compare recent vs early expert counts
        early_count = sum(c for c, _ in self._expert_count_history[:10]) / 10
        recent_count = sum(c for c, _ in self._expert_count_history[-10:]) / 10
        early_reward = sum(r for _, r in self._expert_count_history[:10]) / 10
        recent_reward = sum(r for _, r in self._expert_count_history[-10:]) / 10
        
        # Neuron explosion: expert count grew >threshold but reward didn't improve proportionally
        if early_count > 0:
            count_ratio = recent_count / early_count
            reward_ratio = (recent_reward - early_reward) / (abs(early_reward) + 1e-8)
            
            # If expert count grew >50% but reward improvement < 10% of that growth
            self._neuron_explosion_detected = (
                count_ratio > self._expert_count_growth_threshold and
                reward_ratio < count_ratio * 0.1
            )
        else:
            self._neuron_explosion_detected = False
    
    def _update_architecture_gate(self, is_regressing: bool):
        """Update architecture change gate based on stability."""
        if is_regressing:
            # Freeze structural changes during regression
            self.allow_pruning = False
            self.allow_growth = False
            self.allow_routing_changes = False
        else:
            # Allow changes when stable
            self.allow_pruning = True
            self.allow_growth = True
            self.allow_routing_changes = True
    
    def _check_routing_health(self) -> bool:
        """Check if routing entropy is in healthy range."""
        stats = self.monitor.get_routing_entropy_stats()
        mean_entropy = stats["mean"]
        
        return (
            self.config.routing_entropy_min <= mean_entropy <=
            self.config.routing_entropy_max
        )
    
    def create_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        performance: float = 0.0
    ) -> int:
        """
        Create a checkpoint of the current state.
        
        Args:
            model: Model to checkpoint
            optimizer: Optimizer state (optional)
            performance: Current performance metric
            
        Returns:
            checkpoint_id: ID of created checkpoint
        """
        checkpoint_id = self.checkpoint_counter
        self.checkpoint_counter += 1
        
        checkpoint = {
            "model_state": deepcopy(model.state_dict()),
            "performance": performance,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state"] = deepcopy(optimizer.state_dict())
        
        self.checkpoints[checkpoint_id] = checkpoint
        
        # Track best checkpoint
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_checkpoint_id = checkpoint_id
        
        # Prune old checkpoints
        if len(self.checkpoints) > 10:
            oldest_id = min(self.checkpoints.keys())
            del self.checkpoints[oldest_id]
        
        return checkpoint_id
    
    def rollback(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> bool:
        """
        Rollback to best checkpoint.
        
        Args:
            model: Model to rollback
            optimizer: Optimizer to rollback (optional)
            
        Returns:
            success: Whether rollback succeeded
        """
        if self.best_checkpoint_id is None:
            return False
        
        checkpoint = self.checkpoints[self.best_checkpoint_id]
        
        # Restore model
        model.load_state_dict(checkpoint["model_state"])
        
        # Restore optimizer if provided
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        return True
    
    def track_expert_health(
        self,
        expert_id: int,
        reward_contribution: float,
        activation_utility: float,
        variance: float
    ):
        """
        Track expert health score.
        
        H = α*R + β*A - γ*variance
        
        Args:
            expert_id: Expert ID
            reward_contribution: Reward contribution
            activation_utility: Activation utility
            variance: Output variance
        """
        alpha, beta, gamma = 0.4, 0.4, 0.2
        health = (
            alpha * reward_contribution +
            beta * activation_utility -
            gamma * variance
        )
        
        self.expert_health[expert_id] = health
    
    def isolate_unhealthy_experts(self, threshold: float = 0.0) -> List[int]:
        """
        Identify unhealthy experts for isolation.
        
        Args:
            threshold: Health threshold
            
        Returns:
            unhealthy_ids: List of unhealthy expert IDs
        """
        return [
            exp_id for exp_id, health in self.expert_health.items()
            if health < threshold
        ]
    
    def check_fast_weight_drift(self, fast_weight_norm: float) -> bool:
        """
        Check if fast weights have drifted too far.
        
        Args:
            fast_weight_norm: Current fast weight norm
            
        Returns:
            is_drifting: Whether fast weights are drifting
        """
        return fast_weight_norm > self.config.max_fast_weight_norm
    
    def constrain_fast_weights(
        self,
        fast_weights: nn.Parameter
    ):
        """
        Constrain fast weight norm and apply decay.
        
        Args:
            fast_weights: Fast weight parameter
        """
        # Apply decay
        with torch.no_grad():
            fast_weights.data = fast_weights.data * self.config.fast_weight_decay
        
        # Constrain norm
        current_norm = fast_weights.norm()
        if current_norm > self.config.max_fast_weight_norm:
            with torch.no_grad():
                fast_weights.data = (
                    fast_weights.data * self.config.max_fast_weight_norm / current_norm
                )
    
    def track_memory_usefulness(self, memory_id: int, usefulness: float):
        """
        Track memory entry usefulness.
        
        Args:
            memory_id: Memory entry ID
            usefulness: Usefulness score
        """
        self.memory_usefulness[memory_id] = usefulness
    
    def prune_contaminated_memory(self, threshold: float = 0.1) -> List[int]:
        """
        Identify contaminated memory entries.
        
        Args:
            threshold: Usefulness threshold
            
        Returns:
            contaminated_ids: List of contaminated memory IDs
        """
        return [
            mem_id for mem_id, usefulness in self.memory_usefulness.items()
            if usefulness < threshold
        ]
    
    def get_stats(self) -> Dict:
        """Get SRP statistics."""
        return {
            "monitor_stats": self.monitor.get_stats(),
            "num_checkpoints": len(self.checkpoints),
            "best_checkpoint_id": self.best_checkpoint_id,
            "best_performance": self.best_performance,
            "architecture_gate": {
                "allow_pruning": self.allow_pruning,
                "allow_growth": self.allow_growth,
                "allow_routing_changes": self.allow_routing_changes,
            },
            "num_unhealthy_experts": len(self.isolate_unhealthy_experts()),
            "neuron_explosion_detected": self._neuron_explosion_detected,
        }
    
    def reset(self):
        """Reset SRP state."""
        self.monitor = PerformanceMonitor(self.config.performance_window)
        self.checkpoints = {}
        self.best_checkpoint_id = None
        self.best_performance = float('-inf')
        self.checkpoint_counter = 0
        self.allow_pruning = True
        self.allow_growth = True
        self.allow_routing_changes = True
        self.expert_health = {}
        self.memory_usefulness = {}
        # LEVER 5: Reset neuron explosion detection
        self._expert_count_history = []
        self._neuron_explosion_detected = False
