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
- Stability in the Dark: Capability Density binary gate with rollback
- Stability in the Dark: Dissection Layer checkpoint & rollback
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

        # ----------------------------------------------------------------
        # Stability in the Dark: Capability Density binary gate
        # ----------------------------------------------------------------
        # When a discovery leads to a Capability Density drop, the SRP
        # acts as a strict binary gate, rolling back the Dissection
        # Layer weights to the last stable state.
        self._density_before_last_discovery: float = 0.0
        self._density_rollback_active: bool = False
        self._steps_since_rollback: int = 0
        self._rollback_cooldown: int = 500  # Steps to wait after rollback
        self._density_rollback_gate: float = 0.1  # Fraction drop triggers rollback
        # Checkpoints of the Dissection Layer state for rollback
        self._dissection_checkpoints: List[Dict] = []
        self._last_stable_dissection_state: Optional[Dict] = None
        self._dissection_checkpoint_interval: int = 1000

        # Scientific method: track whether discoveries have been proven
        self._pending_discoveries: List[Dict] = []  # Unproven discoveries
        self._proven_discoveries: List[Dict] = []  # Proven discoveries
        self._scientific_method_proof_threshold: float = 0.05
    
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
    
    # ----------------------------------------------------------------
    # Stability in the Dark: Capability Density Gate
    # ----------------------------------------------------------------

    def check_discovery_impact(
        self,
        capability_density_before: float,
        capability_density_after: float,
    ) -> Tuple[bool, str]:
        """Check whether a discovery caused a Capability Density regression.

        The SRP acts as a strict binary gate: if a discovery leads to a
        Capability Density drop exceeding the rollback gate threshold, the
        discovery is rejected and the system rolls back.

        Args:
            capability_density_before: Density before the discovery.
            capability_density_after: Density after the discovery.

        Returns:
            (approved, reason) tuple.  approved=False means rollback needed.
        """
        if capability_density_before < 1e-8:
            return True, "no_baseline"

        relative_drop = (
            capability_density_before - capability_density_after
        ) / capability_density_before

        if relative_drop > self._density_rollback_gate:
            self._density_rollback_active = True
            self._steps_since_rollback = 0
            self._density_before_last_discovery = capability_density_before
            return False, f"density_drop({relative_drop:.4f} > {self._density_rollback_gate})"

        return True, "density_stable"

    def is_density_gate_active(self) -> bool:
        """Whether the density gate is currently blocking changes."""
        return self._density_rollback_active

    def tick_density_gate(self):
        """Advance the density gate cooldown.

        After a rollback, the gate stays active for
        ``_rollback_cooldown`` steps before allowing new discoveries.
        """
        if self._density_rollback_active:
            self._steps_since_rollback += 1
            if self._steps_since_rollback >= self._rollback_cooldown:
                self._density_rollback_active = False
                self._steps_since_rollback = 0

    def checkpoint_dissection_layer(self, dissection_state: Dict, step: int):
        """Save a checkpoint of the Dissection Layer state.

        This allows the SRP to roll back the Dissection Layer if a
        discovery causes a Capability Density drop.

        Args:
            dissection_state: State dict of the Dissection Layer.
            step: Current step.
        """
        checkpoint = {
            "state": deepcopy(dissection_state),
            "step": step,
            "capability_density": self._density_before_last_discovery,
        }
        self._dissection_checkpoints.append(checkpoint)
        # Keep only last 5 checkpoints
        if len(self._dissection_checkpoints) > 5:
            self._dissection_checkpoints = self._dissection_checkpoints[-5:]
        # Track last stable state
        self._last_stable_dissection_state = checkpoint

    def rollback_dissection_layer(self) -> Optional[Dict]:
        """Roll back the Dissection Layer to the last stable state.

        Returns:
            State dict to restore, or None if no checkpoint available.
        """
        if self._last_stable_dissection_state is None:
            return None
        return deepcopy(self._last_stable_dissection_state["state"])

    # ----------------------------------------------------------------
    # Stability in the Dark: Scientific Method Enforcement
    # ----------------------------------------------------------------

    def register_pending_discovery(self, discovery: Dict):
        """Register a new discovery that has not yet been proven.

        The FVE requires the Dissection Layer to prove that a new
        internal rule actually improves prediction accuracy before it
        can influence the Sparse Cognitive Graph.

        Args:
            discovery: Dict with keys: 'id', 'mechanic_tag_id',
                'prediction_accuracy_before', 'prediction_accuracy_after'.
        """
        self._pending_discoveries.append(discovery)

    def evaluate_discovery_proof(self, discovery_id: str) -> bool:
        """Evaluate whether a pending discovery has been proven.

        A discovery is proven if it improves prediction accuracy by at
        least ``_scientific_method_proof_threshold``.

        Args:
            discovery_id: ID of the discovery to evaluate.

        Returns:
            True if the discovery is proven and can influence the graph.
        """
        for disc in self._pending_discoveries:
            if disc.get("id") == discovery_id:
                acc_before = disc.get("prediction_accuracy_before", 0.0)
                acc_after = disc.get("prediction_accuracy_after", 0.0)
                if acc_before < 1e-8:
                    improvement = acc_after
                else:
                    improvement = (acc_after - acc_before) / acc_before

                if improvement >= self._scientific_method_proof_threshold:
                    self._pending_discoveries.remove(disc)
                    self._proven_discoveries.append(disc)
                    return True
                else:
                    # Not proven yet — keep in pending
                    return False
        return False

    def get_proven_discoveries(self) -> List[Dict]:
        """Get all proven discoveries that can influence the graph."""
        return self._proven_discoveries

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
            "density_gate_active": self._density_rollback_active,
            "pending_discoveries": len(self._pending_discoveries),
            "proven_discoveries": len(self._proven_discoveries),
            "dissection_checkpoints": len(self._dissection_checkpoints),
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
