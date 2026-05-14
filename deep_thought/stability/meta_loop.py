"""
Stable Self-Improvement Component 1: Close the Meta-Loop on Capability Density.

Treats Capability Density as the primary reward signal (R_meta) for the
Context Inference layer, turning architectural changes (pruning, growth,
routing) into a high-level RL problem.

Key ideas:
  - R_meta = Capability_Density = performance / parameter_count
  - Architectural changes are actions in a meta-RL problem
  - SRP vetoes any "improvement" that increases performance at the cost
    of a disproportionate drop in Capability Density
  - Conservative exploration: meta-learning rate is much lower than task RL rate
  - Density buffer tracks history to detect regressions

Stability Guardrail: The SRP module is extended with a Capability Density
gate that vetoes architectural changes whenever:

    delta_performance / delta_params < density_threshold

i.e. the marginal return on new parameters is too low.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class MetaLoopConfig:
    """Configuration for the Capability Density Meta-Loop.

    Attributes:
        use_meta_loop: Whether to enable the meta-loop.
        density_reward_coef: Coefficient for the density reward in the
            meta-objective.  Higher values make the system more aggressive
            about parameter efficiency.
        density_regression_threshold: If capability density drops by more
            than this fraction relative to its running maximum, the system
            is considered to be regressing and architectural changes are
            frozen.
        meta_lr: Learning rate for the meta-optimizer that proposes
            architectural changes.  Should be much lower than the task RL
            rate to ensure conservative exploration.
        meta_action_dim: Dimensionality of the meta-action embedding.
        history_length: Number of recent density observations to keep
            for trend detection.
        min_density_improvement: Minimum relative improvement in density
            required to approve an architectural change.  Prevents
            the system from making changes that are neutral or marginally
            negative for density.
        density_ema_decay: EMA decay for the running density tracker.
            Closer to 1.0 = more stable / longer memory.
    """
    use_meta_loop: bool = True
    density_reward_coef: float = 0.1
    density_regression_threshold: float = 0.15
    meta_lr: float = 1e-5
    meta_action_dim: int = 64
    history_length: int = 200
    min_density_improvement: float = 0.01
    density_ema_decay: float = 0.999


class CapabilityDensityTracker:
    """Tracks capability density over time and detects regressions.

    Capability Density = mean_expert_utility / total_active_param_count

    This is the core metric for the meta-loop.  A declining density
    means the system is adding parameters that don't contribute
    proportionally — the hallmark of neuron explosion.
    """

    def __init__(self, config: MetaLoopConfig):
        self.config = config
        self._density_history: deque = deque(maxlen=config.history_length)
        self._density_ema: float = 0.0
        self._density_max: float = 0.0
        self._step: int = 0

    def update(self, density: float) -> Dict:
        """Record a new density observation and detect regression.

        Args:
            density: Current capability density value.

        Returns:
            Dict with regression status and trend information.
        """
        self._step += 1
        self._density_history.append(density)

        # EMA update
        if self._step == 1:
            self._density_ema = density
        else:
            alpha = 1.0 - self.config.density_ema_decay
            self._density_ema = self.config.density_ema_decay * self._density_ema + alpha * density

        # Track maximum
        self._density_max = max(self._density_max, density)

        # Regression detection: density has dropped significantly from max
        is_regressing = False
        if self._density_max > 1e-8:
            relative_drop = (self._density_max - density) / self._density_max
            is_regressing = relative_drop > self.config.density_regression_threshold

        # Trend detection: is density increasing or decreasing?
        trend = 0.0
        if len(self._density_history) >= 20:
            recent = list(self._density_history)[-20:]
            early_mean = sum(recent[:10]) / 10.0
            late_mean = sum(recent[10:]) / 10.0
            if early_mean > 1e-8:
                trend = (late_mean - early_mean) / abs(early_mean)

        return {
            "density": density,
            "density_ema": self._density_ema,
            "density_max": self._density_max,
            "is_regressing": is_regressing,
            "trend": trend,
            "step": self._step,
        }

    def should_approve_change(
        self,
        current_density: float,
        predicted_density_after: float,
    ) -> Tuple[bool, str]:
        """Decide whether an architectural change should be approved.

        The change is approved only if it improves (or at least doesn't
        significantly hurt) capability density.

        Args:
            current_density: Current capability density.
            predicted_density_after: Predicted density after the change.

        Returns:
            (approved, reason) tuple.
        """
        if current_density < 1e-8:
            # No baseline yet — allow changes cautiously
            return True, "no_baseline"

        relative_change = (predicted_density_after - current_density) / current_density

        if relative_change < -self.config.min_density_improvement:
            return False, f"density_drop({relative_change:.4f})"

        if relative_change < self.config.min_density_improvement:
            return False, f"insufficient_improvement({relative_change:.4f})"

        return True, "density_improved"

    def get_stats(self) -> Dict:
        """Return tracking statistics."""
        return {
            "density_ema": self._density_ema,
            "density_max": self._density_max,
            "history_len": len(self._density_history),
            "step": self._step,
        }


class MetaActionNetwork(nn.Module):
    """Proposes architectural actions based on the current meta-state.

    The meta-state encodes the current health of the system (density,
    expert count, routing entropy, etc.).  The network outputs a
    probability distribution over meta-actions:
      0: no-op (safe default)
      1: prune weakest expert
      2: grow new expert
      3: adjust routing temperature
      4: trigger memory consolidation

    The meta-optimizer (external) updates this network using R_meta
    as the reward signal.
    """

    # Meta-action constants
    NO_OP = 0
    PRUNE_WEAKEST = 1
    GROW_NEW = 2
    ADJUST_ROUTING = 3
    TRIGGER_CONSOLIDATION = 4
    NUM_META_ACTIONS = 5

    def __init__(self, config: MetaLoopConfig, state_dim: int = 32):
        super().__init__()
        self.config = config
        self.state_dim = state_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.meta_action_dim),
            nn.SiLU(),
            nn.Linear(config.meta_action_dim, config.meta_action_dim),
        )

        # Action head
        self.action_head = nn.Linear(config.meta_action_dim, self.NUM_META_ACTIONS)

        # Value head (for meta-RL)
        self.value_head = nn.Linear(config.meta_action_dim, 1)

    def forward(
        self, meta_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Propose a meta-action.

        Args:
            meta_state: Encoded system state (batch, state_dim).

        Returns:
            action_probs: Probability distribution over meta-actions.
            value: Predicted value of the current meta-state.
            info: Additional information.
        """
        h = self.state_encoder(meta_state)
        action_logits = self.action_head(h)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.value_head(h)

        info = {
            "action_logits": action_logits,
            "meta_state_encoded": h,
        }

        return action_probs, value, info

    def encode_meta_state(
        self,
        density: float,
        num_active_experts: int,
        max_experts: int,
        routing_entropy: float,
        mean_utility: float,
        memory_utilization: float,
        compute_budget_used: float,
    ) -> torch.Tensor:
        """Encode the current system state into a meta-state vector.

        Args:
            density: Current capability density.
            num_active_experts: Number of currently active experts.
            max_experts: Maximum allowed experts.
            routing_entropy: Current routing entropy.
            mean_utility: Mean expert utility.
            memory_utilization: Memory usage fraction [0, 1].
            compute_budget_used: Compute budget used fraction [0, 1].

        Returns:
            meta_state: (1, state_dim) tensor.
        """
        raw = torch.tensor([[
            density,
            num_active_experts / max(1, max_experts),
            min(routing_entropy / 5.0, 1.0),  # normalized
            mean_utility,
            memory_utilization,
            compute_budget_used,
        ]], dtype=torch.float32)

        # Pad or project to state_dim
        if raw.size(-1) < self.state_dim:
            padding = torch.zeros(1, self.state_dim - raw.size(-1))
            raw = torch.cat([raw, padding], dim=-1)
        elif raw.size(-1) > self.state_dim:
            raw = raw[:, :self.state_dim]

        return raw


class MetaLoopController(nn.Module):
    """Top-level controller for the Capability Density Meta-Loop.

    Orchestrates:
      1. Density tracking and regression detection
      2. Meta-action proposal and approval
      3. Meta-reward computation (R_meta = capability_density)
      4. Conservative exploration vs exploitation balance

    Usage:
      controller = MetaLoopController(config)
      # Each step:
      obs = controller.observe(density, num_experts, ...)
      should_freeze = controller.should_freeze_architecture()
      action, value, info = controller.propose_action(obs)
      # After executing (or not) the action:
      controller.record_reward(new_density)
    """

    def __init__(self, config: MetaLoopConfig, state_dim: int = 32):
        super().__init__()
        self.config = config
        self.tracker = CapabilityDensityTracker(config)
        self.action_net = MetaActionNetwork(config, state_dim=state_dim)
        self.meta_optimizer = torch.optim.Adam(
            self.action_net.parameters(), lr=config.meta_lr
        )
        self._prev_density: float = 0.0
        self._frozen: bool = False
        self._last_action: int = MetaActionNetwork.NO_OP
        self._cumulative_meta_reward: float = 0.0
        self._last_log_prob: Optional[torch.Tensor] = None
        self._last_value: Optional[torch.Tensor] = None
        self._step: int = 0

    def observe(
        self,
        density: float,
        num_active_experts: int,
        max_experts: int,
        routing_entropy: float,
        mean_utility: float,
        memory_utilization: float = 0.5,
        compute_budget_used: float = 0.5,
    ) -> Dict:
        """Record current system state and detect regressions.

        Returns:
            Observation dict with regression status and trends.
        """
        self._step += 1
        self._prev_density = density

        obs = self.tracker.update(density)
        if obs["is_regressing"]:
            self._frozen = True
        elif obs["trend"] > 0:
            # Density is improving — safe to unfreeze
            self._frozen = False

        # Build meta-state
        meta_state = self.action_net.encode_meta_state(
            density=density,
            num_active_experts=num_active_experts,
            max_experts=max_experts,
            routing_entropy=routing_entropy,
            mean_utility=mean_utility,
            memory_utilization=memory_utilization,
            compute_budget_used=compute_budget_used,
        )
        obs["meta_state"] = meta_state
        obs["frozen"] = self._frozen

        return obs

    def propose_action(self, obs: Dict) -> Tuple[int, float, Dict]:
        """Propose a meta-action based on the current observation.

        If the architecture is frozen (regression detected), always
        returns NO_OP.

        Args:
            obs: Observation dict from self.observe().

        Returns:
            action: Integer meta-action code.
            value: Predicted value of the meta-state.
            info: Additional information.
        """
        if self._frozen:
            return MetaActionNetwork.NO_OP, 0.0, {"reason": "architecture_frozen"}

        meta_state = obs.get("meta_state")
        if meta_state is None:
            return MetaActionNetwork.NO_OP, 0.0, {"reason": "no_meta_state"}

        action_probs, value, info = self.action_net(meta_state)

        # Epsilon-greedy exploration: with probability epsilon, sample
        # randomly from the action distribution instead of taking argmax.
        epsilon = 0.1
        if torch.rand(1).item() < epsilon:
            action = torch.multinomial(action_probs, 1).squeeze(-1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action] + 1e-8)
        else:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action] + 1e-8)
        self._last_action = action

        # Store for REINFORCE gradient update in record_reward
        self._last_log_prob = log_prob
        self._last_value = value

        info["action_probs"] = action_probs.detach()
        info["chosen_action"] = action

        return action, value.item(), info

    def record_reward(self, new_density: float):
        """Record the meta-reward after an action is taken and perform a
        gradient update step.

        R_meta = capability_density (directly).
        A positive meta-reward means the system is becoming more efficient.
        """
        meta_reward = new_density
        self._cumulative_meta_reward += meta_reward

        # Perform meta-gradient update using REINFORCE with baseline
        if self._last_log_prob is not None and self._last_value is not None:
            advantage = meta_reward - self._last_value.detach().item()
            policy_loss = -self._last_log_prob * advantage
            value_loss = F.mse_loss(self._last_value, torch.tensor([meta_reward]))
            total_loss = policy_loss + 0.5 * value_loss

            self.meta_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.action_net.parameters(), 1.0)
            self.meta_optimizer.step()

            self._last_log_prob = None
            self._last_value = None

    def should_freeze_architecture(self) -> bool:
        """Whether architectural changes should be frozen."""
        return self._frozen

    def should_approve_change(
        self, current_density: float, predicted_density_after: float
    ) -> Tuple[bool, str]:
        """Delegate to the tracker for density-based approval."""
        return self.tracker.should_approve_change(
            current_density, predicted_density_after
        )

    def get_meta_reward(self) -> float:
        """Get the latest meta-reward (current density)."""
        return self._prev_density

    def get_stats(self) -> Dict:
        """Return controller statistics."""
        tracker_stats = self.tracker.get_stats()
        return {
            **tracker_stats,
            "frozen": self._frozen,
            "last_action": self._last_action,
            "cumulative_meta_reward": self._cumulative_meta_reward,
            "meta_step": self._step,
        }

    def reset(self):
        """Reset the controller state."""
        self._prev_density = 0.0
        self._frozen = False
        self._last_action = MetaActionNetwork.NO_OP
        self._cumulative_meta_reward = 0.0
        self._last_log_prob = None
        self._last_value = None
        self._step = 0
        self.tracker = CapabilityDensityTracker(self.config)
