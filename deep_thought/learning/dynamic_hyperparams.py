"""
Stable Self-Improvement Component 4: Dynamic Hyperparameter Adaptation.

Self-improvement often breaks because learning rate, pruning threshold,
and other hyperparameters become obsolete as the model grows more complex.

Key ideas:
  - Fast Weights as Meta-Controllers: Use the Fast Adaptation Layer to
    predict optimal learning hyperparameters for the Sparse Cognitive
    Graph based on current task volatility.
  - Neuroplasticity Scaling: Automate the "Representation Warmup" phase
    so the model can trigger its own "re-warmup" if it detects a shift
    in the environment that its current experts can't handle.
  - Volatility detection: Track gradient variance and loss curvature
    to dynamically adjust learning rates and exploration.
  - Automatic warmup trigger: When the system detects a distribution
    shift (sudden increase in prediction error), it can autonomously
    enter a "re-warmup" phase that temporarily increases learning rate
    and exploration while freezing architectural changes.

This component ensures that the system's learning dynamics remain
well-calibrated even as the architecture evolves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class DynamicHyperparamsConfig:
    """Configuration for Dynamic Hyperparameter Adaptation.

    Attributes:
        use_dynamic_hyperparams: Whether to enable dynamic hyperparams.
        volatility_window: Number of recent gradient variance observations
            to use for volatility estimation.
        volatility_ema_decay: EMA decay for volatility tracking.
        lr_min: Minimum allowed learning rate.
        lr_max: Maximum allowed learning rate.
        lr_adjustment_rate: How quickly the meta-controller adjusts the
            learning rate.  Lower values = more conservative.
        pruning_threshold_min: Minimum pruning utility threshold.
        pruning_threshold_max: Maximum pruning utility threshold.
        pruning_threshold_adjustment_rate: How quickly the pruning
            threshold adapts to current conditions.
        warmup_trigger_threshold: If prediction error increases by more
            than this fraction, a warmup phase is triggered.
        warmup_duration: Number of steps the re-warmup phase lasts.
        warmup_lr_multiplier: Learning rate multiplier during warmup.
        warmup_freeze_architecture: Whether to freeze architectural
            changes during warmup (recommended: True).
        curvature_window: Number of observations for loss curvature
            estimation.
        meta_controller_hidden_dim: Hidden dimension of the meta-
            controller network.
    """
    use_dynamic_hyperparams: bool = True
    volatility_window: int = 100
    volatility_ema_decay: float = 0.99
    lr_min: float = 1e-6
    lr_max: float = 1e-2
    lr_adjustment_rate: float = 0.01
    pruning_threshold_min: float = 0.02
    pruning_threshold_max: float = 0.30
    pruning_threshold_adjustment_rate: float = 0.001
    warmup_trigger_threshold: float = 0.5
    warmup_duration: int = 1000
    warmup_lr_multiplier: float = 3.0
    warmup_freeze_architecture: bool = True
    curvature_window: int = 50
    meta_controller_hidden_dim: int = 64


class VolatilityDetector:
    """Detects gradient and loss volatility to inform hyperparameter tuning.

    Volatility is measured as the coefficient of variation (std/mean)
    of recent gradient norms and loss values.  High volatility suggests
    the learning dynamics are unstable and the learning rate should be
    reduced.  Low volatility suggests the system is in a plateau and
    the learning rate could be increased.
    """

    def __init__(self, config: DynamicHyperparamsConfig):
        super().__init__()
        self.config = config
        self._grad_norm_history: deque = deque(maxlen=config.volatility_window)
        self._loss_history: deque = deque(maxlen=config.volatility_window)
        self._prediction_error_history: deque = deque(maxlen=config.volatility_window)
        self._grad_volatility_ema: float = 0.0
        self._loss_volatility_ema: float = 0.0

    def update(self, grad_norm: float, loss: float, prediction_error: float = 0.0):
        """Record a new observation.

        Args:
            grad_norm: Current gradient norm.
            loss: Current loss value.
            prediction_error: Current prediction error.
        """
        self._grad_norm_history.append(grad_norm)
        self._loss_history.append(loss)
        self._prediction_error_history.append(prediction_error)

        # Compute volatility
        grad_vol = self._compute_volatility(list(self._grad_norm_history))
        loss_vol = self._compute_volatility(list(self._loss_history))

        # EMA update
        alpha = 1.0 - self.config.volatility_ema_decay
        self._grad_volatility_ema = (
            self.config.volatility_ema_decay * self._grad_volatility_ema + alpha * grad_vol
        )
        self._loss_volatility_ema = (
            self.config.volatility_ema_decay * self._loss_volatility_ema + alpha * loss_vol
        )

    def _compute_volatility(self, values: List[float]) -> float:
        """Compute coefficient of variation (volatility measure)."""
        if len(values) < 5:
            return 0.0
        mean = sum(values) / len(values)
        if abs(mean) < 1e-10:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance) / abs(mean)

    def get_volatility(self) -> Dict[str, float]:
        """Get current volatility measures."""
        return {
            "grad_volatility": self._grad_volatility_ema,
            "loss_volatility": self._loss_volatility_ema,
            "combined": (self._grad_volatility_ema + self._loss_volatility_ema) / 2.0,
        }

    def detect_distribution_shift(self) -> bool:
        """Detect if there has been a sudden distribution shift.

        A shift is detected if the recent prediction error is much
        higher than the historical baseline, indicating the model's
        current experts can't handle the new data.
        """
        if len(self._prediction_error_history) < 20:
            return False

        history = list(self._prediction_error_history)
        # Use a rolling window: the 10 entries just before the most recent 10
        # as baseline, rather than the very first 10 entries which become stale.
        baseline = sum(history[-20:-10]) / 10.0 if len(history) >= 20 else sum(history[:len(history)//2]) / max(1, len(history)//2)
        recent = sum(history[-10:]) / 10.0

        if baseline < 1e-10:
            return recent > 1.0

        relative_increase = (recent - baseline) / baseline
        return relative_increase > self.config.warmup_trigger_threshold


class MetaController(nn.Module):
    """Neural network that predicts optimal hyperparameters.

    Takes the current system state (volatility, density, entropy, etc.)
    and outputs recommended hyperparameters:
      - learning_rate
      - pruning_threshold
      - entropy_coef
      - exploration_bonus

    This is the "Fast Weights as Meta-Controllers" idea: the Fast
    Adaptation Layer is used to predict optimal hyperparameters for
    the Sparse Cognitive Graph based on current task volatility.
    """

    def __init__(self, config: DynamicHyperparamsConfig, input_dim: int = 8):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        h = config.meta_controller_hidden_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )

        # Output heads (one per hyperparameter)
        self.lr_head = nn.Linear(h, 1)        # learning rate (log-space)
        self.prune_head = nn.Linear(h, 1)     # pruning threshold
        self.entropy_head = nn.Linear(h, 1)   # entropy coefficient
        self.explore_head = nn.Linear(h, 1)   # exploration bonus

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict optimal hyperparameters.

        Args:
            state: System state vector (batch, input_dim) containing
                [volatility, density, entropy, util, ...].

        Returns:
            Dict of recommended hyperparameters (as tensors).
        """
        h = self.trunk(state)

        # Learning rate in log-space, then sigmoid-scaled to [lr_min, lr_max]
        lr_log = torch.sigmoid(self.lr_head(h))
        lr = self.config.lr_min + lr_log * (self.config.lr_max - self.config.lr_min)

        # Pruning threshold in [min, max]
        prune_raw = torch.sigmoid(self.prune_head(h))
        prune_thresh = (
            self.config.pruning_threshold_min +
            prune_raw * (self.config.pruning_threshold_max - self.config.pruning_threshold_min)
        )

        # Entropy coefficient (positive, bounded)
        entropy_coef = F.softplus(self.entropy_head(h)) * 0.1

        # Exploration bonus (positive, bounded)
        explore_bonus = torch.sigmoid(self.explore_head(h)) * 0.5

        return {
            "learning_rate": lr,
            "pruning_threshold": prune_thresh,
            "entropy_coef": entropy_coef,
            "exploration_bonus": explore_bonus,
        }

    @staticmethod
    def encode_state(
        grad_volatility: float = 0.0,
        loss_volatility: float = 0.0,
        capability_density: float = 0.0,
        routing_entropy: float = 1.0,
        mean_utility: float = 0.5,
        active_expert_ratio: float = 0.5,
        compute_budget_used: float = 0.5,
        warmup_phase: float = 0.0,
    ) -> torch.Tensor:
        """Encode system state into a vector for the meta-controller.

        Returns:
            state: (1, input_dim) tensor.
        """
        state = torch.tensor([[
            grad_volatility,
            loss_volatility,
            capability_density,
            min(routing_entropy / 5.0, 1.0),
            mean_utility,
            active_expert_ratio,
            compute_budget_used,
            warmup_phase,
        ]], dtype=torch.float32)
        return state


class DynamicHyperparamController(nn.Module):
    """Top-level controller for dynamic hyperparameter adaptation.

    Orchestrates:
      1. Volatility detection (gradient and loss)
      2. Distribution shift detection (warmup trigger)
      3. Meta-controller predictions (fast weight hyperparameters)
      4. Re-warmup phase management

    Usage:
        controller = DynamicHyperparamController(config)
        # Each step:
        controller.record(grad_norm, loss, pred_error)
        hyperparams = controller.get_hyperparams(density, entropy, ...)
        if controller.in_warmup():
            # Use warmup-specific settings
            ...
    """

    def __init__(self, config: DynamicHyperparamsConfig):
        super().__init__()
        self.config = config
        self.volatility = VolatilityDetector(config)
        self.meta_controller = MetaController(config)
        self._in_warmup: bool = False
        self._warmup_steps_remaining: int = 0
        self._step: int = 0

        # Current hyperparameters (with defaults)
        self._current_lr: float = 3e-4
        self._current_pruning_threshold: float = 0.15
        self._current_entropy_coef: float = 0.01
        self._current_exploration_bonus: float = 0.0

        # Previous prediction error for shift detection
        self._prev_prediction_error_ema: float = 0.0

    def record(
        self,
        grad_norm: float,
        loss: float,
        prediction_error: float = 0.0,
    ):
        """Record a new observation for volatility tracking.

        Also checks for distribution shifts that might trigger a
        re-warmup phase.

        Args:
            grad_norm: Current gradient norm.
            loss: Current loss value.
            prediction_error: Current prediction error.
        """
        self._step += 1
        self.volatility.update(grad_norm, loss, prediction_error)

        # Check for distribution shift
        if not self._in_warmup:
            if self.volatility.detect_distribution_shift():
                self._trigger_warmup()

        # Update warmup counter
        if self._in_warmup:
            self._warmup_steps_remaining -= 1
            if self._warmup_steps_remaining <= 0:
                self._end_warmup()

    def get_hyperparams(
        self,
        capability_density: float = 0.0,
        routing_entropy: float = 1.0,
        mean_utility: float = 0.5,
        active_expert_count: int = 4,
        max_experts: int = 64,
        compute_budget_used: float = 0.5,
    ) -> Dict[str, float]:
        """Get current recommended hyperparameters.

        If in warmup phase, returns warmup-specific settings.
        Otherwise, queries the meta-controller for optimal values.

        Args:
            capability_density: Current capability density.
            routing_entropy: Current routing entropy.
            mean_utility: Mean expert utility.
            active_expert_count: Number of active experts.
            max_experts: Maximum expert count.
            compute_budget_used: Compute budget utilization.

        Returns:
            Dict of hyperparameter values.
        """
        vol = self.volatility.get_volatility()

        if self._in_warmup:
            # Warmup phase: use elevated learning rate, freeze architecture
            return {
                "learning_rate": min(self._current_lr * self.config.warmup_lr_multiplier, self.config.lr_max),
                "pruning_threshold": self.config.pruning_threshold_max,
                "entropy_coef": self._current_entropy_coef * 2.0,
                "exploration_bonus": 0.5,
                "freeze_architecture": self.config.warmup_freeze_architecture,
                "warmup_phase": True,
            }

        # Query meta-controller
        state = MetaController.encode_state(
            grad_volatility=vol["grad_volatility"],
            loss_volatility=vol["loss_volatility"],
            capability_density=capability_density,
            routing_entropy=routing_entropy,
            mean_utility=mean_utility,
            active_expert_ratio=active_expert_count / max(1, max_experts),
            compute_budget_used=compute_budget_used,
            warmup_phase=0.0,
        )

        with torch.no_grad():
            predictions = self.meta_controller(state)

        # Smoothly adjust current values toward predictions
        rate = self.config.lr_adjustment_rate
        self._current_lr = (
            (1 - rate) * self._current_lr +
            rate * predictions["learning_rate"].item()
        )
        self._current_pruning_threshold = (
            (1 - self.config.pruning_threshold_adjustment_rate) * self._current_pruning_threshold +
            self.config.pruning_threshold_adjustment_rate * predictions["pruning_threshold"].item()
        )
        self._current_entropy_coef = (
            (1 - rate) * self._current_entropy_coef +
            rate * predictions["entropy_coef"].item()
        )
        self._current_exploration_bonus = (
            (1 - rate) * self._current_exploration_bonus +
            rate * predictions["exploration_bonus"].item()
        )

        # Clamp to valid ranges
        self._current_lr = max(self.config.lr_min, min(self.config.lr_max, self._current_lr))
        self._current_pruning_threshold = max(
            self.config.pruning_threshold_min,
            min(self.config.pruning_threshold_max, self._current_pruning_threshold)
        )

        return {
            "learning_rate": self._current_lr,
            "pruning_threshold": self._current_pruning_threshold,
            "entropy_coef": self._current_entropy_coef,
            "exploration_bonus": self._current_exploration_bonus,
            "freeze_architecture": False,
            "warmup_phase": False,
        }

    def in_warmup(self) -> bool:
        """Whether the system is currently in a re-warmup phase."""
        return self._in_warmup

    def warmup_progress(self) -> float:
        """Progress of the current warmup phase [0, 1]."""
        if not self._in_warmup:
            return 1.0
        return 1.0 - (self._warmup_steps_remaining / self.config.warmup_duration)

    def _trigger_warmup(self):
        """Enter a re-warmup phase."""
        self._in_warmup = True
        self._warmup_steps_remaining = self.config.warmup_duration

    def _end_warmup(self):
        """Exit the re-warmup phase."""
        self._in_warmup = False
        self._warmup_steps_remaining = 0

    def get_stats(self) -> Dict:
        """Return controller statistics."""
        vol = self.volatility.get_volatility()
        return {
            "step": self._step,
            "in_warmup": self._in_warmup,
            "warmup_progress": self.warmup_progress(),
            "current_lr": self._current_lr,
            "current_pruning_threshold": self._current_pruning_threshold,
            "current_entropy_coef": self._current_entropy_coef,
            "current_exploration_bonus": self._current_exploration_bonus,
            "grad_volatility": vol["grad_volatility"],
            "loss_volatility": vol["loss_volatility"],
        }

    def reset(self):
        """Reset the controller state."""
        self._in_warmup = False
        self._warmup_steps_remaining = 0
        self._step = 0
        self._current_lr = 3e-4
        self._current_pruning_threshold = 0.15
        self._current_entropy_coef = 0.01
        self._current_exploration_bonus = 0.0
        self._prev_prediction_error_ema = 0.0
        self.volatility = VolatilityDetector(self.config)
