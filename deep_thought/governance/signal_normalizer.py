"""
Fix 7: Shared Signal Space Normalization.

Every metric in the system is converted into a common currency:
"expected return impact estimate". This means:

  - Pruning score = predicted delta-reward if expert removed
  - Growth score  = predicted delta-reward if expert added
  - Routing score = predicted delta-reward if expert used
  - Sparsity      = estimated impact on return of using fewer experts
  - Entropy       = estimated impact on return of routing diversity
  - Reconstruction loss = estimated impact on return of world model quality
  - Memory coherence    = estimated impact on return of memory reliability

All signals are normalized to the same scale so they can be compared
and combined. The RL objective is the single dominant objective; all
other losses are regularizers expressed as constraints on it.
"""

import torch
from typing import Dict, Optional
from collections import deque


class SignalNormalizer:
    """
    Normalizes all signals into shared "expected return impact" space.

    Each signal type has its own running statistics (mean, std).
    Normalization converts raw values into z-scores relative to
    recent history, then scales them by the estimated return
    sensitivity for that signal type.

    This implements Fix 1 (single dominant objective) by ensuring
    every subsystem's metrics are expressed in terms of how they
    affect the RL objective.
    """

    def __init__(self, history_size: int = 1000, ema_decay: float = 0.99):
        self._history_size = history_size
        self._ema_decay = ema_decay

        # Running statistics per signal type
        self._means: Dict[str, float] = {}
        self._vars: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

        # Return sensitivity estimates per signal type
        # How much does a 1-sigma change in this signal affect return?
        self._return_sensitivity: Dict[str, float] = {}

        # Recent reward history for sensitivity estimation
        self._reward_history: deque = deque(maxlen=history_size)
        self._signal_history: Dict[str, deque] = {}

    def normalize(self, signal_type: str, value: float) -> float:
        """
        Normalize a raw signal value into expected return impact.

        Args:
            signal_type: The category of signal (e.g., "utility", "sparsity",
                "entropy", "reconstruction_loss", "memory_coherence").
            value: The raw signal value.

        Returns:
            The normalized expected return impact estimate.
        """
        # Update running statistics
        self._update_stats(signal_type, value)

        mean = self._means.get(signal_type, 0.0)
        var = self._vars.get(signal_type, 1.0)
        std = max(var ** 0.5, 1e-8)

        # Z-score
        z_score = (value - mean) / std

        # Scale by return sensitivity
        sensitivity = self._return_sensitivity.get(signal_type, 1.0)

        return z_score * sensitivity

    def normalize_tensor(self, signal_type: str,
                         values: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor of signal values."""
        mean = self._means.get(signal_type, 0.0)
        var = self._vars.get(signal_type, 1.0)
        std = max(var ** 0.5, 1e-8)

        z_score = (values - mean) / std
        sensitivity = self._return_sensitivity.get(signal_type, 1.0)

        return z_score * sensitivity

    def _update_stats(self, signal_type: str, value: float):
        """Update running mean/variance for a signal type using Welford's method."""
        if signal_type not in self._counts:
            self._counts[signal_type] = 0
            self._means[signal_type] = 0.0
            self._vars[signal_type] = 1.0
            self._signal_history[signal_type] = deque(maxlen=self._history_size)

        self._counts[signal_type] += 1
        n = self._counts[signal_type]

        # EMA update
        delta = value - self._means[signal_type]
        self._means[signal_type] += (1 - self._ema_decay) * delta
        delta2 = value - self._means[signal_type]
        self._vars[signal_type] = (
            self._ema_decay * self._vars[signal_type] +
            (1 - self._ema_decay) * delta * delta2
        )

        # Track history for sensitivity estimation
        self._signal_history[signal_type].append(value)

    def update_return_sensitivity(self, signal_type: str, reward: float,
                                   signal_value: float):
        """
        Estimate how sensitive returns are to this signal type.

        Uses correlation between signal changes and reward changes.
        This calibrates the normalization so that the z-scores
        accurately reflect expected return impact.
        """
        self._reward_history.append(reward)

        if (signal_type not in self._signal_history or
                len(self._signal_history[signal_type]) < 10 or
                len(self._reward_history) < 10):
            return

        # Compute recent correlation between signal and reward
        recent_signal = list(self._signal_history[signal_type])[-50:]
        recent_reward = list(self._reward_history)[-50:]

        n = min(len(recent_signal), len(recent_reward))
        if n < 5:
            return

        recent_signal = recent_signal[-n:]
        recent_reward = recent_reward[-n:]

        # Compute covariance
        sig_mean = sum(recent_signal) / n
        rew_mean = sum(recent_reward) / n

        cov = sum(
            (s - sig_mean) * (r - rew_mean)
            for s, r in zip(recent_signal, recent_reward)
        ) / n

        sig_var = sum((s - sig_mean) ** 2 for s in recent_signal) / n
        rew_var = sum((r - rew_mean) ** 2 for r in recent_reward) / n

        if sig_var > 1e-12 and rew_var > 1e-12:
            # Sensitivity = d(reward) / d(signal) approximated by covariance / variance
            sensitivity = cov / sig_var
            # EMA update the sensitivity estimate
            old_sens = self._return_sensitivity.get(signal_type, sensitivity)
            self._return_sensitivity[signal_type] = (
                self._ema_decay * old_sens +
                (1 - self._ema_decay) * sensitivity
            )

    def set_return_sensitivity(self, signal_type: str, sensitivity: float):
        """Manually set the return sensitivity for a signal type."""
        self._return_sensitivity[signal_type] = sensitivity

    def compute_constraint_loss(self, signal_type: str, value: float,
                                 constraint_coef: float = 0.01) -> float:
        """
        Compute a constraint loss for a signal, regularized into RL objective.

        This implements Fix 1: auxiliary losses are constraints, not competing
        objectives. The constraint_coef scales how much this signal can
        influence the total loss relative to the RL objective.

        Args:
            signal_type: The signal category.
            value: The raw signal value.
            constraint_coef: How strongly this constrains the RL objective.

        Returns:
            The constraint loss value.
        """
        normalized = self.normalize(signal_type, value)
        # Only penalize deviations that hurt expected return
        # (positive normalized = helps return, negative = hurts)
        constraint = -normalized if normalized < 0 else 0.0
        return constraint_coef * abs(constraint)

    def get_stats(self) -> Dict:
        """Get normalization statistics."""
        stats = {
            "num_signal_types": len(self._means),
            "signal_types": list(self._means.keys()),
        }
        for sig_type in self._means:
            stats[f"{sig_type}_mean"] = self._means[sig_type]
            stats[f"{sig_type}_std"] = self._vars.get(sig_type, 1.0) ** 0.5
            stats[f"{sig_type}_sensitivity"] = self._return_sensitivity.get(
                sig_type, 0.0
            )
        return stats
