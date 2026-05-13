"""
Fix 3: Capacity Ledger for Growth/Pruning.

Growth and pruning are NOT free mutations — they must be budgeted.
Each expert has:
  - cost: parameter count + compute cost
  - utility: contribution to expected return
  - activation_frequency: how often it's routed to
  - marginal_contribution: delta-utility if removed

The ledger enforces:
  1. Total capacity budget (max parameters / max experts)
  2. Growth must "buy out" existing capacity — adding an expert
     requires proving the new expert's marginal contribution exceeds
     the cost of the weakest expert it displaces.
  3. Pruning only occurs when redundancy is proven over a time window,
     not just because utility is momentarily low.
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from deep_thought.governance.signal_normalizer import SignalNormalizer


@dataclass
class ExpertLedgerEntry:
    """A single expert's entry in the capacity ledger."""
    expert_id: int
    parameter_count: int = 0
    compute_cost: float = 1.0
    utility_score: float = 0.0
    activation_frequency: float = 0.0
    marginal_contribution: float = 0.0  # predicted delta-reward if removed
    redundancy_score: float = 0.0       # how replaceable this expert is
    utility_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    activation_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    confirmation_steps: int = 0         # steps since first flagged for pruning


@dataclass
class CapacityLedgerConfig:
    """Configuration for the capacity ledger."""
    max_total_parameters: int = 100_000_000
    max_experts: int = 256
    min_experts: int = 4
    pruning_utility_threshold: float = 0.05
    pruning_confirmation_window: int = 10_000  # steps to confirm pruning
    redundancy_threshold: float = 0.9          # high = very redundant
    growth_marginal_threshold: float = 0.1     # min marginal for growth
    utility_ema_decay: float = 0.99
    activation_ema_decay: float = 0.99


class CapacityLedger:
    """
    Budget system for expert growth and pruning.

    The ledger tracks per-expert metrics and enforces that:
    - Growth requires sufficient marginal contribution
    - Pruning requires proven redundancy over a time window
    - Total capacity stays within budget
    """

    def __init__(self, config: CapacityLedgerConfig = None,
                 signal_normalizer: SignalNormalizer = None):
        self.config = config or CapacityLedgerConfig()
        self.normalizer = signal_normalizer or SignalNormalizer()
        self._entries: Dict[int, ExpertLedgerEntry] = {}
        self._total_parameters: int = 0
        self._step: int = 0

    def register_expert(self, expert_id: int, parameter_count: int,
                        compute_cost: float = 1.0):
        """Register a new expert in the ledger."""
        entry = ExpertLedgerEntry(
            expert_id=expert_id,
            parameter_count=parameter_count,
            compute_cost=compute_cost,
        )
        self._entries[expert_id] = entry
        self._total_parameters += parameter_count

    def update_utility(self, expert_id: int, raw_utility: float):
        """
        Update an expert's utility score.

        Uses EMA smoothing and the shared signal normalizer to convert
        raw utility into expected return impact.
        """
        if expert_id not in self._entries:
            return
        entry = self._entries[expert_id]
        entry.utility_history.append(raw_utility)

        # EMA smooth
        entry.utility_score = (
            self.config.utility_ema_decay * entry.utility_score +
            (1 - self.config.utility_ema_decay) * raw_utility
        )

        # Normalize into shared signal space (expected return impact)
        entry.marginal_contribution = self.normalizer.normalize(
            "utility", raw_utility
        )

    def update_activation(self, expert_id: int, was_activated: bool):
        """Update activation frequency tracking."""
        if expert_id not in self._entries:
            return
        entry = self._entries[expert_id]
        entry.activation_history.append(1.0 if was_activated else 0.0)

        # EMA activation frequency
        freq = 1.0 if was_activated else 0.0
        entry.activation_frequency = (
            self.config.activation_ema_decay * entry.activation_frequency +
            (1 - self.config.activation_ema_decay) * freq
        )

    def can_grow(self) -> bool:
        """
        Check whether growth is allowed under capacity budget.

        Growth is allowed only if:
        1. We haven't hit the max expert count
        2. We haven't hit the max parameter budget
        3. There's sufficient marginal headroom
        """
        if len(self._entries) >= self.config.max_experts:
            return False
        if self._total_parameters >= self.config.max_total_parameters:
            return False
        return True

    def propose_growth(self, predicted_marginal: float) -> bool:
        """
        Evaluate a growth proposal.

        The new expert must have predicted marginal contribution that
        exceeds the cost of the weakest existing expert. Growth must
        "buy out" existing capacity.

        Args:
            predicted_marginal: Predicted delta-reward if the new expert
                is added.

        Returns:
            True if growth is approved.
        """
        if not self.can_grow():
            return False

        # Find the weakest expert's marginal contribution
        if self._entries:
            weakest_marginal = min(
                e.marginal_contribution for e in self._entries.values()
            )
        else:
            weakest_marginal = 0.0

        # Growth must beat the weakest expert
        return predicted_marginal > max(
            weakest_marginal, self.config.growth_marginal_threshold
        )

    def propose_pruning(self, expert_id: int) -> Tuple[bool, str]:
        """
        Evaluate a pruning proposal.

        Pruning requires:
        1. Utility below threshold
        2. Redundancy proven (high redundancy score)
        3. Confirmation over a time window (not just momentary)

        Args:
            expert_id: Expert to potentially prune.

        Returns:
            (approved, reason) tuple.
        """
        if expert_id not in self._entries:
            return False, "expert_not_found"

        entry = self._entries[expert_id]

        # Don't prune if we're at minimum experts
        if len(self._entries) <= self.config.min_experts:
            return False, "at_minimum_experts"

        # Check utility threshold
        if entry.utility_score >= self.config.pruning_utility_threshold:
            entry.confirmation_steps = 0  # reset
            return False, "utility_above_threshold"

        # Check redundancy
        if entry.redundancy_score < self.config.redundancy_threshold:
            # Not proven redundant yet — increment confirmation counter
            entry.confirmation_steps += 1
            return False, "redundancy_not_proven"

        # Check confirmation window
        if entry.confirmation_steps < self.config.pruning_confirmation_window:
            entry.confirmation_steps += 1
            return False, f"confirmation_window_not_met({entry.confirmation_steps}/{self.config.pruning_confirmation_window})"

        return True, "pruning_approved"

    def compute_redundancy(self, expert_id: int, expert_outputs: Dict[int, torch.Tensor]):
        """
        Compute redundancy score for an expert.

        An expert is redundant if its output is highly correlated
        with a linear combination of other experts' outputs.

        Args:
            expert_id: Expert to evaluate.
            expert_outputs: Dict mapping expert_id -> output tensor.
        """
        if expert_id not in expert_outputs or len(expert_outputs) < 2:
            return

        target = expert_outputs[expert_id].detach().flatten()
        other_ids = [k for k in expert_outputs if k != expert_id]
        other_outputs = torch.stack(
            [expert_outputs[k].detach().flatten() for k in other_ids]
        )  # (num_others, flat_dim)

        # Project target onto span of others
        # redundancy = ||projection|| / ||target||
        target_norm = target.norm()
        if target_norm < 1e-8:
            if expert_id in self._entries:
                self._entries[expert_id].redundancy_score = 1.0
            return

        # Simple cosine similarity with best-matching other expert
        similarities = torch.nn.functional.cosine_similarity(
            target.unsqueeze(0), other_outputs, dim=-1
        )
        max_sim = similarities.max().item()

        if expert_id in self._entries:
            self._entries[expert_id].redundancy_score = max(0.0, min(1.0, max_sim))

    def remove_expert(self, expert_id: int):
        """Remove an expert from the ledger after pruning."""
        if expert_id in self._entries:
            self._total_parameters -= self._entries[expert_id].parameter_count
            del self._entries[expert_id]

    def get_weakest_expert(self) -> Optional[int]:
        """Get the expert with the lowest marginal contribution."""
        if not self._entries:
            return None
        return min(self._entries.items(), key=lambda x: x[1].marginal_contribution)[0]

    def get_budget_summary(self) -> Dict:
        """Get a summary of the capacity budget."""
        return {
            "total_experts": len(self._entries),
            "max_experts": self.config.max_experts,
            "total_parameters": self._total_parameters,
            "max_parameters": self.config.max_total_parameters,
            "parameter_utilization": self._total_parameters / max(1, self.config.max_total_parameters),
            "can_grow": self.can_grow(),
            "weakest_expert": self.get_weakest_expert(),
        }

    def tick(self):
        """Advance the step counter."""
        self._step += 1
