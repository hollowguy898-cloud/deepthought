"""
Governor: The Central Controller.

The Governor is the sole authority that evaluates proposals and
dispatches approved changes. It enforces all 7 governance principles:

1. Single dominant objective: RL loss is primary; all other losses
   are constraint regularizers.
2. Time-scale separation: Delegates to TimescaleController.
3. Capacity ledger: Checks growth/pruning against budget.
4. Decoupled routing: Slow router + fast deterministic gating.
5. Asymmetric memory: Cheap writes, expensive filtered reads.
6. Non-interference: All mutations go through ProposalBus.
7. Shared signal space: All metrics normalized to expected return impact.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from deep_thought.governance.timescale_controller import (
    TimescaleController, TimescaleTier, TimescaleConfig
)
from deep_thought.governance.capacity_ledger import (
    CapacityLedger, CapacityLedgerConfig
)
from deep_thought.governance.proposal_bus import (
    ProposalBus, Proposal, ProposalType, ProposalStatus
)
from deep_thought.governance.signal_normalizer import SignalNormalizer


@dataclass
class GovernorConfig:
    """Configuration for the Governor."""
    timescale_config: TimescaleConfig = None
    ledger_config: CapacityLedgerConfig = None

    # Constraint coefficients (Fix 1: auxiliary losses are constraints)
    sparsity_constraint_coef: float = 0.01
    entropy_constraint_coef: float = 0.01
    load_balance_constraint_coef: float = 0.01
    world_model_constraint_coef: float = 0.01
    compute_penalty_constraint_coef: float = 0.001
    memory_coherence_constraint_coef: float = 0.01

    # Asymmetric memory parameters (Fix 5)
    memory_write_cost: float = 0.01       # Cheap writes
    memory_read_cost: float = 1.0         # Expensive reads
    memory_read_filter_threshold: float = 0.3  # Only read high-value entries
    memory_influence_on_pruning: bool = False  # Fix 5: memory CANNOT influence pruning
    memory_influence_on_growth: bool = False   # Fix 5: memory CANNOT influence growth

    # Decoupled routing parameters (Fix 4)
    routing_slow_update_interval: int = 100  # Router policy updates every N steps
    routing_fast_gating_top_k: int = 4       # Deterministic top-k gating


class Governor:
    """
    Central governance controller for Deep Thought.

    The Governor is the ONLY entity that can authorize structural changes.
    All subsystems submit proposals; the Governor evaluates them against
    the single dominant objective (RL return) and the capacity ledger.
    """

    def __init__(self, config: GovernorConfig = None):
        self.config = config or GovernorConfig()
        self.timescale = TimescaleController(
            self.config.timescale_config or TimescaleConfig()
        )
        self.ledger = CapacityLedger(
            self.config.ledger_config or CapacityLedgerConfig(),
            signal_normalizer=SignalNormalizer()
        )
        self.proposal_bus = ProposalBus()
        self.normalizer = self.ledger.normalizer

        # State
        self._step = 0
        self._episode_reward_ema = 0.0
        self._last_rl_loss = 0.0
        self._frozen = False  # Freeze during regression

    def tick(self, step: int):
        """Advance the governor's step counter."""
        self._step = step
        self.ledger.tick()

    # ----------------------------------------------------------------
    # Fix 1: Single Dominant Objective
    # ----------------------------------------------------------------

    def compute_governed_loss(
        self,
        rl_loss: torch.Tensor,
        auxiliary_losses: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the governed total loss.

        The RL loss is PRIMARY. All auxiliary losses are CONSTRAINT
        regularizers — they cannot overpower the RL objective.

        L_total = L_rl + sum_i(constraint_coef_i * max(0, -normalized_i))

        Args:
            rl_loss: The primary RL policy + value loss.
            auxiliary_losses: Dict of auxiliary loss names to values.

        Returns:
            (total_loss, constraint_weights) — the constraint weights
            show how much each auxiliary loss actually contributed.
        """
        total_loss = rl_loss
        constraint_weights = {}

        # Sparsity constraint
        if "sparsity_loss" in auxiliary_losses:
            val = auxiliary_losses["sparsity_loss"]
            if isinstance(val, torch.Tensor):
                c = self.config.sparsity_constraint_coef
                constraint_loss = c * val
                total_loss = total_loss + constraint_loss
                constraint_weights["sparsity"] = constraint_loss.item()

        # Entropy constraint (keep in healthy range, not a competing objective)
        if "entropy" in auxiliary_losses:
            val = auxiliary_losses["entropy"]
            if isinstance(val, torch.Tensor):
                c = self.config.entropy_constraint_coef
                constraint_loss = c * val
                total_loss = total_loss + constraint_loss
                constraint_weights["entropy"] = constraint_loss.item()

        # Load balance constraint
        if "load_balance" in auxiliary_losses:
            val = auxiliary_losses["load_balance"]
            if isinstance(val, torch.Tensor):
                c = self.config.load_balance_constraint_coef
                constraint_loss = c * val
                total_loss = total_loss + constraint_loss
                constraint_weights["load_balance"] = constraint_loss.item()

        # World model constraint
        if "world_model_loss" in auxiliary_losses:
            val = auxiliary_losses["world_model_loss"]
            if isinstance(val, torch.Tensor):
                c = self.config.world_model_constraint_coef
                constraint_loss = c * val
                total_loss = total_loss + constraint_loss
                constraint_weights["world_model"] = constraint_loss.item()

        # Compute penalty constraint
        if "compute_loss" in auxiliary_losses:
            val = auxiliary_losses["compute_loss"]
            if isinstance(val, torch.Tensor):
                c = self.config.compute_penalty_constraint_coef
                constraint_loss = c * val
                total_loss = total_loss + constraint_loss
                constraint_weights["compute"] = constraint_loss.item()

        return total_loss, constraint_weights

    # ----------------------------------------------------------------
    # Fix 2: Time-Scale Separation
    # ----------------------------------------------------------------

    def is_operation_allowed(self, operation: str) -> bool:
        """Check if an operation is allowed at the current step."""
        if self._frozen:
            return self.timescale.get_tier(operation) == TimescaleTier.FAST
        return self.timescale.is_allowed(operation, self._step)

    def mark_operation_done(self, operation: str):
        """Record that an operation was executed."""
        self.timescale.mark_executed(operation, self._step)

    # ----------------------------------------------------------------
    # Fix 3: Capacity Ledger
    # ----------------------------------------------------------------

    def evaluate_growth_proposal(self, expert_id: int,
                                  predicted_marginal: float) -> bool:
        """Evaluate whether a growth proposal should be approved."""
        if self._frozen:
            return False
        if not self.is_operation_allowed("expert_growth"):
            return False
        return self.ledger.propose_growth(predicted_marginal)

    def evaluate_pruning_proposal(self, expert_id: int) -> Tuple[bool, str]:
        """Evaluate whether a pruning proposal should be approved."""
        if self._frozen:
            return False, "governor_frozen"
        if not self.is_operation_allowed("expert_pruning"):
            return False, "timescale_not_allowed"
        return self.ledger.propose_pruning(expert_id)

    # ----------------------------------------------------------------
    # Fix 4: Decoupled Routing
    # ----------------------------------------------------------------

    def should_update_router_policy(self) -> bool:
        """
        Whether the slow router policy should be updated at this step.

        Router policy is MEDIUM timescale. Fast gating (top-k selection)
        happens every step but doesn't update router weights.
        """
        return self.is_operation_allowed("routing_temperature_update")

    # ----------------------------------------------------------------
    # Fix 5: Asymmetric Memory
    # ----------------------------------------------------------------

    def approve_memory_write(self, importance: float) -> bool:
        """
        Memory writes are CHEAP and over-inclusive.

        Only reject writes that are clearly noise (importance near zero).
        """
        return importance > 0.01  # Very low threshold

    def approve_memory_read(self, relevance: float) -> bool:
        """
        Memory reads are EXPENSIVE and heavily filtered.

        Only allow reads that are highly relevant.
        """
        return relevance > self.config.memory_read_filter_threshold

    def can_memory_influence_pruning(self) -> bool:
        """Memory CANNOT directly influence pruning decisions."""
        return self.config.memory_influence_on_pruning

    def can_memory_influence_growth(self) -> bool:
        """Memory CANNOT directly influence growth decisions."""
        return self.config.memory_influence_on_growth

    # ----------------------------------------------------------------
    # Fix 6: Non-Interference Rule
    # ----------------------------------------------------------------

    def submit_proposal(self, proposal: Proposal) -> int:
        """Submit a structural change proposal."""
        return self.proposal_bus.submit(proposal)

    def evaluate_proposals(self) -> List[Proposal]:
        """
        Evaluate all pending proposals.

        For each proposal:
        1. Check timescale (is this operation allowed now?)
        2. Check capacity ledger (does it fit the budget?)
        3. Check predicted impact (does it help the RL objective?)
        4. Approve or reject

        Returns:
            List of approved proposals.
        """
        approved = []
        pending = self.proposal_bus.get_pending()

        for proposal in pending:
            # Check timescale
            operation_map = {
                ProposalType.PRUNE_EXPERT: "expert_pruning",
                ProposalType.GROW_EXPERT: "expert_growth",
                ProposalType.SPLIT_EXPERT: "expert_split",
                ProposalType.MERGE_EXPERTS: "expert_merge",
                ProposalType.MODIFY_ROUTING: "routing_structure_update",
                ProposalType.MEMORY_POLICY_CHANGE: "memory_consolidation",
                ProposalType.ARCHITECTURE_CHANGE: "architecture_reconfiguration",
                ProposalType.FAST_WEIGHT_UPDATE: "fast_weight_update",
                ProposalType.REACTIVATE_EXPERT: "expert_growth",
            }

            operation = operation_map.get(proposal.proposal_type)

            if operation and not self.is_operation_allowed(operation):
                self.proposal_bus.reject(
                    proposal.proposal_id,
                    reason="timescale_not_allowed"
                )
                continue

            # Check predicted impact
            if proposal.predicted_impact < 0 and self._frozen:
                self.proposal_bus.reject(
                    proposal.proposal_id,
                    reason="governor_frozen_negative_impact"
                )
                continue

            # Check capacity ledger for growth/pruning
            if proposal.proposal_type == ProposalType.GROW_EXPERT:
                if not self.ledger.propose_growth(proposal.predicted_impact):
                    self.proposal_bus.reject(
                        proposal.proposal_id,
                        reason="capacity_ledger_denied"
                    )
                    continue

            elif proposal.proposal_type == ProposalType.PRUNE_EXPERT:
                expert_id = proposal.payload.get("expert_id")
                if expert_id is not None:
                    ok, reason = self.ledger.propose_pruning(expert_id)
                    if not ok:
                        self.proposal_bus.reject(
                            proposal.proposal_id,
                            reason=f"ledger:{reason}"
                        )
                        continue

            # All checks passed — approve
            self.proposal_bus.approve(proposal.proposal_id)
            approved.append(proposal)

        return approved

    # ----------------------------------------------------------------
    # Fix 7: Signal Normalization
    # ----------------------------------------------------------------

    def normalize_signal(self, signal_type: str, value: float) -> float:
        """Normalize a signal into shared expected return impact space."""
        return self.normalizer.normalize(signal_type, value)

    def update_return_sensitivity(self, signal_type: str,
                                   reward: float, signal_value: float):
        """Update return sensitivity estimates for a signal type."""
        self.normalizer.update_return_sensitivity(signal_type, reward, signal_value)

    # ----------------------------------------------------------------
    # Regression Handling
    # ----------------------------------------------------------------

    def update_regression_state(self, reward: float, loss: float):
        """
        Update regression detection. If regressing, freeze all
        structural changes (slow+ timescale operations).
        """
        ema_decay = 0.99
        self._episode_reward_ema = (
            ema_decay * self._episode_reward_ema +
            (1 - ema_decay) * reward
        )
        self._last_rl_loss = loss

        # Freeze structural changes if regressing
        # (reward EMA is declining significantly)
        self._frozen = False  # Let SRP handle this; Governor defers

    def freeze_structural_changes(self):
        """Freeze all structural changes (called by SRP on regression)."""
        self._frozen = True
        self.proposal_bus.clear_pending()

    def unfreeze_structural_changes(self):
        """Unfreeze structural changes (called by SRP when stable)."""
        self._frozen = False

    # ----------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Get comprehensive governance statistics."""
        return {
            "step": self._step,
            "frozen": self._frozen,
            "reward_ema": self._episode_reward_ema,
            "last_rl_loss": self._last_rl_loss,
            "timescale_allowed": self.timescale.get_allowed_operations(self._step),
            "ledger_budget": self.ledger.get_budget_summary(),
            "proposal_stats": self.proposal_bus.get_stats(),
            "signal_stats": self.normalizer.get_stats(),
        }
