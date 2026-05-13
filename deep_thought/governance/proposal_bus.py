"""
Fix 6: Non-Interference Rule.

Subsystems can PROPOSE changes but cannot directly MUTATE another
subsystem's state. Everything goes through:

  propose -> evaluate -> accept/reject

The ProposalBus is the central communication channel. Any structural
change (pruning, growth, routing modification, memory write policy
change) must be submitted as a Proposal. The Governor evaluates
proposals against the single dominant objective and the capacity
ledger before accepting.

This prevents cascading failures where one subsystem's "improvement"
destabilizes another.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class ProposalType(Enum):
    """Types of structural proposals."""
    PRUNE_EXPERT = "prune_expert"
    GROW_EXPERT = "grow_expert"
    SPLIT_EXPERT = "split_expert"
    MERGE_EXPERTS = "merge_experts"
    MODIFY_ROUTING = "modify_routing"
    MEMORY_POLICY_CHANGE = "memory_policy_change"
    ARCHITECTURE_CHANGE = "architecture_change"
    FAST_WEIGHT_UPDATE = "fast_weight_update"
    REACTIVATE_EXPERT = "reactivate_expert"


class ProposalStatus(Enum):
    """Status of a proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Proposal:
    """
    A proposed change to the system.

    Each proposal contains:
    - type: What kind of change
    - source: Which subsystem proposed it
    - payload: The data needed to execute the change
    - predicted_impact: Estimated delta-reward (in shared signal space)
    - cost: Capacity cost of the change
    - priority: Urgency (higher = more urgent)
    """
    proposal_type: ProposalType
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    predicted_impact: float = 0.0
    cost: float = 0.0
    priority: float = 0.5
    status: ProposalStatus = ProposalStatus.PENDING
    created_step: int = 0
    evaluated_step: Optional[int] = None
    rejection_reason: Optional[str] = None
    proposal_id: int = 0

    def __lt__(self, other):
        """For priority queue ordering — higher priority first."""
        return self.priority > other.priority


class ProposalBus:
    """
    Central communication bus for structural proposals.

    All subsystems submit proposals here. The Governor reads, evaluates,
    and dispatches approved proposals. No subsystem may directly mutate
    another subsystem's state.

    This is the enforcement mechanism for the non-interference rule.
    """

    def __init__(self, max_pending: int = 1000):
        self._pending: List[Proposal] = []
        self._history: List[Proposal] = []
        self._max_pending = max_pending
        self._proposal_counter = 0

        # Stats
        self._total_proposed = 0
        self._total_approved = 0
        self._total_rejected = 0

    def submit(self, proposal: Proposal) -> int:
        """
        Submit a proposal to the bus.

        Args:
            proposal: The proposal to submit.

        Returns:
            proposal_id: Unique ID assigned to this proposal.
        """
        proposal.proposal_id = self._proposal_counter
        self._proposal_counter += 1
        proposal.status = ProposalStatus.PENDING

        self._pending.append(proposal)
        self._total_proposed += 1

        # Prevent bus overflow
        if len(self._pending) > self._max_pending:
            # Remove lowest-priority pending proposals
            self._pending.sort(key=lambda p: p.priority, reverse=True)
            removed = self._pending[self._max_pending:]
            for p in removed:
                p.status = ProposalStatus.REJECTED
                p.rejection_reason = "bus_overflow"
                self._history.append(p)
                self._total_rejected += 1
            self._pending = self._pending[:self._max_pending]

        return proposal.proposal_id

    def get_pending(self) -> List[Proposal]:
        """Get all pending proposals, sorted by priority."""
        return sorted(self._pending, key=lambda p: p.priority, reverse=True)

    def get_pending_by_type(self, proposal_type: ProposalType) -> List[Proposal]:
        """Get pending proposals of a specific type."""
        return [
            p for p in self._pending
            if p.proposal_type == proposal_type and p.status == ProposalStatus.PENDING
        ]

    def approve(self, proposal_id: int) -> Optional[Proposal]:
        """
        Mark a proposal as approved.

        Returns the approved proposal, or None if not found.
        """
        for p in self._pending:
            if p.proposal_id == proposal_id:
                p.status = ProposalStatus.APPROVED
                self._total_approved += 1
                self._pending.remove(p)
                self._history.append(p)
                return p
        return None

    def reject(self, proposal_id: int, reason: str = "") -> Optional[Proposal]:
        """
        Mark a proposal as rejected.

        Returns the rejected proposal, or None if not found.
        """
        for p in self._pending:
            if p.proposal_id == proposal_id:
                p.status = ProposalStatus.REJECTED
                p.rejection_reason = reason
                self._total_rejected += 1
                self._pending.remove(p)
                self._history.append(p)
                return p
        return None

    def mark_executed(self, proposal_id: int) -> Optional[Proposal]:
        """Mark an approved proposal as successfully executed."""
        for p in self._history:
            if p.proposal_id == proposal_id and p.status == ProposalStatus.APPROVED:
                p.status = ProposalStatus.EXECUTED
                return p
        return None

    def mark_rolled_back(self, proposal_id: int) -> Optional[Proposal]:
        """Mark a proposal as rolled back after execution failure."""
        for p in self._history:
            if p.proposal_id == proposal_id:
                p.status = ProposalStatus.ROLLED_BACK
                return p
        return None

    def clear_pending(self):
        """Clear all pending proposals (e.g., during regression)."""
        for p in self._pending:
            p.status = ProposalStatus.REJECTED
            p.rejection_reason = "cleared_on_regression"
            self._total_rejected += 1
            self._history.append(p)
        self._pending.clear()

    def get_stats(self) -> Dict:
        """Get proposal bus statistics."""
        return {
            "pending_count": len(self._pending),
            "total_proposed": self._total_proposed,
            "total_approved": self._total_approved,
            "total_rejected": self._total_rejected,
            "approval_rate": (
                self._total_approved / max(1, self._total_proposed)
            ),
            "history_size": len(self._history),
        }
