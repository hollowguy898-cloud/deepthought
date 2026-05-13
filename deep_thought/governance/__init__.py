"""
Architectural Governance Layer for Deep Thought.

Implements the 7 governance principles that turn a swarm of competing
optimizers into a single optimizer with delegated organs:

1. Single dominant objective (RL objective is primary)
2. Hard time-scale separation (fast/medium/slow/very-slow)
3. Capacity ledger for growth/pruning
4. Decoupled routing (slow router policy + fast deterministic gating)
5. Asymmetric memory read/write with firewalls
6. Non-interference rule (propose -> evaluate -> accept)
7. Shared signal space normalization (expected return impact estimate)
"""

from deep_thought.governance.timescale_controller import TimescaleController
from deep_thought.governance.capacity_ledger import CapacityLedger
from deep_thought.governance.proposal_bus import ProposalBus, Proposal, ProposalType
from deep_thought.governance.signal_normalizer import SignalNormalizer
from deep_thought.governance.governor import Governor

__all__ = [
    "TimescaleController",
    "CapacityLedger",
    "ProposalBus",
    "Proposal",
    "ProposalType",
    "SignalNormalizer",
    "Governor",
]
