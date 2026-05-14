"""
Compute Market — Competitive Compute Allocation for Deep Thought.

Implements a market-based mechanism where experts bid for compute resources.
The core philosophy: "Convince the system your computation is worth spending."

Architecture Overview:
    ┌─────────────────────────────────────────────────────┐
    │  ComputeMarket                                       │
    │                                                      │
    │  ┌──────────────┐    ┌──────────────────────────┐   │
    │  │ ExpertBidder │    │ ComputeAuction            │   │
    │  │ (per-expert) │───▶│ sealed-bid / vickrey      │   │
    │  │ Learns bid   │    │ Clears within energy bgt  │   │
    │  │ strategy     │    └────────────┬─────────────┘   │
    │  └──────────────┘                 │                  │
    │                                   ▼                  │
    │  ┌──────────────┐    ┌──────────────────────────┐   │
    │  │ EnergyConstr. │    │ BudgetAllocator           │   │
    │  │ Tracks global │◀───│ Converts auction results  │   │
    │  │ compute bgt   │    │ into actual compute bgt   │   │
    │  └──────────────┘    └──────────────────────────┘   │
    │                                                      │
    │  Credit System: High-value experts earn more credit   │
    │  for future bidding, creating a virtuous cycle.       │
    └─────────────────────────────────────────────────────┘

Market Flow:
    1. Each expert observes its utility, routing gate, and context.
    2. ExpertBidder networks produce bids (amount, price, expected_value).
    3. ComputeAuction clears by selecting highest-value bids within the
       total energy budget.
    4. BudgetAllocator converts winning bids into compute budgets.
    5. After computation, experts that contributed high value are credited,
       giving them more purchasing power in future auctions.
    6. The energy budget recharges slowly, enforcing compute conservation.

Supported Auction Types:
    - ``sealed_bid``: Standard pay-what-you-bid auction. Winners pay
      their offered price.
    - ``vickrey``: Second-price (Vickrey) auction. Winners pay the
      highest losing bid price, encouraging truthful bidding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from deep_thought.config import ComputeEconomyConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Bid:
    """A single bid submitted by an expert in the compute market.

    Attributes:
        expert_id: Unique identifier of the bidding expert.
        amount: How much compute the expert is requesting.
        offered_price: The price the expert is willing to pay per unit
            of compute.  In a sealed-bid auction this is what the expert
            pays if they win; in a Vickrey auction they pay the second-
            highest price instead.
        expected_value: The expert's self-estimated value contribution
            if granted the requested compute.  This is the signal the
            auction uses to prioritize bids.
    """
    expert_id: int
    amount: float           # How much compute they want
    offered_price: float    # What they're willing to pay
    expected_value: float   # Their estimated value contribution


@dataclass
class MarketInfo:
    """Aggregated market information returned after each forward pass.

    This dataclass serves as a diagnostic and logging structure that
    captures the state of the market after an auction round.

    Attributes:
        winning_bids: List of bids that won the auction.
        losing_bids: List of bids that lost the auction.
        total_energy_spent: Total energy consumed by the winning allocation.
        total_energy_remaining: Energy left in the global budget.
        auction_clearing_price: The marginal price at which the auction
            cleared (price of the last winning bid).  For Vickrey auctions
            this is the highest losing price.
        credit_balances: Current credit balance for each expert.
        compute_allocations: Final compute allocation per expert.
        bid_stats: Per-bid statistics (expert_id -> bid details).
    """
    winning_bids: List[Bid] = field(default_factory=list)
    losing_bids: List[Bid] = field(default_factory=list)
    total_energy_spent: float = 0.0
    total_energy_remaining: float = 0.0
    auction_clearing_price: float = 0.0
    credit_balances: Dict[int, float] = field(default_factory=dict)
    compute_allocations: Dict[int, float] = field(default_factory=dict)
    bid_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExpertBidder — per-expert learned bidding network
# ---------------------------------------------------------------------------

class ExpertBidder(nn.Module):
    """Learned bidding network for a single expert.

    Each expert has its own ``ExpertBidder`` that takes as input the
    expert's current utility, routing gate value, and a context vector,
    and produces a bid for compute resources.

    The network outputs three quantities:
        - **bid_amount**: How much compute the expert requests.
          Passed through a softplus to guarantee positivity.
        - **bid_price**: What the expert is willing to pay per unit.
          Passed through a softplus and clamped to ``min_bid_price``.
        - **expected_value**: The expert's self-estimated value
          contribution.  Also softplus-ensured positive.

    Args:
        latent_dim: Dimension of the context input.
        hidden_dim: Width of the hidden layer in the bidding MLP.
        min_bid_price: Minimum allowed bid price (floor).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        min_bid_price: float = 0.01,
    ):
        super().__init__()
        self.min_bid_price = min_bid_price

        # Bidding MLP: context + utility + gate → bid parameters
        # Input: [context (latent_dim), utility (1), gate (1)]
        input_dim = latent_dim + 2
        self.bidding_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Separate heads for each bid component
        self.amount_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        utility: torch.Tensor,
        gate: torch.Tensor,
        credit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce a bid given the expert's current state.

        Args:
            context: Context vector of shape ``(batch, latent_dim)``.
            utility: Expert utility score of shape ``(batch, 1)``.
            gate: Routing gate value of shape ``(batch, 1)``.
            credit: Expert's current credit balance of shape ``(batch, 1)``.

        Returns:
            bid_amount: Requested compute amount, shape ``(batch, 1)``.
            bid_price: Offered price per unit, shape ``(batch, 1)``.
            expected_value: Estimated value contribution, shape ``(batch, 1)``.
        """
        # Concatenate inputs
        x = torch.cat([context, utility, gate], dim=-1)
        h = self.bidding_network(x)

        # Bid amount: softplus ensures positivity
        bid_amount = F.softplus(self.amount_head(h))

        # Bid price: softplus with a floor at min_bid_price
        bid_price = F.softplus(self.price_head(h)) + self.min_bid_price

        # Credit cap: an expert cannot bid more than they can afford.
        # The effective price is the minimum of their bid and their credit.
        # We scale the amount by credit / price to enforce budget constraint.
        max_affordable_amount = credit / (bid_price + 1e-8)
        bid_amount = torch.min(bid_amount, max_affordable_amount.detach())

        # Expected value: softplus ensures positivity
        expected_value = F.softplus(self.value_head(h))

        return bid_amount, bid_price, expected_value


# ---------------------------------------------------------------------------
# ComputeAuction — sealed-bid and Vickrey auction mechanisms
# ---------------------------------------------------------------------------

class ComputeAuction:
    """Auction mechanism for clearing compute bids under an energy budget.

    Supports two auction types:

    - **sealed_bid** (first-price): Winners pay their own offered price.
      Simple and fast, but can encourage overbidding.
    - **vickrey** (second-price): Winners pay the highest losing bid price.
      Encourages truthful bidding because the price you pay is independent
      of your own bid.

    The auction sorts all bids by a *priority score* (expected value per
    unit of compute, scaled by offered price) and greedily selects bids
    until the energy budget is exhausted.

    Args:
        auction_type: One of ``"sealed_bid"`` or ``"vickrey"``.
        temperature: Softmax temperature for converting priority scores
            into selection probabilities when using a soft allocation.
    """

    def __init__(
        self,
        auction_type: str = "sealed_bid",
        temperature: float = 1.0,
    ):
        if auction_type not in ("sealed_bid", "vickrey"):
            raise ValueError(
                f"Unknown auction type '{auction_type}'. "
                f"Must be 'sealed_bid' or 'vickrey'."
            )
        self.auction_type = auction_type
        self.temperature = temperature

    def clear(
        self,
        bids: List[Bid],
        energy_budget: float,
    ) -> Tuple[Dict[int, float], List[Bid], List[Bid], float]:
        """Clear the auction by selecting winning bids.

        Bids are ranked by priority score, defined as::

            priority = expected_value / (amount + ε) × offered_price

        This favors experts that promise high value per unit of compute
        and are willing to pay more.  The auction greedily selects bids
        from highest to lowest priority until the cumulative compute
        amount exceeds the energy budget.

        For the **Vickrey** variant, each winner pays the price of the
        highest-priority *losing* bid rather than their own offered price.

        Args:
            bids: List of :class:`Bid` objects submitted by experts.
            energy_budget: Total compute energy available for allocation.

        Returns:
            allocations: Mapping ``expert_id → allocated_compute``.
            winning_bids: List of bids that were selected.
            losing_bids: List of bids that were not selected.
            clearing_price: The marginal price at which the auction
                cleared (price of the last winning bid for sealed-bid,
                or highest losing price for Vickrey).
        """
        if not bids:
            return {}, [], [], 0.0

        # Sort bids by priority score (descending)
        sorted_bids = sorted(
            bids,
            key=lambda b: (b.expected_value / (b.amount + 1e-8)) * b.offered_price,
            reverse=True,
        )

        allocations: Dict[int, float] = {}
        winning_bids: List[Bid] = []
        losing_bids: List[Bid] = []
        energy_remaining = energy_budget
        clearing_price = 0.0

        for bid in sorted_bids:
            cost = bid.amount * bid.offered_price
            if cost <= energy_remaining + 1e-8:
                # This bid fits within the remaining budget
                actual_amount = min(bid.amount, energy_remaining / (bid.offered_price + 1e-8))
                allocations[bid.expert_id] = actual_amount
                winning_bids.append(bid)
                energy_remaining -= actual_amount * bid.offered_price
                clearing_price = bid.offered_price
            else:
                losing_bids.append(bid)

        # For Vickrey auction: adjust the price each winner pays
        if self.auction_type == "vickrey" and losing_bids:
            vickrey_price = max(b.offered_price for b in losing_bids)
            clearing_price = vickrey_price
            # Re-compute allocations at the Vickrey price
            new_allocations: Dict[int, float] = {}
            energy_remaining = energy_budget
            for bid in winning_bids:
                effective_price = min(bid.offered_price, vickrey_price)
                actual_amount = min(
                    bid.amount,
                    energy_remaining / (effective_price + 1e-8),
                )
                new_allocations[bid.expert_id] = actual_amount
                energy_remaining -= actual_amount * effective_price
            # Clamp total allocation to energy budget
            # (Vickrey's lower effective prices can cause total raw
            # allocation to exceed the budget even though cost fits)
            total_allocated = sum(new_allocations.values())
            if total_allocated > energy_budget:
                scale = energy_budget / (total_allocated + 1e-8)
                new_allocations = {
                    eid: amt * scale for eid, amt in new_allocations.items()
                }
            allocations = new_allocations

        return allocations, winning_bids, losing_bids, clearing_price

    def soft_clear(
        self,
        bid_amounts: torch.Tensor,
        bid_prices: torch.Tensor,
        expected_values: torch.Tensor,
        energy_budget: float,
    ) -> torch.Tensor:
        """Differentiable soft auction using softmax allocation.

        Instead of a hard winner-take-all selection, this method computes
        a soft allocation proportional to each bid's priority score.  The
        result is fully differentiable, allowing gradient flow back to the
        bidding networks.

        The priority score for each expert is::

            priority_i = expected_value_i / (amount_i + ε) × price_i

        These are converted to allocation weights via softmax with
        temperature, then scaled by the energy budget.

        Args:
            bid_amounts: Tensor of shape ``(num_experts,)`` with requested
                compute amounts.
            bid_prices: Tensor of shape ``(num_experts,)`` with offered
                prices.
            expected_values: Tensor of shape ``(num_experts,)`` with
                self-estimated value contributions.
            energy_budget: Total compute energy available.

        Returns:
            allocations: Tensor of shape ``(num_experts,)`` with allocated
                compute for each expert.  Sum is bounded by energy budget.
        """
        # Compute priority scores
        priorities = expected_values / (bid_amounts + 1e-8) * bid_prices

        # Softmax over priorities (temperature-scaled)
        alloc_weights = F.softmax(priorities / self.temperature, dim=0)

        # Scale allocations: each expert gets a fraction of the budget
        # proportional to their priority weight, but capped by their
        # requested amount
        raw_allocations = alloc_weights * energy_budget
        allocations = torch.min(raw_allocations, bid_amounts)

        return allocations


# ---------------------------------------------------------------------------
# BudgetAllocator — converts auction results into compute budgets
# ---------------------------------------------------------------------------

class BudgetAllocator(nn.Module):
    """Converts raw auction allocations into normalized compute budgets.

    The allocator takes the raw compute amounts from the auction and
    produces a budget vector that:

    1. Is normalized so the total does not exceed the energy budget.
    2. Has a minimum floor allocation so every expert gets at least a
       tiny amount of compute (prevents total starvation).
    3. Applies a learnable scaling factor to adjust allocation sharpness.

    Args:
        num_experts: Total number of experts in the system.
        min_allocation: Minimum compute allocation for any expert.
            Ensures no expert is completely starved.
    """

    def __init__(
        self,
        num_experts: int,
        min_allocation: float = 0.001,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.min_allocation = min_allocation

        # Learnable scaling: controls how sharply the allocation
        # concentrates on high-priority experts vs. spreading evenly
        self.sharpness = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        raw_allocations: Dict[int, float],
        energy_budget: float,
    ) -> torch.Tensor:
        """Convert raw auction allocations into a normalized budget tensor.

        Args:
            raw_allocations: Mapping ``expert_id → allocated_compute``
                from the auction.
            energy_budget: Total compute energy budget.

        Returns:
            budgets: Tensor of shape ``(num_experts,)`` with normalized
                compute budgets.  Each entry is at least
                ``min_allocation`` and the sum is bounded by
                ``energy_budget``.
        """
        budgets = torch.full(
            (self.num_experts,), self.min_allocation, dtype=torch.float32,
            device=self.sharpness.device,
        )

        total_raw = sum(raw_allocations.values()) + self.num_experts * self.min_allocation
        scale = min(1.0, energy_budget / (total_raw + 1e-8))

        for expert_id, amount in raw_allocations.items():
            if 0 <= expert_id < self.num_experts:
                budgets[expert_id] = self.min_allocation + amount * scale

        # Apply learnable sharpness: softmax-like reweighting
        if self.sharpness.item() > 0.01:
            sharpness_val = torch.clamp(self.sharpness, min=0.01, max=10.0)
            budget_weights = F.softmax(
                torch.log(budgets + 1e-8) * sharpness_val, dim=0
            )
            budgets = budget_weights * energy_budget

        # Ensure minimum allocation
        budgets = torch.clamp(budgets, min=self.min_allocation)

        return budgets


# ---------------------------------------------------------------------------
# EnergyConstraint — global compute energy budget tracker
# ---------------------------------------------------------------------------

class EnergyConstraint:
    """Tracks and enforces a global compute energy budget.

    The energy budget represents the total amount of compute that can be
    spent across all experts in a single step.  It depletes as experts
    consume compute and recharges at a configurable rate.

    The energy constraint acts as a conservation law: the system cannot
    spend more compute than it has energy, forcing experts to compete
    for limited resources.

    Args:
        total_energy_budget: Maximum energy capacity.
        recharge_rate: How much energy is restored per recharge step.
            A rate of 1.0 means the budget fully recharges each step.
    """

    def __init__(
        self,
        total_energy_budget: float = 100.0,
        recharge_rate: float = 1.0,
    ):
        self.total_energy_budget = total_energy_budget
        self.current_energy = total_energy_budget
        self.recharge_rate = recharge_rate

    def check_budget(self, allocation: Dict[int, float]) -> bool:
        """Check whether an allocation fits within the current energy budget.

        The cost of an allocation is the sum of all allocated compute
        amounts (each unit of compute costs one unit of energy).

        Args:
            allocation: Mapping ``expert_id → allocated_compute``.

        Returns:
            True if the total allocation fits within the remaining energy,
            False otherwise.
        """
        total_cost = sum(allocation.values())
        return total_cost <= self.current_energy + 1e-8

    def consume(self, amount: float) -> float:
        """Consume energy from the budget.

        Args:
            amount: Amount of energy to consume.

        Returns:
            The actual amount consumed (may be less than requested if
            the budget is insufficient).
        """
        consumed = min(amount, self.current_energy)
        self.current_energy -= consumed
        self.current_energy = max(0.0, self.current_energy)
        return consumed

    def recharge(self, amount: Optional[float] = None) -> None:
        """Recharge the energy budget.

        If ``amount`` is ``None``, the budget recharges by
        ``recharge_rate``.  Otherwise, it recharges by the specified
        amount.  The budget is capped at ``total_energy_budget``.

        Args:
            amount: Optional explicit recharge amount.  If ``None``,
                uses ``self.recharge_rate``.
        """
        if amount is None:
            amount = self.recharge_rate
        self.current_energy = min(
            self.total_energy_budget,
            self.current_energy + amount,
        )

    def get_fraction_remaining(self) -> float:
        """Return the fraction of energy budget remaining.

        Returns:
            A float in ``[0, 1]`` representing how much energy is left
            as a fraction of the total budget.
        """
        if self.total_energy_budget <= 0:
            return 0.0
        return self.current_energy / self.total_energy_budget

    def reset(self) -> None:
        """Reset energy to full capacity."""
        self.current_energy = self.total_energy_budget


# ---------------------------------------------------------------------------
# ComputeMarket — the main module
# ---------------------------------------------------------------------------

class ComputeMarket(nn.Module):
    """Competitive market for compute allocation in Deep Thought.

    ``ComputeMarket`` implements a full market economy where experts bid
    for limited compute resources.  The market ensures that compute is
    allocated to the experts that can make the best use of it, while
    enforcing a global energy budget that prevents runaway computation.

    The market cycle each forward pass:

    1. **Bidding**: Each expert's ``ExpertBidder`` network observes the
       expert's utility, routing gate, and context, then produces a
       bid (amount, price, expected_value).
    2. **Auction**: The ``ComputeAuction`` mechanism clears the bids
       within the global energy budget, selecting the highest-priority
       bids as winners.
    3. **Allocation**: The ``BudgetAllocator`` converts auction results
       into a normalized compute budget tensor.
    4. **Credit Update**: After computation, experts that contributed
       high value receive credit, increasing their purchasing power
       for future auctions.
    5. **Energy Recharge**: The energy budget partially recharges,
       enforcing compute conservation over time.

    Args:
        config: A :class:`ComputeEconomyConfig` controlling market
            hyperparameters.
        num_experts: Number of experts participating in the market.
        latent_dim: Dimensionality of the shared latent representation.
    """

    def __init__(
        self,
        config: Optional[ComputeEconomyConfig] = None,
        num_experts: int = 128,
        latent_dim: int = 1024,
    ):
        super().__init__()
        self.config = config or ComputeEconomyConfig()
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.use_compute_market = self.config.use_compute_market

        # ── Expert bidding networks ──────────────────────────────────
        # Each expert has its own bidder that learns to estimate its
        # expected value contribution and produce competitive bids.
        self.expert_bidders = nn.ModuleList([
            ExpertBidder(
                latent_dim=latent_dim,
                hidden_dim=self.config.bidding_hidden_dim,
                min_bid_price=self.config.min_bid_price,
            )
            for _ in range(num_experts)
        ])

        # ── Auction mechanism ────────────────────────────────────────
        self.auction = ComputeAuction(
            auction_type=self.config.auction_type,
            temperature=self.config.market_temperature,
        )

        # ── Budget allocator ─────────────────────────────────────────
        self.budget_allocator = BudgetAllocator(
            num_experts=num_experts,
            min_allocation=self.config.min_bid_price,
        )

        # ── Energy constraint ────────────────────────────────────────
        self.energy_constraint = EnergyConstraint(
            total_energy_budget=self.config.total_energy_budget,
            recharge_rate=self.config.energy_recharge_rate,
        )

        # ── Credit system ────────────────────────────────────────────
        # Each expert starts with equal credit.  Credit is updated via
        # an exponential moving average when the expert contributes value.
        self.register_buffer(
            "expert_credits",
            torch.ones(num_experts, dtype=torch.float32),
        )
        self.credit_ema = self.config.credit_ema

        # ── Running statistics ───────────────────────────────────────
        # Track auction history for diagnostics and potential
        # meta-learning feedback.
        self._step_count = 0
        self._total_energy_spent = 0.0
        self._total_credit_earned = 0.0
        self._auction_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        expert_utilities: torch.Tensor,
        routing_gates: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, MarketInfo]:
        """Run one market cycle: bid → auction → allocate.

        Args:
            expert_utilities: Per-expert utility scores of shape
                ``(batch, num_experts)``.  These represent how useful
                each expert has been recently (e.g., reward contribution).
            routing_gates: Per-expert routing gate values of shape
                ``(batch, num_experts)``.  These indicate how strongly
                each expert is currently being selected by the router.
            context: Context vector of shape ``(batch, latent_dim)``.
                Represents the current state of the system (e.g.,
                encoded observation + hidden state).

        Returns:
            compute_allocations: Tensor of shape ``(num_experts,)``
                with the compute budget allocated to each expert.
            market_info: A :class:`MarketInfo` object with detailed
                auction diagnostics.
        """
        if not self.use_compute_market:
            # If the market is disabled, allocate equally
            equal_alloc = torch.ones(
                self.num_experts, device=expert_utilities.device
            ) / self.num_experts
            return equal_alloc, MarketInfo()

        batch_size = expert_utilities.size(0)
        device = expert_utilities.device

        # ── Step 1: Collect bids from all experts ────────────────────
        bids: List[Bid] = []
        bid_amounts_list: List[torch.Tensor] = []
        bid_prices_list: List[torch.Tensor] = []
        expected_values_list: List[torch.Tensor] = []

        # Aggregate utilities and gates across the batch (mean)
        mean_utilities = expert_utilities.mean(dim=0)  # (num_experts,)
        mean_gates = routing_gates.mean(dim=0)         # (num_experts,)

        for expert_id in range(self.num_experts):
            # Prepare per-expert inputs (broadcast over batch)
            utility_input = mean_utilities[expert_id].unsqueeze(0).unsqueeze(-1)  # (1, 1)
            gate_input = mean_gates[expert_id].unsqueeze(0).unsqueeze(-1)         # (1, 1)
            credit_input = self.expert_credits[expert_id].unsqueeze(0).unsqueeze(-1)  # (1, 1)
            context_input = context.mean(dim=0, keepdim=True)  # (1, latent_dim)

            # Get bid from this expert's bidder network
            bid_amount, bid_price, expected_value = self.expert_bidders[expert_id](
                context_input, utility_input, gate_input, credit_input
            )

            # Extract scalar values (detach for the discrete auction)
            amount_val = bid_amount.squeeze().item()
            price_val = bid_price.squeeze().item()
            value_val = expected_value.squeeze().item()

            # Enforce minimum bid price
            price_val = max(price_val, self.config.min_bid_price)

            bids.append(Bid(
                expert_id=expert_id,
                amount=amount_val,
                offered_price=price_val,
                expected_value=value_val,
            ))

            # Keep tensors for differentiable allocation
            bid_amounts_list.append(bid_amount.squeeze())
            bid_prices_list.append(bid_price.squeeze())
            expected_values_list.append(expected_value.squeeze())

        # ── Step 2: Run the auction ──────────────────────────────────
        # Use the discrete auction for the primary allocation
        allocations, winning_bids, losing_bids, clearing_price = \
            self.auction.clear(bids, self.energy_constraint.current_energy)

        # Also compute a differentiable soft allocation for gradient flow
        bid_amounts_t = torch.stack(bid_amounts_list).to(device)
        bid_prices_t = torch.stack(bid_prices_list).to(device)
        expected_values_t = torch.stack(expected_values_list).to(device)

        soft_allocations = self.auction.soft_clear(
            bid_amounts_t, bid_prices_t, expected_values_t,
            self.energy_constraint.current_energy,
        )

        # ── Step 3: Blend hard and soft allocations ─────────────────
        # The hard allocation determines the actual compute budgets,
        # but we blend with the soft allocation for gradient flow.
        # The blend weight decays from 1.0 (fully soft) early in
        # training to 0.1 (mostly hard) later.
        blend_weight = max(0.1, 1.0 / (1.0 + self._step_count / 1000.0))

        # Create hard allocation tensor
        hard_allocations = torch.zeros(self.num_experts, device=device)
        for expert_id, amount in allocations.items():
            hard_allocations[expert_id] = amount

        # Blend: gradients flow through the soft component
        compute_allocations = (
            (1.0 - blend_weight) * hard_allocations.detach()
            + blend_weight * soft_allocations
        )

        # ── Step 4: Consume energy ───────────────────────────────────
        total_alloc = compute_allocations.sum().item()
        energy_consumed = self.energy_constraint.consume(total_alloc)
        self._total_energy_spent += energy_consumed

        # ── Step 5: Recharge energy ──────────────────────────────────
        self.energy_constraint.recharge()

        # ── Step 6: Build market info ────────────────────────────────
        market_info = MarketInfo(
            winning_bids=winning_bids,
            losing_bids=losing_bids,
            total_energy_spent=energy_consumed,
            total_energy_remaining=self.energy_constraint.current_energy,
            auction_clearing_price=clearing_price,
            credit_balances={
                i: self.expert_credits[i].item()
                for i in range(self.num_experts)
            },
            compute_allocations={
                i: compute_allocations[i].item()
                for i in range(self.num_experts)
            },
            bid_stats={
                b.expert_id: {
                    "amount": b.amount,
                    "price": b.offered_price,
                    "expected_value": b.expected_value,
                    "won": b in winning_bids,
                }
                for b in bids
            },
        )

        self._step_count += 1

        # Store a summary in the auction history (keep last 100)
        self._auction_history.append({
            "step": self._step_count,
            "num_winners": len(winning_bids),
            "energy_consumed": energy_consumed,
            "clearing_price": clearing_price,
        })
        if len(self._auction_history) > 100:
            self._auction_history = self._auction_history[-100:]

        return compute_allocations, market_info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_auction(self, bids: List[Bid]) -> Dict[int, float]:
        """Run a standalone auction without the full forward pass.

        This is useful for testing, debugging, or when you want to
        run the auction with manually constructed bids.

        Args:
            bids: List of :class:`Bid` objects from experts.

        Returns:
            allocations: Mapping ``expert_id → allocated_compute``
                for the winning bids.
        """
        allocations, _, _, _ = self.auction.clear(
            bids, self.energy_constraint.current_energy
        )
        return allocations

    def compute_energy_budget(self, allocation: Dict[int, float]) -> bool:
        """Check whether an allocation is within the current energy budget.

        Args:
            allocation: Mapping ``expert_id → allocated_compute``.

        Returns:
            True if the total allocation cost fits within the remaining
            energy budget, False otherwise.
        """
        return self.energy_constraint.check_budget(allocation)

    def recharge_energy(self, amount: float) -> None:
        """Recharge the global compute energy budget.

        This can be called explicitly to add energy to the budget,
        e.g., when the system enters a low-activity phase.

        Args:
            amount: Amount of energy to add.  The budget is capped
                at ``total_energy_budget``.
        """
        self.energy_constraint.recharge(amount)

    def get_market_stats(self) -> Dict[str, Any]:
        """Return current market statistics for logging and diagnostics.

        Returns:
            A dictionary with the following keys:
            - ``step_count``: Number of market cycles completed.
            - ``total_energy_spent``: Cumulative energy consumed.
            - ``current_energy``: Energy currently available.
            - ``energy_fraction_remaining``: Fraction of budget left.
            - ``mean_credit``: Average expert credit balance.
            - ``max_credit``: Maximum expert credit balance.
            - ``min_credit``: Minimum expert credit balance.
            - ``recent_auction_history``: Last 10 auction summaries.
        """
        return {
            "step_count": self._step_count,
            "total_energy_spent": self._total_energy_spent,
            "current_energy": self.energy_constraint.current_energy,
            "energy_fraction_remaining": self.energy_constraint.get_fraction_remaining(),
            "mean_credit": self.expert_credits.mean().item(),
            "max_credit": self.expert_credits.max().item(),
            "min_credit": self.expert_credits.min().item(),
            "credit_std": self.expert_credits.std().item(),
            "recent_auction_history": self._auction_history[-10:],
        }

    def update_expert_credit(
        self,
        expert_id: int,
        value_contributed: float,
    ) -> None:
        """Update an expert's credit balance after it contributes value.

        When an expert produces a useful computation, its credit is
        increased proportionally to the value it contributed.  This gives
        the expert more purchasing power in future auctions, creating a
        virtuous cycle where high-value experts earn more compute.

        The update uses an exponential moving average (EMA) to smoothly
        adjust the credit over time::

            credit_new = ema × credit_old + (1 - ema) × value_contributed

        A high ``credit_ema`` (close to 1.0) means credit changes slowly,
        providing stability.  A low ``credit_ema`` means credit reacts
        quickly to recent performance.

        Args:
            expert_id: The expert whose credit should be updated.
            value_contributed: The value the expert contributed in the
                most recent computation.  Should be non-negative.
        """
        if 0 <= expert_id < self.num_experts:
            old_credit = self.expert_credits[expert_id].item()
            new_credit = (
                self.credit_ema * old_credit
                + (1.0 - self.credit_ema) * max(value_contributed, 0.0)
            )
            self.expert_credits[expert_id] = new_credit
            self._total_credit_earned += max(value_contributed, 0.0)

    def update_expert_credits_batch(
        self,
        value_contributions: Dict[int, float],
    ) -> None:
        """Update credit for multiple experts at once.

        This is more efficient than calling
        :meth:`update_expert_credit` individually for each expert.

        Args:
            value_contributions: Mapping ``expert_id → value_contributed``.
        """
        for expert_id, value in value_contributions.items():
            self.update_expert_credit(expert_id, value)

    def reset_energy(self) -> None:
        """Reset the energy budget to full capacity.

        Useful at the start of a new episode or after a long idle
        period.
        """
        self.energy_constraint.reset()

    def reset_credits(self) -> None:
        """Reset all expert credits to equal initial values.

        Useful for ablation studies or when restarting training.
        """
        self.expert_credits.fill_(1.0)

    def get_top_bidders(self, k: int = 10) -> List[Tuple[int, float]]:
        """Return the top-k experts by credit balance.

        Args:
            k: Number of top experts to return.

        Returns:
            List of ``(expert_id, credit)`` tuples sorted by credit
            descending.
        """
        credits = self.expert_credits
        top_k_values, top_k_indices = torch.topk(credits, min(k, self.num_experts))
        return [
            (idx.item(), val.item())
            for idx, val in zip(top_k_indices, top_k_values)
        ]

    def get_allocation_concentration(self) -> float:
        """Measure how concentrated the current allocation is.

        Uses the Herfindahl-Hirschman Index (HHI), defined as the
        sum of squared market shares.  An HHI of 1.0 means all
        compute goes to a single expert; an HHI of ``1/N`` means
        perfectly equal distribution.

        Returns:
            The HHI value, a float in ``[1/N, 1.0]``.
        """
        if self.expert_credits.sum() < 1e-8:
            return 1.0 / self.num_experts
        shares = self.expert_credits / self.expert_credits.sum()
        hhi = (shares ** 2).sum().item()
        return hhi
