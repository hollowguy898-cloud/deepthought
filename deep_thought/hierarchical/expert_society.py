"""
Hierarchical Expert Society for Deep Thought.

Implements a multi-tier expert hierarchy where experts at different levels
of abstraction coordinate to produce intelligent, adaptive behavior. The
hierarchy spans from fast reactive reflex experts at the bottom to
meta-routing experts at the top that decide which other experts should think.

Architecture Overview:
    ┌─────────────────────────────────────────────┐
    │  MetaTier (Level 3) — Meta-routing experts  │
    │  Decide which other experts should think.    │
    ├─────────────────────────────────────────────┤
    │  StrategicTier (Level 2) — Manager experts  │
    │  Set goals and allocate compute budgets.     │
    ├─────────────────────────────────────────────┤
    │  TacticalTier (Level 1) — Coordination      │
    │  Coordinate reflex experts, short-term plan. │
    ├─────────────────────────────────────────────┤
    │  ReflexTier (Level 0) — Reactive experts    │
    │  Fast stimulus-response, low latency.        │
    └─────────────────────────────────────────────┘

Routing Flow:
    1. Input arrives at MetaTier, which decides which tiers/experts to activate.
    2. StrategicTier receives meta-guidance and produces goals + compute budgets.
    3. TacticalTier receives strategic goals and coordinates reflex experts.
    4. ReflexTier provides fast reactive outputs under tactical supervision.
    5. Outputs from all active tiers are combined (weighted by compute budget)
       into the final output.

Each tier has its own routing network (small gating MLP) and a pool of
specialized expert MLPs. Higher tiers produce goal/context vectors that
condition the behavior of lower tiers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from deep_thought.config import HierarchicalConfig


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class TierLevel(Enum):
    """Enumeration of hierarchy tier levels."""
    REFLEX = 0
    TACTICAL = 1
    STRATEGIC = 2
    META = 3


@dataclass
class TierExpertStats:
    """Statistics for a single expert within a tier."""
    activation_count: int = 0
    total_gate_value: float = 0.0
    utility_score: float = 0.0
    tier: TierLevel = TierLevel.REFLEX


@dataclass
class RoutingInfo:
    """Aggregated routing information across all tiers."""
    meta_selection: Optional[torch.Tensor] = None
    meta_gates: Optional[torch.Tensor] = None
    strategic_selection: Optional[torch.Tensor] = None
    strategic_gates: Optional[torch.Tensor] = None
    strategic_goals: Optional[torch.Tensor] = None
    tactical_selection: Optional[torch.Tensor] = None
    tactical_gates: Optional[torch.Tensor] = None
    reflex_selection: Optional[torch.Tensor] = None
    reflex_gates: Optional[torch.Tensor] = None
    compute_budgets: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Tier-specific expert modules
# ---------------------------------------------------------------------------

class ReflexExpert(nn.Module):
    """
    A single reflex expert — a lightweight MLP for immediate
    stimulus-response mappings.

    Reflex experts are the fastest and simplest in the hierarchy.
    They use a narrow hidden dimension and at most two linear layers
    to minimize inference latency.

    Args:
        input_dim: Dimension of the input vector.
        hidden_dim: Width of the hidden layer.
        output_dim: Dimension of the output vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the reflex expert."""
        return self.network(x)


class TacticalExpert(nn.Module):
    """
    A tactical expert — a mid-level MLP that coordinates reflex experts
    and handles short-term planning.

    Tactical experts receive goal vectors from the strategic tier and
    produce both an output and a coordination signal that modulates
    reflex expert selection.

    Args:
        input_dim: Dimension of the input vector (includes goal context).
        hidden_dim: Width of the hidden layer.
        output_dim: Dimension of the output vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Coordination signal: influences which reflex experts are preferred
        self.coord_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and coordination signal.

        Returns:
            output: The expert's output contribution.
            coord_signal: A vector that conditions lower-tier routing.
        """
        h = F.silu(self.network[0](x))          # first layer + activation
        h = F.silu(self.network[2](h))           # second layer + activation
        output = self.network[4](h)              # output projection
        coord_signal = self.coord_head(h)        # coordination signal
        return output, coord_signal


class StrategicExpert(nn.Module):
    """
    A strategic expert — a high-level manager that sets goals and
    allocates compute budgets for lower tiers.

    Strategic experts receive meta-guidance from the meta tier and
    produce:
      - A goal vector that conditions the tactical tier.
      - A compute-budget allocation across lower tiers.

    Args:
        input_dim: Dimension of the input vector (includes meta context).
        hidden_dim: Width of the hidden layer.
        goal_dim: Dimension of the goal vector produced.
        num_budget_slots: Number of tiers that receive budget allocations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        goal_dim: int,
        num_budget_slots: int = 3,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.goal_head = nn.Linear(hidden_dim, goal_dim)
        self.budget_head = nn.Linear(hidden_dim, num_budget_slots)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning goals and budget allocation.

        Returns:
            goals: Goal vector for the tactical tier.
            budget_logits: Unnormalized budget allocations (softmax to normalize).
        """
        h = self.network(x)
        goals = self.goal_head(h)
        budget_logits = self.budget_head(h)
        return goals, budget_logits


class MetaExpert(nn.Module):
    """
    A meta-routing expert — decides which other experts should think.

    Meta experts are the "middle management" of the hierarchy. They
    receive the full context and produce:
      - A routing mask that gates which experts at lower tiers are
        allowed to participate.
      - A priority signal for each lower tier.

    Args:
        input_dim: Dimension of the input vector.
        hidden_dim: Width of the hidden layer.
        num_lower_tiers: Number of tiers below meta (typically 3).
        num_strategic_experts: Number of strategic experts to gate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_lower_tiers: int = 3,
        num_strategic_experts: int = 8,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Tier-level priority: how much compute to give each lower tier
        self.tier_priority_head = nn.Linear(hidden_dim, num_lower_tiers)
        # Expert-level gating: which strategic experts to activate
        self.expert_gate_head = nn.Linear(hidden_dim, num_strategic_experts)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning tier priorities and expert gates.

        Returns:
            tier_priorities: Priority weight for each lower tier.
            expert_gates: Gating logits for each strategic expert.
            context: Internal context representation for downstream use.
        """
        h = self.network(x)
        tier_priorities = F.softmax(self.tier_priority_head(h), dim=-1)
        expert_gates = self.expert_gate_head(h)
        return tier_priorities, expert_gates, h


# ---------------------------------------------------------------------------
# Tier-level routing networks
# ---------------------------------------------------------------------------

class TierRouter(nn.Module):
    """
    A small routing network for a single tier.

    Given an input vector (which may include context from higher tiers),
    produces top-k gate values and indices over the tier's experts.

    Args:
        input_dim: Dimension of the router input.
        num_experts: Number of experts in this tier.
        top_k: Number of experts to select per forward pass.
        hidden_dim: Width of the router's hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        x: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route input to top-k experts in this tier.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            available_mask: Optional boolean/binary mask of shape
                ``(num_experts,)`` indicating which experts are available.

        Returns:
            gates: Normalized gate values of shape ``(batch, top_k)``.
            indices: Selected expert indices of shape ``(batch, top_k)``.
            logits: Raw routing logits of shape ``(batch, num_experts)``.
        """
        logits = self.gate(x)

        # Mask unavailable experts with large negative value
        if available_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~available_mask.bool(), mask_value)

        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(
            probs, self.top_k, dim=-1
        )
        gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return gates, top_k_indices, logits


# ---------------------------------------------------------------------------
# Full tier modules (experts + router + statistics)
# ---------------------------------------------------------------------------

class ReflexTier(nn.Module):
    """
    Reflex tier (Level 0): Fast, reactive experts for immediate
    stimulus-response.

    This tier contains simple MLPs with low latency. It is the
    bottom-most tier and does not produce any downstream context
    for other tiers — it only produces output contributions.

    Args:
        num_experts: Number of reflex experts.
        input_dim: Input dimension per expert.
        hidden_dim: Hidden dimension per expert.
        output_dim: Output dimension per expert.
        top_k: Number of experts to activate per forward pass.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        top_k: int = 4,
    ):
        super().__init__()
        self.tier_level = TierLevel.REFLEX
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert pool
        self.experts = nn.ModuleList([
            ReflexExpert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

        # Router (input may include tactical coordination signal)
        self.router = TierRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Per-expert statistics
        self.expert_stats: Dict[int, TierExpertStats] = {
            i: TierExpertStats(tier=TierLevel.REFLEX) for i in range(num_experts)
        }

    def forward(
        self,
        x: torch.Tensor,
        available_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the reflex tier.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            available_experts: Optional mask over experts.

        Returns:
            output: Weighted sum of expert outputs, shape ``(batch, output_dim)``.
            gates: Gate values for selected experts.
            indices: Indices of selected experts.
        """
        gates, indices, logits = self.router(x, available_experts)
        batch_size = x.size(0)

        # Compute weighted expert output
        output = torch.zeros(batch_size, x.size(-1), device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            expert_idx = indices[:, k]                     # (batch,)
            gate_val = gates[:, k : k + 1]                 # (batch, 1)

            # Apply each unique expert once and mask
            unique_ids = expert_idx.unique()
            for eid in unique_ids:
                mask = (expert_idx == eid).unsqueeze(-1).float()   # (batch, 1)
                expert_out = self.experts[eid.item()](x)           # (batch, dim)
                output = output + mask * gate_val * expert_out

                # Update stats
                with torch.no_grad():
                    self.expert_stats[eid.item()].activation_count += int(mask.sum().item())
                    self.expert_stats[eid.item()].total_gate_value += float(gate_val[mask.squeeze(-1) > 0].sum().item())

        return output, gates, indices


class TacticalTier(nn.Module):
    """
    Tactical tier (Level 1): Mid-level experts that coordinate reflex
    experts and handle short-term planning.

    Tactical experts receive strategic goals and produce both an output
    contribution and a coordination signal that modulates reflex-tier
    routing.

    Args:
        num_experts: Number of tactical experts.
        input_dim: Input dimension (observation + strategic goal).
        hidden_dim: Hidden dimension per expert.
        output_dim: Output dimension per expert.
        top_k: Number of experts to activate per forward pass.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        top_k: int = 4,
    ):
        super().__init__()
        self.tier_level = TierLevel.TACTICAL
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim

        # Expert pool
        self.experts = nn.ModuleList([
            TacticalExpert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

        # Router
        self.router = TierRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Per-expert statistics
        self.expert_stats: Dict[int, TierExpertStats] = {
            i: TierExpertStats(tier=TierLevel.TACTICAL) for i in range(num_experts)
        }

    def forward(
        self,
        x: torch.Tensor,
        available_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the tactical tier.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            available_experts: Optional mask over experts.

        Returns:
            output: Weighted sum of expert outputs.
            coord_signal: Aggregated coordination signal for the reflex tier.
            gates: Gate values for selected experts.
            indices: Indices of selected experts.
        """
        gates, indices, logits = self.router(x, available_experts)
        batch_size = x.size(0)
        out_dim = self.output_dim

        output = torch.zeros(batch_size, out_dim, device=x.device, dtype=x.dtype)
        coord_signal = torch.zeros(batch_size, out_dim, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            gate_val = gates[:, k : k + 1]

            unique_ids = expert_idx.unique()
            for eid in unique_ids:
                mask = (expert_idx == eid).unsqueeze(-1).float()
                expert_out, expert_coord = self.experts[eid.item()](x)
                output = output + mask * gate_val * expert_out
                coord_signal = coord_signal + mask * gate_val * expert_coord

                with torch.no_grad():
                    self.expert_stats[eid.item()].activation_count += int(mask.sum().item())
                    self.expert_stats[eid.item()].total_gate_value += float(gate_val[mask.squeeze(-1) > 0].sum().item())

        return output, coord_signal, gates, indices


class StrategicTier(nn.Module):
    """
    Strategic tier (Level 2): High-level manager experts that set goals
    and allocate compute budgets for lower tiers.

    Strategic experts receive meta-level context and produce:
      - Goal vectors for the tactical tier.
      - Compute budget allocations across lower tiers.

    Args:
        num_experts: Number of strategic experts.
        input_dim: Input dimension (observation + meta context).
        hidden_dim: Hidden dimension per expert.
        goal_dim: Dimension of the goal vector produced.
        num_budget_slots: Number of lower tiers receiving budgets.
        top_k: Number of experts to activate per forward pass.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        goal_dim: int,
        num_budget_slots: int = 3,
        top_k: int = 2,
    ):
        super().__init__()
        self.tier_level = TierLevel.STRATEGIC
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert pool
        self.experts = nn.ModuleList([
            StrategicExpert(input_dim, hidden_dim, goal_dim, num_budget_slots)
            for _ in range(num_experts)
        ])

        # Router
        self.router = TierRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Per-expert statistics
        self.expert_stats: Dict[int, TierExpertStats] = {
            i: TierExpertStats(tier=TierLevel.STRATEGIC) for i in range(num_experts)
        }

    def forward(
        self,
        x: torch.Tensor,
        available_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the strategic tier.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            available_experts: Optional mask over experts.

        Returns:
            goals: Aggregated goal vector for the tactical tier.
            budget_logits: Aggregated budget allocation logits.
            gates: Gate values for selected experts.
            indices: Indices of selected experts.
        """
        gates, indices, logits = self.router(x, available_experts)
        batch_size = x.size(0)
        goal_dim = self.experts[0].goal_head.out_features
        num_budget_slots = self.experts[0].budget_head.out_features

        goals = torch.zeros(batch_size, goal_dim, device=x.device, dtype=x.dtype)
        budget_logits = torch.zeros(batch_size, num_budget_slots, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            gate_val = gates[:, k : k + 1]

            unique_ids = expert_idx.unique()
            for eid in unique_ids:
                mask = (expert_idx == eid).unsqueeze(-1).float()
                expert_goals, expert_budget = self.experts[eid.item()](x)
                goals = goals + mask * gate_val * expert_goals
                budget_logits = budget_logits + mask * gate_val * expert_budget

                with torch.no_grad():
                    self.expert_stats[eid.item()].activation_count += int(mask.sum().item())
                    self.expert_stats[eid.item()].total_gate_value += float(gate_val[mask.squeeze(-1) > 0].sum().item())

        return goals, budget_logits, gates, indices


class MetaTier(nn.Module):
    """
    Meta tier (Level 3): Meta-routing experts that decide which other
    experts should think — the "middle management" tier.

    Meta experts receive the full observation context and produce:
      - Tier-level priorities: how much compute to allocate to each
        lower tier.
      - Expert-level gates: which strategic experts to activate.

    Args:
        num_experts: Number of meta experts.
        input_dim: Input dimension (full observation context).
        hidden_dim: Hidden dimension per expert.
        num_lower_tiers: Number of tiers below meta.
        num_strategic_experts: Number of strategic experts to gate.
        top_k: Number of meta experts to activate per forward pass.
    """

    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        num_lower_tiers: int = 3,
        num_strategic_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.tier_level = TierLevel.META
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert pool
        self.experts = nn.ModuleList([
            MetaExpert(input_dim, hidden_dim, num_lower_tiers, num_strategic_experts)
            for _ in range(num_experts)
        ])

        # Router
        self.router = TierRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Per-expert statistics
        self.expert_stats: Dict[int, TierExpertStats] = {
            i: TierExpertStats(tier=TierLevel.META) for i in range(num_experts)
        }

    def forward(
        self,
        x: torch.Tensor,
        available_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the meta tier.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            available_experts: Optional mask over experts.

        Returns:
            tier_priorities: Priority weights for each lower tier.
            expert_gates: Gating logits for strategic experts.
            meta_context: Internal representation from selected experts.
            gates: Gate values for selected meta experts.
            indices: Indices of selected meta experts.
        """
        gates, indices, logits = self.router(x, available_experts)
        batch_size = x.size(0)
        num_lower_tiers = self.experts[0].tier_priority_head.out_features
        num_strategic = self.experts[0].expert_gate_head.out_features
        hidden_dim = self.experts[0].network[-1].out_features

        tier_priorities = torch.zeros(batch_size, num_lower_tiers, device=x.device, dtype=x.dtype)
        expert_gates = torch.zeros(batch_size, num_strategic, device=x.device, dtype=x.dtype)
        meta_context = torch.zeros(batch_size, hidden_dim, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = indices[:, k]
            gate_val = gates[:, k : k + 1]

            unique_ids = expert_idx.unique()
            for eid in unique_ids:
                mask = (expert_idx == eid).unsqueeze(-1).float()
                tp, eg, ctx = self.experts[eid.item()](x)
                tier_priorities = tier_priorities + mask * gate_val * tp
                expert_gates = expert_gates + mask * gate_val * eg
                meta_context = meta_context + mask * gate_val * ctx

                with torch.no_grad():
                    self.expert_stats[eid.item()].activation_count += int(mask.sum().item())
                    self.expert_stats[eid.item()].total_gate_value += float(gate_val[mask.squeeze(-1) > 0].sum().item())

        return tier_priorities, expert_gates, meta_context, gates, indices


# ---------------------------------------------------------------------------
# Hierarchical Expert Society
# ---------------------------------------------------------------------------

class HierarchicalExpertSociety(nn.Module):
    """
    Multi-tier hierarchical expert society for Deep Thought.

    Experts at different levels of abstraction coordinate through a
    top-down control flow:

    1. **MetaTier** decides which strategic experts should think and how
       much compute each lower tier receives.
    2. **StrategicTier** sets goals and allocates compute budgets for the
       tactical and reflex tiers.
    3. **TacticalTier** coordinates reflex experts and produces a
       coordination signal that conditions reflex routing.
    4. **ReflexTier** produces fast, reactive outputs.

    The final output is a compute-budget-weighted combination of
    contributions from all active tiers.

    Args:
        config: A :class:`HierarchicalConfig` instance controlling tier
            sizes, dimensions, and other hyperparameters.
        latent_dim: Dimensionality of the shared latent representation.
            All tiers produce and consume vectors of this size.
    """

    def __init__(
        self,
        config: Optional[HierarchicalConfig] = None,
        latent_dim: int = 1024,
    ):
        super().__init__()
        self.config = config or HierarchicalConfig()
        self.latent_dim = latent_dim

        # Only build tiers if hierarchy is enabled
        self.use_hierarchy = self.config.use_hierarchy

        if self.use_hierarchy and self.config.num_tiers >= 1:
            self.reflex_tier = ReflexTier(
                num_experts=self.config.reflex_experts,
                input_dim=latent_dim,
                hidden_dim=self.config.reflex_hidden_dim,
                output_dim=latent_dim,
                top_k=min(4, self.config.reflex_experts),
            )
        if self.use_hierarchy and self.config.num_tiers >= 2:
            self.tactical_tier = TacticalTier(
                num_experts=self.config.tactical_experts,
                input_dim=latent_dim * 2,   # observation + strategic goal
                hidden_dim=self.config.tactical_hidden_dim,
                output_dim=latent_dim,
                top_k=min(4, self.config.tactical_experts),
            )
        if self.use_hierarchy and self.config.num_tiers >= 3:
            self.strategic_tier = StrategicTier(
                num_experts=self.config.strategic_experts,
                input_dim=latent_dim * 2,   # observation + meta context
                hidden_dim=self.config.strategic_hidden_dim,
                goal_dim=latent_dim,
                num_budget_slots=2,         # tactical + reflex
                top_k=min(2, self.config.strategic_experts),
            )
        if self.use_hierarchy and self.config.num_tiers >= 4:
            self.meta_tier = MetaTier(
                num_experts=self.config.meta_experts,
                input_dim=latent_dim * 2,   # h_t + x_t concatenated
                hidden_dim=self.config.meta_hidden_dim,
                num_lower_tiers=3,
                num_strategic_experts=self.config.strategic_experts,
                top_k=min(2, self.config.meta_experts),
            )

        # Tier name mapping for convenience
        self._tier_map: Dict[TierLevel, Optional[nn.Module]] = {
            TierLevel.REFLEX: getattr(self, "reflex_tier", None),
            TierLevel.TACTICAL: getattr(self, "tactical_tier", None),
            TierLevel.STRATEGIC: getattr(self, "strategic_tier", None),
            TierLevel.META: getattr(self, "meta_tier", None),
        }

        # Expert registry: maps expert_id -> (TierLevel, local_index)
        # Used for promotion/demotion lookups
        self._expert_registry: Dict[int, Tuple[TierLevel, int]] = {}
        self._next_expert_id = 0
        self._register_initial_experts()

        # Learnable budget-allocation parameters (used by
        # allocate_compute_budget as a learnable baseline)
        self.budget_scale = nn.Parameter(
            torch.ones(len(TierLevel)) * self.config.compute_budget_total / len(TierLevel)
        )

        # Learned linear projections to replace adaptive_avg_pool1d
        # (which destroys information by averaging). These preserve
        # information through learnable weight matrices.
        if self.use_hierarchy and self.config.num_tiers >= 4:
            self.meta_context_proj = nn.Linear(self.config.meta_hidden_dim, latent_dim, bias=False)
            self.meta_output_proj = nn.Linear(self.config.meta_hidden_dim, latent_dim, bias=False)
        if self.use_hierarchy and self.config.num_tiers >= 3:
            self.goals_proj = nn.Linear(latent_dim, latent_dim, bias=False)

    # ------------------------------------------------------------------
    # Expert registry helpers
    # ------------------------------------------------------------------

    def _register_initial_experts(self) -> None:
        """Populate the expert registry with all initially created experts."""
        tier_counts = {
            TierLevel.REFLEX: self.config.reflex_experts,
            TierLevel.TACTICAL: self.config.tactical_experts,
            TierLevel.STRATEGIC: self.config.strategic_experts,
            TierLevel.META: self.config.meta_experts,
        }
        for tier_level, count in tier_counts.items():
            for local_idx in range(count):
                self._expert_registry[self._next_expert_id] = (tier_level, local_idx)
                self._next_expert_id += 1

    def _get_tier_module(self, tier_level: TierLevel) -> Optional[nn.Module]:
        """Return the nn.Module for a given tier level."""
        return self._tier_map.get(tier_level)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        available_experts: Optional[Dict[TierLevel, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, RoutingInfo]:
        """
        Hierarchical routing through all tiers.

        Processing order is top-down: Meta → Strategic → Tactical → Reflex.
        Each tier's output conditions the next lower tier's behavior.

        Args:
            h_t: Current hidden state, shape ``(batch, latent_dim)``.
            x_t: Encoded observation, shape ``(batch, latent_dim)``.
            context: Optional context vector, shape ``(batch, latent_dim)``.
                If ``None``, ``h_t`` is used as context.
            available_experts: Optional dict mapping each tier level to a
                boolean mask of shape ``(num_experts_in_tier,)`` indicating
                which experts are available for routing.

        Returns:
            output: Combined output tensor of shape ``(batch, latent_dim)``.
            routing_info: A :class:`RoutingInfo` object with detailed
                routing information for every tier.
        """
        if context is None:
            context = h_t

        routing_info = RoutingInfo()
        batch_size = h_t.size(0)
        device = h_t.device
        dtype = h_t.dtype

        # Combined input for the meta tier
        meta_input = torch.cat([h_t, x_t], dim=-1) if self.config.num_tiers >= 4 else h_t

        # ---- Meta tier (Level 3) ----
        if self.config.num_tiers >= 4 and hasattr(self, "meta_tier"):
            meta_avail = available_experts.get(TierLevel.META) if available_experts else None
            tier_priorities, expert_gates, meta_context, meta_gates, meta_indices = \
                self.meta_tier(meta_input, meta_avail)
            routing_info.meta_selection = meta_indices
            routing_info.meta_gates = meta_gates
        else:
            # Defaults: uniform priority, all experts available
            tier_priorities = torch.ones(batch_size, 3, device=device, dtype=dtype) / 3.0
            expert_gates = None
            meta_context = torch.zeros(batch_size, self.config.meta_hidden_dim, device=device, dtype=dtype)

        # ---- Strategic tier (Level 2) ----
        if self.config.num_tiers >= 3 and hasattr(self, "strategic_tier"):
            # Condition strategic input with meta context (learned projection)
            meta_ctx_proj = self.meta_context_proj(meta_context) if meta_context.size(-1) != self.latent_dim else meta_context
            strategic_input = torch.cat([h_t, meta_ctx_proj], dim=-1)

            # If meta tier produced expert gates, use them to mask strategic experts
            if expert_gates is not None:
                # Binarize: only allow experts with gate > 0.5
                strategic_mask = (expert_gates > 0.0).float()
                if available_experts and TierLevel.STRATEGIC in available_experts:
                    strategic_mask = strategic_mask * available_experts[TierLevel.STRATEGIC]
            else:
                strategic_mask = available_experts.get(TierLevel.STRATEGIC) if available_experts else None

            goals, budget_logits, strategic_gates, strategic_indices = \
                self.strategic_tier(strategic_input, strategic_mask)
            routing_info.strategic_selection = strategic_indices
            routing_info.strategic_gates = strategic_gates
            routing_info.strategic_goals = goals
        else:
            goals = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
            budget_logits = None

        # Compute budget allocation
        compute_budgets = self.allocate_compute_budget(
            budget_logits, tier_priorities
        )
        routing_info.compute_budgets = compute_budgets

        # ---- Tactical tier (Level 1) ----
        if self.config.num_tiers >= 2 and hasattr(self, "tactical_tier"):
            tactical_input = torch.cat([h_t, goals], dim=-1)
            tactical_avail = available_experts.get(TierLevel.TACTICAL) if available_experts else None

            tactical_output, coord_signal, tactical_gates, tactical_indices = \
                self.tactical_tier(tactical_input, tactical_avail)
            routing_info.tactical_selection = tactical_indices
            routing_info.tactical_gates = tactical_gates
        else:
            tactical_output = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
            coord_signal = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)

        # ---- Reflex tier (Level 0) ----
        if self.config.num_tiers >= 1 and hasattr(self, "reflex_tier"):
            # Reflex input is conditioned by tactical coordination signal
            reflex_input = h_t + coord_signal
            reflex_avail = available_experts.get(TierLevel.REFLEX) if available_experts else None

            reflex_output, reflex_gates, reflex_indices = \
                self.reflex_tier(reflex_input, reflex_avail)
            routing_info.reflex_selection = reflex_indices
            routing_info.reflex_gates = reflex_gates
        else:
            reflex_output = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)

        # ---- Combine outputs weighted by compute budgets ----
        budget_reflex = compute_budgets.get("reflex", 0.25)
        budget_tactical = compute_budgets.get("tactical", 0.25)
        budget_strategic = compute_budgets.get("strategic", 0.25)
        budget_meta = compute_budgets.get("meta", 0.25)

        # Meta contribution: the meta_context (learned projection)
        if meta_context.size(-1) != self.latent_dim:
            meta_output = self.meta_output_proj(meta_context)
        else:
            meta_output = meta_context

        output = (
            budget_reflex * reflex_output
            + budget_tactical * tactical_output
            + budget_strategic * self.goals_proj(goals)
            + budget_meta * meta_output
        )

        return output, routing_info

    def route_tier(
        self,
        tier: TierLevel,
        input: torch.Tensor,
        available_sub_experts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route within a single tier, independent of the full forward pass.

        This is useful for debugging, analysis, or when you want to
        inspect a specific tier's routing behavior in isolation.

        Args:
            tier: Which tier level to route within.
            input: Input tensor for the tier's router.
            available_sub_experts: Optional boolean mask of shape
                ``(num_experts_in_tier,)``.

        Returns:
            selected: Indices of selected experts, shape ``(batch, top_k)``.
            gates: Gate values for selected experts, shape ``(batch, top_k)``.

        Raises:
            ValueError: If the requested tier is not enabled.
        """
        tier_module = self._get_tier_module(tier)
        if tier_module is None:
            raise ValueError(
                f"Tier {tier.name} is not enabled. "
                f"Enable it by setting num_tiers >= {tier.value + 1} "
                f"in HierarchicalConfig."
            )

        # Each tier module has a router attribute
        gates, indices, _ = tier_module.router(input, available_sub_experts)
        return indices, gates

    def allocate_compute_budget(
        self,
        strategy_output: Optional[torch.Tensor] = None,
        tier_priorities: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute the compute budget allocated to each tier.

        The budget is determined by combining three signals:

        1. **Learnable baseline** (``self.budget_scale``): a parameter
           that is updated via gradient descent.
        2. **Strategy output**: budget logits produced by the strategic
           tier (if available).
        3. **Tier priorities**: priority weights produced by the meta
           tier (if available).

        The final allocation is normalized so that the total budget
        equals ``config.compute_budget_total``.

        Args:
            strategy_output: Budget logits from strategic experts,
                shape ``(batch, num_budget_slots)``. If ``None``,
                only the learnable baseline is used.
            tier_priorities: Priority weights from the meta tier,
                shape ``(batch, num_lower_tiers)``. If ``None``, equal
                priorities are assumed.

        Returns:
            A dictionary mapping tier names (``"reflex"``, ``"tactical"``,
            ``"strategic"``, ``"meta"``) to their allocated budget
            fraction.
        """
        # Start from the learnable baseline
        baseline = F.softmax(self.budget_scale, dim=0)

        # Incorporate strategy output (detached — budget is a control signal)
        if strategy_output is not None:
            strat_budget = F.softmax(strategy_output.detach().mean(dim=0), dim=0)
            # strat_budget covers tactical + reflex (2 slots)
            # Pad to 4 slots if needed
            if strat_budget.numel() < 4:
                padding = torch.zeros(
                    4 - strat_budget.numel(),
                    device=strat_budget.device,
                    dtype=strat_budget.dtype,
                )
                strat_budget = torch.cat([strat_budget, padding])
            alpha = self.config.budget_allocation_lr
            blended = (1 - alpha) * baseline + alpha * strat_budget
        else:
            blended = baseline

        # Incorporate tier priorities from meta tier
        if tier_priorities is not None:
            tp = tier_priorities.detach().mean(dim=0)
            # tp has 3 entries: reflex, tactical, strategic
            # meta gets the remainder
            if tp.numel() >= 3:
                meta_priority = 1.0 - tp[:3].sum().item()
                meta_priority = max(meta_priority, 0.01)
                full_priorities = torch.cat([
                    tp[:3],
                    torch.tensor([meta_priority], device=tp.device),
                ])
                alpha = self.config.budget_allocation_lr
                blended = (1 - alpha) * blended[:4] + alpha * full_priorities

        # Normalize
        total = blended[:4].sum()
        if total > 0:
            blended = blended[:4] / total

        # Scale by total budget
        blended = blended * self.config.compute_budget_total

        return {
            "reflex": float(blended[0].item()),
            "tactical": float(blended[1].item()),
            "strategic": float(blended[2].item()),
            "meta": float(blended[3].item()),
        }

    def get_tier_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Return statistics for each tier.

        For each tier, the statistics include:
          - ``num_experts``: Total number of experts in the tier.
          - ``total_activations``: Sum of activation counts across experts.
          - ``mean_gate_value``: Average gate value across experts.
          - ``max_utility`` / ``min_utility``: Extreme utility scores.
          - ``expert_details``: Per-expert activation and gate info.

        Returns:
            A dictionary keyed by tier name, each mapping to a stats dict.
        """
        result: Dict[str, Dict[str, Any]] = {}
        tier_modules = {
            "reflex": getattr(self, "reflex_tier", None),
            "tactical": getattr(self, "tactical_tier", None),
            "strategic": getattr(self, "strategic_tier", None),
            "meta": getattr(self, "meta_tier", None),
        }

        for tier_name, tier_module in tier_modules.items():
            if tier_module is None:
                result[tier_name] = {"enabled": False}
                continue

            stats = tier_module.expert_stats
            total_activations = sum(s.activation_count for s in stats.values())
            mean_gate = (
                sum(s.total_gate_value for s in stats.values())
                / max(len(stats), 1)
            )
            utilities = [s.utility_score for s in stats.values()]

            expert_details = {
                eid: {
                    "activation_count": s.activation_count,
                    "total_gate_value": s.total_gate_value,
                    "utility_score": s.utility_score,
                }
                for eid, s in stats.items()
            }

            result[tier_name] = {
                "enabled": True,
                "num_experts": len(stats),
                "total_activations": total_activations,
                "mean_gate_value": mean_gate,
                "max_utility": max(utilities) if utilities else 0.0,
                "min_utility": min(utilities) if utilities else 0.0,
                "expert_details": expert_details,
            }

        return result

    def promote_expert(
        self,
        expert_id: int,
        from_tier: TierLevel,
        to_tier: TierLevel,
    ) -> bool:
        """
        Promote an expert from a lower tier to a higher tier.

        Promotion copies the expert's learned representation (where
        architecturally feasible) into a slot in the destination tier.
        If the destination tier is full, the lowest-utility expert is
        replaced.

        The expert is removed from its original tier and a fresh expert
        is inserted in its place.

        Args:
            expert_id: Global expert ID in the registry.
            from_tier: The tier the expert currently belongs to.
            to_tier: The target tier (must be a higher level).

        Returns:
            ``True`` if the promotion succeeded, ``False`` otherwise.

        Raises:
            ValueError: If the source tier is not lower than the target.
        """
        if from_tier.value >= to_tier.value:
            raise ValueError(
                f"Cannot promote from {from_tier.name} to {to_tier.name}: "
                f"source tier must be lower than target tier."
            )

        if expert_id not in self._expert_registry:
            return False

        registered_tier, local_idx = self._expert_registry[expert_id]
        if registered_tier != from_tier:
            return False

        src_module = self._get_tier_module(from_tier)
        dst_module = self._get_tier_module(to_tier)
        if src_module is None or dst_module is None:
            return False

        # Find a slot in the destination tier (replace lowest-utility expert)
        dst_stats = dst_module.expert_stats
        worst_local = min(dst_stats, key=lambda k: dst_stats[k].utility_score)

        # Transfer parameters where dimensions allow it
        src_expert = src_module.experts[local_idx]
        dst_expert = dst_module.experts[worst_local]

        with torch.no_grad():
            src_params = list(src_expert.parameters())
            dst_params = list(dst_expert.parameters())
            for sp, dp in zip(src_params, dst_params):
                if sp.shape == dp.shape:
                    dp.copy_(sp)
                else:
                    # Incompatible shapes — initialize with scaled noise
                    dp.normal_(0, 0.02)

        # Update registries
        new_id = self._next_expert_id
        self._next_expert_id += 1
        self._expert_registry[new_id] = (to_tier, worst_local)

        # Reset the source expert (replace with a fresh one)
        self._expert_registry[expert_id] = (from_tier, local_idx)
        with torch.no_grad():
            for p in src_expert.parameters():
                p.normal_(0, 0.02)

        # Update stats
        src_module.expert_stats[local_idx] = TierExpertStats(tier=from_tier)
        dst_module.expert_stats[worst_local] = TierExpertStats(tier=to_tier)

        return True

    def demote_expert(
        self,
        expert_id: int,
        from_tier: TierLevel,
        to_tier: TierLevel,
    ) -> bool:
        """
        Demote an expert from a higher tier to a lower tier.

        Demotion copies the expert's learned representation (where
        architecturally feasible) into a slot in the destination tier.
        If the destination tier is full, the lowest-utility expert is
        replaced.

        The expert is removed from its original tier and a fresh expert
        is inserted in its place.

        Args:
            expert_id: Global expert ID in the registry.
            from_tier: The tier the expert currently belongs to.
            to_tier: The target tier (must be a lower level).

        Returns:
            ``True`` if the demotion succeeded, ``False`` otherwise.

        Raises:
            ValueError: If the source tier is not higher than the target.
        """
        if from_tier.value <= to_tier.value:
            raise ValueError(
                f"Cannot demote from {from_tier.name} to {to_tier.name}: "
                f"source tier must be higher than target tier."
            )

        if expert_id not in self._expert_registry:
            return False

        registered_tier, local_idx = self._expert_registry[expert_id]
        if registered_tier != from_tier:
            return False

        src_module = self._get_tier_module(from_tier)
        dst_module = self._get_tier_module(to_tier)
        if src_module is None or dst_module is None:
            return False

        # Find a slot in the destination tier (replace lowest-utility expert)
        dst_stats = dst_module.expert_stats
        worst_local = min(dst_stats, key=lambda k: dst_stats[k].utility_score)

        # Transfer parameters where dimensions allow it
        src_expert = src_module.experts[local_idx]
        dst_expert = dst_module.experts[worst_local]

        with torch.no_grad():
            src_params = list(src_expert.parameters())
            dst_params = list(dst_expert.parameters())
            for sp, dp in zip(src_params, dst_params):
                if sp.shape == dp.shape:
                    dp.copy_(sp)
                else:
                    dp.normal_(0, 0.02)

        # Update registries
        new_id = self._next_expert_id
        self._next_expert_id += 1
        self._expert_registry[new_id] = (to_tier, worst_local)

        # Reset the source expert
        self._expert_registry[expert_id] = (from_tier, local_idx)
        with torch.no_grad():
            for p in src_expert.parameters():
                p.normal_(0, 0.02)

        # Update stats
        src_module.expert_stats[local_idx] = TierExpertStats(tier=from_tier)
        dst_module.expert_stats[worst_local] = TierExpertStats(tier=to_tier)

        return True
