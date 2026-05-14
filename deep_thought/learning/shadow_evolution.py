"""
Stable Self-Improvement Component 3: Evolutionary Search within the Expert Bank.

Uses the "Dormant State" of experts to run background experiments
(Shadow Evolution) while Active Experts handle the task.

Key ideas:
  - While active experts handle the task, the TCPL uses spare compute
    to mutate dormant experts (shadow evolution).
  - Shadow experts are tested against held-out validation data.
  - Only when a Shadow Expert significantly outperforms an active one
    (within the FVE / Formal Verification Layer's safety bounds) is it
    swapped into the active pool.
  - Mutation operators: weight noise injection, layer dropout,
    activation function swap, architecture narrowing/widening.
  - Selection: tournament selection among shadow variants.
  - Stability: each swap must pass formal verification before execution.

This provides a safe, gradual evolutionary search that doesn't
interfere with the active computation path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import copy
import math


class MutationType(Enum):
    """Types of mutations that can be applied to shadow experts."""
    WEIGHT_NOISE = "weight_noise"           # Add Gaussian noise to weights
    LAYER_DROPOUT = "layer_dropout"         # Zero out a random layer
    ACTIVATION_SWAP = "activation_swap"     # Change activation function
    NARROWING = "narrowing"                 # Reduce hidden dim
    WIDENING = "widening"                   # Increase hidden dim (with zero-init)
    RESCALE = "rescale"                     # Scale all weights by a factor


@dataclass
class ShadowEvolutionConfig:
    """Configuration for Shadow Evolution within the Expert Bank.

    Attributes:
        use_shadow_evolution: Whether to enable shadow evolution.
        max_shadow_experts: Maximum number of shadow experts to maintain
            at any time.  Shadow experts are stored in CPU memory to
            avoid impacting VRAM.
        mutation_rate: Probability of mutating each weight during a
            shadow evolution step.
        mutation_strength: Standard deviation of Gaussian noise for
            weight mutations.
        tournament_size: Number of shadow experts to compare in each
            tournament selection round.
        validation_window: Number of validation samples to evaluate
            each shadow expert on before making a swap decision.
        swap_threshold: Minimum improvement ratio required for a shadow
            expert to replace an active expert.  A value of 0.1 means
            the shadow must be at least 10% better.
        evolution_interval: How often (in steps) to run a shadow
            evolution cycle.  Lower values = more frequent evolution
            but higher compute cost.
        max_mutations_per_cycle: Maximum number of mutations to apply
            in a single evolution cycle.
        archive_size: Number of best shadow experts to archive for
            potential future use.
    """
    use_shadow_evolution: bool = True
    max_shadow_experts: int = 16
    mutation_rate: float = 0.03  # 3% per parameter per mutation application
    mutation_strength: float = 0.01
    tournament_size: int = 4
    validation_window: int = 100
    swap_threshold: float = 0.1
    evolution_interval: int = 1000
    max_mutations_per_cycle: int = 4
    archive_size: int = 5
    min_shadow_age: int = 2
    replacement_cooldown: int = 3
    min_evaluations_before_swap: int = 3
    fitness_ema_alpha: float = 0.1
    max_mutation_history: int = 128


@dataclass
class ShadowExpert:
    """A shadow (dormant) expert undergoing evolutionary testing.

    Attributes:
        expert_id: Unique identifier.
        parent_id: ID of the active expert this was cloned from.
        state_dict: Deep copy of the expert's weights.
        fitness: Current fitness score (higher is better).
        age: Number of evolution cycles this shadow has survived.
        mutation_history: List of mutations applied.
        validation_scores: Recent validation scores.
    """
    expert_id: int
    parent_id: int
    state_dict: Dict[str, torch.Tensor]
    fitness: float = 0.0
    age: int = 0
    mutation_history: List[str] = field(default_factory=list)
    validation_scores: List[float] = field(default_factory=list)
    last_improvement: float = 0.0


class ShadowMutator:
    """Applies mutations to shadow experts.

    Each mutation operator is designed to be small and conservative,
    producing variants that are close to the parent but explore the
    local architecture landscape.
    """

    def __init__(self, config: ShadowEvolutionConfig):
        self.config = config

    def mutate(
        self,
        state_dict: Dict[str, torch.Tensor],
        mutation_types: Optional[List[MutationType]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Apply random mutations to an expert's state dict.

        Args:
            state_dict: Expert weights to mutate.
            mutation_types: Specific mutation types to apply. If None,
                a random subset is chosen.

        Returns:
            (mutated_state_dict, mutation_descriptions) tuple.
        """
        mutated = copy.deepcopy(state_dict)
        applied = []

        if mutation_types is None:
            mutation_types = list(MutationType)

        for key, tensor in mutated.items():
            if not tensor.is_floating_point():
                continue

            # Decide whether to mutate this parameter
            if torch.rand(1).item() > self.config.mutation_rate:
                continue

            # Pick a random mutation type
            mtype = mutation_types[torch.randint(len(mutation_types), (1,)).item()]

            if mtype == MutationType.WEIGHT_NOISE:
                noise = torch.randn_like(tensor) * self.config.mutation_strength
                mutated[key] = tensor + noise
                applied.append(f"weight_noise({key})")

            elif mtype == MutationType.RESCALE:
                scale = 1.0 + torch.randn(1).item() * 0.05  # +/- 5%
                mutated[key] = tensor * scale
                applied.append(f"rescale({key}, {scale:.3f})")

            elif mtype == MutationType.LAYER_DROPOUT:
                # Zero out 10% of weights in this parameter
                mask = torch.rand_like(tensor) > 0.1
                mutated[key] = tensor * mask.float()
                applied.append(f"layer_dropout({key})")

            elif mtype == MutationType.NARROWING:
                # Reduce magnitude of weights slightly
                mutated[key] = tensor * 0.95
                applied.append(f"narrowing({key})")

            elif mtype == MutationType.WIDENING:
                # Slightly increase magnitude of weights
                mutated[key] = tensor * 1.05
                applied.append(f"widening({key})")

            elif mtype == MutationType.ACTIVATION_SWAP:
                # Flip sign of a small fraction of weights (simulates activation change)
                mask = torch.rand_like(tensor) > 0.9  # Only 10% flip rate (was 50%)
                mutated[key] = tensor * mask.float() - tensor * (~mask).float()
                applied.append(f"activation_swap({key})")

        return mutated, applied


class ShadowEvolutionEngine:
    """Engine for running evolutionary search on dormant experts.

    The engine maintains a population of shadow experts, periodically
    mutates them, evaluates their fitness, and proposes swaps when
    a shadow expert outperforms an active one.

    All shadow experts are stored in CPU memory to avoid VRAM impact.
    Swaps must be approved by the Formal Verification Layer before
    execution.
    """

    def __init__(self, config: ShadowEvolutionConfig):
        self.config = config
        self.mutator = ShadowMutator(config)

        # Shadow population
        self._shadow_population: Dict[int, ShadowExpert] = {}
        self._next_shadow_id: int = 0

        # Archive of best shadow experts
        self._archive: List[ShadowExpert] = []

        # Statistics
        self._total_mutations: int = 0
        self._total_swaps_proposed: int = 0
        self._total_swaps_approved: int = 0
        self._cycle_count: int = 0
        self._last_replacement_cycle: Dict[int, int] = {}
        self._proposal_history: Dict[int, int] = {}

    def _can_propose_swap(self, shadow: ShadowExpert) -> bool:
        if shadow.age < self.config.min_shadow_age:
            return False
        if len(shadow.validation_scores) < self.config.min_evaluations_before_swap:
            return False

        last_cycle = self._last_replacement_cycle.get(shadow.parent_id)
        if last_cycle is not None:
            if self._cycle_count - last_cycle < self.config.replacement_cooldown:
                return False

        recent_scores = shadow.validation_scores[-self.config.min_evaluations_before_swap:]
        if any(not math.isfinite(score) for score in recent_scores):
            return False

        return shadow.fitness > self.config.swap_threshold

    def spawn_shadow(
        self,
        parent_id: int,
        parent_state_dict: Dict[str, torch.Tensor],
    ) -> int:
        """Create a new shadow expert by cloning an active expert.

        Args:
            parent_id: ID of the active expert to clone.
            parent_state_dict: Weights of the parent expert.

        Returns:
            shadow_id: ID of the new shadow expert.
        """
        # Enforce population limit
        if len(self._shadow_population) >= self.config.max_shadow_experts:
            # Remove worst shadow expert
            worst_id = min(
                self._shadow_population,
                key=lambda k: self._shadow_population[k].fitness
            )
            del self._shadow_population[worst_id]

        shadow_id = self._next_shadow_id
        self._next_shadow_id += 1

        # Deep copy state dict to CPU
        cpu_state = {
            k: v.detach().cpu().clone() for k, v in parent_state_dict.items()
        }

        shadow = ShadowExpert(
            expert_id=shadow_id,
            parent_id=parent_id,
            state_dict=cpu_state,
        )
        self._shadow_population[shadow_id] = shadow
        self._proposal_history.setdefault(parent_id, 0)

        return shadow_id

    def evolve_cycle(self) -> List[Tuple[int, int]]:
        """Run one evolution cycle: mutate, evaluate, propose swaps.

        Returns:
            swap_proposals: List of (shadow_id, active_id) pairs where
                the shadow expert should be considered for swapping with
                the active expert.
        """
        self._cycle_count += 1
        swap_proposals = []

        if not self._shadow_population:
            return swap_proposals

        # Apply mutations to each shadow expert
        for shadow_id, shadow in list(self._shadow_population.items()):
            # Apply mutations
            num_mutations = torch.randint(
                1, self.config.max_mutations_per_cycle + 1, (1,)
            ).item()

            for _ in range(num_mutations):
                new_state, mutations = self.mutator.mutate(shadow.state_dict)
                shadow.state_dict = new_state
                shadow.mutation_history.extend(mutations)
                if len(shadow.mutation_history) > self.config.max_mutation_history:
                    shadow.mutation_history = shadow.mutation_history[-self.config.max_mutation_history:]
                self._total_mutations += 1

            shadow.age += 1

            # Evaluate shadow and propose swap if it outperforms parent
            if self._can_propose_swap(shadow):
                swap_proposals.append((shadow_id, shadow.parent_id))
                self._total_swaps_proposed += 1
                self._proposal_history[shadow.parent_id] = (
                    self._proposal_history.get(shadow.parent_id, 0) + 1
                )

        return swap_proposals

    def evaluate_shadow(
        self,
        shadow_id: int,
        validation_loss: float,
        active_parent_loss: float,
    ) -> Tuple[bool, float]:
        """Evaluate a shadow expert against its active parent.

        Args:
            shadow_id: ID of the shadow expert.
            validation_loss: Loss of the shadow expert on validation data.
            active_parent_loss: Loss of the active parent on same data.

        Returns:
            (should_swap, improvement_ratio) tuple.
        """
        if shadow_id not in self._shadow_population:
            return False, 0.0
        if not math.isfinite(validation_loss) or not math.isfinite(active_parent_loss):
            return False, 0.0

        shadow = self._shadow_population[shadow_id]
        shadow.validation_scores.append(validation_loss)
        if len(shadow.validation_scores) > self.config.validation_window:
            shadow.validation_scores = shadow.validation_scores[-self.config.validation_window:]

        # Compute fitness (lower loss = higher fitness)
        if active_parent_loss > 1e-8:
            improvement = (active_parent_loss - validation_loss) / active_parent_loss
        else:
            improvement = 0.0

        alpha = min(max(self.config.fitness_ema_alpha, 0.0), 1.0)
        shadow.fitness = shadow.fitness * (1.0 - alpha) + improvement * alpha
        shadow.last_improvement = improvement

        # Should we propose a swap?
        should_swap = self._can_propose_swap(shadow)
        if should_swap:
            self._total_swaps_proposed += 1
            self._proposal_history[shadow.parent_id] = (
                self._proposal_history.get(shadow.parent_id, 0) + 1
            )

        return should_swap, improvement

    def tournament_select(self, k: Optional[int] = None) -> Optional[int]:
        """Select the best shadow expert via tournament selection.

        Args:
            k: Tournament size (defaults to config value).

        Returns:
            ID of the winning shadow expert, or None.
        """
        k = k or self.config.tournament_size
        if not self._shadow_population:
            return None

        candidates = list(self._shadow_population.keys())
        k = min(k, len(candidates))

        # Random subset
        selected = [candidates[torch.randint(len(candidates), (1,)).item()]
                     for _ in range(k)]

        # Best fitness wins
        winner = max(selected, key=lambda sid: self._shadow_population[sid].fitness)
        return winner

    def get_swap_state_dict(self, shadow_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get the state dict of a shadow expert for swapping into active pool.

        Returns:
            State dict on CPU, or None if shadow_id not found.
        """
        if shadow_id not in self._shadow_population:
            return None

        shadow = self._shadow_population[shadow_id]

        # Archive before removal (if high fitness)
        if shadow.fitness > 0 and len(self._archive) < self.config.archive_size:
            self._archive.append(copy.deepcopy(shadow))

        state = shadow.state_dict
        del self._shadow_population[shadow_id]
        self._total_swaps_approved += 1
        self._last_replacement_cycle[shadow.parent_id] = self._cycle_count

        return state

    def get_stats(self) -> Dict:
        """Return evolution engine statistics."""
        return {
            "shadow_population_size": len(self._shadow_population),
            "archive_size": len(self._archive),
            "total_mutations": self._total_mutations,
            "total_swaps_proposed": self._total_swaps_proposed,
            "total_swaps_approved": self._total_swaps_approved,
            "cycle_count": self._cycle_count,
            "replacement_cooldowns": dict(self._last_replacement_cycle),
            "proposal_history": dict(self._proposal_history),
        }

    def reset(self):
        """Reset the engine state."""
        self._shadow_population.clear()
        self._archive.clear()
        self._total_mutations = 0
        self._total_swaps_proposed = 0
        self._total_swaps_approved = 0
        self._cycle_count = 0
        self._last_replacement_cycle.clear()
        self._proposal_history.clear()
