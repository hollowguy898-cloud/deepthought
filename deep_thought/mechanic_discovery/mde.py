"""
Mechanic Discovery Engine (MDE) for Deep Thought.

Discovers invariants in action-observation relationships by tracking how
actions consistently affect observations across different contexts.  An
*invariant* (or *mechanic*) is a relationship between an action and an
observation change that remains stable regardless of the surrounding
context.

Architecture
------------
The MDE comprises three cooperating sub-systems:

1. **InvariantDetector** — Maintains a sliding window of
   ``(action, observation, context, reward)`` tuples and groups them by
   action similarity (cosine similarity > 0.9).  For each group it
   computes a *stability score*: the mean pairwise cosine similarity of
   the observation-effect vectors across distinct contexts.  If the
   stability score exceeds a threshold the correlation is flagged as a
   *Universal Mechanic*.

2. **MechanicLabeler** — Generates deterministic symbolic tags for
   discovered invariants (e.g. ``"Mechanic_042"``).  Each tag carries
   a *routing weight adjustment* that biases the Sparse Router toward
   experts that have historically succeeded under that mechanic.

3. **MechanicStore** — Persistent storage for all discovered mechanics,
   their tags, stability histories, and expert associations.  Supports
   O(1) lookups by tag, action signature, context, or expert ID.

The top-level ``MechanicDiscoveryEngine`` orchestrates the three
sub-systems and exposes the main API consumed by the rest of the Deep
Thought system.

Important design choices
~~~~~~~~~~~~~~~~~~~~~~~~
- The MDE is **not** an ``nn.Module`` — it is a pure algorithmic
  component with no learned parameters and no gradient flow.
- All tensors that pass through the MDE are **detached** before any
  computation.
- Expert-mechanic affinities use EMA tracking so that the routing
  hints remain stable yet adaptive.

References
----------
- Schmidhuber, "Developmental Robotics, Optimal Artificial Curiosity,
  Creativity, Music, and the Fine Arts" (2006) — invariant discovery
  as a driver of curiosity.
- Pathak et al., "Curiosity-driven Exploration by Self-supervised
  Prediction" (2017) — action-observation consistency as a learning
  signal.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Invariant:
    """A discovered invariant between an action and an observation effect.

    An invariant represents a consistent relationship where performing a
    particular action produces a predictable change in the observation,
    regardless of the surrounding context.

    Attributes:
        invariant_id: Unique integer identifier.
        action_signature: Hash / embedding of the action pattern.
        observation_effect: The consistent observation change vector
            (mean effect across contexts).
        stability_score: How consistent the correlation is across
            different contexts, in ``[0, 1]``.
        context_span: Number of distinct contexts where the invariant
            has held.
        confidence: Combined metric computed as
            ``stability_score * context_span / threshold``.
        is_universal: ``True`` when ``stability_score`` exceeds the
            configured threshold **and** ``context_span`` is at least
            ``min_context_span``.
        discovery_step: Global step at which the invariant was first
            discovered.
        last_validated_step: Step at which the invariant was last
            re-validated.
        contradicted: Whether the invariant has been found to be noise
            after validation.
        associated_expert_ids: Experts that have succeeded under this
            invariant.
        validation_history: Recent validation scores (rolling window).
    """

    invariant_id: int
    action_signature: torch.Tensor
    observation_effect: torch.Tensor
    stability_score: float
    context_span: int
    confidence: float
    is_universal: bool
    discovery_step: int
    last_validated_step: int
    contradicted: bool
    associated_expert_ids: List[int]
    validation_history: List[float]


@dataclass
class MechanicTag:
    """Symbolic tag attached to a discovered invariant.

    Tags serve as the interface between the MDE and the Sparse Router.
    When a mechanic is active the router uses the tag's
    ``routing_weight_adjustment`` to bias expert selection.

    Attributes:
        tag_id: Human-readable identifier, e.g. ``"Mechanic_042"``.
        invariant_id: Back-reference to the source invariant.
        action_signature_hash: Deterministic string hash of the action
            signature for quick lookup.
        confidence: Invariant confidence at the time of tagging.
        context_span: Number of distinct contexts spanned.
        expert_affinities: Mapping ``expert_id -> success_rate`` under
            this mechanic.  Updated via EMA.
        routing_weight_adjustment: Mapping ``expert_id -> weight bonus``
            for the Sparse Router.
        is_active: Whether the mechanic is currently considered valid.
    """

    tag_id: str
    invariant_id: int
    action_signature_hash: str
    confidence: float
    context_span: int
    expert_affinities: Dict[int, float]
    routing_weight_adjustment: Dict[int, float]
    is_active: bool


@dataclass
class MechanicDiscoveryConfig:
    """Configuration for the Mechanic Discovery Engine.

    Attributes:
        use_mde: Master switch.  When ``False`` all MDE operations
            become no-ops.
        observation_dim: Dimensionality of observation embeddings.
        action_dim: Dimensionality of the action space.
        context_dim: Dimensionality of context embeddings.
        stability_threshold: Stability score above which a candidate
            is flagged as a Universal Mechanic.
        min_context_span: Minimum number of distinct contexts required
            before a candidate can be promoted to invariant.
        window_size: Maximum number of recent steps retained in the
            sliding window.
        validation_interval: Steps between periodic re-validation of
            existing mechanics.
        contradiction_threshold: Validation score below which a
            mechanic is considered noise.
        max_mechanics: Maximum number of tracked mechanics.  When the
            limit is reached, the lowest-confidence mechanic is
            evicted.
        tag_prefix: Prefix for generated tag IDs.
        routing_hint_strength: Scaling factor for routing weight
            adjustments.  Controls how strongly mechanic tags influence
            the Sparse Router.
        expert_affinity_decay: EMA decay for expert-mechanic affinity
            tracking.  Closer to 1.0 → more stable affinities.
    """

    use_mde: bool = True
    observation_dim: int = 64
    action_dim: int = 4
    context_dim: int = 32
    stability_threshold: float = 0.85
    min_context_span: int = 3
    window_size: int = 200
    validation_interval: int = 100
    contradiction_threshold: float = 0.3
    max_mechanics: int = 100
    tag_prefix: str = "Mechanic"
    routing_hint_strength: float = 0.1
    expert_affinity_decay: float = 0.99


# ---------------------------------------------------------------------------
# Internal step record
# ---------------------------------------------------------------------------


@dataclass
class _StepRecord:
    """Internal record stored in the sliding window.

    Attributes:
        action: Action vector (detached).
        observation_before: Observation *before* the action (detached).
        observation_after: Observation *after* the action (detached).
        context: Context vector at the time of the action (detached).
        reward: Reward received after the action.
        step: Global step counter value.
    """

    action: torch.Tensor
    observation_before: torch.Tensor
    observation_after: torch.Tensor
    context: torch.Tensor
    reward: float
    step: int


# ---------------------------------------------------------------------------
# InvariantDetector
# ---------------------------------------------------------------------------


class InvariantDetector:
    """Tracks action-observation pairs and detects stable invariants.

    The detector maintains a sliding window of recent
    ``(action, observation_before, observation_after, context, reward)``
    records.  For each candidate action-observation correlation it
    computes a **stability score** — the mean pairwise cosine similarity
    of the observation-effect vectors across distinct contexts.  When the
    stability score exceeds the configured threshold and the invariant
    spans enough distinct contexts, it is promoted to a *Universal
    Mechanic*.

    The observation effect for a single step is defined as::

        effect = observation_after - observation_before

    Two steps are considered to involve the *same action* when the
    cosine similarity of their action vectors exceeds 0.9.
    """

    # Minimum cosine similarity between two action vectors to be
    # considered the "same action" for grouping purposes.
    _ACTION_SIMILARITY_THRESHOLD: float = 0.9

    def __init__(self, config: MechanicDiscoveryConfig) -> None:
        self.config = config
        # Sliding window of recent step records
        self._window: deque[_StepRecord] = deque(maxlen=config.window_size)
        # Candidate invariants indexed by a representative action signature
        # key = integer candidate_id
        self._candidates: Dict[int, _CandidateGroup] = {}
        self._next_candidate_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_step(
        self,
        action: torch.Tensor,
        observation_before: torch.Tensor,
        observation_after: torch.Tensor,
        context: torch.Tensor,
        reward: float,
        step: int,
    ) -> None:
        """Record a new step into the sliding window.

        All tensors are **detached** before storage to ensure no
        gradient flow through the MDE.

        Args:
            action: Action vector of shape ``(action_dim,)``.
            observation_before: Observation before the action, shape
                ``(observation_dim,)``.
            observation_after: Observation after the action, shape
                ``(observation_dim,)``.
            context: Context vector of shape ``(context_dim,)``.
            reward: Scalar reward.
            step: Global step counter.
        """
        record = _StepRecord(
            action=action.detach().cpu(),
            observation_before=observation_before.detach().cpu(),
            observation_after=observation_after.detach().cpu(),
            context=context.detach().cpu(),
            reward=reward,
            step=step,
        )
        self._window.append(record)

    def detect(self, current_step: int) -> List[Invariant]:
        """Run invariant detection over the current sliding window.

        Groups steps by action similarity, computes stability scores,
        and returns any newly-promoted invariants.

        Args:
            current_step: Current global step (used for
                ``discovery_step``).

        Returns:
            List of newly discovered ``Invariant`` objects.
        """
        new_invariants: List[Invariant] = []

        # Build groups from scratch each time (simpler and correct with
        # sliding window eviction).  For large windows this can be
        # optimised with incremental updates.
        groups = self._group_by_action()

        for action_repr, records in groups.items():
            if len(records) < self.config.min_context_span:
                continue

            # Distinct contexts — two contexts are "distinct" when
            # their cosine similarity < 0.95.
            distinct_contexts = self._count_distinct_contexts(records)
            if distinct_contexts < self.config.min_context_span:
                continue

            # Compute mean observation effect and stability score
            effects = [
                (r.observation_after - r.observation_before) for r in records
            ]
            mean_effect = torch.stack(effects).mean(dim=0)
            stability = self._compute_stability(effects)

            # Confidence metric
            confidence = (
                stability * distinct_contexts / self.config.stability_threshold
            )

            is_universal = (
                stability >= self.config.stability_threshold
                and distinct_contexts >= self.config.min_context_span
            )

            # Only create an invariant for universal mechanics
            if is_universal:
                inv = Invariant(
                    invariant_id=self._next_candidate_id,
                    action_signature=action_repr.detach(),
                    observation_effect=mean_effect.detach(),
                    stability_score=stability,
                    context_span=distinct_contexts,
                    confidence=confidence,
                    is_universal=is_universal,
                    discovery_step=current_step,
                    last_validated_step=current_step,
                    contradicted=False,
                    associated_expert_ids=[],
                    validation_history=[stability],
                )
                self._next_candidate_id += 1
                new_invariants.append(inv)

        return new_invariants

    def validate(
        self,
        invariant: Invariant,
        new_observation: torch.Tensor,
        new_reward: float,
        current_step: int,
    ) -> float:
        """Re-validate an invariant with fresh observation data.

        Computes the cosine similarity between the stored observation
        effect and the new observation as a proxy for whether the
        mechanic still applies.  A low score signals that the mechanic
        may be noise.

        Args:
            invariant: The invariant to validate.
            new_observation: The latest observation vector.
            new_reward: Reward accompanying the new observation.
            current_step: Current global step.

        Returns:
            Validation score in ``[0, 1]``.  Values below
            ``contradiction_threshold`` indicate noise.
        """
        with torch.no_grad():
            new_obs = new_observation.detach().cpu()
            effect = invariant.observation_effect.detach().cpu()

            # Cosine similarity between the stored effect and the new
            # observation (as a proxy for whether the effect still
            # applies).
            effect_norm = effect.norm()
            obs_norm = new_obs.norm()
            if effect_norm < 1e-8 or obs_norm < 1e-8:
                val_score = 0.0
            else:
                val_score = F.cosine_similarity(
                    effect.unsqueeze(0), new_obs.unsqueeze(0), dim=-1
                ).item()
                val_score = max(0.0, min(1.0, val_score))

        # Update validation history (keep last 20 scores)
        invariant.validation_history.append(val_score)
        if len(invariant.validation_history) > 20:
            invariant.validation_history = invariant.validation_history[-20:]

        # Update stability as running mean of validation scores
        if invariant.validation_history:
            invariant.stability_score = sum(invariant.validation_history) / len(
                invariant.validation_history
            )

        # Check for contradiction
        if val_score < self.config.contradiction_threshold:
            invariant.contradicted = True

        invariant.last_validated_step = current_step
        return val_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_by_action(
        self,
    ) -> Dict[torch.Tensor, List[_StepRecord]]:
        """Group step records by action similarity.

        Uses greedy clustering: iterate through records and assign each
        to the first existing group whose representative action has
        cosine similarity > ``_ACTION_SIMILARITY_THRESHOLD``.  If no
        match is found, start a new group.

        Returns:
            Mapping from representative action tensor to list of records.
        """
        groups: Dict[torch.Tensor, List[_StepRecord]] = {}

        for record in self._window:
            assigned = False
            for repr_action in groups:
                sim = F.cosine_similarity(
                    repr_action.unsqueeze(0),
                    record.action.unsqueeze(0),
                    dim=-1,
                ).item()
                if sim > self._ACTION_SIMILARITY_THRESHOLD:
                    groups[repr_action].append(record)
                    assigned = True
                    break

            if not assigned:
                groups[record.action] = [record]

        return groups

    @staticmethod
    def _count_distinct_contexts(records: List[_StepRecord]) -> int:
        """Count the number of distinct contexts in a group.

        Two contexts are considered *distinct* when their cosine
        similarity is below 0.95.
        """
        if not records:
            return 0

        distinct: List[torch.Tensor] = []
        for r in records:
            is_distinct = True
            for existing in distinct:
                sim = F.cosine_similarity(
                    existing.unsqueeze(0),
                    r.context.unsqueeze(0),
                    dim=-1,
                ).item()
                if sim >= 0.95:
                    is_distinct = False
                    break
            if is_distinct:
                distinct.append(r.context)

        return len(distinct)

    @staticmethod
    def _compute_stability(effects: List[torch.Tensor]) -> float:
        """Compute stability score for a list of observation-effect vectors.

        The stability score is the mean pairwise cosine similarity of
        all effect vectors.  A score close to 1.0 means the effects are
        nearly identical; a score close to 0.0 means they are
        orthogonal or inconsistent.

        For efficiency, when there are many effects we subsample to at
        most 50 pairwise comparisons.

        Args:
            effects: List of effect vectors.

        Returns:
            Mean pairwise cosine similarity in ``[0, 1]``.
        """
        if len(effects) < 2:
            return 0.0

        n = len(effects)
        # Subsample for efficiency
        max_pairs = 50
        pair_indices: List[Tuple[int, int]] = []
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_indices.append((i, j))
                count += 1
                if count >= max_pairs:
                    break
            if count >= max_pairs:
                break

        similarities: List[float] = []
        for i, j in pair_indices:
            e_i = effects[i]
            e_j = effects[j]
            norm_i = e_i.norm()
            norm_j = e_j.norm()
            if norm_i < 1e-8 or norm_j < 1e-8:
                similarities.append(0.0)
            else:
                sim = F.cosine_similarity(
                    e_i.unsqueeze(0), e_j.unsqueeze(0), dim=-1
                ).item()
                similarities.append(max(0.0, sim))

        return sum(similarities) / len(similarities) if similarities else 0.0


# ---------------------------------------------------------------------------
# Internal candidate group (used during detection)
# ---------------------------------------------------------------------------


@dataclass
class _CandidateGroup:
    """Temporary storage for a candidate invariant during detection.

    Attributes:
        representative_action: The action vector that represents this
            group.
        records: Step records belonging to this group.
    """

    representative_action: torch.Tensor
    records: List[_StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MechanicLabeler
# ---------------------------------------------------------------------------


class MechanicLabeler:
    """Generates symbolic tags for discovered invariants.

    Each tag carries a deterministic ID (e.g. ``"Mechanic_042"``), a
    string hash of the action signature for quick lookups, and routing
    weight adjustments derived from expert-mechanic affinities.

    The labeler also maintains the mapping ``mechanic_tag ->
    expert_ids`` that guides the Sparse Router toward experts with a
    proven track record under a given mechanic.
    """

    def __init__(self, config: MechanicDiscoveryConfig) -> None:
        self.config = config
        # Mapping: tag_id -> MechanicTag
        self._tags: Dict[str, MechanicTag] = {}
        # Running counter for deterministic tag generation
        self._tag_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, invariant: Invariant) -> MechanicTag:
        """Create a symbolic tag for an invariant.

        The tag ID is deterministic: ``"{prefix}_{invariant_id:03d}"``.
        The action signature hash is derived from the tensor bytes via
        SHA-256 for collision resistance.

        Args:
            invariant: The invariant to label.

        Returns:
            A new ``MechanicTag`` instance.
        """
        tag_id = f"{self.config.tag_prefix}_{invariant.invariant_id:03d}"
        action_hash = self._hash_action_signature(invariant.action_signature)

        tag = MechanicTag(
            tag_id=tag_id,
            invariant_id=invariant.invariant_id,
            action_signature_hash=action_hash,
            confidence=invariant.confidence,
            context_span=invariant.context_span,
            expert_affinities={},
            routing_weight_adjustment={},
            is_active=True,
        )
        self._tags[tag_id] = tag
        self._tag_counter += 1
        return tag

    def update_expert_affinity(
        self,
        tag_id: str,
        expert_id: int,
        success: float,
    ) -> None:
        """Update the EMA affinity between a mechanic and an expert.

        When an expert produces a good outcome while a mechanic is
        active, call this method with a high ``success`` value.  The
        affinity is updated via exponential moving average::

            affinity = decay * affinity + (1 - decay) * success

        The routing weight adjustment is then recomputed as::

            adjustment = hint_strength * affinity

        Args:
            tag_id: The mechanic tag to update.
            expert_id: The expert whose affinity should change.
            success: Success signal in ``[0, 1]``.
        """
        if tag_id not in self._tags:
            return

        tag = self._tags[tag_id]
        decay = self.config.expert_affinity_decay

        old_affinity = tag.expert_affinities.get(expert_id, 0.0)
        new_affinity = decay * old_affinity + (1 - decay) * success
        tag.expert_affinities[expert_id] = new_affinity

        # Recompute routing weight adjustment
        tag.routing_weight_adjustment[expert_id] = (
            self.config.routing_hint_strength * new_affinity
        )

    def get_tag(self, tag_id: str) -> Optional[MechanicTag]:
        """Retrieve a tag by its ID.

        Args:
            tag_id: The tag identifier.

        Returns:
            The ``MechanicTag`` if it exists, else ``None``.
        """
        return self._tags.get(tag_id)

    def deactivate_tag(self, tag_id: str) -> None:
        """Mark a tag as inactive (e.g. after contradiction).

        Inactive tags are not included in routing hints.

        Args:
            tag_id: The tag to deactivate.
        """
        if tag_id in self._tags:
            self._tags[tag_id].is_active = False

    def get_active_tags(self) -> List[MechanicTag]:
        """Return all currently active tags.

        Returns:
            List of active ``MechanicTag`` instances.
        """
        return [t for t in self._tags.values() if t.is_active]

    def get_tag_for_invariant(self, invariant_id: int) -> Optional[MechanicTag]:
        """Look up the tag associated with an invariant.

        Args:
            invariant_id: The invariant to look up.

        Returns:
            The corresponding ``MechanicTag`` if found, else ``None``.
        """
        for tag in self._tags.values():
            if tag.invariant_id == invariant_id:
                return tag
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_action_signature(action_signature: torch.Tensor) -> str:
        """Compute a deterministic string hash of an action signature.

        Uses SHA-256 over the raw tensor bytes for collision resistance.

        Args:
            action_signature: The action tensor.

        Returns:
            Hex digest string (first 16 characters).
        """
        with torch.no_grad():
            data = action_signature.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]


# ---------------------------------------------------------------------------
# MechanicStore
# ---------------------------------------------------------------------------


class MechanicStore:
    """Persistent storage for discovered mechanics.

    Stores all discovered mechanics with their tags, stability scores,
    and histories.  Supports efficient lookups by tag, action signature,
    context, or expert ID.

    The store provides a *mechanic context vector* — a summary of all
    active mechanics — that the Sparse Router can consume.
    """

    def __init__(self, config: MechanicDiscoveryConfig) -> None:
        self.config = config
        # Primary storage: invariant_id -> Invariant
        self._invariants: Dict[int, Invariant] = {}
        # Secondary indices
        self._tag_to_invariant: Dict[str, int] = {}  # tag_id -> invariant_id
        self._action_hash_to_invariants: Dict[str, Set[int]] = (
            {}
        )  # hash -> {inv_ids}
        self._expert_to_invariants: Dict[int, Set[int]] = (
            {}
        )  # expert_id -> {inv_ids}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, invariant: Invariant, tag: MechanicTag) -> None:
        """Store an invariant and its tag.

        If the store is at capacity the lowest-confidence invariant is
        evicted first.

        Args:
            invariant: The invariant to store.
            tag: The associated mechanic tag.
        """
        # Evict if at capacity
        if len(self._invariants) >= self.config.max_mechanics:
            self._evict_lowest_confidence()

        self._invariants[invariant.invariant_id] = invariant
        self._tag_to_invariant[tag.tag_id] = invariant.invariant_id
        self._action_hash_to_invariants.setdefault(
            tag.action_signature_hash, set()
        ).add(invariant.invariant_id)

        # Update expert index
        for expert_id in invariant.associated_expert_ids:
            self._expert_to_invariants.setdefault(expert_id, set()).add(
                invariant.invariant_id
            )

    def get_by_invariant_id(self, invariant_id: int) -> Optional[Invariant]:
        """Look up an invariant by its ID.

        Args:
            invariant_id: The invariant identifier.

        Returns:
            The ``Invariant`` if found, else ``None``.
        """
        return self._invariants.get(invariant_id)

    def get_by_tag(self, tag_id: str) -> Optional[Invariant]:
        """Look up an invariant by its tag ID.

        Args:
            tag_id: The mechanic tag identifier.

        Returns:
            The ``Invariant`` if found, else ``None``.
        """
        inv_id = self._tag_to_invariant.get(tag_id)
        if inv_id is None:
            return None
        return self._invariants.get(inv_id)

    def get_by_action_hash(self, action_hash: str) -> List[Invariant]:
        """Look up invariants by action signature hash.

        Args:
            action_hash: SHA-256 hash of the action signature.

        Returns:
            List of matching ``Invariant`` objects.
        """
        inv_ids = self._action_hash_to_invariants.get(action_hash, set())
        return [
            self._invariants[iid]
            for iid in inv_ids
            if iid in self._invariants
        ]

    def get_by_expert(self, expert_id: int) -> List[Invariant]:
        """Look up invariants associated with a specific expert.

        Args:
            expert_id: The expert identifier.

        Returns:
            List of ``Invariant`` objects associated with that expert.
        """
        inv_ids = self._expert_to_invariants.get(expert_id, set())
        return [
            self._invariants[iid]
            for iid in inv_ids
            if iid in self._invariants
        ]

    def get_active_invariants(self) -> List[Invariant]:
        """Return all active (non-contradicted, universal) invariants.

        Returns:
            List of active ``Invariant`` objects.
        """
        return [
            inv
            for inv in self._invariants.values()
            if inv.is_universal and not inv.contradicted
        ]

    def get_mechanic_context_vector(
        self, context_dim: Optional[int] = None
    ) -> torch.Tensor:
        """Build a summary context vector from all active mechanics.

        The vector is the mean of all active observation-effect vectors,
        zero-padded or truncated to ``context_dim``.

        Args:
            context_dim: Desired output dimension.  Defaults to
                ``config.context_dim``.

        Returns:
            Tensor of shape ``(context_dim,)``.
        """
        if context_dim is None:
            context_dim = self.config.context_dim

        active = self.get_active_invariants()
        if not active:
            return torch.zeros(context_dim)

        effects = [inv.observation_effect for inv in active]
        mean_effect = torch.stack(effects).mean(dim=0)

        # Resize to context_dim
        if mean_effect.shape[0] < context_dim:
            padding = torch.zeros(
                context_dim - mean_effect.shape[0],
            )
            result = torch.cat([mean_effect, padding])
        elif mean_effect.shape[0] > context_dim:
            result = mean_effect[:context_dim]
        else:
            result = mean_effect

        return result.detach()

    def update_expert_association(
        self, invariant_id: int, expert_id: int
    ) -> None:
        """Associate an expert with an invariant.

        Args:
            invariant_id: The invariant to update.
            expert_id: The expert to associate.
        """
        inv = self._invariants.get(invariant_id)
        if inv is None:
            return

        if expert_id not in inv.associated_expert_ids:
            inv.associated_expert_ids.append(expert_id)
            self._expert_to_invariants.setdefault(expert_id, set()).add(
                invariant_id
            )

    def remove_expert_association(
        self, invariant_id: int, expert_id: int
    ) -> None:
        """Remove an expert association from an invariant.

        Args:
            invariant_id: The invariant to update.
            expert_id: The expert to disassociate.
        """
        inv = self._invariants.get(invariant_id)
        if inv is None:
            return

        if expert_id in inv.associated_expert_ids:
            inv.associated_expert_ids.remove(expert_id)
            if expert_id in self._expert_to_invariants:
                self._expert_to_invariants[expert_id].discard(invariant_id)

    @property
    def num_invariants(self) -> int:
        """Total number of stored invariants."""
        return len(self._invariants)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_lowest_confidence(self) -> None:
        """Evict the invariant with the lowest confidence score.

        Also cleans up secondary indices.
        """
        if not self._invariants:
            return

        min_id = min(
            self._invariants, key=lambda k: self._invariants[k].confidence
        )
        inv = self._invariants.pop(min_id)

        # Clean up tag index
        tags_to_remove = [
            tid
            for tid, iid in self._tag_to_invariant.items()
            if iid == min_id
        ]
        for tid in tags_to_remove:
            del self._tag_to_invariant[tid]

        # Clean up action hash index
        for hash_key, inv_ids in self._action_hash_to_invariants.items():
            inv_ids.discard(min_id)

        # Clean up expert index
        for expert_id in inv.associated_expert_ids:
            if expert_id in self._expert_to_invariants:
                self._expert_to_invariants[expert_id].discard(min_id)


# ---------------------------------------------------------------------------
# MechanicDiscoveryEngine
# ---------------------------------------------------------------------------


class MechanicDiscoveryEngine:
    """Top-level orchestrator for the Mechanic Discovery Engine.

    Coordinates the ``InvariantDetector``, ``MechanicLabeler``, and
    ``MechanicStore`` to process environment steps, discover mechanics,
    and provide routing hints to the Sparse Router.

    The engine is a **pure algorithmic component** — it is *not* an
    ``nn.Module`` and has no learned parameters.  All tensors that enter
    the engine are immediately detached.

    Example::

        config = MechanicDiscoveryConfig(
            observation_dim=64,
            action_dim=4,
            context_dim=32,
        )
        mde = MechanicDiscoveryEngine(config)

        # During interaction loop:
        active_tags, hints = mde.process_step(
            action=action_tensor,
            observation=obs_tensor,
            context=ctx_tensor,
            reward=0.5,
        )
    """

    def __init__(self, config: Optional[MechanicDiscoveryConfig] = None) -> None:
        if config is None:
            config = MechanicDiscoveryConfig()
        self.config = config

        self._detector = InvariantDetector(config)
        self._labeler = MechanicLabeler(config)
        self._store = MechanicStore(config)

        # Bookkeeping
        self._step_counter: int = 0
        # Previous observation for computing observation deltas
        self._prev_observation: Optional[torch.Tensor] = None
        # Cache of active tags from the most recent detection pass
        self._active_tags: List[MechanicTag] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_step(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        context: torch.Tensor,
        reward: float,
    ) -> Tuple[List[MechanicTag], Dict[int, float]]:
        """Process a single environment step.

        1. Records the step in the sliding window.
        2. Runs invariant detection if enough data has accumulated.
        3. Labels any newly discovered invariants.
        4. Periodically re-validates existing mechanics.
        5. Returns active mechanic tags and routing hints.

        Args:
            action: Action vector of shape ``(action_dim,)``.
            observation: Current observation of shape
                ``(observation_dim,)``.
            context: Current context of shape ``(context_dim,)``.
            reward: Scalar reward.

        Returns:
            A tuple ``(active_tags, routing_hints)`` where:

            - ``active_tags`` is a list of currently active
              ``MechanicTag`` objects relevant to this step.
            - ``routing_hints`` is a dict mapping ``expert_id`` to a
              weight adjustment (float) for the Sparse Router.
        """
        if not self.config.use_mde:
            return [], {}

        # Detach all inputs
        action = action.detach()
        observation = observation.detach()
        context = context.detach()

        # We need a "before" and "after" observation.  On the first
        # call we only have the current observation; use it as the
        # "before" for the next step.
        if self._prev_observation is not None:
            self._detector.add_step(
                action=action,
                observation_before=self._prev_observation,
                observation_after=observation,
                context=context,
                reward=reward,
                step=self._step_counter,
            )
        self._prev_observation = observation.clone()

        self._step_counter += 1

        # --- Invariant detection (every few steps to amortise cost) ---
        new_invariants: List[Invariant] = []
        if (
            self._step_counter >= self.config.min_context_span * 2
            and self._step_counter % 5 == 0
        ):
            new_invariants = self._detector.detect(self._step_counter)

        # --- Label and store new invariants ---
        for inv in new_invariants:
            tag = self._labeler.label(inv)
            self._store.store(inv, tag)

        # --- Periodic re-validation ---
        if self._step_counter % self.config.validation_interval == 0:
            self._run_validation(observation, reward)

        # --- Determine active tags for this step ---
        self._active_tags = self._find_active_tags(action)
        routing_hints = self._compute_routing_hints(self._active_tags)

        return self._active_tags, routing_hints

    def get_routing_hints(
        self, mechanic_tags: List[MechanicTag]
    ) -> Dict[int, float]:
        """Compute expert weight adjustments based on active mechanics.

        Aggregates the ``routing_weight_adjustment`` from all provided
        tags.  When multiple tags recommend a weight for the same
        expert, the adjustments are **summed**.

        Args:
            mechanic_tags: List of active mechanic tags.

        Returns:
            Mapping ``expert_id -> total_weight_adjustment``.
        """
        return self._compute_routing_hints(mechanic_tags)

    def validate_mechanic(
        self,
        invariant_id: int,
        new_observation: torch.Tensor,
        new_reward: float,
    ) -> Optional[float]:
        """Re-validate a mechanic with new observation data.

        If the validation score falls below ``contradiction_threshold``
        the mechanic is flagged as noise.

        Args:
            invariant_id: The invariant to validate.
            new_observation: Fresh observation vector.
            new_reward: Reward accompanying the observation.

        Returns:
            Validation score in ``[0, 1]``, or ``None`` if the
            invariant does not exist.
        """
        inv = self._store.get_by_invariant_id(invariant_id)
        if inv is None:
            return None

        score = self._detector.validate(
            inv, new_observation, new_reward, self._step_counter
        )

        # If contradicted, deactivate the corresponding tag
        if inv.contradicted:
            tag = self._labeler.get_tag_for_invariant(invariant_id)
            if tag is not None:
                self._labeler.deactivate_tag(tag.tag_id)

        return score

    def contradict_mechanic(self, invariant_id: int) -> bool:
        """Explicitly contradict a mechanic.

        Marks the invariant as contradicted and deactivates its tag.
        This triggers the SRP (Self-Regression Prevention) to prune
        the associated expert.

        Args:
            invariant_id: The invariant to contradict.

        Returns:
            ``True`` if the invariant was found and contradicted,
            ``False`` otherwise.
        """
        inv = self._store.get_by_invariant_id(invariant_id)
        if inv is None:
            return False

        inv.contradicted = True
        inv.is_universal = False

        # Deactivate the corresponding tag
        tag = self._labeler.get_tag_for_invariant(invariant_id)
        if tag is not None:
            self._labeler.deactivate_tag(tag.tag_id)

        return True

    def record_expert_success(
        self,
        mechanic_tag_id: str,
        expert_id: int,
        success: float,
    ) -> None:
        """Record that an expert succeeded under a mechanic.

        Updates the EMA affinity between the mechanic and the expert,
        and ensures the expert is associated with the invariant in the
        store.

        Args:
            mechanic_tag_id: The tag whose affinity should be updated.
            expert_id: The expert that succeeded.
            success: Success signal in ``[0, 1]``.
        """
        self._labeler.update_expert_affinity(mechanic_tag_id, expert_id, success)

        tag = self._labeler.get_tag(mechanic_tag_id)
        if tag is not None:
            self._store.update_expert_association(tag.invariant_id, expert_id)

    def get_mechanic_context_vector(self) -> torch.Tensor:
        """Get a summary context vector from all active mechanics.

        Useful for the Sparse Router as an additional input signal.

        Returns:
            Tensor of shape ``(config.context_dim,)``.
        """
        return self._store.get_mechanic_context_vector()

    @property
    def step_counter(self) -> int:
        """Current global step counter."""
        return self._step_counter

    @property
    def num_mechanics(self) -> int:
        """Total number of stored mechanics."""
        return self._store.num_invariants

    def get_stats(self) -> Dict[str, object]:
        """Return diagnostic statistics about the MDE.

        Returns:
            Dictionary with counts and scores.
        """
        active_invs = self._store.get_active_invariants()
        active_tags = self._labeler.get_active_tags()

        avg_stability = (
            sum(inv.stability_score for inv in active_invs) / len(active_invs)
            if active_invs
            else 0.0
        )
        avg_confidence = (
            sum(inv.confidence for inv in active_invs) / len(active_invs)
            if active_invs
            else 0.0
        )

        return {
            "step": self._step_counter,
            "total_invariants": self._store.num_invariants,
            "active_invariants": len(active_invs),
            "active_tags": len(active_tags),
            "contradicted_invariants": sum(
                1 for inv in self._store._invariants.values() if inv.contradicted
            ),
            "avg_stability": avg_stability,
            "avg_confidence": avg_confidence,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_active_tags(
        self, action: torch.Tensor
    ) -> List[MechanicTag]:
        """Determine which mechanic tags are relevant for the current action.

        A tag is considered *active* if:
        1. It is marked as active in the labeler, **and**
        2. The cosine similarity between the current action and the
           stored action signature exceeds 0.9.

        Args:
            action: Current action vector (already detached).

        Returns:
            List of matching ``MechanicTag`` objects.
        """
        active_tags = self._labeler.get_active_tags()
        matching: List[MechanicTag] = []

        for tag in active_tags:
            inv = self._store.get_by_tag(tag.tag_id)
            if inv is None:
                continue

            # Compare action to stored signature
            sig = inv.action_signature
            if sig.norm() < 1e-8 or action.norm() < 1e-8:
                continue

            sim = F.cosine_similarity(
                sig.unsqueeze(0), action.unsqueeze(0), dim=-1
            ).item()

            if sim > 0.9:
                matching.append(tag)

        return matching

    def _compute_routing_hints(
        self, tags: List[MechanicTag]
    ) -> Dict[int, float]:
        """Aggregate routing weight adjustments from multiple tags.

        Args:
            tags: Active mechanic tags.

        Returns:
            Mapping ``expert_id -> total_weight_adjustment``.
        """
        hints: Dict[int, float] = {}
        for tag in tags:
            for expert_id, adjustment in tag.routing_weight_adjustment.items():
                hints[expert_id] = hints.get(expert_id, 0.0) + adjustment
        return hints

    def _run_validation(
        self, current_observation: torch.Tensor, current_reward: float
    ) -> None:
        """Re-validate all active invariants.

        Called periodically according to ``validation_interval``.

        Args:
            current_observation: Latest observation vector.
            current_reward: Latest reward.
        """
        active_invs = self._store.get_active_invariants()
        for inv in active_invs:
            self.validate_mechanic(
                inv.invariant_id, current_observation, current_reward
            )
