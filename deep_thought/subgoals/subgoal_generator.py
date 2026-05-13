"""
Self-Generated Subgoal System for Deep Thought.

Instead of merely maximizing external reward, this system invents intermediate
objectives — *subgoals* — that structure the agent's behavior across time.
Subgoals act as stepping stones that decompose the vast space of possible
behaviors into manageable, learnable chunks.

Supported subgoal types:
    - **explore**  : Scout unknown territory, discover new states.
    - **secure**   : Ensure an escape path or safety margin.
    - **conserve** : Preserve resources (energy, health, ammo, etc.).
    - **reduce_uncertainty** : Actively reduce epistemic uncertainty about
      the environment or opponent.
    - **probe**    : Deliberately test opponent strategy or environment
      dynamics.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  SubgoalEncoder                                       │
    │  Encodes current state into a goal-oriented embedding │
    │  space.                                               │
    ├──────────────────────────────────────────────────────┤
    │  SubgoalProposer                                      │
    │  Proposes candidate subgoals from the current state   │
    │  context. Uses learned proposal networks conditioned  │
    │  on subgoal type.                                     │
    ├──────────────────────────────────────────────────────┤
    │  SubgoalEvaluator                                     │
    │  Evaluates how useful a subgoal is via a learned      │
    │  value function over (state, subgoal) pairs.          │
    ├──────────────────────────────────────────────────────┤
    │  SubgoalDecomposer                                    │
    │  Decomposes high-level goals into sequences of lower- │
    │  level subgoals with parent-child relationships.      │
    └──────────────────────────────────────────────────────┘

Processing Flow:
    1. The agent's current state (latent h_t, observation x_t, reward,
       uncertainty, episode progress) is fed to ``forward()``.
    2. ``SubgoalEncoder`` projects the state into a goal-aware embedding.
    3. ``SubgoalProposer`` generates a set of candidate subgoals.
    4. ``SubgoalEvaluator`` scores each candidate; the best-scoring
       subgoal becomes the *active* subgoal.
    5. ``SubgoalDecomposer`` can further decompose the active subgoal
       into a sequence of child subgoals.
    6. An intrinsic reward signal is produced for progress toward the
       active subgoal, supplementing the external reward.

All sub-modules are implemented as ``nn.Module`` and are trained jointly
with the rest of the Deep Thought system.

References:
    - Nachum et al., "Data-Efficient Hierarchical Reinforcement Learning"
      (2018)
    - Vezhnevets et al., "FeUdal Networks for Hierarchical RL" (2017)
    - Levy et al., "Learning Multi-Level Hierarchies with HINDSIGHT"
      (2019)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_thought.config import SubgoalConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Subgoal:
    """
    A single self-generated subgoal.

    Attributes:
        goal_vector: Latent tensor representing the subgoal in an abstract
            goal space.  Shape ``(goal_embedding_dim,)``.
        goal_type: Human-readable type label, e.g. ``"explore"``,
            ``"secure"``, ``"conserve"``, ``"reduce_uncertainty"``,
            ``"probe"``.
        priority: Scalar priority score — higher means more urgent.
        completion_estimate: Estimated number of environment steps
            required to complete this subgoal.
        parent_id: ID of the parent subgoal if this was produced by
            decomposition, or ``None`` for top-level subgoals.
        children: List of child subgoal IDs (populated when the subgoal
            is decomposed).
        subgoal_id: Unique integer identifier assigned upon creation.
        progress: Current completion progress in ``[0, 1]``.
    """

    goal_vector: torch.Tensor
    goal_type: str
    priority: float
    completion_estimate: float
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)
    subgoal_id: int = 0
    progress: float = 0.0


# ---------------------------------------------------------------------------
# Sub-module: SubgoalEncoder
# ---------------------------------------------------------------------------


class SubgoalEncoder(nn.Module):
    """
    Encodes the current agent state into a goal-oriented embedding space.

    The encoder concatenates:
      - ``h_t``        : latent hidden state
      - ``x_t``        : encoded observation
      - reward signal (scalar)
      - uncertainty signal (scalar)
      - episode progress (scalar)

    and projects the result through a multi-layer MLP with SiLU activation
    and LayerNorm, producing a ``goal_embedding_dim``-dimensional vector
    that captures *what matters for deciding subgoals*.

    Args:
        latent_dim: Dimensionality of ``h_t`` and ``x_t``.
        goal_embedding_dim: Output dimension of the goal-aware embedding.
        hidden_dim: Width of intermediate MLP layers.
    """

    def __init__(
        self,
        latent_dim: int,
        goal_embedding_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        # Input: h_t (latent_dim) + x_t (latent_dim) + reward (1)
        #        + uncertainty (1) + progress (1)
        input_dim = latent_dim * 2 + 3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, goal_embedding_dim),
        )

    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        reward: torch.Tensor,
        uncertainty: torch.Tensor,
        episode_progress: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode the current state into a goal-aware embedding.

        Args:
            h_t: Hidden state tensor, shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.
            x_t: Encoded observation, same shape constraints as ``h_t``.
            reward: Scalar or per-sample reward, shape ``()`` or
                ``(batch,)`` or ``(batch, 1)``.
            uncertainty: Scalar or per-sample uncertainty, same shape
                constraints as ``reward``.
            episode_progress: Scalar or per-sample progress in ``[0, 1]``,
                same shape constraints as ``reward``.

        Returns:
            Goal-aware state embedding of shape ``(batch,
            goal_embedding_dim)`` or ``(goal_embedding_dim,)``.
        """
        # Ensure batch dimension on h_t / x_t
        squeeze = False
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
            x_t = x_t.unsqueeze(0)
            squeeze = True

        batch_size = h_t.size(0)
        device = h_t.device

        # Ensure correct shape for scalars: (batch, 1)
        def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                return t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
            if t.dim() == 1:
                return t.unsqueeze(1)
            return t  # already (batch, 1) or (batch, k)

        reward = _ensure_2d(reward)
        uncertainty = _ensure_2d(uncertainty)
        episode_progress = _ensure_2d(episode_progress)

        combined = torch.cat([h_t, x_t, reward, uncertainty, episode_progress], dim=-1)
        embedding = self.encoder(combined)

        if squeeze:
            embedding = embedding.squeeze(0)
        return embedding


# ---------------------------------------------------------------------------
# Sub-module: SubgoalProposer
# ---------------------------------------------------------------------------


class SubgoalProposer(nn.Module):
    """
    Proposes candidate subgoals from a goal-aware state embedding.

    For each supported subgoal type, a separate *type head* projects the
    state embedding into a goal vector of dimension ``goal_embedding_dim``.
    A shared *priority head`` scores each candidate, and a *completion
    estimate head* predicts how many steps the subgoal will take.

    The proposer also maintains learnable *type embeddings* — one per
    subgoal type — that are added to the output goal vector so that
    the downstream evaluator can distinguish subgoal types without
    relying solely on the geometry of the goal vector.

    Args:
        goal_embedding_dim: Dimension of the goal embedding and output
            goal vectors.
        num_subgoal_types: Number of distinct subgoal types.
        hidden_dim: Width of internal MLP layers.
    """

    def __init__(
        self,
        goal_embedding_dim: int = 256,
        num_subgoal_types: int = 5,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.goal_embedding_dim = goal_embedding_dim
        self.num_subgoal_types = num_subgoal_types

        # Per-type goal vector heads
        self.type_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(goal_embedding_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, goal_embedding_dim),
            )
            for _ in range(num_subgoal_types)
        ])

        # Shared priority scorer: (goal_embedding) -> scalar priority
        self.priority_head = nn.Sequential(
            nn.Linear(goal_embedding_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Shared completion estimator: (goal_embedding) -> scalar steps
        self.completion_head = nn.Sequential(
            nn.Linear(goal_embedding_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # ensure positive
        )

        # Learnable type embeddings — added to goal vectors
        self.type_embeddings = nn.Parameter(
            torch.randn(num_subgoal_types, goal_embedding_dim) * 0.1
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        num_candidates: int = 5,
    ) -> List[Subgoal]:
        """
        Propose a list of candidate subgoals.

        For each of the ``num_subgoal_types`` (or ``num_candidates``,
        whichever is smaller), the method:
          1. Projects the state embedding through the corresponding
             type head to get a goal vector.
          2. Adds the learnable type embedding.
          3. Scores priority and estimates completion steps.

        Args:
            state_embedding: Goal-aware state embedding from
                :class:`SubgoalEncoder`, shape ``(batch,
                goal_embedding_dim)`` or ``(goal_embedding_dim,)``.
            num_candidates: Maximum number of candidates to propose.
                Capped at ``num_subgoal_types``.

        Returns:
            List of :class:`Subgoal` objects (one per type, up to
            ``num_candidates``).  All tensors inside are detached from
            the computation graph for storage; gradients flow through
            the proposer during training via the loss function.
        """
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        num_candidates = min(num_candidates, self.num_subgoal_types)
        candidates: List[Subgoal] = []

        for i in range(num_candidates):
            # Goal vector for this type
            goal_vec = self.type_heads[i](state_embedding)  # (batch, goal_dim)
            # Add type embedding (broadcast over batch)
            goal_vec = goal_vec + self.type_embeddings[i].unsqueeze(0)

            # Priority score (batch, 1)
            priority = self.priority_head(goal_vec).squeeze(-1)  # (batch,)

            # Completion estimate (batch, 1)
            completion = self.completion_head(goal_vec).squeeze(-1)  # (batch,)

            # Take the first (and possibly only) batch element
            if squeeze or state_embedding.size(0) == 1:
                goal_vec_out = goal_vec[0].detach()
                priority_out = float(priority[0].item())
                completion_out = float(completion[0].item())
            else:
                goal_vec_out = goal_vec.detach()
                priority_out = float(priority.mean().item())
                completion_out = float(completion.mean().item())

            candidates.append(
                Subgoal(
                    goal_vector=goal_vec_out,
                    goal_type=self._type_index_to_name(i),
                    priority=priority_out,
                    completion_estimate=completion_out,
                )
            )

        return candidates

    @staticmethod
    def _type_index_to_name(index: int) -> str:
        """Map a subgoal-type index to its canonical name."""
        names = ["explore", "secure", "conserve", "reduce_uncertainty", "probe"]
        if index < len(names):
            return names[index]
        return f"unknown_{index}"


# ---------------------------------------------------------------------------
# Sub-module: SubgoalEvaluator
# ---------------------------------------------------------------------------


class SubgoalEvaluator(nn.Module):
    """
    Evaluates how useful a subgoal is given the current state.

    Implements a learned value function over (state, subgoal) pairs.
    The evaluator takes the goal-aware state embedding and a subgoal
    vector, concatenates them, and produces a scalar value estimate
    that represents the expected long-term benefit of pursuing this
    subgoal from this state.

    The evaluator is trained jointly with the rest of the system,
    using TD-style updates on the subgoal value target.

    Args:
        goal_embedding_dim: Dimension of both the state embedding and
            the subgoal goal vector.
        hidden_dim: Width of internal MLP layers.
    """

    def __init__(
        self,
        goal_embedding_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        # Input: state_embedding (goal_embedding_dim) + goal_vector (goal_embedding_dim)
        input_dim = goal_embedding_dim * 2
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        subgoal_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the expected value of pursuing a subgoal from a state.

        Args:
            state_embedding: Goal-aware state embedding, shape
                ``(batch, goal_embedding_dim)`` or
                ``(goal_embedding_dim,)``.
            subgoal_vector: Subgoal goal vector, same shape constraints.

        Returns:
            Scalar value estimate of shape ``(batch, 1)`` or ``(1,)``.
        """
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        if subgoal_vector.dim() == 1:
            subgoal_vector = subgoal_vector.unsqueeze(0)

        combined = torch.cat([state_embedding, subgoal_vector], dim=-1)
        return self.value_network(combined)


# ---------------------------------------------------------------------------
# Sub-module: SubgoalDecomposer
# ---------------------------------------------------------------------------


class SubgoalDecomposer(nn.Module):
    """
    Decomposes a high-level subgoal into a sequence of lower-level
    child subgoals.

    The decomposer uses a GRU to autoregressively generate child
    subgoal vectors conditioned on the parent subgoal vector and the
    current state.  Each step of the GRU produces:
      - A child goal vector
      - A type prediction (softmax over subgoal types)
      - A step-level priority and completion estimate

    The number of children is controlled by
    ``config.decomposition_depth``.

    Args:
        goal_embedding_dim: Dimension of goal vectors.
        num_subgoal_types: Number of distinct subgoal types.
        hidden_dim: GRU hidden state size.
        max_depth: Maximum decomposition depth (max children per parent).
    """

    def __init__(
        self,
        goal_embedding_dim: int = 256,
        num_subgoal_types: int = 5,
        hidden_dim: int = 512,
        max_depth: int = 3,
    ):
        super().__init__()
        self.goal_embedding_dim = goal_embedding_dim
        self.num_subgoal_types = num_subgoal_types
        self.max_depth = max_depth

        # GRU: input = parent_goal + prev_child_goal (or zeros for first step)
        self.gru = nn.GRUCell(
            input_size=goal_embedding_dim * 2,
            hidden_size=hidden_dim,
        )

        # Child goal vector head
        self.child_goal_head = nn.Sequential(
            nn.Linear(hidden_dim, goal_embedding_dim),
            nn.SiLU(),
            nn.Linear(goal_embedding_dim, goal_embedding_dim),
        )

        # Type prediction head
        self.type_head = nn.Linear(hidden_dim, num_subgoal_types)

        # Priority head for child
        self.child_priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Completion estimate head for child
        self.child_completion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

        # Stop token: predicts whether decomposition should continue
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        parent_subgoal: Subgoal,
        state_embedding: torch.Tensor,
        max_children: Optional[int] = None,
    ) -> List[Subgoal]:
        """
        Decompose a parent subgoal into child subgoals.

        The GRU generates children one at a time.  After each step a
        *stop probability* is computed; if it exceeds 0.5 the
        decomposition terminates early.

        Args:
            parent_subgoal: The :class:`Subgoal` to decompose.
            state_embedding: Goal-aware state embedding, shape
                ``(batch, goal_embedding_dim)`` or
                ``(goal_embedding_dim,)``.
            max_children: Override for ``self.max_depth``.

        Returns:
            List of child :class:`Subgoal` objects, each with
            ``parent_id`` set to ``parent_subgoal.subgoal_id``.
        """
        max_children = max_children or self.max_depth

        parent_vec = parent_subgoal.goal_vector
        if parent_vec.dim() == 1:
            parent_vec = parent_vec.unsqueeze(0)
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)

        batch_size = parent_vec.size(0)
        device = parent_vec.device
        dtype = parent_vec.dtype

        # Initialize GRU hidden state from state embedding
        # Project state_embedding to hidden_dim if needed
        h = state_embedding
        if h.size(-1) != self.gru.hidden_size:
            # Simple linear projection
            if not hasattr(self, "_state_proj"):
                self._state_proj = nn.Linear(
                    h.size(-1), self.gru.hidden_size, bias=False
                ).to(device)
            h = self._state_proj(h)

        prev_child = torch.zeros(batch_size, self.goal_embedding_dim, device=device, dtype=dtype)

        children: List[Subgoal] = []

        for step in range(max_children):
            # GRU input: parent + previous child
            gru_input = torch.cat([parent_vec, prev_child], dim=-1)
            h = self.gru(gru_input, h)

            # Child goal vector
            child_vec = self.child_goal_head(h)  # (batch, goal_dim)

            # Type prediction
            type_logits = self.type_head(h)  # (batch, num_types)
            type_idx = type_logits.argmax(dim=-1)  # (batch,)

            # Priority
            priority = self.child_priority_head(h).squeeze(-1)  # (batch,)

            # Completion estimate
            completion = self.child_completion_head(h).squeeze(-1)  # (batch,)

            # Stop probability
            stop_prob = self.stop_head(h).squeeze(-1)  # (batch,)

            # Take first batch element for Subgoal creation
            type_name = SubgoalProposer._type_index_to_name(type_idx[0].item())
            child_goal = Subgoal(
                goal_vector=child_vec[0].detach(),
                goal_type=type_name,
                priority=float(priority[0].item()),
                completion_estimate=float(completion[0].item()),
                parent_id=parent_subgoal.subgoal_id,
            )
            children.append(child_goal)

            # Update prev_child for autoregressive generation
            prev_child = child_vec.detach()

            # Early stopping
            if stop_prob[0].item() > 0.5:
                break

        return children


# ---------------------------------------------------------------------------
# Sub-module: CompletionChecker
# ---------------------------------------------------------------------------


class CompletionChecker(nn.Module):
    """
    Checks whether a subgoal has been completed given the current state.

    Implements a learned binary classifier over (state, subgoal) pairs.
    The output is a probability in ``[0, 1]`` that the subgoal is
    complete.  A subgoal is considered *completed* when this probability
    exceeds the configured ``completion_threshold``.

    The checker is trained with a supervised signal derived from
    hindsight analysis: after an episode, subgoals that were being
    actively pursued when a reward spike occurred are labelled as
    completed.

    Args:
        goal_embedding_dim: Dimension of both the state embedding and
            the subgoal vector.
        hidden_dim: Width of internal MLP layers.
    """

    def __init__(
        self,
        goal_embedding_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = goal_embedding_dim * 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        subgoal_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the probability that a subgoal is completed.

        Args:
            state_embedding: Goal-aware state embedding, shape
                ``(batch, goal_embedding_dim)`` or
                ``(goal_embedding_dim,)``.
            subgoal_vector: Subgoal goal vector, same shape constraints.

        Returns:
            Completion probability in ``[0, 1]``, shape ``(batch, 1)``
            or ``(1,)``.
        """
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        if subgoal_vector.dim() == 1:
            subgoal_vector = subgoal_vector.unsqueeze(0)

        combined = torch.cat([state_embedding, subgoal_vector], dim=-1)
        return self.network(combined)


# ---------------------------------------------------------------------------
# Sub-module: SubgoalRewardComputer
# ---------------------------------------------------------------------------


class SubgoalRewardComputer(nn.Module):
    """
    Computes intrinsic reward for subgoal progress.

    The intrinsic reward is proportional to how much *closer* the agent
    has moved toward the subgoal in goal space.  Concretely:

        r_intrinsic = max(0, d_{t-1} - d_t) / (d_0 + eps)

    where ``d_t`` is the distance between the current state embedding
    and the subgoal vector at time *t*.  This gives a normalized
    progress signal: a reward of 1.0 means the agent has closed the
    entire initial gap in one step.

    A small learnable bonus is also added when the subgoal is newly
    completed.

    Args:
        goal_embedding_dim: Dimension of the goal / state embedding.
        hidden_dim: Width of the bonus network.
    """

    def __init__(
        self,
        goal_embedding_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # Completion bonus network
        self.completion_bonus_net = nn.Sequential(
            nn.Linear(goal_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Running distance statistics for normalization
        self.register_buffer("initial_distance", torch.tensor(0.0))
        self.register_buffer("previous_distance", torch.tensor(0.0))
        self.register_buffer("distance_initialized", torch.tensor(False))

    def forward(
        self,
        state_embedding: torch.Tensor,
        subgoal_vector: torch.Tensor,
        is_completed: bool = False,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward for progress toward a subgoal.

        Args:
            state_embedding: Current goal-aware state embedding, shape
                ``(batch, goal_embedding_dim)`` or
                ``(goal_embedding_dim,)``.
            subgoal_vector: Target subgoal goal vector, same shape.
            is_completed: Whether the subgoal was just completed.

        Returns:
            Intrinsic reward scalar tensor.
        """
        if state_embedding.dim() == 1:
            state_embedding = state_embedding.unsqueeze(0)
        if subgoal_vector.dim() == 1:
            subgoal_vector = subgoal_vector.unsqueeze(0)

        # Cosine distance: 1 - cos_similarity
        cos_sim = F.cosine_similarity(state_embedding, subgoal_vector, dim=-1)
        current_distance = 1.0 - cos_sim  # (batch,)

        if not self.distance_initialized.item():
            with torch.no_grad():
                self.initial_distance.copy_(current_distance.mean().detach())
                self.previous_distance.copy_(current_distance.mean().detach())
                self.distance_initialized.fill_(True)
            return torch.tensor(0.0, device=state_embedding.device)

        # Progress reward: reduction in distance, normalized
        with torch.no_grad():
            progress = (self.previous_distance - current_distance.mean()).clamp(min=0.0)
            normalized_progress = progress / (self.initial_distance + 1e-8)
            self.previous_distance.copy_(current_distance.mean().detach())

        intrinsic_reward = normalized_progress

        # Completion bonus
        if is_completed:
            bonus = self.completion_bonus_net(subgoal_vector.mean(dim=0, keepdim=True))
            intrinsic_reward = intrinsic_reward + bonus.squeeze()

        return intrinsic_reward


# ---------------------------------------------------------------------------
# Main module: SubgoalGenerator
# ---------------------------------------------------------------------------


class SubgoalGenerator(nn.Module):
    """
    Self-generated subgoal system for Deep Thought.

    The ``SubgoalGenerator`` invents intermediate objectives that
    structure the agent's behavior.  Rather than always optimizing the
    external reward directly, the system proposes, evaluates, and
    tracks subgoals that serve as stepping stones toward long-term
    success.

    Lifecycle:
        1. **Propose**: Every ``subgoal_proposal_interval`` steps, the
           proposer generates candidate subgoals from the current state
           embedding.
        2. **Evaluate**: The evaluator scores each candidate; the highest-
           valued candidate becomes the new active subgoal (subject to
           the ``max_active_subgoals`` cap).
        3. **Decompose**: Optionally, the decomposer breaks a high-level
           subgoal into a sequence of child subgoals.
        4. **Track**: The completion checker monitors progress.  When a
           subgoal's completion probability exceeds
           ``completion_threshold``, it is marked as done and removed
           from the active set.
        5. **Reward**: An intrinsic reward signal is produced proportional
           to progress toward the active subgoal.

    Args:
        config: A :class:`SubgoalConfig` instance.  When ``None``,
            defaults are used.
        latent_dim: Dimensionality of the shared latent representation
            (``h_t`` and ``x_t``).

    Example::

        config = SubgoalConfig(goal_embedding_dim=256, max_active_subgoals=5)
        generator = SubgoalGenerator(config, latent_dim=1024)

        # During interaction:
        active_subgoal, info = generator(
            h_t=hidden,
            x_t=observation,
            reward=0.5,
            uncertainty=0.3,
            episode_progress=0.4,
        )
        intrinsic_reward = info["subgoal_reward"]
    """

    def __init__(
        self,
        config: Optional[SubgoalConfig] = None,
        latent_dim: int = 1024,
    ):
        super().__init__()

        if config is None:
            config = SubgoalConfig()
        self.config = config
        self.latent_dim = latent_dim
        self.use_subgoals = config.use_subgoals

        # ---- Sub-modules ------------------------------------------------
        self.encoder = SubgoalEncoder(
            latent_dim=latent_dim,
            goal_embedding_dim=config.goal_embedding_dim,
            hidden_dim=config.goal_embedding_dim * 2,
        )

        self.proposer = SubgoalProposer(
            goal_embedding_dim=config.goal_embedding_dim,
            num_subgoal_types=len(config.subgoal_types),
            hidden_dim=config.goal_embedding_dim * 2,
        )

        self.evaluator = SubgoalEvaluator(
            goal_embedding_dim=config.goal_embedding_dim,
            hidden_dim=config.goal_embedding_dim * 2,
        )

        self.decomposer = SubgoalDecomposer(
            goal_embedding_dim=config.goal_embedding_dim,
            num_subgoal_types=len(config.subgoal_types),
            hidden_dim=config.goal_embedding_dim * 2,
            max_depth=config.decomposition_depth,
        )

        self.completion_checker = CompletionChecker(
            goal_embedding_dim=config.goal_embedding_dim,
            hidden_dim=config.goal_embedding_dim,
        )

        self.reward_computer = SubgoalRewardComputer(
            goal_embedding_dim=config.goal_embedding_dim,
            hidden_dim=config.goal_embedding_dim // 2,
        )

        # ---- Internal state ---------------------------------------------
        self._active_subgoals: List[Subgoal] = []
        self._completed_subgoals: List[Subgoal] = []
        self._next_subgoal_id: int = 0
        self._step_count: int = 0
        self._current_state_embedding: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allocate_subgoal_id(self) -> int:
        """Return the next unique subgoal ID and increment the counter."""
        sid = self._next_subgoal_id
        self._next_subgoal_id += 1
        return sid

    def _assign_ids(self, subgoals: List[Subgoal]) -> None:
        """Assign unique IDs to a list of subgoals in-place."""
        for sg in subgoals:
            sg.subgoal_id = self._allocate_subgoal_id()

    def _enforce_active_cap(self) -> None:
        """
        Trim the active subgoal list to ``max_active_subgoals``.

        Subgoals with the lowest priority are removed first.
        """
        cap = self.config.max_active_subgoals
        if len(self._active_subgoals) > cap:
            # Sort by priority descending, keep top-cap
            self._active_subgoals.sort(key=lambda s: s.priority, reverse=True)
            removed = self._active_subgoals[cap:]
            self._active_subgoals = self._active_subgoals[:cap]
            # Move removed subgoals to completed (they were deprioritized)
            self._completed_subgoals.extend(removed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        reward: torch.Tensor,
        uncertainty: torch.Tensor,
        episode_progress: torch.Tensor,
    ) -> Tuple[Optional[Subgoal], Dict[str, Any]]:
        """
        Main entry point: propose, evaluate, and track subgoals.

        On every call the completion of existing subgoals is checked.  At
        intervals of ``subgoal_proposal_interval`` steps, new candidates
        are proposed and the best one is added to the active set.

        Args:
            h_t: Hidden state tensor, shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.
            x_t: Encoded observation, same shape constraints.
            reward: Current external reward scalar.
            uncertainty: Current uncertainty scalar.
            episode_progress: Episode progress in ``[0, 1]``.

        Returns:
            A tuple ``(active_subgoal, info)`` where:

            - ``active_subgoal`` is the highest-priority active
              :class:`Subgoal`, or ``None`` if no subgoals are active.
            - ``info`` is a diagnostic dictionary containing reward
              signals, active subgoal count, etc.
        """
        if not self.use_subgoals:
            return None, {
                "subgoal_reward": torch.tensor(0.0),
                "active_subgoal_count": 0,
                "active_subgoal_type": "none",
                "subgoal_progress": 0.0,
                "proposed_count": 0,
            }

        self._step_count += 1

        # ---- Encode current state ----
        state_embedding = self.encoder(h_t, x_t, reward, uncertainty, episode_progress)
        self._current_state_embedding = state_embedding.detach()

        # ---- Check completion of existing subgoals ----
        newly_completed: List[Subgoal] = []
        for sg in list(self._active_subgoals):
            is_done = self.check_completion(
                state_embedding.detach(), sg
            )
            if is_done:
                sg.progress = 1.0
                newly_completed.append(sg)

        # Remove completed subgoals
        for sg in newly_completed:
            self._active_subgoals.remove(sg)
            self._completed_subgoals.append(sg)

        # ---- Propose new subgoals at intervals ----
        proposed_count = 0
        if self._step_count % self.config.subgoal_proposal_interval == 0:
            candidates = self.propose_subgoals(state_embedding.detach())
            proposed_count = len(candidates)

            # Evaluate and select the best candidate
            if candidates:
                best_subgoal = max(candidates, key=lambda s: s.priority)
                # Also evaluate with the learned value function
                best_value = self.evaluate_subgoal(state_embedding.detach(), best_subgoal)
                best_subgoal.priority = float(best_value.item()) + best_subgoal.priority

                # Assign ID and add to active set
                best_subgoal.subgoal_id = self._allocate_subgoal_id()
                self._active_subgoals.append(best_subgoal)
                self._enforce_active_cap()

        # ---- Compute intrinsic reward for the top subgoal ----
        subgoal_reward = torch.tensor(0.0, device=h_t.device)
        active_subgoal: Optional[Subgoal] = None
        active_type = "none"
        active_progress = 0.0

        if self._active_subgoals:
            # Sort by priority
            self._active_subgoals.sort(key=lambda s: s.priority, reverse=True)
            active_subgoal = self._active_subgoals[0]
            active_type = active_subgoal.goal_type
            active_progress = active_subgoal.progress

            subgoal_reward = self.get_subgoal_reward(
                state_embedding.detach(), active_subgoal
            )
            subgoal_reward = subgoal_reward * self.config.subgoal_reward_coef

        info: Dict[str, Any] = {
            "subgoal_reward": subgoal_reward,
            "active_subgoal_count": len(self._active_subgoals),
            "active_subgoal_type": active_type,
            "subgoal_progress": active_progress,
            "proposed_count": proposed_count,
            "completed_this_step": len(newly_completed),
        }

        return active_subgoal, info

    def propose_subgoals(
        self,
        state_context: torch.Tensor,
        num_candidates: int = 5,
    ) -> List[Subgoal]:
        """
        Propose candidate subgoals from the current state context.

        Delegates to :class:`SubgoalProposer`.  The number of candidates
        is capped at the number of configured subgoal types.

        Args:
            state_context: Goal-aware state embedding, shape
                ``(batch, goal_embedding_dim)`` or
                ``(goal_embedding_dim,)``.
            num_candidates: Maximum number of candidates to propose.

        Returns:
            List of proposed :class:`Subgoal` objects.
        """
        num_candidates = min(num_candidates, len(self.config.subgoal_types))
        return self.proposer(state_context, num_candidates=num_candidates)

    def evaluate_subgoal(
        self,
        state: torch.Tensor,
        subgoal: Subgoal,
    ) -> torch.Tensor:
        """
        Evaluate the expected value of a subgoal from the current state.

        Args:
            state: Goal-aware state embedding, shape
                ``(batch, goal_embedding_dim)`` or
                ``(goal_embedding_dim,)``.
            subgoal: The :class:`Subgoal` to evaluate.

        Returns:
            Scalar value estimate.
        """
        subgoal_vec = subgoal.goal_vector.to(state.device)
        value = self.evaluator(state, subgoal_vec)
        return value.squeeze()

    def decompose_subgoal(self, subgoal: Subgoal) -> List[Subgoal]:
        """
        Decompose a high-level subgoal into a sequence of child subgoals.

        Requires a stored state embedding (set during ``forward()``).
        If no state embedding is available, returns an empty list.

        Args:
            subgoal: The :class:`Subgoal` to decompose.

        Returns:
            List of child :class:`Subgoal` objects with ``parent_id``
            set to ``subgoal.subgoal_id``.
        """
        if self._current_state_embedding is None:
            return []

        state_emb = self._current_state_embedding
        children = self.decomposer(subgoal, state_emb)

        # Assign IDs and register parent-child relationships
        for child in children:
            child.subgoal_id = self._allocate_subgoal_id()
            subgoal.children.append(child.subgoal_id)

        return children

    def check_completion(
        self,
        state: torch.Tensor,
        subgoal: Subgoal,
    ) -> bool:
        """
        Check whether a subgoal has been completed.

        Uses :class:`CompletionChecker` to predict a completion
        probability.  The subgoal is considered complete if this
        probability exceeds ``config.completion_threshold``.

        The subgoal's ``progress`` field is also updated with the
        completion probability.

        Args:
            state: Goal-aware state embedding.
            subgoal: The :class:`Subgoal` to check.

        Returns:
            ``True`` if the subgoal is complete, ``False`` otherwise.
        """
        subgoal_vec = subgoal.goal_vector.to(state.device)
        completion_prob = self.completion_checker(state, subgoal_vec)
        prob_val = float(completion_prob.mean().item())

        # Update progress
        subgoal.progress = max(subgoal.progress, prob_val)

        return prob_val >= self.config.completion_threshold

    def get_subgoal_reward(
        self,
        state: torch.Tensor,
        subgoal: Subgoal,
    ) -> torch.Tensor:
        """
        Compute intrinsic reward for progress toward a subgoal.

        The reward reflects *distance reduction* in the goal embedding
        space between the current state and the subgoal.  A bonus is
        added when the subgoal is newly completed.

        Args:
            state: Goal-aware state embedding.
            subgoal: The :class:`Subgoal` being pursued.

        Returns:
            Intrinsic reward scalar tensor.
        """
        subgoal_vec = subgoal.goal_vector.to(state.device)
        is_completed = subgoal.progress >= self.config.completion_threshold
        reward = self.reward_computer(state, subgoal_vec, is_completed=is_completed)
        return reward

    def get_active_subgoals(self) -> List[Subgoal]:
        """
        Return the list of currently active subgoals.

        Returns:
            List of active :class:`Subgoal` objects, sorted by priority
            (highest first).
        """
        return sorted(self._active_subgoals, key=lambda s: s.priority, reverse=True)

    def reset(self) -> None:
        """
        Reset all subgoal state for a new episode.

        Clears active and completed subgoal lists, resets the step
        counter, and clears the stored state embedding.
        """
        self._active_subgoals.clear()
        self._completed_subgoals.clear()
        self._step_count = 0
        self._next_subgoal_id = 0
        self._current_state_embedding = None
        # Reset distance trackers in reward computer
        self.reward_computer.initial_distance.zero_()
        self.reward_computer.previous_distance.zero_()
        self.reward_computer.distance_initialized.zero_()

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return comprehensive diagnostic information about the subgoal
        system.

        Returns:
            Dictionary containing:
            - ``active_subgoal_count``: Number of active subgoals.
            - ``completed_subgoal_count``: Number of completed subgoals.
            - ``total_subgoals_created``: Total subgoals ever created.
            - ``step_count``: Current step within the episode.
            - ``active_types``: List of goal types for active subgoals.
            - ``active_priorities``: Priority values of active subgoals.
            - ``active_progress``: Progress values of active subgoals.
        """
        active = self.get_active_subgoals()
        return {
            "active_subgoal_count": len(self._active_subgoals),
            "completed_subgoal_count": len(self._completed_subgoals),
            "total_subgoals_created": self._next_subgoal_id,
            "step_count": self._step_count,
            "active_types": [s.goal_type for s in active],
            "active_priorities": [s.priority for s in active],
            "active_progress": [s.progress for s in active],
        }
