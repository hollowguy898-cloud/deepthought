"""
Opponent and social modeling systems for Deep Thought.

Models not just "Where are they?" but "What kind of player are they?"
Tracks opponent tendencies, deception patterns, risk preferences, and
habitual movement patterns to build rich, actionable opponent profiles.

Architecture
~~~~~~~~~~~~

The system is composed of five synergistic sub-modules:

1. **OpponentEncoder** — Encodes raw opponent observations (positions,
   actions, visible states) into a compact opponent latent space that
   captures the essential features of each opponent's behaviour.

2. **TendencyTracker** — A GRU-based recurrent module that tracks
   opponent behavioural tendencies over time.  It learns to categorise
   opponents along multiple behavioural axes such as aggressive,
   defensive, deceptive, and exploratory, producing a learned tendency
   embedding for each tracked opponent.

3. **StrategyPredictor** — Predicts the opponent's likely future
   strategy by auto-regressively rolling out the opponent's tendency
   embedding over a configurable horizon.  The predicted strategy
   embedding can be consumed by the planner to anticipate opponent
   moves.

4. **DeceptionDetector** — Detects when an opponent is being deceptive
   by comparing claimed (or inferred-intended) actions against observed
   actions.  A learnable discrepancy scorer maps the divergence to a
   deception likelihood score.

5. **RiskProfileEstimator** — Estimates an opponent's risk preferences
   on a continuous scale from risk-averse (0) to risk-seeking (1) using
   an exponential moving average of observed risk-indicating signals.

All sub-modules are differentiable and trained as part of the larger
Deep Thought system.  Per-opponent state (tendency vectors, risk
profiles, interaction counts) is stored in persistent buffers so that
profiles accumulate across interaction steps.

References:
    - Albrecht et al., "Autonomous Agents Modelling Other Agents:
      A Comprehensive Survey" (2018)
    - Shum et al., "Theory of Mind for Multi-Agent Reinforcement
      Learning" (2019)
    - De Weerd et al., "Estimating Repeated Second-Order Beliefs
      from Communication" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

from deep_thought.config import OpponentModelingConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OpponentProfile:
    """
    A rich, structured profile of a single opponent.

    This dataclass is returned by :meth:`OpponentModelingSystem.get_opponent_profile`
    and provides a snapshot of everything the system believes about an opponent
    at the current time-step.

    Attributes:
        opponent_id: Integer identifier of the opponent (index into the
            tracked-opponent buffer).
        tendency_vector: Learned tendency embedding capturing the opponent's
            behavioural style.  Shape ``(tendency_dim,)``.
        risk_preference: Estimated risk preference on a continuous scale
            from 0 (risk-averse) to 1 (risk-seeking).
        deception_likelihood: Estimated probability that the opponent is
            currently being deceptive, from 0 (honest) to 1 (deceptive).
        strategy_embedding: Current predicted strategy embedding for the
            opponent.  Shape ``(opponent_latent_dim,)``.
        predictability: How predictable the opponent's behaviour is, from
            0 (unpredictable / random) to 1 (highly predictable / consistent).
        interaction_count: Total number of interaction steps observed for
            this opponent.
    """

    opponent_id: int
    tendency_vector: torch.Tensor    # (tendency_dim,)
    risk_preference: float           # 0=risk-averse, 1=risk-seeking
    deception_likelihood: float      # 0=honest, 1=deceptive
    strategy_embedding: torch.Tensor # (opponent_latent_dim,)
    predictability: float            # 0=unpredictable, 1=highly predictable
    interaction_count: int


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class OpponentEncoder(nn.Module):
    """
    Encodes raw opponent observations into an opponent latent space.

    Takes a concatenation of the opponent's observable features (position,
    last action, visible resources, etc.) and maps them through a multi-layer
    perceptron to produce a fixed-size latent representation that captures
    the essential features of the opponent's current state.

    A projection head further distils the latent representation into a
    tendency-aware embedding that is suitable for consumption by the
    :class:`TendencyTracker`.

    Args:
        observation_dim: Dimensionality of the raw opponent observation
            vector.
        latent_dim: Dimensionality of the output latent representation.
        hidden_dim: Hidden layer width for the encoder MLP.
        num_layers: Number of hidden layers in the encoder MLP.
    """

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim

        # Build multi-layer encoder
        layers: list[nn.Module] = []
        in_dim = observation_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Projection head for tendency-aware embedding
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, opponent_obs: torch.Tensor) -> torch.Tensor:
        """
        Encode opponent observations into latent space.

        Args:
            opponent_obs: Raw opponent observation tensor of shape
                ``(batch, observation_dim)`` or ``(observation_dim,)``.

        Returns:
            Latent opponent embedding of shape ``(batch, latent_dim)``
            or ``(latent_dim,)``.
        """
        squeeze = False
        if opponent_obs.dim() == 1:
            opponent_obs = opponent_obs.unsqueeze(0)
            squeeze = True

        latent = self.encoder(opponent_obs)
        latent = F.silu(latent)

        if squeeze:
            latent = latent.squeeze(0)
        return latent

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Project a latent embedding through the tendency-aware projection head.

        This is used to prepare encoded observations for the tendency tracker,
        ensuring that the representation emphasises behaviourally-relevant
        features.

        Args:
            latent: Latent embedding of shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.

        Returns:
            Projected embedding of the same shape.
        """
        squeeze = False
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            squeeze = True

        projected = self.projection(latent)

        if squeeze:
            projected = projected.squeeze(0)
        return projected


class TendencyTracker(nn.Module):
    """
    Tracks opponent behavioural tendencies using a GRU-based recurrent model.

    Processes a sequence of opponent observations (already encoded) and
    produces a tendency embedding that captures the opponent's behavioural
    style along multiple axes.  The tendency embedding is further
    decomposed into discrete tendency types (aggressive, defensive,
    deceptive, etc.) via a learned projection.

    The tracker maintains per-opponent hidden states stored as buffers
    so they persist across forward passes and can be incrementally
    updated.

    Args:
        latent_dim: Dimensionality of the input opponent latent
            representation.
        tendency_dim: Dimensionality of the output tendency embedding.
        num_tendency_types: Number of discrete tendency categories to
            track (e.g., aggressive, defensive, deceptive, exploratory,
            conservative, chaotic, cooperative, opportunistic).
        hidden_dim: Hidden state dimensionality for the GRU.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        tendency_dim: int = 64,
        num_tendency_types: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.tendency_dim = tendency_dim
        self.num_tendency_types = num_tendency_types

        # GRU that processes encoded opponent observations
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Project GRU output to tendency embedding
        self.tendency_head = nn.Sequential(
            nn.Linear(hidden_dim, tendency_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(tendency_dim * 2),
            nn.Linear(tendency_dim * 2, tendency_dim),
        )

        # Project tendency embedding to discrete tendency-type logits
        self.type_head = nn.Sequential(
            nn.Linear(tendency_dim, tendency_dim),
            nn.SiLU(),
            nn.Linear(tendency_dim, num_tendency_types),
        )

        # Predictability estimator — maps tendency embedding to a
        # scalar in [0, 1] indicating how predictable the opponent is.
        self.predictability_head = nn.Sequential(
            nn.Linear(tendency_dim, tendency_dim // 2),
            nn.SiLU(),
            nn.Linear(tendency_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        encoded_obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process an encoded opponent observation and produce tendency outputs.

        Args:
            encoded_obs: Encoded opponent observation of shape
                ``(batch, seq_len, latent_dim)`` or ``(batch, latent_dim)``
                or ``(latent_dim,)``.  If 2-D or 1-D, a sequence length of
                1 is assumed.
            hidden_state: Optional previous GRU hidden state of shape
                ``(num_layers, batch, hidden_dim)``.  If ``None``, the GRU
                starts from a zero hidden state.

        Returns:
            A tuple ``(tendency_embedding, tendency_logits, predictability, new_hidden)``:

            - ``tendency_embedding``: Tendency embedding of shape
              ``(batch, tendency_dim)``.
            - ``tendency_logits``: Per-tendency-type logits of shape
              ``(batch, num_tendency_types)``.
            - ``predictability``: Predictability score of shape
              ``(batch, 1)``.
            - ``new_hidden``: Updated GRU hidden state of shape
              ``(num_layers, batch, hidden_dim)``.
        """
        # Normalise input to (batch, seq_len, latent_dim)
        if encoded_obs.dim() == 1:
            encoded_obs = encoded_obs.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        elif encoded_obs.dim() == 2:
            encoded_obs = encoded_obs.unsqueeze(1)  # (B, 1, D)

        gru_out, new_hidden = self.gru(encoded_obs, hidden_state)
        # Take the last time-step output
        last_out = gru_out[:, -1, :]  # (batch, hidden_dim)

        tendency_embedding = self.tendency_head(last_out)   # (batch, tendency_dim)
        tendency_logits = self.type_head(tendency_embedding) # (batch, num_tendency_types)
        predictability = self.predictability_head(tendency_embedding)  # (batch, 1)

        return tendency_embedding, tendency_logits, predictability, new_hidden


class StrategyPredictor(nn.Module):
    """
    Predicts the opponent's likely future strategy.

    Given an opponent's current tendency embedding, auto-regressively
    rolls out predicted strategy embeddings over a configurable horizon.
    Each step of the rollout is produced by a shared GRU cell followed
    by a projection to the strategy embedding space.

    The predicted strategy embeddings can be consumed by the planning
    layer to anticipate opponent moves and prepare counter-strategies.

    Args:
        tendency_dim: Dimensionality of the input tendency embedding.
        strategy_dim: Dimensionality of the output strategy embedding.
        horizon: Number of future steps to predict.
        hidden_dim: Hidden state dimensionality for the auto-regressive
            GRU cell.
    """

    def __init__(
        self,
        tendency_dim: int = 64,
        strategy_dim: int = 256,
        horizon: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.tendency_dim = tendency_dim
        self.strategy_dim = strategy_dim
        self.horizon = horizon

        # Project tendency embedding to initial hidden state
        self.init_projector = nn.Sequential(
            nn.Linear(tendency_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # GRU cell for auto-regressive rollout
        self.gru_cell = nn.GRUCell(
            input_size=strategy_dim,
            hidden_size=hidden_dim,
        )

        # Project hidden state to strategy embedding at each step
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim, strategy_dim),
            nn.SiLU(),
            nn.LayerNorm(strategy_dim),
        )

        # Seed vector for the first step of the rollout
        self.register_buffer(
            "seed", torch.randn(strategy_dim)
        )
        # Normalise the seed so it starts as a unit vector
        self.seed.div_(self.seed.norm() + 1e-8)

    def forward(
        self,
        tendency_embedding: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict future strategy embeddings for an opponent.

        Args:
            tendency_embedding: Current tendency embedding of shape
                ``(batch, tendency_dim)`` or ``(tendency_dim,)``.
            horizon: Number of future steps to predict.  If ``None``,
                ``self.horizon`` is used.

        Returns:
            Predicted strategy embeddings of shape
            ``(batch, horizon, strategy_dim)``.
        """
        if tendency_embedding.dim() == 1:
            tendency_embedding = tendency_embedding.unsqueeze(0)

        batch_size = tendency_embedding.shape[0]
        h = horizon if horizon is not None else self.horizon

        # Initialise hidden state from tendency embedding
        hidden = self.init_projector(tendency_embedding)  # (batch, hidden_dim)

        # First input is the learned seed vector
        x = self.seed.unsqueeze(0).expand(batch_size, -1)  # (batch, strategy_dim)

        predictions: list[torch.Tensor] = []
        for _ in range(h):
            hidden = self.gru_cell(x, hidden)               # (batch, hidden_dim)
            strategy = self.strategy_head(hidden)             # (batch, strategy_dim)
            predictions.append(strategy)
            x = strategy  # auto-regressive: output becomes next input

        # Stack along the time dimension
        return torch.stack(predictions, dim=1)  # (batch, horizon, strategy_dim)


class DeceptionDetector(nn.Module):
    """
    Detects when an opponent is being deceptive.

    Compares an opponent's *claimed* (or inferred-intended) action against
    the *observed* action.  The discrepancy is encoded through a learnable
    scoring network that maps the pair to a deception likelihood in [0, 1].

    The detector also considers the opponent's tendency embedding as
    context, since a historically deceptive opponent is more likely to be
    deceptive again (Bayesian prior).

    Args:
        action_dim: Dimensionality of the action representation.
        tendency_dim: Dimensionality of the tendency embedding.
        hidden_dim: Hidden layer width for the scoring network.
    """

    def __init__(
        self,
        action_dim: int,
        tendency_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim

        # Encode the discrepancy between claimed and observed actions
        self.discrepancy_encoder = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )

        # Combine discrepancy features with tendency context
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2 + tendency_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(
        self,
        claimed_action: torch.Tensor,
        observed_action: torch.Tensor,
        tendency_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute deception likelihood given claimed and observed actions.

        Args:
            claimed_action: The action the opponent claimed or appeared to
                intend, of shape ``(batch, action_dim)`` or ``(action_dim,)``.
            observed_action: The action the opponent actually took, of the
                same shape as ``claimed_action``.
            tendency_embedding: Optional tendency embedding for the opponent
                of shape ``(batch, tendency_dim)`` or ``(tendency_dim,)``.
                When provided, the scorer uses it as additional context.

        Returns:
            Deception likelihood of shape ``(batch, 1)`` or ``(1,)``,
            where 0 means honest and 1 means deceptive.
        """
        squeeze = False
        if claimed_action.dim() == 1:
            claimed_action = claimed_action.unsqueeze(0)
            observed_action = observed_action.unsqueeze(0)
            squeeze = True

        # Encode the discrepancy
        discrepancy_input = torch.cat([claimed_action, observed_action], dim=-1)
        discrepancy_features = self.discrepancy_encoder(discrepancy_input)

        # Combine with tendency context if available
        if tendency_embedding is not None:
            if tendency_embedding.dim() == 1:
                tendency_embedding = tendency_embedding.unsqueeze(0)
            combined = torch.cat([discrepancy_features, tendency_embedding], dim=-1)
        else:
            # Use zero context when no tendency embedding is available
            batch_size = discrepancy_features.shape[0]
            zero_context = torch.zeros(
                batch_size, self.scorer[0].in_features - discrepancy_features.shape[-1],
                device=discrepancy_features.device,
                dtype=discrepancy_features.dtype,
            )
            combined = torch.cat([discrepancy_features, zero_context], dim=-1)

        score = self.scorer(combined)

        if squeeze:
            score = score.squeeze(0)
        return score


class RiskProfileEstimator(nn.Module):
    """
    Estimates an opponent's risk preferences.

    Produces a continuous risk preference score from 0 (risk-averse) to
    1 (risk-seeking).  The estimator uses two complementary mechanisms:

    1. **Neural risk scorer** — A learnable MLP that maps the opponent's
       tendency embedding and recent action variance to a risk score.
       This captures complex, non-linear risk patterns.

    2. **EMA risk tracker** — An exponential moving average of observed
       risk-indicating signals (e.g., variance of actions, frequency of
       high-stakes decisions).  This provides a stable, long-term risk
       estimate that is robust to temporary fluctuations.

    Args:
        tendency_dim: Dimensionality of the tendency embedding.
        action_dim: Dimensionality of the action representation.
        hidden_dim: Hidden layer width for the neural scorer.
        ema_decay: Exponential moving average decay factor for the
            long-term risk tracker.  Higher values make the estimate
            more stable; lower values make it more reactive.
    """

    def __init__(
        self,
        tendency_dim: int = 64,
        action_dim: int = 64,
        hidden_dim: int = 128,
        ema_decay: float = 0.95,
    ):
        super().__init__()
        self.tendency_dim = tendency_dim
        self.action_dim = action_dim
        self.ema_decay = ema_decay

        # Neural risk scorer
        self.scorer = nn.Sequential(
            nn.Linear(tendency_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

        # EMA buffers for per-opponent risk tracking (initialised in
        # the parent OpponentModelingSystem based on max_opponents)
        # These are registered by the parent, not here.

    def forward(
        self,
        tendency_embedding: torch.Tensor,
        action_signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute an instantaneous risk preference estimate.

        Args:
            tendency_embedding: Tendency embedding of shape
                ``(batch, tendency_dim)`` or ``(tendency_dim,)``.
            action_signal: A signal derived from the opponent's recent
                actions (e.g., variance, magnitude) of shape
                ``(batch, action_dim)`` or ``(action_dim,)``.

        Returns:
            Risk preference of shape ``(batch, 1)`` or ``(1,)``.
        """
        squeeze = False
        if tendency_embedding.dim() == 1:
            tendency_embedding = tendency_embedding.unsqueeze(0)
            action_signal = action_signal.unsqueeze(0)
            squeeze = True

        combined = torch.cat([tendency_embedding, action_signal], dim=-1)
        risk = self.scorer(combined)

        if squeeze:
            risk = risk.squeeze(0)
        return risk

    def update_ema(
        self,
        current_ema: torch.Tensor,
        new_value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update the EMA risk estimate with a new observation.

        Args:
            current_ema: Current EMA risk estimate (scalar or tensor).
            new_value: New risk-indicating value to blend in.

        Returns:
            Updated EMA risk estimate.
        """
        return self.ema_decay * current_ema + (1.0 - self.ema_decay) * new_value


# ---------------------------------------------------------------------------
# Main system
# ---------------------------------------------------------------------------


class OpponentModelingSystem(nn.Module):
    """
    Opponent and social modeling system for Deep Thought.

    Models not just "Where are they?" but "What kind of player are they?"
    Tracks opponent tendencies, deception patterns, risk preferences, and
    habitual movement patterns.  For each tracked opponent the system
    maintains an :class:`OpponentProfile` that can be queried by the
    planning and decision-making layers.

    The system is fully differentiable and is trained end-to-end as part
    of the larger Deep Thought architecture.  Per-opponent state is
    maintained in buffers that persist across forward passes.

    Usage::

        config = OpponentModelingConfig(opponent_latent_dim=256)
        model = OpponentModelingSystem(config, latent_dim=1024)

        # During interaction:
        opponent_context, info = model(
            opponent_obs=obs_batch,             # (batch, obs_dim)
            interaction_history=history_batch,   # (batch, seq, obs_dim)
        )

        # Query individual profiles:
        profile = model.get_opponent_profile(opponent_id=0)

        # Detect deception:
        deception_score = model.detect_deception(
            opponent_id=0,
            claimed_action=claimed,
            observed_action=observed,
        )

    Args:
        config: A :class:`OpponentModelingConfig` instance controlling
            all hyperparameters.  When ``None``, defaults are used.
        latent_dim: Dimensionality of the main Deep Thought latent
            representation.  Used to produce the final opponent context
            that integrates into the rest of the system.
    """

    def __init__(
        self,
        config: Optional[OpponentModelingConfig] = None,
        latent_dim: int = 1024,
    ):
        super().__init__()

        if config is None:
            config = OpponentModelingConfig()

        self.config = config
        self.latent_dim = latent_dim

        # Inferred dimensions
        opponent_obs_dim = latent_dim  # opponent obs share the main latent dim
        action_dim = latent_dim // 4   # action signal dimensionality

        # ---- Sub-modules ------------------------------------------------

        self.opponent_encoder = OpponentEncoder(
            observation_dim=opponent_obs_dim,
            latent_dim=config.opponent_latent_dim,
            hidden_dim=config.opponent_latent_dim * 2,
            num_layers=3,
        )

        self.tendency_tracker = TendencyTracker(
            latent_dim=config.opponent_latent_dim,
            tendency_dim=config.tendency_dim,
            num_tendency_types=config.num_tendency_types,
            hidden_dim=config.opponent_latent_dim,
        )

        self.strategy_predictor = StrategyPredictor(
            tendency_dim=config.tendency_dim,
            strategy_dim=config.opponent_latent_dim,
            horizon=config.strategy_horizon,
            hidden_dim=config.opponent_latent_dim,
        )

        self.deception_detector = DeceptionDetector(
            action_dim=action_dim,
            tendency_dim=config.tendency_dim,
            hidden_dim=config.opponent_latent_dim,
        )

        self.risk_estimator = RiskProfileEstimator(
            tendency_dim=config.tendency_dim,
            action_dim=action_dim,
            hidden_dim=config.opponent_latent_dim // 2,
            ema_decay=config.risk_estimation_ema,
        )

        # ---- Social context aggregator ----------------------------------
        # Combines all opponent profiles into a single social context
        # vector that can be consumed by the main agent.
        self.social_aggregator = nn.Sequential(
            nn.Linear(
                config.max_opponents * (config.tendency_dim + config.opponent_latent_dim),
                config.opponent_latent_dim,
            ),
            nn.SiLU(),
            nn.LayerNorm(config.opponent_latent_dim),
            nn.Linear(config.opponent_latent_dim, latent_dim),
        )

        # ---- Per-opponent persistent state (buffers) --------------------
        # These are not gradient-tracked; they are updated via EMA-style
        # operations in ``update_tendencies`` and similar methods.

        # Tendency embeddings: (max_opponents, tendency_dim)
        self.register_buffer(
            "tendency_vectors",
            torch.zeros(config.max_opponents, config.tendency_dim),
        )

        # Strategy embeddings: (max_opponents, opponent_latent_dim)
        self.register_buffer(
            "strategy_embeddings",
            torch.zeros(config.max_opponents, config.opponent_latent_dim),
        )

        # Risk EMA: (max_opponents,)
        self.register_buffer(
            "risk_ema",
            torch.full((config.max_opponents,), 0.5),
        )

        # Deception likelihood: (max_opponents,)
        self.register_buffer(
            "deception_scores",
            torch.zeros(config.max_opponents),
        )

        # Predictability: (max_opponents,)
        self.register_buffer(
            "predictability_scores",
            torch.full((config.max_opponents,), 0.5),
        )

        # Interaction counts: (max_opponents,)
        self.register_buffer(
            "interaction_counts",
            torch.zeros(config.max_opponents, dtype=torch.long),
        )

        # GRU hidden states for tendency tracker: (num_layers, max_opponents, hidden_dim)
        # Stored separately to handle the 3-D shape properly
        self.register_buffer(
            "gru_hidden_states",
            torch.zeros(2, config.max_opponents, config.opponent_latent_dim),
        )

        # ---- Action signal projector ------------------------------------
        # Projects encoded observations to an action-signal space for the
        # risk estimator.
        self.action_signal_projector = nn.Sequential(
            nn.Linear(config.opponent_latent_dim, action_dim),
            nn.SiLU(),
        )

        # ---- Action encoder for deception detection ---------------------
        self.action_encoder = nn.Sequential(
            nn.Linear(config.opponent_latent_dim, action_dim),
            nn.SiLU(),
        )

        # ---- Context projection -----------------------------------------
        # Projects per-opponent context to the main latent space
        self.context_projector = nn.Sequential(
            nn.Linear(config.opponent_latent_dim + config.tendency_dim, latent_dim),
            nn.SiLU(),
            nn.LayerNorm(latent_dim),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        opponent_obs: torch.Tensor,
        interaction_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process opponent observations and produce an opponent context vector.

        This is the main entry point during agent interaction.  It encodes
        the current opponent observations, updates internal tracking state,
        and produces a rich opponent context that can be integrated into
        the agent's decision-making process.

        Args:
            opponent_obs: Raw opponent observations of shape
                ``(batch, num_opponents, obs_dim)`` or
                ``(num_opponents, obs_dim)`` or ``(obs_dim,)``.
                When the batch dimension is omitted, a batch size of 1
                is assumed.  When the opponent dimension is omitted,
                only a single opponent is assumed.
            interaction_history: Optional sequence of past opponent
                observations for the tendency tracker, of shape
                ``(batch, num_opponents, seq_len, obs_dim)`` or
                ``(num_opponents, seq_len, obs_dim)`` or ``None``.
                When ``None``, only the current observation is used.

        Returns:
            A tuple ``(opponent_context, info)`` where:

            - ``opponent_context`` is a tensor of shape ``(batch, latent_dim)``
              containing the aggregated social context for the agent.
            - ``info`` is a dictionary of diagnostic information.
        """
        if not self.config.use_opponent_modeling:
            batch_size = opponent_obs.shape[0] if opponent_obs.dim() > 1 else 1
            device = opponent_obs.device
            return torch.zeros(batch_size, self.latent_dim, device=device), {
                "opponent_context_norm": 0.0,
                "num_active_opponents": 0,
                "mean_risk": 0.0,
                "mean_deception": 0.0,
                "mean_predictability": 0.0,
            }

        # Normalise input dimensions
        squeeze_batch = False
        if opponent_obs.dim() == 1:
            # (obs_dim,) -> (1, 1, obs_dim)
            opponent_obs = opponent_obs.unsqueeze(0).unsqueeze(0)
            squeeze_batch = True
        elif opponent_obs.dim() == 2:
            # (num_opponents, obs_dim) -> (1, num_opponents, obs_dim)
            opponent_obs = opponent_obs.unsqueeze(0)
            squeeze_batch = True

        batch_size, num_opponents, obs_dim = opponent_obs.shape
        num_opponents = min(num_opponents, self.config.max_opponents)

        # Encode all opponent observations
        # Reshape to (batch * num_opponents, obs_dim) for batched encoding
        flat_obs = opponent_obs[:, :num_opponents, :].reshape(
            batch_size * num_opponents, obs_dim
        )
        encoded = self.opponent_encoder(flat_obs)  # (B*O, latent)
        encoded = encoded.reshape(batch_size, num_opponents, -1)  # (B, O, latent)

        # Process through tendency tracker for each opponent
        all_tendencies: list[torch.Tensor] = []
        all_tendency_logits: list[torch.Tensor] = []
        all_predictabilities: list[torch.Tensor] = []

        for opp_idx in range(num_opponents):
            # Get the sequence for this opponent
            if interaction_history is not None:
                if interaction_history.dim() == 3:
                    # (num_opponents, seq_len, obs_dim)
                    opp_history = interaction_history[opp_idx]  # (seq_len, obs_dim)
                elif interaction_history.dim() == 4:
                    # (batch, num_opponents, seq_len, obs_dim)
                    opp_history = interaction_history[:, opp_idx, :, :]  # (batch, seq, obs)
                else:
                    opp_history = None
            else:
                opp_history = None

            # Current encoded observation for this opponent
            current_encoded = encoded[:, opp_idx, :]  # (batch, latent)

            # Use stored GRU hidden state, expanded to full batch dimension
            # so every batch element gets the correct hidden state
            hidden = self.gru_hidden_states[:, opp_idx, :].unsqueeze(1).expand(-1, batch_size, -1).contiguous()  # (layers, batch_size, hidden)

            # If we have interaction history, encode it first
            if opp_history is not None and opp_history.dim() == 2:
                # (seq_len, obs_dim) — encode history
                encoded_history = self.opponent_encoder(opp_history)  # (seq, latent)
                encoded_history = encoded_history.unsqueeze(0)  # (1, seq, latent)
                tendency_emb, tendency_logits, predictability, new_hidden = (
                    self.tendency_tracker(encoded_history, hidden)
                )
            else:
                # Use just the current observation
                tendency_emb, tendency_logits, predictability, new_hidden = (
                    self.tendency_tracker(current_encoded, hidden)
                )

            all_tendencies.append(tendency_emb.mean(dim=0))  # average over batch
            all_tendency_logits.append(tendency_logits.mean(dim=0))
            all_predictabilities.append(predictability.mean(dim=0))

            # Update persistent buffers (using first batch element)
            with torch.no_grad():
                # Update tendency vector with EMA
                current_tendency = self.tendency_vectors[opp_idx]
                new_tendency = (
                    (1.0 - self.config.tendency_update_rate) * current_tendency
                    + self.config.tendency_update_rate * tendency_emb[0].detach()
                )
                self.tendency_vectors[opp_idx] = new_tendency

                # Update strategy embedding
                strategy_preds = self.strategy_predictor(tendency_emb[0].detach().unsqueeze(0))
                self.strategy_embeddings[opp_idx] = strategy_preds[0, 0, :].detach()

                # Update risk estimate
                action_signal = self.action_signal_projector(current_encoded[0].detach().unsqueeze(0))
                instant_risk = self.risk_estimator(
                    tendency_emb[0].detach().unsqueeze(0), action_signal
                )
                self.risk_ema[opp_idx] = self.risk_estimator.update_ema(
                    self.risk_ema[opp_idx], instant_risk.item()
                )

                # Update predictability
                self.predictability_scores[opp_idx] = (
                    (1.0 - self.config.tendency_update_rate) * self.predictability_scores[opp_idx]
                    + self.config.tendency_update_rate * predictability[0, 0].detach().item()
                )

                # Increment interaction count
                self.interaction_counts[opp_idx] += 1

                # Update GRU hidden state
                self.gru_hidden_states[:, opp_idx, :] = new_hidden[:, 0, :].detach()

        # ---- Build opponent context ------------------------------------
        # For each opponent, concatenate tendency + strategy embeddings
        # and project to the main latent space.  Then aggregate across
        # opponents into a single social context vector.
        per_opponent_contexts: list[torch.Tensor] = []
        for opp_idx in range(num_opponents):
            combined = torch.cat(
                [self.tendency_vectors[opp_idx], self.strategy_embeddings[opp_idx]],
                dim=-1,
            )
            context = self.context_projector(combined)  # (latent_dim,)
            per_opponent_contexts.append(context)

        # Stack and aggregate: (num_opponents, latent_dim)
        if per_opponent_contexts:
            stacked = torch.stack(per_opponent_contexts, dim=0)  # (O, latent)
            # Mean across opponents as a simple aggregation
            opponent_context = stacked.mean(dim=0).unsqueeze(0)  # (1, latent)
        else:
            opponent_context = torch.zeros(1, self.latent_dim, device=opponent_obs.device)

        # Also produce a full social context using the social aggregator
        # Pad to max_opponents for the aggregator
        social_input = torch.zeros(
            1,
            self.config.max_opponents * (self.config.tendency_dim + self.config.opponent_latent_dim),
            device=opponent_obs.device,
        )
        for opp_idx in range(num_opponents):
            start = opp_idx * (self.config.tendency_dim + self.config.opponent_latent_dim)
            end = start + self.config.tendency_dim
            social_input[0, start:end] = self.tendency_vectors[opp_idx]
            social_input[0, end:end + self.config.opponent_latent_dim] = (
                self.strategy_embeddings[opp_idx]
            )

        social_context = self.social_aggregator(social_input)  # (1, latent_dim)

        # Blend opponent context with social context
        opponent_context = 0.5 * opponent_context + 0.5 * social_context

        if squeeze_batch:
            opponent_context = opponent_context.squeeze(0)

        # ---- Diagnostic info -------------------------------------------
        active_mask = self.interaction_counts[:num_opponents] > 0
        num_active = active_mask.sum().item()

        info: Dict[str, Any] = {
            "opponent_context_norm": opponent_context.norm().item(),
            "num_active_opponents": num_active,
            "mean_risk": self.risk_ema[:num_opponents].mean().item() if num_opponents > 0 else 0.0,
            "mean_deception": self.deception_scores[:num_opponents].mean().item() if num_opponents > 0 else 0.0,
            "mean_predictability": self.predictability_scores[:num_opponents].mean().item() if num_opponents > 0 else 0.0,
        }

        return opponent_context, info

    # ------------------------------------------------------------------
    # Individual API methods
    # ------------------------------------------------------------------

    def encode_opponent(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode a raw opponent observation into the opponent latent space.

        This is a convenience method that calls
        :class:`OpponentEncoder` directly.

        Args:
            obs: Raw opponent observation of shape ``(obs_dim,)`` or
                ``(batch, obs_dim)``.

        Returns:
            Encoded opponent representation of shape ``(latent_dim,)`` or
            ``(batch, latent_dim)``.
        """
        return self.opponent_encoder(obs)

    def update_tendencies(
        self,
        opponent_id: int,
        new_observation: torch.Tensor,
    ) -> OpponentProfile:
        """
        Update the tendency profile for a specific opponent.

        Encodes the new observation, runs it through the tendency tracker,
        and updates all persistent state for the given opponent.

        Args:
            opponent_id: Index of the opponent to update.  Must be in
                ``[0, max_opponents)``.
            new_observation: Raw opponent observation of shape
                ``(obs_dim,)``.

        Returns:
            Updated :class:`OpponentProfile` for the opponent.

        Raises:
            ValueError: If ``opponent_id`` is out of range.
        """
        if opponent_id < 0 or opponent_id >= self.config.max_opponents:
            raise ValueError(
                f"opponent_id must be in [0, {self.config.max_opponents}), "
                f"got {opponent_id}"
            )

        # Encode the observation
        encoded = self.opponent_encoder(new_observation)  # (latent,)

        # Run through tendency tracker with stored hidden state
        hidden = self.gru_hidden_states[:, opponent_id, :].unsqueeze(1)  # (layers, 1, hidden)
        tendency_emb, tendency_logits, predictability, new_hidden = (
            self.tendency_tracker(encoded.unsqueeze(0), hidden)
        )
        tendency_emb = tendency_emb[0]   # (tendency_dim,)
        predictability_val = predictability[0, 0].item()

        with torch.no_grad():
            # EMA update for tendency vector
            current_tendency = self.tendency_vectors[opponent_id]
            self.tendency_vectors[opponent_id] = (
                (1.0 - self.config.tendency_update_rate) * current_tendency
                + self.config.tendency_update_rate * tendency_emb.detach()
            )

            # Update strategy embedding
            strategy_preds = self.strategy_predictor(tendency_emb.detach().unsqueeze(0))
            self.strategy_embeddings[opponent_id] = strategy_preds[0, 0, :].detach()

            # Update risk
            action_signal = self.action_signal_projector(encoded.detach().unsqueeze(0))
            instant_risk = self.risk_estimator(
                tendency_emb.detach().unsqueeze(0), action_signal
            )
            self.risk_ema[opponent_id] = self.risk_estimator.update_ema(
                self.risk_ema[opponent_id], instant_risk.item()
            )

            # Update predictability
            self.predictability_scores[opponent_id] = (
                (1.0 - self.config.tendency_update_rate) * self.predictability_scores[opponent_id]
                + self.config.tendency_update_rate * predictability_val
            )

            # Increment interaction count
            self.interaction_counts[opponent_id] += 1

            # Update GRU hidden state
            self.gru_hidden_states[:, opponent_id, :] = new_hidden[:, 0, :].detach()

        return self.get_opponent_profile(opponent_id)

    def predict_strategy(
        self,
        opponent_id: int,
        horizon: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict the opponent's likely future strategy.

        Uses the current tendency embedding for the specified opponent
        to auto-regressively roll out predicted strategy embeddings.

        Args:
            opponent_id: Index of the opponent.
            horizon: Number of future steps to predict.  If ``None``,
                ``config.strategy_horizon`` is used.

        Returns:
            Predicted strategy embeddings of shape
            ``(horizon, opponent_latent_dim)``.

        Raises:
            ValueError: If ``opponent_id`` is out of range.
        """
        if opponent_id < 0 or opponent_id >= self.config.max_opponents:
            raise ValueError(
                f"opponent_id must be in [0, {self.config.max_opponents}), "
                f"got {opponent_id}"
            )

        tendency_emb = self.tendency_vectors[opponent_id].unsqueeze(0)  # (1, tendency_dim)
        predictions = self.strategy_predictor(tendency_emb, horizon=horizon)  # (1, H, D)
        return predictions.squeeze(0)  # (H, D)

    def detect_deception(
        self,
        opponent_id: int,
        claimed_action: torch.Tensor,
        observed_action: torch.Tensor,
    ) -> float:
        """
        Detect whether a specific opponent is being deceptive.

        Compares the claimed action against the observed action using
        the :class:`DeceptionDetector`, taking into account the
        opponent's historical tendency embedding.

        Also updates the stored deception likelihood for this opponent
        using an EMA update.

        Args:
            opponent_id: Index of the opponent.
            claimed_action: The action the opponent claimed or appeared
                to intend, of shape ``(action_dim,)`` or any compatible
                shape.  Will be encoded through the action encoder.
            observed_action: The action the opponent actually took, of
                the same shape as ``claimed_action``.
        Returns:
            Deception likelihood as a float in [0, 1], where 0 means
            honest and 1 means deceptive.

        Raises:
            ValueError: If ``opponent_id`` is out of range.
        """
        if opponent_id < 0 or opponent_id >= self.config.max_opponents:
            raise ValueError(
                f"opponent_id must be in [0, {self.config.max_opponents}), "
                f"got {opponent_id}"
            )

        # Encode actions into the action signal space
        # Ensure 2D shape (batch, dim) for the encoder
        if claimed_action.dim() == 1:
            claimed_action = claimed_action.unsqueeze(0)
        if observed_action.dim() == 1:
            observed_action = observed_action.unsqueeze(0)
        claimed_encoded = self.action_encoder(claimed_action)  # (batch, action_dim)
        observed_encoded = self.action_encoder(observed_action)  # (batch, action_dim)

        # Get tendency embedding as context
        tendency_emb = self.tendency_vectors[opponent_id].unsqueeze(0)  # (1, tendency_dim)
        # Expand tendency to match batch if needed
        if claimed_encoded.size(0) > 1:
            tendency_emb = tendency_emb.expand(claimed_encoded.size(0), -1)

        # Compute deception score
        deception_score = self.deception_detector(
            claimed_encoded, observed_encoded, tendency_emb
        )  # (1, 1)

        score_val = deception_score[0, 0].item()

        # Update stored deception score with EMA
        with torch.no_grad():
            self.deception_scores[opponent_id] = (
                self.config.risk_estimation_ema * self.deception_scores[opponent_id]
                + (1.0 - self.config.risk_estimation_ema) * score_val
            )

        return score_val

    def estimate_risk(self, opponent_id: int) -> float:
        """
        Estimate the risk preference of a specific opponent.

        Returns the EMA risk estimate that has been accumulated over
        all interactions with this opponent.

        Args:
            opponent_id: Index of the opponent.

        Returns:
            Risk preference as a float in [0, 1], where 0 means
            risk-averse and 1 means risk-seeking.

        Raises:
            ValueError: If ``opponent_id`` is out of range.
        """
        if opponent_id < 0 or opponent_id >= self.config.max_opponents:
            raise ValueError(
                f"opponent_id must be in [0, {self.config.max_opponents}), "
                f"got {opponent_id}"
            )

        return self.risk_ema[opponent_id].item()

    def get_opponent_profile(self, opponent_id: int) -> OpponentProfile:
        """
        Get the full profile for a specific opponent.

        Returns an :class:`OpponentProfile` dataclass containing all
        tracked information about the opponent.

        Args:
            opponent_id: Index of the opponent.

        Returns:
            An :class:`OpponentProfile` instance with the current
            estimates for the opponent.

        Raises:
            ValueError: If ``opponent_id`` is out of range.
        """
        if opponent_id < 0 or opponent_id >= self.config.max_opponents:
            raise ValueError(
                f"opponent_id must be in [0, {self.config.max_opponents}), "
                f"got {opponent_id}"
            )

        return OpponentProfile(
            opponent_id=opponent_id,
            tendency_vector=self.tendency_vectors[opponent_id].clone(),
            risk_preference=self.risk_ema[opponent_id].item(),
            deception_likelihood=self.deception_scores[opponent_id].item(),
            strategy_embedding=self.strategy_embeddings[opponent_id].clone(),
            predictability=self.predictability_scores[opponent_id].item(),
            interaction_count=self.interaction_counts[opponent_id].item(),
        )

    def get_social_context(self) -> torch.Tensor:
        """
        Compute an aggregate social context vector across all opponents.

        Concatenates all opponent tendency vectors and strategy embeddings
        into a single fixed-length vector and projects it through the
        social aggregator network.  This provides the rest of the Deep
        Thought system with a summary of the entire social landscape.

        Returns:
            Social context tensor of shape ``(latent_dim,)``.
        """
        # Build the concatenated social input
        social_input = torch.cat(
            [self.tendency_vectors, self.strategy_embeddings],
            dim=-1,
        )  # (max_opponents, tendency_dim + latent_dim)

        # Flatten and project
        social_input = social_input.reshape(1, -1)  # (1, max_opp * (tendency + strategy))
        social_context = self.social_aggregator(social_input)  # (1, latent_dim)

        return social_context.squeeze(0)  # (latent_dim,)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def reset_opponent(self, opponent_id: int) -> None:
        """
        Reset all tracked state for a specific opponent.

        Useful when an opponent leaves the environment or when a
        profile needs to be re-initialised.

        Args:
            opponent_id: Index of the opponent to reset.

        Raises:
            ValueError: If ``opponent_id`` is out of range.
        """
        if opponent_id < 0 or opponent_id >= self.config.max_opponents:
            raise ValueError(
                f"opponent_id must be in [0, {self.config.max_opponents}), "
                f"got {opponent_id}"
            )

        with torch.no_grad():
            self.tendency_vectors[opponent_id].zero_()
            self.strategy_embeddings[opponent_id].zero_()
            self.risk_ema[opponent_id] = 0.5
            self.deception_scores[opponent_id] = 0.0
            self.predictability_scores[opponent_id] = 0.5
            self.interaction_counts[opponent_id] = 0
            self.gru_hidden_states[:, opponent_id, :].zero_()

    def reset_all_opponents(self) -> None:
        """
        Reset all tracked opponent profiles.

        Useful at the start of a new episode or when the environment
        changes significantly.
        """
        with torch.no_grad():
            self.tendency_vectors.zero_()
            self.strategy_embeddings.zero_()
            self.risk_ema.fill_(0.5)
            self.deception_scores.zero_()
            self.predictability_scores.fill_(0.5)
            self.interaction_counts.zero_()
            self.gru_hidden_states.zero_()

    def get_active_opponent_ids(self) -> list[int]:
        """
        Return a list of opponent IDs that have been interacted with.

        Returns:
            List of integer opponent IDs with ``interaction_count > 0``.
        """
        active = (self.interaction_counts > 0).nonzero(as_tuple=True)[0]
        return active.tolist()

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return comprehensive diagnostic information about the opponent
        modeling system.

        Returns:
            Dictionary containing:
            - ``num_active_opponents``: Number of opponents with
              ``interaction_count > 0``.
            - ``mean_risk``: Mean risk preference across active opponents.
            - ``mean_deception``: Mean deception likelihood across active
              opponents.
            - ``mean_predictability``: Mean predictability across active
              opponents.
            - ``per_opponent_risk``: Per-opponent risk estimates.
            - ``per_opponent_deception``: Per-opponent deception scores.
            - ``per_opponent_predictability``: Per-opponent predictability.
            - ``per_opponent_interactions``: Per-opponent interaction counts.
        """
        active_ids = self.get_active_opponent_ids()
        num_active = len(active_ids)

        if num_active > 0:
            active_indices = torch.tensor(active_ids, device=self.risk_ema.device)
            mean_risk = self.risk_ema[active_indices].mean().item()
            mean_deception = self.deception_scores[active_indices].mean().item()
            mean_predictability = self.predictability_scores[active_indices].mean().item()
        else:
            mean_risk = 0.0
            mean_deception = 0.0
            mean_predictability = 0.0

        return {
            "num_active_opponents": num_active,
            "mean_risk": mean_risk,
            "mean_deception": mean_deception,
            "mean_predictability": mean_predictability,
            "per_opponent_risk": self.risk_ema.tolist(),
            "per_opponent_deception": self.deception_scores.tolist(),
            "per_opponent_predictability": self.predictability_scores.tolist(),
            "per_opponent_interactions": self.interaction_counts.tolist(),
        }
