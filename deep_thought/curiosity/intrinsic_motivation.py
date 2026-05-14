"""
Intrinsic motivation and curiosity systems for Deep Thought.

Implements multiple complementary curiosity drives that provide exploration
bonuses beyond external rewards:

1. **PredictionErrorCuriosity** — Uses prediction error from the world model
   as a curiosity signal. States that are hard to predict are inherently
   interesting and deserve further exploration.

2. **NoveltyBonus** — Count-based exploration using state visitation counts
   with learnable hashing. Novel (rarely visited) states receive bonus
   rewards that diminish as the agent becomes familiar with them.

3. **UncertaintyReduction** — Ensemble disagreement as a curiosity signal.
   Regions of the state space where an ensemble of models disagree are
   more interesting because they represent epistemic uncertainty.

4. **InformationGainBonus** — Estimates how much observing a state reduces
   uncertainty about the environment, encouraging the agent to seek out
   informative experiences.

All components are combined in `IntrinsicMotivationSystem`, which produces a
single intrinsic reward signal and tracks detailed diagnostic information.
Curiosity naturally decays over time as the agent learns, preventing
inefficient perpetual exploration.

References:
    - Pathak et al., "Curiosity-driven Exploration by Self-supervised
      Prediction" (2017)
    - Bellemare et al., "Unifying Count-Based Exploration and Intrinsic
      Motivation" (2016)
    - Burda et al., "Exploration by Random Network Distillation" (2018)
    - Pathak et al., "Self-Supervised Exploration via Disagreement" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, Any

from deep_thought.config import CuriosityConfig


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class PredictionErrorCuriosity(nn.Module):
    """
    Prediction-error-based curiosity module.

    Uses the prediction error from a world model as the curiosity signal.
    If the agent cannot predict what happens next, the state is considered
    surprising and the agent receives an intrinsic reward proportional to the
    prediction error. This encourages the agent to seek out states where its
    model is inaccurate, thereby improving the world model over time.

    The prediction error is passed through a learnable scaling network so the
    agent can adaptively weight different dimensions of the error.

    Args:
        latent_dim: Dimensionality of the latent state representation.
        error_dim: Dimensionality of the prediction error vector.
            Defaults to ``latent_dim``.
        hidden_dim: Hidden layer size for the scaling network.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        error_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.error_dim = error_dim

        # Learnable scaling network: maps prediction error to a scalar
        # curiosity bonus, allowing the agent to focus on error dimensions
        # that are most informative.
        self.scaler = nn.Sequential(
            nn.Linear(error_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Running statistics for normalising prediction errors across
        # training (prevents scale drift).
        self.register_buffer("error_mean", torch.zeros(error_dim))
        self.register_buffer("error_var", torch.ones(error_dim))
        self.register_buffer("update_count", torch.tensor(0.0))
        self._eps = 1e-8
        self._momentum = 0.01

    def update_error_stats(self, prediction_error: torch.Tensor) -> None:
        """
        Update running mean/variance of prediction errors for normalisation.

        Uses Welford-style online updates with exponential momentum.

        Args:
            prediction_error: Raw prediction error tensor of shape
                ``(batch, error_dim)`` or ``(error_dim,)``.
        """
        with torch.no_grad():
            if prediction_error.dim() == 1:
                prediction_error = prediction_error.unsqueeze(0)

            batch_mean = prediction_error.mean(dim=0)
            batch_var = prediction_error.var(dim=0, unbiased=False)

            self.update_count += 1
            n = self.update_count

            # Exponential moving average update
            self.error_mean = (1 - self._momentum) * self.error_mean + self._momentum * batch_mean
            self.error_var = (1 - self._momentum) * self.error_var + self._momentum * batch_var

    def forward(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction-error-based curiosity bonus.

        The raw prediction error is first normalised using running statistics
        and then passed through a learnable scaler to produce a scalar bonus.

        Args:
            prediction_error: Prediction error tensor of shape
                ``(batch, error_dim)`` or ``(error_dim,)``.

        Returns:
            Curiosity bonus of shape ``(batch, 1)`` or ``(1,)``.
        """
        if prediction_error.dim() == 1:
            prediction_error = prediction_error.unsqueeze(0)

        # Normalise by running statistics
        normalised = (prediction_error - self.error_mean) / torch.sqrt(self.error_var + self._eps)

        # Learnable scaling
        bonus = self.scaler(normalised)
        return F.softplus(bonus)  # ensure non-negative


class NoveltyBonus(nn.Module):
    """
    Count-based novelty bonus using learnable state hashing.

    Maintains a fixed-size hash table of state visitation counts.  States
    are embedded and quantised into hash buckets via a learnable projection
    followed by a sign-based hash.  The novelty bonus is inversely
    proportional to the square root of the visit count, following the
    established ``1/sqrt(N)`` bonus schedule (Bellemare et al., 2016).

    The learnable projection allows the agent to discover a hashing scheme
    that groups states meaningfully, rather than relying on a fixed
    discretisation.

    Args:
        latent_dim: Dimensionality of the latent state representation.
        hash_size: Number of buckets in the hash table.
        num_projections: Number of random projection vectors used for
            the hash function.  More projections give finer-grained
            state discrimination at higher memory cost.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hash_size: int = 10000,
        num_projections: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hash_size = hash_size
        self.num_projections = num_projections

        # Learnable projection for hashing — projects latent states to a
        # lower-dimensional space whose sign pattern determines the hash.
        self.projection = nn.Linear(latent_dim, num_projections, bias=False)
        # Initialise with small orthogonal-ish weights for stable hashing
        nn.init.xavier_normal_(self.projection.weight)

        # Visitation counts stored as a buffer so they persist across
        # save/load but are *not* updated by gradient descent.
        self.register_buffer(
            "visit_counts", torch.zeros(hash_size, dtype=torch.float32)
        )

        # Total visits for global statistics
        self.register_buffer("total_visits", torch.tensor(0.0))

    def _compute_hash(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute hash indices for a batch of latent states.

        Projects the latent through a learnable linear layer, takes the sign
        of each dimension, interprets the sign vector as a binary code, and
        maps the code to a hash-table index via modulo.

        Args:
            latent: Latent state tensor of shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.

        Returns:
            Integer hash indices of shape ``(batch,)``.
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        projected = self.projection(latent)  # (batch, num_projections)
        # Sign-based binary hash: each dimension contributes one bit
        binary_code = (projected > 0).float()  # (batch, num_projections)
        # Convert binary code to integer via positional weighting
        powers = 2.0 ** torch.arange(
            self.num_projections, device=latent.device, dtype=torch.float32
        )
        int_code = (binary_code * powers).sum(dim=-1)  # (batch,)
        # Map to hash-table range
        hash_indices = int_code.long() % self.hash_size
        return hash_indices

    def update_visit_counts(self, latent: torch.Tensor) -> None:
        """
        Increment visitation counts for the given latent states.

        Args:
            latent: Latent state tensor of shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.
        """
        with torch.no_grad():
            hash_indices = self._compute_hash(latent)
            batch_size = hash_indices.shape[0]

            # Increment counts (scatter-add for batch support)
            ones = torch.ones(batch_size, device=latent.device)
            self.visit_counts.scatter_add_(0, hash_indices, ones)
            self.total_visits += batch_size

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty bonus for the given latent states.

        The bonus is ``1 / sqrt(N(s) + 1)`` where ``N(s)`` is the visit
        count of the hash bucket corresponding to state *s*.

        Args:
            latent: Latent state tensor of shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.

        Returns:
            Novelty bonus of shape ``(batch,)``.
        """
        with torch.no_grad():
            hash_indices = self._compute_hash(latent)
            counts = self.visit_counts[hash_indices]
            bonus = 1.0 / torch.sqrt(counts + 1.0)
        return bonus


class UncertaintyReduction(nn.Module):
    """
    Ensemble-disagreement-based curiosity module.

    Maintains a small ensemble of lightweight prediction heads.  The
    disagreement (variance) across ensemble members for a given state
    quantifies epistemic uncertainty — the model doesn't know what it
    doesn't know.  High disagreement signals unexplored or uncertain
    regions of the state space that deserve further visits.

    The ensemble disagreement is mapped to a scalar curiosity bonus
    through a learnable scaling network.

    Args:
        latent_dim: Dimensionality of the latent state representation.
        ensemble_size: Number of ensemble members.
        hidden_dim: Hidden layer size for each ensemble head.
        output_dim: Prediction output dimension for each head.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        ensemble_size: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ensemble_size = ensemble_size
        self.output_dim = output_dim

        # Ensemble heads — each is a small MLP predicting a latent target
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(ensemble_size)
        ])

        # Learnable scaler: maps disagreement vector to scalar bonus
        self.scaler = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, latent: torch.Tensor, ensemble_uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute uncertainty-based curiosity bonus.

        If ``ensemble_uncertainty`` is provided directly it is used;
        otherwise the ensemble disagreement is computed from the latent
        by running all heads.

        Args:
            latent: Latent state tensor of shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.
            ensemble_uncertainty: Pre-computed ensemble disagreement tensor
                of shape ``(batch, output_dim)``.  When ``None``, the
                disagreement is computed internally.

        Returns:
            Curiosity bonus of shape ``(batch, 1)``.
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        if ensemble_uncertainty is not None:
            disagreement = ensemble_uncertainty
            if disagreement.dim() == 1:
                disagreement = disagreement.unsqueeze(0)
        else:
            # Compute ensemble disagreement
            predictions = torch.stack(
                [head(latent) for head in self.heads], dim=0
            )  # (ensemble, batch, output_dim)
            # Variance across ensemble members
            disagreement = predictions.var(dim=0)  # (batch, output_dim)

        bonus = self.scaler(disagreement)  # (batch, 1)
        return F.softplus(bonus)


class InformationGainBonus(nn.Module):
    """
    Information-gain-based curiosity module.

    Estimates how much a new observation reduces uncertainty about the
    environment.  This is implemented via a *surprise* model: a
    learnable density estimator that tracks how "surprising" each state
    is.  States that significantly change the density estimate carry
    high information gain and are rewarded.

    Concretely, the module maintains an autoencoder-style density model.
    The reconstruction error of a state under the *current* density model
    is used as a proxy for information gain — states that are poorly
    reconstructed (high reconstruction error) are far from the current
    manifold and thus carry new information.

    As the model adapts to previously surprising states, the information
    gain naturally decreases, implementing an automatic curriculum.

    Args:
        latent_dim: Dimensionality of the latent state representation.
        hidden_dim: Hidden layer size for the density model.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Density / autoencoder model
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Running reconstruction error statistics for normalisation
        self.register_buffer("recon_mean", torch.tensor(0.0))
        self.register_buffer("recon_var", torch.tensor(1.0))
        self._momentum = 0.01

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute information-gain-based curiosity bonus.

        Args:
            latent: Latent state tensor of shape ``(batch, latent_dim)`` or
                ``(latent_dim,)``.

        Returns:
            Information gain bonus of shape ``(batch,)``.
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Encode → decode
        z = self.encoder(latent)
        recon = self.decoder(z)

        # Per-sample reconstruction error
        recon_error = (latent - recon).pow(2).sum(dim=-1)  # (batch,)

        # Normalise by running statistics
        with torch.no_grad():
            batch_mean = recon_error.mean()
            batch_var = recon_error.var() if recon_error.shape[0] > 1 else torch.tensor(1.0, device=latent.device)
            self.recon_mean = (1 - self._momentum) * self.recon_mean + self._momentum * batch_mean
            self.recon_var = (1 - self._momentum) * self.recon_var + self._momentum * batch_var

        normalised = (recon_error - self.recon_mean) / torch.sqrt(self.recon_var + 1e-8)
        return F.softplus(normalised)


# ---------------------------------------------------------------------------
# Main system
# ---------------------------------------------------------------------------


class IntrinsicMotivationSystem(nn.Module):
    """
    Intrinsic motivation system that provides exploration bonuses beyond
    external rewards.

    Combines four complementary curiosity signals:

    1. **Prediction Error Curiosity** — Surprising (hard-to-predict) states
       are intrinsically rewarding.
    2. **Novelty Bonus** — Rarely-visited states receive count-based
       exploration bonuses.
    3. **Uncertainty Reduction** — Ensemble disagreement signals epistemic
       uncertainty and drives exploration.
    4. **Information Gain Bonus** — States that substantially change the
       agent's belief about the environment are rewarded.

    The total intrinsic reward is a weighted sum of these components,
    subject to clamping and optional temporal decay.  The decay factor
    gradually reduces curiosity over time, preventing inefficient
    perpetual exploration as the agent's world model improves.

    All sub-modules contain learnable parameters and are trained as part
    of the larger Deep Thought system.

    Args:
        config: A :class:`CuriosityConfig` instance controlling all
            hyperparameters.  When ``None``, defaults are used.

    Example::

        config = CuriosityConfig(
            prediction_error_coef=0.1,
            novelty_coef=0.05,
            uncertainty_coef=0.05,
            info_gain_coef=0.02,
        )
        curiosity = IntrinsicMotivationSystem(config)

        # During interaction:
        intrinsic_reward, info = curiosity(
            latent=state_embedding,          # (batch, latent_dim)
            prediction_error=pred_err,       # (batch, latent_dim)
            ensemble_uncertainty=unc,        # (batch, output_dim)
        )
        curiosity.update_visit_counts(state_embedding)
    """

    def __init__(self, config: Optional[CuriosityConfig] = None):
        super().__init__()

        if config is None:
            config = CuriosityConfig()

        self.config = config

        # ---- Sub-modules ------------------------------------------------
        self.prediction_curiosity = PredictionErrorCuriosity(
            latent_dim=config.state_embedding_dim,
            error_dim=config.state_embedding_dim,
            hidden_dim=config.state_embedding_dim * 2,
        )

        self.novelty_bonus = NoveltyBonus(
            latent_dim=config.state_embedding_dim,
            hash_size=config.visit_count_hash_size,
            num_projections=32,
        )

        self.uncertainty_curiosity = UncertaintyReduction(
            latent_dim=config.state_embedding_dim,
            ensemble_size=5,
            hidden_dim=config.state_embedding_dim * 2,
            output_dim=32,
        )

        self.info_gain_bonus = InformationGainBonus(
            latent_dim=config.state_embedding_dim,
            hidden_dim=config.state_embedding_dim * 2,
        )

        # Learnable state embedding — projects arbitrary latent
        # representations into a consistent embedding space for the
        # curiosity sub-modules.
        self.state_embedding = nn.Sequential(
            nn.Linear(config.state_embedding_dim, config.state_embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(config.state_embedding_dim),
        )

        # ---- Curiosity decay tracking -----------------------------------
        # ``curiosity_scale`` starts at 1.0 and decays toward
        # ``min_curiosity`` over time.
        self.register_buffer("curiosity_scale", torch.tensor(1.0))
        self.register_buffer("decay_step", torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        latent: torch.Tensor,
        prediction_error: torch.Tensor,
        ensemble_uncertainty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the total intrinsic reward for a batch of states.

        Args:
            latent: Latent state representation of shape ``(batch,
                state_embedding_dim)`` or ``(state_embedding_dim,)``.
            prediction_error: World-model prediction error of shape
                ``(batch, state_embedding_dim)`` or
                ``(state_embedding_dim,)``.
            ensemble_uncertainty: Pre-computed ensemble disagreement of
                shape ``(batch, output_dim)`` or ``None`` (will be
                computed internally).

        Returns:
            A tuple ``(intrinsic_reward, info)`` where:

            - ``intrinsic_reward`` is a tensor of shape ``(batch,)`` with
              the total intrinsic reward per sample.
            - ``info`` is a dictionary of diagnostic information
              containing individual component bonuses and metadata.
        """
        if not self.config.use_curiosity:
            batch_size = latent.shape[0] if latent.dim() > 1 else 1
            return torch.zeros(batch_size, device=latent.device), {
                "intrinsic_reward": 0.0,
                "prediction_curiosity": 0.0,
                "novelty_bonus": 0.0,
                "uncertainty_curiosity": 0.0,
                "info_gain_bonus": 0.0,
                "curiosity_scale": 0.0,
            }

        # Ensure batch dimension
        squeeze_output = False
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            squeeze_output = True
        if prediction_error.dim() == 1:
            prediction_error = prediction_error.unsqueeze(0)
        if ensemble_uncertainty is not None and ensemble_uncertainty.dim() == 1:
            ensemble_uncertainty = ensemble_uncertainty.unsqueeze(0)

        # Project latent through the learnable embedding
        embedded = self.state_embedding(latent)

        # --- Component bonuses ---
        pred_bonus = self.get_prediction_curiosity(prediction_error)  # (batch, 1)
        novelty = self.get_curiosity_bonus(embedded)                  # (batch,)
        unc_bonus = self.get_uncertainty_curiosity(
            embedded, ensemble_uncertainty=ensemble_uncertainty
        )                                                              # (batch, 1)
        info_gain = self.info_gain_bonus(embedded)                    # (batch,)

        # Squeeze extra dims for consistent shapes → (batch,)
        pred_bonus = pred_bonus.squeeze(-1)
        unc_bonus = unc_bonus.squeeze(-1)

        # Weighted sum
        total = (
            self.config.prediction_error_coef * pred_bonus
            + self.config.novelty_coef * novelty
            + self.config.uncertainty_coef * unc_bonus
            + self.config.info_gain_coef * info_gain
        )

        # Apply curiosity decay scale
        total = total * self.curiosity_scale

        # Clamp to maximum intrinsic reward
        total = total.clamp(max=self.config.max_intrinsic_reward)

        # Diagnostic info
        info: Dict[str, Any] = {
            "intrinsic_reward": total.detach().mean().item(),
            "prediction_curiosity": pred_bonus.detach().mean().item(),
            "novelty_bonus": novelty.detach().mean().item(),
            "uncertainty_curiosity": unc_bonus.detach().mean().item(),
            "info_gain_bonus": info_gain.detach().mean().item(),
            "curiosity_scale": self.curiosity_scale.item(),
            "total_visits": self.novelty_bonus.total_visits.item(),
        }

        if squeeze_output:
            total = total.squeeze(0)

        return total, info

    def update_visit_counts(self, latent: torch.Tensor) -> None:
        """
        Update state visitation counts for novelty tracking.

        Should be called once per environment step with the current
        latent state so that the novelty bonus can correctly track how
        often each state (or state cluster) has been visited.

        Args:
            latent: Latent state representation of shape ``(batch,
                state_embedding_dim)`` or ``(state_embedding_dim,)``.
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        embedded = self.state_embedding(latent)
        self.novelty_bonus.update_visit_counts(embedded)

    def get_curiosity_bonus(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Get the count-based novelty bonus for a latent state.

        The bonus is inversely proportional to the square root of the
        visit count: ``1 / sqrt(N(s) + 1)``.

        Args:
            latent: Embedded latent state of shape ``(batch,
                state_embedding_dim)`` or ``(state_embedding_dim,)``.

        Returns:
            Novelty bonus of shape ``(batch,)``.
        """
        return self.novelty_bonus(latent)

    def get_prediction_curiosity(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """
        Scale a world-model prediction error into a curiosity bonus.

        The prediction error is normalised by running statistics and
        passed through a learnable scaler, then softplus-activated to
        ensure non-negativity.

        Args:
            prediction_error: Raw prediction error of shape ``(batch,
                state_embedding_dim)`` or ``(state_embedding_dim,)``.

        Returns:
            Curiosity bonus of shape ``(batch, 1)``.
        """
        # Update running statistics for normalisation
        self.prediction_curiosity.update_error_stats(prediction_error)
        return self.prediction_curiosity(prediction_error)

    def get_uncertainty_curiosity(
        self,
        latent: torch.Tensor,
        ensemble_uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scale ensemble disagreement into a curiosity bonus.

        When ``ensemble_uncertainty`` is provided it is used directly;
        otherwise the disagreement is computed from the latent by running
        the internal ensemble heads.

        Args:
            latent: Embedded latent state of shape ``(batch,
                state_embedding_dim)`` or ``(state_embedding_dim,)``.
            ensemble_uncertainty: Pre-computed ensemble disagreement of
                shape ``(batch, output_dim)`` or ``None``.

        Returns:
            Curiosity bonus of shape ``(batch, 1)``.
        """
        return self.uncertainty_curiosity(latent, ensemble_uncertainty)

    def decay_curiosity(self) -> None:
        """
        Decay the global curiosity scale by the configured decay factor.

        As the agent learns more about the environment, curiosity should
        naturally decrease to shift the focus from exploration to
        exploitation.  The scale never drops below
        ``config.min_curiosity``.

        Should be called once per training step.
        """
        self.curiosity_scale.mul_(self.config.curiosity_decay)
        self.curiosity_scale.clamp_(min=self.config.min_curiosity)
        self.decay_step.add_(1)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def reset_visit_counts(self) -> None:
        """
        Reset all visitation counts to zero.

        Useful when starting a new episode or when the state space
        changes significantly (e.g., after a transfer to a new
        environment).
        """
        self.novelty_bonus.visit_counts.zero_()
        self.novelty_bonus.total_visits.zero_()

    def get_curiosity_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive curiosity diagnostic statistics.

        Returns:
            Dictionary containing:
            - ``curiosity_scale``: Current global decay scale.
            - ``total_visits``: Total state visits recorded.
            - ``mean_visit_count``: Mean visits per hash bucket.
            - ``max_visit_count``: Maximum visits in any bucket.
            - ``num_visited_buckets``: Number of buckets visited at least once.
            - ``decay_step``: Number of decay steps applied.
        """
        counts = self.novelty_bonus.visit_counts
        visited_mask = counts > 0
        return {
            "curiosity_scale": self.curiosity_scale.item(),
            "total_visits": self.novelty_bonus.total_visits.item(),
            "mean_visit_count": counts.mean().item(),
            "max_visit_count": counts.max().item(),
            "num_visited_buckets": visited_mask.sum().item(),
            "decay_step": self.decay_step.item(),
        }
