"""
Attention-Based Probability Maps for Deep Thought RL Framework.

Implements attention-driven probability maps that allocate compute dynamically
based on confidence, temporal evolution, and uncertainty focus.  Regions with
low confidence receive more compute; high-risk regions are prioritised while
low-information zones stay compressed.

Components
----------
ConfidenceTracker
    Tracks per-feature confidence via exponential moving averages of prediction
    errors.  Low prediction error → high confidence → less compute needed.
    High prediction error → low confidence → more compute needed.

UncertaintyFocus
    Identifies high-uncertainty regions that require additional compute by
    measuring ensemble disagreement (when available) or prediction variance.

TemporalEvolution
    A GRU-based module that evolves the attention map over time, incorporating
    prediction error and novelty signals to adaptively shift compute allocation.

AttentionProbabilityMap
    The main module that orchestrates confidence tracking, uncertainty focus,
    and temporal evolution to produce attention-weighted latent representations
    with dynamic compute allocation.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_thought.config import AttentionMapsConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A computationally efficient alternative to LayerNorm that normalises by
    the root mean square of the input without requiring a bias term.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5 + self.eps)
        return self.weight * x / (norm + self.eps)


# ---------------------------------------------------------------------------
# ConfidenceTracker
# ---------------------------------------------------------------------------


class ConfidenceTracker(nn.Module):
    """Tracks confidence per feature dimension using exponential moving averages.

    Confidence is inversely related to the running average of prediction errors:
    low prediction error ⟹ high confidence ⟹ less compute needed;
    high prediction error ⟹ low confidence ⟹ more compute needed.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation being tracked.
    decay : float
        Exponential moving average decay factor.  Values closer to 1.0 make
        confidence estimates more stable; values closer to 0.0 make them more
        reactive to recent errors.
    """

    def __init__(self, latent_dim: int, decay: float = 0.99) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.decay = decay

        # Running EMA of per-dimension squared prediction errors.
        # Initialised to 1.0 (maximum uncertainty → zero confidence).
        self.register_buffer(
            "running_error",
            torch.ones(latent_dim),
        )

        # Number of updates applied so far (used for bias correction).
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, prediction_error: torch.Tensor) -> torch.Tensor:
        """Update the running error estimate and return current confidence.

        Parameters
        ----------
        prediction_error : Tensor
            Per-dimension prediction error, shape ``[B, D]`` or ``[D]``.
            If a batch is supplied the mean over the batch dimension is used.

        Returns
        -------
        Tensor
            Per-dimension confidence in ``[0, 1]``, shape ``[D]``.
        """
        if prediction_error.dim() == 1:
            error = prediction_error
        else:
            error = prediction_error.mean(dim=0)

        # EMA update
        self.num_updates += 1
        self.running_error.mul_(self.decay).add_(
            error.detach(), alpha=1.0 - self.decay
        )

        return self.get_confidence()

    def get_confidence(self) -> torch.Tensor:
        """Return the current per-dimension confidence.

        Confidence is defined as ``1 / (1 + running_error)`` so that it is
        bounded in ``(0, 1]`` with 1 representing full confidence.
        """
        return 1.0 / (1.0 + self.running_error)

    def get_attention_weights(self, min_attention: float = 0.01) -> torch.Tensor:
        """Compute attention weights derived from confidence.

        Low confidence ⟹ high weight (needs more compute).
        The weights are normalised to sum to 1 and clamped above
        *min_attention* to prevent zero attention.

        Parameters
        ----------
        min_attention : float
            Floor value for each weight before normalisation.

        Returns
        -------
        Tensor
            Normalised attention weights, shape ``[D]``.
        """
        confidence = self.get_confidence()
        # Invert: low confidence → high attention needed
        weights = 1.0 - confidence
        weights = weights.clamp(min=min_attention)
        weights = weights / weights.sum()
        return weights

    def reset(self) -> None:
        """Reset tracker to initial state (maximum uncertainty)."""
        self.running_error.fill_(1.0)
        self.num_updates.fill_(0)


# ---------------------------------------------------------------------------
# UncertaintyFocus
# ---------------------------------------------------------------------------


class UncertaintyFocus(nn.Module):
    """Identifies high-uncertainty regions that require additional compute.

    When an ensemble of predictions is available the module measures
    *disagreement* across ensemble members.  Otherwise it falls back to
    per-dimension prediction variance.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation.
    threshold : float
        Uncertainty threshold above which a region is considered
        "high-uncertainty" and given additional compute.
    """

    def __init__(
        self, latent_dim: int, threshold: float = 0.5
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.threshold = threshold

        # Small projection to convert raw uncertainty into attention-compatible
        # logits while preserving differentiability.
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            RMSNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        uncertainty: Optional[torch.Tensor] = None,
        ensemble_predictions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute uncertainty-based attention weights.

        At least one of *uncertainty* or *ensemble_predictions* must be
        provided.  If both are given, their contributions are summed.

        Parameters
        ----------
        uncertainty : Tensor or None
            Per-dimension uncertainty estimates, shape ``[B, D]``.
            When ``None`` the module falls back to ensemble disagreement.
        ensemble_predictions : Tensor or None
            Stacked predictions from an ensemble, shape ``[E, B, D]`` where
            *E* is the number of ensemble members.  When ``None`` the module
            falls back to the supplied *uncertainty* tensor.

        Returns
        -------
        attention_weights : Tensor
            Normalised attention weights, shape ``[B, D]``.
        raw_uncertainty : Tensor
            Raw (unnormalised) per-dimension uncertainty, shape ``[B, D]``.
        """
        raw_uncertainty: torch.Tensor

        if ensemble_predictions is not None:
            # Variance across ensemble members → epistemic uncertainty
            raw_uncertainty = ensemble_predictions.var(dim=0)  # [B, D]
            if uncertainty is not None:
                raw_uncertainty = raw_uncertainty + uncertainty
        elif uncertainty is not None:
            raw_uncertainty = uncertainty
        else:
            # No information available – return uniform attention
            device = self.uncertainty_proj[0].weight.device
            raw_uncertainty = torch.zeros(1, self.latent_dim, device=device)

        # Project through learned transformation
        projected = self.uncertainty_proj(raw_uncertainty)

        # Threshold mask: regions above threshold get boosted
        high_uncertainty_mask = (raw_uncertainty > self.threshold).float()
        boosted = projected * (1.0 + high_uncertainty_mask)

        # Softmax normalisation across the feature dimension
        attention_weights = F.softmax(boosted, dim=-1)

        return attention_weights, raw_uncertainty

    def get_focus_ratio(self, raw_uncertainty: torch.Tensor) -> float:
        """Return the fraction of dimensions classified as high-uncertainty.

        Parameters
        ----------
        raw_uncertainty : Tensor
            Raw per-dimension uncertainty, shape ``[B, D]`` or ``[D]``.

        Returns
        -------
        float
            Fraction of dimensions above the uncertainty threshold.
        """
        if raw_uncertainty.dim() == 1:
            return (raw_uncertainty > self.threshold).float().mean().item()
        return (raw_uncertainty > self.threshold).float().mean().item()


# ---------------------------------------------------------------------------
# TemporalEvolution
# ---------------------------------------------------------------------------


class TemporalEvolution(nn.Module):
    """GRU-based temporal evolution of the attention map.

    Takes the current attention map, prediction error, and a novelty signal
    and produces an updated attention map.  This allows compute allocation to
    evolve smoothly over time, responding to changing task demands rather
    than reacting instantaneously to each new observation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space / attention map.
    hidden_dim : int
        Hidden size of the GRU cell.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection: [attention_map | prediction_error | novelty] → hidden
        input_dim = latent_dim * 3  # attention + error + novelty
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            RMSNorm(hidden_dim),
        )

        # GRU cell for temporal updates
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Output projection: hidden → attention map
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            RMSNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

        # Initial hidden state (learnable)
        self.register_buffer(
            "initial_hidden",
            torch.zeros(hidden_dim),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        attention_map: torch.Tensor,
        prediction_error: Optional[torch.Tensor] = None,
        novelty: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evolve the attention map one time-step.

        Parameters
        ----------
        attention_map : Tensor
            Current attention map, shape ``[B, D]``.
        prediction_error : Tensor or None
            Per-dimension prediction error, shape ``[B, D]``.
            Defaults to zeros when not supplied.
        novelty : Tensor or None
            Per-dimension novelty signal, shape ``[B, D]``.
            Defaults to zeros when not supplied.
        hidden_state : Tensor or None
            Previous GRU hidden state, shape ``[B, H]``.
            Uses the learnable initial state when ``None``.

        Returns
        -------
        updated_map : Tensor
            Evolved attention map (softmax-normalised), shape ``[B, D]``.
        new_hidden : Tensor
            Updated GRU hidden state, shape ``[B, H]``.
        """
        batch_size = attention_map.size(0)
        device = attention_map.device

        # Default missing signals to zeros
        if prediction_error is None:
            prediction_error = torch.zeros_like(attention_map)
        if novelty is None:
            novelty = torch.zeros_like(attention_map)

        # Concatenate inputs
        gru_input = torch.cat([attention_map, prediction_error, novelty], dim=-1)
        gru_input = self.input_proj(gru_input)

        # Initialise hidden state if needed
        if hidden_state is None:
            hidden_state = self.initial_hidden.unsqueeze(0).expand(batch_size, -1)

        # GRU step
        new_hidden = self.gru(gru_input, hidden_state)

        # Project to attention map
        updated_map = self.output_proj(new_hidden)

        # Softmax normalisation so the map remains a valid distribution
        updated_map = F.softmax(updated_map, dim=-1)

        return updated_map, new_hidden

    def reset_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Return a fresh hidden state for a given batch size.

        Parameters
        ----------
        batch_size : int
            Number of parallel sequences.

        Returns
        -------
        Tensor
            Initial hidden state, shape ``[B, H]``.
        """
        return self.initial_hidden.unsqueeze(0).expand(batch_size, -1).clone()


# ---------------------------------------------------------------------------
# AttentionProbabilityMap (main module)
# ---------------------------------------------------------------------------


class AttentionProbabilityMap(nn.Module):
    """Attention-Driven Probability Map for dynamic compute allocation.

    Maintains and updates a spatial/feature attention map that determines how
    compute is distributed across the latent representation.  Three mechanisms
    drive allocation:

    1. **Confidence-weighted maps** – regions with low confidence (high
       prediction error) receive more compute.
    2. **Uncertainty focus** – high-risk / high-uncertainty regions are
       prioritised; low-information zones stay compressed.
    3. **Temporal evolution** – a GRU smoothly evolves the attention map over
       time, incorporating prediction error trends and novelty signals.

    Parameters
    ----------
    config : AttentionMapsConfig
        Configuration dataclass controlling all hyper-parameters.
    latent_dim : int
        Dimensionality of the latent representation.  Defaults to the value
        in *config*.
    """

    def __init__(
        self,
        config: AttentionMapsConfig,
        latent_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.num_heads = config.num_heads
        self.min_attention = config.min_attention

        # --- Sub-modules ------------------------------------------------- #
        self.confidence_tracker = ConfidenceTracker(
            latent_dim=latent_dim,
            decay=config.confidence_decay,
        )
        self.uncertainty_focus = UncertaintyFocus(
            latent_dim=latent_dim,
            threshold=config.uncertainty_threshold,
        )
        self.temporal_evolution = TemporalEvolution(
            latent_dim=latent_dim,
            hidden_dim=config.evolution_hidden_dim,
        )

        # --- Learnable attention map (uniform initialisation) ----------- #
        self.register_buffer(
            "attention_map",
            torch.ones(latent_dim) / latent_dim,
        )

        # --- Multi-head attention --------------------------------------- #
        # Projects the latent into Q, K, V for multi-head attention that
        # refines the attention weights based on content.
        self.q_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.out_proj = nn.Linear(latent_dim, latent_dim, bias=False)

        # Scaling factor for dot-product attention
        self.head_dim = latent_dim // self.num_heads
        self.scale = math.sqrt(self.head_dim)

        # --- GRU hidden state ------------------------------------------- #
        self.register_buffer(
            "_temporal_hidden",
            torch.zeros(config.evolution_hidden_dim),
        )
        self._batch_size: int = 1

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        latent: torch.Tensor,
        prediction_error: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        novelty: Optional[torch.Tensor] = None,
        ensemble_predictions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute attention-weighted latent with dynamic compute allocation.

        Parameters
        ----------
        latent : Tensor
            Input latent representation, shape ``[B, D]``.
        prediction_error : Tensor or None
            Per-dimension prediction error, shape ``[B, D]``.
        uncertainty : Tensor or None
            Per-dimension uncertainty, shape ``[B, D]``.
        novelty : Tensor or None
            Per-dimension novelty signal, shape ``[B, D]``.
        ensemble_predictions : Tensor or None
            Stacked ensemble predictions, shape ``[E, B, D]``.

        Returns
        -------
        weighted_latent : Tensor
            Attention-weighted latent, shape ``[B, D]``.
        info : dict
            Diagnostic information including attention maps, confidence,
            uncertainty weights, and compute allocation statistics.
        """
        batch_size = latent.size(0)
        device = latent.device
        self._batch_size = batch_size
        info: Dict[str, torch.Tensor] = {}

        # ---- 1. Update confidence from prediction error ---------------- #
        if prediction_error is not None:
            confidence = self.confidence_tracker.update(prediction_error)
        else:
            confidence = self.confidence_tracker.get_confidence()
        info["confidence"] = confidence  # [D]

        # ---- 2. Compute confidence-based attention weights ------------- #
        confidence_weights = self.confidence_tracker.get_attention_weights(
            min_attention=self.min_attention,
        )  # [D]
        info["confidence_attention"] = confidence_weights

        # ---- 3. Compute uncertainty-based attention weights ------------- #
        uncertainty_weights, raw_uncertainty = self.uncertainty_focus(
            uncertainty=uncertainty,
            ensemble_predictions=ensemble_predictions,
        )  # [B, D], [B, D]
        info["uncertainty_attention"] = uncertainty_weights
        info["raw_uncertainty"] = raw_uncertainty

        # ---- 4. Multi-head content attention --------------------------- #
        content_weights = self._multihead_attention(latent)  # [B, D]
        info["content_attention"] = content_weights

        # ---- 5. Combine attention sources ------------------------------ #
        # Expand confidence weights to [B, D]
        cw = confidence_weights.unsqueeze(0).expand(batch_size, -1)
        # Use current stored attention map expanded to [B, D]
        stored_map = self.attention_map.unsqueeze(0).expand(batch_size, -1)

        # Weighted combination (confidence + uncertainty + content + stored)
        combined = (
            0.25 * cw
            + 0.35 * uncertainty_weights
            + 0.20 * content_weights
            + 0.20 * stored_map
        )
        info["combined_raw"] = combined

        # ---- 6. Temporal evolution ------------------------------------- #
        evolved_map, new_hidden = self.temporal_evolution(
            attention_map=combined,
            prediction_error=prediction_error,
            novelty=novelty,
            hidden_state=self._temporal_hidden.unsqueeze(0).expand(batch_size, -1),
        )  # [B, D], [B, H]
        info["evolved_attention"] = evolved_map

        # Update stored hidden state (detached to avoid graph retention)
        self._temporal_hidden = new_hidden.detach().mean(dim=0)

        # ---- 7. Floor and normalise ------------------------------------ #
        final_map = evolved_map.clamp(min=self.min_attention)
        final_map = final_map / final_map.sum(dim=-1, keepdim=True)

        # Update the stored attention map (batch-averaged)
        self.attention_map = final_map.detach().mean(dim=0)
        info["final_attention"] = final_map

        # ---- 8. Apply attention to latent ------------------------------ #
        weighted_latent = latent * final_map

        # ---- 9. Compute allocation statistics -------------------------- #
        info["compute_allocation"] = final_map
        info["focus_ratio"] = torch.tensor(
            self.uncertainty_focus.get_focus_ratio(raw_uncertainty)
        )

        return weighted_latent, info

    # ------------------------------------------------------------------ #
    # Multi-head attention helper                                          #
    # ------------------------------------------------------------------ #

    def _multihead_attention(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention to produce content-based weights.

        Parameters
        ----------
        latent : Tensor
            Input latent, shape ``[B, D]``.

        Returns
        -------
        Tensor
            Content-based attention weights, shape ``[B, D]``.
        """
        batch_size = latent.size(0)

        # Degenerate case: with a single token the self-attention dot
        # product reduces to a scalar per head that always gets weight 1.0
        # after softmax, making the whole attention block a no-op.  Return
        # uniform attention so the content signal does not distort the
        # combined map.
        if batch_size == 1:
            return torch.ones(1, self.latent_dim, device=latent.device) / self.latent_dim

        q = self.q_proj(latent)  # [B, D]
        k = self.k_proj(latent)  # [B, D]
        v = self.v_proj(latent)  # [B, D]

        # Reshape for multi-head: [B, H, L, head_dim]
        # Here L=1 since we have a single latent vector per sample;
        # we treat the *feature* dimension as the sequence for attention.
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        # Dot-product attention across feature groups within each head
        # q, k, v: [B, H, head_dim] → treat as [B, H, 1, head_dim]
        q = q.unsqueeze(2)  # [B, H, 1, head_dim]
        k = k.unsqueeze(2)  # [B, H, 1, head_dim]

        # Scaled dot-product: [B, H, 1, 1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Apply to v: [B, H, 1, head_dim] → [B, H, head_dim]
        attended = torch.matmul(attn_weights, v.unsqueeze(2)).squeeze(2)

        # Reshape back: [B, D]
        attended = attended.reshape(batch_size, self.latent_dim)
        attended = self.out_proj(attended)

        # Convert to probability distribution over features
        weights = F.softmax(attended, dim=-1)
        return weights

    # ------------------------------------------------------------------ #
    # Public helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_attention_map(self) -> torch.Tensor:
        """Return the current attention map.

        Returns
        -------
        Tensor
            Per-dimension attention weights, shape ``[D]``.
        """
        return self.attention_map.clone()

    def get_compute_allocation(self) -> Dict[str, float]:
        """Return per-region compute allocation percentages.

        Splits the latent dimensions into equal-sized "regions" and reports
        the fraction of total compute allocated to each region.

        Returns
        -------
        dict
            Mapping ``"region_{i}"`` → percentage of compute allocated.
        """
        allocation = self.attention_map.detach().cpu()
        total = allocation.sum().item()
        if total == 0:
            total = 1.0  # avoid division by zero

        # Divide into 8 regions for reporting
        num_regions = 8
        region_size = self.latent_dim // num_regions
        result: Dict[str, float] = {}
        for i in range(num_regions):
            start = i * region_size
            end = start + region_size if i < num_regions - 1 else self.latent_dim
            region_pct = allocation[start:end].sum().item() / total * 100.0
            result[f"region_{i}"] = round(region_pct, 2)
        return result

    def reset(self) -> None:
        """Reset attention map to uniform and clear all running state."""
        self.attention_map = torch.ones(self.latent_dim, device=self.attention_map.device) / self.latent_dim
        self.confidence_tracker.reset()
        self._temporal_hidden = torch.zeros(
            self.config.evolution_hidden_dim,
            device=self._temporal_hidden.device,
        )
