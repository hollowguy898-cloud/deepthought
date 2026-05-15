"""
Sparse Router module for Deep Thought.

Implements top-k expert selection with load balancing, entropy regularization,
and noise-augmented gating for stable sparse activation.

CRITICAL FIX: Routing Collapse Prevention
  - DIFFERENTIABLE sparse gating: Gate values retain gradient flow to router
    weights during training. The old code detached gates, which meant the router
    never received gradients from the main loss — only from a weak EMA-based
    auxiliary loss. This caused routing collapse.
  - Switch Transformer load balancing loss: Operates on the current batch's
    routing probabilities directly, not on a stale EMA buffer. Much stronger
    signal to prevent expert underuse.
  - Expert utilization regularization: Penalizes any expert receiving less
    than 1/N of the routing probability mass, ensuring all experts stay alive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

from deep_thought.config import RouterConfig


class NoisyTopKRouter(nn.Module):
    """
    Router with noisy top-k gating for load balancing.

    Uses additive noise during training to encourage expert diversity
    and prevent routing collapse.

    Key design: Gate values are ALWAYS differentiable during training.
    Gradients flow through the selected gate probabilities back to the
    router weights, so the router learns from the main objective.
    """

    def __init__(self, config: RouterConfig, latent_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.active_experts = config.active_experts
        self.hidden_dim = config.hidden_dim
        self.noise_epsilon = config.noise_epsilon

        # Router network - input is concatenated h_t, x_t, m_t (each latent_dim)
        self._latent_dim = latent_dim
        if latent_dim is not None:
            self.router = nn.Sequential(
                nn.Linear(latent_dim * 3, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.num_experts),
            )
        else:
            # Will be initialized lazily
            self.router = None

        # Load balancing loss tracking (kept for monitoring only)
        self.register_buffer("expert_usage", torch.zeros(self.num_experts))

    def _ensure_router(self, h_t: torch.Tensor):
        """Lazily initialize router if needed."""
        if self.router is None:
            latent_dim = h_t.size(-1)
            self._latent_dim = latent_dim
            self.router = nn.Sequential(
                nn.Linear(latent_dim * 3, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.num_experts),
            )
            # Move to same device as input
            self.router = self.router.to(h_t.device)

    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        training: bool = True,
        detach_gates: Optional[bool] = None,  # Deprecated; kept for API compat
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route to top-k experts with differentiable gating.

        Gate values retain gradients during training so the router learns
        from the main loss. The detach_gates parameter is deprecated —
        during training, gates are always differentiable; during eval,
        gradients don't matter.

        Args:
            h_t: Hidden state
            x_t: Encoded observation
            m_t: Memory read
            training: Whether in training mode
            detach_gates: DEPRECATED — ignored. Kept for backward API compat.

        Returns:
            gates: Gate values for selected experts (differentiable during training)
            selected_indices: Indices of selected experts
            info: Routing information dict with keys:
                - logits: raw router logits
                - probs: full routing probabilities (batch, num_experts)
                - entropy: mean routing entropy
                - expert_usage: expert usage statistics
                - selected_indices: same as returned value (for loss computation)
        """
        # Concatenate inputs
        combined = torch.cat([h_t, x_t, m_t], dim=-1)

        # Ensure router is initialized
        self._ensure_router(h_t)

        # Compute router logits
        logits = self.router(combined)

        # Add noise during training for exploration
        if training:
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits = logits + noise

        # Softmax for full routing probabilities
        probs = F.softmax(logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            probs,
            self.active_experts,
            dim=-1
        )

        # Differentiable gating: use the raw probs for selected experts.
        # This is the key fix — we do NOT detach. Gradients flow through
        # the gate values back to the router weights via the softmax.
        # Normalize selected gates so they sum to 1.
        gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Update expert usage statistics for monitoring
        if training:
            with torch.no_grad():
                # Use full probs to track usage (more informative than hard assignments)
                self.expert_usage = 0.99 * self.expert_usage + 0.01 * probs.mean(dim=0)

        # Compute routing entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        info = {
            "logits": logits,
            "probs": probs,
            "entropy": entropy,
            "expert_usage": self.expert_usage.clone(),
            "selected_indices": top_k_indices,  # Include for loss computation
            "gates": gates,  # Include for loss computation
        }

        return gates, top_k_indices, info

    def load_balance_loss(
        self,
        probs: torch.Tensor,
        selected_indices: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Switch Transformer load balancing loss.

        Computes auxiliary loss that encourages balanced expert utilization
        by penalizing the product of (fraction of tokens dispatched to expert i)
        and (average routing probability for expert i). When experts are
        perfectly balanced, this product is minimized.

        This operates on the CURRENT BATCH, not a stale EMA buffer, so it
        provides immediate and strong gradient signal.

        Args:
            probs: (batch, num_experts) - full routing probabilities
            selected_indices: (batch, k) - selected expert indices
            gates: (batch, k) - gate values for selected experts

        Returns:
            auxiliary_loss: scalar that encourages balanced expert usage
        """
        num_experts = probs.size(-1)

        # f_i: fraction of tokens dispatched to each expert
        expert_mask = torch.zeros_like(probs)
        for k_idx in range(selected_indices.size(1)):
            expert_mask.scatter_(1, selected_indices[:, k_idx:k_idx+1], 1.0)
        f = expert_mask.mean(dim=0)  # (num_experts,)

        # P_i: mean routing probability per expert
        P = probs.mean(dim=0)  # (num_experts,)

        # Auxiliary loss: N * sum(f_i * P_i)
        # Minimized when f_i = P_i = 1/N for all i (uniform distribution)
        aux_loss = num_experts * (f * P).sum()

        return self.config.load_balance_loss_coef * aux_loss

    def expert_utilization_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Penalize experts that receive near-zero routing probability.

        Encourages all experts to be used at least 1/N of the probability
        mass. This prevents "dead" experts that the router never selects,
        which was the primary symptom of routing collapse.

        Args:
            probs: (batch, num_experts) - full routing probabilities

        Returns:
            utilization_loss: scalar penalty for underused experts
        """
        P = probs.mean(dim=0)  # (num_experts,)
        target = 1.0 / self.num_experts
        underuse = F.relu(target - P)  # Only penalize underuse, not overuse
        return underuse.sum() * self.config.load_balance_loss_coef * 10.0

    def entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularization loss.

        Keeps routing entropy in a healthy range. Low entropy means
        the router is collapsed (selecting same experts repeatedly).
        High entropy means routing is near-uniform (not selective enough).

        Args:
            entropy: scalar mean routing entropy

        Returns:
            entropy_loss: penalty if entropy outside healthy range
        """
        if entropy < self.config.min_entropy:
            return self.config.entropy_coef * (self.config.min_entropy - entropy)
        elif entropy > self.config.max_entropy:
            return self.config.entropy_coef * (entropy - self.config.max_entropy)
        return torch.tensor(0.0, device=entropy.device)


class AdaptiveRouter(nn.Module):
    """
    Adaptive router that adjusts based on context and prediction error.

    Modifies routing distribution based on:
    - Context embedding
    - Prediction error signals
    - Meta-router controller

    Key design: Gate values are always differentiable during training,
    allowing the adapter to learn from the main objective.
    """

    def __init__(self, config: RouterConfig, context_dim: int = 256, latent_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.context_dim = context_dim

        # Base router
        self.base_router = NoisyTopKRouter(config, latent_dim=latent_dim)

        # Context encoder - will be lazily initialized if latent_dim not known
        self._latent_dim = latent_dim
        if latent_dim is not None:
            self.context_encoder = nn.Linear(latent_dim * 3, self.context_dim)
        else:
            self.context_encoder = None

        # Adapter - will be lazily initialized
        self.adapter = None
        self._adapter_input_dim = None

    def _ensure_modules(self, h_t: torch.Tensor, x_t: torch.Tensor, m_t: torch.Tensor):
        """Lazily initialize context encoder and adapter."""
        combined = torch.cat([h_t, x_t, m_t], dim=-1)

        if self.context_encoder is None:
            input_dim = combined.size(-1)
            self.context_encoder = nn.Linear(input_dim, self.context_dim).to(combined.device)

        if self.adapter is None:
            # adapter_input = context + prediction_error (1 dim)
            self._adapter_input_dim = self.context_dim + 1
            self.adapter = nn.Sequential(
                nn.Linear(self._adapter_input_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.hidden_dim, self.config.num_experts),
            ).to(combined.device)

    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prediction_error: Optional[torch.Tensor] = None,
        training: bool = True,
        detach_gates: Optional[bool] = None,  # Deprecated; kept for API compat
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route with adaptive modification.

        Gate values are always differentiable during training. The adapter
        output flows through softmax and top-k with gradients intact.

        Args:
            h_t: Hidden state
            x_t: Encoded observation
            m_t: Memory read
            context: Context embedding
            prediction_error: Prediction error signal
            training: Whether in training mode
            detach_gates: DEPRECATED — ignored. Kept for backward API compat.

        Returns:
            gates: Gate values for selected experts (differentiable during training)
            selected_indices: Indices of selected experts
            info: Routing information dict
        """
        # Base routing — always differentiable during training
        gates, indices, base_info = self.base_router(
            h_t, x_t, m_t, training
        )

        # Ensure adapter modules are initialized
        self._ensure_modules(h_t, x_t, m_t)

        # Encode context if not provided
        if context is None:
            combined = torch.cat([h_t, x_t, m_t], dim=-1)
            context = self.context_encoder(combined)

        # Adaptation based on context and error
        if prediction_error is not None:
            # Normalize error
            if prediction_error.dim() == 0:
                # Scalar tensor
                error_norm = prediction_error.unsqueeze(0).unsqueeze(0)
                if error_norm.size(0) != context.size(0):
                    error_norm = error_norm.expand(context.size(0), 1)
            else:
                error_norm = (prediction_error - prediction_error.mean()) / \
                            (prediction_error.std() + 1e-8)
                if error_norm.dim() == 1:
                    error_norm = error_norm.unsqueeze(-1)
                elif error_norm.dim() == 0:
                    error_norm = error_norm.unsqueeze(0).unsqueeze(0)
                    error_norm = error_norm.expand(context.size(0), 1)
            adapter_input = torch.cat([context, error_norm], dim=-1)
        else:
            # Use zeros for prediction_error placeholder
            adapter_input = torch.cat([context, torch.zeros(context.size(0), 1, device=context.device)], dim=-1)

        # Compute adaptation — gradient flows through adapter during training
        adaptation = self.adapter(adapter_input)

        # Modify routing logits and re-select
        modified_logits = base_info["logits"] + adaptation
        modified_probs = F.softmax(modified_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(
            modified_probs,
            self.config.active_experts,
            dim=-1
        )

        # Differentiable gating: use raw probs, normalize to sum to 1
        gates = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Update info (create new dict to avoid mutating base_info)
        info = {**base_info}
        info["modified_logits"] = modified_logits
        info["adaptation"] = adaptation
        info["probs"] = modified_probs  # Use modified probs for loss computation
        info["selected_indices"] = top_k_indices  # Use modified indices for loss computation
        info["gates"] = gates  # Include for loss computation

        # Update expert usage statistics for monitoring
        if training:
            with torch.no_grad():
                self.base_router.expert_usage = 0.99 * self.base_router.expert_usage + \
                    0.01 * modified_probs.mean(dim=0)
            info["expert_usage"] = self.base_router.expert_usage.clone()

        return gates, top_k_indices, info


class SparseRouter(nn.Module):
    """
    Main sparse router for Deep Thought.

    Combines base routing with adaptive modification based on
    context and error signals.

    Key design: Differentiable sparse gating ensures the router
    receives gradient signal from the main objective, preventing
    routing collapse to a few experts.
    """

    def __init__(self, config: RouterConfig, use_adaptive: bool = True,
                 latent_dim: Optional[int] = None, context_dim: int = 256):
        super().__init__()
        self.config = config
        self.use_adaptive = use_adaptive

        if use_adaptive:
            self.router = AdaptiveRouter(config, context_dim=context_dim, latent_dim=latent_dim)
        else:
            self.router = NoisyTopKRouter(config, latent_dim=latent_dim)

    def forward(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        prediction_error: Optional[torch.Tensor] = None,
        training: bool = True,
        detach_gates: Optional[bool] = None,  # Deprecated; kept for API compat
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route to experts with differentiable gating.

        The detach_gates parameter is deprecated — gates are always
        differentiable during training. Kept for backward API compatibility.
        """
        if self.use_adaptive:
            return self.router(
                h_t, x_t, m_t, context, prediction_error, training,
                detach_gates=detach_gates
            )
        else:
            return self.router(h_t, x_t, m_t, training, detach_gates=detach_gates)

    def compute_losses(self, info: dict, gates: Optional[torch.Tensor] = None) -> dict:
        """
        Compute routing losses using the current batch's routing data.

        Args:
            info: Routing information dict from forward() containing
                'probs', 'selected_indices', 'entropy', and optionally 'gates'
            gates: Optional gate values (if not in info dict)

        Returns:
            losses: Dict with 'load_balance', 'expert_utilization', and 'entropy' losses
        """
        losses = {}

        # Get the base router for loss computation
        if self.use_adaptive:
            base_router = self.router.base_router
        else:
            base_router = self.router

        # Extract routing data for loss computation
        probs = info.get("probs", None)
        selected_indices = info.get("selected_indices", None)
        if gates is None:
            gates = info.get("gates", None)

        # Switch Transformer load balance loss
        if probs is not None and selected_indices is not None and gates is not None:
            losses["load_balance"] = base_router.load_balance_loss(
                probs, selected_indices, gates
            )

        # Expert utilization loss
        if probs is not None:
            losses["expert_utilization"] = base_router.expert_utilization_loss(probs)

        # Entropy loss
        if "entropy" in info:
            losses["entropy"] = base_router.entropy_loss(info["entropy"])

        return losses

    def get_expert_usage(self) -> torch.Tensor:
        """Get current expert usage statistics."""
        if self.use_adaptive:
            return self.router.base_router.expert_usage.clone()
        else:
            return self.router.expert_usage.clone()
