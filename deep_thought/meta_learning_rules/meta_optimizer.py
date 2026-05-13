"""Meta-Learning of Learning Rules for Deep Thought RL Framework.

Instead of hand-designed learning rate schedules, pruning rules, and routing
logic, this module EVOLVES the learning process itself.  Experts don't just
learn tasks — they learn *how to learn better*.

The core idea is that a small LSTM-based network observes gradient statistics
and produces per-parameter-group hyperparameters (learning rate, momentum,
weight decay).  This replaces fixed schedules with adaptive, learned
optimisation rules.

Classes
-------
UpdateRuleNetwork
    LSTM that maps gradient statistics to per-parameter-group hyperparameters.
GradientStatistics
    Tracks running statistics (mean, variance, norm) of gradients via EMA.
MetaLearningRule
    Applies a learned update rule to a single parameter group.
MetaOptimizer
    Top-level ``nn.Module`` that orchestrates all learned rules and provides
    a standard ``zero_grad`` / ``step`` / ``meta_loss`` interface.

References
----------
* Andrychowicz et al., "Learning to Learn by Gradient Descent by Gradient
  Descent", NeurIPS 2016.
* Wichrowska et al., "Learned Optimizers that Scale and Generalize", ICML 2017.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_thought.config import MetaLearningRulesConfig


# ---------------------------------------------------------------------------
# Gradient Statistics Tracker
# ---------------------------------------------------------------------------


class GradientStatistics:
    """Track running statistics of gradients for a single parameter group.

    Uses exponential moving averages (EMA) to maintain cheap, differentiable
    estimates of the gradient mean, variance, and L2 norm.  All internal
    buffers are stored as regular Python floats / tensors (not registered
    buffers) because they are *not* part of the model's learned parameters.

    Parameters
    ----------
    decay : float
        EMA decay factor (closer to 1.0 → smoother / longer memory).
        Corresponds to ``statistics_decay`` in :class:`MetaLearningRulesConfig`.
    device : torch.device | str
        Device on which to allocate the running-stat tensors.

    Attributes
    ----------
    grad_mean : torch.Tensor
        Running EMA of the (scalar-mean) gradient value.
    grad_var : torch.Tensor
        Running EMA of the (scalar) gradient variance.
    grad_norm : torch.Tensor
        Running EMA of the gradient L2 norm.
    step_count : torch.Tensor
        Monotonically increasing step counter (as a tensor so it can be
        concatenated into the feature vector fed to the LSTM).
    """

    def __init__(self, decay: float = 0.99, device: torch.device | str = "cpu") -> None:
        self.decay = decay
        self.device = torch.device(device)
        self._initialized = False

        # Lazy-initialised tensors (created on first ``update`` call)
        self.grad_mean: torch.Tensor = torch.zeros(1, device=self.device)
        self.grad_var: torch.Tensor = torch.ones(1, device=self.device)
        self.grad_norm: torch.Tensor = torch.zeros(1, device=self.device)
        self.step_count: torch.Tensor = torch.zeros(1, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, grad: torch.Tensor) -> None:
        """Ingest a new gradient tensor and refresh running statistics.

        Parameters
        ----------
        grad : torch.Tensor
            The gradient tensor of a parameter group.
        """
        with torch.no_grad():
            scalar_mean = grad.mean().item()
            scalar_var = grad.var().item() if grad.numel() > 1 else 0.0
            scalar_norm = grad.norm(2).item()

            mean_t = torch.tensor([scalar_mean], device=self.device)
            var_t = torch.tensor([scalar_var], device=self.device)
            norm_t = torch.tensor([scalar_norm], device=self.device)

            if not self._initialized:
                self.grad_mean = mean_t
                self.grad_var = var_t
                self.grad_norm = norm_t
                self._initialized = True
            else:
                d = self.decay
                self.grad_mean = d * self.grad_mean + (1.0 - d) * mean_t
                self.grad_var = d * self.grad_var + (1.0 - d) * var_t
                self.grad_norm = d * self.grad_norm + (1.0 - d) * norm_t

            self.step_count = self.step_count + 1.0

    def get_features(self) -> torch.Tensor:
        """Return a 4-dimensional feature vector for the LSTM.

        The features are:
        ``[grad_norm, grad_mean, grad_var, log(1 + step_count)]``

        The step count is log-transformed so that its magnitude stays bounded
        even after millions of updates.

        Returns
        -------
        torch.Tensor
            Shape ``(4,)`` on ``self.device``.
        """
        log_step = torch.log1p(self.step_count)
        return torch.cat([
            self.grad_norm,
            self.grad_mean,
            self.grad_var,
            log_step,
        ]).clamp(-1e6, 1e6)


# ---------------------------------------------------------------------------
# Update Rule Network (LSTM)
# ---------------------------------------------------------------------------


class UpdateRuleNetwork(nn.Module):
    """LSTM that maps gradient statistics to optimiser hyperparameters.

    For each parameter group, the network receives a 4-dimensional feature
    vector (see :meth:`GradientStatistics.get_features`) and outputs three
    scalars that are subsequently squashed into valid ranges:

    * **learning rate**  ∈ ``[min_lr, max_lr]``
    * **momentum**       ∈ ``[0, 1]``
    * **weight decay**   ∈ ``[0, max_weight_decay]``

    Parameters
    ----------
    input_dim : int
        Dimensionality of the per-step input (default 4 — one for each
        gradient statistic feature).
    hidden_dim : int
        Hidden size of the LSTM layers.
    num_lstm_layers : int
        Number of stacked LSTM layers.
    max_learning_rate : float
        Upper bound on the learning rate output.
    min_learning_rate : float
        Lower bound on the learning rate output.
    max_weight_decay : float
        Upper bound on the weight-decay output.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        max_learning_rate: float = 0.1,
        min_learning_rate: float = 1e-6,
        max_weight_decay: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_weight_decay = max_weight_decay

        # Core recurrent network
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Output projection: hidden → raw (unconstrained) hyperparameters
        self.output_proj = nn.Linear(hidden_dim, 3)

        # Initialise output layer to produce small, reasonable defaults
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Hidden state (one per parameter-group, managed externally)
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_hidden(self, device: torch.device | str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a fresh zero hidden state on the given device.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(h_0, c_0)`` each of shape ``(num_lstm_layers, 1, hidden_dim)``.
        """
        device = torch.device(device)
        h_0 = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_lstm_layers, 1, self.hidden_dim, device=device)
        return (h_0, c_0)

    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Produce hyperparameters from gradient-statistic features.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(1, 1, input_dim)`` — batch 1, seq-len 1, feature dim.
        hidden : tuple[torch.Tensor, torch.Tensor] | None
            Previous LSTM hidden state.  ``None`` → initialise to zeros.

        Returns
        -------
        hyperparams : torch.Tensor
            Shape ``(1, 3)`` — ``[lr, momentum, weight_decay]`` already
            clamped to their valid ranges.
        new_hidden : tuple[torch.Tensor, torch.Tensor]
            Updated LSTM hidden state for the next call.
        """
        if hidden is None:
            hidden = self.init_hidden(device=features.device)

        lstm_out, new_hidden = self.lstm(features, hidden)
        # lstm_out: (1, 1, hidden_dim) → squeeze seq dim
        last_hidden = lstm_out.squeeze(1)  # (1, hidden_dim)
        raw = self.output_proj(last_hidden)  # (1, 3)

        lr_raw = raw[:, 0]
        momentum_raw = raw[:, 1]
        wd_raw = raw[:, 2]

        # Squash into valid ranges
        lr = self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * torch.sigmoid(lr_raw)
        momentum = torch.sigmoid(momentum_raw)  # [0, 1]
        weight_decay = self.max_weight_decay * torch.sigmoid(wd_raw)  # [0, max_wd]

        hyperparams = torch.stack([lr, momentum, weight_decay], dim=-1)  # (1, 3)
        return hyperparams, new_hidden


# ---------------------------------------------------------------------------
# Meta Learning Rule (per parameter group)
# ---------------------------------------------------------------------------


class MetaLearningRule:
    """A learned update rule that replaces standard SGD/Adam for one group.

    For each parameter group it:

    1. Collects gradient statistics via :class:`GradientStatistics`.
    2. Feeds them through an :class:`UpdateRuleNetwork`.
    3. Produces adaptive hyperparameters (lr, momentum, weight_decay).
    4. Applies the update::

           param = param - lr * (momentum * prev_update
                                 + (1 - momentum) * grad
                                 + weight_decay * param)

    Parameters
    ----------
    rule_network : UpdateRuleNetwork
        The shared (or per-group) LSTM rule network.
    stats : GradientStatistics
        Gradient statistic tracker for this group.
    config : MetaLearningRulesConfig
        Hyperparameter bounds and other settings.
    device : torch.device | str
        Target device.
    """

    def __init__(
        self,
        rule_network: UpdateRuleNetwork,
        stats: GradientStatistics,
        config: MetaLearningRulesConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.rule_network = rule_network
        self.stats = stats
        self.config = config
        self.device = torch.device(device)

        # LSTM hidden state for this group
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Previous update tensor (for momentum).  Lazy-initialised.
        self._prev_updates: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, params: List[torch.Tensor]) -> float:
        """Compute and apply the learned update to all parameters in the group.

        Parameters
        ----------
        params : list[torch.Tensor]
            The parameter tensors belonging to this group.

        Returns
        -------
        float
            Total L2 norm of the applied updates (diagnostic).
        """
        # 1. Aggregate gradient statistics across the group
        all_grads = []
        for p in params:
            if p.grad is not None:
                all_grads.append(p.grad)

        if len(all_grads) == 0:
            return 0.0

        # Concatenate into a flat vector for summary statistics
        flat_grad = torch.cat([g.reshape(-1) for g in all_grads])
        self.stats.update(flat_grad)

        # 2. Build LSTM input features
        features = self.stats.get_features().unsqueeze(0).unsqueeze(0)  # (1, 1, 4)

        # 3. Run the LSTM
        hyperparams, new_hidden = self.rule_network(features, self._hidden)
        self._hidden = new_hidden

        lr = hyperparams[0, 0]
        momentum = hyperparams[0, 1]
        weight_decay = hyperparams[0, 2]

        # 4. Apply the update to each parameter
        total_update_norm = 0.0
        eps = 1e-12

        for p in params:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            # Previous update (momentum buffer)
            if pid not in self._prev_updates:
                self._prev_updates[pid] = torch.zeros_like(p.data)

            prev_update = self._prev_updates[pid]

            # Learned update rule
            #   update = momentum * prev_update + (1 - momentum) * grad + weight_decay * p
            update = (
                momentum * prev_update
                + (1.0 - momentum) * grad
                + weight_decay * p.data
            )

            # Store for next step
            self._prev_updates[pid] = update.detach()

            # Apply the update
            p.data.sub_(lr * update)

            total_update_norm += update.norm(2).item() ** 2

        total_update_norm = math.sqrt(total_update_norm + eps)
        return total_update_norm


# ---------------------------------------------------------------------------
# Meta Optimizer (main class)
# ---------------------------------------------------------------------------


class MetaOptimizer(nn.Module):
    """Meta-learning system that learns to generate optimiser updates.

    Instead of using fixed learning-rate schedules, a small LSTM-based network
    observes gradient statistics and produces per-parameter-group learning
    rates, momentum values, and weight-decay coefficients.

    This class is a drop-in replacement for standard PyTorch optimisers with
    an extended interface for meta-learning:

    * :meth:`step` — perform one optimisation step using the learned rules.
    * :meth:`meta_loss` — compute a meta-loss on validation data to update
      the *rule network* itself via a second-level optimiser.

    Parameters
    ----------
    config : MetaLearningRulesConfig
        Configuration dataclass.  If ``None`` is passed the defaults are used.
    param_groups : list[dict]
        Parameter groups (same format as :class:`torch.optim.Optimizer`).
        Each dict must contain a ``"params"`` key with an iterable of tensors.
        If empty, a single default group is created.

    Example
    -------
    >>> from deep_thought.config import MetaLearningRulesConfig
    >>> cfg = MetaLearningRulesConfig()
    >>> model = torch.nn.Linear(10, 5)
    >>> meta_opt = MetaOptimizer(cfg, [{"params": model.parameters()}])
    >>> loss = model(torch.randn(2, 10)).sum()
    >>> loss.backward()
    >>> update_norm = meta_opt.step(loss)
    >>> meta_loss = meta_opt.meta_loss(validation_loss)
    """

    def __init__(
        self,
        config: Optional[MetaLearningRulesConfig] = None,
        param_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()

        self.config = config or MetaLearningRulesConfig()
        self.device = torch.device("cpu")

        # Normalise param_groups
        if param_groups is None or len(param_groups) == 0:
            param_groups = [{"params": []}]

        self.param_groups: List[Dict[str, Any]] = []
        self._group_params: List[List[torch.Tensor]] = []  # resolved tensors

        for group in param_groups:
            params = list(group.get("params", []))
            self._group_params.append(params)
            self.param_groups.append({k: v for k, v in group.items() if k != "params"})

        # Determine device from the first available parameter
        for params in self._group_params:
            for p in params:
                if p is not None:
                    self.device = p.device
                    break
            if self.device != torch.device("cpu"):
                break

        # Create the shared UpdateRuleNetwork
        self.rule_network = UpdateRuleNetwork(
            input_dim=4,
            hidden_dim=self.config.hidden_dim,
            num_lstm_layers=self.config.num_lstm_layers,
            max_learning_rate=self.config.max_learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            max_weight_decay=self.config.max_weight_decay,
        ).to(self.device)

        # Per-group gradient statistics + learning rules
        self.gradient_stats: List[GradientStatistics] = []
        self.learning_rules: List[MetaLearningRule] = []

        for _ in self._group_params:
            stats = GradientStatistics(
                decay=self.config.statistics_decay,
                device=self.device,
            )
            rule = MetaLearningRule(
                rule_network=self.rule_network,
                stats=stats,
                config=self.config,
                device=self.device,
            )
            self.gradient_stats.append(stats)
            self.learning_rules.append(rule)

        # Meta-optimiser for updating the rule network itself
        self.meta_optimizer = torch.optim.Adam(
            self.rule_network.parameters(),
            lr=self.config.meta_lr,
        )

        # Register a buffer for the step counter (used in diagnostics)
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # Standard optimiser interface
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero all parameter gradients.

        Parameters
        ----------
        set_to_none : bool
            If True (default) set gradient tensors to ``None`` instead of
            filling with zeros — this is the modern PyTorch best practice for
            memory efficiency.
        """
        for params in self._group_params:
            for p in params:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, loss: Optional[torch.Tensor] = None) -> float:
        """Perform one optimisation step using the learned rules.

        The method iterates over all parameter groups, applies the
        :class:`MetaLearningRule` for each group, and returns the total
        update norm for diagnostic purposes.

        Parameters
        ----------
        loss : torch.Tensor | None
            The current loss tensor.  Kept for API compatibility; the
            actual gradient information comes from ``param.grad``.

        Returns
        -------
        float
            Total L2 norm of the applied updates across all parameter groups.
        """
        total_norm = 0.0

        for group_idx, (params, rule) in enumerate(
            zip(self._group_params, self.learning_rules)
        ):
            group_update_norm = rule.apply(params)
            total_norm += group_update_norm ** 2

        total_norm = math.sqrt(total_norm + 1e-12)
        self._global_step.add_(1)

        return total_norm

    # ------------------------------------------------------------------
    # Meta-learning interface
    # ------------------------------------------------------------------

    def meta_loss(self, validation_loss: torch.Tensor) -> torch.Tensor:
        """Compute a meta-loss that measures how well the learned rules perform.

        The meta-loss is:

        .. math::

            \\mathcal{L}_{\\text{meta}} = \\mathcal{L}_{\\text{val}}
                + \\lambda \\cdot R

        where :math:`R` is a regularisation term that penalises extreme
        hyperparameters (very high learning rates or weight decay values).

        Parameters
        ----------
        validation_loss : torch.Tensor
            Scalar tensor with the validation loss obtained *after* the
            latest ``step`` call.

        Returns
        -------
        torch.Tensor
            Scalar meta-loss (differentiable w.r.t. rule-network parameters).
        """
        # Gather the hyperparameters produced by the rule network in the
        # most recent forward pass.  We accumulate them for regularisation.
        lr_values: List[torch.Tensor] = []
        wd_values: List[torch.Tensor] = []

        for stats in self.gradient_stats:
            features = stats.get_features().unsqueeze(0).unsqueeze(0)  # (1,1,4)
            hyperparams, _ = self.rule_network(features)
            lr_values.append(hyperparams[0, 0])
            wd_values.append(hyperparams[0, 2])

        lr_stack = torch.stack(lr_values) if lr_values else torch.zeros(1, device=self.device)
        wd_stack = torch.stack(wd_values) if wd_values else torch.zeros(1, device=self.device)

        # Regularisation: penalise learning rates that are close to the
        # maximum (encourages staying in a moderate range)
        lr_reg = F.relu(lr_stack - 0.5 * self.config.max_learning_rate).mean()
        wd_reg = F.relu(wd_stack - 0.5 * self.config.max_weight_decay).mean()

        regularisation = lr_reg + wd_reg

        meta_loss_val = validation_loss + self.config.regularization_coef * regularisation
        return meta_loss_val

    def meta_step(self, validation_loss: torch.Tensor) -> float:
        """Perform one meta-optimisation step on the rule network.

        This is a convenience method that calls :meth:`meta_loss` and then
        updates the rule-network parameters via the internal Adam optimiser.

        Parameters
        ----------
        validation_loss : torch.Tensor
            Scalar validation loss.

        Returns
        -------
        float
            The scalar meta-loss value (before the meta-optimiser step).
        """
        self.meta_optimizer.zero_grad()
        m_loss = self.meta_loss(validation_loss)
        m_loss.backward()
        # Clip gradients to prevent destabilising the rule network
        torch.nn.utils.clip_grad_norm_(self.rule_network.parameters(), max_norm=1.0)
        self.meta_optimizer.step()
        return m_loss.item()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the meta-optimiser.

        Returns
        -------
        dict
            Keys:
            * ``global_step`` — total number of ``step`` calls.
            * ``num_param_groups`` — number of parameter groups.
            * ``mean_lr`` / ``std_lr`` — mean / std of current learning rates.
            * ``mean_momentum`` / ``std_momentum`` — same for momentum.
            * ``mean_weight_decay`` / ``std_weight_decay`` — same for weight decay.
            * ``gradient_norms`` — list of per-group gradient norms.
            * ``gradient_means`` — list of per-group gradient means.
            * ``gradient_vars`` — list of per-group gradient variances.
        """
        lrs: List[float] = []
        momenta: List[float] = []
        wds: List[float] = []
        grad_norms: List[float] = []
        grad_means: List[float] = []
        grad_vars: List[float] = []

        with torch.no_grad():
            for stats in self.gradient_stats:
                features = stats.get_features().unsqueeze(0).unsqueeze(0)
                hyperparams, _ = self.rule_network(features)
                lrs.append(hyperparams[0, 0].item())
                momenta.append(hyperparams[0, 1].item())
                wds.append(hyperparams[0, 2].item())

                grad_norms.append(stats.grad_norm.item())
                grad_means.append(stats.grad_mean.item())
                grad_vars.append(stats.grad_var.item())

        n = len(lrs) if lrs else 1
        return {
            "global_step": self._global_step.item(),
            "num_param_groups": len(self._group_params),
            "mean_lr": sum(lrs) / n if lrs else 0.0,
            "std_lr": (sum((x - sum(lrs) / n) ** 2 for x in lrs) / n) ** 0.5 if lrs else 0.0,
            "mean_momentum": sum(momenta) / n if momenta else 0.0,
            "std_momentum": (sum((x - sum(momenta) / n) ** 2 for x in momenta) / n) ** 0.5 if momenta else 0.0,
            "mean_weight_decay": sum(wds) / n if wds else 0.0,
            "std_weight_decay": (sum((x - sum(wds) / n) ** 2 for x in wds) / n) ** 0.5 if wds else 0.0,
            "gradient_norms": grad_norms,
            "gradient_means": grad_means,
            "gradient_vars": grad_vars,
        }
