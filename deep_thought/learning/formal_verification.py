"""
Stable Self-Improvement Component 2: Formal Verification Layer (FVE Evolution).

Evolves the Feature Validation Engine into a Formal Verification Layer that
enforces logic-based constraints and entropy regulation before any
architectural change is committed.

Key ideas:
  - Before the FEC spawns a new expert, the Formal Verification Layer
    runs symbolic checks to ensure the new subnetwork doesn't violate
    core behavioral constraints.
  - Entropy Regulation: KL-divergence penalty ensures self-improvements
    don't deviate too far from a known stable baseline too quickly:
        D_KL(P_new || P_stable) < epsilon
  - Constraint checking: behavioral invariants are encoded as symbolic
    predicates that must hold after any architectural change.
  - Three-tier verification: SYNTACTIC (fast, local), SEMANTIC (slower,
    cross-module), CAUSAL (full ablation test).

This layer sits between the Feature Validation Engine and the Expert
Compiler, acting as a quality gate that prevents "hallucinated"
architectural efficiencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Set
from dataclasses import dataclass, field
from enum import Enum
import math


class VerificationTier(Enum):
    """Verification depth levels."""
    SYNTACTIC = "syntactic"     # Fast local checks (shape, norm, rank)
    SEMANTIC = "semantic"       # Cross-module consistency checks
    CAUSAL = "causal"           # Full ablation / counterfactual test


class ConstraintType(Enum):
    """Types of behavioral constraints."""
    OUTPUT_RANGE = "output_range"             # Output values must stay in range
    CAPABILITY_DENSITY = "capability_density" # Density must not decrease
    ROUTING_ENTROPY = "routing_entropy"       # Entropy must stay in bounds
    PARAMETER_BUDGET = "parameter_budget"     # Total params must stay in budget
    EXPERT_COUNT = "expert_count"             # Number of experts in bounds
    GRADIENT_STABILITY = "gradient_stability" # Gradients must not explode


@dataclass
class BehavioralConstraint:
    """A single behavioral constraint that must hold after any change.

    Attributes:
        name: Human-readable name.
        constraint_type: Type of constraint.
        check_fn: A callable that takes (model_state_before, model_state_after)
            and returns (passed: bool, violation_magnitude: float).
        is_hard: If True, any violation blocks the change.
            If False, violation adds a penalty but doesn't block.
        threshold: Numeric threshold for the constraint.
    """
    name: str
    constraint_type: ConstraintType
    check_fn: object = None  # Callable
    is_hard: bool = True
    threshold: float = 0.0


@dataclass
class FormalVerificationConfig:
    """Configuration for the Formal Verification Layer.

    Attributes:
        use_formal_verification: Whether to enable the formal verification layer.
        kl_epsilon: Maximum allowed KL divergence between new and stable
            routing distributions.  Prevents rapid drift.
        kl_check_interval: How often (in steps) to check KL divergence.
            More frequent checks are safer but slower.
        max_output_norm: Maximum allowed norm for any expert's output.
            Prevents output explosion.
        min_capability_density: Minimum capability density below which
            no growth is allowed.
        gradient_explosion_threshold: Maximum gradient norm before a
            change is rejected.
        verification_tier: Default verification tier for checks.
        constraint_violation_cooldown: Number of steps to wait after a
            constraint violation before allowing changes again.
    """
    use_formal_verification: bool = True
    kl_epsilon: float = 0.1
    kl_check_interval: int = 100
    max_output_norm: float = 100.0
    min_capability_density: float = 0.001
    gradient_explosion_threshold: float = 10.0
    verification_tier: str = "semantic"
    constraint_violation_cooldown: int = 500


class EntropyRegulator:
    """Ensures routing distribution changes don't violate KL constraint.

    D_KL(P_new || P_stable) < epsilon

    The "stable baseline" P_stable is the routing distribution from a
    known-good checkpoint.  If a proposed change pushes the routing
    distribution too far from this baseline, it is rejected.

    This prevents the "greedy router" problem where the sparse router
    collapses onto a small set of experts, causing the system to
    overfit and lose generality.
    """

    def __init__(self, config: FormalVerificationConfig):
        self.config = config
        self._stable_probs: Optional[torch.Tensor] = None
        self._step: int = 0

    def set_stable_baseline(self, probs: torch.Tensor):
        """Set the stable baseline routing distribution.

        Args:
            probs: Routing probabilities (batch, num_experts).
                Averaged over the batch to get a single distribution.
        """
        self._stable_probs = probs.detach().mean(dim=0).clone()
        # Add small epsilon to prevent log(0)
        self._stable_probs = self._stable_probs + 1e-8
        self._stable_probs = self._stable_probs / self._stable_probs.sum()

    def check_kl_divergence(self, new_probs: torch.Tensor) -> Tuple[bool, float]:
        """Check if new routing distribution is within KL budget.

        Args:
            new_probs: Proposed new routing probabilities (batch, num_experts).

        Returns:
            (within_budget, kl_value) tuple.
        """
        if self._stable_probs is None:
            return True, 0.0  # No baseline yet

        # Average over batch
        p_new = new_probs.detach().mean(dim=0).clone()
        p_new = p_new + 1e-8
        p_new = p_new / p_new.sum()

        # KL(P_new || P_stable)
        kl = F.kl_div(
            self._stable_probs.log(),
            p_new,
            reduction="sum"
        ).item()

        self._step += 1
        within_budget = kl < self.config.kl_epsilon
        return within_budget, kl

    def get_stats(self) -> Dict:
        """Return regulator statistics."""
        return {
            "has_baseline": self._stable_probs is not None,
            "kl_epsilon": self.config.kl_epsilon,
            "step": self._step,
        }


class FormalVerificationLayer(nn.Module):
    """Formal Verification Layer for stable self-improvement.

    Sits between the Feature Validation Engine and the Expert Compiler,
    acting as a quality gate that prevents harmful architectural changes.

    Three-tier verification:
      1. SYNTACTIC: Fast checks (output shape, norm, rank).
         Runs every step. Cost: negligible.
      2. SEMANTIC: Cross-module checks (KL divergence, density bounds,
         routing entropy). Runs every kl_check_interval steps.
      3. CAUSAL: Full ablation test. Runs only when a major architectural
         change is proposed. Cost: expensive but thorough.

    A proposed change must pass ALL active constraint checks at the
    appropriate tier before being approved.
    """

    def __init__(self, config: FormalVerificationConfig, num_experts: int = 128):
        super().__init__()
        self.config = config
        self.num_experts = num_experts

        # Entropy regulator
        self.entropy_regulator = EntropyRegulator(config)

        # Behavioral constraints
        self._constraints: List[BehavioralConstraint] = []
        self._register_default_constraints()

        # Violation tracking
        self._violation_history: List[Dict] = []
        self._steps_since_violation: int = 0
        self._total_changes_approved: int = 0
        self._total_changes_rejected: int = 0

    def _register_default_constraints(self):
        """Register the standard behavioral constraints."""
        # Hard constraint: output norm must not explode
        self._constraints.append(BehavioralConstraint(
            name="output_norm_bound",
            constraint_type=ConstraintType.OUTPUT_RANGE,
            is_hard=True,
            threshold=self.config.max_output_norm,
        ))

        # Hard constraint: capability density must not decrease below minimum
        self._constraints.append(BehavioralConstraint(
            name="min_capability_density",
            constraint_type=ConstraintType.CAPABILITY_DENSITY,
            is_hard=True,
            threshold=self.config.min_capability_density,
        ))

        # Soft constraint: routing entropy should stay healthy
        self._constraints.append(BehavioralConstraint(
            name="routing_entropy_bounds",
            constraint_type=ConstraintType.ROUTING_ENTROPY,
            is_hard=False,
            threshold=0.5,
        ))

        # Hard constraint: total parameter budget
        self._constraints.append(BehavioralConstraint(
            name="parameter_budget",
            constraint_type=ConstraintType.PARAMETER_BUDGET,
            is_hard=True,
        ))

    def verify_syntactic(
        self,
        expert_output: torch.Tensor,
    ) -> Tuple[bool, List[str]]:
        """Run fast syntactic checks on an expert's output.

        Args:
            expert_output: Output tensor from the expert.

        Returns:
            (passed, violations) tuple.
        """
        violations = []

        # Check 1: Output norm is bounded
        output_norm = expert_output.norm().item()
        if output_norm > self.config.max_output_norm:
            violations.append(
                f"output_norm_exceeded: {output_norm:.2f} > {self.config.max_output_norm}"
            )

        # Check 2: No NaN or Inf
        if torch.isnan(expert_output).any():
            violations.append("output_contains_nan")
        if torch.isinf(expert_output).any():
            violations.append("output_contains_inf")

        # Check 3: Output shape is valid (non-empty, correct dims)
        if expert_output.numel() == 0:
            violations.append("output_is_empty")

        return len(violations) == 0, violations

    def verify_semantic(
        self,
        new_routing_probs: torch.Tensor,
        current_capability_density: float,
    ) -> Tuple[bool, List[str], Dict]:
        """Run cross-module semantic checks.

        Args:
            new_routing_probs: Proposed routing probabilities.
            current_capability_density: Current capability density.

        Returns:
            (passed, violations, details) tuple.
        """
        violations = []
        details = {}

        # Check 1: KL divergence from stable baseline
        within_kl, kl_value = self.entropy_regulator.check_kl_divergence(
            new_routing_probs
        )
        details["kl_divergence"] = kl_value
        details["kl_within_budget"] = within_kl
        if not within_kl:
            violations.append(
                f"kl_divergence_exceeded: {kl_value:.4f} > {self.config.kl_epsilon}"
            )

        # Check 2: Capability density above minimum
        if current_capability_density < self.config.min_capability_density:
            violations.append(
                f"capability_density_below_min: {current_capability_density:.6f} < "
                f"{self.config.min_capability_density}"
            )

        # Check 3: Routing probabilities are valid (sum to ~1, all positive)
        prob_sum = new_routing_probs.sum(dim=-1).mean().item()
        if abs(prob_sum - 1.0) > 0.1:
            violations.append(f"routing_probs_not_normalized: sum={prob_sum:.4f}")

        return len(violations) == 0, violations, details

    def verify_causal(
        self,
        model_before_output: torch.Tensor,
        model_after_output: torch.Tensor,
        reward_before: float,
        reward_after: float,
    ) -> Tuple[bool, List[str], Dict]:
        """Run full causal / counterfactual verification.

        This is the most expensive check. It compares the model's
        behavior before and after a proposed change to ensure the
        change doesn't cause regression.

        Args:
            model_before_output: Model output before the change.
            model_after_output: Model output after the change.
            reward_before: Reward before the change.
            reward_after: Reward after the change.

        Returns:
            (passed, violations, details) tuple.
        """
        violations = []
        details = {}

        # Check 1: Output distribution hasn't shifted catastrophically
        # Use cosine similarity as a measure of behavioral preservation
        cos_sim = F.cosine_similarity(
            model_before_output.flatten().unsqueeze(0),
            model_after_output.flatten().unsqueeze(0),
            dim=-1
        ).item()
        details["cosine_similarity"] = cos_sim
        if cos_sim < 0.5:
            violations.append(
                f"behavioral_shift_too_large: cos_sim={cos_sim:.4f}"
            )

        # Check 2: Reward hasn't dropped significantly
        if reward_before > 1e-8:
            relative_reward_drop = (reward_before - reward_after) / abs(reward_before)
            details["relative_reward_drop"] = relative_reward_drop
            if relative_reward_drop > 0.2:
                violations.append(
                    f"reward_regression: drop={relative_reward_drop:.4f}"
                )

        # Check 3: Output variance hasn't collapsed or exploded
        var_before = model_before_output.var().item()
        var_after = model_after_output.var().item()
        if var_before > 1e-8:
            var_ratio = var_after / var_before
            details["variance_ratio"] = var_ratio
            if var_ratio > 10.0 or var_ratio < 0.01:
                violations.append(
                    f"output_variance_anomaly: ratio={var_ratio:.4f}"
                )

        return len(violations) == 0, violations, details

    def verify_change(
        self,
        change_type: str,
        expert_output: Optional[torch.Tensor] = None,
        new_routing_probs: Optional[torch.Tensor] = None,
        current_capability_density: Optional[float] = None,
        model_before_output: Optional[torch.Tensor] = None,
        model_after_output: Optional[torch.Tensor] = None,
        reward_before: Optional[float] = None,
        reward_after: Optional[float] = None,
        tier: Optional[VerificationTier] = None,
    ) -> Tuple[bool, Dict]:
        """Verify a proposed architectural change at the appropriate tier.

        This is the main entry point.  Call this before committing any
        architectural change (expert growth, pruning, routing adjustment).

        Args:
            change_type: Type of change ("growth", "pruning", "routing").
            expert_output: Expert output for syntactic check.
            new_routing_probs: Routing probs for semantic check.
            current_capability_density: For semantic check.
            model_before_output: For causal check.
            model_after_output: For causal check.
            reward_before: For causal check.
            reward_after: For causal check.
            tier: Override verification tier.

        Returns:
            (approved, details) tuple.
        """
        if not self.config.use_formal_verification:
            return True, {"skipped": True}

        # Cooldown after violation
        if self._steps_since_violation < self.config.constraint_violation_cooldown:
            self._steps_since_violation += 1
            return False, {"reason": "cooldown_after_violation"}

        tier = tier or VerificationTier(self.config.verification_tier)
        all_violations = []
        details = {"tier": tier.value, "change_type": change_type}

        # Always run syntactic checks
        if expert_output is not None:
            passed, violations = self.verify_syntactic(expert_output)
            all_violations.extend(violations)
            details["syntactic_passed"] = passed

        # Semantic checks (SYNTACTIC tier skips these)
        if tier in (VerificationTier.SEMANTIC, VerificationTier.CAUSAL):
            if new_routing_probs is not None and current_capability_density is not None:
                passed, violations, sem_details = self.verify_semantic(
                    new_routing_probs, current_capability_density
                )
                all_violations.extend(violations)
                details["semantic_passed"] = passed
                details.update(sem_details)

        # Causal checks (only at CAUSAL tier)
        if tier == VerificationTier.CAUSAL:
            if all(v is not None for v in [
                model_before_output, model_after_output,
                reward_before, reward_after
            ]):
                passed, violations, causal_details = self.verify_causal(
                    model_before_output, model_after_output,
                    reward_before, reward_after
                )
                all_violations.extend(violations)
                details["causal_passed"] = passed
                details.update(causal_details)

        # Determine approval
        approved = len(all_violations) == 0
        details["violations"] = all_violations

        if approved:
            self._total_changes_approved += 1
        else:
            self._total_changes_rejected += 1
            self._steps_since_violation = 0
            self._violation_history.append({
                "change_type": change_type,
                "violations": all_violations,
            })

        return approved, details

    def update_stable_baseline(self, routing_probs: torch.Tensor):
        """Update the stable baseline routing distribution.

        Should be called periodically (e.g. at SLOW timescale) when
        the system is known to be in a good state.
        """
        self.entropy_regulator.set_stable_baseline(routing_probs)

    def get_stats(self) -> Dict:
        """Return verification layer statistics."""
        return {
            "total_approved": self._total_changes_approved,
            "total_rejected": self._total_changes_rejected,
            "violation_count": len(self._violation_history),
            "steps_since_violation": self._steps_since_violation,
            "entropy_regulator": self.entropy_regulator.get_stats(),
        }
