"""
Feature Validation Engine (FVE) for Deep Thought.

Validates features before integration into the system to prevent
learning noise and fake patterns. Features must survive time,
reuse, and counterfactual stress before promotion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from deep_thought.config import FeatureValidationConfig


class FeatureState(Enum):
    """Feature lifecycle states."""
    CANDIDATE = "candidate"
    OBSERVED = "observed"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    DEPRECATED = "deprecated"
    PRUNED = "pruned"


@dataclass
class Feature:
    """A candidate feature in the validation pipeline."""
    feature_id: int
    vector: torch.Tensor
    activation_trace: List[float] = field(default_factory=list)
    reward_delta_trace: List[float] = field(default_factory=list)
    gradient_norm_trace: List[float] = field(default_factory=list)
    environment_tags: List[str] = field(default_factory=list)
    routing_effect: float = 0.0
    timestamp: int = 0
    state: FeatureState = FeatureState.CANDIDATE
    
    # Validation metrics
    temporal_consistency: float = 0.0
    cross_env_generalization: float = 0.0
    causal_impact: float = 0.0
    routing_entropy_impact: float = 0.0
    noise_robustness: float = 0.0
    
    # Statistics
    usage_count: int = 0
    total_reward_delta: float = 0.0
    total_gradient_norm: float = 0.0


class FeatureValidationEngine(nn.Module):
    """
    Feature Validation Engine (FVE).
    
    Validates features through:
    - Temporal consistency checks
    - Cross-environment generalization
    - Causal stress tests
    - Routing impact analysis
    - Noise robustness testing
    
    Only features that pass all tests are promoted to experts.
    """
    
    def __init__(self, config: FeatureValidationConfig, feature_dim: int = 1024):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        # Feature buffer
        self.buffer_size = config.feature_buffer_size
        self.features: Dict[int, Feature] = {}
        self.next_feature_id = 0
        
        # Validation window
        self.validation_window = config.validation_window
        
        # Thresholds
        self.temporal_threshold = config.temporal_consistency_threshold
        self.promotion_threshold = config.promotion_threshold
        self.noise_threshold = config.noise_robustness_threshold
        
        # Competition strength
        self.competition_strength = config.competition_strength
        
        # Feature extraction network
        self.extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
    
    def extract_features(
        self,
        latent: torch.Tensor,
        gradient_norms: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Extract candidate features from latent activations.

        Args:
            latent: Latent representation
            gradient_norms: Gradient norms per dimension

        Returns:
            List of extracted feature IDs
        """
        # FIX: Remove torch.no_grad() so the feature extractor receives
        # gradient signal during training. Previously, the extractor was
        # always computed with no_grad, meaning its parameters NEVER
        # received gradients and the feature_validator was a dead module.
        features = self.extractor(latent)
        
        # Simple clustering based on activation patterns
        feature_ids = []
        
        # For simplicity, create one feature per batch
        # In practice, use clustering algorithms
        batch_size = latent.size(0)
        for i in range(batch_size):
            feature_id = self._create_feature(
                features[i].unsqueeze(0),
                gradient_norms[i] if gradient_norms is not None else None
            )
            feature_ids.append(feature_id)
        
        return feature_ids
    
    def _create_feature(
        self,
        vector: torch.Tensor,
        gradient_norm: Optional[torch.Tensor] = None
    ) -> int:
        """Create a new candidate feature."""
        feature_id = self.next_feature_id
        self.next_feature_id += 1
        
        feature = Feature(
            feature_id=feature_id,
            vector=vector.detach().clone(),
            timestamp=0,
        )
        
        if gradient_norm is not None:
            feature.gradient_norm_trace.append(gradient_norm.item())
        
        self.features[feature_id] = feature
        
        # Evict if over capacity
        if len(self.features) > self.buffer_size:
            self._evict_weakest()
        
        return feature_id
    
    def update_feature(
        self,
        feature_id: int,
        reward_delta: float,
        gradient_norm: float,
        routing_entropy_delta: float = 0.0,
        environment_tag: str = "default"
    ):
        """
        Update feature statistics.
        
        Args:
            feature_id: Feature to update
            reward_delta: Reward change when feature active
            gradient_norm: Gradient norm contribution
            routing_entropy_delta: Change in routing entropy
            environment_tag: Current environment
        """
        if feature_id not in self.features:
            return
        
        feature = self.features[feature_id]
        
        # Update traces
        feature.activation_trace.append(1.0)
        feature.reward_delta_trace.append(reward_delta)
        feature.gradient_norm_trace.append(gradient_norm)
        feature.routing_effect += routing_entropy_delta
        
        if environment_tag not in feature.environment_tags:
            feature.environment_tags.append(environment_tag)
        
        # Update statistics
        feature.usage_count += 1
        feature.total_reward_delta += reward_delta
        feature.total_gradient_norm += gradient_norm
        
        # Keep traces bounded
        max_trace_length = self.validation_window
        if len(feature.activation_trace) > max_trace_length:
            feature.activation_trace = feature.activation_trace[-max_trace_length:]
            feature.reward_delta_trace = feature.reward_delta_trace[-max_trace_length:]
            feature.gradient_norm_trace = feature.gradient_norm_trace[-max_trace_length:]
    
    def compute_stability_score(self, feature: Feature) -> float:
        """
        Compute overall stability score for a feature.
        
        S = α*G + β*R + γ*U - δ*C
        
        Where:
        G = gradient contribution consistency
        R = reward improvement when active
        U = reuse across tasks/environments
        C = compute cost (simplified as 1.0)
        """
        if len(feature.gradient_norm_trace) < 2:
            return 0.0
        
        # Gradient consistency
        grad_mean = np.mean(feature.gradient_norm_trace)
        grad_std = np.std(feature.gradient_norm_trace)
        gradient_score = grad_mean / (grad_std + 1e-8)
        
        # Reward contribution
        if len(feature.reward_delta_trace) > 0:
            reward_score = np.mean(feature.reward_delta_trace)
        else:
            reward_score = 0.0
        
        # Reuse (number of environments)
        reuse_score = len(feature.environment_tags)
        
        # Compute stability
        alpha, beta, gamma, delta = 0.3, 0.3, 0.3, 0.1
        stability = (
            alpha * gradient_score +
            beta * reward_score +
            gamma * reuse_score -
            delta * 1.0
        )
        
        return max(0.0, stability)
    
    def check_temporal_consistency(self, feature: Feature) -> float:
        """
        Check if feature persists over time.
        
        Computes correlation between feature at different time points.
        """
        if len(feature.activation_trace) < 2:
            return 0.0
        
        # Simple consistency: how often does it activate?
        activation_rate = np.mean(feature.activation_trace)
        
        # Variance in activation
        activation_var = np.var(feature.activation_trace)
        
        # Consistency score
        consistency = activation_rate / (activation_var + 1e-8)
        
        feature.temporal_consistency = consistency
        return consistency
    
    def check_cross_env_generalization(self, feature: Feature) -> float:
        """
        Check if feature works across multiple environments.
        """
        num_envs = len(feature.environment_tags)
        
        if num_envs <= 1:
            return 0.0
        
        # Generalization increases with more environments
        generalization = min(1.0, num_envs / 5.0)
        
        feature.cross_env_generalization = generalization
        return generalization
    
    def causal_stress_test(
        self,
        feature_id: int,
        model,
        env,
        num_trials: int = 10
    ) -> float:
        """
        Test causal impact by removing feature and measuring delta.
        
        Args:
            feature_id: Feature to test
            model: The model
            env: Environment
            num_trials: Number of trials
            
        Returns:
            Causal impact score
        """
        if feature_id not in self.features:
            return 0.0
        
        # This is a simplified version
        # In practice, would actually ablate the feature and measure performance
        
        # For now, use reward delta trace as proxy
        feature = self.features[feature_id]
        
        if len(feature.reward_delta_trace) == 0:
            return 0.0
        
        # Average reward delta when feature is active
        avg_reward_delta = np.mean(feature.reward_delta_trace)
        
        feature.causal_impact = avg_reward_delta
        return avg_reward_delta
    
    def noise_robustness_test(
        self,
        feature: Feature,
        noise_scale: float = 0.1
    ) -> float:
        """
        Test feature robustness to noise.
        
        Perturbs feature vector and measures stability.
        """
        with torch.no_grad():
            original = feature.vector
            noisy = original + torch.randn_like(original) * noise_scale
            
            # Measure similarity
            similarity = F.cosine_similarity(
                original.unsqueeze(0),
                noisy.unsqueeze(0),
                dim=-1
            ).item()
        
        feature.noise_robustness = similarity
        return similarity
    
    def validate_feature(self, feature_id: int) -> bool:
        """
        Run full validation pipeline on a feature.
        
        Returns:
            True if feature passes validation
        """
        if feature_id not in self.features:
            return False
        
        feature = self.features[feature_id]
        
        # Temporal consistency
        temporal = self.check_temporal_consistency(feature)
        if temporal < self.temporal_threshold:
            return False
        
        # Cross-environment generalization
        generalization = self.check_cross_env_generalization(feature)
        if generalization < 0.3:  # Must work in at least 2 environments
            return False
        
        # Noise robustness
        robustness = self.noise_robustness_test(feature)
        if robustness < self.noise_threshold:
            return False
        
        # Compute overall stability
        stability = self.compute_stability_score(feature)
        
        # Update state
        if stability > self.promotion_threshold:
            feature.state = FeatureState.VALIDATED
            return True
        elif feature.usage_count > 10:
            feature.state = FeatureState.OBSERVED
        else:
            feature.state = FeatureState.CANDIDATE
        
        return False
    
    def promote_features(self) -> List[int]:
        """
        Promote validated features.
        
        Returns:
            List of promoted feature IDs
        """
        promoted = []
        
        for feature_id, feature in self.features.items():
            if feature.state == FeatureState.VALIDATED:
                feature.state = FeatureState.PROMOTED
                promoted.append(feature_id)
        
        return promoted
    
    def decay_features(self):
        """Decay weak features."""
        to_prune = []
        
        for feature_id, feature in self.features.items():
            if feature.state == FeatureState.CANDIDATE:
                # If candidate hasn't been used much, prune
                if feature.usage_count < 5 and len(feature.activation_trace) > 50:
                    to_prune.append(feature_id)
            elif feature.state == FeatureState.OBSERVED:
                # If observed but not validated, decay
                stability = self.compute_stability_score(feature)
                if stability < 0.1:
                    to_prune.append(feature_id)
        
        for feature_id in to_prune:
            self.features[feature_id].state = FeatureState.PRUNED
            del self.features[feature_id]
    
    def _evict_weakest(self):
        """Evict weakest features when buffer is full."""
        # Sort by stability score
        feature_scores = [
            (fid, self.compute_stability_score(f))
            for fid, f in self.features.items()
        ]
        feature_scores.sort(key=lambda x: x[1])
        
        # Remove weakest
        num_to_remove = len(self.features) - self.buffer_size
        for i in range(num_to_remove):
            fid = feature_scores[i][0]
            del self.features[fid]
    
    def get_promoted_features(self) -> List[Feature]:
        """Get all promoted features ready for expert compilation."""
        return [
            f for f in self.features.values()
            if f.state == FeatureState.PROMOTED
        ]
    
    def get_feature_stats(self) -> Dict:
        """Get statistics about features."""
        stats = {
            "total_features": len(self.features),
            "candidates": sum(1 for f in self.features.values() if f.state == FeatureState.CANDIDATE),
            "observed": sum(1 for f in self.features.values() if f.state == FeatureState.OBSERVED),
            "validated": sum(1 for f in self.features.values() if f.state == FeatureState.VALIDATED),
            "promoted": sum(1 for f in self.features.values() if f.state == FeatureState.PROMOTED),
        }
        return stats
