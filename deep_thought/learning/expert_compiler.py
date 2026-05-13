"""
Feature → Expert Compiler (FEC) for Deep Thought.

Converts validated features into reusable, sparse, specialized
computation modules (experts) with safety constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

from deep_thought.config import ExpertCompilerConfig
from deep_thought.learning.feature_validation import Feature, FeatureState
from deep_thought.architecture.experts import Expert, ExpertConfig, ExpertState


@dataclass
class ExpertCandidate:
    """Candidate expert during compilation."""
    expert: Expert
    feature: Feature
    quarantine_steps_remaining: int
    stability_score: float
    anchor_loss: float = 0.0


class ExpertCompiler(nn.Module):
    """
    Feature → Expert Compiler (FEC).
    
    Converts validated features into experts through:
    - Feature crystallization
    - Expert initialization (cloning, not random)
    - Feature anchoring
    - Controlled specialization
    - Quarantine phase
    - Expert evaluation
    """
    
    def __init__(
        self,
        config: ExpertCompilerConfig,
        expert_config: ExpertConfig,
        latent_dim: int = 1024
    ):
        super().__init__()
        self.config = config
        self.expert_config = expert_config
        self.latent_dim = latent_dim
        
        # Candidate experts in quarantine
        self.candidates: Dict[int, ExpertCandidate] = {}
        
        # Best expert for cloning
        self.best_expert_id = 0
        
        # Anchor loss coefficient
        self.anchor_coef = config.anchor_loss_coef
    
    def crystallize_feature(self, feature: Feature) -> torch.Tensor:
        """
        Compress feature activation traces into prototype vector.
        
        Args:
            feature: Feature to crystallize
            
        Returns:
            prototype: Prototype latent vector
        """
        # Use feature vector as prototype
        # In practice, would average over activation traces
        prototype = feature.vector.detach().clone()
        
        return prototype
    
    def initialize_expert(
        self,
        prototype: torch.Tensor,
        parent_expert: Optional[Expert] = None
    ) -> Expert:
        """
        Initialize expert from prototype and parent.
        
        Args:
            prototype: Feature prototype
            parent_expert: Expert to clone (if None, use best)
            
        Returns:
            expert: Initialized expert
        """
        if parent_expert is None:
            # Create new expert with default initialization
            expert_id = 0  # Will be assigned by expert bank
            expert = Expert(self.expert_config, expert_id)
        else:
            # Clone parent with noise
            expert_id = 0
            expert = parent_expert.clone(expert_id, noise_scale=0.01)
        
        return expert
    
    def add_anchor_loss(
        self,
        expert: Expert,
        prototype: torch.Tensor,
        h_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Add anchor loss to keep expert faithful to feature.
        
        L_anchor = ||E(h) - f(h)||^2
        
        Args:
            expert: Expert to anchor
            prototype: Feature prototype
            h_t: Hidden state
            
        Returns:
            anchor_loss: Anchor loss
        """
        # Apply expert
        expert_output = expert(h_t)
        
        # Compute distance to prototype
        anchor_loss = F.mse_loss(expert_output, prototype)
        
        return anchor_loss
    
    def create_candidate(
        self,
        feature: Feature,
        parent_expert: Optional[Expert] = None
    ) -> int:
        """
        Create a candidate expert from a feature.
        
        Args:
            feature: Validated feature
            parent_expert: Parent expert to clone
            
        Returns:
            candidate_id: ID of created candidate
        """
        # Crystallize feature
        prototype = self.crystallize_feature(feature)
        
        # Initialize expert
        expert = self.initialize_expert(prototype, parent_expert)
        
        # Create candidate
        candidate_id = len(self.candidates)
        candidate = ExpertCandidate(
            expert=expert,
            feature=feature,
            quarantine_steps_remaining=self.config.quarantine_steps,
            stability_score=0.0,
        )
        
        self.candidates[candidate_id] = candidate
        
        return candidate_id
    
    def evaluate_candidate(
        self,
        candidate_id: int,
        h_t: torch.Tensor,
        reward: float
    ) -> float:
        """
        Evaluate candidate expert performance.
        
        Args:
            candidate_id: Candidate to evaluate
            h_t: Hidden state
            reward: Reward obtained
            
        Returns:
            stability_score: Updated stability score
        """
        if candidate_id not in self.candidates:
            return 0.0
        
        candidate = self.candidates[candidate_id]
        
        # Apply expert
        expert_output = candidate.expert(h_t)
        
        # Compute anchor loss
        prototype = self.crystallize_feature(candidate.feature)
        anchor_loss = self.add_anchor_loss(candidate.expert, prototype, h_t)
        candidate.anchor_loss = anchor_loss.item()
        
        # Update stability score based on reward and anchor loss
        # Higher reward, lower anchor loss = higher stability
        stability = reward - 0.1 * anchor_loss.item()
        
        # EMA update
        candidate.stability_score = (
            0.9 * candidate.stability_score +
            0.1 * stability
        )
        
        # Decrement quarantine
        candidate.quarantine_steps_remaining -= 1
        
        return candidate.stability_score
    
    def promote_candidate(
        self,
        candidate_id: int
    ) -> Optional[Expert]:
        """
        Promote candidate to full expert if ready.
        
        Args:
            candidate_id: Candidate to promote
            
        Returns:
            expert: Promoted expert (or None if not ready)
        """
        if candidate_id not in self.candidates:
            return None
        
        candidate = self.candidates[candidate_id]
        
        # Check if quarantine is over and stability is good
        if candidate.quarantine_steps_remaining <= 0:
            if candidate.stability_score > 0.0:
                # Promote
                expert = candidate.expert
                del self.candidates[candidate_id]
                return expert
        
        return None
    
    def prune_candidates(self, threshold: float = -1.0):
        """Remove candidates with poor stability."""
        to_remove = []
        
        for cid, candidate in self.candidates.items():
            if candidate.stability_score < threshold:
                to_remove.append(cid)
        
        for cid in to_remove:
            del self.candidates[cid]
    
    def split_expert(
        self,
        expert: Expert,
        variance_threshold: float = 0.5
    ) -> Tuple[Expert, Expert]:
        """
        Split an expert if it has high variance (overloaded).
        
        Args:
            expert: Expert to split
            variance_threshold: Variance threshold
            
        Returns:
            child1, child2: Split experts
        """
        # Create two children with small perturbations
        child1 = expert.clone(0, noise_scale=0.005)
        child2 = expert.clone(0, noise_scale=-0.005)
        
        return child1, child2
    
    def merge_experts(
        self,
        expert1: Expert,
        expert2: Expert,
        distance_threshold: float = 0.1
    ) -> Optional[Expert]:
        """
        Merge two similar experts.
        
        Args:
            expert1: First expert
            expert2: Second expert
            distance_threshold: Distance threshold
            
        Returns:
            merged: Merged expert (or None if too different)
        """
        # Compute parameter distance
        distance = 0.0
        for p1, p2 in zip(expert1.parameters(), expert2.parameters()):
            distance += (p1 - p2).norm().item()
        
        if distance < distance_threshold:
            # Average parameters
            merged = deepcopy(expert1)
            with torch.no_grad():
                for p_merged, p1, p2 in zip(
                    merged.parameters(),
                    expert1.parameters(),
                    expert2.parameters()
                ):
                    p_merged.data = (p1.data + p2.data) / 2
            return merged
        
        return None
    
    def get_candidate_stats(self) -> Dict:
        """Get statistics about candidates."""
        stats = {
            "num_candidates": len(self.candidates),
            "avg_stability": 0.0,
            "avg_quarantine": 0.0,
        }
        
        if self.candidates:
            stats["avg_stability"] = sum(
                c.stability_score for c in self.candidates.values()
            ) / len(self.candidates)
            stats["avg_quarantine"] = sum(
                c.quarantine_steps_remaining for c in self.candidates.values()
            ) / len(self.candidates)
        
        return stats
