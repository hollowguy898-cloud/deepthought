"""
Semantic memory module for Deep Thought.

Implements compressed knowledge storage for generalized world structure,
abstract rules, and learned invariants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from deep_thought.config import MemoryConfig


@dataclass
class Concept:
    """Abstract concept in semantic memory."""
    embedding: torch.Tensor
    prototype: torch.Tensor
    strength: float
    usage_count: int = 0


class SemanticMemory(nn.Module):
    """
    Semantic memory for generalized knowledge.
    
    Stores:
    - Generalized world structure
    - Abstract rules
    - Learned invariants
    
    Updated via slow consolidation from episodic memory.
    """
    
    def __init__(self, config: MemoryConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.semantic_dim = config.semantic_dim
        
        # Concept encoder
        self.concept_encoder = nn.Sequential(
            nn.Linear(latent_dim, self.semantic_dim),
            nn.ReLU(),
            nn.Linear(self.semantic_dim, self.semantic_dim),
        )
        
        # Concept storage
        self.capacity = config.semantic_capacity
        self.concepts: list[Concept] = []
        
        # Consolidation rate
        self.consolidation_rate = config.consolidation_rate
    
    def write(
        self,
        latent: torch.Tensor,
        observation: torch.Tensor
    ):
        """
        Write to semantic memory (consolidation).
        
        Args:
            latent: Latent representation
            observation: Observation
        """
        # Encode to concept
        with torch.no_grad():
            embedding = self.concept_encoder(latent).squeeze(0)
        
        # Check for similar existing concept
        similar_idx = self._find_similar(embedding)
        
        if similar_idx is not None:
            # Strengthen existing concept
            self.concepts[similar_idx].strength = (
                (1 - self.consolidation_rate) * self.concepts[similar_idx].strength +
                self.consolidation_rate
            )
            self.concepts[similar_idx].usage_count += 1
        else:
            # Create new concept
            if len(self.concepts) < self.capacity:
                concept = Concept(
                    embedding=embedding,
                    prototype=latent.squeeze(0).detach(),
                    strength=0.5,
                    usage_count=1
                )
                self.concepts.append(concept)
            else:
                # Replace weakest concept
                weakest_idx = min(
                    range(len(self.concepts)),
                    key=lambda i: self.concepts[i].strength
                )
                self.concepts[weakest_idx] = Concept(
                    embedding=embedding,
                    prototype=latent.squeeze(0).detach(),
                    strength=0.5,
                    usage_count=1
                )
    
    def read(
        self,
        query: torch.Tensor,
        k: int = 3
    ) -> Tuple[torch.Tensor, list[Concept]]:
        """
        Retrieve relevant concepts.
        
        Args:
            query: Query latent
            k: Number of concepts to retrieve
            
        Returns:
            semantic_read: Aggregated semantic read
            concepts: Retrieved concepts
        """
        if len(self.concepts) == 0:
            device = query.device
            return torch.zeros(1, self.latent_dim, device=device), []
        
        # Encode query
        with torch.no_grad():
            q = self.concept_encoder(query).squeeze(0)
        
        # Compute similarities
        similarities = []
        for concept in self.concepts:
            sim = F.cosine_similarity(
                q.unsqueeze(0),
                concept.embedding.unsqueeze(0),
                dim=-1
            ).item()
            similarities.append(sim)
        
        # Get top-k
        top_k_indices = torch.topk(
            torch.tensor(similarities),
            min(k, len(similarities))
        ).indices
        
        # Retrieve concepts
        retrieved = [self.concepts[i] for i in top_k_indices]
        
        # Weight by strength
        similarities_k = [similarities[i] for i in top_k_indices]
        strengths = [retrieved[i].strength for i in range(len(retrieved))]
        weights = torch.tensor(similarities_k) * torch.tensor(strengths)
        weights = F.softmax(weights, dim=0)
        
        # Aggregate prototypes
        prototypes = torch.stack([c.prototype for c in retrieved])
        semantic_read = (weights.unsqueeze(-1) * prototypes).sum(dim=0, keepdim=True)
        
        return semantic_read, retrieved
    
    def _find_similar(
        self,
        embedding: torch.Tensor,
        threshold: float = 0.9
    ) -> Optional[int]:
        """Find similar existing concept."""
        for i, concept in enumerate(self.concepts):
            sim = F.cosine_similarity(
                embedding.unsqueeze(0),
                concept.embedding.unsqueeze(0),
                dim=-1
            ).item()
            if sim > threshold:
                return i
        return None
    
    def decay_strength(self):
        """Decay strength of all concepts."""
        for concept in self.concepts:
            concept.strength *= 0.999
    
    def prune_weak(self, threshold: float = 0.1):
        """Prune weak concepts."""
        self.concepts = [
            c for c in self.concepts
            if c.strength > threshold or c.usage_count > 10
        ]
    
    def get_size(self) -> int:
        """Get current number of concepts."""
        return len(self.concepts)
