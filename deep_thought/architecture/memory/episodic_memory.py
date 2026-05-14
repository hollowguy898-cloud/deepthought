"""
Episodic memory module for Deep Thought.

Implements key-value memory for storing and retrieving specific experiences.
Stores high-reward events, high prediction error events, and novel trajectories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random

from deep_thought.config import MemoryConfig


@dataclass
class MemoryEntry:
    """Single episodic memory entry."""
    key: torch.Tensor
    value: torch.Tensor
    observation: torch.Tensor
    action: torch.Tensor
    reward: float
    done: bool
    importance: float
    age: int = 0
    retrieval_count: int = 0


class EpisodicMemory(nn.Module):
    """
    Episodic memory for specific experiences.
    
    Stores:
    - High reward events
    - High prediction error events
    - Novelty spikes
    - Failure states
    
    Uses attention-based retrieval.
    """
    
    def __init__(self, config: MemoryConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        # Key and value projections
        self.key_proj = nn.Linear(latent_dim, config.episodic_key_dim)
        self.value_proj = nn.Linear(latent_dim, config.episodic_value_dim)
        
        # Query projection
        self.query_proj = nn.Linear(latent_dim, config.episodic_key_dim)
        
        # Memory buffer
        self.capacity = config.episodic_capacity
        self.buffer: List[MemoryEntry] = []
        
        # Importance computation
        self.importance_net = nn.Sequential(
            nn.Linear(latent_dim + 3, 128),  # latent + reward + error + novelty
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def compute_importance(
        self,
        latent: torch.Tensor,
        reward: float,
        prediction_error: float,
        novelty: float
    ) -> float:
        """
        Compute importance score for memory entry.
        
        Args:
            latent: Latent representation
            reward: Reward value
            prediction_error: Prediction error
            novelty: Novelty score
            
        Returns:
            Importance score
        """
        with torch.no_grad():
            # Handle batch dimension - use mean across batch
            if latent.dim() > 1:
                latent_flat = latent.mean(dim=0)
            else:
                latent_flat = latent
            
            features = torch.cat([
                latent_flat,
                torch.tensor([reward, prediction_error, novelty], device=latent.device)
            ])
            importance = self.importance_net(features).item()
        return importance
    
    def write(
        self,
        latent: torch.Tensor,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        prediction_error: float = 0.0,
        novelty: float = 0.0
    ):
        """
        Write entry to episodic memory if important enough.
        
        Args:
            latent: Latent representation
            observation: Observation
            action: Action taken
            reward: Reward received
            done: Episode done flag
            prediction_error: Prediction error
            novelty: Novelty score
        """
        # Compute importance
        importance = self.compute_importance(
            latent, reward, prediction_error, novelty
        )
        
        # Only store if important enough
        if importance < self.config.importance_threshold:
            return
        
        # Create entry - handle batch dimension gracefully
        with torch.no_grad():
            key = self.key_proj(latent.detach())
            value = self.value_proj(latent.detach())
            
            # Squeeze batch dimension if present
            if key.dim() > 1 and key.size(0) == 1:
                key = key.squeeze(0)
            if value.dim() > 1 and value.size(0) == 1:
                value = value.squeeze(0)
            
            obs_detached = observation.detach()
            if obs_detached.dim() > 1 and obs_detached.size(0) == 1:
                obs_detached = obs_detached.squeeze(0)
            
            act_detached = action.detach()
            if act_detached.dim() > 1 and act_detached.size(0) == 1:
                act_detached = act_detached.squeeze(0)
        
        entry = MemoryEntry(
            key=key,
            value=value,
            observation=obs_detached,
            action=act_detached,
            reward=reward,
            done=done,
            importance=importance,
        )
        
        # Add to buffer
        self.buffer.append(entry)
        
        # Evict if over capacity
        if len(self.buffer) > self.capacity:
            self._evict()
    
    def read(
        self,
        query: torch.Tensor,
        k: int = 5
    ) -> Tuple[torch.Tensor, List[MemoryEntry]]:
        """
        Retrieve k most relevant memories.
        
        Args:
            query: Query latent
            k: Number of memories to retrieve
            
        Returns:
            memory_read: Aggregated memory read
            entries: Retrieved entries
        """
        if len(self.buffer) == 0:
            device = query.device
            batch_size = query.size(0)
            return torch.zeros(batch_size, self.latent_dim, device=device), []
        
        # Compute query - handle batch dimension
        with torch.no_grad():
            q = self.query_proj(query)
            # Use first element for similarity search if batched
            if q.dim() > 1:
                q_search = q[0]
            else:
                q_search = q
        
        # Compute similarities
        similarities = []
        for entry in self.buffer:
            sim = F.cosine_similarity(
                q_search.unsqueeze(0),
                entry.key.unsqueeze(0),
                dim=-1
            ).item()
            similarities.append(sim)
        
        # Get top-k
        top_k_indices = torch.topk(
            torch.tensor(similarities),
            min(k, len(similarities))
        ).indices
        
        # Retrieve entries
        entries = [self.buffer[i] for i in top_k_indices]
        
        # Compute attention weights
        # Clamp negative similarities (cosine similarity can be negative for anti-correlated vectors)
        similarities_k = [max(0.0, similarities[i]) for i in top_k_indices]
        attention = F.softmax(torch.tensor(similarities_k, dtype=torch.float32, device=query.device), dim=0)
        
        # Aggregate values
        values = torch.stack([entry.value for entry in entries])
        memory_read = (attention.unsqueeze(-1) * values).sum(dim=0, keepdim=True)
        
        # Expand to match batch dimension
        batch_size = query.size(0)
        memory_read = memory_read.expand(batch_size, -1)
        
        # If value dim != latent_dim, we need a projection (handled by memory_system fusion)
        # For now, pad or truncate to latent_dim
        if memory_read.size(-1) != self.latent_dim:
            if memory_read.size(-1) < self.latent_dim:
                padding = torch.zeros(batch_size, self.latent_dim - memory_read.size(-1), device=query.device)
                memory_read = torch.cat([memory_read, padding], dim=-1)
            else:
                memory_read = memory_read[:, :self.latent_dim]
        
        # Update retrieval counts
        for idx in top_k_indices:
            self.buffer[idx].retrieval_count += 1
        
        return memory_read, entries
    
    def _evict(self):
        """Evict least important entries."""
        # Sort by importance and age
        self.buffer.sort(
            key=lambda x: (x.importance, -x.age),
            reverse=True
        )
        # Remove oldest/least important
        self.buffer = self.buffer[:self.capacity]
    
    def age_memories(self):
        """Increment age of all memories."""
        for entry in self.buffer:
            entry.age += 1
    
    def consolidate(self, semantic_memory):
        """
        Consolidate important episodic memories into semantic memory.
        
        Args:
            semantic_memory: Semantic memory module
        """
        # Get high-importance, frequently-retrieved memories
        candidates = [
            entry for entry in self.buffer
            if entry.importance > 0.8 and entry.retrieval_count > 5
        ]
        
        for entry in candidates:
            # Add to semantic memory - use observation as both latent and observation inputs
            # since semantic_memory.write() expects latent_dim for its concept_encoder
            obs_input = entry.observation.unsqueeze(0) if entry.observation.dim() == 1 else entry.observation
            semantic_memory.write(obs_input, obs_input)
        
        # Remove consolidated entries
        self.buffer = [
            entry for entry in self.buffer
            if not (entry.importance > 0.8 and entry.retrieval_count > 5)
        ]
    
    def get_size(self) -> int:
        """Get current memory size."""
        return len(self.buffer)
