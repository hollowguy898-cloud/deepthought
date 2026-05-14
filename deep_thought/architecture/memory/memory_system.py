"""
Integrated memory system for Deep Thought.

Combines working, episodic, and semantic memory into a unified
memory management system.

Fix 5: Asymmetric Memory Read/Write
  - Writes are CHEAP and over-inclusive: low threshold, noisy storage ok
  - Reads are EXPENSIVE and heavily filtered: high relevance threshold
  - Memory CANNOT directly influence pruning or growth decisions
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from deep_thought.config import MemoryConfig
from deep_thought.architecture.memory.working_memory import WorkingMemory
from deep_thought.architecture.memory.episodic_memory import EpisodicMemory
from deep_thought.architecture.memory.semantic_memory import SemanticMemory


class MemorySystem(nn.Module):
    """
    Unified memory system combining all memory types.

    Coordinates:
    - Working memory (fast, volatile)
    - Episodic memory (specific experiences)
    - Semantic memory (generalized knowledge)

    Fix 5: Asymmetric read/write with firewalls.
    """

    def __init__(self, config: MemoryConfig, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim

        # Memory components
        if config.use_working_memory:
            self.working = WorkingMemory(config, latent_dim)
        else:
            self.working = None

        if config.use_episodic_memory:
            self.episodic = EpisodicMemory(config, latent_dim)
        else:
            self.episodic = None

        if config.use_semantic_memory:
            self.semantic = SemanticMemory(config, latent_dim)
        else:
            self.semantic = None

        # Memory fusion
        if config.use_episodic_memory and config.use_semantic_memory:
            self.fusion = nn.Linear(
                latent_dim * 2,
                latent_dim
            )
        else:
            self.fusion = None

        # Fix 5: Asymmetric memory parameters
        self.read_filter_threshold = 0.3  # High threshold for reads
        self.write_threshold = 0.01       # Low threshold for writes (cheap)

    def forward(
        self,
        h_prev: torch.Tensor,
        x_t: torch.Tensor,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        prediction_error: float = 0.0,
        novelty: float = 0.0,
        write: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Memory forward pass.

        Fix 5: Implements asymmetric read/write semantics.
        - Writes: cheap, over-inclusive (low threshold)
        - Reads: expensive, heavily filtered (high relevance threshold)

        Args:
            h_prev: Previous hidden state
            x_t: Current encoded observation
            observation: Raw observation
            action: Action taken
            reward: Reward received
            done: Episode done flag
            prediction_error: Prediction error
            novelty: Novelty score
            write: Whether to write to memory (governor-approved)

        Returns:
            h_t: Updated hidden state
            memory_info: Dictionary with memory information
        """
        memory_info = {}

        # Fix 5: EXPENSIVE, heavily filtered reads
        batch_size = x_t.size(0)
        episodic_read = torch.zeros(batch_size, self.latent_dim, device=x_t.device)
        semantic_read = torch.zeros(batch_size, self.latent_dim, device=x_t.device)

        if self.episodic is not None:
            episodic_read, episodic_entries = self.episodic.read(x_t, k=5)
            memory_info["episodic_entries"] = len(episodic_entries)

            # Fix 5: Filter reads by relevance — only use high-relevance entries
            # The read already returns top-k, but we apply an additional
            # relevance filter to zero out low-relevance reads.
            #
            # KNOWN LIMITATION: Using vector norm as a proxy for relevance
            # conflates magnitude with semantic meaning.  A memory entry
            # with small norm but high cosine similarity to the query can
            # be incorrectly filtered out, while a large-norm but
            # irrelevant entry may pass.  Consider replacing this with
            # cosine-similarity-based filtering for production use.
            read_norm = episodic_read.norm(dim=-1, keepdim=True)
            mean_norm = read_norm.mean() + 1e-8
            relevance = (read_norm / mean_norm).squeeze(-1)
            # Soft gating: allows gradient flow while still filtering low-relevance entries
            read_mask = torch.sigmoid((relevance - self.read_filter_threshold) * 10.0).unsqueeze(-1)
            episodic_read = episodic_read * read_mask

        if self.semantic is not None:
            semantic_read, semantic_concepts = self.semantic.read(x_t, k=3)
            memory_info["semantic_concepts"] = len(semantic_concepts)

            # Fix 5: Same soft gating for semantic reads
            # (Same norm-based limitation as episodic filter above)
            read_norm = semantic_read.norm(dim=-1, keepdim=True)
            mean_norm = read_norm.mean() + 1e-8
            relevance = (read_norm / mean_norm).squeeze(-1)
            read_mask = torch.sigmoid((relevance - self.read_filter_threshold) * 10.0).unsqueeze(-1)
            semantic_read = semantic_read * read_mask

        # Fuse memory reads
        if self.fusion is not None:
            memory_read = self.fusion(
                torch.cat([episodic_read, semantic_read], dim=-1)
            )
        else:
            memory_read = episodic_read + semantic_read

        # Update working memory
        if self.working is not None:
            h_t, delta_h = self.working(h_prev, x_t, memory_read)
            memory_info["delta_h"] = delta_h
        else:
            h_t = h_prev
            memory_info["delta_h"] = torch.zeros_like(h_t)

        # Fix 5: CHEAP, over-inclusive writes
        # Write threshold is very low — almost everything gets written
        if write and self.episodic is not None:
            self.episodic.write(
                x_t, observation, action, reward, done,
                prediction_error, novelty
            )

        memory_info["memory_read"] = memory_read
        memory_info["episodic_read"] = episodic_read
        memory_info["semantic_read"] = semantic_read

        return h_t, memory_info

    def consolidate(self):
        """Consolidate episodic memory into semantic memory."""
        if self.episodic is not None and self.semantic is not None:
            self.episodic.consolidate(self.semantic)

    def age_memories(self):
        """Age episodic memories."""
        if self.episodic is not None:
            self.episodic.age_memories()

    def decay_semantic(self):
        """Decay semantic memory strengths."""
        if self.semantic is not None:
            self.semantic.decay_strength()

    def prune_semantic(self):
        """Prune weak semantic concepts."""
        if self.semantic is not None:
            self.semantic.prune_weak()

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory systems."""
        stats = {}

        if self.working is not None:
            stats["working_memory_size"] = self.config.working_memory_size

        if self.episodic is not None:
            stats["episodic_size"] = self.episodic.get_size()
            stats["episodic_capacity"] = self.episodic.capacity

        if self.semantic is not None:
            stats["semantic_size"] = self.semantic.get_size()
            stats["semantic_capacity"] = self.semantic.capacity

        return stats

    def reset_working_memory(self, batch_size: int, device: torch.device):
        """Reset working memory."""
        if self.working is not None:
            return self.working.reset(batch_size, device)
        return torch.zeros(batch_size, self.latent_dim, device=device)
