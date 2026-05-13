"""
Expert bank module for Deep Thought.

Implements sparse mixture-of-experts with specialized MLP modules,
utility tracking, and lifecycle management (active, dormant, dead).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import copy

from deep_thought.config import ExpertConfig


class ExpertState(Enum):
    """Expert lifecycle states."""
    ACTIVE = "active"
    COOLING = "cooling"
    DORMANT = "dormant"
    CANDIDATE_DELETE = "candidate_delete"
    DEAD = "dead"


@dataclass
class ExpertStats:
    """Statistics for tracking expert utility."""
    activation_count: int = 0
    gradient_norm: float = 0.0
    reward_contribution: float = 0.0
    compute_cost: float = 1.0
    dormancy_age: int = 0
    utility_score: float = 0.0
    state: ExpertState = ExpertState.ACTIVE


class Expert(nn.Module):
    """
    Specialized MLP expert module.
    
    Each expert is a compact MLP that learns a narrow skill.
    Uses SwiGLU activation for efficient computation.
    """
    
    def __init__(self, config: ExpertConfig, expert_id: int, latent_dim: int = 1024):
        super().__init__()
        self.config = config
        self.expert_id = expert_id
        self.latent_dim = latent_dim
        
        # Expert network
        layers = []
        input_dim = latent_dim
        
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            if config.activation == "swiglu":
                layers.append(SwiGLU(config.hidden_dim))
            elif config.activation == "silu":
                layers.append(nn.SiLU())
                layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            else:
                layers.append(nn.ReLU())
                layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            input_dim = config.hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output projection
        self.output = nn.Linear(config.hidden_dim, latent_dim)
        
        # Stats
        self.stats = ExpertStats()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply expert transformation."""
        h = self.network(x)
        out = self.output(h)
        
        if self.config.use_residual:
            out = out + x
        
        return out
    
    def clone(self, new_id: int, noise_scale: float = 0.01) -> "Expert":
        """Clone expert with small noise for neurogenesis."""
        new_expert = Expert(self.config, new_id, self.latent_dim)
        new_expert.load_state_dict(self.state_dict())
        
        # Add noise to weights
        with torch.no_grad():
            for param in new_expert.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
        
        return new_expert


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate(x))
        value = self.value(x)
        return self.output(gate * value)


class ExpertBank(nn.Module):
    """
    Bank of sparse experts with lifecycle management.
    
    Manages:
    - Expert activation and routing
    - Utility tracking
    - Lifecycle states (active, dormant, dead)
    - Pruning and growth
    - LEVER 3: Hard expert cap (max_experts budget)
    - LEVER 4: Dormant expert offloading (compress to float16, free VRAM)
    
    Uses a mapping dict (expert_id -> position_in_modulelist) to maintain
    consistent indexing even after pruning operations.
    """
    
    def __init__(self, config: ExpertConfig, num_experts: int = 128, latent_dim: int = 1024,
                 max_experts: int = 256):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        # LEVER 3: Hard cap on total expert count (active + dormant + cached)
        self.max_experts = max_experts
        
        # Create experts - each expert has a unique ID that persists
        self.experts = nn.ModuleDict({
            str(i): Expert(config, i, latent_dim) for i in range(num_experts)
        })
        
        # Expert statistics keyed by expert ID
        self.expert_stats: Dict[int, ExpertStats] = {
            i: ExpertStats() for i in range(num_experts)
        }
        
        # LEVER 4: Dormant cache - stores compressed (float16) weights of
        # offloaded dormant experts.  When an expert goes dormant, its weights
        # are quantized to float16 and stored here, then removed from the
        # ModuleDict to free VRAM.  This means dormant experts contribute
        # ~zero compute tax instead of sitting in memory unused.
        self.dormant_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Utility computation parameters
        self.alpha = 0.3  # Activation weight
        self.beta = 0.3   # Gradient weight
        self.gamma = 0.3  # Reward weight
        self.delta = 0.1  # Cost weight
        self.eta = 0.1    # Dormancy weight
        
        # EMA for utility smoothing
        self.utility_ema = 0.99
    
    def _get_expert(self, expert_id: int) -> Optional[Expert]:
        """Get expert by ID, returns None if not found."""
        key = str(expert_id)
        if key in self.experts:
            return self.experts[key]
        return None
    
    def forward(
        self,
        h_t: torch.Tensor,
        selected_indices: torch.Tensor,
        gates: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Apply selected experts to hidden state.
        
        Args:
            h_t: Hidden state
            selected_indices: Indices of selected experts
            gates: Gate values for selected experts
            
        Returns:
            delta_h: Expert contributions
            compute_costs: Compute cost per expert
        """
        batch_size = h_t.size(0)
        delta_h = torch.zeros_like(h_t)
        compute_costs = {}
        
        # Apply each selected expert
        num_active = selected_indices.size(-1)  # Number of active experts per sample
        for i in range(num_active):
            expert_idx = selected_indices[:, i]
            gate = gates[:, i:i+1]

            # Get unique experts in batch
            unique_experts = expert_idx.unique()

            for exp_id_tensor in unique_experts:
                exp_id = exp_id_tensor.item()
                expert = self._get_expert(exp_id)
                if expert is None:
                    continue

                # Check if expert is active
                if exp_id not in self.expert_stats:
                    continue
                if self.expert_stats[exp_id].state != ExpertState.ACTIVE:
                    continue

                # Mask for this expert
                mask = (expert_idx == exp_id).unsqueeze(-1)

                # Apply expert
                expert_output = expert(h_t)
                delta_h = delta_h + mask * gate * expert_output

                # Track compute cost
                compute_costs[exp_id] = mask.sum().item()

                # Update stats
                self.expert_stats[exp_id].activation_count += mask.sum().item()
        
        return delta_h, compute_costs
    
    def update_utility(
        self,
        gradient_norms: Dict[int, float],
        reward_contributions: Dict[int, float]
    ):
        """
        Update utility scores for all experts.
        
        Args:
            gradient_norms: Gradient norm per expert
            reward_contributions: Reward contribution per expert
        """
        for exp_id, stats in self.expert_stats.items():
            if stats.state == ExpertState.DEAD:
                continue
            
            # Normalize components
            act_norm = stats.activation_count / (stats.activation_count + 1)
            grad_norm = gradient_norms.get(exp_id, 0.0)
            rew_norm = reward_contributions.get(exp_id, 0.0)
            cost_norm = stats.compute_cost
            dorm_norm = stats.dormancy_age / 1000.0
            
            # Compute utility
            utility = (
                self.alpha * act_norm +
                self.beta * grad_norm +
                self.gamma * rew_norm -
                self.delta * cost_norm -
                self.eta * dorm_norm
            )
            
            # EMA smooth
            stats.utility_score = (
                self.utility_ema * stats.utility_score +
                (1 - self.utility_ema) * utility
            )
            
            # Reset counters
            stats.activation_count = 0
    
    def mark_dormant(self, threshold: float = 0.15):
        """Mark low-utility experts as dormant and offload them (LEVER 4)."""
        for exp_id, stats in self.expert_stats.items():
            if stats.utility_score < threshold:
                if stats.state == ExpertState.ACTIVE:
                    stats.state = ExpertState.DORMANT
                    stats.dormancy_age = 0
                    # LEVER 4: Offload dormant expert to save VRAM
                    self._offload_dormant_expert(exp_id)
    
    def mark_dead(self, threshold: float = 0.05, confirmation_steps: int = 1000000):
        """Mark long-dormant experts as dead."""
        for exp_id, stats in self.expert_stats.items():
            if stats.state == ExpertState.DORMANT:
                stats.dormancy_age += 1
                if stats.dormancy_age > confirmation_steps and \
                   stats.utility_score < threshold:
                    stats.state = ExpertState.DEAD
    
    def reactivate(self, expert_ids: List[int]):
        """Reactivate dormant experts, restoring from cache if needed (LEVER 4)."""
        for exp_id in expert_ids:
            if exp_id in self.expert_stats:
                # LEVER 4: Check if expert is in dormant cache
                if exp_id in self.dormant_cache:
                    self._reactivate_from_cache(exp_id)
                self.expert_stats[exp_id].state = ExpertState.ACTIVE
                self.expert_stats[exp_id].dormancy_age = 0
    
    def prune_dead_experts(self) -> List[int]:
        """Remove dead experts and return their IDs."""
        dead_ids = [
            exp_id for exp_id, stats in self.expert_stats.items()
            if stats.state == ExpertState.DEAD
        ]
        
        # Remove from module dict
        for exp_id in dead_ids:
            key = str(exp_id)
            if key in self.experts:
                del self.experts[key]
        
        # Update stats
        for exp_id in dead_ids:
            del self.expert_stats[exp_id]
        
        return dead_ids
    
    def grow_expert(
        self,
        parent_id: Optional[int] = None,
        noise_scale: float = 0.01
    ) -> Optional[int]:
        """
        Grow a new expert.
        
        LEVER 3: Enforces hard cap on expert count.  If the total number
        of experts (active + dormant + cached) is at or above max_experts,
        growth is refused and None is returned.
        
        Args:
            parent_id: Expert to clone (if None, clone best)
            noise_scale: Noise for initialization
            
        Returns:
            New expert ID, or None if growth is denied by the cap.
        """
        # LEVER 3: Hard budget check
        total_experts = len(self.experts) + len(self.dormant_cache)
        if total_experts >= self.max_experts:
            return None  # Growth denied - at capacity
        
        # Find best expert if no parent specified
        if parent_id is None:
            active_stats = {k: v for k, v in self.expert_stats.items()
                            if v.state == ExpertState.ACTIVE}
            if not active_stats:
                return None  # No active experts to clone from
            parent_id = max(active_stats.items(), key=lambda x: x[1].utility_score)[0]
        
        # Generate new unique ID
        all_ids = set(self.expert_stats.keys())
        new_id = max(all_ids) + 1 if all_ids else 0
        
        # Clone parent
        parent_expert = self._get_expert(parent_id)
        if parent_expert is not None:
            new_expert = parent_expert.clone(new_id, noise_scale)
        else:
            new_expert = Expert(self.config, new_id, self.latent_dim)
        
        self.experts[str(new_id)] = new_expert
        self.expert_stats[new_id] = ExpertStats(state=ExpertState.ACTIVE)
        
        return new_id
    
    def get_active_experts(self) -> List[int]:
        """Get list of active expert IDs."""
        return [
            exp_id for exp_id, stats in self.expert_stats.items()
            if stats.state == ExpertState.ACTIVE
        ]
    
    def get_dormant_experts(self) -> List[int]:
        """Get list of dormant expert IDs."""
        return [
            exp_id for exp_id, stats in self.expert_stats.items()
            if stats.state == ExpertState.DORMANT
        ]
    
    def compute_total_cost(self) -> float:
        """Compute total compute cost of active experts."""
        return sum(
            stats.compute_cost
            for stats in self.expert_stats.values()
            if stats.state == ExpertState.ACTIVE
        )
    
    def split_expert(self, expert_id: int, variance_threshold: float = 0.5):
        """
        Split an expert if it has high variance (overloaded).
        
        Args:
            expert_id: Expert to split
            variance_threshold: Variance threshold for splitting
        """
        if expert_id not in self.expert_stats:
            return
        
        # Create two children
        child1_id = self.grow_expert(expert_id, noise_scale=0.005)
        child2_id = self.grow_expert(expert_id, noise_scale=-0.005)
        
        # Mark parent as cooling
        self.expert_stats[expert_id].state = ExpertState.COOLING
    
    def merge_experts(self, expert_id1: int, expert_id2: int, 
                      distance_threshold: float = 0.1):
        """
        Merge two similar experts.
        
        Args:
            expert_id1: First expert
            expert_id2: Second expert
            distance_threshold: Distance threshold for merging
        """
        if expert_id1 not in self.expert_stats or expert_id2 not in self.expert_stats:
            return
        
        expert1 = self._get_expert(expert_id1)
        expert2 = self._get_expert(expert_id2)
        if expert1 is None or expert2 is None:
            return
        
        # Average weights
        with torch.no_grad():
            for p1, p2 in zip(expert1.parameters(), expert2.parameters()):
                p1.data = (p1.data + p2.data) / 2
        
        # Mark second as dead
        self.expert_stats[expert_id2].state = ExpertState.DEAD
    
    def __len__(self):
        """Return total number of experts (active + cached)."""
        return len(self.experts) + len(self.dormant_cache)
    
    # ----------------------------------------------------------------
    # LEVER 3: Capability Density metric
    # ----------------------------------------------------------------
    def capability_density(self) -> float:
        """Compute capability density = mean_utility / total_param_count.
        
        This metric rewards the system for being efficient — high performance
        with few parameters.  A low density means the model is wasting capacity
        on experts that aren't contributing proportionally to their cost.
        
        Returns:
            Capability density (higher is better).
        """
        active_experts = self.get_active_experts()
        if not active_experts:
            return 0.0
        mean_utility = sum(
            self.expert_stats[eid].utility_score for eid in active_experts
        ) / len(active_experts)
        total_params = sum(
            sum(p.numel() for p in self.experts[str(eid)].parameters())
            for eid in active_experts
            if str(eid) in self.experts
        )
        if total_params == 0:
            return 0.0
        return mean_utility / (total_params / 1e6)  # Normalize to millions of params
    
    # ----------------------------------------------------------------
    # LEVER 4: Dormant expert offloading / compression
    # ----------------------------------------------------------------
    def _offload_dormant_expert(self, expert_id: int):
        """Compress a dormant expert's weights to float16 and offload from VRAM.
        
        Instead of keeping dormant experts in the ModuleDict where they
        consume VRAM and incur a compute tax, this method:
        1. Extracts all weight tensors
        2. Compresses them to float16 (2x memory savings)
        3. Stores them in dormant_cache (a plain dict, not a ModuleDict)
        4. Removes the expert from self.experts ModuleDict
        
        The expert can be restored later via _reactivate_from_cache().
        """
        key = str(expert_id)
        if key not in self.experts:
            return  # Already offloaded or doesn't exist
        
        expert = self.experts[key]
        compressed = {}
        with torch.no_grad():
            for name, param in expert.named_parameters():
                compressed[name] = param.data.cpu().half()  # float16 compression
        
        self.dormant_cache[expert_id] = compressed
        del self.experts[key]  # Remove from ModuleDict to free VRAM
    
    def _reactivate_from_cache(self, expert_id: int):
        """Restore a cached dormant expert from float16 back to float32.
        
        Decompresses the expert's weights and adds it back to the
        ModuleDict so it can participate in forward passes again.
        """
        if expert_id not in self.dormant_cache:
            return  # Not in cache
        
        compressed = self.dormant_cache.pop(expert_id)
        
        # Create a fresh expert and restore weights
        expert = Expert(self.config, expert_id, self.latent_dim)
        with torch.no_grad():
            for name, param in expert.named_parameters():
                if name in compressed:
                    param.data = compressed[name].float().to(param.device)
        
        self.experts[str(expert_id)] = expert
