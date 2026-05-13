"""
Temporal Cognition & Planning Layer (TCPL) for Deep Thought.

Orchestrates experts over time with hierarchical planning,
plan correction, and counterfactual simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from deep_thought.config import PlanningConfig
from deep_thought.architecture.planning.plan_memory import PlanMemory


@dataclass
class ExpertSchedule:
    """Schedule for expert activation over time."""
    expert_id: int
    start_step: int
    duration: int
    priority: float


@dataclass
class Plan:
    """A plan with expert sequence."""
    schedules: List[ExpertSchedule]
    expected_reward: float
    confidence: float


class TemporalPlanningLayer(nn.Module):
    """
    Temporal Cognition & Planning Layer (TCPL).
    
    Orchestrates experts over time through:
    - Latent timeline construction
    - Plan decomposition into expert sequences
    - Temporal routing
    - Execution controller
    - Plan correction loop
    - Hierarchical time scales
    - Counterfactual simulation
    """
    
    def __init__(self, config: PlanningConfig, num_experts: int = 128, 
                 latent_dim: Optional[int] = None, action_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_experts = num_experts
        self._latent_dim = latent_dim
        
        # Planning network - will be lazily initialized if latent_dim not provided
        if latent_dim is not None:
            self._init_networks(latent_dim, action_dim)
        else:
            self.planner = None
            self.controller = None
            self._networks_initialized = False

        # Goal compression network - fixed size
        self.goal_compressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        # Plan memory
        self.plan_memory = PlanMemory(config)
        
        # Current plan state
        self.current_plan: Optional[Plan] = None
        self.plan_step = 0
        self.controller_state = None
        
        self._action_dim = action_dim
    
    def _init_networks(self, latent_dim: int, action_dim: Optional[int] = None):
        """Initialize planner and controller networks."""
        self._latent_dim = latent_dim
        input_dim = latent_dim * 3  # h_t, x_t, m_t
        
        self.planner = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.planning_horizon * self.num_experts),
        )

        # Controller uses h_t, x_t, m_t concatenated (same as planner input)
        controller_input_dim = input_dim
        self.controller = nn.GRUCell(
            input_size=controller_input_dim,
            hidden_size=512
        )
        
        self._networks_initialized = True
    
    def _ensure_initialized(self, h_t: torch.Tensor):
        """Lazily initialize planner and controller once latent dim is known."""
        if self._networks_initialized:
            return
        latent_dim = h_t.size(-1)
        self._init_networks(latent_dim, self._action_dim)
        
        # Move to same device
        device = h_t.device
        self.planner = self.planner.to(device)
        self.controller = self.controller.to(device)
    
    def construct_timeline(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Construct latent timeline representation.
        
        Args:
            h_t: Hidden state
            x_t: Encoded observation
            m_t: Memory read
            context: Context embedding (ignored - planner uses fixed input dim)
            
        Returns:
            timeline: Compressed future trajectory
        """
        # Initialize planner and controller if needed
        self._ensure_initialized(h_t)

        # Concatenate inputs (context is intentionally not included to keep
        # input dimensions consistent with the planner network)
        combined = torch.cat([h_t, x_t, m_t], dim=-1)
        
        # Plan timeline
        timeline = self.planner(combined)
        
        # Reshape to [horizon, num_experts]
        horizon = self.config.planning_horizon
        timeline = timeline.view(-1, horizon, self.num_experts)
        
        return timeline
    
    def decompose_plan(
        self,
        timeline: torch.Tensor,
        k: int = 4
    ) -> List[ExpertSchedule]:
        """
        Decompose timeline into expert sequence.
        
        Args:
            timeline: Timeline from planner
            k: Number of experts per step
            
        Returns:
            schedules: List of expert schedules
        """
        horizon = timeline.size(1)
        schedules = []
        
        # Clamp k to actual number of experts available
        actual_k = min(k, timeline.size(-1))
        
        for t in range(horizon):
            # Get top-k experts for this timestep
            step_logits = timeline[:, t, :]
            top_k_vals, top_k_idx = torch.topk(step_logits, actual_k, dim=-1)
            
            for i in range(actual_k):
                expert_id = top_k_idx[0, i].item()
                priority = top_k_vals[0, i].item()
                
                schedule = ExpertSchedule(
                    expert_id=expert_id,
                    start_step=t,
                    duration=1,
                    priority=priority
                )
                schedules.append(schedule)
        
        return schedules
    
    def temporal_routing(
        self,
        timeline: torch.Tensor,
        k: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route experts over future horizon.
        
        Args:
            timeline: Timeline representation
            k: Number of experts per timestep
            
        Returns:
            gates: Gate values over horizon
            indices: Expert indices over horizon
        """
        horizon = timeline.size(1)
        
        # Clamp k to actual number of experts available
        actual_k = min(k, timeline.size(-1))
        
        # Get top-k per timestep
        gates = []
        indices = []
        
        for t in range(horizon):
            step_logits = timeline[:, t, :]
            top_k_vals, top_k_idx = torch.topk(step_logits, actual_k, dim=-1)
            
            # Normalize gates
            normalized_gates = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)
            
            gates.append(normalized_gates)
            indices.append(top_k_idx)
        
        gates = torch.stack(gates, dim=1)  # [B, horizon, k]
        indices = torch.stack(indices, dim=1)  # [B, horizon, k]
        
        return gates, indices
    
    def update_controller(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update execution controller state.
        
        Args:
            h_t: Hidden state
            x_t: Encoded observation
            m_t: Memory read
            action: Action taken (unused - controller uses same input as planner)
            
        Returns:
            controller_state: Updated controller state
        """
        # Ensure controller is initialized
        self._ensure_initialized(h_t)

        # Concatenate inputs (same as planner input)
        combined = torch.cat([h_t, x_t, m_t], dim=-1)
        
        # Update controller
        if self.controller_state is None:
            batch_size = h_t.size(0)
            device = h_t.device
            self.controller_state = torch.zeros(
                batch_size, 512, device=device
            )
        
        self.controller_state = self.controller(
            combined, self.controller_state
        )
        
        return self.controller_state
    
    def check_plan_deviation(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        threshold: float = 0.3
    ) -> bool:
        """
        Check if plan has deviated from reality.
        
        Args:
            predicted: Predicted state
            actual: Actual state
            threshold: Deviation threshold
            
        Returns:
            needs_correction: Whether plan needs correction
        """
        error = F.mse_loss(predicted, actual)
        return error > threshold
    
    def correct_plan(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Plan:
        """
        Generate corrected plan.
        
        Args:
            h_t: Current hidden state
            x_t: Encoded observation
            m_t: Memory read
            context: Context embedding
            
        Returns:
            corrected_plan: New plan
        """
        # Reconstruct timeline
        timeline = self.construct_timeline(h_t, x_t, m_t, context)
        
        # Decompose into schedules
        schedules = self.decompose_plan(timeline)
        
        # Create plan
        plan = Plan(
            schedules=schedules,
            expected_reward=0.0,  # Will be estimated
            confidence=0.5
        )
        
        self.current_plan = plan
        self.plan_step = 0
        
        return plan
    
    def compress_goals(
        self,
        rewards: torch.Tensor,
        horizon: int = 10
    ) -> torch.Tensor:
        """
        Compress reward sequence into goal vector.
        
        Args:
            rewards: Reward sequence
            horizon: Planning horizon
            
        Returns:
            goal: Compressed goal vector
        """
        # Pad or truncate to horizon
        if rewards.numel() < horizon:
            padded = F.pad(rewards.view(-1), (0, horizon - rewards.numel()))
        else:
            padded = rewards.view(-1)[:horizon]
        
        # Compress - pad to 1024 for goal_compressor input
        padded_1024 = F.pad(padded, (0, 1024 - padded.numel()))
        goal = self.goal_compressor(padded_1024.unsqueeze(0))
        
        return goal
    
    def simulate_counterfactual(
        self,
        h_t: torch.Tensor,
        plan: Plan,
        world_model,
        policy_fn,
        num_rollouts: int = 4
    ) -> float:
        """
        Simulate counterfactual rollouts to evaluate plan.
        
        Args:
            h_t: Current hidden state
            plan: Plan to evaluate
            world_model: World model for simulation
            policy_fn: Policy function
            num_rollouts: Number of rollouts
            
        Returns:
            expected_value: Expected value of plan
        """
        total_reward = 0.0
        
        for _ in range(num_rollouts):
            # Simulate following the plan
            z = h_t.clone()
            rollout_reward = 0.0
            
            for schedule in plan.schedules:
                # Get action from policy
                action = policy_fn(z)
                
                # Predict next state
                z_next, r_pred, _ = world_model(z, action)
                
                rollout_reward += r_pred.mean().item() if r_pred is not None else 0.0
                z = z_next
            
            total_reward += rollout_reward
        
        return total_reward / num_rollouts
    
    def get_current_experts(
        self,
        step: Optional[int] = None
    ) -> List[int]:
        """
        Get experts that should be active at current step.
        
        Args:
            step: Current step (uses plan_step if None)
            
        Returns:
            expert_ids: Active expert IDs
        """
        if self.current_plan is None:
            return []
        
        step = step if step is not None else self.plan_step
        
        active_experts = [
            s.expert_id for s in self.current_plan.schedules
            if s.start_step <= step < s.start_step + s.duration
        ]
        
        return active_experts
    
    def advance_step(self):
        """Advance plan by one step."""
        self.plan_step += 1
        
        # Check if plan is complete
        if self.current_plan is not None and self.current_plan.schedules:
            max_step = max(s.start_step + s.duration for s in self.current_plan.schedules)
            if self.plan_step >= max_step:
                self.current_plan = None
                self.plan_step = 0
    
    def reset_plan(self):
        """Reset current plan."""
        self.current_plan = None
        self.plan_step = 0
        self.controller_state = None
    
    def hierarchical_planning(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        m_t: torch.Tensor
    ) -> Dict[str, Plan]:
        """
        Generate plans at multiple time scales.
        
        Args:
            h_t: Hidden state
            x_t: Encoded observation
            m_t: Memory read
            
        Returns:
            plans: Dictionary of plans at different scales
        """
        plans = {}
        
        # Micro plan (immediate actions)
        micro_timeline = self.construct_timeline(h_t, x_t, m_t)
        micro_schedules = self.decompose_plan(micro_timeline, k=2)
        plans["micro"] = Plan(
            schedules=micro_schedules[:self.config.micro_horizon],
            expected_reward=0.0,
            confidence=0.8
        )
        
        # Tactical plan (seconds to minutes)
        tactical_schedules = self.decompose_plan(micro_timeline, k=4)
        plans["tactical"] = Plan(
            schedules=tactical_schedules[:self.config.tactical_horizon],
            expected_reward=0.0,
            confidence=0.6
        )
        
        # Strategic plan (episode-long)
        strategic_schedules = self.decompose_plan(micro_timeline, k=8)
        plans["strategic"] = Plan(
            schedules=strategic_schedules[:self.config.strategic_horizon],
            expected_reward=0.0,
            confidence=0.4
        )
        
        return plans
