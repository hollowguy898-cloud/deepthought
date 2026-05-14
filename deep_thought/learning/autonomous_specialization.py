"""
Autonomous Expert Specialization for Deep Thought.

With a Black Box layer (MDE), the Feature → Expert Compiler (FEC) becomes
much more powerful:

1. Targeted Growth: When the Dissection Layer (MDE) identifies a
   high-confidence pattern that doesn't match any existing experts, it
   triggers Neurogenesis to spawn a "Specialist Expert" for that specific
   latent pattern.

2. Pruning via Contradiction: If the Dissection Layer realizes a
   previously identified "mechanic" was actually just noise, it signals
   the Synaptic Pruning module to dissolve the expert that was trained
   on that false premise.

This module bridges the MDE's invariant discovery with the Expert Bank's
lifecycle management, ensuring that expert growth and pruning are driven
by discovered mechanics rather than arbitrary thresholds.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from deep_thought.config import AutonomousSpecializationConfig


@dataclass
class SpecialistRecord:
    """Record of a specialist expert created for a specific mechanic.

    Attributes:
        expert_id: ID of the specialist expert in the ExpertBank.
        mechanic_tag_id: Tag ID of the mechanic that triggered growth.
        invariant_id: ID of the invariant that this expert specializes in.
        creation_step: Step when the specialist was created.
        parent_expert_id: ID of the parent expert that was cloned.
        validation_score: Latest validation score for this specialist.
        is_active: Whether the specialist is still active.
    """
    expert_id: int
    mechanic_tag_id: str
    invariant_id: int
    creation_step: int
    parent_expert_id: int
    validation_score: float = 0.0
    is_active: bool = True


class AutonomousSpecialization:
    """Bridges MDE discoveries with Expert Bank lifecycle management.

    When the MDE discovers a high-confidence invariant that has no
    matching expert, this module triggers targeted growth (Neurogenesis).
    When an invariant is contradicted, this module triggers pruning of
    the associated specialist expert.

    This module is NOT an nn.Module — it is a pure algorithmic component.
    """

    def __init__(self, config: AutonomousSpecializationConfig):
        self.config = config
        # Track specialist experts: mechanic_tag_id -> List[SpecialistRecord]
        self._specialists: Dict[str, List[SpecialistRecord]] = {}
        # Track how many specialists per mechanic
        self._specialist_count: Dict[str, int] = {}
        # Track contradicted mechanics whose specialists should be pruned
        self._contradicted_mechanics: List[str] = []
        # Step counter
        self._step: int = 0
        # Statistics
        self._total_specialists_created: int = 0
        self._total_specialists_pruned: int = 0

    def should_trigger_growth(
        self,
        mechanic_tag_id: str,
        invariant_confidence: float,
        invariant_context_span: int,
        has_matching_expert: bool,
    ) -> bool:
        """Determine whether targeted growth should be triggered.

        Growth is triggered when:
        1. The MDE has discovered a high-confidence invariant.
        2. No existing expert specializes in this invariant.
        3. The invariant spans enough contexts.
        4. We haven't already created too many specialists for this mechanic.

        Args:
            mechanic_tag_id: Tag ID of the discovered mechanic.
            invariant_confidence: Confidence score of the invariant.
            invariant_context_span: Number of contexts the invariant spans.
            has_matching_expert: Whether an expert already handles this
                mechanic.

        Returns:
            True if targeted growth should be triggered.
        """
        if not self.config.use_autonomous_specialization:
            return False

        if has_matching_expert:
            return False  # Already have an expert for this

        if invariant_confidence < self.config.growth_confidence_threshold:
            return False  # Not confident enough

        if invariant_context_span < self.config.growth_context_span_min:
            return False  # Not enough context evidence

        # Check specialist cap per mechanic
        current_count = self._specialist_count.get(mechanic_tag_id, 0)
        if current_count >= self.config.max_specialists_per_mechanic:
            return False  # Already at cap

        return True

    def record_specialist(
        self,
        expert_id: int,
        mechanic_tag_id: str,
        invariant_id: int,
        parent_expert_id: int,
    ) -> None:
        """Record that a specialist expert was created.

        Args:
            expert_id: ID of the new specialist expert.
            mechanic_tag_id: Tag ID of the mechanic.
            invariant_id: ID of the invariant.
            parent_expert_id: ID of the parent expert that was cloned.
        """
        record = SpecialistRecord(
            expert_id=expert_id,
            mechanic_tag_id=mechanic_tag_id,
            invariant_id=invariant_id,
            creation_step=self._step,
            parent_expert_id=parent_expert_id,
        )

        if mechanic_tag_id not in self._specialists:
            self._specialists[mechanic_tag_id] = []
        self._specialists[mechanic_tag_id].append(record)

        self._specialist_count[mechanic_tag_id] = (
            self._specialist_count.get(mechanic_tag_id, 0) + 1
        )
        self._total_specialists_created += 1

    def handle_contradiction(
        self,
        mechanic_tag_id: str,
    ) -> List[int]:
        """Handle a contradicted mechanic by marking its specialists for pruning.

        When the MDE determines that a mechanic was actually noise (its
        validation score dropped below the contradiction threshold), this
        method identifies the specialist experts that were created for
        that mechanic and marks them for pruning.

        Args:
            mechanic_tag_id: Tag ID of the contradicted mechanic.

        Returns:
            List of expert IDs that should be pruned.
        """
        experts_to_prune: List[int] = []

        if mechanic_tag_id not in self._specialists:
            return experts_to_prune

        for record in self._specialists[mechanic_tag_id]:
            if record.is_active:
                record.is_active = False
                record.validation_score = 0.0
                experts_to_prune.append(record.expert_id)
                self._total_specialists_pruned += 1

        self._contradicted_mechanics.append(mechanic_tag_id)
        return experts_to_prune

    def update_specialist_validation(
        self,
        expert_id: int,
        mechanic_tag_id: str,
        validation_score: float,
    ) -> bool:
        """Update a specialist's validation score.

        If the validation score drops below the contradiction threshold,
        the specialist is marked for pruning.

        Args:
            expert_id: ID of the specialist expert.
            mechanic_tag_id: Tag ID of the mechanic.
            validation_score: Latest validation score.

        Returns:
            True if the specialist should be pruned.
        """
        if mechanic_tag_id not in self._specialists:
            return False

        for record in self._specialists[mechanic_tag_id]:
            if record.expert_id == expert_id:
                record.validation_score = validation_score
                if validation_score < self.config.contradiction_prune_threshold:
                    record.is_active = False
                    self._total_specialists_pruned += 1
                    return True
                break

        return False

    def get_specialists_for_mechanic(
        self, mechanic_tag_id: str
    ) -> List[SpecialistRecord]:
        """Get all specialist records for a given mechanic.

        Args:
            mechanic_tag_id: Tag ID of the mechanic.

        Returns:
            List of SpecialistRecord instances.
        """
        return self._specialists.get(mechanic_tag_id, [])

    def get_active_specialist_ids(self) -> List[int]:
        """Get IDs of all currently active specialist experts.

        Returns:
            List of expert IDs.
        """
        active_ids: List[int] = []
        for records in self._specialists.values():
            for record in records:
                if record.is_active:
                    active_ids.append(record.expert_id)
        return active_ids

    def has_specialist_for_mechanic(self, mechanic_tag_id: str) -> bool:
        """Check if there's an active specialist for a mechanic.

        Args:
            mechanic_tag_id: Tag ID of the mechanic.

        Returns:
            True if at least one active specialist exists.
        """
        records = self._specialists.get(mechanic_tag_id, [])
        return any(r.is_active for r in records)

    def tick(self):
        """Advance the step counter."""
        self._step += 1

    def get_stats(self) -> Dict:
        """Return specialization statistics."""
        return {
            "total_specialists_created": self._total_specialists_created,
            "total_specialists_pruned": self._total_specialists_pruned,
            "active_specialists": len(self.get_active_specialist_ids()),
            "contradicted_mechanics": len(self._contradicted_mechanics),
            "mechanics_with_specialists": len(self._specialists),
            "step": self._step,
        }

    def reset(self):
        """Reset all state."""
        self._specialists.clear()
        self._specialist_count.clear()
        self._contradicted_mechanics.clear()
        self._step = 0
        self._total_specialists_created = 0
        self._total_specialists_pruned = 0
