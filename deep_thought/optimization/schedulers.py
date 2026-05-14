"""
Learning rate schedulers for Deep Thought training.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with warmup.
    
    Combines linear warmup with cosine annealing decay.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.warmup_steps > 0 and self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.min_lr + (base_lr - self.min_lr) *
                0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class ExponentialDecayScheduler(_LRScheduler):
    """
    Exponential decay with optional warmup.
    """
    
    def __init__(
        self,
        optimizer,
        decay_rate: float = 0.999,
        warmup_steps: int = 0,
        last_epoch: int = -1
    ):
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.warmup_steps > 0 and self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Exponential decay
            return [
                base_lr * (self.decay_rate ** (self.last_epoch - self.warmup_steps))
                for base_lr in self.base_lrs
            ]
