"""
Checkpoint utilities for Deep Thought.
"""

import torch
import os
from typing import Optional, Dict
from pathlib import Path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    path: str,
    extra_data: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        step: Current training step
        path: Path to save checkpoint
        extra_data: Additional data to save
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if extra_data is not None:
        checkpoint.update(extra_data)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load into (optional)
        device: Device to load to
        
    Returns:
        checkpoint: Loaded checkpoint data
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint
