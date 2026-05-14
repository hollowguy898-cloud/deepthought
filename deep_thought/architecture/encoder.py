"""
Encoder module for Deep Thought.

Compresses observations into factorized latent space with orthogonal subspaces
to maximize information density without representational collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from deep_thought.config import EncoderConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5 + self.eps)
        return self.weight * x / (norm + self.eps)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit."""
    
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


class FactorizedLatent(nn.Module):
    """
    Factorized latent representation with orthogonal subspaces.
    
    Splits latent into specialized subspaces to prevent feature interference
    and maximize information density.
    """
    
    def __init__(self, latent_dim: int, num_subspaces: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces
        self.subspace_dim = latent_dim // num_subspaces
        self.output_dim = self.subspace_dim * num_subspaces  # May be < latent_dim if not evenly divisible
        
        # Subspace projections
        self.projections = nn.ModuleList([
            nn.Linear(latent_dim, self.subspace_dim, bias=False)
            for _ in range(num_subspaces)
        ])
        
        # Orthogonality regularization
        self.register_buffer(
            "orthogonality_target",
            torch.eye(num_subspaces)
        )
        
        # Project back to full latent_dim if needed
        if self.output_dim != latent_dim:
            self.output_proj = nn.Linear(self.output_dim, latent_dim, bias=False)
        else:
            self.output_proj = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode into factorized subspaces.
        
        Returns:
            h: Full latent vector
            subspaces: Tuple of subspace vectors
        """
        subspaces = []
        for proj in self.projections:
            subspaces.append(proj(x))
        
        h = torch.cat(subspaces, dim=-1)
        
        # Project back to latent_dim if dimensions don't match
        if self.output_proj is not None:
            h = self.output_proj(h)
        
        return h, tuple(subspaces)
    
    def orthogonality_loss(self, subspaces: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute orthogonality constraint loss."""
        num = len(subspaces)
        loss = 0.0
        for i in range(num):
            for j in range(i + 1, num):
                # Compute correlation between subspaces
                corr = torch.abs(
                    (subspaces[i] * subspaces[j]).sum(dim=-1).mean()
                )
                loss += corr
        return loss / (num * (num - 1) / 2)


class SparseCoding(nn.Module):
    """
    Sparse coding layer for compressed representation.
    
    Enforces sparsity in latent activations to create interpretable
    compressed representations.
    """
    
    def __init__(self, dim: int, sparsity_coef: float = 0.01):
        super().__init__()
        self.dim = dim
        self.sparsity_coef = sparsity_coef
        self.encoder = nn.Linear(dim, dim * 2, bias=False)
        self.decoder = nn.Linear(dim * 2, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with sparse activation."""
        z = self.encoder(x)
        # Apply sparsity via k-winners
        k = max(1, int(z.size(-1) * 0.1))  # Top 10% activate
        topk_vals, topk_idx = torch.topk(z.abs(), k, dim=-1)
        sparse_z = torch.zeros_like(z)
        sparse_z.scatter_(-1, topk_idx, z.gather(-1, topk_idx))

        # Straight-through estimator: allow a small residual gradient
        # for non-winning features so they can recover from "dead" state.
        # During forward pass sparse_z is unchanged; during backward pass
        # gradients flow through z with a small residual scale.
        residual_scale = 0.01
        sparse_z = sparse_z + (z - z.detach()) * residual_scale

        h = self.decoder(sparse_z)
        return h, sparse_z
    
    def sparsity_loss(self, sparse_z: torch.Tensor) -> torch.Tensor:
        """L1 sparsity loss."""
        return self.sparsity_coef * sparse_z.abs().sum(dim=-1).mean()


class Encoder(nn.Module):
    """
    Main encoder for Deep Thought.
    
    Compresses observations into factorized latent space with:
    - Multi-layer projection
    - Factorized subspaces
    - Sparse coding
    - Orthogonality constraints
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_subspaces = 5
        
        # Input projection - observation_dim must be set before creating encoder
        if config.observation_dim is None:
            raise ValueError(
                "EncoderConfig.observation_dim must be set before creating Encoder. "
                "Set it via config.encoder.observation_dim = env.observation_space.shape[0]"
            )
        
        layers = []
        input_dim = config.observation_dim
        
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            if config.activation == "silu":
                layers.append(nn.SiLU())
            elif config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.SiLU())
            
            if config.use_layer_norm:
                layers.append(RMSNorm(config.hidden_dim))
            
            input_dim = config.hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Project from hidden_dim to latent_dim if they differ
        if config.hidden_dim != config.latent_dim:
            self.hidden_to_latent = nn.Linear(config.hidden_dim, config.latent_dim)
        else:
            self.hidden_to_latent = nn.Identity()
        
        # Factorized latent
        self.factorized = FactorizedLatent(
            config.latent_dim,
            self.num_subspaces
        )
        
        # Sparse coding
        self.sparse_coding = SparseCoding(config.latent_dim)
        
        # Final projection
        self.output = nn.Linear(config.latent_dim, config.latent_dim)
    
    def forward(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Encode observation into latent representation.
        
        Args:
            observation: Input observation tensor
            
        Returns:
            latent: Compressed latent representation
            info: Dictionary with intermediate representations
        """
        # Initial encoding
        h = self.encoder(observation)
        
        # Project to latent dim
        h = self.hidden_to_latent(h)
        
        # Factorized subspaces
        h_factorized, subspaces = self.factorized(h)
        
        # Sparse coding
        h_sparse, sparse_z = self.sparse_coding(h_factorized)
        
        # Final output
        latent = self.output(h_sparse)
        
        info = {
            "subspaces": subspaces,
            "sparse_z": sparse_z,
            "h_factorized": h_factorized,
        }
        
        return latent, info
    
    def compute_losses(self, info: dict) -> dict:
        """Compute encoder regularization losses."""
        losses = {}
        
        # Orthogonality loss
        orth_loss = self.factorized.orthogonality_loss(info["subspaces"])
        losses["orthogonality"] = orth_loss
        
        # Sparsity loss
        sparse_loss = self.sparse_coding.sparsity_loss(info["sparse_z"])
        losses["sparsity"] = sparse_loss
        
        return losses


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for image observations.
    
    Uses ResNet-style blocks with attention pooling.
    """
    
    def __init__(
        self,
        image_size: int = 84,
        latent_dim: int = 1024,
        channels: int = 3
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Convolutional stem
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Compute flattened size
        conv_out_size = self._get_conv_output_size()
        
        # Projection to latent
        self.fc = nn.Linear(conv_out_size, latent_dim)
        self.norm = RMSNorm(latent_dim)
    
    def _get_conv_output_size(self) -> int:
        """Compute output size after convolutions."""
        x = torch.zeros(1, 3, self.image_size, self.image_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.norm(x)
        return x


def create_encoder(config: EncoderConfig) -> nn.Module:
    """Factory function to create appropriate encoder."""
    if config.use_conv and config.image_size is not None:
        return ConvEncoder(
            image_size=config.image_size,
            latent_dim=config.latent_dim,
            channels=3  # Assuming RGB
        )
    else:
        return Encoder(config)
