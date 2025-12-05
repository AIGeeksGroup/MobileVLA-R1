from typing import Optional

import torch
import torch.nn as nn


class PointTransformerEncoder(nn.Module):
    """Lightweight Point Transformer style encoder for Nav-CoT point clouds.

    The implementation intentionally keeps the dependency footprint small by
    reusing ``nn.TransformerEncoder`` while injecting coordinate-aware
    positional encodings that mimic Point Transformer blocks. The module
    outputs a sequence of length 1 (global token) that can be consumed by the
    multimodal projector.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode a batch of point clouds.

        Args:
            points: ``(B, N, 3)`` coordinates in metric space.
            features: Optional per-point extra channels ``(B, N, C)``. When
                provided, the coordinates are concatenated with these features
                before the input projection.

        Returns:
            Tensor of shape ``(B, 1, hidden_size)`` representing a global token.
        """

        if points.dim() != 3 or points.size(-1) != 3:
            raise ValueError(f"Expected points of shape (B, N, 3) but received {points.shape}.")

        coords = points
        if features is not None:
            if features.shape[:2] != points.shape[:2]:
                raise ValueError("Point features must share the first two dimensions with coordinates.")
            tokens = torch.cat([coords, features], dim=-1)
        else:
            tokens = coords

        if tokens.size(-1) != self.in_channels:
            # Pad or crop to match the expected dimensionality.
            if tokens.size(-1) < self.in_channels:
                pad_dim = self.in_channels - tokens.size(-1)
                tokens = torch.nn.functional.pad(tokens, (0, pad_dim))
            else:
                tokens = tokens[..., : self.in_channels]

        token_embeddings = self.input_proj(tokens) + self.pos_mlp(coords)
        encoded = self.encoder(token_embeddings)
        pooled = self.norm(encoded.mean(dim=1, keepdim=True))
        return pooled

