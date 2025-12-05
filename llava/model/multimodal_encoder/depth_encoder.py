import warnings
from typing import Optional

import torch
import torch.nn as nn


class DepthFeatureEncoder(nn.Module):
    """Encode Depth Anything v2 maps into transformer-friendly tokens."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 1024,
        intermediate_channels: int = 256,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = []
        ch = in_channels
        for out_ch in [64, 128, intermediate_channels]:
            conv = nn.Conv2d(ch, out_ch, kernel_size=3, stride=2, padding=1, bias=not use_batch_norm)
            layers.append(conv)
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            ch = out_ch

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intermediate_channels, hidden_size),
        )

    def forward(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """Convert ``(B, C, H, W)`` depth maps to ``(B, 1, hidden_size)`` tokens."""

        if depth_maps.dim() == 3:
            depth_maps = depth_maps.unsqueeze(1)
        if depth_maps.dim() != 4:
            raise ValueError(f"Expected depth maps of shape (B, 1, H, W) but received {depth_maps.shape}.")

        feats = self.backbone(depth_maps)
        pooled = self.pool(feats)
        token = self.head(pooled).unsqueeze(1)
        return token

