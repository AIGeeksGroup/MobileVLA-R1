import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def normalize_nav_path(raw_path: str, override_root: Optional[str] = None) -> str:
    """Normalize MP3D-style Windows paths to the current platform."""

    path = Path(str(raw_path).replace("\\", "/"))
    if override_root is None:
        return str(path)

    relative = path
    if "scans" in path.parts:
        idx = path.parts.index("scans")
        relative = Path(*path.parts[idx + 1 :])
    elif len(path.parts) > 1:
        relative = Path(*path.parts[1:])
    return str(Path(override_root) / relative)


def load_depth_map(depth_path: str, scale: float = 1000.0) -> torch.Tensor:
    """Load a depth map stored as png/npy/pt and return a float32 tensor."""

    suffix = Path(depth_path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        depth = np.array(Image.open(depth_path)).astype(np.float32)
        depth = depth / scale if scale > 0 else depth
    elif suffix == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    elif suffix == ".pt":
        depth = torch.load(depth_path).float().cpu().numpy()
    else:
        raise ValueError(f"Unsupported depth file extension: {suffix}")

    depth = torch.from_numpy(depth)
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    return depth.unsqueeze(0)  # (1, 1, H, W)


def depth_to_point_cloud(
    depth_tensor: torch.Tensor,
    max_points: int = 2048,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a ``(1, 1, H, W)`` depth map to a ``(max_points, 3)`` point cloud."""

    if depth_tensor.dim() != 4:
        raise ValueError("Depth tensor must have shape (B, 1, H, W).")

    _, _, height, width = depth_tensor.shape
    depth = depth_tensor[0, 0]

    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=depth.device),
        torch.linspace(-1.0, 1.0, width, device=depth.device),
        indexing="ij",
    )
    z = depth
    x = xs * z
    y = ys * z
    points = torch.stack([x, y, z], dim=-1).view(-1, 3)

    valid_mask = torch.isfinite(points).all(dim=-1)
    points = points[valid_mask]
    if points.shape[0] == 0:
        points = torch.zeros(1, 3, device=depth.device)

    if points.shape[0] > max_points:
        idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
        points = points[idx]
    elif points.shape[0] < max_points:
        pad = max_points - points.shape[0]
        points = torch.cat([points, torch.zeros(pad, 3, device=points.device)], dim=0)

    if normalize:
        mean = points.mean(dim=0, keepdim=True)
        std = points.std(dim=0, keepdim=True).clamp_min(1e-6)
        points = (points - mean) / std

    return points

