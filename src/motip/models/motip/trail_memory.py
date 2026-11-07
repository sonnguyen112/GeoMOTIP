# Trail-MOTIP V0
# Based on the MOTIP research codebase (Apache-2.0).
# New Trail-MOTIP implementation code in this file.

from __future__ import annotations

import torch
import torch.nn as nn


class BBoxTrailEncoder(nn.Module):
    """Encode a decayed per-track bounding-box trail into a feature vector.

    V0 deliberately uses only bounding boxes, not RGB/ROI features. The goal is
    to isolate the contribution of an explicit spatial-temporal motion trail.

    Expected MOTIP tensor layout:
        trajectory_boxes: [B, G, T, N, 4], normalized cx, cy, w, h
        trajectory_masks: [B, G, T, N], True means padded/missing

    The trail recursion for trajectory n is:
        M_t = alpha * M_{t-1} + B_t

    B_t is a differentiable soft rasterization of the normalized box on a small
    Hs x Ws grid. Missing observations do not add a box, but the previous trail
    still decays, which naturally models occlusion gaps.
    """

    def __init__(
        self,
        feature_dim: int,
        grid_h: int = 12,
        grid_w: int = 20,
        alpha: float = 0.8,
        hidden_dim: int = 128,
        raster_sharpness: float = 80.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("grid_h and grid_w must be positive")

        self.feature_dim = int(feature_dim)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.alpha = float(alpha)
        self.hidden_dim = int(hidden_dim)
        self.raster_sharpness = float(raster_sharpness)

        # Grid-cell centers in normalized image coordinates [0, 1].
        xs = (torch.arange(self.grid_w, dtype=torch.float32) + 0.5) / self.grid_w
        ys = (torch.arange(self.grid_h, dtype=torch.float32) + 0.5) / self.grid_h
        self.register_buffer(
            "grid_x", xs.view(1, 1, 1, 1, self.grid_w), persistent=False
        )
        self.register_buffer(
            "grid_y", ys.view(1, 1, 1, self.grid_h, 1), persistent=False
        )

        # A small MLP is used instead of a CNN in V0 to keep memory use modest
        # when MOTIP creates many augmented ID-binding groups during training.
        self.encoder = nn.Sequential(
            nn.Linear(self.grid_h * self.grid_w, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        first = self.encoder[0]
        second = self.encoder[2]
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)

        # Start from the exact MOTIP baseline: the trail branch initially emits
        # zero, so the pretrained/original trajectory path is not perturbed.
        nn.init.zeros_(second.weight)
        nn.init.zeros_(second.bias)

        layer_norm = self.encoder[3]
        nn.init.ones_(layer_norm.weight)
        nn.init.zeros_(layer_norm.bias)

    def _soft_rasterize(
        self,
        boxes: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiably rasterize normalized cxcywh boxes.

        Args:
            boxes: [..., 4], normalized cx, cy, w, h.
            invalid_mask: [...], True for missing/padded observations.

        Returns:
            Soft masks with shape [..., grid_h, grid_w].
        """
        if boxes.shape[-1] != 4:
            raise ValueError(f"Expected boxes[..., 4], got {tuple(boxes.shape)}")
        if invalid_mask.shape != boxes.shape[:-1]:
            raise ValueError(
                "invalid_mask must match boxes without the last coordinate dimension: "
                f"boxes={tuple(boxes.shape)}, mask={tuple(invalid_mask.shape)}"
            )

        # Use the box dtype/device so mixed-precision inference works.
        gx = self.grid_x.to(device=boxes.device, dtype=boxes.dtype)
        gy = self.grid_y.to(device=boxes.device, dtype=boxes.dtype)

        cx, cy, w, h = boxes.unbind(dim=-1)
        half_w = 0.5 * w.clamp_min(0.0)
        half_h = 0.5 * h.clamp_min(0.0)

        left = (cx - half_w).unsqueeze(-1).unsqueeze(-1)
        right = (cx + half_w).unsqueeze(-1).unsqueeze(-1)
        top = (cy - half_h).unsqueeze(-1).unsqueeze(-1)
        bottom = (cy + half_h).unsqueeze(-1).unsqueeze(-1)

        s = self.raster_sharpness
        inside_x = torch.sigmoid(s * (gx - left)) * torch.sigmoid(s * (right - gx))
        inside_y = torch.sigmoid(s * (gy - top)) * torch.sigmoid(s * (bottom - gy))
        soft_box = inside_x * inside_y

        valid = (~invalid_mask).to(dtype=boxes.dtype).unsqueeze(-1).unsqueeze(-1)
        return soft_box * valid

    def build_trails(
        self,
        trajectory_boxes: torch.Tensor,
        trajectory_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Build all intermediate trail maps, mainly for debugging/visualization.

        Returns:
            [B, G, T, N, grid_h, grid_w]
        """
        self._validate_inputs(trajectory_boxes, trajectory_masks)
        b, g, t_len, n, _ = trajectory_boxes.shape
        state = trajectory_boxes.new_zeros((b, g, n, self.grid_h, self.grid_w))
        trails = []

        for t in range(t_len):
            observation = self._soft_rasterize(
                trajectory_boxes[:, :, t], trajectory_masks[:, :, t]
            )
            state = self.alpha * state + observation
            trails.append(state)

        if len(trails) == 0:
            return trajectory_boxes.new_zeros(
                (b, g, 0, n, self.grid_h, self.grid_w)
            )
        return torch.stack(trails, dim=2)

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        # state: [B, G, N, H, W] -> [B, G, N, feature_dim]
        flat = state.flatten(start_dim=-2)
        return self.encoder(flat)

    def forward(
        self,
        trajectory_boxes: torch.Tensor,
        trajectory_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return a trail embedding for every historical trajectory token.

        This streaming implementation avoids materializing all trail maps before
        encoding them, which reduces peak memory compared with build_trails().

        Returns:
            [B, G, T, N, feature_dim]
        """
        self._validate_inputs(trajectory_boxes, trajectory_masks)
        b, g, t_len, n, _ = trajectory_boxes.shape
        state = trajectory_boxes.new_zeros((b, g, n, self.grid_h, self.grid_w))
        embeddings = []

        for t in range(t_len):
            current_invalid = trajectory_masks[:, :, t]
            observation = self._soft_rasterize(
                trajectory_boxes[:, :, t], current_invalid
            )
            state = self.alpha * state + observation
            trail_feature = self._encode_state(state)

            # Padded/missing trajectory tokens remain masked exactly as in MOTIP.
            trail_feature = trail_feature.masked_fill(
                current_invalid.unsqueeze(-1), 0.0
            )
            embeddings.append(trail_feature)

        if len(embeddings) == 0:
            return trajectory_boxes.new_zeros(
                (b, g, 0, n, self.feature_dim)
            )
        return torch.stack(embeddings, dim=2)

    @staticmethod
    def _validate_inputs(
        trajectory_boxes: torch.Tensor,
        trajectory_masks: torch.Tensor,
    ) -> None:
        if trajectory_boxes.ndim != 5:
            raise ValueError(
                "trajectory_boxes must have shape [B, G, T, N, 4], got "
                f"{tuple(trajectory_boxes.shape)}"
            )
        if trajectory_masks.ndim != 4:
            raise ValueError(
                "trajectory_masks must have shape [B, G, T, N], got "
                f"{tuple(trajectory_masks.shape)}"
            )
        if trajectory_boxes.shape[:-1] != trajectory_masks.shape:
            raise ValueError(
                "trajectory_boxes and trajectory_masks are shape-incompatible: "
                f"boxes={tuple(trajectory_boxes.shape)}, mask={tuple(trajectory_masks.shape)}"
            )
        if trajectory_boxes.shape[-1] != 4:
            raise ValueError("The last trajectory_boxes dimension must be 4 (cx, cy, w, h).")
