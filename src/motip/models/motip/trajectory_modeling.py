# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Modified for Trail-MOTIP V0.

import torch
import torch.nn as nn
from contextlib import nullcontext

from models.ffn import FFN
from models.motip.trail_memory import BBoxTrailEncoder


class TrajectoryModeling(nn.Module):

    def __init__(
        self,
        detr_dim: int,
        ffn_dim_ratio: int,
        feature_dim: int,
        trail_enabled: bool = False,
        trail_grid_h: int = 12,
        trail_grid_w: int = 20,
        trail_alpha: float = 0.8,
        trail_encoder_hidden_dim: int = 128,
        trail_raster_sharpness: float = 80.0,
        trail_fusion_scale: float = 1.0,
        preserve_trail_rng: bool = False,
        ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim
        self.trail_enabled = trail_enabled
        self.trail_fusion_scale = float(trail_fusion_scale)

        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)

        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        # Keep the original MOTIP initialization for the original branch.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            pass

        # Instantiate the new branch after the loop above so its final projection
        # can remain zero-initialized. This makes step 0 exactly the MOTIP baseline.
        self.trail_encoder = None
        if self.trail_enabled:
            with torch.random.fork_rng(devices=[]) if preserve_trail_rng else nullcontext():
                self.trail_encoder = BBoxTrailEncoder(
                    feature_dim=feature_dim,
                    grid_h=trail_grid_h,
                    grid_w=trail_grid_w,
                    alpha=trail_alpha,
                    hidden_dim=trail_encoder_hidden_dim,
                    raster_sharpness=trail_raster_sharpness,
                )

    def forward(self, seq_info):
        trajectory_features = seq_info["trajectory_features"]

        trajectory_features = trajectory_features + self.adapter(trajectory_features)
        trajectory_features = self.norm(trajectory_features)

        if self.trail_enabled:
            trajectory_boxes = seq_info["trajectory_boxes"]
            trajectory_masks = seq_info["trajectory_masks"]
            trail_features = self.trail_encoder(
                trajectory_boxes=trajectory_boxes,
                trajectory_masks=trajectory_masks,
            )
            trajectory_features = (
                trajectory_features + self.trail_fusion_scale * trail_features
            )

        trajectory_features = trajectory_features + self.ffn(trajectory_features)
        trajectory_features = self.ffn_norm(trajectory_features)

        seq_info["trajectory_features"] = trajectory_features
        return seq_info
