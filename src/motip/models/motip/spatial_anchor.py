"""Minimal bbox-position prior for a controlled MOTIP association ablation.

This is NOT a reproduction of MOTIP2: that method reuses detector positional
embeddings and changes several other components. Here the frozen detector and
baseline decoder stay unchanged; a single learned box projection is added to
both history and current object features at the ID-decoder handoff.
"""

import torch
from torch import nn


class BoxSpatialAnchor(nn.Module):
    """Normalized cxcywh -> feature vector, with neutral masked observations.

    Zero initialization preserves baseline logits initially. Unlike a zero
    scalar multiplying a deep branch, this single linear map receives a direct
    nonzero gradient on the first backward pass.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.projection = nn.Linear(4, feature_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, boxes: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if boxes.shape[-1] != 4 or boxes.shape[:-1] != masks.shape:
            raise ValueError("Box anchor expects normalized cxcywh [...,4] and padding masks [...]")
        if masks.dtype != torch.bool:
            raise ValueError("Box-anchor masks must be boolean (True = invalid)")
        # Padding must not introduce NaN gradients, even if padded boxes are NaN.
        inputs = boxes.detach().masked_fill(masks.unsqueeze(-1), 0)
        return self.projection(inputs).masked_fill(masks.unsqueeze(-1), 0)
