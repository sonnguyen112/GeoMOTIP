import math

import torch
from torch import nn
from torch.nn import functional as F


class TrailEncoder(nn.Module):
    """R memories -> small convolution -> P spatial tokens + fixed 2-D PE."""

    def __init__(self, channels, scales, feature_dim, num_tokens=8, hidden=32):
        super().__init__()
        if num_tokens <= 0 or feature_dim % 4:
            raise ValueError("Positive token count and feature dimension divisible by 4 required")
        ph = max(d for d in range(1, int(math.sqrt(num_tokens)) + 1) if num_tokens % d == 0)
        self.pool_size = (ph, num_tokens // ph)
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(nn.Conv2d(channels * scales, hidden, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.projection = nn.Linear(hidden, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        y, x = torch.meshgrid((torch.arange(ph) + .5) / ph,
                              (torch.arange(num_tokens // ph) + .5) / (num_tokens // ph),
                              indexing="ij")
        freq = 10000 ** (-torch.arange(feature_dim // 4).float() / (feature_dim // 4))
        x, y = x.flatten()[:, None] * (2 * math.pi) * freq, y.flatten()[:, None] * (2 * math.pi) * freq
        self.register_buffer("position", torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1), persistent=False)

    def forward(self, state):
        # [*,R,C,H,W] -> [*,P,F]. State stays float32 during the recurrence.
        shape = state.shape[:-4]
        x = state.flatten(-4, -3).reshape(-1, state.shape[-4] * state.shape[-3], *state.shape[-2:])
        x = x.to(self.conv[0].weight.dtype)
        x = F.adaptive_avg_pool2d(self.conv(x), self.pool_size).flatten(2).transpose(1, 2)
        x = self.norm(self.projection(x)) + self.position.to(x.dtype)
        return x.reshape(*shape, self.num_tokens, self.feature_dim)
