"""Small trail-only attention branch for Trail-MOTIP V2.

The branch never sees the original MOTIP history tokens.  It produces a
correction in the ID-logit space, which is combined with the untouched
baseline decoder by a scalar initialized to zero.
"""

from __future__ import annotations

import torch
from torch import nn

from models.ffn import FFN


class TrailResidualBranch(nn.Module):
    """Lightweight trail-only cross-attention and ID head."""

    def __init__(self, feature_dim: int, id_dim: int, head_dim: int,
                 num_id_vocabulary: int, ffn_dim_ratio: int = 2):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.id_dim = int(id_dim)
        self.model_dim = self.feature_dim + self.id_dim
        self.num_id_vocabulary = int(num_id_vocabulary)
        self.num_heads = self.model_dim // int(head_dim)
        if self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("Trail residual model dimension must divide into attention heads")

        self.token_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.query_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            self.model_dim, self.num_heads, dropout=0.0,
            batch_first=True, add_zero_attn=True,
        )
        self.norm = nn.LayerNorm(self.model_dim)
        self.ffn = FFN(
            d_model=self.model_dim,
            d_ffn=self.model_dim * int(ffn_dim_ratio),
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(self.model_dim)
        self.id_head = nn.Linear(self.id_dim, self.num_id_vocabulary + 1, bias=False)

        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, seq, id_label_to_embed, generate_empty_id_embed):
        unknown_features = seq["unknown_features"]
        trail_tokens = seq["trail_tokens"]
        trail_slot_ids = seq["trail_slot_ids"]
        trail_masks = seq["trail_masks"]
        b, g, t, n, _ = unknown_features.shape
        k, p = trail_tokens.shape[3:5]
        query = torch.cat((unknown_features, generate_empty_id_embed(unknown_features)), -1)
        query = self.query_projection(query).reshape(b * g * t, n, self.model_dim)

        slot_ids = id_label_to_embed(trail_slot_ids)
        slot_ids = slot_ids.unsqueeze(-2).expand(b, g, t, k, p, self.id_dim)
        keys = torch.cat((trail_tokens, slot_ids), -1)
        keys = self.token_projection(keys).reshape(b * g * t, k * p, self.model_dim)
        padding = trail_masks.unsqueeze(-1).expand(b, g, t, k, p).reshape(b * g * t, k * p)

        attended, _ = self.cross_attn(
            query=query, key=keys, value=keys,
            key_padding_mask=padding,
        )
        attended = self.norm(query + attended)
        attended = self.ffn_norm(attended + self.ffn(attended))
        logits = self.id_head(attended[..., -self.id_dim:])
        return logits.reshape(b, g, t, n, self.num_id_vocabulary + 1)
