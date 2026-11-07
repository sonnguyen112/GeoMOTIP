import torch
from torch import nn


class TrailGate(nn.Module):
    """Observation reliability, learned solely through MOTIP's ID loss."""

    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim + 1, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, confidence):
        return self.net(torch.cat((features, confidence.unsqueeze(-1)), dim=-1)).sigmoid().squeeze(-1)
