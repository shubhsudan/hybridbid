"""
Behavior Cloning network for post-RTC+B BESS bidding.

Input: flattened (price_history (32×12) + static_features (14)) = 398 dims
Output: 6D physical MW action
  dim 0 (p_energy): tanh → [-P_max, +P_max]  (signed; + discharge, − charge)
  dim 1-5 (c_as):   sigmoid → [0, P_max]     (non-negative AS capacity offers)

Battery constants hardcoded per CLAUDE.md.
"""

import torch
import torch.nn as nn

P_MAX    = 10.0
OBS_DIM  = 398    # 32 × 12 + 14
ACT_DIM  = 6
HIDDEN   = 256


class BCNet(nn.Module):
    def __init__(
        self,
        obs_dim: int  = OBS_DIM,
        hidden_dim: int = HIDDEN,
        p_max: float  = P_MAX,
    ):
        super().__init__()
        self.p_max = p_max
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, ACT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, obs_dim) float32
        returns: (B, 6) float32 in physical MW
        """
        h = self.trunk(x)
        out = self.head(h)
        p_energy = torch.tanh(out[:, 0:1]) * self.p_max       # [-P_max, P_max]
        c_as     = torch.sigmoid(out[:, 1:]) * self.p_max      # [0, P_max]
        return torch.cat([p_energy, c_as], dim=-1)
