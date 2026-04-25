"""
Cal-QL network architecture.

Reference: Nakamoto et al. 2023, NeurIPS. "Cal-QL: Calibrated Offline RL Pre-Training
for Efficient Online Fine-Tuning."

Networks:
  Actor   : (B, 398) → (B, 6) squashed Gaussian over mixed action space
  TwinQ   : (B, 404) → (B,1) × 2 twin Q-networks

Action space (p.u., training space):
  p_energy ∈ (−1, 1) via tanh
  c_as     ∈ (0, 1)×5 via (tanh + 1) / 2

De-normalized to physical MW in eval_policy.py (× P_MAX=10).

Note: No TTFE, no shared encoder. Flat MLP per sprint spec.
"""

import math
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

OBS_DIM = 32 * 12 + 14   # 398
ACT_DIM = 6
HIDDEN  = 256

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _mlp(in_dim: int, out_dim: int, final_act: bool = False) -> nn.Sequential:
    """2 hidden layers × 256 ReLU, matching spec. final_act=True adds ReLU after output."""
    layers = [
        nn.Linear(in_dim, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, out_dim),
    ]
    if final_act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ── Actor ────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Squashed-Gaussian policy over 6D continuous action (p.u. space).

    Mixed squash:
      p_energy (dim 0): tanh → (−1, 1)
      c_as (dims 1–5):  (tanh + 1)/2 → (0, 1)

    Deterministic eval mode: squashed mean (no sampling noise).
    """

    def __init__(self):
        super().__init__()
        self.trunk        = _mlp(OBS_DIM, HIDDEN, final_act=True)
        self.mu_head      = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std_head = nn.Linear(HIDDEN, ACT_DIM)

    def _forward_raw(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    @staticmethod
    def _squash(z: torch.Tensor) -> torch.Tensor:
        """Map tanh output z ∈ (−1,1)^6 to mixed action p.u. space."""
        return torch.cat([z[:, 0:1], (z[:, 1:] + 1.0) / 2.0], dim=-1)

    @staticmethod
    def _log_jac(z: torch.Tensor) -> torch.Tensor:
        """
        Sum of log |d action_i / d x_i| for the change-of-variables correction.

        p_energy: d(tanh(x))/dx = 1 − tanh²(x)
        c_as:     d((tanh(x)+1)/2)/dx = (1 − tanh²(x)) / 2
        """
        eps = 1e-6
        log_jac_energy = torch.log(1.0 - z[:, 0:1].pow(2) + eps)
        log_jac_as     = torch.log((1.0 - z[:, 1:].pow(2) + eps) / 2.0)
        return torch.cat([log_jac_energy, log_jac_as], dim=-1).sum(dim=-1)  # (B,)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample. Returns (action, log_prob) both with gradient.

        action   : (B, 6) p.u. in mixed bounds
        log_prob : (B,) log π(a|s), change-of-variables corrected
        """
        mu, log_std = self._forward_raw(obs)
        std = log_std.exp()
        x   = mu + std * torch.randn_like(mu)      # reparameterized
        z   = torch.tanh(x)
        action   = self._squash(z)
        log_prob = Normal(mu, std).log_prob(x).sum(dim=-1) - self._log_jac(z)
        return action, log_prob

    @torch.no_grad()
    def sample_n_ood(self, obs: torch.Tensor, n: int) -> torch.Tensor:
        """
        Sample n OOD actions per obs for CQL penalty. No gradient.
        obs: (B, 398) → returns (B*n, 6) in p.u. space.
        Caller is responsible for obs.repeat(n,1) reshape.
        """
        obs_rep = obs.unsqueeze(1).expand(-1, n, -1).reshape(-1, obs.shape[-1])
        mu, log_std = self._forward_raw(obs_rep)
        std = log_std.exp()
        x   = mu + std * torch.randn_like(mu)
        z   = torch.tanh(x)
        return self._squash(z)   # (B*n, 6)

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Eval mode: squashed mean, no sampling noise."""
        mu, _ = self._forward_raw(obs)
        z = torch.tanh(mu)
        return self._squash(z)   # (B, 6)


# ── TwinQ ─────────────────────────────────────────────────────────────────────

class TwinQ(nn.Module):
    """
    Twin Q-networks. Input: (obs, action) concatenated → (B, OBS_DIM + ACT_DIM = 404).
    Each Q outputs (B, 1) scalar.
    """

    def __init__(self):
        super().__init__()
        in_dim = OBS_DIM + ACT_DIM   # 404
        self.q1 = _mlp(in_dim, 1)
        self.q2 = _mlp(in_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, act)
        return torch.min(q1, q2)

    def update_target_from(self, online: "TwinQ", tau: float) -> None:
        """Polyak-average self ← τ*online + (1−τ)*self."""
        for p_tgt, p_src in zip(self.parameters(), online.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)
