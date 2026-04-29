"""
Diffusion-QL network architecture.

Reference: Wang, Hunt, Zhou. "Diffusion Policies as an Expressive Policy Class
for Offline Reinforcement Learning." ICLR 2023. arXiv:2208.06193.

Architecture:
  ObsEncoder     : (N, 398) → (N, 256)  shared MLP
  DiffusionNet   : (obs_feat, action, t_emb) → (N, 6)  predicts noise ε
  TwinQ          : (obs_feat, action) → (N, 1) × 2  critic pair

Diffusion:
  T = 5 steps, linear β schedule β_1=1e-4 → β_T=0.02
  Noise prediction parameterization (predict ε, not x0)
  Action space: p.u. (energy ∈ [-1,1], AS ∈ [0,1]×5)

Note: TTFE is NOT used (HybridBid v5.1 scope). Flat MLP encoder per paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

OBS_DIM = 398
ACT_DIM = 6
HIDDEN = 256
TIME_EMB_DIM = 128

# ── Diffusion schedule ──────────────────────────────────────────────────────
T_STEPS = 5
BETA_MIN = 1e-4
BETA_MAX = 0.02


def make_schedule(T: int = T_STEPS) -> dict:
    """Precompute linear β schedule and derived α quantities."""
    betas = torch.linspace(BETA_MIN, BETA_MAX, T)        # (T,)
    alphas = 1.0 - betas                                  # (T,)
    alpha_cumprod = torch.cumprod(alphas, dim=0)          # (T,)
    alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]])  # (T,)
    sqrt_alpha_cumprod = alpha_cumprod.sqrt()
    sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt()
    # Posterior variance for reverse process (DDPM)
    posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
    return dict(
        betas=betas,
        alphas=alphas,
        alpha_cumprod=alpha_cumprod,
        alpha_cumprod_prev=alpha_cumprod_prev,
        sqrt_alpha_cumprod=sqrt_alpha_cumprod,
        sqrt_one_minus_alpha_cumprod=sqrt_one_minus_alpha_cumprod,
        posterior_variance=posterior_variance,
    )


# ── Building blocks ─────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 3,
         final_activation: bool = False) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.Mish()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.Mish()]
    layers.append(nn.Linear(hidden, out_dim))
    if final_activation:
        layers.append(nn.Mish())
    return nn.Sequential(*layers)


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal time embedding → linear projection to TIME_EMB_DIM."""

    def __init__(self, dim: int = TIME_EMB_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.Mish(), nn.Linear(dim * 2, dim)
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (N,) integer step indices (0-based)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (N, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)     # (N, dim)
        return self.proj(emb)


# ── Main modules ─────────────────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """Flatten (32×12 + 14) = 398 → 256 feature vector."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(OBS_DIM, HIDDEN, HIDDEN, n_layers=3, final_activation=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (N, 398) → (N, 256)."""
        return self.net(obs)


class DiffusionNet(nn.Module):
    """
    Denoising network: predicts added noise ε given (obs_feat, noisy_action, t).
    Input dim: 256 + 6 + 128 = 390.
    """

    def __init__(self):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(TIME_EMB_DIM)
        in_dim = HIDDEN + ACT_DIM + TIME_EMB_DIM
        self.net = _mlp(in_dim, HIDDEN, ACT_DIM, n_layers=3, final_activation=False)

    def forward(self, obs_feat: torch.Tensor, noisy_action: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        obs_feat    : (N, 256)
        noisy_action: (N, 6)
        t           : (N,) int — diffusion step index (0-based)
        returns     : (N, 6) predicted noise ε
        """
        t_emb = self.time_emb(t)                          # (N, 128)
        x = torch.cat([obs_feat, noisy_action, t_emb], dim=-1)  # (N, 390)
        return self.net(x)


class TwinQ(nn.Module):
    """Twin Q-networks (SAC-style). Input: (obs_feat, action) → two scalar Q-values."""

    def __init__(self):
        super().__init__()
        in_dim = HIDDEN + ACT_DIM
        self.q1 = _mlp(in_dim, HIDDEN, 1, n_layers=3)
        self.q2 = _mlp(in_dim, HIDDEN, 1, n_layers=3)

    def forward(self, obs_feat: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns Q1(s,a), Q2(s,a) each shape (N, 1)."""
        x = torch.cat([obs_feat, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs_feat, action)
        return torch.min(q1, q2)


class DiffusionQL(nn.Module):
    """
    Full Diffusion-QL model combining encoder, diffusion policy, and twin critics.

    All forward methods share obs encoder weights between policy/critic paths
    via separate encoder instances (policy_enc and critic_enc) to avoid
    conflicting gradient directions between BC loss and Q-loss.
    """

    def __init__(self):
        super().__init__()
        schedule = make_schedule(T_STEPS)
        for k, v in schedule.items():
            self.register_buffer(k, v)

        self.policy_enc  = ObsEncoder()
        self.critic_enc  = ObsEncoder()
        self.diffusion   = DiffusionNet()
        self.twin_q      = TwinQ()

        # Target critics (no grad, updated via EMA)
        self.twin_q_tgt = TwinQ()
        for p in self.twin_q_tgt.parameters():
            p.requires_grad_(False)
        self.twin_q_tgt.load_state_dict(self.twin_q.state_dict())

    # ── Noise / denoise utilities ─────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(ᾱ_t)×x0 + sqrt(1-ᾱ_t)×ε."""
        s = self.sqrt_alpha_cumprod[t].view(-1, 1)
        r = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1)
        return s * x0 + r * noise

    @torch.no_grad()
    def p_sample_step(self, obs_feat: torch.Tensor, x_t: torch.Tensor,
                      t_idx: int) -> torch.Tensor:
        """One reverse-diffusion step: x_{t-1} | x_t, obs."""
        N = x_t.shape[0]
        t_tensor = torch.full((N,), t_idx, dtype=torch.long, device=x_t.device)

        eps_pred = self.diffusion(obs_feat, x_t, t_tensor)

        alpha    = self.alphas[t_idx]
        alpha_cp = self.alpha_cumprod[t_idx]
        beta     = self.betas[t_idx]

        # Predicted x0
        x0_pred = (x_t - (1 - alpha_cp).sqrt() * eps_pred) / alpha_cp.sqrt()
        x0_pred = self._clip_action(x0_pred)

        # Posterior mean
        mu = alpha.sqrt() * (1 - self.alpha_cumprod_prev[t_idx]) / (1 - alpha_cp) * x_t \
           + self.alpha_cumprod_prev[t_idx].sqrt() * beta / (1 - alpha_cp) * x0_pred

        if t_idx == 0:
            return mu
        noise = torch.randn_like(x_t)
        return mu + self.posterior_variance[t_idx].sqrt() * noise

    def _clip_action(self, a: torch.Tensor) -> torch.Tensor:
        """Clip to feasible p.u. range: energy∈[-1,1], AS∈[0,1].
        Uses torch.cat (not in-place) so autograd graph stays valid."""
        return torch.cat([
            torch.clamp(a[:, 0:1], -1.0, 1.0),
            torch.clamp(a[:, 1:],   0.0, 1.0),
        ], dim=1)

    @torch.no_grad()
    def sample_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Full denoising chain: xT ~ N(0,I) → x0.
        Returns clipped action (N, 6) in p.u. space.
        Inference: no gradient, use for Q-target computation and eval.
        """
        obs_feat = self.policy_enc(obs)
        x = torch.randn(obs.shape[0], ACT_DIM, device=obs.device)
        for t in reversed(range(T_STEPS)):
            x = self.p_sample_step(obs_feat, x, t)
        return self._clip_action(x)

    def sample_action_grad(self, obs_feat: torch.Tensor) -> torch.Tensor:
        """
        Full denoising chain with gradient (for policy loss).
        Keeps computation graph through denoising steps for backprop.
        """
        x = torch.randn(obs_feat.shape[0], ACT_DIM, device=obs_feat.device)
        for t in reversed(range(T_STEPS)):
            t_tensor = torch.full((obs_feat.shape[0],), t,
                                  dtype=torch.long, device=obs_feat.device)
            eps_pred = self.diffusion(obs_feat, x, t_tensor)

            alpha    = self.alphas[t]
            alpha_cp = self.alpha_cumprod[t]
            beta     = self.betas[t]

            x0_pred = (x - (1 - alpha_cp).sqrt() * eps_pred) / alpha_cp.sqrt()
            x0_pred = self._clip_action(x0_pred)

            mu = alpha.sqrt() * (1 - self.alpha_cumprod_prev[t]) / (1 - alpha_cp) * x \
               + self.alpha_cumprod_prev[t].sqrt() * beta / (1 - alpha_cp) * x0_pred

            if t == 0:
                x = mu
            else:
                x = mu + self.posterior_variance[t].sqrt() * torch.randn_like(x)

        return self._clip_action(x)

    # ── Target network update ────────────────────────────────────────────

    def update_target(self, tau: float = 0.005) -> None:
        for p_tgt, p_src in zip(self.twin_q_tgt.parameters(),
                                 self.twin_q.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)
