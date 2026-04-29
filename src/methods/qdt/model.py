"""
QDT network architecture.

Reference: Yamagata, Khalil, Santos-Rodriguez. "Q-learning Decision Transformer:
Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL."
ICML 2023. arXiv:2209.03993.

Three components:

CQLCritic     — Twin Q-networks trained with CQL conservatism penalty (Stage 1)
RelabelerQ    — Thin wrapper exposing (obs_flat, action) → Q_min for relabeling (Stage 2)
DecisionTransformer — Causal transformer predicting action given (RTG, obs, past actions) (Stage 3)

Observation: flat (398,) = flattened price_history(32,12) + static_features(14)
Action: p.u. (6,)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple

OBS_DIM = 398
ACT_DIM = 6
HIDDEN  = 256
K       = 20      # DT context length

# ── CQL Critic ──────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 3) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.Mish()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.Mish()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class CQLCritic(nn.Module):
    """
    Twin Q-networks with CQL conservatism.

    Loss (per batch, Stage 1):
      L = L_TD + alpha * L_CQL
      L_TD  = MSE(Q(s,a), r + gamma*Q_target(s',a'))   [standard Bellman]
      L_CQL = E_a_rand[Q(s,a_rand)] - E_(s,a)~D[Q(s,a)]  [pushes down OOD Q-values]

    alpha = 1.0 per paper default for medium-replay datasets.
    """

    def __init__(self):
        super().__init__()
        in_dim = OBS_DIM + ACT_DIM
        self.q1 = _mlp(in_dim, HIDDEN, 1)
        self.q2 = _mlp(in_dim, HIDDEN, 1)

        self.q1_tgt = _mlp(in_dim, HIDDEN, 1)
        self.q2_tgt = _mlp(in_dim, HIDDEN, 1)
        for p in list(self.q1_tgt.parameters()) + list(self.q2_tgt.parameters()):
            p.requires_grad_(False)
        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())

    def forward(self, obs: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

    def q_min_target(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return torch.min(self.q1_tgt(x), self.q2_tgt(x))

    def update_target(self, tau: float = 0.005) -> None:
        for p_t, p_s in zip(self.q1_tgt.parameters(), self.q1.parameters()):
            p_t.data.mul_(1 - tau).add_(tau * p_s.data)
        for p_t, p_s in zip(self.q2_tgt.parameters(), self.q2.parameters()):
            p_t.data.mul_(1 - tau).add_(tau * p_s.data)


# ── Decision Transformer ────────────────────────────────────────────────────

DT_LAYERS  = 4
DT_HEADS   = 8
DT_HIDDEN  = 128    # embedding dim per token
DT_DROPOUT = 0.1


class DecisionTransformer(nn.Module):
    """
    Causal Decision Transformer for action prediction.

    Token sequence (3K tokens): [RTG_0, obs_0, act_0, RTG_1, obs_1, act_1, ...]

    Embeddings:
      RTG    : scalar → linear → DT_HIDDEN
      obs    : (398,) → linear → DT_HIDDEN
      action : (6,)   → linear → DT_HIDDEN
      position: learned, per (3K) positions

    Predicts action at each obs position (only action tokens are supervised).
    At inference: slide K-step window, feed current RTG target.
    """

    def __init__(self):
        super().__init__()
        # Token embeddings
        self.rtg_emb   = nn.Linear(1, DT_HIDDEN)
        self.obs_emb   = nn.Linear(OBS_DIM, DT_HIDDEN)
        self.act_emb   = nn.Linear(ACT_DIM, DT_HIDDEN)
        self.pos_emb   = nn.Embedding(3 * K, DT_HIDDEN)

        self.ln_in  = nn.LayerNorm(DT_HIDDEN)
        self.drop   = nn.Dropout(DT_DROPOUT)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=DT_HIDDEN, nhead=DT_HEADS,
            dim_feedforward=DT_HIDDEN * 4,
            dropout=DT_DROPOUT, batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=DT_LAYERS)
        self.act_head    = nn.Linear(DT_HIDDEN, ACT_DIM)

        # Causal mask (upper-triangular, -inf for future tokens)
        self.register_buffer("causal_mask",
                             torch.triu(torch.full((3*K, 3*K), float("-inf")), diagonal=1))

    def forward(self, rtg: torch.Tensor, obs: torch.Tensor,
                act: torch.Tensor) -> torch.Tensor:
        """
        rtg : (B, K)       target return-to-go per step
        obs : (B, K, 398)
        act : (B, K, 6)
        returns: (B, K, 6) predicted actions (supervised at obs positions)
        """
        B = rtg.shape[0]

        r_tok = self.rtg_emb(rtg.unsqueeze(-1))               # (B, K, H)
        o_tok = self.obs_emb(obs)                              # (B, K, H)
        a_tok = self.act_emb(act)                              # (B, K, H)

        # Interleave: [r0, o0, a0, r1, o1, a1, ...]
        tokens = torch.stack([r_tok, o_tok, a_tok], dim=2)    # (B, K, 3, H)
        tokens = tokens.reshape(B, 3 * K, DT_HIDDEN)          # (B, 3K, H)

        pos = self.pos_emb(torch.arange(3 * K, device=rtg.device))  # (3K, H)
        tokens = self.drop(self.ln_in(tokens + pos))

        out = self.transformer(tokens, mask=self.causal_mask)  # (B, 3K, H)

        # Extract obs-position outputs (every 3rd token starting at index 1)
        obs_out = out[:, 1::3, :]                              # (B, K, H)
        return self.act_head(obs_out)                          # (B, K, 6)

    def predict_action(self, rtg_ctx: torch.Tensor, obs_ctx: torch.Tensor,
                       act_ctx: torch.Tensor) -> torch.Tensor:
        """
        Context may be < K steps (e.g., at episode start).
        Pads with zeros to length K, returns action for the last obs position.
        """
        T = rtg_ctx.shape[1]
        if T < K:
            pad = K - T
            rtg_ctx = torch.cat([torch.zeros(rtg_ctx.shape[0], pad, device=rtg_ctx.device), rtg_ctx], dim=1)
            obs_ctx = torch.cat([torch.zeros(obs_ctx.shape[0], pad, OBS_DIM, device=obs_ctx.device), obs_ctx], dim=1)
            act_ctx = torch.cat([torch.zeros(act_ctx.shape[0], pad, ACT_DIM, device=act_ctx.device), act_ctx], dim=1)

        pred = self.forward(rtg_ctx, obs_ctx, act_ctx)         # (B, K, 6)
        return pred[:, -1, :]                                   # (B, 6) last step
