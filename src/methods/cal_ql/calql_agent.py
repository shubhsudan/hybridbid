"""
Cal-QL offline update step.

Implements:
  update_critic : TD loss + Cal-QL calibrated CQL penalty
  update_actor  : Q-maximization (alpha_entropy=0 → pure Q-max)
  update_target : Polyak EMA on twin_q_tgt

Cal-QL calibration (Nakamoto et al. 2023, Algorithm 1):
  Standard CQL pushes up on Q(s, a_dataset). Cal-QL replaces the push-up target
  with max(Q(s, a_dataset), V_behavior(s)), preventing over-conservatism when
  Q(s, a_dataset) < V_behavior(s) (the failure mode that killed QDT Stage 2
  at P50=-$127 with both alpha_cql=1.0 and alpha_cql=0.3).

CQL penalty (per-Q network):
  push_down = logsumexp([Q_rand, Q_pi]) − log(2 · n_random)   [log-mean-exp approx]
  push_up   = max(Q_dataset.detach(), V_behavior)
  cql_term  = alpha_cql · (push_down − push_up)

Bootstrap target (SARSA-style, per sprint spec):
  y = r + gamma · (1 − sarsa_done) · min(Q1_tgt, Q2_tgt)(s', a'_pi)
  sarsa_done zeros the bootstrap at CT-midnight episode boundaries.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict

from src.methods.cal_ql.networks import Actor, TwinQ


class CalQLAgent:
    """
    Manages Cal-QL offline updates. Owns actor and critic optimizers.

    Parameters
    ----------
    actor       : Actor
    twin_q      : TwinQ (online)
    twin_q_tgt  : TwinQ (target, no grad)
    actor_opt   : Adam optimizer for actor
    critic_opt  : Adam optimizer for twin_q
    cfg         : dict with keys matching config.yaml
    device      : torch.device
    """

    def __init__(self, actor: Actor, twin_q: TwinQ, twin_q_tgt: TwinQ,
                 actor_opt: torch.optim.Optimizer,
                 critic_opt: torch.optim.Optimizer,
                 cfg: dict, device: torch.device):
        self.actor      = actor
        self.twin_q     = twin_q
        self.twin_q_tgt = twin_q_tgt
        self.actor_opt  = actor_opt
        self.critic_opt = critic_opt
        self.cfg        = cfg
        self.device     = device
        self._n         = cfg["n_random_actions"]
        self._gamma     = cfg["gamma"]
        self._alpha_cql = cfg["alpha_cql"]
        self._alpha_ent = cfg.get("alpha_entropy", 0.0)
        self._log_2n    = math.log(2 * self._n)   # normalization for log-mean-exp

    # ── Critic update ────────────────────────────────────────────────────────

    def update_critic(self, obs: torch.Tensor, act: torch.Tensor,
                      rew: torch.Tensor, next_obs: torch.Tensor,
                      sarsa_done: torch.Tensor, v_beh: torch.Tensor) -> Dict:
        """
        One Cal-QL critic update step.

        Args (all on device):
          obs, next_obs : (B, 398)
          act           : (B, 6) p.u. dataset action
          rew           : (B, 1)
          sarsa_done    : (B, 1) 1.0 at CT-midnight episode boundaries
          v_beh         : (B, 1) V_behavior calibration anchor
        """
        B, n = obs.shape[0], self._n

        # ── TD target (no grad) ─────────────────────────────────────────────
        with torch.no_grad():
            a_next, _ = self.actor.sample(next_obs)
            q1_tgt, q2_tgt = self.twin_q_tgt(next_obs, a_next)
            q_target = rew + self._gamma * (1.0 - sarsa_done) * torch.min(q1_tgt, q2_tgt)

        # ── TD loss on dataset actions ───────────────────────────────────────
        q1, q2 = self.twin_q(obs, act)
        td_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # ── CQL OOD actions ─────────────────────────────────────────────────
        obs_rep = obs.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)  # (B*n, 398)

        # Random uniform OOD: p_energy ~ U(-1,1), c_as ~ U(0,1)
        with torch.no_grad():
            energy_rand = torch.rand(B * n, 1, device=self.device) * 2.0 - 1.0
            as_rand     = torch.rand(B * n, 5, device=self.device)
            a_rand      = torch.cat([energy_rand, as_rand], dim=-1)   # (B*n, 6)

        # Policy OOD (no grad through actor during critic update)
        a_pi_ood = self.actor.sample_n_ood(obs, n)   # (B*n, 6); @torch.no_grad inside

        # Q for all OOD actions
        Q1_rand, Q2_rand = self.twin_q(obs_rep, a_rand)    # (B*n, 1) each
        Q1_pi,   Q2_pi   = self.twin_q(obs_rep, a_pi_ood)  # (B*n, 1) each

        Q1_rand = Q1_rand.reshape(B, n)
        Q2_rand = Q2_rand.reshape(B, n)
        Q1_pi   = Q1_pi.reshape(B, n)
        Q2_pi   = Q2_pi.reshape(B, n)

        # Log-mean-exp over 2n OOD samples (approximates log E_mu[exp(Q)])
        Q1_cat = torch.cat([Q1_rand, Q1_pi], dim=1)   # (B, 2n)
        Q2_cat = torch.cat([Q2_rand, Q2_pi], dim=1)   # (B, 2n)
        Q1_lme = torch.logsumexp(Q1_cat, dim=1, keepdim=True) - self._log_2n  # (B,1)
        Q2_lme = torch.logsumexp(Q2_cat, dim=1, keepdim=True) - self._log_2n  # (B,1)

        # ── Cal-QL calibration: push-up = max(Q_dataset, V_behavior) ────────
        Q1_push_up = torch.max(q1.detach(), v_beh)   # (B, 1)
        Q2_push_up = torch.max(q2.detach(), v_beh)   # (B, 1)

        cql1 = (Q1_lme - Q1_push_up).mean()
        cql2 = (Q2_lme - Q2_push_up).mean()
        cql_loss = self._alpha_cql * (cql1 + cql2)

        # ── Total critic loss ─────────────────────────────────────────────────
        total_loss = td_loss + cql_loss

        self.critic_opt.zero_grad()
        total_loss.backward()
        self.critic_opt.step()

        q_min = torch.min(q1, q2)
        return {
            "td_loss":    td_loss.item(),
            "cql_term":   (cql1.item() + cql2.item()) / 2.0,   # per-Q average
            "cql_loss":   cql_loss.item(),
            "Q1_mean":    q1.mean().item(),
            "Q2_mean":    q2.mean().item(),
            "Q_mean":     q_min.mean().item(),
            "Q_max":      q_min.max().item(),
            "Q1_lme":     Q1_lme.mean().item(),
            "V_beh_mean": v_beh.mean().item(),
            "push_up_mean": Q1_push_up.mean().item(),
        }

    # ── Actor update ─────────────────────────────────────────────────────────

    def update_actor(self, obs: torch.Tensor) -> Dict:
        """
        Q-maximization actor update.
        alpha_entropy=0 (per config): no SAC entropy regularization.
        Gradient flows through reparameterized sample → Q → actor params.
        """
        a_pi, log_pi = self.actor.sample(obs)
        q_pi = self.twin_q.q_min(obs, a_pi)   # critic not in actor_opt; grad ignored there

        actor_loss = (-q_pi + self._alpha_ent * log_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "log_pi":     log_pi.mean().item(),
            "q_pi_mean":  q_pi.mean().item(),
        }

    # ── Target network ────────────────────────────────────────────────────────

    def update_target(self) -> None:
        """Polyak-average target Q from online Q."""
        self.twin_q_tgt.update_target_from(self.twin_q, self.cfg["tau_polyak"])
