"""
Diffusion-QL training script.

Usage:
  python -m methods.diffusion_ql.train --mode smoke [--gpu 0]
  python -m methods.diffusion_ql.train --mode full  [--gpu 0]

Smoke: 5k steps, eval at 5k, commit SMOKE_REPORT.md. Exits.
Full: trains to 50k steps, sys.exit(0) at checkpoint. Karthik reviews before continuing.

Sprint discipline:
  - sys.exit(0) at CHECKPOINT_STEP (50k). No auto-continuation.
  - One variable per experiment. Do not change hyperparameters mid-run.
  - Smoke pass criteria checked and logged.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT)

from methods.diffusion_ql.data_loader import PostbreakDataset, make_dataloader
from methods.diffusion_ql.model import DiffusionQL, T_STEPS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dql")

# ── Hyperparameters ────────────────────────────────────────────────────────
CFG = dict(
    lr          = 3e-4,
    batch_size  = 256,
    gamma       = 0.99,
    tau         = 0.005,
    beta_q      = 1.0,     # Q-maximization weight in policy loss (paper default)
    log_every   = 500,
    eval_every  = 10_000,
    smoke_steps = 5_000,
    full_steps  = 100_000,
    checkpoint_step = 50_000,  # sys.exit here — Karthik reviews before continuing
)

# Smoke-pass thresholds
SMOKE_MAX_AS_MEAN = 7.0     # MW: flag if mean AS bid > 7 MW per dimension
SMOKE_MIN_REWARD  = -200.0  # $/interval: warn below
SMOKE_MAX_REWARD  = 300.0   # $/interval: warn above


def build_infinite_loader(npz_path: str, batch_size: int):
    """Infinite random-sampler over the dataset."""
    dataset = PostbreakDataset(npz_path)
    while True:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        yield from loader


def train(mode: str, gpu: int, train_path: str, val_path: str,
          data_dir: str, results_dir: str):

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    n_steps = CFG["smoke_steps"] if mode == "smoke" else CFG["full_steps"]
    ckpt_dir = Path(ROOT) / "checkpoints" / "sprint" / "dql"
    log_dir  = Path(ROOT) / "logs" / "sprint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"dql_{mode}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    model = DiffusionQL().to(device)
    opt_policy = torch.optim.Adam(
        list(model.policy_enc.parameters()) + list(model.diffusion.parameters()),
        lr=CFG["lr"]
    )
    opt_critic = torch.optim.Adam(
        list(model.critic_enc.parameters()) + list(model.twin_q.parameters()),
        lr=CFG["lr"]
    )

    data_iter = build_infinite_loader(train_path, CFG["batch_size"])

    # For smoke reward-statistics check
    reward_buf = []

    t0_total = time.time()
    for step in range(1, n_steps + 1):

        obs, act, rew, next_obs, done = next(data_iter)
        obs      = obs.to(device)
        act      = act.to(device)
        rew      = rew.to(device).unsqueeze(1)  # (B, 1)
        next_obs = next_obs.to(device)
        done     = done.to(device).unsqueeze(1)  # (B, 1) — always 0; kept for correctness

        reward_buf.extend(rew.squeeze(1).cpu().tolist())

        # ── Critic update ────────────────────────────────────────────────
        with torch.no_grad():
            next_obs_feat = model.policy_enc(next_obs)
            a_next = model.sample_action_grad(next_obs_feat).detach()
            # Use target critic for Q-target
            next_obs_feat_tgt = model.critic_enc(next_obs)
            q1_tgt, q2_tgt = model.twin_q_tgt(next_obs_feat_tgt, a_next)
            q_target = rew + CFG["gamma"] * (1.0 - done) * torch.min(q1_tgt, q2_tgt)

        obs_feat = model.critic_enc(obs)
        q1, q2 = model.twin_q(obs_feat, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()

        # ── Policy (diffusion) update ────────────────────────────────────
        # BC loss: standard diffusion denoising on dataset actions
        t_rand = torch.randint(0, T_STEPS, (obs.shape[0],), device=device)
        noise = torch.randn_like(act)
        noisy_act = model.q_sample(act, t_rand, noise)
        obs_feat_policy = model.policy_enc(obs)
        eps_pred = model.diffusion(obs_feat_policy, noisy_act, t_rand)
        bc_loss = F.mse_loss(eps_pred, noise)

        # Q-maximization: sample actions from policy, maximize Q
        a_pi = model.sample_action_grad(obs_feat_policy)
        obs_feat_for_q = model.critic_enc(obs).detach()  # critic enc, stop grad
        q_pi = model.twin_q.q_min(obs_feat_for_q, a_pi)

        # Normalize Q to prevent scale dominating BC loss (Wang et al. §C)
        q_norm = q_pi / (q_pi.abs().mean().detach() + 1e-8)
        policy_loss = bc_loss - CFG["beta_q"] * q_norm.mean()

        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

        # ── Target network EMA ───────────────────────────────────────────
        model.update_target(CFG["tau"])

        # ── Logging ─────────────────────────────────────────────────────
        if step % CFG["log_every"] == 0:
            q_val = q_pi.mean().item()
            log.info(
                f"step={step:>7}  bc_loss={bc_loss.item():.4f}  "
                f"critic_loss={critic_loss.item():.4f}  "
                f"q_mean={q_val:.2f}  q_max={q_pi.max().item():.2f}  "
                f"q_min={q_pi.min().item():.2f}"
            )

        # ── Periodic eval ────────────────────────────────────────────────
        if step % CFG["eval_every"] == 0 or step == CFG["smoke_steps"]:
            _run_eval(model, device, step, data_dir, results_dir, mode)

        # ── Sprint discipline checkpoint ─────────────────────────────────
        if mode == "full" and step == CFG["checkpoint_step"]:
            ckpt_path = ckpt_dir / f"dql_step{step}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "opt_policy": opt_policy.state_dict(),
                "opt_critic": opt_critic.state_dict(),
            }, ckpt_path)
            log.info(f"[CHECKPOINT] Step {step}: saved to {ckpt_path}")
            log.info("[CHECKPOINT] Halting for Karthik's review. Re-launch to continue.")
            sys.exit(0)

    # ── Smoke completion ─────────────────────────────────────────────────────
    wall_time = time.time() - t0_total
    log.info(f"Smoke complete: {n_steps} steps in {wall_time:.1f}s "
             f"({wall_time / n_steps * 1000:.1f}ms/step)")
    log.info(f"Projected full-run (100k steps): {wall_time / n_steps * 100_000 / 3600:.1f}h")

    _smoke_pass_checks(model, device, reward_buf, n_steps, log_dir, ckpt_dir)

    ckpt_path = ckpt_dir / "dql_smoke_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    log.info(f"Smoke checkpoint saved: {ckpt_path}")


def _run_eval(model, device, step, data_dir, results_dir, mode):
    """Run T-60 eval harness if data available; otherwise skip with warning."""
    try:
        sys.path.insert(0, ROOT)
        from methods.diffusion_ql.policy import DiffusionQLPolicy
        from experiments.prepare_postbreak import evaluate
        policy = DiffusionQLPolicy(model, device)
        result = evaluate(policy, f"dql_{mode}_step{step}",
                          data_dir=data_dir, results_dir=results_dir)
        log.info(
            f"[EVAL step={step}] all_days=${result['all_days']['per_kw_yr']:.2f}/kW-yr  "
            f"ex_fern=${result['ex_fern']['per_kw_yr']:.2f}/kW-yr  "
            f"fern=${result['fern_only']['per_kw_yr']:.2f}/kW-yr"
        )
    except Exception as exc:
        log.warning(f"[EVAL step={step}] skipped: {exc}")


def _smoke_pass_checks(model, device, reward_buf, n_steps, log_dir, ckpt_dir):
    """Log smoke pass/fail criteria and write SMOKE_REPORT.md."""
    log.info("=" * 60)
    log.info("SMOKE PASS CHECKS")
    log.info("=" * 60)

    # Reward statistics
    r = np.array(reward_buf)
    log.info(f"[REWARD STATS] mean={r.mean():.3f}  std={r.std():.3f}  "
             f"min={r.min():.3f}  max={r.max():.3f}  "
             f"(batches sample from 19,584 transitions)")
    if r.mean() < SMOKE_MIN_REWARD or r.mean() > SMOKE_MAX_REWARD:
        log.warning(f"[REWARD WARN] mean reward {r.mean():.3f} outside expected "
                    f"[{SMOKE_MIN_REWARD}, {SMOKE_MAX_REWARD}] $/interval")
        log.warning("  → may still be in mixed-unit space; check data_loader")
    else:
        log.info("[REWARD OK] reward scale consistent with physical $/interval")

    # Action distribution at final step
    model.eval()
    with torch.no_grad():
        test_obs = torch.zeros(256, 398, device=device)
        actions_pu = model.sample_action(test_obs).cpu().numpy()
    model.train()

    dim_names = ["p_energy", "c_regup", "c_regdn", "c_rrs", "c_ecrs", "c_nsrs"]
    log.info("[ACTION DISTRIBUTION at step_end]")
    action_flags = []
    for i, name in enumerate(dim_names):
        col = actions_pu[:, i]
        mean_abs = np.mean(np.abs(col))
        p95      = np.percentile(np.abs(col), 95)
        mean_mw  = mean_abs * 10.0
        flag = "WARN-AS-BIAS" if i > 0 and mean_mw > SMOKE_MAX_AS_MEAN else "ok"
        action_flags.append(flag)
        log.info(f"  [{flag}] {name}: mean|x|={mean_abs:.3f} ({mean_mw:.1f}MW)  "
                 f"P95|x|={p95:.3f} ({p95*10:.1f}MW)")

    smoke_pass = "PASS" if all(f == "ok" for f in action_flags) else "FLAG"
    log.info(f"[SMOKE RESULT] {smoke_pass}")

    # Write report placeholder (full report filled in after actual Narnia run)
    report_path = Path(ckpt_dir).parent.parent.parent / "methods" / "diffusion_ql" / "SMOKE_REPORT.md"
    with open(report_path, "w") as f:
        f.write(f"# Diffusion-QL Smoke Report\n")
        f.write(f"**Steps:** {n_steps}\n")
        f.write(f"**Smoke result:** {smoke_pass}\n\n")
        f.write(f"## Reward stats\n")
        f.write(f"mean={r.mean():.3f}  std={r.std():.3f}  "
                f"min={r.min():.3f}  max={r.max():.3f}\n\n")
        f.write(f"## Action distribution\n")
        for i, (name, flag) in enumerate(zip(dim_names, action_flags)):
            f.write(f"- {name}: {flag}\n")
        f.write(f"\n*Full loss curves and eval metrics filled in after Narnia run.*\n")
    log.info(f"SMOKE_REPORT.md written to {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--train-path", default="data/expert_trajectories/receding_horizon_postbreak_train.npz")
    ap.add_argument("--val-path",   default="data/expert_trajectories/receding_horizon_postbreak_val.npz")
    ap.add_argument("--data-dir",   default="data/processed")
    ap.add_argument("--results-dir", default="data/results")
    args = ap.parse_args()

    log.info(f"Diffusion-QL training: mode={args.mode}  gpu={args.gpu}")
    log.info(f"Config: {CFG}")

    train(
        mode=args.mode,
        gpu=args.gpu,
        train_path=args.train_path,
        val_path=args.val_path,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
