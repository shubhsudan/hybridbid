"""
QDT training pipeline (3 stages).

Usage:
  python -m methods.qdt.train --stage 1 --mode smoke [--gpu 0]   # CQL 5k steps
  python -m methods.qdt.train --stage 1 --mode full  [--gpu 0]   # CQL 50k steps
  python -m methods.qdt.train --stage 2  [--gpu 0]               # RTG relabeling (fast)
  python -m methods.qdt.train --stage 3 --mode smoke [--gpu 0]   # DT 1k steps
  python -m methods.qdt.train --stage 3 --mode full  [--gpu 0]   # DT 50k steps

Sprint discipline:
  - Stage 1 full: sys.exit(0) at 50k checkpoint
  - Stage 3 full: sys.exit(0) at 50k checkpoint
  - Stage 2 (relabeling): runs to completion, no checkpoint needed
  - No auto-continuation of any stage
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

from methods.qdt.data_loader import (
    PostbreakDataset, make_cql_loader, relabel_rtg, save_relabeled,
    SequenceDataset, make_sequence_loader, K, OBS_DIM, ACT_DIM
)
from methods.qdt.model import CQLCritic, DecisionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("qdt")

# ── Hyperparameters ────────────────────────────────────────────────────────
CFG_S1 = dict(
    lr            = 3e-4,
    batch_size    = 256,
    gamma         = 0.99,
    tau           = 0.005,
    alpha_cql     = 1.0,      # CQL conservatism weight (paper default, medium-replay)
    n_rand_actions = 10,      # Number of random actions sampled for CQL penalty
    log_every     = 500,
    smoke_steps   = 5_000,
    full_steps    = 50_000,
    checkpoint_step = 50_000,
)

CFG_S3 = dict(
    lr          = 1e-4,
    batch_size  = 64,
    log_every   = 200,
    smoke_steps = 1_000,
    full_steps  = 50_000,
    checkpoint_step = 50_000,
)

# Inference: target RTG at eval time. Set to P95 of training Q-values after Stage 2.
# Populated dynamically from relabeled dataset at Stage 3 eval time.
TARGET_RTG = None

CKPT_DIR  = Path(ROOT) / "checkpoints" / "sprint" / "qdt"
LOG_DIR   = Path(ROOT) / "logs" / "sprint"


def _setup_dirs():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _add_file_handler(name: str):
    fh = logging.FileHandler(LOG_DIR / f"qdt_{name}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: CQL Critic
# ─────────────────────────────────────────────────────────────────────────────

def build_infinite_cql(npz_path: str, batch_size: int):
    dataset = PostbreakDataset(npz_path)
    while True:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        yield from loader


def run_stage1(mode: str, gpu: int, train_path: str, data_dir: str,
               results_dir: str, resume_ckpt: str = ""):
    """Train CQL critic. Smoke: 5k steps. Full: 50k steps with sys.exit checkpoint."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Stage 1 CQL | device={device} | mode={mode}")
    log.info(f"Config: {CFG_S1}")

    model = CQLCritic().to(device)
    opt   = torch.optim.Adam(
        list(model.q1.parameters()) + list(model.q2.parameters()), lr=CFG_S1["lr"]
    )

    start_step = 0
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        log.info(f"Resumed from {resume_ckpt} at step {start_step}")

    n_steps   = CFG_S1["smoke_steps"] if mode == "smoke" else CFG_S1["full_steps"]
    data_iter = build_infinite_cql(train_path, CFG_S1["batch_size"])
    t0        = time.time()

    for step in range(start_step + 1, n_steps + 1):
        obs, act, rew, next_obs, done, next_act, sarsa_done = next(data_iter)
        obs        = obs.to(device)
        act        = act.to(device)
        rew        = rew.to(device).unsqueeze(1)
        next_obs   = next_obs.to(device)
        done       = done.to(device).unsqueeze(1)        # always 0; no terminal states
        next_act   = next_act.to(device)
        sarsa_done = sarsa_done.to(device).unsqueeze(1)  # 1 at CT-midnight boundaries only

        # ── TD target (in-sample SARSA-style bootstrap) ──────────────────
        # Use the dataset's actual next action. This keeps the bootstrap in-distribution
        # and avoids the OOD pessimism that random actions cause with a conservative critic.
        # sarsa_done zeros the bootstrap at CT-midnight episode boundaries where
        # next_act belongs to the following day's episode, not the current one.
        with torch.no_grad():
            q_tgt = rew + CFG_S1["gamma"] * (1.0 - sarsa_done) * model.q_min_target(next_obs, next_act)

        # ── Bellman loss ──────────────────────────────────────────────────
        q1, q2 = model(obs, act)
        td_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

        # ── CQL penalty: push down Q on random OOD actions ─────────────
        # Sample n_rand random actions for CQL lower bound
        n_rand = CFG_S1["n_rand_actions"]
        obs_rep = obs.unsqueeze(1).expand(-1, n_rand, -1).reshape(-1, OBS_DIM)
        a_rand  = torch.cat([
            torch.rand(obs_rep.shape[0], 1, device=device) * 2 - 1,
            torch.rand(obs_rep.shape[0], 5, device=device),
        ], dim=1)
        q_rand  = model.q_min(obs_rep, a_rand).reshape(-1, n_rand)  # (B, n_rand)
        cql_loss = (q_rand.logsumexp(dim=1).mean() - q1.mean())
        total_loss = td_loss + CFG_S1["alpha_cql"] * cql_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        model.update_target(CFG_S1["tau"])

        if step % CFG_S1["log_every"] == 0:
            q_mean = q1.mean().item()
            q_max  = q1.max().item()
            log.info(f"step={step:>7}  td={td_loss.item():.4f}  cql={cql_loss.item():.4f}  "
                     f"total={total_loss.item():.4f}  q_mean={q_mean:.2f}  q_max={q_max:.2f}")

        # ── Sprint checkpoint ─────────────────────────────────────────────
        if mode == "full" and step == CFG_S1["checkpoint_step"]:
            ckpt_path = CKPT_DIR / f"qdt_s1_step{step}.pt"
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()},
                       ckpt_path)
            log.info(f"[CHECKPOINT Stage1] Step {step}: saved {ckpt_path}")
            log.info(f"[CHECKPOINT Stage1] Q_mean={q1.mean().item():.1f}  Q_max={q1.max().item():.1f}")
            log.info("[CHECKPOINT Stage1] Halting. Run Stage 2 (relabeling) then Stage 3.")
            sys.exit(0)

    wall = time.time() - t0
    log.info(f"Stage 1 {mode} done: {n_steps} steps in {wall:.1f}s ({wall/n_steps*1000:.1f}ms/step)")

    # Save smoke checkpoint for Stage 3 smoke test
    ckpt_path = CKPT_DIR / f"qdt_s1_{mode}_final.pt"
    torch.save({"step": n_steps, "model": model.state_dict(), "opt": opt.state_dict()},
               ckpt_path)
    log.info(f"Saved: {ckpt_path}")

    # ── Strengthened smoke checks ─────────────────────────────────────────────
    # Flag usage by stage (for traceability):
    #   done       (always 0.0) — Q-learning terminal flag; never zeros bootstrap (no terminal states)
    #   sarsa_done (truncated|last_idx, 69 positions) — zeros γ·Q(s',a') in Bellman target HERE only
    #   truncated  (68 CT-midnight positions) — used by SequenceDataset to validate DT context windows
    q_vals   = q1.detach().cpu().numpy().flatten()
    cql_val  = cql_loss.item()
    td_val   = td_loss.item()

    log.info(f"[SMOKE Stage1] Q distribution (per-transition critic Q-values on last batch):")
    log.info(f"  mean={q_vals.mean():.2f}  std={q_vals.std():.2f}  "
             f"min={q_vals.min():.2f}  P10={np.percentile(q_vals,10):.2f}  "
             f"P50={np.percentile(q_vals,50):.2f}  P90={np.percentile(q_vals,90):.2f}  "
             f"max={q_vals.max():.2f}")
    log.info(f"  NOTE: These are per-transition Q(s,a) values; effective horizon ~"
             f"{int(1/(1-CFG_S1['gamma']))} steps; expected range ≈ "
             f"mean_reward/(1-γ) ≈ ${5.95/(1-CFG_S1['gamma']):.0f}")

    # Check 1: No NaN
    if np.isnan(q_vals).any():
        log.error("[SMOKE FAIL] NaN in Q-values!")
    else:
        log.info("[SMOKE Stage1 C1] No NaN — OK")

    # Check 2: Q_mean positive (>0)
    if q_vals.mean() <= 0:
        log.error(f"[SMOKE FAIL Stage1 C2] Q_mean={q_vals.mean():.2f} ≤ 0 — "
                  f"TD bootstrap still pessimistic. Check in-sample a_next construction.")
    else:
        log.info(f"[SMOKE Stage1 C2] Q_mean={q_vals.mean():.2f} > 0 — OK")

    # Check 3: per-transition Q P90 > $200.
    # These are per-transition Q-values from the critic, NOT relabeled RTG values.
    # Expected scale: mean_reward/(1-γ) ≈ $595 for this dataset. $200 ≈ 1/3 of that,
    # a floor that flags badly undercalibrated critics. Actual P90 is informational.
    P90_FLOOR = 200.0
    q_p90 = float(np.percentile(q_vals, 90))
    if q_p90 < P90_FLOOR:
        log.warning(f"[SMOKE FLAG Stage1 C3] per-transition Q P90={q_p90:.2f} < ${P90_FLOOR:.0f} — "
                    f"critic undercalibrated; RTG labels will produce a low TARGET_RTG for DT.")
    else:
        log.info(f"[SMOKE Stage1 C3] per-transaction Q P90={q_p90:.2f} ≥ ${P90_FLOOR:.0f} — OK")

    # Check 4: CQL gradient direction correct.
    # Spec: "gradient direction correct" means Q_data ≥ Q_rand, i.e. cql_loss ≤ 0.
    # cql_loss = logsumexp(Q_rand) - Q_data
    #   < 0 : Q_data > Q_rand — correct conservatism direction; CQL reinforces this
    #   > 0 : Q_rand > Q_data — conservatism not yet established (expected early; FAIL at 50k)
    # Gradient of cql_loss always pushes Q_data UP and Q_rand DOWN regardless of sign —
    # sign only tells us whether we're already conservative.
    # CQL/TD ratio (informational): <0.01× → barely engaged (watch α); >10× → dominating.
    ratio = abs(cql_val) / (td_val + 1e-8)
    if cql_val > 0:
        # Q_rand > Q_data: conservatism not established.
        # At smoke (5k) this is a WARNING — CQL is still working toward the goal.
        # At full (50k) checkpoint this would be a FAIL.
        flag_str = "WARN-not-yet-conservative (OK at 5k smoke; FAIL if persists at 50k)"
        log.warning(f"[SMOKE WARN Stage1 C4] CQL={cql_val:.4f} > 0 — "
                    f"Q_rand > Q_data: conservatism not yet established. "
                    f"ratio={ratio:.4f}×  {flag_str}")
    else:
        # Q_data > Q_rand: correct direction.
        if ratio > 10.0:
            engagement = "FLAG-DOMINANT (CQL >> TD; reduce α_cql)"
        elif ratio < 0.01:
            engagement = "FLAG-WEAK (CQL barely engaged; may need larger α_cql)"
        else:
            engagement = "OK"
        log.info(f"[SMOKE Stage1 C4] CQL={cql_val:.4f}  TD={td_val:.4f}  "
                 f"ratio={ratio:.4f}×  direction=correct  engagement={engagement}")

    # Check 5: Manual bootstrap spot-check (10 transitions)
    # Verify next_act looks like actual dataset actions (not random, not shuffled).
    log.info("[SMOKE Stage1 C5] Bootstrap spot-check (last batch, 10 samples):")
    act_np      = act.detach().cpu().numpy()
    next_act_np = next_act.detach().cpu().numpy()
    s_done_np   = sarsa_done.detach().cpu().numpy().flatten()
    for k in range(min(10, len(act_np))):
        log.info(f"  [{k}] act[0]={act_np[k,0]:.4f}  next_act[0]={next_act_np[k,0]:.4f}  "
                 f"sarsa_done={s_done_np[k]:.0f}")

    log.info("[SMOKE Stage1] DONE — review C3/C4 before Stage 1 full.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: RTG relabeling
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(gpu: int, train_path: str, critic_ckpt: str, out_path: str):
    """Load Stage 1 critic, relabel entire training dataset with Q-values."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Stage 2 RTG relabeling | device={device}")

    model = CQLCritic().to(device)
    ckpt  = torch.load(critic_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    log.info(f"Loaded CQL critic from {critic_ckpt} (step {ckpt['step']})")

    rtg_values = relabel_rtg(train_path, model, device)
    save_relabeled(train_path, rtg_values, out_path)
    log.info(f"Stage 2 complete. Relabeled dataset at: {out_path}")
    return rtg_values


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Decision Transformer
# ─────────────────────────────────────────────────────────────────────────────

def build_infinite_seq(relabeled_path: str, batch_size: int):
    dataset = SequenceDataset(relabeled_path)
    while True:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        yield from loader


def run_stage3(mode: str, gpu: int, relabeled_path: str, data_dir: str,
               results_dir: str, resume_ckpt: str = ""):
    """Train Decision Transformer on relabeled sequences."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Stage 3 DT | device={device} | mode={mode}")
    log.info(f"Config: {CFG_S3}")

    model = DecisionTransformer().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=CFG_S3["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG_S3["full_steps"], eta_min=1e-5
    )

    start_step = 0
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        log.info(f"Resumed from {resume_ckpt} at step {start_step}")

    # Compute target RTG for eval: P90 of Q-values in relabeled dataset
    global TARGET_RTG
    rel_data = np.load(relabeled_path, allow_pickle=False)
    TARGET_RTG = float(np.percentile(rel_data["rtg"], 90))
    log.info(f"Target RTG for inference (P90 of training Q-values): {TARGET_RTG:.2f}")

    n_steps   = CFG_S3["smoke_steps"] if mode == "smoke" else CFG_S3["full_steps"]
    data_iter = build_infinite_seq(relabeled_path, CFG_S3["batch_size"])
    t0        = time.time()

    for step in range(start_step + 1, n_steps + 1):
        rtg, obs, act = next(data_iter)
        rtg = rtg.to(device)   # (B, K)
        obs = obs.to(device)   # (B, K, 398)
        act = act.to(device)   # (B, K, 6)

        # DT loss: MSE between predicted and target actions at every position
        # Teacher-forcing: feed ground-truth past actions
        act_in  = act.clone()
        act_out = model(rtg, obs, act_in)          # (B, K, 6)

        # Clamp target to valid p.u. range before computing loss
        act_tgt_e = torch.clamp(act[:, :, 0:1], -1.0, 1.0)
        act_tgt_a = torch.clamp(act[:, :, 1:],   0.0, 1.0)
        act_tgt   = torch.cat([act_tgt_e, act_tgt_a], dim=-1)

        loss = F.mse_loss(act_out, act_tgt)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step()
        sched.step()

        if step % CFG_S3["log_every"] == 0:
            log.info(f"step={step:>7}  dt_loss={loss.item():.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if step % 5_000 == 0 or step == CFG_S3["smoke_steps"]:
            _eval_dt(model, device, step, data_dir, results_dir, mode, TARGET_RTG)

        # ── Sprint checkpoint ─────────────────────────────────────────────
        if mode == "full" and step == CFG_S3["checkpoint_step"]:
            ckpt_path = CKPT_DIR / f"qdt_s3_step{step}.pt"
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()},
                       ckpt_path)
            log.info(f"[CHECKPOINT Stage3] Step {step}: saved {ckpt_path}")
            log.info(f"[CHECKPOINT Stage3] DT_loss={loss.item():.4f}")
            log.info("[CHECKPOINT Stage3] Halting for Karthik's review.")
            sys.exit(0)

    wall = time.time() - t0
    log.info(f"Stage 3 {mode} done: {n_steps} steps in {wall:.1f}s ({wall/n_steps*1000:.1f}ms/step)")

    ckpt_path = CKPT_DIR / f"qdt_s3_{mode}_final.pt"
    torch.save({"step": n_steps, "model": model.state_dict(), "opt": opt.state_dict(),
                "target_rtg": TARGET_RTG}, ckpt_path)
    log.info(f"Saved: {ckpt_path}")
    return model


def _eval_dt(model, device, step, data_dir, results_dir, mode, target_rtg):
    """Run T-60 eval harness with current DT policy."""
    try:
        from methods.qdt.policy import QDTPolicy
        from experiments.prepare_postbreak import evaluate
        policy = QDTPolicy(model, device, target_rtg)
        result = evaluate(policy, f"qdt_{mode}_step{step}",
                          data_dir=data_dir, results_dir=results_dir)
        log.info(
            f"[EVAL step={step}] all_days=${result['all_days']['annualized_kw_yr']:.2f}/kW-yr  "
            f"ex_fern=${result['ex_fern']['annualized_kw_yr']:.2f}/kW-yr  "
            f"fern=${result['fern_only']['annualized_kw_yr']:.2f}/kW-yr"
        )
    except Exception as exc:
        log.warning(f"[EVAL step={step}] skipped: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",  type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--mode",   choices=["smoke", "full"], default="smoke")
    ap.add_argument("--gpu",    type=int, default=0)
    ap.add_argument("--train-path",      default="data/expert_trajectories/receding_horizon_postbreak_train.npz")
    ap.add_argument("--relabeled-path",  default="methods/qdt/dataset_relabeled.npz")
    ap.add_argument("--critic-ckpt",     default="",   help="Stage 1 checkpoint for Stage 2")
    ap.add_argument("--data-dir",        default="data/processed")
    ap.add_argument("--results-dir",     default="data/results")
    ap.add_argument("--resume",          default="", help="Checkpoint to resume from")
    args = ap.parse_args()

    _setup_dirs()
    _add_file_handler(f"s{args.stage}_{args.mode}")

    log.info(f"QDT Stage {args.stage} | mode={args.mode} | gpu={args.gpu}")

    if args.stage == 1:
        run_stage1(args.mode, args.gpu, args.train_path, args.data_dir,
                   args.results_dir, args.resume)
    elif args.stage == 2:
        ckpt = args.critic_ckpt or str(CKPT_DIR / "qdt_s1_full_final.pt")
        if not Path(ckpt).exists():
            # Try smoke checkpoint for smoke pipeline
            ckpt = str(CKPT_DIR / "qdt_s1_smoke_final.pt")
        run_stage2(args.gpu, args.train_path, ckpt, args.relabeled_path)
    elif args.stage == 3:
        run_stage3(args.mode, args.gpu, args.relabeled_path, args.data_dir,
                   args.results_dir, args.resume)


if __name__ == "__main__":
    main()
