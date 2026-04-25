"""
Cal-QL offline training entry point.

Usage:
  python -m methods.cal_ql.train_offline --mode smoke [--gpu 0]
  python -m methods.cal_ql.train_offline --mode full  [--gpu 0]
  python -m methods.cal_ql.train_offline --mode full  --resume checkpoints/sprint/cal_ql/calql_step25000.pt [--gpu 0]

Sprint discipline:
  - sys.exit(0) at checkpoint_step (25k). Do NOT auto-continue.
  - Smoke (5k) writes SMOKE_RESULTS.md then exits.
  - Kill criteria checked every eval_every (5k) steps.
  - One variable per experiment: do not change hyperparams mid-run.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, ROOT)

from methods.cal_ql.networks import Actor, TwinQ
from methods.cal_ql.calql_agent import CalQLAgent
from methods.cal_ql.data_loader import PostbreakDatasetCalQL, make_infinite_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calql")


def load_cfg(path: str = None) -> dict:
    cfg_path = path or str(Path(__file__).parent / "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_agent(cfg: dict, device: torch.device) -> tuple:
    actor     = Actor().to(device)
    twin_q    = TwinQ().to(device)
    twin_q_tgt = TwinQ().to(device)

    # Initialize target = online
    for p_tgt, p_src in zip(twin_q_tgt.parameters(), twin_q.parameters()):
        p_tgt.data.copy_(p_src.data)
    for p in twin_q_tgt.parameters():
        p.requires_grad_(False)

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=cfg["lr_actor"])
    critic_opt = torch.optim.Adam(twin_q.parameters(), lr=cfg["lr_critic"])

    agent = CalQLAgent(actor, twin_q, twin_q_tgt, actor_opt, critic_opt, cfg, device)
    return actor, twin_q, twin_q_tgt, agent


def _action_stats_for_mask(act_arr: np.ndarray, mask: np.ndarray,
                            dim_names: list) -> dict:
    """Per-dim action stats for a boolean mask over rows of act_arr."""
    sub = act_arr[mask]
    if len(sub) == 0:
        return {}
    stats = {}
    for i, name in enumerate(dim_names):
        col = sub[:, i]
        stats[name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "p5": float(np.percentile(col, 5)), "p95": float(np.percentile(col, 95)),
            "frac_zero": float((np.abs(col) < 1e-3).mean()),
            "n": int(mask.sum()),
        }
    return stats


def eval_on_val(actor: Actor, twin_q: TwinQ, val_path: str,
                device: torch.device, step: int) -> dict:
    """
    Lightweight val-NPZ eval: Q statistics and action distribution.

    Computes two action-distribution slices:
      - Global (all val transitions)
      - Negative-V_beh slice (val transitions where V_behavior < 0;
        ~17% of train — QDT's failure zone without a calibration floor)

    NOT the full T-60 harness (Karthik runs that separately after 50k).
    """
    from methods.cal_ql.data_loader import PostbreakDatasetCalQL
    val_ds = PostbreakDatasetCalQL(val_path, v_behavior_cache="", gamma=0.99)

    all_q, all_act = [], []
    actor.eval()
    twin_q.eval()

    with torch.no_grad():
        bs = 512
        N  = len(val_ds)
        for start in range(0, N, bs):
            end       = min(start + bs, N)
            obs_batch = val_ds.obs[start:end].to(device)
            a_det     = actor.deterministic(obs_batch)   # (B, 6)
            q_val     = twin_q.q_min(obs_batch, a_det)  # (B, 1)
            all_q.append(q_val.cpu().numpy())
            all_act.append(a_det.cpu().numpy())

    actor.train()
    twin_q.train()

    q_arr   = np.concatenate(all_q).squeeze()   # (N,)
    act_arr = np.concatenate(all_act)            # (N, 6)
    v_beh   = val_ds.v_beh.numpy()              # (N,)

    dim_names = ["p_energy", "c_regup", "c_regdn", "c_rrs", "c_ecrs", "c_nsrs"]

    # Global action stats
    act_stats = {}
    for i, name in enumerate(dim_names):
        col = act_arr[:, i]
        act_stats[name] = {
            "mean": float(col.mean()), "std": float(col.std()),
            "p5": float(np.percentile(col, 5)), "p95": float(np.percentile(col, 95)),
            "frac_zero": float((np.abs(col) < 1e-3).mean()),
        }

    # Negative-V_beh slice — states where calibration floor provides no protection
    neg_mask = v_beh < 0.0
    neg_act_stats = _action_stats_for_mask(act_arr, neg_mask, dim_names)

    result = {
        "step":    step,
        "Q_mean":  float(q_arr.mean()),
        "Q_max":   float(q_arr.max()),
        "Q_min":   float(q_arr.min()),
        "Q_p50":   float(np.percentile(q_arr, 50)),
        "Q_p90":   float(np.percentile(q_arr, 90)),
        "action_stats":     act_stats,
        "neg_vbeh_stats":   neg_act_stats,
        "neg_vbeh_n":       int(neg_mask.sum()),
        "neg_vbeh_frac":    float(neg_mask.mean()),
    }

    log.info(f"[EVAL step={step}] Q_mean={result['Q_mean']:.2f}  Q_max={result['Q_max']:.2f}  "
             f"Q_min={result['Q_min']:.2f}  Q_p50={result['Q_p50']:.2f}")
    log.info(f"[EVAL step={step}] Global action distribution (p.u.):")
    for name, s in act_stats.items():
        log.info(f"  {name}: mean={s['mean']:.3f}  std={s['std']:.3f}  "
                 f"p5={s['p5']:.3f}  p95={s['p95']:.3f}  frac_zero={s['frac_zero']:.3f}")
    log.info(f"[EVAL step={step}] Negative-V_beh slice ({result['neg_vbeh_n']} states, "
             f"{result['neg_vbeh_frac']*100:.1f}% of val — no calibration floor):")
    for name, s in neg_act_stats.items():
        log.info(f"  {name}: mean={s['mean']:.3f}  std={s['std']:.3f}  "
                 f"frac_zero={s['frac_zero']:.3f}")

    return result


def check_kill_criteria(metrics: dict, q_max_smoke: float, cfg: dict, step: int) -> bool:
    """
    Returns True if a kill criterion is met.
    Kill criteria for 25k–50k continuation (checked every eval_every).
    """
    q_max_now  = metrics["Q_max"]
    q_mean_now = metrics["Q_mean"]
    cql_term   = metrics.get("cql_term_last", 0.0)

    killed = False

    if q_max_smoke > 0 and q_max_now > cfg["kill_q_max_multiplier"] * q_max_smoke:
        log.error(f"[KILL] Step {step}: Q_max={q_max_now:.1f} > "
                  f"{cfg['kill_q_max_multiplier']}× Q_max_smoke ({q_max_smoke:.1f}). "
                  f"DQL-class divergence. Writing diagnostics and exiting.")
        killed = True

    if q_mean_now < cfg["kill_q_mean_floor"]:
        log.error(f"[KILL] Step {step}: Q_mean={q_mean_now:.2f} < "
                  f"{cfg['kill_q_mean_floor']}. QDT-class over-conservatism. "
                  f"Writing diagnostics and exiting.")
        killed = True

    if abs(cql_term) > cfg["kill_cql_term_max"]:
        log.error(f"[KILL] Step {step}: |CQL term|={abs(cql_term):.1f} > "
                  f"{cfg['kill_cql_term_max']}. Calibration mechanism broken. Exiting.")
        killed = True

    return killed


def save_checkpoint(step: int, actor: Actor, twin_q: TwinQ, twin_q_tgt: TwinQ,
                    agent: CalQLAgent, ckpt_dir: Path) -> str:
    ckpt_path = str(ckpt_dir / f"calql_step{step}.pt")
    torch.save({
        "step":         step,
        "actor":        actor.state_dict(),
        "twin_q":       twin_q.state_dict(),
        "twin_q_tgt":   twin_q_tgt.state_dict(),
        "actor_opt":    agent.actor_opt.state_dict(),
        "critic_opt":   agent.critic_opt.state_dict(),
    }, ckpt_path)
    return ckpt_path


def train(mode: str, gpu: int, train_path: str, val_path: str,
          v_beh_cache: str, resume_ckpt: str = ""):

    cfg    = load_cfg()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  mode={mode}  resume={resume_ckpt or 'none'}")
    log.info(f"Config: {cfg}")

    ckpt_dir = Path(ROOT) / "checkpoints" / "sprint" / "cal_ql"
    log_dir  = Path(ROOT) / "logs" / "sprint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"calql_{mode}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    # ── Build model and agent ────────────────────────────────────────────────
    actor, twin_q, twin_q_tgt, agent = build_agent(cfg, device)

    start_step  = 0
    q_max_smoke = 0.0   # set after smoke eval, used for kill threshold in full run

    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        twin_q.load_state_dict(ckpt["twin_q"])
        twin_q_tgt.load_state_dict(ckpt["twin_q_tgt"])
        agent.actor_opt.load_state_dict(ckpt["actor_opt"])
        agent.critic_opt.load_state_dict(ckpt["critic_opt"])
        start_step = ckpt["step"]
        q_max_smoke = ckpt.get("q_max_smoke", 0.0)
        log.info(f"Resumed from {resume_ckpt} at step {start_step}  "
                 f"Q_max_smoke={q_max_smoke:.1f}")

    n_steps = cfg["smoke_steps"] if mode == "smoke" else cfg["total_steps"]
    data_iter = make_infinite_loader(train_path, v_beh_cache, cfg["batch_size"], cfg["gamma"])

    # ── Training metrics accumulators ────────────────────────────────────────
    metric_buf: dict = {k: [] for k in
                        ["td_loss", "cql_term", "cql_loss", "Q_mean", "Q_max",
                         "actor_loss", "log_pi"]}
    cql_term_last = 0.0

    t0 = time.time()

    for step in range(start_step + 1, start_step + n_steps + 1):

        obs, act, rew, next_obs, sarsa_done, v_beh = next(data_iter)

        obs        = obs.to(device)
        act        = act.to(device)
        rew        = rew.to(device).unsqueeze(1)           # (B, 1)
        next_obs   = next_obs.to(device)
        sarsa_done = sarsa_done.to(device).unsqueeze(1)    # (B, 1)
        v_beh      = v_beh.to(device).unsqueeze(1)         # (B, 1)

        # ── NaN guard ────────────────────────────────────────────────────────
        if torch.isnan(rew).any() or torch.isnan(obs).any():
            log.error(f"[ABORT] NaN in input data at step {step}. Exiting.")
            sys.exit(1)

        # ── Update critic ─────────────────────────────────────────────────────
        c_info = agent.update_critic(obs, act, rew, next_obs, sarsa_done, v_beh)

        # ── Update actor ──────────────────────────────────────────────────────
        a_info = agent.update_actor(obs)

        # ── Polyak target update ──────────────────────────────────────────────
        agent.update_target()

        # ── NaN check on losses ──────────────────────────────────────────────
        if math.isnan(c_info["td_loss"]) or math.isnan(a_info["actor_loss"]):
            log.error(f"[ABORT] NaN in loss at step {step}: "
                      f"td_loss={c_info['td_loss']}  actor_loss={a_info['actor_loss']}. Exiting.")
            sys.exit(1)

        cql_term_last = c_info["cql_term"]

        for k in ["td_loss", "cql_term", "cql_loss", "Q_mean", "Q_max"]:
            metric_buf[k].append(c_info[k])
        metric_buf["actor_loss"].append(a_info["actor_loss"])
        metric_buf["log_pi"].append(a_info["log_pi"])

        # ── Logging every 500 steps ───────────────────────────────────────────
        if step % 500 == 0:
            def m(k): return np.mean(metric_buf[k][-500:]) if metric_buf[k] else 0
            log.info(
                f"step={step:>7}  td={m('td_loss'):.3f}  cql_term={m('cql_term'):.2f}  "
                f"Q_mean={m('Q_mean'):.2f}  Q_max={m('Q_max'):.2f}  "
                f"actor={m('actor_loss'):.3f}  log_pi={m('log_pi'):.3f}"
            )

        # ── Periodic eval (every eval_every) ──────────────────────────────────
        if step % cfg["eval_every"] == 0 or (mode == "smoke" and step == cfg["smoke_steps"]):
            eval_result = eval_on_val(actor, twin_q, val_path, device, step)
            eval_result["cql_term_last"] = cql_term_last

            if mode == "smoke" and step == cfg["smoke_steps"]:
                q_max_smoke = eval_result["Q_max"]
                log.info(f"[SMOKE] Q_max_smoke={q_max_smoke:.2f}  "
                         f"Q_mean={eval_result['Q_mean']:.2f}")

            elif mode == "full" and start_step > 0 and q_max_smoke > 0:
                # Kill criteria — only active for full run after smoke
                if check_kill_criteria(eval_result, q_max_smoke, cfg, step):
                    diag_path = ckpt_dir / f"diagnostics_step{step}.json"
                    with open(diag_path, "w") as f:
                        json.dump({**eval_result,
                                   "q_max_smoke": q_max_smoke,
                                   "step": step,
                                   "metric_history": {k: metric_buf[k][-100:]
                                                      for k in metric_buf}},
                                  f, indent=2)
                    log.error(f"[KILL] Diagnostics written to {diag_path}")
                    sys.exit(1)

        # ── Sprint checkpoint: sys.exit(0) at checkpoint_step ─────────────────
        if mode == "full" and step == cfg["checkpoint_step"]:
            eval_result = eval_on_val(actor, twin_q, val_path, device, step)
            q_max_at_ckpt = eval_result["Q_max"]
            ckpt_path = save_checkpoint(step, actor, twin_q, twin_q_tgt, agent, ckpt_dir)

            # Save Q_max_smoke into checkpoint so resume can load it
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            ckpt_data["q_max_smoke"] = q_max_smoke
            torch.save(ckpt_data, ckpt_path)

            log.info(f"[CHECKPOINT] Step {step}: checkpoint saved to {ckpt_path}")
            log.info(f"[CHECKPOINT] Q_mean={eval_result['Q_mean']:.2f}  "
                     f"Q_max={q_max_at_ckpt:.2f}  Q_p50={eval_result['Q_p50']:.2f}")
            log.info(f"[CHECKPOINT] Q_max_smoke (from smoke run): {q_max_smoke:.2f}")
            if q_max_smoke > 0:
                ratio = q_max_at_ckpt / max(q_max_smoke, 1)
                log.info(f"[CHECKPOINT] Q_max ratio vs smoke: {ratio:.2f}×  "
                         f"(kill threshold: {cfg['kill_q_max_multiplier']}×)")
            log.info("[CHECKPOINT] Halting for Karthik's review. "
                     "Re-launch with --mode full --resume <path> to continue.")
            sys.exit(0)

    # ── Smoke completion ─────────────────────────────────────────────────────
    if mode == "smoke":
        wall_time = time.time() - t0
        log.info(f"Smoke complete: {cfg['smoke_steps']} steps in {wall_time:.1f}s "
                 f"({wall_time / cfg['smoke_steps'] * 1000:.1f}ms/step)")
        log.info(f"Projected full (50k steps): {wall_time / cfg['smoke_steps'] * 50_000 / 3600:.2f}h")

        # Final eval for smoke results
        eval_result = eval_on_val(actor, twin_q, val_path, device, cfg["smoke_steps"])
        q_max_smoke = eval_result["Q_max"]
        q_mean_smoke = eval_result["Q_mean"]

        # Save smoke checkpoint
        ckpt_path = save_checkpoint(cfg["smoke_steps"], actor, twin_q, twin_q_tgt, agent, ckpt_dir)
        log.info(f"Smoke checkpoint: {ckpt_path}")

        # Write SMOKE_RESULTS.md
        _write_smoke_results(eval_result, metric_buf, cfg, wall_time, ckpt_path)

        log.info("[SMOKE] Review SMOKE_RESULTS.md and confirm before launching 25k full run.")
        sys.exit(0)

    # ── Full run completion (50k) ────────────────────────────────────────────
    wall_time = time.time() - t0
    log.info(f"Full run complete: {n_steps} steps in {wall_time:.1f}s")
    ckpt_path = save_checkpoint(start_step + n_steps, actor, twin_q, twin_q_tgt, agent, ckpt_dir)
    log.info(f"Final checkpoint: {ckpt_path}")


def _write_smoke_results(eval_result: dict, metric_buf: dict, cfg: dict,
                          wall_time: float, ckpt_path: str):
    """Write SMOKE_RESULTS.md for Karthik's review."""
    results_path = Path(ROOT) / "methods" / "cal_ql" / "SMOKE_RESULTS.md"

    act_stats     = eval_result.get("action_stats", {})
    neg_vbeh_stats = eval_result.get("neg_vbeh_stats", {})
    neg_vbeh_n    = eval_result.get("neg_vbeh_n", 0)
    neg_vbeh_frac = eval_result.get("neg_vbeh_frac", 0.0)

    # Pass/fail checks
    q_max  = eval_result["Q_max"]
    q_mean = eval_result["Q_mean"]
    cql    = np.mean(metric_buf["cql_term"][-500:]) if metric_buf["cql_term"] else 0
    cql_range = (min(metric_buf["cql_term"]) if metric_buf["cql_term"] else 0,
                 max(metric_buf["cql_term"]) if metric_buf["cql_term"] else 0)

    def degenerate(stats: dict) -> bool:
        # 0.999 threshold: c_rrs/c_ecrs are legitimately ~96% zero in MILP expert;
        # 0.99 would false-flag them. Only flag true dead-neuron collapse.
        return stats["frac_zero"] > 0.999 or stats["std"] < 5e-4

    action_flags = {name: "DEGENERATE" if degenerate(s) else "ok"
                    for name, s in act_stats.items()}
    q_bounded    = q_max < 50_000
    q_positive   = q_mean > 0
    cql_bounded  = abs(cql) < cfg["kill_cql_term_max"]
    action_ok    = all(f == "ok" for f in action_flags.values())

    smoke_pass = all([q_bounded, q_positive, cql_bounded, action_ok])

    # ── Adaptive kill threshold recommendation (Karthik's guidance) ──────────
    # V_beh max = $11,952. A correctly-learning Q must reach that range for Fern
    # states. Q_max_smoke in $8–15k = spike states engaged; 4× is reasonable.
    # Q_max_smoke < $3k = 4× would false-positive on legitimate Fern learning.
    if q_max >= 8_000:
        thresh_rec  = f"**4× = {q_max * 4:.0f}** (spike states engaged; 4× is a safe divergence signal)"
        thresh_mode = "4x"
    elif q_max >= 3_000:
        thresh_rec  = (f"**4× = {q_max * 4:.0f}** (borderline — inspect 4× breach but also watch "
                       f"absolute 50k threshold)")
        thresh_mode = "4x_inspect"
    else:
        thresh_rec  = (f"**Absolute 50k** (4× = {q_max * 4:.0f} would false-positive on Fern states; "
                       f"use 4× = {q_max * 4:.0f} as inspection-only, firm guard = $50,000)")
        thresh_mode = "absolute_50k"
    # ─────────────────────────────────────────────────────────────────────────

    lines = [
        "# Cal-QL Smoke Results",
        f"**Steps:** {cfg['smoke_steps']}",
        f"**Mode:** offline, smoke",
        f"**Wall time:** {wall_time:.1f}s ({wall_time/cfg['smoke_steps']*1000:.1f}ms/step)",
        f"**Smoke result:** {'PASS' if smoke_pass else 'FAIL'}",
        "",
        "## Smoke pass criteria",
        "| Criterion | Value | Status |",
        "|-----------|-------|--------|",
        f"| Q_max bounded (<50k) | {q_max:.2f} | {'✓' if q_bounded else '✗ FAIL'} |",
        f"| Q_mean > 0 | {q_mean:.2f} | {'✓' if q_positive else '✗ FAIL'} |",
        f"| CQL term bounded | [{cql_range[0]:.2f}, {cql_range[1]:.2f}] | {'✓' if cql_bounded else '✗ FAIL'} |",
        f"| Action dist. non-degenerate | — | {'✓' if action_ok else '✗ FAIL'} |",
        "",
        "## Q-value statistics (val NPZ, deterministic policy)",
        f"- **Q_max_smoke: {q_max:.2f}**",
        f"- Q_mean:  {q_mean:.2f}",
        f"- Q_min:   {eval_result['Q_min']:.2f}",
        f"- Q_p50:   {eval_result['Q_p50']:.2f}",
        f"- Q_p90:   {eval_result['Q_p90']:.2f}",
        "",
        "## Kill threshold recommendation for 25k–50k run",
        f"*(V_beh max = $11,952; correctly-learning Q must reach spike-state range)*",
        f"- Q_max_smoke = **{q_max:.2f}**",
        f"- Recommended kill guard: {thresh_rec}",
        f"- Mode: `{thresh_mode}`",
        (f"- To apply: update `kill_q_max_multiplier` in config.yaml (4×) or "
         f"treat 50k absolute as hard guard and use 4× as alert-only."),
        "",
        "## Calibration term (CQL term, last 500 steps avg)",
        f"- Mean: {cql:.3f}",
        f"- Range: [{cql_range[0]:.3f}, {cql_range[1]:.3f}]",
        "",
        "## Action distribution — global (val NPZ, p.u.)",
    ]
    for name, stats in act_stats.items():
        flag = action_flags.get(name, "?")
        lines.append(f"- **{name}** [{flag}]: mean={stats['mean']:.3f}  "
                     f"std={stats['std']:.3f}  p5={stats['p5']:.3f}  "
                     f"p95={stats['p95']:.3f}  frac_zero={stats['frac_zero']:.3f}")

    # Negative-V_beh slice
    lines += [
        "",
        f"## Action distribution — negative-V_beh slice ({neg_vbeh_n} states, "
        f"{neg_vbeh_frac*100:.1f}% of val)",
        "*(States with V_behavior < 0 — no calibration floor; QDT's failure zone in miniature)*",
    ]
    if neg_vbeh_stats:
        global_means = {n: act_stats[n]["mean"] for n in act_stats}
        for name, stats in neg_vbeh_stats.items():
            g_mean = global_means.get(name, 0.0)
            drift  = stats["mean"] - g_mean
            flag   = "DRIFT" if abs(drift) > 0.15 else "ok"
            lines.append(f"- **{name}** [{flag}]: mean={stats['mean']:.3f} "
                         f"(global {g_mean:.3f}, drift={drift:+.3f})  "
                         f"std={stats['std']:.3f}  frac_zero={stats['frac_zero']:.3f}")
    else:
        lines.append("*(no negative-V_beh states in val split)*")

    lines += [
        "",
        "## Training loss (last 500 steps avg)",
        f"- TD loss: {np.mean(metric_buf['td_loss'][-500:]):.4f}",
        f"- CQL loss: {np.mean(metric_buf['cql_loss'][-500:]):.4f}",
        f"- Actor loss: {np.mean(metric_buf['actor_loss'][-500:]):.4f}",
        f"- log_pi: {np.mean(metric_buf['log_pi'][-500:]):.3f}",
        "",
        "## Checkpoint",
        f"- `{ckpt_path}`",
        "",
        "## Next step",
        "Karthik reviews. If PASS, set kill threshold per recommendation above, then:",
        "```",
        "CUDA_VISIBLE_DEVICES=<gpu> python -m methods.cal_ql.train_offline --mode full --gpu 0",
        "```",
        "Halts at step 25k (sys.exit 0). Resume with `--resume checkpoints/sprint/cal_ql/calql_step25000.pt`.",
    ]

    with open(results_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"SMOKE_RESULTS.md written to {results_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    choices=["smoke", "full"], required=True)
    ap.add_argument("--gpu",     type=int, default=0)
    ap.add_argument("--train-path", default="data/expert_trajectories/receding_horizon_postbreak_train.npz")
    ap.add_argument("--val-path",   default="data/expert_trajectories/receding_horizon_postbreak_val.npz")
    ap.add_argument("--v-beh-cache", default="data/cal_ql/V_behavior.npy")
    ap.add_argument("--resume",  default="", help="Checkpoint to resume from (full mode only)")
    args = ap.parse_args()

    log.info(f"Cal-QL offline training: mode={args.mode}  gpu={args.gpu}")
    train(
        mode=args.mode,
        gpu=args.gpu,
        train_path=args.train_path,
        val_path=args.val_path,
        v_beh_cache=args.v_beh_cache,
        resume_ckpt=args.resume,
    )


if __name__ == "__main__":
    main()
