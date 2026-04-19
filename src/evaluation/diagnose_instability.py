"""
Diagnostic evaluation for Stage 1 v5.9.1 training instability.

Reports per-checkpoint:
  - Mean/median $/day
  - Mode distribution (charge/discharge/idle %)
  - SoC violation count + mean SoC at violation steps
  - Mean SoC across test set
  - Mean energy_mag (p.u. magnitude from raw_action[3])
  - Alpha value stored in checkpoint

Usage:
  python -m src.evaluation.diagnose_instability
  python -m src.evaluation.diagnose_instability --device cpu
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.env.ercot_env import ERCOTBatteryEnv
from src.models.sac import SACAgent
from src.training.config import Stage1Config

TBEX_DAILY = 870.0
PERFECT_FORESIGHT_DAILY = 1519.0
TEST_START = "2025-10-01"
TEST_END   = "2025-12-04"

GOOD_STEPS = [50000, 175000, 250000, 575000, 600000, 850000]
BAD_STEPS  = [300000, 425000, 550000, 675000, 700000, 750000, 900000, 950000]


def load_alpha_from_checkpoint(checkpoint_path: str) -> float:
    """Extract the SAC entropy temperature alpha from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Try common storage locations
    for key in ("log_alpha", "alpha"):
        if key in ckpt:
            val = ckpt[key]
            if key == "log_alpha":
                return float(torch.exp(torch.tensor(val)) if not isinstance(val, torch.Tensor) else val.exp())
            return float(val)
    # Search inside nested dicts
    for top_key, top_val in ckpt.items():
        if isinstance(top_val, dict):
            for k, v in top_val.items():
                if "log_alpha" in k:
                    return float(torch.tensor(v).exp())
                if k == "alpha":
                    return float(v)
    return float("nan")


def evaluate_checkpoint(checkpoint_path: str, config: Stage1Config, device: str) -> dict:
    """Run deterministic rollout and collect diagnostic metrics."""
    config.device = device

    battery_config = dict(
        p_max=config.p_max, e_max=config.e_max,
        soc_min_frac=config.soc_min_frac, soc_max_frac=config.soc_max_frac,
        soc_initial_frac=config.soc_initial_frac,
        eta_ch=config.eta_ch, eta_dch=config.eta_dch,
        degradation_cost=config.degradation_cost,
    )
    env = ERCOTBatteryEnv(
        data_dir=config.data_dir,
        mode="energy_only",
        battery_config=battery_config,
        seq_len=config.seq_len,
        date_range=(TEST_START, TEST_END),
    )
    n_days = len(env.day_starts)

    agent = SACAgent(
        stage=1,
        device=device,
        n_prices=config.n_prices,
        n_prices_flat=getattr(config, "n_prices_flat", None),
        d_model=config.d_model,
        nhead=config.nhead,
        n_layers=config.n_layers,
        seq_len=config.seq_len,
        static_dim=config.static_dim,
        hidden_dim=config.hidden_dim,
        tau_gumbel=config.tau_gumbel_final,
    )
    agent.load_checkpoint(checkpoint_path)

    alpha = load_alpha_from_checkpoint(checkpoint_path)

    daily_revenues = []
    daily_modes = []
    daily_mean_soc = []
    daily_mean_emag = []
    violation_socs = []  # SoC values at each violation step
    total_violations = 0

    for day_idx in range(n_days):
        obs, _ = env.reset(options={"day_idx": day_idx})
        day_rev = 0.0
        mode_counts = {0: 0, 1: 0, 2: 0}
        socs = []
        emags = []
        done = False

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            day_rev += info["energy_revenue"] * config.p_max
            mode_counts[info["mode"]] += 1
            socs.append(info["soc"])

            # energy_mag from raw_action[3] (always non-negative per env._parse_action)
            raw = info.get("raw_action", action)
            emags.append(float(abs(raw[3])))

            if info["soc_violated"]:
                total_violations += 1
                violation_socs.append(info["soc"])

        daily_revenues.append(day_rev)
        total_steps = sum(mode_counts.values())
        daily_modes.append([
            mode_counts[0] / total_steps,
            mode_counts[1] / total_steps,
            mode_counts[2] / total_steps,
        ])
        daily_mean_soc.append(np.mean(socs))
        daily_mean_emag.append(np.mean(emags))

    revenues = np.array(daily_revenues)
    modes = np.array(daily_modes)

    return {
        "mean": revenues.mean(),
        "median": np.median(revenues),
        "std": revenues.std(),
        "best": revenues.max(),
        "worst": revenues.min(),
        "n_negative": int((revenues < 0).sum()),
        "charge_pct": modes[:, 0].mean() * 100,
        "discharge_pct": modes[:, 1].mean() * 100,
        "idle_pct": modes[:, 2].mean() * 100,
        "mean_soc": np.mean(daily_mean_soc),
        "mean_emag": np.mean(daily_mean_emag),
        "violations": total_violations,
        "mean_soc_at_violation": float(np.mean(violation_socs)) if violation_socs else float("nan"),
        "alpha": alpha,
        "n_days": n_days,
    }


def print_results(label: str, step: int, r: dict):
    tag = "GOOD" if step in GOOD_STEPS else "BAD "
    print(f"\n[{tag}] step {step:>7d}  {label}")
    print(f"  Revenue    mean ${r['mean']:>8.2f}/day   median ${r['median']:>8.2f}/day   "
          f"std ${r['std']:>7.2f}   best ${r['best']:>8.2f}   worst ${r['worst']:>8.2f}   "
          f"n_negative={r['n_negative']}/{r['n_days']}")
    print(f"  Modes      ch={r['charge_pct']:>5.1f}%  dc={r['discharge_pct']:>5.1f}%  "
          f"idle={r['idle_pct']:>5.1f}%")
    print(f"  SoC        mean={r['mean_soc']:>5.2f} MWh   violations={r['violations']}   "
          f"mean_soc@violation={r['mean_soc_at_violation']:.2f} MWh")
    print(f"  energy_mag mean={r['mean_emag']:.4f} p.u.")
    print(f"  Alpha      {r['alpha']:.6f}")


def main(device: str):
    config = Stage1Config()
    config.device = device

    all_steps = sorted(set(GOOD_STEPS + BAD_STEPS))
    results = {}

    print(f"\n{'='*70}")
    print(f"Stage 1 v5.9.1 Instability Diagnostic")
    print(f"Test set: {TEST_START} → {TEST_END}  |  Device: {device}")
    print(f"{'='*70}")

    for step in all_steps:
        ckpt = f"checkpoints/stage1/checkpoint_step{step}.pt"
        if not os.path.exists(ckpt):
            print(f"\n  [SKIP] step {step}: checkpoint not found")
            continue
        print(f"\n  Evaluating step {step}...", flush=True)
        r = evaluate_checkpoint(ckpt, config, device)
        results[step] = r
        label = "✓ good" if step in GOOD_STEPS else "✗ bad"
        print_results(label, step, r)

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"{'Step':>8}  {'Cat':4}  {'Mean$/d':>9}  {'Med$/d':>9}  "
          f"{'Ch%':>5}  {'Dc%':>5}  {'Id%':>5}  "
          f"{'MnSoC':>6}  {'Viols':>5}  {'EmgMn':>6}  {'Alpha':>8}")
    print(f"{'─'*8}  {'─'*4}  {'─'*9}  {'─'*9}  "
          f"{'─'*5}  {'─'*5}  {'─'*5}  "
          f"{'─'*6}  {'─'*5}  {'─'*6}  {'─'*8}")
    for step in all_steps:
        if step not in results:
            continue
        r = results[step]
        cat = "GOOD" if step in GOOD_STEPS else "BAD "
        print(f"{step:>8d}  {cat}  {r['mean']:>9.2f}  {r['median']:>9.2f}  "
              f"{r['charge_pct']:>5.1f}  {r['discharge_pct']:>5.1f}  {r['idle_pct']:>5.1f}  "
              f"{r['mean_soc']:>6.2f}  {r['violations']:>5d}  "
              f"{r['mean_emag']:>6.4f}  {r['alpha']:>8.6f}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.device)
