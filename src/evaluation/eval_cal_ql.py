"""
Driver script: evaluate Cal-QL 25k checkpoint on the T-60 frozen harness.

Usage:
    python experiments/eval_cal_ql_25k.py \
        --checkpoint checkpoints/sprint/cal_ql/calql_step25000.pt \
        --data-dir data/processed \
        --results-dir data/results \
        --output results/cal_ql_25k_eval.json

Outputs:
    data/results/eval_cal_ql_25k/trajectory.parquet
    data/results/eval_cal_ql_25k/summary.json
    data/results/eval_cal_ql_25k/comparison_card.md
    results/cal_ql_25k_eval.json   (full report with projection stats)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT)

from src.evaluation.eval_t60 import (
    P_MAX, E_MAX, ETA, SOC_INIT, SOC_MIN, SOC_MAX, DT,
    PRICE_COLS, SYSTEM_COLS,
    load_data, _find_t60_indices, _build_obs, project_action, evaluate,
)
from src.methods.cal_ql.eval_policy import CalQLPolicy


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(policy: CalQLPolicy, val_npz_path: str) -> None:
    """Load one obs from val NPZ and confirm shape, bounds, determinism."""
    data = np.load(val_npz_path, allow_pickle=False)
    ph = data["price_history"][0].astype(np.float32)   # (32, 12)
    sf = data["static_features"][0].astype(np.float32) # (14,)
    obs = {"price_history": ph, "static_features": sf}

    a1 = policy(obs)
    a2 = policy(obs)

    assert a1.shape == (6,), f"Bad shape: {a1.shape}"
    assert np.allclose(a1, a2), "Not deterministic!"
    assert -P_MAX <= a1[0] <= P_MAX, f"p_energy out of range: {a1[0]}"
    assert np.all(a1[1:] >= 0) and np.all(a1[1:] <= P_MAX), f"c_as out of range: {a1[1:]}"

    print("[sanity] shape (6,) OK | deterministic OK | p.u.-to-MW bounds OK")
    print(f"[sanity] sample action (MW): p_energy={a1[0]:.3f}  "
          f"c_regup={a1[1]:.3f}  c_regdn={a1[2]:.3f}  "
          f"c_rrs={a1[3]:.3f}  c_ecrs={a1[4]:.3f}  c_nsrs={a1[5]:.3f}")


# ---------------------------------------------------------------------------
# Projection frequency pass
# ---------------------------------------------------------------------------

def compute_projection_stats(
    policy: CalQLPolicy,
    data_dir: str,
) -> dict:
    """
    Run the full eval loop, tracking pre- vs post-projection actions.
    Mirrors harness loop exactly (same obs build, same SoC update).
    """
    print("[projection] Running projection-tracking pass ...")
    merged = load_data(data_dir)
    start_idx, end_idx, ts_ct = _find_t60_indices(merged)

    price_data  = merged[PRICE_COLS].values.astype(np.float32)
    system_data = merged[SYSTEM_COLS].values.astype(np.float32)

    n_steps = end_idx - start_idx
    policy.reset()
    soc = SOC_INIT

    n_projected = 0
    scale_factors = []          # (pre_as_sum / post_as_sum) when joint cap was binding
    pre_cap_sums  = []          # |p_energy| + sum(c_as) before projection

    for step in range(n_steps):
        idx = start_idx + step
        obs = _build_obs(price_data, system_data, ts_ct, idx, soc)
        raw_mw = policy(obs).astype(np.float64)   # (6,) physical MW, pre-projection

        proj_mw = project_action(raw_mw, soc).astype(np.float64)

        # Detect joint-cap binding: raw total > P_MAX (after individual clipping)
        raw_cap = abs(raw_mw[0]) + float(np.sum(np.clip(raw_mw[1:], 0.0, P_MAX)))
        if raw_cap > P_MAX + 1e-6:
            n_projected += 1
            pre_cap_sums.append(raw_cap)
            raw_as_sum = float(np.sum(np.clip(raw_mw[1:], 0.0, P_MAX)))
            proj_as_sum = float(np.sum(proj_mw[1:]))
            if proj_as_sum > 1e-9:
                scale_factors.append(proj_as_sum / raw_as_sum)

        # Replicate harness SoC update on projected action
        p_energy = float(proj_mw[0])
        if p_energy >= 0:
            soc -= p_energy / ETA * DT
        else:
            soc += abs(p_energy) * ETA * DT
        soc = float(np.clip(soc, SOC_MIN, SOC_MAX))

    frac_projected = n_projected / n_steps
    mean_scale = float(np.mean(scale_factors)) if scale_factors else float("nan")
    mean_pre_cap = float(np.mean(pre_cap_sums)) if pre_cap_sums else float("nan")

    print(f"[projection] Steps with joint cap binding: {n_projected}/{n_steps} "
          f"({frac_projected*100:.1f}%)")
    print(f"[projection] Mean pre-proj cap sum:  {mean_pre_cap:.3f} MW  "
          f"(limit {P_MAX:.1f} MW)")
    print(f"[projection] Mean AS scale factor:   {mean_scale:.4f}  "
          f"(1.0 = no scaling; < 1.0 = AS cut)")

    return {
        "n_steps_total": n_steps,
        "n_steps_joint_cap_binding": n_projected,
        "frac_steps_projected": round(frac_projected, 4),
        "mean_pre_proj_cap_sum_mw": round(mean_pre_cap, 4),
        "mean_as_scale_factor": round(mean_scale, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="checkpoints/sprint/cal_ql/calql_step25000.pt")
    parser.add_argument("--data-dir",    default="data/processed")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--output",      default="results/cal_ql_25k_eval.json")
    parser.add_argument("--val-npz",
        default="data/expert_trajectories/receding_horizon_postbreak_val.npz")
    args = parser.parse_args()

    # ── Load policy ──────────────────────────────────────────────────────────
    print(f"[eval] Loading checkpoint: {args.checkpoint}")
    policy = CalQLPolicy(args.checkpoint)

    # ── Sanity check ─────────────────────────────────────────────────────────
    sanity_check(policy, args.val_npz)

    # ── Canary check (cached result) ─────────────────────────────────────────
    canary_path = os.path.join(args.results_dir, "eval_milp_replay_ct", "summary.json")
    if os.path.exists(canary_path):
        with open(canary_path) as f:
            canary = json.load(f)
        canary_kw_yr = canary["all_days"]["annualized_kw_yr"]
        canary_ok = abs(canary_kw_yr - 58.40) < 58.40 * 0.02
        print(f"[canary] milp_replay_ct = ${canary_kw_yr:.4f}/kW-yr  "
              f"({'OK' if canary_ok else 'DRIFT — HALT'})")
        if not canary_ok:
            print("[canary] HARNESS DRIFT DETECTED. Halting.")
            sys.exit(1)
    else:
        print(f"[canary] WARNING: no milp_replay_ct result at {canary_path}")
        print("[canary] Skipping canary validation.")

    # ── Official harness eval ─────────────────────────────────────────────────
    print("\n[eval] Running frozen harness ...")
    policy.reset()
    summary = evaluate(
        policy,
        method_name="cal_ql_25k",
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )

    # ── AS revenue breakdown from trajectory parquet ─────────────────────────
    import pandas as pd
    traj_path = os.path.join(args.results_dir, "eval_cal_ql_25k", "trajectory.parquet")
    df = pd.read_parquet(traj_path)
    as_cols = ["as_rev_regup", "as_rev_regdn", "as_rev_rrs", "as_rev_ecrs", "as_rev_nsrs"]
    as_totals = {c: round(float(df[c].sum()), 2) for c in as_cols}
    total_rev = float(df["step_rev"].sum())
    as_rev_total = float(df[as_cols].sum().sum())
    energy_rev_total = float(df["energy_rev"].sum())
    as_share = as_rev_total / total_rev if total_rev != 0 else float("nan")

    print("\n[eval] AS revenue breakdown:")
    for c, v in as_totals.items():
        print(f"  {c}: ${v:>10,.2f}")
    print(f"  energy_rev:  ${energy_rev_total:>10,.2f}")
    print(f"  AS share:    {as_share*100:.1f}%")

    # ── Projection stats ─────────────────────────────────────────────────────
    proj_stats = compute_projection_stats(policy, args.data_dir)

    # ── Full report ──────────────────────────────────────────────────────────
    full_report = {
        "checkpoint": args.checkpoint,
        "canary_milp_replay_ct_kw_yr": canary_kw_yr if os.path.exists(canary_path) else None,
        "harness_summary": summary,
        "as_rev_breakdown": as_totals,
        "energy_rev_total": round(energy_rev_total, 2),
        "as_rev_total": round(as_rev_total, 2),
        "as_share": round(as_share, 4),
        "projection_stats": proj_stats,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[eval] Full report written to {args.output}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    cal_ql_kw_yr = summary["all_days"]["annualized_kw_yr"]
    bc_kw_yr = 29.16
    print(f"\n[verdict] Cal-QL 25k: {cal_ql_kw_yr:.2f} $/kW-yr vs BC: {bc_kw_yr:.2f} $/kW-yr")
    if cal_ql_kw_yr > bc_kw_yr + 0.05:
        print("[verdict] BEATS BC")
    elif cal_ql_kw_yr > bc_kw_yr - 0.5:
        print("[verdict] MATCHES BC (within 0.5 $/kW-yr)")
    else:
        print("[verdict] BELOW BC")


if __name__ == "__main__":
    main()
