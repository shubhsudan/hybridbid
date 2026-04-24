"""
experiments/prepare_postbreak.py

Method-agnostic eval harness for the T-60 post-break window
(2026-01-01 → 2026-02-23, 54 days, 15,552 five-minute steps).

Policy contract
---------------
class PolicyInterface:
    def reset(self) -> None: ...
    def __call__(self, obs: dict) -> np.ndarray:  # shape (6,), physical MW
        # obs = {"price_history": (32, 12) float32, "static_features": (14,) float32}
        # returns [p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs] in MW
        # p_energy signed (+ = discharge, − = charge); c_as non-negative

Revenue (physical $, per step)
-------------------------------
    energy_rev = p_energy_mw * rt_lmp * dt          # signed
    as_rev_j   = c_as_j_mw  * rt_mcpc_j * dt        # availability-based, unconditional

Feasibility projection (numpy, NOT ercot_env.project_co_optimize)
------------------------------------------------------------------
    AS sustain durations match ERCOT spec (feasibility.py has wrong values):
        regup=1.0h, regdn=1.0h, rrs=1/6h, ecrs=1/4h, nsrs=0.5h
    Joint shared capacity: |p_energy| + sum(c_as) <= P_max  (matches MILP)

Outputs per method
------------------
    data/results/eval_{method_name}/
        trajectory.parquet   # per-interval records
        summary.json         # aggregate metrics
        comparison_card.md   # human-readable
"""

from __future__ import annotations

import glob
import json
import os
from datetime import date, datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

P_MAX = 10.0       # MW
E_MAX = 20.0       # MWh
ETA = 0.95         # η_ch = η_dch (Li et al. Table I)
SOC_MIN = 2.0      # 10% of 20 MWh
SOC_MAX = 18.0     # 90% of 20 MWh
SOC_INIT = 10.0    # 50% of 20 MWh (fixed initial, per paper)
DT = 5.0 / 60.0   # hours per 5-min step

SEQ_LEN = 32
STEPS_PER_DAY = 288

EVAL_START_CT = date(2026, 1, 1)
EVAL_END_CT = date(2026, 2, 23)
EVAL_DAYS = 54
FERN_DATE = date(2026, 1, 26)  # Winter Storm Fern

# Fleet benchmarks — commit bd07a9c, 54-day T-60 window, DO NOT RECOMPUTE
FLEET_MEDIAN_KW_YR = 24.93
FLEET_TOP_Q_KW_YR = 32.23

# MILP replay validation target (CLARABEL, 54 days solved fresh)
MILP_T60_USD = 90_814.0

# Column ordering matches ercot_env.PRICE_COLS and SYSTEM_COLS exactly
PRICE_COLS = [
    "rt_lmp",
    "rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs",
    "dam_spp",
    "dam_as_regup", "dam_as_regdn", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs",
]
SYSTEM_COLS = [
    "total_load_mw", "load_forecast_mw",
    "wind_actual_mw", "wind_forecast_mw",
    "solar_actual_mw", "solar_forecast_mw",
    "net_load_mw",
]
# Normalization scales match ercot_env._system_scales
SYSTEM_SCALES = np.array(
    [50_000, 50_000, 15_000, 15_000, 10_000, 10_000, 40_000], dtype=np.float32
)

# AS sustain durations (ERCOT correct; feasibility.py values are WRONG — see EVAL_HARNESS_RECON.md §4)
# Order: [regup, regdn, rrs, ecrs, nsrs] — aligns with action dims 1:6
AS_SUSTAIN_H = np.array([1.0, 1.0, 1.0 / 6.0, 1.0 / 4.0, 0.5], dtype=np.float64)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> pd.DataFrame:
    """Load and merge all three Parquet tables from processed/ directory."""

    def _read(directory: str) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No Parquet files in {directory}")
        return pd.concat([pd.read_parquet(f) for f in files]).sort_index()

    ep = _read(os.path.join(data_dir, "energy_prices"))
    ap = _read(os.path.join(data_dir, "as_prices"))
    sc = _read(os.path.join(data_dir, "system_conditions"))

    for df in [ep, ap, sc]:
        if "is_post_rtcb" in df.columns:
            df.drop(columns=["is_post_rtcb"], inplace=True)

    merged = ep.join(ap, how="outer").join(sc, how="outer")

    # Load through 2026-02-24 UTC to capture all of Feb 23 in Central Time
    # (Feb 23 18:00–24:00 CT = Feb 24 00:00–06:00 UTC, which UTC-date "2026-02-23" excludes)
    # _find_t60_indices trims to CT date ≤ EVAL_END_CT, so Feb 24 UTC rows are discarded there.
    merged = merged.loc["2025-12-28":"2026-02-24"]

    merged[PRICE_COLS] = merged[PRICE_COLS].fillna(0.0)
    merged[SYSTEM_COLS] = merged[SYSTEM_COLS].ffill().fillna(0.0)

    if len(merged) == 0:
        raise ValueError("No data loaded for T-60 window. Check data_dir.")

    return merged


def _find_t60_indices(merged: pd.DataFrame):
    """Return (start_idx, end_idx, ts_ct) for T-60 eval window in Central Time."""
    ts = merged.index
    if ts.tz is not None:
        ts_ct = ts.tz_convert("US/Central")
    else:
        ts_ct = ts

    ct_dates = np.array([t.date() for t in ts_ct])

    start_mask = ct_dates == EVAL_START_CT
    end_mask = ct_dates == EVAL_END_CT

    if not start_mask.any():
        raise ValueError(f"No data found for T-60 start {EVAL_START_CT}")
    if not end_mask.any():
        raise ValueError(f"No data found for T-60 end {EVAL_END_CT}")

    start_idx = int(np.where(start_mask)[0][0])
    end_idx = int(np.where(end_mask)[0][-1]) + 1  # exclusive

    n_steps = end_idx - start_idx
    expected = EVAL_DAYS * STEPS_PER_DAY
    if n_steps != expected:
        print(
            f"[prepare_postbreak] WARNING: expected {expected} T-60 steps, "
            f"got {n_steps}. DST or data gap?"
        )

    return start_idx, end_idx, ts_ct


# ---------------------------------------------------------------------------
# Observation construction (replicates ERCOTBatteryEnv._get_observation standard mode)
# ---------------------------------------------------------------------------

def _get_time_features(ts) -> np.ndarray:
    """Cyclical time encoding matching ercot_env._get_time_features."""
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    month = ts.month
    return np.array(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ],
        dtype=np.float32,
    )


def _build_obs(
    price_data: np.ndarray,
    system_data: np.ndarray,
    ts_ct,
    idx: int,
    soc: float,
) -> dict:
    """Build observation dict matching ERCOTBatteryEnv._get_observation standard mode."""
    start = max(0, idx - SEQ_LEN + 1)
    price_history = price_data[start : idx + 1].copy()
    if len(price_history) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(price_history), 12), dtype=np.float32)
        price_history = np.vstack([pad, price_history])

    system = system_data[idx] / SYSTEM_SCALES
    time_feats = _get_time_features(ts_ct[idx])
    soc_frac = np.array([soc / E_MAX], dtype=np.float32)
    static_features = np.concatenate([system, time_feats, soc_frac]).astype(np.float32)

    return {
        "price_history": price_history.astype(np.float32),
        "static_features": static_features,
    }


# ---------------------------------------------------------------------------
# Feasibility projection
# ---------------------------------------------------------------------------

def project_action(action_mw: np.ndarray, soc: float) -> np.ndarray:
    """
    Project 6D physical MW action to feasibility.

    action_mw: [p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs] in MW
      p_energy signed (+ = discharge, − = charge)
      c_as non-negative

    Uses ERCOT-correct AS sustain durations and joint shared capacity.
    Does NOT call src/models/feasibility.project_co_optimize (wrong sustains).
    """
    p_energy = float(action_mw[0])
    c_as = action_mw[1:].astype(np.float64).copy()  # [regup, regdn, rrs, ecrs, nsrs]

    # 1. Clip AS magnitudes to [0, P_max]
    c_as = np.clip(c_as, 0.0, P_MAX)

    # 2. Clip energy to ±P_max, then SoC limits
    p_energy = float(np.clip(p_energy, -P_MAX, P_MAX))
    soc_headroom_dch = max(0.0, soc - SOC_MIN)
    soc_headroom_chg = max(0.0, SOC_MAX - soc)

    if p_energy >= 0:  # discharge
        max_dch_mw = soc_headroom_dch * ETA / DT
        p_energy = min(p_energy, min(P_MAX, max_dch_mw))
    else:  # charge
        max_chg_mw = soc_headroom_chg / (ETA * DT)
        p_energy = max(p_energy, max(-P_MAX, -max_chg_mw))

    # 3. AS SoC sustain feasibility (individual constraints, matching MILP formulation)
    # Discharge-direction (regup[0], rrs[2], ecrs[3], nsrs[4]):
    #   c_j * sustain_h[j] / eta <= (soc - soc_min)
    for i in [0, 2, 3, 4]:
        max_c = soc_headroom_dch * ETA / AS_SUSTAIN_H[i]
        c_as[i] = min(c_as[i], max_c)

    # Charge-direction (regdn[1]):
    #   c_regdn * sustain_h[1] * eta <= (soc_max - soc)
    max_regdn = soc_headroom_chg / (ETA * AS_SUSTAIN_H[1])
    c_as[1] = min(c_as[1], max_regdn)

    # 4. Joint shared capacity: |p_energy| + sum(c_as) <= P_max
    total = abs(p_energy) + float(np.sum(c_as))
    if total > P_MAX:
        as_sum = float(np.sum(c_as))
        as_budget = max(0.0, P_MAX - abs(p_energy))
        if as_sum > 1e-9:
            c_as *= as_budget / as_sum

    return np.array([p_energy, *c_as], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    policy,
    method_name: str,
    data_dir: str = "data/processed",
    results_dir: str = "data/results",
) -> dict:
    """
    Evaluate a policy on the T-60 post-break window.

    Parameters
    ----------
    policy : PolicyInterface
        Object with reset() and __call__(obs) → np.ndarray shape (6,) in physical MW.
    method_name : str
        Label for output files and RESULT line.
    data_dir : str
        Path to processed/ directory with energy_prices/, as_prices/, system_conditions/.
    results_dir : str
        Output root. Writes to {results_dir}/eval_{method_name}/.

    Returns
    -------
    dict with keys: method, eval_date, all_days, ex_fern, fern_only, fleet,
                    energy_rev_total, as_rev_total
    """
    print(f"[prepare_postbreak] Loading data from {data_dir}")
    merged = load_data(data_dir)
    start_idx, end_idx, ts_ct = _find_t60_indices(merged)

    price_data = merged[PRICE_COLS].values.astype(np.float32)
    system_data = merged[SYSTEM_COLS].values.astype(np.float32)
    timestamps = merged.index

    n_steps = end_idx - start_idx
    print(
        f"[prepare_postbreak] Evaluating '{method_name}' over {n_steps} steps "
        f"({n_steps // STEPS_PER_DAY} days)"
    )

    policy.reset()
    soc = SOC_INIT

    records = []
    for step in range(n_steps):
        idx = start_idx + step
        ct_ts = ts_ct[idx]
        ct_date = ct_ts.date()

        obs = _build_obs(price_data, system_data, ts_ct, idx, soc)
        action_mw = np.asarray(policy(obs), dtype=np.float32)
        proj_mw = project_action(action_mw, soc)

        p_energy = float(proj_mw[0])
        c_as = proj_mw[1:].astype(np.float64)  # [regup, regdn, rrs, ecrs, nsrs]

        rt_lmp = float(price_data[idx, 0])
        rt_mcpc = price_data[idx, 1:6].astype(np.float64)  # [regup, regdn, rrs, ecrs, nsrs]

        energy_rev = p_energy * rt_lmp * DT
        as_rev = c_as * rt_mcpc * DT  # (5,) element-wise, always ≥ 0

        # SoC update from energy dispatch (AS is availability reservation, not deployed)
        if p_energy >= 0:
            soc -= p_energy / ETA * DT
        else:
            soc += abs(p_energy) * ETA * DT
        soc = float(np.clip(soc, SOC_MIN, SOC_MAX))

        records.append(
            {
                "timestamp": timestamps[idx],
                "ct_date": ct_date,
                "p_energy_mw": p_energy,
                "c_regup_mw": float(c_as[0]),
                "c_regdn_mw": float(c_as[1]),
                "c_rrs_mw": float(c_as[2]),
                "c_ecrs_mw": float(c_as[3]),
                "c_nsrs_mw": float(c_as[4]),
                "rt_lmp": rt_lmp,
                "rt_mcpc_regup": float(rt_mcpc[0]),
                "rt_mcpc_regdn": float(rt_mcpc[1]),
                "rt_mcpc_rrs": float(rt_mcpc[2]),
                "rt_mcpc_ecrs": float(rt_mcpc[3]),
                "rt_mcpc_nsrs": float(rt_mcpc[4]),
                "energy_rev": energy_rev,
                "as_rev_regup": float(as_rev[0]),
                "as_rev_regdn": float(as_rev[1]),
                "as_rev_rrs": float(as_rev[2]),
                "as_rev_ecrs": float(as_rev[3]),
                "as_rev_nsrs": float(as_rev[4]),
                "step_rev": energy_rev + float(np.sum(as_rev)),
                "soc_mwh": soc,
            }
        )

    df = pd.DataFrame(records)
    df["cumulative_rev"] = df["step_rev"].cumsum()

    # Per-day revenues
    day_revs = df.groupby("ct_date")["step_rev"].sum()
    fern_rev = float(df[df["ct_date"] == FERN_DATE]["step_rev"].sum())

    def _metrics(revs: pd.Series) -> dict:
        total = float(revs.sum())
        n = int(len(revs))
        daily_avg = total / n if n > 0 else 0.0
        ann = (daily_avg * 365) / (P_MAX * 1000)
        return {
            "n_days": n,
            "total_revenue_usd": round(total, 2),
            "daily_avg_revenue": round(daily_avg, 2),
            "annualized_kw_yr": round(ann, 4),
        }

    all_m = _metrics(day_revs)
    exfern_m = _metrics(day_revs[day_revs.index != FERN_DATE])
    fern_ann = round((fern_rev * 365) / (P_MAX * 1000), 4)
    fern_m = {
        "n_days": 1,
        "total_revenue_usd": round(fern_rev, 2),
        "annualized_kw_yr": fern_ann,
    }

    total_ann = all_m["annualized_kw_yr"]
    fleet = {
        "median_kw_yr": FLEET_MEDIAN_KW_YR,
        "top_q_kw_yr": FLEET_TOP_Q_KW_YR,
        "vs_median_pct": round((total_ann / FLEET_MEDIAN_KW_YR - 1) * 100, 1),
        "vs_top_q_pct": round((total_ann / FLEET_TOP_Q_KW_YR - 1) * 100, 1),
    }

    as_cols = ["as_rev_regup", "as_rev_regdn", "as_rev_rrs", "as_rev_ecrs", "as_rev_nsrs"]
    total_energy_rev = round(float(df["energy_rev"].sum()), 2)
    total_as_rev = round(float(df[as_cols].sum().sum()), 2)

    summary = {
        "method": method_name,
        "eval_date": str(date.today()),
        "all_days": all_m,
        "ex_fern": exfern_m,
        "fern_only": fern_m,
        "fleet": fleet,
        "energy_rev_total": total_energy_rev,
        "as_rev_total": total_as_rev,
    }

    out_dir = os.path.join(results_dir, f"eval_{method_name}")
    os.makedirs(out_dir, exist_ok=True)

    df.drop(columns=["ct_date"]).to_parquet(
        os.path.join(out_dir, "trajectory.parquet"), index=False
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_comparison_card(summary, out_dir)
    _print_summary(summary)

    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_comparison_card(summary: dict, out_dir: str) -> None:
    a = summary["all_days"]
    e = summary["ex_fern"]
    f = summary["fern_only"]
    fl = summary["fleet"]
    total_rev = summary["energy_rev_total"] + summary["as_rev_total"]
    as_share = (
        f"{summary['as_rev_total'] / total_rev * 100:.1f}%"
        if total_rev != 0 else "N/A"
    )

    lines = [
        f"# Eval Results: {summary['method']}",
        f"**Date:** {summary['eval_date']}",
        "**Window:** 2026-01-01 → 2026-02-23 (54 days, T-60 post-break)",
        "",
        "## Revenue Summary",
        "",
        "| Segment | Days | Total USD | $/day avg | $/kW-yr |",
        "|---------|------|-----------|-----------|---------|",
        f"| All 54 days | {a['n_days']} | ${a['total_revenue_usd']:>12,.2f} | ${a['daily_avg_revenue']:>8,.2f} | {a['annualized_kw_yr']:.2f} |",
        f"| Ex-Fern (53 days) | {e['n_days']} | ${e['total_revenue_usd']:>12,.2f} | ${e['daily_avg_revenue']:>8,.2f} | {e['annualized_kw_yr']:.2f} |",
        f"| Fern only (Jan 26) | {f['n_days']} | ${f['total_revenue_usd']:>12,.2f} | {'—':>9} | {f['annualized_kw_yr']:.2f} |",
        "",
        "## Revenue Composition",
        "",
        f"- Energy revenue: ${summary['energy_rev_total']:>10,.2f}",
        f"- AS revenue:     ${summary['as_rev_total']:>10,.2f}",
        f"- AS share:       {as_share}",
        "",
        "## Fleet Comparison (T-60 Window, commit bd07a9c)",
        "",
        "| Benchmark | $/kW-yr | This method | Delta |",
        "|-----------|---------|-------------|-------|",
        f"| Fleet median | {fl['median_kw_yr']:.2f} | {a['annualized_kw_yr']:.2f} | {fl['vs_median_pct']:+.1f}% |",
        f"| Fleet top-quartile | {fl['top_q_kw_yr']:.2f} | {a['annualized_kw_yr']:.2f} | {fl['vs_top_q_pct']:+.1f}% |",
    ]

    with open(os.path.join(out_dir, "comparison_card.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _print_summary(summary: dict) -> None:
    a = summary["all_days"]
    fl = summary["fleet"]
    total_rev = summary["energy_rev_total"] + summary["as_rev_total"]
    as_share = (
        f"{summary['as_rev_total'] / total_rev * 100:.1f}%"
        if total_rev != 0 else "N/A"
    )

    print(f"\n[prepare_postbreak] === {summary['method']} ===")
    print(f"  All 54 days:  ${a['total_revenue_usd']:>12,.2f}  ({a['annualized_kw_yr']:.2f} $/kW-yr)")
    print(f"  Ex-Fern:      ${summary['ex_fern']['total_revenue_usd']:>12,.2f}  ({summary['ex_fern']['annualized_kw_yr']:.2f} $/kW-yr)")
    print(f"  Fern only:    ${summary['fern_only']['total_revenue_usd']:>12,.2f}")
    print(f"  AS share:     {as_share}")
    print(f"  vs fleet median:  {fl['vs_median_pct']:+.1f}%  (target: {fl['median_kw_yr']:.2f} $/kW-yr)")
    print(f"  vs fleet top-Q:   {fl['vs_top_q_pct']:+.1f}%  (target: {fl['top_q_kw_yr']:.2f} $/kW-yr)")
    print(
        f"RESULT method={summary['method']} "
        f"all_days_usd={a['total_revenue_usd']:.2f} "
        f"kw_yr={a['annualized_kw_yr']:.4f} "
        f"vs_median_pct={fl['vs_median_pct']:.1f} "
        f"vs_top_q_pct={fl['vs_top_q_pct']:.1f}"
    )


# ---------------------------------------------------------------------------
# Test policies (Phase 2 validation)
# ---------------------------------------------------------------------------

class ZeroPolicy:
    """Always returns zero action — idle, no AS bids. Revenue ≈ 0."""

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)


class RandomPolicy:
    """Uniform random actions: energy ∈ [−P_max, +P_max], AS ∈ [0, P_max] each."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        p_energy = float(self._rng.uniform(-P_MAX, P_MAX))
        c_as = self._rng.uniform(0.0, P_MAX, size=5).tolist()
        return np.array([p_energy] + c_as, dtype=np.float32)


class MILPReplayPolicy:
    """
    Replays MILP expert actions for the T-60 eval window.

    Action source (NPZ actions in p.u., converted to MW via × P_max):
        Train NPZ [7776:19584]  Jan 1 – Feb 10  (41 days, 11 808 transitions)
        Val NPZ   [0:3744]      Feb 11 – Feb 23 (13 days,  3 744 transitions)

    Validation target: total revenue within ±2% of $90,814 (±$1,816).
    """

    TRAIN_SLICE = slice(7776, 19584)
    VAL_SLICE = slice(0, 3744)

    def __init__(self, npz_train_path: str, npz_val_path: str):
        train = np.load(npz_train_path)
        val = np.load(npz_val_path)

        train_slice = train["actions"][self.TRAIN_SLICE]  # (11808, 6) p.u.
        val_slice = val["actions"][self.VAL_SLICE]        # (3744, 6)  p.u.

        # NPZ actions: p_energy ∈ [−1, 1], c_as ∈ [0, 1] — multiply by P_max to get MW
        actions_pu = np.concatenate([train_slice, val_slice], axis=0)  # (15552, 6)
        self.actions_mw = (actions_pu * P_MAX).astype(np.float32)

        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def __call__(self, obs: dict) -> np.ndarray:
        if self._step >= len(self.actions_mw):
            return np.zeros(6, dtype=np.float32)
        action = self.actions_mw[self._step].copy()
        self._step += 1
        return action


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a test policy on the T-60 post-break window."
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument(
        "--method",
        choices=["zero", "random", "milp_replay", "all"],
        default="all",
    )
    parser.add_argument(
        "--npz-train",
        default="data/expert_trajectories/receding_horizon_postbreak_train.npz",
    )
    parser.add_argument(
        "--npz-val",
        default="data/expert_trajectories/receding_horizon_postbreak_val.npz",
    )
    args = parser.parse_args()

    methods = (
        ["zero", "random", "milp_replay"] if args.method == "all" else [args.method]
    )

    for method in methods:
        if method == "zero":
            policy = ZeroPolicy()
        elif method == "random":
            policy = RandomPolicy(seed=42)
        else:
            policy = MILPReplayPolicy(args.npz_train, args.npz_val)

        evaluate(
            policy,
            method_name=method,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
        )
