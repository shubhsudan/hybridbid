"""
Eval utilities shared across all methods.

Provides:
  - enrich_summary(): add vs_pf_oracle + vs_milp_replay_ceiling to summary.json
  - run_fern_slice(): 7-day Fern-inclusive in-loop probe (Jan 23-29, SoC reset to 50%)

These are the only eval-path utilities in the methods/ tree. They do not modify
experiments/prepare_postbreak.py (frozen harness).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.prepare_postbreak import (
    load_data,
    _find_t60_indices,
    _build_obs,
    project_action,
    PRICE_COLS,
    SYSTEM_COLS,
    P_MAX,
    E_MAX,
    ETA,
    SOC_MIN,
    SOC_MAX,
    SOC_INIT,
    DT,
    STEPS_PER_DAY,
    FERN_DATE,
)

# ── Reference ceilings (locked; do NOT recompute mid-sprint) ──────────────────
PF_ORACLE_KW_YR      = 63.1556   # Phase 1 result, commit 1eb529e
MILP_REPLAY_KW_YR    = 58.40     # commit 92c5a49 / eval_milp_replay_ct
FLEET_MEDIAN_KW_YR   = 24.93     # commit bd07a9c

FERN_SLICE_START = date(2026, 1, 23)
FERN_SLICE_END   = date(2026, 1, 29)   # inclusive


# ── Summary enrichment ────────────────────────────────────────────────────────

def add_ceiling_metrics(summary: dict) -> dict:
    """
    Add vs_pf_oracle and vs_milp_replay_ceiling to an existing summary dict.
    Mutates and returns the dict. Does NOT write to disk.
    """
    kw_yr = summary["all_days"]["annualized_kw_yr"]
    summary["vs_pf_oracle"]           = round((kw_yr / PF_ORACLE_KW_YR    - 1) * 100, 1)
    summary["vs_milp_replay_ceiling"] = round((kw_yr / MILP_REPLAY_KW_YR  - 1) * 100, 1)
    summary["vs_fleet_median"]        = round((kw_yr / FLEET_MEDIAN_KW_YR - 1) * 100, 1)
    return summary


def enrich_summary_file(summary_path: str) -> dict:
    """
    Read an existing summary.json, add ceiling metrics, re-write it.
    Returns the enriched dict.
    """
    with open(summary_path) as f:
        summary = json.load(f)
    add_ceiling_metrics(summary)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ── Fern slice evaluator ──────────────────────────────────────────────────────

def _find_fern_slice_indices(ts_ct) -> tuple[int, int]:
    """Return (slice_start, slice_end_exclusive) for Jan 23-29 CT in the full ts_ct array."""
    ct_dates = np.array([t.date() for t in ts_ct])
    start_mask = ct_dates == FERN_SLICE_START
    end_mask   = ct_dates == FERN_SLICE_END
    if not start_mask.any():
        raise ValueError(f"Fern slice start {FERN_SLICE_START} not found in data")
    if not end_mask.any():
        raise ValueError(f"Fern slice end {FERN_SLICE_END} not found in data")
    return int(np.where(start_mask)[0][0]), int(np.where(end_mask)[0][-1]) + 1


def run_fern_slice(
    policy,
    price_data:  np.ndarray,   # (N_full, 12) float32 — full merged price array
    system_data: np.ndarray,   # (N_full, 7) float32
    ts_ct,                     # CT timestamps for full merged array
    fern_start_idx: int,
    fern_end_idx:   int,
    soc_init: float = SOC_INIT,
) -> dict:
    """
    Run a 7-day Fern-inclusive in-loop probe (Jan 23-29).

    SoC is reset to soc_init (default 50% = 10 MWh) at slice start.
    This is intentional: probe is a failure-mode detector, not production accounting.

    Returns dict with revenue breakdown (physical $).
    """
    policy.reset()
    soc = soc_init
    n_steps = fern_end_idx - fern_start_idx

    total_energy_rev = 0.0
    total_as_rev     = 0.0
    fern_rev         = 0.0

    for step in range(n_steps):
        idx = fern_start_idx + step
        ct_date = ts_ct[idx].date()

        obs        = _build_obs(price_data, system_data, ts_ct, idx, soc)
        action_mw  = np.asarray(policy(obs), dtype=np.float32)
        proj_mw    = project_action(action_mw, soc)

        p_energy = float(proj_mw[0])
        c_as     = proj_mw[1:].astype(np.float64)

        rt_lmp  = float(price_data[idx, 0])
        rt_mcpc = price_data[idx, 1:6].astype(np.float64)

        e_rev  = p_energy * rt_lmp * DT
        as_rev_vec = c_as * rt_mcpc * DT
        step_rev = e_rev + float(np.sum(as_rev_vec))

        total_energy_rev += e_rev
        total_as_rev     += float(np.sum(as_rev_vec))
        if ct_date == FERN_DATE:
            fern_rev += step_rev

        if p_energy >= 0:
            soc -= p_energy / ETA * DT
        else:
            soc += abs(p_energy) * ETA * DT
        soc = float(np.clip(soc, SOC_MIN, SOC_MAX))

    total_rev  = total_energy_rev + total_as_rev
    daily_avg  = total_rev / 7.0
    kw_yr      = (daily_avg * 365) / (P_MAX * 1000)
    total_rev_rnd = round(total_rev, 2)

    return {
        "n_steps":          n_steps,
        "total_revenue_usd": total_rev_rnd,
        "daily_avg_revenue": round(daily_avg, 2),
        "annualized_kw_yr":  round(kw_yr, 4),
        "energy_rev_usd":    round(total_energy_rev, 2),
        "as_rev_usd":        round(total_as_rev, 2),
        "fern_day_rev_usd":  round(fern_rev, 2),
        "soc_final_mwh":     round(soc, 3),
    }


def prepare_fern_slice_data(data_dir: str = "data/processed"):
    """
    Load merged data once, extract Fern slice indices and arrays.
    Returns (price_data, system_data, ts_ct, fern_start_idx, fern_end_idx).
    Cache-friendly: call once at startup.
    """
    merged = load_data(data_dir)
    _, _, ts_ct = _find_t60_indices(merged)   # validates T-60 window
    price_data  = merged[PRICE_COLS].values.astype(np.float32)
    system_data = merged[SYSTEM_COLS].values.astype(np.float32)
    fern_start, fern_end = _find_fern_slice_indices(ts_ct)
    return price_data, system_data, ts_ct, fern_start, fern_end
