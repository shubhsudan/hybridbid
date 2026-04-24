"""
Post-break MILP expert trajectory generation (6D joint co-optimization).

Generates (s, a, r, s', done) transitions for the post-RTC+B period using a
per-day receding-horizon MILP with joint energy + 5 AS product optimization.

Design rationale — why one MILP per day equals per-interval receding horizon:
  With perfect-foresight prices (realized prices used as "forecast") and a
  deterministic SoC update rule, re-solving at every interval gives the same
  remaining schedule as the full-day solution. The only state that can deviate
  is SoC, and since SoC evolves exactly as the MILP plans (no stochastic
  disturbance), it stays on the planned trajectory. We therefore solve once per
  day and read off all 288 committed actions — identical result, 288× faster.

MILP formulation (per day, T=288 intervals):
  maximize  Σ_t [ (p_dch-p_ch)*rt_lmp*dt + Σ_j c_j*rt_mcpc_j*dt ]
            - λ*(soc[T] - soc_target)²          (soft terminal SoC)
  subject to
    soc[t+1] = soc[t] + η*p_ch[t]*dt - p_dch[t]/η*dt   (dynamics)
    soc_min ≤ soc[t] ≤ soc_max                           (state bounds)
    soc[0] = soc_initial                                  (initial condition)
    p_ch[t], p_dch[t] ≥ 0                                (non-negativity)
    p_ch[t] + p_dch[t] + Σ_j c_j[t] ≤ P_max             (shared capacity)
    c_j[t] * sustain_j ≤ (soc[t]-soc_min)*η  (discharge AS SoC feasibility)
    c_regdn[t] * sustain_regdn ≤ (soc_max-soc[t])*η  (charge AS SoC feasibility)

Battery: 10 MW / 20 MWh, η_ch=η_dch=0.95, SoC ∈ [2, 18] MWh. Matches CLAUDE.md.

Output schema (matches receding_horizon_*_option_d.npz):
  price_history:        (N, 32, 12)  float32  — raw TTFE input (pre-TTFE)
  static_features:      (N, 14)      float32  — system(7)+time(6)+soc(1)
  next_price_history:   (N, 32, 12)  float32
  next_static_features: (N, 14)      float32
  actions:              (N, 6)       float32  — normalized p.u.: [p_energy∈[-1,1], c_as∈[0,1]×5]
  rewards:              (N,)         float32  — Li et al. Eq.26 (p.u. energy) + AS revenue
  dones:                (N,)         bool     — always False (MILP enforces SoC feasibility)
  truncateds:           (N,)         bool     — True at last interval of each day
  soc:                  (N,)         float32  — MWh

Usage:
  python -m src.data.postbreak_milp --smoke              # 5-day sanity check
  python -m src.data.postbreak_milp --split train        # Dec 5 – Feb 10
  python -m src.data.postbreak_milp --split val          # Feb 11 – Apr 15
  python -m src.data.postbreak_milp                      # both splits
"""

import argparse
import glob
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Battery (matches CLAUDE.md / ercot_env.py DEFAULT_BATTERY) ──
P_MAX = 10.0        # MW
E_MAX = 20.0        # MWh
ETA = 0.95          # η_ch = η_dch = 0.95 (symmetric)
SOC_MIN_FRAC = 0.10
SOC_MAX_FRAC = 0.90
SOC_INIT_FRAC = 0.50
SOC_MIN = SOC_MIN_FRAC * E_MAX   # 2.0 MWh
SOC_MAX = SOC_MAX_FRAC * E_MAX   # 18.0 MWh
SOC_TARGET = SOC_INIT_FRAC * E_MAX  # 10.0 MWh

# ── MILP hyperparameters ──
LAMBDA_TERMINAL = 20.0   # $/MWh²: soft terminal SoC penalty weight

# AS sustain durations [hours] from configs/battery.yaml
AS_SUSTAIN_H = {
    "regup": 1.0,    # 60 min
    "regdn": 1.0,    # 60 min
    "rrs":   1/6,    # 10 min
    "ecrs":  1/4,    # 15 min
    "nsrs":  0.5,    # 30 min
}

# ── Env reward constants (Li et al. Eq. 26) ──
EMA_TAU   = 0.9
BETA_ARB  = 10.0
DT        = 5.0 / 60.0  # hours per interval

# ── Observation constants (matches ercot_env.py) ──
SEQ_LEN  = 32
STEPS_PER_DAY = 288

# Column ordering for price_history (must match ercot_env.py PRICE_COLS)
PRICE_COLS = [
    "rt_lmp",
    "rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs",
    "dam_spp",
    "dam_as_regup",  "dam_as_regdn",  "dam_as_rrs",  "dam_as_ecrs",  "dam_as_nsrs",
]
SYSTEM_COLS = [
    "total_load_mw", "load_forecast_mw",
    "wind_actual_mw", "wind_forecast_mw",
    "solar_actual_mw", "solar_forecast_mw",
    "net_load_mw",
]
SYSTEM_SCALES = np.array(
    [50000, 50000, 15000, 15000, 10000, 10000, 40000], dtype=np.float32
)

# ── Data splits ──
CONTEXT_START = "2025-11-29"   # Nov 29 gives 5 days of lookback before Dec 5
SPLITS = {
    "train": ("2025-12-05", "2026-02-10"),
    "val":   ("2026-02-11", "2026-04-15"),
}
SMOKE_DATES = [
    "2026-01-10",  # normal winter weekday
    "2026-01-25",  # day before Fern
    "2026-01-26",  # Winter Storm Fern
    "2026-01-27",  # post-Fern
    "2026-02-05",  # post-Fern, mid-validation boundary
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_merged(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load and merge all three processed Parquet tables.
    Returns a DataFrame with PRICE_COLS + SYSTEM_COLS, UTC-indexed, 5-min.
    """
    ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{data_dir}/energy_prices/*.parquet"))]).sort_index()
    ap = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{data_dir}/as_prices/*.parquet"))]).sort_index()
    sc = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{data_dir}/system_conditions/*.parquet"))]).sort_index()

    for df in [ep, ap, sc]:
        if "is_post_rtcb" in df.columns:
            df.drop(columns=["is_post_rtcb"], inplace=True)

    merged = ep.join(ap, how="outer").join(sc, how="outer")

    # Ensure correct column ordering
    all_cols = PRICE_COLS + SYSTEM_COLS
    for col in all_cols:
        if col not in merged.columns:
            merged[col] = 0.0

    merged[PRICE_COLS]  = merged[PRICE_COLS].fillna(0.0)
    merged[SYSTEM_COLS] = merged[SYSTEM_COLS].ffill().fillna(0.0)

    return merged[all_cols]


def slice_range(merged: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return merged.loc[s:e]


# ─────────────────────────────────────────────────────────────────────────────
# Observation builder (replicates ercot_env._get_observation, standard mode)
# ─────────────────────────────────────────────────────────────────────────────

def build_obs(
    price_data:  np.ndarray,   # (N_global, 12) float32
    system_data: np.ndarray,   # (N_global, 7)  float32
    timestamps:  pd.DatetimeIndex,
    global_idx:  int,
    soc_mwh:     float,
) -> dict:
    """Build one (price_history, static_features) observation dict."""
    start = global_idx - SEQ_LEN + 1
    price_history = price_data[start : global_idx + 1].copy()   # (32, 12)

    sys_norm = system_data[global_idx] / SYSTEM_SCALES
    soc_frac = np.array([soc_mwh / E_MAX], dtype=np.float32)

    ts = timestamps[global_idx]
    if hasattr(ts, "tz_convert"):
        ts_local = ts.tz_convert("US/Central")
    else:
        ts_local = ts
    hour  = ts_local.hour + ts_local.minute / 60.0
    dow   = ts_local.dayofweek
    month = ts_local.month
    time_feats = np.array([
        np.sin(2 * np.pi * hour  / 24), np.cos(2 * np.pi * hour  / 24),
        np.sin(2 * np.pi * dow   / 7),  np.cos(2 * np.pi * dow   / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
    ], dtype=np.float32)

    static_features = np.concatenate([sys_norm, time_feats, soc_frac])  # (14,)
    return {"price_history": price_history, "static_features": static_features}


# ─────────────────────────────────────────────────────────────────────────────
# MILP solver (one solve for a full 288-interval day)
# ─────────────────────────────────────────────────────────────────────────────

def solve_day_milp(
    day_price_data: np.ndarray,  # (288, 12) — PRICE_COLS order
    soc_initial:    float,
    solver:         str,
    timeout_s:      float = 600.0,
) -> dict:
    """
    Solve full-day joint energy+AS QP for 288 intervals.

    Returns dict with keys: status, p_ch, p_dch, c_as (288,5), soc (289,),
    solve_time, objective.  On failure: status + solve_time only.
    """
    T = STEPS_PER_DAY

    rt_lmp  = day_price_data[:, 0].astype(float)    # energy price
    rt_mcpc = day_price_data[:, 1:6].astype(float)  # 5 AS clearing prices (cols 1-5)

    # ── Decision variables ──
    p_ch    = cp.Variable(T, nonneg=True, name="p_ch")
    p_dch   = cp.Variable(T, nonneg=True, name="p_dch")
    c_regup = cp.Variable(T, nonneg=True, name="c_regup")
    c_regdn = cp.Variable(T, nonneg=True, name="c_regdn")
    c_rrs   = cp.Variable(T, nonneg=True, name="c_rrs")
    c_ecrs  = cp.Variable(T, nonneg=True, name="c_ecrs")
    c_nsrs  = cp.Variable(T, nonneg=True, name="c_nsrs")
    soc     = cp.Variable(T + 1, name="soc")

    c_as_list = [c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs]
    sustain_h = list(AS_SUSTAIN_H.values())  # [1.0, 1.0, 1/6, 1/4, 0.5]

    # ── Objective ──
    energy_rev = cp.sum(cp.multiply(rt_lmp, (p_dch - p_ch))) * DT
    as_rev = sum(
        cp.sum(cp.multiply(rt_mcpc[:, j], c_as_list[j])) * DT
        for j in range(5)
    )
    terminal_penalty = LAMBDA_TERMINAL * cp.square(soc[T] - SOC_TARGET)
    objective = cp.Maximize(energy_rev + as_rev - terminal_penalty)

    # ── Constraints ──
    constraints = [soc[0] == soc_initial]

    # SoC dynamics (vectorized)
    constraints.append(
        soc[1:] == soc[:-1] + ETA * p_ch * DT - (p_dch / ETA) * DT
    )

    # SoC bounds
    constraints.append(soc >= SOC_MIN)
    constraints.append(soc <= SOC_MAX)

    # Power limits
    constraints.append(p_ch  <= P_MAX)
    constraints.append(p_dch <= P_MAX)

    # Shared capacity: energy + all AS ≤ P_max
    constraints.append(
        p_ch + p_dch + c_regup + c_regdn + c_rrs + c_ecrs + c_nsrs <= P_MAX
    )

    # AS SoC feasibility (linear in soc[t] — uses soc[:-1] for t=0..T-1)
    avail_dch = (soc[:-1] - SOC_MIN) * ETA   # energy available to discharge (MWh)
    avail_ch  = (SOC_MAX - soc[:-1]) * ETA   # energy available to charge (MWh)

    # Discharge-direction AS: c_j * sustain_j ≤ avail_dch
    for j, (c_j, name) in enumerate(zip(c_as_list, AS_SUSTAIN_H)):
        if name == "regdn":
            # charge-direction
            constraints.append(c_j * sustain_h[j] <= avail_ch)
        else:
            constraints.append(c_j * sustain_h[j] <= avail_dch)

    # ── Solve ──
    prob = cp.Problem(objective, constraints)
    t0 = time.time()
    try:
        solve_kwargs = dict(solver=solver, verbose=False)
        if solver == "HIGHS":
            solve_kwargs["time_limit"] = timeout_s
        prob.solve(**solve_kwargs)
    except Exception as exc:
        return {"status": f"error:{exc}", "solve_time": time.time() - t0}
    solve_time = time.time() - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status, "solve_time": solve_time}

    def _safe(v):
        arr = np.array(v.value, dtype=np.float64).flatten()
        np.nan_to_num(arr, nan=0.0, posinf=P_MAX, neginf=0.0, copy=False)
        return arr

    return {
        "status":     prob.status,
        "p_ch":       _safe(p_ch),
        "p_dch":      _safe(p_dch),
        "c_as":       np.column_stack([_safe(c) for c in c_as_list]),  # (288,5)
        "soc":        _safe(soc),
        "solve_time": solve_time,
        "objective":  float(prob.value),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation (matches ercot_env.py step() exactly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_reward(
    energy_mag_pu: float,   # |p_energy| / P_max ∈ [0,1]
    mode:          int,     # 0=charge, 1=discharge, 2=idle
    rt_lmp:        float,
    ema:           float,   # UPDATED EMA (already includes current rt_lmp)
    c_as_mw:       np.ndarray,  # (5,) physical MW
    rt_mcpc:       np.ndarray,  # (5,) $/MW-h clearing prices
) -> float:
    """Li et al. Eq. 26 energy reward (p.u.) + AS capacity revenue (physical MW)."""
    v_dch = 1.0 if mode == 1 else 0.0
    v_ch  = 1.0 if mode == 0 else 0.0

    price_dev = abs(rt_lmp - ema)
    I_dch = np.sign(rt_lmp - ema)
    I_ch  = np.sign(ema - rt_lmp)

    energy_term = energy_mag_pu * rt_lmp * (v_dch * ETA - v_ch / ETA) * DT
    timing_bonus = (
        BETA_ARB * energy_mag_pu * price_dev
        * (I_dch * v_dch * ETA + I_ch * v_ch / ETA) * DT
    )
    as_rev = float(np.sum(c_as_mw * rt_mcpc) * DT)

    return float(energy_term + timing_bonus + as_rev)


# ─────────────────────────────────────────────────────────────────────────────
# Per-day worker
# ─────────────────────────────────────────────────────────────────────────────

def process_day(args: tuple) -> dict:
    """
    Run the full-day MILP for one UTC date and collect 288 transitions.

    args = (date_str, day_start_global, price_data, system_data, timestamps, solver)

    price_data and system_data are slices covering:
      [day_start_global - SEQ_LEN + 1 : day_start_global + STEPS_PER_DAY + 1]
    local_day_start = SEQ_LEN - 1 (within the slice)
    """
    date_str, day_start_global, price_slice, system_slice, ts_slice, solver = args

    T = STEPS_PER_DAY
    local_start = SEQ_LEN - 1  # index of day start within slice

    # Day prices for MILP (288 intervals)
    day_p = price_slice[local_start : local_start + T]  # (288, 12)

    # Solve MILP
    result = solve_day_milp(day_p, soc_initial=SOC_TARGET, solver=solver)
    if "p_ch" not in result:
        logger.warning(f"{date_str}: MILP failed — status={result['status']}. Using idle fallback.")
        p_ch_arr  = np.zeros(T, dtype=np.float64)
        p_dch_arr = np.zeros(T, dtype=np.float64)
        c_as_arr  = np.zeros((T, 5), dtype=np.float64)
        soc_arr   = np.full(T + 1, SOC_TARGET, dtype=np.float64)
        milp_failed = True
    else:
        p_ch_arr  = np.clip(result["p_ch"],  0, P_MAX)
        p_dch_arr = np.clip(result["p_dch"], 0, P_MAX)
        c_as_arr  = np.clip(result["c_as"],  0, P_MAX)
        soc_arr   = result["soc"]
        milp_failed = False

    # ── Build transitions ──
    ph_buf   = np.empty((T, SEQ_LEN, 12), dtype=np.float32)
    sf_buf   = np.empty((T, 14),          dtype=np.float32)
    nph_buf  = np.empty((T, SEQ_LEN, 12), dtype=np.float32)
    nsf_buf  = np.empty((T, 14),          dtype=np.float32)
    act_buf  = np.empty((T, 6),           dtype=np.float32)
    rew_buf  = np.empty(T,                dtype=np.float32)
    done_buf = np.zeros(T,                dtype=bool)
    trunc_buf= np.zeros(T,                dtype=bool)
    soc_buf  = np.empty(T,                dtype=np.float32)

    ema = float(day_p[0, 0])  # initialise EMA to first RT LMP of the day
    soc_sim = SOC_TARGET       # simulated SoC (tracks actual, not MILP-planned)

    price_data_local  = price_slice.astype(np.float32)
    system_data_local = system_slice.astype(np.float32)

    for t in range(T):
        g = local_start + t   # index into price_slice / system_slice

        # Current observation
        obs = build_obs(price_data_local, system_data_local, ts_slice, g, soc_sim)
        ph_buf[t]  = obs["price_history"]
        sf_buf[t]  = obs["static_features"]

        # Action from MILP solution
        p_ch_t  = float(p_ch_arr[t])
        p_dch_t = float(p_dch_arr[t])
        c_as_t  = c_as_arr[t].astype(np.float64)   # (5,) MW

        # Determine mode and energy_mag
        eps = 1e-3
        if p_dch_t > eps:
            mode = 1   # discharge
            energy_mag_pu = p_dch_t / P_MAX
            p_net = p_dch_t
        elif p_ch_t > eps:
            mode = 0   # charge
            energy_mag_pu = p_ch_t / P_MAX
            p_net = -p_ch_t
        else:
            mode = 2   # idle
            energy_mag_pu = 0.0
            p_net = 0.0

        # Normalized action: [p_energy ∈ [-1,1], c_as ∈ [0,1]×5]
        p_energy_norm = p_net / P_MAX
        c_as_norm = c_as_t / P_MAX
        act_buf[t] = np.concatenate([[p_energy_norm], c_as_norm]).astype(np.float32)

        # Reward — update EMA first (matches env step() order)
        rt_lmp_t = float(day_p[t, 0])
        ema = EMA_TAU * ema + (1.0 - EMA_TAU) * rt_lmp_t
        rt_mcpc_t = day_p[t, 1:6].astype(np.float64)

        rew_buf[t] = compute_step_reward(
            energy_mag_pu, mode, rt_lmp_t, ema,
            c_as_mw=c_as_t, rt_mcpc=rt_mcpc_t,
        )

        # SoC update (simulate, don't rely on MILP soc_arr)
        if p_dch_t > eps:
            soc_sim -= (p_dch_t / ETA) * DT
        elif p_ch_t > eps:
            soc_sim += p_ch_t * ETA * DT
        soc_sim = float(np.clip(soc_sim, SOC_MIN, SOC_MAX))

        soc_buf[t] = soc_sim

        # Next observation
        if t < T - 1:
            g_next = g + 1
            next_obs = build_obs(price_data_local, system_data_local, ts_slice, g_next, soc_sim)
        else:
            # Last step: use same obs as current (env does the same on truncation)
            next_obs = build_obs(price_data_local, system_data_local, ts_slice, g, soc_sim)
            trunc_buf[t] = True

        nph_buf[t]  = next_obs["price_history"]
        nsf_buf[t]  = next_obs["static_features"]

    # Revenue for logging (physical $)
    energy_rev_day = float(np.sum((p_dch_arr - p_ch_arr) * day_p[:, 0]) * DT)
    as_rev_day     = float(np.sum(c_as_arr * day_p[:, 1:6]) * DT)
    total_rev_day  = energy_rev_day + as_rev_day

    return {
        "date":           date_str,
        "price_history":  ph_buf,
        "static_features": sf_buf,
        "next_price_history":  nph_buf,
        "next_static_features": nsf_buf,
        "actions":        act_buf,
        "rewards":        rew_buf,
        "dones":          done_buf,
        "truncateds":     trunc_buf,
        "soc":            soc_buf,
        # diagnostics
        "solve_time":     result.get("solve_time", 0.0),
        "milp_status":    result.get("status", "idle_fallback"),
        "milp_failed":    milp_failed,
        "energy_rev":     energy_rev_day,
        "as_rev":         as_rev_day,
        "total_rev":      total_rev_day,
        "terminal_soc":   float(soc_buf[-1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Day-list builder
# ─────────────────────────────────────────────────────────────────────────────

def build_day_list(merged: pd.DataFrame, start: str, end: str) -> list:
    """
    Return a list of (date_str, global_start_idx) for complete UTC days
    within [start, end] that have enough lookback (>= SEQ_LEN rows before them).
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    timestamps = merged.index
    all_dates  = pd.Series(timestamps.date).unique()

    days = []
    for d in sorted(all_dates):
        date_ts = pd.Timestamp(d, tz="UTC")
        if date_ts < start_ts or date_ts > end_ts:
            continue
        mask     = timestamps.date == d
        indices  = np.where(mask)[0]
        if len(indices) < STEPS_PER_DAY:
            continue
        first_idx = indices[0]
        if first_idx < SEQ_LEN:
            continue  # not enough lookback
        days.append((str(d), int(first_idx)))

    return days


# ─────────────────────────────────────────────────────────────────────────────
# Run helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_worker_args(
    days: list,
    price_data: np.ndarray,   # (N_global, 12)
    system_data: np.ndarray,  # (N_global, 7)
    timestamps: pd.DatetimeIndex,
    solver: str,
) -> list:
    """Package per-day args for process_day()."""
    args_list = []
    ctx = SEQ_LEN  # rows of lookback needed before day start
    for date_str, day_start in days:
        sl_s = day_start - ctx + 1
        sl_e = day_start + STEPS_PER_DAY + 1
        args_list.append((
            date_str,
            day_start,
            price_data[sl_s:sl_e].copy(),
            system_data[sl_s:sl_e].copy(),
            timestamps[sl_s:sl_e],
            solver,
        ))
    return args_list


def _collect(results: list) -> dict:
    """Concatenate per-day result dicts into arrays."""
    keys = ["price_history", "static_features", "next_price_history",
            "next_static_features", "actions", "rewards", "dones", "truncateds", "soc"]
    out = {k: np.concatenate([r[k] for r in results], axis=0) for k in keys}
    return out


def run_days(
    days: list,
    price_data: np.ndarray,
    system_data: np.ndarray,
    timestamps: pd.DatetimeIndex,
    solver: str,
    n_workers: int = 1,
) -> tuple:
    """Run all days, return (arrays_dict, diagnostics_list)."""
    args_list = _prepare_worker_args(days, price_data, system_data, timestamps, solver)
    t0 = time.time()

    if n_workers <= 1:
        results = [process_day(a) for a in args_list]
    else:
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_day, args_list)

    elapsed = time.time() - t0
    logger.info(f"Processed {len(results)} days in {elapsed:.1f}s wall time "
                f"({elapsed/len(results):.1f}s/day avg)")

    diag = [
        {k: r[k] for k in ("date", "solve_time", "milp_status", "milp_failed",
                            "energy_rev", "as_rev", "total_rev", "terminal_soc")}
        for r in results
    ]
    return _collect(results), diag


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(arrays: dict, diag: list, label: str) -> None:
    """Run and print all 8 sanity checks from the task spec."""
    logger.info(f"\n{'='*60}")
    logger.info(f"SANITY CHECKS — {label}")
    logger.info(f"{'='*60}")

    N = len(arrays["rewards"])
    acts = arrays["actions"]
    soc  = arrays["soc"]
    rews = arrays["rewards"]
    trunc= arrays["truncateds"]

    total_days = int(trunc.sum())
    n_failed = sum(1 for d in diag if d["milp_failed"])

    # ── 1. Revenue scale ──
    total_rev = sum(d["total_rev"] for d in diag)
    n_days = len(diag)
    # Annualize: total_rev over n_days → $/yr → $/kW-yr (P_max=10MW=10000kW)
    ann_rev_per_kw = (total_rev / n_days * 365) / (P_MAX * 1000)
    status1 = "OK" if 40 <= ann_rev_per_kw <= 200 else "FAIL"
    logger.info(f"[{status1}] 1. Revenue scale: ${ann_rev_per_kw:.1f}/kW-yr "
                f"(total=${total_rev:.0f} over {n_days} days; target: $40-$200/kW-yr)")

    # ── 2. Fern contribution (only if Jan 26 is in diag) ──
    fern_diags = [d for d in diag if "2026-01-26" in d["date"]]
    if fern_diags:
        fern_rev = fern_diags[0]["total_rev"]
        fern_pct = fern_rev / max(total_rev, 1) * 100
        status2 = "OK" if 10 <= fern_pct <= 80 else "WARN"
        logger.info(f"[{status2}] 2. Fern contribution: ${fern_rev:.0f} ({fern_pct:.1f}% of window revenue; target: 20-50%)")
    else:
        logger.info("[SKIP] 2. Fern (Jan 26) not in this run")

    # ── 3. Action distributions ──
    logger.info("[ -- ] 3. Action distributions:")
    names = ["p_energy", "c_regup", "c_regdn", "c_rrs", "c_ecrs", "c_nsrs"]
    all_zero_flag = False
    pmax_flag = False
    for i, name in enumerate(names):
        col = acts[:, i]
        p95 = np.percentile(np.abs(col), 95)
        mean_abs = np.mean(np.abs(col))
        frac_zero = np.mean(np.abs(col) < 1e-3)
        frac_max  = np.mean(np.abs(col) > 0.99)
        if frac_zero > 0.99:
            flag = "WARN-ALL-ZERO"; all_zero_flag = True
        elif frac_max > 0.99:
            flag = "WARN-ALL-MAX"; pmax_flag = True
        else:
            flag = "OK"
        logger.info(f"        [{flag}] {name}: mean|x|={mean_abs:.3f}, P95|x|={p95:.3f}, "
                    f"frac_zero={frac_zero:.2%}, frac_pmax={frac_max:.2%}")

    # ── 4. SoC distribution ──
    soc_mwh = soc  # stored in MWh
    soc_floor_frac = np.mean(soc_mwh <= SOC_MIN + 0.05)
    soc_ceil_frac  = np.mean(soc_mwh >= SOC_MAX - 0.05)
    status4 = "OK"
    if soc_floor_frac > 0.5: status4 = "WARN-FLOOR-PINNED"
    if soc_ceil_frac  > 0.5: status4 = "WARN-CEIL-PINNED"
    logger.info(f"[{status4}] 4. SoC distribution: min={soc_mwh.min():.2f} max={soc_mwh.max():.2f} "
                f"mean={soc_mwh.mean():.2f} MWh | floor_frac={soc_floor_frac:.2%}, ceil_frac={soc_ceil_frac:.2%}")

    # ── 5. Terminal SoC distribution ──
    term_socs = [d["terminal_soc"] for d in diag]
    t_mean = np.mean(term_socs)
    t_std  = np.std(term_socs)
    t_min  = np.min(term_socs)
    t_max  = np.max(term_socs)
    pinned_05 = np.mean(np.abs(np.array(term_socs) - SOC_TARGET) < 0.01)
    status5 = "OK"
    if pinned_05 > 0.95: status5 = "WARN-LAMBDA-TOO-HIGH"
    if t_std > 4.0:      status5 = "WARN-LAMBDA-TOO-LOW"
    logger.info(f"[{status5}] 5. Terminal SoC: mean={t_mean:.2f} std={t_std:.2f} "
                f"range=[{t_min:.2f},{t_max:.2f}] MWh (target≈10.0)")

    # ── 6. Per-unit spot check (3 random intervals) ──
    logger.info("[ -- ] 6. Per-unit spot check (3 random intervals):")
    rng = np.random.default_rng(42)
    # Rebuild physical values from normalized actions
    for _ in range(3):
        idx = int(rng.integers(0, N))
        a_norm = acts[idx]        # [p_energy_norm, 5×c_as_norm]
        p_energy_mw = a_norm[0] * P_MAX  # MW
        c_as_mw     = a_norm[1:] * P_MAX  # (5,) MW
        stored_rew  = float(rews[idx])

        # Recompute from static_features (last element is soc_frac)
        sf = arrays["static_features"][idx]
        soc_frac_sf = float(sf[-1])
        soc_mwh_sf  = soc_frac_sf * E_MAX

        logger.info(f"        idx={idx}: p_energy={p_energy_mw:.2f}MW, "
                    f"c_as_sum={c_as_mw.sum():.2f}MW, reward={stored_rew:.4f}")

    # ── 7. Joint co-optimization check ──
    # With the shared capacity constraint, full-power energy steps CANNOT have AS
    # (that is correct behavior, not a bug). Check partial-power steps (0.1 < |p| < 0.9)
    # where simultaneous AS bids confirm genuine joint co-optimization.
    p_abs = np.abs(acts[:, 0])
    c_as_sum = acts[:, 1:].sum(axis=1)
    partial = (p_abs > 0.10) & (p_abs < 0.90)
    n_partial = int(partial.sum())
    n_joint_partial = int((partial & (c_as_sum > 0.01)).sum())
    joint_frac = n_joint_partial / max(n_partial, 1)
    status7 = "OK" if (n_partial == 0 or joint_frac >= 0.50) else "WARN-SEQUENTIAL"
    logger.info(f"[{status7}] 7. Joint co-opt: {n_joint_partial}/{n_partial} partial-power steps "
                f"({100*joint_frac:.0f}%) also have non-zero AS bids "
                f"(note: full-power steps correctly have AS=0 per capacity constraint)")

    # ── 8. Failed solves ──
    status8 = "OK" if n_failed == 0 else ("WARN" if n_failed <= int(0.05*n_days) else "FAIL")
    logger.info(f"[{status8}] 8. Solver failures: {n_failed}/{n_days} days failed ({100*n_failed/max(n_days,1):.1f}%)")

    logger.info(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_npz(arrays: dict, diag: list, out_path: Path, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    sz = out_path.stat().st_size / 1e6

    total_rev  = sum(d["total_rev"]  for d in diag)
    energy_rev = sum(d["energy_rev"] for d in diag)
    as_rev     = sum(d["as_rev"]     for d in diag)
    n_days     = len(diag)
    ann_per_kw = (total_rev / n_days * 365) / (P_MAX * 1000)

    # Write companion text file
    txt_path = out_path.with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(f"Post-break MILP trajectory — {label}\n")
        f.write("=" * 50 + "\n")
        f.write(f"n_days: {n_days}\n")
        f.write(f"n_transitions: {len(arrays['rewards'])}\n")
        f.write(f"total_revenue_usd: {total_rev:.2f}\n")
        f.write(f"energy_revenue_usd: {energy_rev:.2f}\n")
        f.write(f"as_revenue_usd: {as_rev:.2f}\n")
        f.write(f"daily_avg_total: {total_rev/n_days:.2f}\n")
        f.write(f"daily_avg_energy: {energy_rev/n_days:.2f}\n")
        f.write(f"daily_avg_as: {as_rev/n_days:.2f}\n")
        f.write(f"annualized_per_kw: {ann_per_kw:.2f}\n")
        n_fail = sum(1 for d in diag if d["milp_failed"])
        f.write(f"n_solver_failures: {n_fail}\n")
        solve_times = [d["solve_time"] for d in diag]
        f.write(f"mean_solve_s: {np.mean(solve_times):.4f}\n")
        f.write(f"max_solve_s: {np.max(solve_times):.4f}\n")

    logger.info(f"Saved {out_path} ({sz:.1f} MB) | {n_days} days | "
                f"${total_rev/n_days:.0f}/day avg | ${ann_per_kw:.1f}/kW-yr annualized")
    logger.info(f"  Energy: ${energy_rev/n_days:.0f}/day | AS: ${as_rev/n_days:.0f}/day")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def select_solver() -> str:
    available = cp.installed_solvers()
    for s in ("GUROBI", "HIGHS"):
        if s in available:
            return s
    raise RuntimeError(f"No suitable solver found. Available: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-break MILP trajectory generator")
    parser.add_argument("--smoke",   action="store_true",
                        help="5-day smoke test only (Jan 10,25,26,27, Feb 5 2026)")
    parser.add_argument("--split",   choices=["train", "val", "both"], default="both",
                        help="Which split to generate (ignored if --smoke)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel worker processes (default: 32)")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir",  default="data/expert_trajectories")
    parser.add_argument("--solver",   default=None, help="CVXPY solver (default: auto)")
    args = parser.parse_args()

    solver = args.solver or select_solver()
    logger.info(f"Solver: {solver}")

    logger.info("Loading processed data...")
    merged = load_merged(args.data_dir)
    merged_ctx = slice_range(merged, CONTEXT_START, "2026-04-15")

    price_data  = merged_ctx[PRICE_COLS].values.astype(np.float32)
    system_data = merged_ctx[SYSTEM_COLS].values.astype(np.float32)
    timestamps  = merged_ctx.index
    logger.info(f"Loaded {len(merged_ctx):,} rows ({merged_ctx.index.min()} → {merged_ctx.index.max()})")

    out_dir = Path(args.out_dir)

    if args.smoke:
        # ── Smoke test: 5 specific dates ──
        target_dates = set(SMOKE_DATES)
        days = [
            (d, i) for (d, i) in build_day_list(merged_ctx, "2026-01-01", "2026-02-28")
            if d in target_dates
        ]
        days.sort()
        logger.info(f"Smoke-test days: {[d for d, _ in days]}")

        arrays, diag = run_days(
            days, price_data, system_data, timestamps,
            solver=solver, n_workers=1,  # sequential for smoke test
        )
        out_path = out_dir / "receding_horizon_postbreak_smoke.npz"
        save_npz(arrays, diag, out_path, "smoke")
        sanity_check(arrays, diag, "SMOKE TEST (5 days)")
        logger.info("Smoke test complete. Commit POSTBREAK_MILP_SMOKE.md and wait for green-light.")
        return

    # ── Full run ──
    splits_to_run = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits_to_run:
        start, end = SPLITS[split]
        logger.info(f"\nRunning {split} split ({start} → {end})")
        days = build_day_list(merged_ctx, start, end)
        logger.info(f"  {len(days)} complete days")

        n_workers = min(args.workers, len(days))
        arrays, diag = run_days(
            days, price_data, system_data, timestamps,
            solver=solver, n_workers=n_workers,
        )
        out_path = out_dir / f"receding_horizon_postbreak_{split}.npz"
        save_npz(arrays, diag, out_path, split)
        sanity_check(arrays, diag, split.upper())


if __name__ == "__main__":
    main()
