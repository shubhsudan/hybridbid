"""
Perfect Foresight MIP oracle for T-60 window.

Solves the joint 6D energy+AS LP over the full 54-day eval window (15,552
intervals) with continuous SoC — no daily reset, no terminal penalty.
This is the absolute ceiling: optimal revenue with a crystal ball.

Formulation mirrors src/data/postbreak_milp.py exactly, minus:
  - Daily SoC reset (single continuous horizon)
  - Soft terminal SoC penalty (λ=0; oracle doesn't need end-of-horizon nudge)

Fallback: if the 15,552-step LP times out in TIMEOUT_FULL seconds, solves 8
weekly sub-horizons (~2016 steps each) with continuous SoC between windows.
"""

import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np

P_MAX   = 10.0
E_MAX   = 20.0
ETA     = 0.95
SOC_MIN = 2.0
SOC_MAX = 18.0
SOC_INIT = 10.0
DT      = 5.0 / 60.0   # hours per 5-min step

# ERCOT AS sustain durations [h]: [regup, regdn, rrs, ecrs, nsrs]
AS_SUSTAIN_H = np.array([1.0, 1.0, 1.0 / 6.0, 1.0 / 4.0, 0.5], dtype=np.float64)

TIMEOUT_FULL    = 300.0   # seconds — try single full horizon
TIMEOUT_WEEKLY  = 180.0   # seconds — per-week fallback


def _solve_lp(
    rt_lmp:  np.ndarray,   # (T,) $/MWh
    rt_mcpc: np.ndarray,   # (T, 5) $/MWh, columns = [regup, regdn, rrs, ecrs, nsrs]
    soc_init: float,
    timeout_s: float,
) -> dict:
    """
    Solve the joint energy+AS LP for a time horizon of T intervals.

    Returns dict with:
      status   : str ('optimal', 'optimal_inaccurate', or error string)
      p_ch     : (T,) float64 [MW]
      p_dch    : (T,) float64 [MW]
      c_as     : (T, 5) float64 [MW]
      soc      : (T+1,) float64 [MWh]
      solve_time: float [s]
      revenue  : float [$] physical
    """
    T = len(rt_lmp)

    p_ch    = cp.Variable(T, nonneg=True, name="p_ch")
    p_dch   = cp.Variable(T, nonneg=True, name="p_dch")
    c_regup = cp.Variable(T, nonneg=True, name="c_regup")
    c_regdn = cp.Variable(T, nonneg=True, name="c_regdn")
    c_rrs   = cp.Variable(T, nonneg=True, name="c_rrs")
    c_ecrs  = cp.Variable(T, nonneg=True, name="c_ecrs")
    c_nsrs  = cp.Variable(T, nonneg=True, name="c_nsrs")
    soc     = cp.Variable(T + 1, name="soc")

    c_as_vars = [c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs]

    # Objective: physical $ revenue, no terminal penalty
    energy_rev = cp.sum(cp.multiply(rt_lmp, (p_dch - p_ch))) * DT
    as_rev = cp.sum(
        cp.sum(cp.multiply(rt_mcpc[:, j], c_as_vars[j])) * DT
        for j in range(5)
    )
    objective = cp.Maximize(energy_rev + as_rev)

    constraints = [
        soc[0] == soc_init,
        soc[1:] == soc[:-1] + ETA * p_ch * DT - (p_dch / ETA) * DT,
        soc >= SOC_MIN,
        soc <= SOC_MAX,
        p_ch  <= P_MAX,
        p_dch <= P_MAX,
        # Shared capacity: energy + all AS ≤ P_max
        p_ch + p_dch + c_regup + c_regdn + c_rrs + c_ecrs + c_nsrs <= P_MAX,
    ]

    # AS SoC sustain feasibility (matches postbreak_milp.py formulation exactly)
    avail_dch = (soc[:-1] - SOC_MIN) * ETA   # MWh available for discharge-direction AS
    avail_ch  = (SOC_MAX - soc[:-1]) * ETA   # MWh available for charge-direction AS
    for j, c_j in enumerate(c_as_vars):
        if j == 1:  # regdn: charge direction
            constraints.append(c_j * AS_SUSTAIN_H[j] <= avail_ch)
        else:       # regup, rrs, ecrs, nsrs: discharge direction
            constraints.append(c_j * AS_SUSTAIN_H[j] <= avail_dch)

    prob = cp.Problem(objective, constraints)

    t0 = time.time()
    try:
        prob.solve(solver="HIGHS", verbose=False, time_limit=timeout_s)
    except Exception as exc:
        return {"status": f"error:{exc}", "solve_time": time.time() - t0}
    solve_time = time.time() - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"status": prob.status or "timeout", "solve_time": solve_time}

    def _v(var):
        arr = np.array(var.value, dtype=np.float64).flatten()
        np.nan_to_num(arr, nan=0.0, posinf=P_MAX, neginf=0.0, copy=False)
        return np.clip(arr, 0.0, P_MAX)

    p_ch_v  = _v(p_ch)
    p_dch_v = _v(p_dch)
    c_as_v  = np.column_stack([_v(c) for c in c_as_vars])   # (T, 5)
    soc_v   = np.array(soc.value, dtype=np.float64).flatten()

    rev = float(np.sum((p_dch_v - p_ch_v) * rt_lmp) * DT
                + np.sum(c_as_v * rt_mcpc) * DT)

    return {
        "status":     prob.status,
        "p_ch":       p_ch_v,
        "p_dch":      p_dch_v,
        "c_as":       c_as_v,
        "soc":        soc_v,
        "solve_time": solve_time,
        "revenue":    rev,
    }


def _actions_from_result(result: dict, T: int) -> np.ndarray:
    """Convert LP result to (T, 6) physical MW action array."""
    p_energy = result["p_dch"] - result["p_ch"]   # signed: + discharge, - charge
    return np.column_stack([p_energy, result["c_as"]]).astype(np.float32)


def solve_pf_milp(
    rt_lmp:  np.ndarray,   # (T,) where T = 15552 for T-60
    rt_mcpc: np.ndarray,   # (T, 5)
    soc_init: float = SOC_INIT,
) -> tuple[np.ndarray, dict]:
    """
    Solve perfect-foresight oracle LP for the T-60 window.

    Tries full single-horizon first (TIMEOUT_FULL seconds). Falls back to
    8 weekly sub-horizons with continuous SoC if the full LP times out.

    Returns
    -------
    actions_mw : (T, 6) float32 physical MW
                 [p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs]
    meta       : dict with solve info
    """
    T = len(rt_lmp)
    print(f"[pf_policy] Solving full {T}-step LP (timeout={TIMEOUT_FULL:.0f}s)...")

    t_start = time.time()
    result = _solve_lp(rt_lmp, rt_mcpc, soc_init, timeout_s=TIMEOUT_FULL)
    elapsed = time.time() - t_start

    if result["status"] in ("optimal", "optimal_inaccurate"):
        actions_mw = _actions_from_result(result, T)
        rev = result["revenue"]
        print(
            f"[pf_policy] Full LP solved: status={result['status']} "
            f"revenue=${rev:,.2f} solve_time={result['solve_time']:.1f}s"
        )
        return actions_mw, {
            "approach":   "full_horizon",
            "status":     result["status"],
            "revenue":    rev,
            "solve_time": result["solve_time"],
        }

    print(
        f"[pf_policy] Full LP status={result['status']} after {elapsed:.1f}s. "
        "Falling back to weekly sub-horizons."
    )

    # Fallback: solve in weekly chunks with continuous SoC
    actions_mw = np.zeros((T, 6), dtype=np.float32)
    soc = soc_init
    week_steps = 7 * 288   # 2016 per week
    n_weeks = (T + week_steps - 1) // week_steps
    week_revenues = []
    week_statuses = []

    for w in range(n_weeks):
        ws = w * week_steps
        we = min(ws + week_steps, T)
        n  = we - ws

        print(f"[pf_policy]   Week {w+1}/{n_weeks} steps [{ws}:{we}] soc_init={soc:.2f} MWh")
        res = _solve_lp(
            rt_lmp[ws:we], rt_mcpc[ws:we], soc,
            timeout_s=TIMEOUT_WEEKLY,
        )
        week_statuses.append(res["status"])

        if res["status"] in ("optimal", "optimal_inaccurate"):
            actions_mw[ws:we] = _actions_from_result(res, n)
            soc = float(res["soc"][-1])
            week_revenues.append(res["revenue"])
            print(
                f"[pf_policy]     solved: ${res['revenue']:,.2f}  "
                f"soc_end={soc:.2f}  t={res['solve_time']:.1f}s"
            )
        else:
            week_revenues.append(0.0)
            # Carry SoC forward unchanged (zero actions = no SoC change)
            print(f"[pf_policy]     FAILED ({res['status']}): using zero actions this week")

    total_rev = sum(week_revenues)
    print(f"[pf_policy] Weekly fallback complete: total_revenue=${total_rev:,.2f}")

    return actions_mw, {
        "approach":      "weekly_fallback",
        "n_weeks":       n_weeks,
        "week_statuses": week_statuses,
        "week_revenues": week_revenues,
        "total_revenue": total_rev,
    }


class PrecomputedPolicy:
    """
    Replays a pre-solved action sequence through the eval harness.
    Ignores obs (actions are determined at construction time by the MILP solve).
    """

    def __init__(self, actions_mw: np.ndarray):
        # actions_mw: (T, 6) float32, physical MW
        self.actions_mw = actions_mw
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def __call__(self, obs: dict) -> np.ndarray:
        if self._step >= len(self.actions_mw):
            return np.zeros(6, dtype=np.float32)
        action = self.actions_mw[self._step].copy()
        self._step += 1
        return action
