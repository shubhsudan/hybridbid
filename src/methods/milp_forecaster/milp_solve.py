"""
24-hour MILP re-solve for MILP+forecaster policy.

Wraps the same CVXPY LP formulation as pf_policy._solve_lp, applied to
a 288-step forecasted price horizon (instead of perfect-foresight prices).

Re-used as-is from methods/baselines/pf_policy.py; kept here to avoid
cross-module imports from non-shared baselines code.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

P_MAX   = 10.0
E_MAX   = 20.0
ETA     = 0.95
SOC_MIN = 2.0
SOC_MAX = 18.0
DT      = 5.0 / 60.0

AS_SUSTAIN_H = np.array([1.0, 1.0, 1.0 / 6.0, 1.0 / 4.0, 0.5], dtype=np.float64)

DAILY_TIMEOUT  = 10.0   # seconds per 288-step solve
TERMINAL_LAMBDA = 20.0  # soft terminal SoC penalty (matches training MILP)


def solve_daily_milp(
    rt_lmp_forecast:  np.ndarray,    # (288,) $/MWh — forecasted
    rt_mcpc_forecast: np.ndarray,    # (288, 5) $/MWh — forecasted
    soc_init: float,
    use_terminal_penalty: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Solve 24h (288-step) MILP with forecasted prices and current SoC.

    Terminal SoC penalty λ*(SoC_T - 0.5)^2 discourages extreme SoC at day end,
    matching the training MILP formulation (λ=20, target=50%).

    Returns
    -------
    actions_mw : (288, 6) float32  [p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs]
    meta       : dict with status, revenue, solve_time
    """
    T = 288
    rt_lmp  = np.clip(rt_lmp_forecast.astype(np.float64), -9999, 9999)
    rt_mcpc = np.clip(rt_mcpc_forecast.astype(np.float64), 0, 9999)

    p_ch    = cp.Variable(T, nonneg=True)
    p_dch   = cp.Variable(T, nonneg=True)
    c_regup = cp.Variable(T, nonneg=True)
    c_regdn = cp.Variable(T, nonneg=True)
    c_rrs   = cp.Variable(T, nonneg=True)
    c_ecrs  = cp.Variable(T, nonneg=True)
    c_nsrs  = cp.Variable(T, nonneg=True)
    soc     = cp.Variable(T + 1)
    c_as_vars = [c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs]

    energy_rev = cp.sum(cp.multiply(rt_lmp, p_dch - p_ch)) * DT
    as_rev = cp.sum(
        cp.sum(cp.multiply(rt_mcpc[:, j], c_as_vars[j])) * DT
        for j in range(5)
    )
    objective_expr = energy_rev + as_rev

    if use_terminal_penalty:
        soc_target = E_MAX * 0.5   # 10 MWh = 50%
        terminal_pen = TERMINAL_LAMBDA * cp.square(soc[-1] - soc_target)
        objective_expr = objective_expr - terminal_pen

    objective = cp.Maximize(objective_expr)

    avail_dch = (soc[:-1] - SOC_MIN) * ETA
    avail_ch  = (SOC_MAX  - soc[:-1]) * ETA

    constraints = [
        soc[0] == soc_init,
        soc[1:] == soc[:-1] + ETA * p_ch * DT - (p_dch / ETA) * DT,
        soc >= SOC_MIN,
        soc <= SOC_MAX,
        p_ch  <= P_MAX,
        p_dch <= P_MAX,
        p_ch + p_dch + c_regup + c_regdn + c_rrs + c_ecrs + c_nsrs <= P_MAX,
    ]
    for j, c_j in enumerate(c_as_vars):
        if j == 1:
            constraints.append(c_j * AS_SUSTAIN_H[j] <= avail_ch)
        else:
            constraints.append(c_j * AS_SUSTAIN_H[j] <= avail_dch)

    prob = cp.Problem(objective, constraints)
    t0 = time.time()
    try:
        prob.solve(solver="HIGHS", verbose=False, time_limit=DAILY_TIMEOUT)
    except Exception as exc:
        prob._status = f"error:{exc}"
    solve_time = time.time() - t0

    # CLARABEL fallback: used when HiGHS reports unbounded or errors on extreme-price instances
    # (matches the training MILP fallback pattern from src/data/postbreak_milp.py)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        highs_status = prob.status
        try:
            prob.solve(solver="CLARABEL", verbose=False)
        except Exception:
            pass
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return np.zeros((T, 6), dtype=np.float32), {
                "status": f"highs:{highs_status}/clarabel:{prob.status}",
                "solve_time": time.time() - t0,
            }
        solve_time = time.time() - t0

    def _v(var):
        arr = np.array(var.value, dtype=np.float64).flatten()
        np.nan_to_num(arr, nan=0.0, posinf=P_MAX, neginf=0.0, copy=False)
        return np.clip(arr, 0.0, P_MAX)

    p_ch_v  = _v(p_ch)
    p_dch_v = _v(p_dch)
    c_as_v  = np.column_stack([_v(c) for c in c_as_vars])

    p_energy = p_dch_v - p_ch_v
    actions  = np.column_stack([p_energy, c_as_v]).astype(np.float32)
    rev      = float(np.sum(p_energy * rt_lmp) * DT + np.sum(c_as_v * rt_mcpc) * DT)

    return actions, {
        "status":     prob.status,
        "revenue":    rev,
        "solve_time": solve_time,
        "soc_end":    float(soc.value[-1]),
    }
