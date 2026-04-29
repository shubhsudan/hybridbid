# Dec 26 Timeout Root Cause Investigation
**Date:** 2026-04-24
**Branch:** sprint-offline-rl

---

## Summary

Root cause: **HiGHS QP solver instability**. Dec 26 is not a structurally hard optimization problem. CLARABEL solves it in 27ms to the certified optimal. Zero other training days were degraded. The train NPZ has been patched; the idle fallback has been replaced with the correct CLARABEL solution.

---

## Root Cause

The MILP objective contains a quadratic terminal-SoC penalty term:

```
maximize  Σ(energy_rev + AS_rev) - λ*(soc[T] - soc_target)²
```

This makes the problem a **quadratic program (QP)**, not a linear program. HiGHS uses an interior-point method for QP, which is distinct from its simplex LP solver. On Dec 26, HiGHS's QP interior-point solver failed to converge within the 600-second timeout.

**The problem is not numerically difficult.** Dec 26 prices are entirely ordinary:
- rt_lmp: min=$9.38, max=$46.11, mean=$15.96, std=$6.00 — normal low-demand winter day
- No negative prices, no extreme spikes, no NaN
- rt_mcpc_regup has only 35/288 non-zero intervals — sparse but not degenerate

**CLARABEL** (Clarabel.rs interior-point QP solver) solves the identical problem in **27ms** with status `optimal`. This confirms the issue is HiGHS-specific, not a property of Dec 26's price data.

**Likely mechanism:** HiGHS's QP path applies a Schur-complement factorization update at each interior-point iteration. With sparse AS prices (mostly-zero rt_mcpc_regup creates near-zero reduced costs for `c_regup`), the solver may encounter near-singular factorizations, leading to extremely slow convergence or numerical stall. The CVXPY 1.8.2 + HiGHS interface on Narnia (Conda env) may also use older HiGHS QP configuration defaults that do not include the heuristic workarounds present in newer releases.

---

## Scope of Degraded Solutions

All 68 training days were re-solved locally with CLARABEL as a reference solver:

| Metric | Result |
|--------|--------|
| Days with `optimal` status | 68 / 68 |
| Days with `optimal_inaccurate` | 0 / 68 |
| Solve times | mean=22.6ms, max=34.7ms, all < 35ms |
| Days with non-zero CLARABEL revenue | 68 / 68 |

**Cross-validation against Narnia HiGHS totals:**
- CLARABEL total (68 days): $112,053.84
- Narnia HiGHS total (68 days): $111,519.25
- Difference: **$534.59 = exactly Dec 26's optimal revenue**

This confirms that the 67 non-Dec26 days solved to the same optimum on both solvers. The train NPZ was contaminated only by the single Dec 26 idle fallback. No other days were degraded.

---

## Dec 26 Optimal Solution (CLARABEL)

| Metric | Idle fallback (original) | CLARABEL optimal (patched) |
|--------|--------------------------|---------------------------|
| Status | user_limit (timeout) | optimal |
| Revenue | $0.00 | $534.59 |
| Energy revenue | $0.00 | $397.18 |
| AS revenue | $0.00 | $137.41 |
| Terminal SoC | 10.00 MWh (unchanged) | 9.23 MWh |
| Active intervals (|p|>0.01) | 0/288 | 117/288 |
| AS-active intervals | 0/288 | 196/288 |
| Solve time | >600s (HiGHS timeout) | 0.027s (CLARABEL) |

Dec 26 is the 14th-highest-revenue day in the 68-day training set — a mid-range day with normal Christmas-week price patterns.

---

## Impact of Patch

The idle fallback had two effects on the training corpus:

1. **Revenue contamination (minor):** $534.59 missing from a $112,054 total = **0.48% of total training revenue**. Offline RL methods reward-weight by return, so Dec 26's contribution is limited even with the correct solution.

2. **Action distribution contamination (significant):** 288 all-zero action transitions appear as the dominant "idle" pattern. These could bias:
   - **BC (behavioral cloning):** idle actions get 288 gradient steps on Dec 26, artificially inflating the frequency of idle behavior in loss-weighted updates.
   - **Cal-QL / IQL:** transitions with zero reward cluster near the bottom of the return distribution; if used as a reference for pessimistic value estimation, they pull down Q-values for Dec 26-like price conditions.
   - With the patch, Dec 26 shows 117 active energy steps and 196 AS-active steps — consistent with the rest of the training set.

---

## Corrected Train Statistics

| Metric | Original (with idle fallback) | Patched (CLARABEL Dec 26) |
|--------|-------------------------------|---------------------------|
| total_revenue_usd | $111,519.25 | **$112,053.84** |
| daily_avg_total | $1,639.99 | **$1,647.85** |
| annualized_per_kw | $59.86/kW-yr | **$60.15/kW-yr** |
| n_solver_failures | 1 | **0** |
| mean_solve_s | 9.70s (dominated by 600s timeout) | **0.87s** |
| max_solve_s | 600.22s | **~1.83s** |

---

## Fix for Future Runs

The HiGHS QP solver is vulnerable to stalls on this problem class. Two options:

**Option A (recommended): CLARABEL as primary solver.**
CLARABEL is an interior-point cone solver that handles QPs natively and reliably. On all 68 training days, it solves in 20–35ms. Replace `solver="HIGHS"` with `solver="CLARABEL"` in `src/data/postbreak_milp.py`, and remove the `time_limit` kwarg (CLARABEL doesn't support it, but also doesn't need it).

**Option B: LP reformulation.**
Replace the quadratic terminal penalty with a piecewise-linear approximation:

```python
# Instead of: LAMBDA_TERMINAL * cp.square(soc[T] - SOC_TARGET)
# Use:
slack = cp.Variable(nonneg=True)
constraints.append(slack >= soc[T] - SOC_TARGET)
constraints.append(slack >= SOC_TARGET - soc[T])
# penalty: LAMBDA_TERMINAL_L1 * slack  (linear)
```

This converts the QP to a pure LP, which HiGHS handles reliably and fast. The L1 penalty has slightly different terminal SoC distribution characteristics than L2, but the effect is minor at the λ=20 scale.

**Recommendation:** Use CLARABEL (Option A). It's a one-line change, eliminates the QP/LP distinction, and requires no reformulation. Val split already has 0 failures with HiGHS (all < 2s), suggesting the val price distribution is easier — but CLARABEL eliminates the risk entirely.

---

## Files Changed

- `data/expert_trajectories/receding_horizon_postbreak_train.npz` — Dec 26 transitions (indices 6048–6335) replaced with CLARABEL-optimal solution
- `data/expert_trajectories/receding_horizon_postbreak_train.txt` — corrected revenue stats and solver failure count
- `data/expert_trajectories/dec26_patch.npz` — intermediate patch file (can be deleted after commit)
