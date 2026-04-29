# Post-Break MILP Trajectory Generation — Phase 0 Recon
**Date:** 2026-04-24
**Branch:** sprint-offline-rl
**Status:** STOP — flagging 6 issues before Phase 1 can proceed. Green-light required.

---

## 1. Observation Space Spec

**Result: Mismatch with task spec. Env does NOT produce a flat 90-dim vector.**

The env (`src/env/ercot_env.py`) returns a `gym.spaces.Dict` observation, not a flat array:

| Field | Shape | Content |
|-------|-------|---------|
| `price_history` | `(32, 12)` | Raw prices for TTFE input (12-dim × 32 timesteps) |
| `static_features` | `(14,)` | system(7) + time(6) + soc(1) |

The 90-dim described in the task spec (`64 TTFE + 12 raw prices + 7 system + 6 time + 1 SoC`) is the **network architecture** output — the 64-dim comes from running the TTFE network over `price_history`, not from the raw env observation. The raw env observation is the Dict above.

The existing pre-break trajectories (`_with_obs.npz`) store the Dict format directly:
- `price_history`: `(N, 32, 12)` float32
- `static_features`: `(N, 14)` float32

For the post-break trajectories, the observation dimension will be the same if we use the same standard (non-enriched) observation mode. **No change required to observation format — just confirm we use `enriched_obs=False, enriched_flat=False`.**

---

## 2. Pre-Break Trajectory Schema

**Result: Two schemas exist. Neither matches the task spec output format exactly.**

### Schema A — Raw traces (`receding_horizon_{split}.npz`)
| Key | Shape | dtype | Range | Notes |
|-----|-------|-------|-------|-------|
| `timestamps` | `(420424,)` | `<U25` | UTC ISO strings | |
| `modes` | `(420424,)` | `int8` | {0, 1, 2} | 0=charge, 1=discharge, 2=idle |
| `magnitudes` | `(420424,)` | `float32` | [0, 1] | normalized bid power |
| `socs` | `(420424,)` | `float32` | [2.0, 18.0] | MWh |
| `rewards_env` | `(420424,)` | `float32` | [-2357, +5779] | Li et al. Eq. 26 |
| `rewards_raw` | `(420424,)` | `float32` | [-6671, +7554] | plain energy revenue |
| `rt_lmp` | `(420424,)` | `float32` | [-216, +9065] | $/MWh |

### Schema B — Preprocessed (`receding_horizon_{split}_with_obs.npz` / `_option_d.npz`)
| Key | Shape | dtype | Notes |
|-----|-------|-------|-------|
| `price_history` | `(N, 32, 12)` | float32 | current obs TTFE input |
| `static_features` | `(N, 14)` | float32 | current obs static |
| `next_price_history` | `(N, 32, 12)` | float32 | next obs TTFE input |
| `next_static_features` | `(N, 14)` | float32 | next obs static |
| `actions` | `(N,)` | int64 | **quantized to 7 atoms** — not continuous 6D |
| `rewards` | `(N,)` | float32 | Li et al. Eq. 26 |
| `dones` | `(N,)` | bool | SoC violation terminations |
| `truncateds` | `(N,)` | bool | day-end truncations |
| `quantization_error` | `(N,)` | float32 | |
| `reward_milp_stored` | `(N,)` | float32 | |

**Task spec output format is different from both schemas:**

```
observations:      (N, obs_dim)        — flat? or dict?
actions:           (N, 6)              — continuous 6D
rewards:           (N,)
next_observations: (N, obs_dim)
dones:             (N,)
soc:               (N,)
```

**Issue:** The task spec format (flat `observations` array, continuous 6D `actions`) does not match either existing schema. The pre-break pipeline stored observations as split `price_history`/`static_features` arrays, not a single flat `observations` array. The existing offline-RL training code (e.g., `experiments/prepare.py`) will need to be consulted to know which format it expects.

**Recommendation:** Adopt Schema B style (separate `price_history`/`static_features` for obs and next_obs) but with continuous 6D actions instead of quantized 1D. Add `soc` array for diagnostics. This makes the post-break file structurally parallel to the pre-break `_option_d` file while accommodating the 6D action space.

---

## 3. Pre-Break MILP Battery Config

**Result: η mismatch between config file and MILP generator. MILP generator is correct.**

| Location | η_ch | η_dch | Degradation |
|----------|------|-------|-------------|
| `configs/battery.yaml` | **0.92** | **0.92** | $2.00/MWh |
| `src/utils/battery_sim.py` (BatteryParams default) | **0.92** | **0.92** | $2.00/MWh |
| `src/env/ercot_env.py` DEFAULT_BATTERY | **0.95** | **0.95** | $2.00/MWh |
| `src/data/receding_horizon_milp.py` (git history) | **0.95** | **0.95** | $0.00/MWh |
| CLAUDE.md spec | 0.95 | 0.95 | not in step reward |

The pre-break MILP explicitly hardcodes η=0.95 and degradation_cost=0.0, **overriding the BatteryParams defaults** that load from battery.yaml. The env also uses η=0.95.

**For post-break MILP:** hardcode η_ch=η_dch=0.95 and degradation_cost=0.0 directly in the MILP script. Do NOT use `BatteryParams.from_yaml()` — it will give η=0.92.

---

## 4. Pre-Break MILP Forecast Methodology

**Result: Pre-break is NOT per-interval receding horizon. Task spec requires a different design.**

### Pre-break actual design (from `receding_horizon_milp.py` git history + `.txt` metadata):
- **Commit window: 1 hour (12 intervals)**
- **Lookahead: 24 hours**
- **24 MILP solves per day** (one per hour, each committing 12 intervals)
- **NO daily SoC reset** — SoC carries across day boundaries continuously
- **Energy-only** (no AS products)
- Mean solve time: 0.149s with GUROBI, 0.134s max 5.12s

### Task spec design (post-break):
- **Commit window: 1 interval (5 minutes)** — per-interval
- **Lookahead: 24 hours**
- **288 MILP solves per day** — 12× more than pre-break
- **Daily SoC reset** to 0.5 at midnight
- **Joint 6D co-optimization** (energy + 5 AS products)

**Wall-clock implication:** The co-optimize MILP has more variables (6 action dimensions + capacity sharing constraints + AS feasibility constraints), so individual solve times will be higher than the 0.15s pre-break energy-only average. With 288 solves/day × 132 days = 37,904 total solves, if each solve takes 1-2s → 10-21 CPU-hours → ~20-40 minutes wall time with 32-way parallelism. This is plausible but tight.

**These are confirmed as deliberate design differences, not bugs.** The per-interval commitment and daily SoC reset are new design choices for the post-break trajectories as described in the task spec.

---

## 5. CVXPY + HiGHS Availability

**Result: Available on both M4 and Narnia. GUROBI NOT available in Narnia conda env.**

| Machine | CVXPY | Solvers available |
|---------|-------|-------------------|
| M4 (local) | 1.8.2 | CLARABEL, **GUROBI**, **HIGHS**, OSQP, SCIPY, SCS |
| Narnia (`hybridbid` conda env) | 1.8.2 | CLARABEL, **HIGHS**, OSQP, SCIPY, SCS |

**GUROBI is NOT in Narnia's `hybridbid` conda env.** The pre-break trajectories used GUROBI (per the `.txt` metadata: `solver: GUROBI`), likely via a separate WLS license set up outside conda. For the post-break run, HiGHS will be used on Narnia — this should work fine, just slightly slower.

---

## 6. Git Branch Status

**Result: Already on `sprint-offline-rl` ✓**

```
* sprint-offline-rl   (current)
  main
  ...
```

---

## CRITICAL BLOCKER: Data Coverage Gap

**AS prices and system_conditions are only available through March 31, 2026.**

| Table | Coverage | Gap |
|-------|----------|-----|
| `data/processed/energy_prices/` | Through 2026-04-15 | None |
| `data/processed/as_prices/` | Through **2026-03-31** | Apr 1–15 missing |
| `data/processed/system_conditions/` | Through **2026-03-31** | Apr 1–15 missing |

The task validation split is **Feb 11 → Apr 15, 2026**, but AS prices and system conditions are missing for Apr 1–15. This is a 2-week gap at the end of the validation window.

**Options (requires Karthik decision before proceeding):**
1. **Fetch April data from ERCOT API** — `src/data/ercot_fetcher.py` likely supports this; will take ~15-30 min to download and preprocess.
2. **Truncate validation split to Feb 11 → Mar 31, 2026** — 48 days instead of 63. Simpler but loses 2 weeks.
3. **Zero-fill April AS prices** — low quality for validation; the AS product bids will be wrong.

---

## SECONDARY FLAG: No Joint Energy+AS MILP Formulation Exists

The task requires a **6D co-optimization MILP** (energy + 5 AS products). The existing `src/baselines/perfect_foresight.py` only has `solve_energy_only_mip()` — no AS formulation exists in the codebase.

The post-break MILP script will need to build the joint LP/MIP from scratch. Key formulation elements needed (per task spec):
1. `|p_energy| + sum(c_AS) ≤ P_max` — shared capacity constraint
2. AS availability constraint: `c_AS * t_sustain ≤ (SoC - SoC_min) / η_dch` for each product
3. Soft terminal SoC penalty: `λ * (SoC[T] - 0.5)²` with λ=$20
4. Revenue: energy revenue + AS capacity payments (MCPC × bid × Δt)

**This is the main implementation work for Phase 1** — the MILP formulation itself must be written.

---

## Summary of Stop Conditions

| # | Issue | Severity | Decision needed |
|---|-------|----------|-----------------|
| A | AS + system_conditions data gap (Apr 1–15) | **BLOCKER** | Fetch data, truncate, or zero-fill? |
| B | Task spec output format differs from existing schemas | **Must decide** | Use split price_history/static_features, or flat 90-dim? |
| C | Commit window mismatch (1h pre-break vs 1-interval task spec) | **Informational** | Confirmed as deliberate new design |
| D | η=0.92 in battery.yaml/BatteryParams — must hardcode 0.95 in MILP | **Known fix** | Always use 0.95; never load from yaml |
| E | No joint energy+AS MILP formulation exists | **Implementation work** | Proceed once green-lit |
| F | GUROBI not in Narnia conda env → HiGHS only on Narnia | **Known** | HiGHS fine; acknowledge in Phase 1 |

**STOP. Waiting for green-light on items A and B before proceeding to Phase 1.**
