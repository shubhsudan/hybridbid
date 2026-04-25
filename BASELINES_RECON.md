# Phase 0 Recon — cc-baselines
**Date:** 2026-04-25
**Branch:** sprint-offline-rl
**Session:** cc-baselines (M4)

---

## 1. Git State

- Branch: `sprint-offline-rl` ✓
- Modified tracked files: `CLAUDE.md`, `data/expert_trajectories/receding_horizon_postbreak_train.npz`, `data/expert_trajectories/receding_horizon_postbreak_val.npz` (and .txt)
- Untracked: eval sweep scripts, `logs/`, new trajectory files — none in cc-baselines domain
- Working tree is not clean, but no cc-baselines owned files are uncommitted or conflicted.

---

## 2. Existing Baseline Code

### 2a. TBx — `src/baselines/tbx.py`

**Function signatures:**
```python
identify_tbx_schedule(prices: pd.Series, n_charge_hours=4, n_discharge_hours=4) → pd.DataFrame
run_tbx(prices: pd.Series, params: BatteryParams, n_charge_hours=4, n_discharge_hours=4) → pd.DataFrame
run_tbx_daily(prices: pd.Series, params: BatteryParams, n_charge_hours=4, n_discharge_hours=4) → pd.DataFrame
```

**Action space:** **1D energy-only.** Uses `BatteryAction(p_charge_mw=..., p_discharge_mw=...)`. All 5 AS capacity dimensions are hard-coded to 0. Returns `BatterySimulator.get_history_df()` — not in `PolicyInterface` format.

**Logic:** Identify cheapest 4h (charge) and most expensive 4h (discharge) per day; dispatch full available power in each window.

**UTC-vs-CT bug:** `hourly_prices.groupby(hourly_prices.index.date)` — if the index is UTC-aware, `.date` returns UTC dates, not CT dates. The same bug exists in `run_tbx_daily` at the outer `prices.groupby(prices.index.date)` groupby. **This is the Day 1 bug class we already fixed in the eval harness; confirmed present here too.**

**Eta:** Uses `BatteryParams` which defaults to `eta_charge=eta_discharge=0.92` and reads from `configs/battery.yaml` (also 0.92). Sprint requires 0.95 hardcoded.

**Degradation:** `BatterySimulator.step()` includes degradation cost in `net_revenue_usd`. CLAUDE.md excludes degradation from rewards. For fair comparison, T-60 baselines must use no-degradation revenue (energy + AS only).

---

### 2b. Perfect Foresight MIP — `src/baselines/perfect_foresight.py`

**Function signatures:**
```python
solve_energy_only_mip(prices: np.ndarray, params: BatteryParams, soc_initial=None, solver=None, verbose=False) → dict
run_perfect_foresight(prices: pd.Series, params: BatteryParams, horizon_hours=24, step_hours=24, solver=None) → pd.DataFrame
run_perfect_foresight_daily(prices: pd.Series, params: BatteryParams, solver=None) → pd.DataFrame
```

**Action space:** **1D energy-only.** `solve_energy_only_mip` only models `p_charge` (MW) and `p_discharge` (MW) with no AS variables. Joint 6D co-optimization entirely absent.

**Formulation differences vs. sprint MILP:**
- Uses a binary mutual-exclusivity variable `u(t) ∈ {0,1}` (Big-M MIP). Sprint's `postbreak_milp.py` uses a continuous LP with shared-capacity constraint (`p_ch + p_dch + Σ_j c_j ≤ P_max`) — no binary variable.
- Includes `degradation_cost_per_mwh` in objective. CLAUDE.md spec: no degradation cost.
- No soft terminal SoC penalty.
- No AS sustain feasibility constraints.

**UTC-vs-CT bug at line 291:** `run_perfect_foresight_daily` loops `for date, day_prices in prices.groupby(prices.index.date)` — same UTC-date groupby bug. **Flagged in TIMEZONE_AUDIT.md, deferred to main; confirmed here.**

**Eta:** Same issue — reads `BatteryParams` from `configs/battery.yaml` (0.92).

**run_perfect_foresight (rolling-horizon):** Does carry SoC window-to-window (`soc_initial=sim.soc_mwh` at each window). The function is structurally correct for continuous SoC — just energy-only.

---

### 2c. run_baselines.py — `src/baselines/run_baselines.py`

Not directly reused for sprint. Uses `configs/battery.yaml` (stale eta), energy-only formulation, and pre/post split at UTC `2025-12-05`. **Will not be called for Phase 1.**

---

## 3. Eval Harness Contract — `experiments/prepare_postbreak.py`

**Status:** Frozen, validated (Day 1). Reproduced MILP-replay $58.40/kW-yr ($86,394.20).

**PolicyInterface:**
```python
class PolicyInterface:
    def reset(self) -> None: ...
    def __call__(self, obs: dict) -> np.ndarray:  # shape (6,), physical MW
        # obs = {"price_history": (32, 12) float32, "static_features": (14,) float32}
        # returns [p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs]
        # p_energy signed (+ discharge, − charge); c_as ≥ 0
```

**Harness correctness checklist (verified):**
- [x] DT = 5/60 throughout
- [x] Physical MW (not p.u.) from policy; MILPReplayPolicy multiplies p.u. NPZ by P_max
- [x] CT timezone alignment in `_find_t60_indices`, `_build_obs`
- [x] AS unconditional (no energy-dispatch gate)
- [x] Continuous SoC across 54-day window, initial SoC_INIT = 10.0 MWh
- [x] Correct ERCOT AS sustain durations in `project_action` (not `src/models/feasibility.py` — that file has wrong values per harness comments)
- [x] Three-way split: all_days / ex_fern / fern_only
- [x] `summary.json` and `comparison_card.md` written per eval

---

## 4. Trajectory Schema Spot-Check

**File:** `data/expert_trajectories/receding_horizon_postbreak_train.npz`

| Key | Shape | dtype | Notes |
|-----|-------|-------|-------|
| price_history | (19584, 32, 12) | float32 | Rolling 32-step window, 12 price dims |
| static_features | (19584, 14) | float32 | 7 system + 6 cyclical time + 1 SoC |
| next_price_history | (19584, 32, 12) | float32 | |
| next_static_features | (19584, 14) | float32 | |
| actions | (19584, 6) | float32 | p.u.: p_energy ∈ [-1,1], c_as ∈ [0,1] |
| rewards | (19584,) | float32 | Per-interval revenue, p.u. |
| dones | (19584,) | bool | False (SoC feasibility maintained by MILP) |
| truncateds | (19584,) | bool | True at last step of each day |
| soc | (19584,) | float32 | MWh (diagnostic) |

Matches CLAUDE.md schema spec. N=19584 = 68 days × 288 steps/day ✓

**Val file:** `data/expert_trajectories/receding_horizon_postbreak_val.npz` — N=17856 (62 days × 288) ✓

**DISCREPANCY TO FLAG:** CLAUDE.md data state section references files at:
- `data/processed/receding_horizon_postbreak_train_option_d.npz`
- `data/processed/receding_horizon_postbreak_val_option_d.npz`

These paths **do not exist.** The actual files are at:
- `data/expert_trajectories/receding_horizon_postbreak_train.npz` (confirmed train)
- `data/expert_trajectories/receding_horizon_postbreak_val.npz` (confirmed val)

The schemas are identical to the `_option_d` spec — the `_option_d` naming convention denotes "Dict-style schema" and the postbreak files already use it, just without the suffix in the filename. The eval harness CLI defaults already reference the correct `data/expert_trajectories/` paths.

**This is a documentation discrepancy, not a data quality issue.** The eval harness is correct. The `_option_d` naming convention was not applied to postbreak files — the postbreak files *are* the option_d schema.

---

## 5. What Phase 1 Must Build (vs. Reuse)

Phase 1 **cannot reuse** any of the existing baseline code as-is due to:

1. **Eta = 0.92** throughout existing code; sprint requires 0.95
2. **Energy-only action space** (1D); T-60 baselines must produce 6D output through `PolicyInterface`
3. **UTC-vs-CT groupby bugs** in both `tbx.py` and `perfect_foresight.py`
4. **Degradation cost** included in `BatterySimulator` revenue; must be excluded
5. **BatterySimulator** is not compatible with the eval harness `PolicyInterface` — it manages state internally, but the harness manages SoC externally. TBx and PF policies must be stateless wrappers that receive the obs dict (including SoC) and return an action.

**Plan for Phase 1 implementations:** Write new `PolicyInterface`-compatible wrappers in `methods/baselines/tbx_policy.py` and `methods/baselines/pf_policy.py`. These will be fresh implementations using the harness's SoC-as-input pattern, not the `BatterySimulator`-manages-state pattern. The existing `src/baselines/` files are read-only reference only.

---

## 6. Summary Table

| Item | Status |
|------|--------|
| Branch: sprint-offline-rl | ✓ |
| Eval harness validated ($58.40/kW-yr) | ✓ |
| Trajectory files confirmed (correct schema) | ✓ |
| Pre-break TBx file found | ✓ (energy-only, UTC bug, eta=0.92) |
| Pre-break PF file found | ✓ (energy-only, UTC bug, eta=0.92, no AS) |
| CLAUDE.md path discrepancy (option_d) | **FLAG** (doc only, not blocking) |
| Phase 1 reuse of src/baselines/ | **No — must write fresh PolicyInterface wrappers** |
| Phase 1 blocking issues | None |

---

**STOP: Awaiting Karthik's green-light for Phase 1.**
