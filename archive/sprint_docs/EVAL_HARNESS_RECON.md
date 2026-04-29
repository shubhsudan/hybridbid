# Eval Harness Recon — Post-Break T-60 Window
**Date:** 2026-04-24
**Branch:** sprint-offline-rl
**Session:** cc-baselines

---

## 1. Existing `experiments/prepare.py` Interface

**What it takes:**
- `checkpoint_path: str` — path to `.pt` SAC checkpoint
- `config: Stage1Config` — hyperparameters including device, data_dir, seq_len
- `experiment_name: str` — label for the RESULT line

**What it does:**
- Instantiates `ERCOTBatteryEnv` on the pre-RTC+B test set (2025-10-01 → 2025-12-04)
- Runs 5 seeds × `evaluate()` from `src.evaluation.evaluate_stage1`
- Computes IQM (drop min/max of 5 seeds, mean of middle 3)
- Applies `net_return = iqm − 50 × soc_violations` to penalise dump-and-terminate policies
- **No file I/O** — only outputs to stdout
- Output format: one machine-parseable `RESULT` line, always last

**What it returns:**
```python
{
  "iqm_daily_revenue": float,   # $/day, IQM across seeds
  "net_daily_revenue": float,   # iqm − 50×violations
  "capture_rate": float,        # vs TBx $870/day
  "soc_violations": int,
  "n_days": int,
  "per_seed_revenues": list[float],
}
```

**File structure of outputs:** None — pure stdout. No parquet, no JSON, no markdown.

**Key locked constants** (from `experiments/program.md` — DO NOT MODIFY):
- Test range: 2025-10-01 → 2025-12-04
- TBx baseline: $870/day
- Eval seeds: [10, 11, 12, 13, 14]
- Primary metric: `net_return`

---

## 2. Policy Interface Expectations

**Pre-break harness (`evaluate_stage1.py`):**
```python
action = agent.select_action(obs: dict, deterministic: bool=True) → np.ndarray shape (4,)
```
- `obs` is a dict: `{"price_history": (32, 12) float32, "static_features": (14,) float32}`
- Returns 4D action: `[mode_logit(3), energy_mag(1)]` — Gumbel-Softmax format
- Not stateful (env manages EMA / SoC internally)
- Not batched (single observation)

**Post-break harness (new):**
```python
class PolicyInterface:
    def reset(self) -> None: ...
    def __call__(self, obs: dict) -> np.ndarray:  # shape (6,), physical MW
```
- Same obs format: `{"price_history": (32, 12), "static_features": (14,)}`
- Returns **6D physical MW** `[p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs]`
- `p_energy` signed (+ = discharge, − = charge); AS non-negative
- Stateful via `reset()` — called once before eval begins
- Not batched — single obs per call

**Critical difference from pre-break:** The action format changes from Gumbel-mode + p.u. magnitude (4D, network output) to direct physical MW (6D, post-projection). Method wrappers translate their model's native output into this contract.

---

## 3. Revenue Computation

**Pre-break (`evaluate_stage1.py` line 100):**
```python
day_revenue += info["energy_revenue"] * config.p_max
```
Where `info["energy_revenue"]` = `energy_mag × rt_lmp × (v_dch×η − v_ch/η) × Δt` — this is in **p.u.** (energy_mag ∈ [0,1]). Multiply by `P_max=10 MW` to get actual dollars.

AS revenue is NOT included in the pre-break revenue computation (energy_only mode).

**Post-break (new harness):**
```python
energy_rev  = p_energy_mw * rt_lmp * dt          # physical $, signed
as_rev_j    = c_as_j_mw  * rt_mcpc_j * dt        # physical $, per product
```
Both already in physical MW — no p.u. scaling needed. AS revenue is unconditional (ERCOT availability-based, not Li et al. deployment-based).

The `ercot_env.py` co_optimize step already computes AS revenue this way (`projected_action[i+1]` is in MW after `as_phys = as_mags * p_max`). The formula is reused verbatim.

**T-60 MILP reference revenues (CLARABEL, all 54 days solved fresh):**
| Segment | Revenue | $/kW-yr |
|---------|---------|---------|
| All 54 days | $90,814 | $61.38 |
| Ex-Fern (53 days) | $79,594 | $54.82 |
| Fern only (Jan 26) | $11,220 | $1.122/kW |

MILP replay validation target: within 2% of $90,814 (tolerance ±$1,816).

---

## 4. Feasibility Projection — Critical Deviations

`src/models/feasibility.py` `project_co_optimize` is **NOT** used verbatim in the post-break harness. Two deviations:

### A. AS sustain durations are wrong in feasibility.py

| Product | `feasibility.py` | MILP / ERCOT spec | Delta |
|---------|-----------------|-------------------|-------|
| regup | 0.5h | 1.0h | 2× |
| regdn | 0.5h | 1.0h | 2× |
| rrs   | 0.5h | 1/6h (10 min) | 3× |
| ecrs  | 1.0h | 1/4h (15 min) | 4× |
| nsrs  | 4.0h | 0.5h (30 min) | 8× |

Using `feasibility.py`'s durations would incorrectly clip MILP-generated actions (especially nsrs), causing spurious revenue discrepancies in the MILP replay validation. The post-break harness uses the MILP/ERCOT-correct durations.

### B. Shared capacity constraint differs

- `feasibility.py`: separate upward (`p_discharge + regup + rrs + ecrs ≤ P_max`) and downward (`p_charge + regdn ≤ P_max`)
- MILP: joint shared (`p_ch + p_dch + all_AS ≤ P_max`)

The post-break harness uses joint shared capacity (matching MILP). This ensures MILP-generated actions pass through projection unmodified when SoC is at the planned level.

**Both deviations are intentional and documented.** `src/env/ercot_env.py` is not modified (per task constraint). The projection is re-implemented in numpy with correct parameters.

---

## 5. Continuous SoC vs MILP Daily Reset

The eval harness runs a **continuous 54-day trajectory** (no midnight reset). The MILP trajectories were generated with **daily reset to 10 MWh** at the start of each day.

Consequence for MILP replay validation:
- The harness SoC at the start of each day = previous day's terminal SoC
- MILP actions assume SoC = 10 MWh at day start
- When terminal SoC ≠ 10 MWh, the harness feasibility projection may clip actions for the next day
- Revenue discrepancy is bounded by the terminal SoC deviation × (revenue opportunities lost from clipping)

From the MILP trajectory data: train terminal SoC mean=8.88 MWh, std=0.98 MWh. The deviation from 10.0 is typically 1–2 MWh, affecting the first few intervals of each day. Estimated revenue impact: <2% of total. The 2% replay tolerance accounts for this.

---

## 6. Observation Construction

Standard (non-enriched) mode matches `ercot_env._get_observation()` exactly:
- `price_history`: rolling (32, 12) float32 window of `PRICE_COLS`
- `static_features`: (14,) = system(7, normalized) + time(6, cyclical) + soc(1, p.u.)

No v6.0 enriched mode (36-dim TTFE, 32-dim static) — post-break methods use standard obs to match the MILP trajectory schema.

---

## 7. Output Structure (new — no equivalent in pre-break harness)

```
data/results/eval_{method_name}/
├── trajectory.parquet   # per-interval: timestamp, actions (MW), prices, revenues, SoC, cumulative
├── summary.json         # aggregate metrics (all_days/ex_fern/fern_only + fleet comparison)
└── comparison_card.md   # human-readable summary
```

Pre-break harness has no file output. Post-break harness adds this structure so all methods produce comparable artifacts in one directory.

---

## 8. Summary of Design Decisions

| Dimension | Pre-break harness | Post-break harness |
|-----------|-------------------|-------------------|
| Policy interface | checkpoint → SACAgent | PolicyInterface wrapping any model |
| Action format | 4D Gumbel + p.u. mag | 6D physical MW |
| Revenue units | p.u. × P_max → $ | Physical $ directly |
| AS revenue | Not included | Included, availability-based |
| SoC continuity | Daily reset (env handles) | Continuous, no reset |
| Output | stdout RESULT line | parquet + JSON + markdown |
| Feasibility | project_co_optimize (wrong sustains) | Numpy reimplementation (correct) |
| Fleet comparison | vs TBx $870/day | vs fleet median $24.93/kW-yr |
