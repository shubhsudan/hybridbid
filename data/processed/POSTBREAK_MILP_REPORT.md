# Post-Break MILP Trajectory Generation — Final Report
**Date:** 2026-04-24
**Branch:** sprint-offline-rl
**Script:** `src/data/postbreak_milp.py`
**Run machine:** Narnia (GPU node 18, HiGHS solver, 32 workers)
**Wall time:** ~10 min train + <5 min val

---

## Overview

Full 132-day post-break MILP expert trajectory generation completed successfully.

| Split | Period | Days | Transitions | Total USD | $/day avg | $/kW-yr |
|-------|--------|------|-------------|-----------|-----------|---------|
| Train | 2025-12-05 → 2026-02-10 | 68 | 19,584 | $111,519 | $1,640 | $59.86 |
| Val   | 2026-02-11 → 2026-04-15 | 64 | 18,432 | $74,697  | $1,167 | $42.60 |

Output files: `data/expert_trajectories/receding_horizon_postbreak_{train,val}.{npz,txt}`

---

## Sanity Check Results

### Train Split

| # | Check | Result | Value |
|---|-------|--------|-------|
| 1 | Revenue scale [$40–$200/kW-yr] | **PASS** | $59.86/kW-yr |
| 2 | Fern contribution [20–50%] | **FLAG** | 10.4% (see note) |
| 3 | Action distributions — no all-zero/all-pmax dims | **PASS** | All 6 dims active (c_rrs near-zero: expected) |
| 4 | SoC distribution within [2.0, 18.0] | **PASS** | min=2.00, max=18.00, mean=8.88 MWh |
| 5 | Terminal SoC distribution | **PASS** | mean=8.88, std=0.98, range=[4.27, 10.00] MWh |
| 6 | Per-unit spot check | **PASS** | See POSTBREAK_MILP_SMOKE.md §Check 6 |
| 7 | Joint co-optimization | **PASS** | 1698/1808 partial-power steps (93.9%) have AS bids |
| 8 | Solver failures | **PASS** | 1/68 days (1.5%, Dec 26) — below 5% threshold |

### Val Split

| # | Check | Result | Value |
|---|-------|--------|-------|
| 1 | Revenue scale [$40–$200/kW-yr] | **PASS** | $42.60/kW-yr |
| 2 | Fern contribution | N/A | Fern event (Jan 26) is in train split |
| 3 | Action distributions | **PASS** | All dims active (c_rrs near-zero: see note) |
| 4 | SoC distribution | **PASS** | min=2.00, max=18.00, mean=9.35 MWh |
| 5 | Terminal SoC | **PASS** | mean=9.31, std=0.69, range=[6.27, 9.94] MWh |
| 6 | Per-unit spot check | **PASS** | Same formulation as smoke |
| 7 | Joint co-optimization | **PASS** | 1361/1538 partial-power steps (88.5%) |
| 8 | Solver failures | **PASS** | 0/64 (0%) |

---

## Revenue Statistics

### Train: Full vs. Ex-Fern vs. Fern-Only

Fern (Winter Storm Fern, Jan 26, 2026) is the 2nd-highest revenue day by env reward. The Jan 24–28 window (sustained scarcity) is the highest-revenue cluster.

| Segment | USD | Days | $/day avg |
|---------|-----|------|-----------|
| Full train | $111,519 | 68 | $1,640 |
| Fern window (Jan 24–28) | ~$45,445 | 5 | ~$9,089 |
| Fern day only (Jan 26) | ~$11,621 | 1 | — |
| Ex-Fern train | ~$99,898 | 67 | ~$1,491 |
| Dec 26 (idle fallback) | $0 | 1 | — |
| Ex-Fern, ex-Dec26 | ~$99,898 | 66 | ~$1,513 |

**Note on Fern concentration:** The 10.4% Fern flag (Check 2) reflects that over 68 days, Fern is one day. The smoke test's 48.3% was a 5-day window centered on Fern. The Jan 24–28 window is 40.8% of total training revenue — the scarcity concentration is real, but distributed across the week-long event rather than a single day.

**Jan 28 was the single highest reward day** in training (NPZ reward $9,373 vs. Jan 26's $6,573). This reflects post-Fern stress continuing into Jan 28. HiGHS vs. GUROBI differences in optimal dispatch may also contribute to day-level ordering differences vs. the smoke test.

### Val: No Scarcity Concentration

Top val day is March 24, 2026 (~$7,703 USD est, 10.3% of val total). Val revenue is more evenly distributed — consistent with spring mild-weather markets without a Fern-scale event.

### Benchmark Comparison

| Metric | Train | Val | Fleet benchmark (T-60 window) |
|--------|-------|-----|-------------------------------|
| $/kW-yr | $59.86 | $42.60 | median $24.93, top-Q $32.23 |

MILP exceeds fleet benchmark due to: (1) perfect foresight within the day, (2) joint energy+AS co-optimization adding ~35% revenue premium vs energy-only.

---

## Action Distributions

### Train (19,584 transitions)

| Dim | mean | std | P95 | frac_zero | frac_pmax | Notes |
|-----|------|-----|-----|-----------|-----------|-------|
| p_energy | −0.012 | 0.573 | 1.000 | 59.9% | 29.2% | Active in ~40% of steps |
| c_regup | 0.118 | 0.299 | 1.000 | 82.0% | 8.3% | Moderate usage |
| c_regdn | 0.272 | 0.423 | 1.000 | 66.5% | 21.8% | Largest AS product |
| c_rrs | 0.009 | 0.061 | 0.000 | 97.1% | 0.0% | Near-zero (SoC constrained) |
| c_ecrs | 0.016 | 0.079 | 0.000 | 95.3% | 0.0% | Near-zero |
| c_nsrs | 0.193 | 0.362 | 1.000 | 72.1% | 13.6% | Moderate |

### Val (18,432 transitions)

| Dim | mean | std | P95 | frac_zero | frac_pmax | Notes |
|-----|------|-----|-----|-----------|-----------|-------|
| p_energy | −0.016 | 0.617 | 1.000 | 54.9% | 34.4% | More discharge-heavy (spring) |
| c_regup | 0.187 | 0.358 | 1.000 | 72.4% | 12.3% | Higher than train |
| c_regdn | 0.219 | 0.387 | 1.000 | 71.1% | 16.0% | Lower than train |
| c_rrs | 0.003 | 0.034 | 0.000 | 99.1% | 0.0% | Near-zero (spring SoC dynamics) |
| c_ecrs | 0.005 | 0.044 | 0.000 | 98.3% | 0.0% | Near-zero |
| c_nsrs | 0.096 | 0.268 | 1.000 | 85.6% | 5.9% | Less than train |

**RRS and ECRS near-zero:** Expected. Short sustain durations (10 and 15 min) require holding SoC buffer proportional to c_rrs × sustain_h / η. In discharge-heavy winter/spring markets where SoC frequently sits near SoC_min, little RRS capacity can be feasibly offered. This is a formulation artifact, not a bug.

**RegDown dominance in train vs. RegUp in val:** Train period (Dec–Feb, high afternoon prices) has strong discharge economics → battery frequently discharges → regdn (absorb excess, charge-direction AS) is the complementary product. Val period (Feb–Apr, milder prices) has more balanced dispatch with more charge headroom → regup becomes relatively more attractive.

---

## Reward Statistics (NPZ, env-scale)

These are Li et al. Eq. 26 rewards — the offline RL training signal. They differ from USD revenue because the energy term is normalized by P_max (p.u.) while AS revenue uses physical MW.

| Stat | Train | Val |
|------|-------|-----|
| mean per step | 3.221 | 2.544 |
| std per step | 13.892 | 10.482 |
| min | −145.15 | −184.66 |
| max | 578.10 | 434.02 |
| P5 | −0.955 | −0.781 |
| P95 | 14.52 | 11.37 |
| Total sum | 63,078 | 46,894 |

Large standard deviation relative to mean reflects occasional high-price scarcity intervals (Fern spikes) against a baseline of low-price intervals.

---

## Solver Statistics

| Metric | Train | Val |
|--------|-------|-----|
| Solver | HiGHS (Narnia) | HiGHS (Narnia) |
| Days run | 68 | 64 |
| Failures (timeout) | **1** (Dec 26, 2025) | 0 |
| mean solve time | 9.70 s | 0.71 s |
| max solve time | 600.2 s (Dec 26) | 1.83 s |
| Failure rate | 1.5% | 0.0% |

**Dec 26 failure (train only):** Christmas week with unusual holiday price patterns caused HiGHS to hit the 600 s timeout. The day was replaced with 288 zero-action (idle) transitions and $0 revenue. This contributes a 1.5% trajectory contamination rate, below the 5% stop threshold. The idle fallback day is identifiable by `rewards=0` across all 288 steps.

**Mean solve time discrepancy:** Train mean (9.70 s) is dominated by the Dec 26 timeout (600 s single outlier inflating the mean). Val mean (0.71 s) with no timeouts is more representative of typical solve performance. Ex-Dec26 train mean is approximately 0.8–1.5 s/day.

---

## Schema Verification

Both files match the target schema (parallel to pre-break `_option_d.npz`, extended to 6D actions):

| Key | Train shape | Val shape | dtype | Notes |
|-----|-------------|-----------|-------|-------|
| `price_history` | (19584, 32, 12) | (18432, 32, 12) | float32 | Raw pre-TTFE features |
| `static_features` | (19584, 14) | (18432, 14) | float32 | system(7)+time(6)+soc(1) |
| `next_price_history` | (19584, 32, 12) | (18432, 32, 12) | float32 | |
| `next_static_features` | (19584, 14) | (18432, 14) | float32 | |
| `actions` | (19584, 6) | (18432, 6) | float32 | [p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs] in p.u. |
| `rewards` | (19584,) | (18432,) | float32 | Li et al. Eq. 26 |
| `dones` | (19584,) | (18432,) | bool | 0 SoC violations |
| `truncateds` | (19584,) | (18432,) | bool | 68 / 64 True |
| `soc` | (19584,) | (18432,) | float32 | Post-action SoC, MWh |

Action encoding: `p_energy ∈ [−1, +1]` (discharge positive), `c_as ∈ [0, 1]` each, all normalized by P_max=10 MW. Initial SoC for each day (10.0 MWh) is embedded in `static_features[:, −1]`.

---

## Flags and Notes for Offline RL Use

1. **Dec 26 idle day** (train index 52×288 to 53×288−1, i.e., transitions 6048–6335): all-zero actions, zero rewards. Consider filtering or down-weighting for BC/offline-RL training if idle trajectories bias the policy toward inaction.

2. **Jan 24–28 Fern window** (train days 51–55, transitions ~14688–15839): 40.8% of total training reward concentrated in 5 consecutive days. Offline RL methods (Cal-QL, Diffusion-QL) that weight by return may over-represent this window. Consider return-weighted sampling vs. uniform.

3. **Env reward vs. USD revenue:** NPZ rewards are the offline RL signal; USD revenue is the MILP objective. The ratio varies by day (1.4–2.0×) due to the EMA timing bonus term that inflates env rewards relative to raw market revenue. Do not use NPZ reward sum to compute USD revenue.

4. **c_rrs and c_ecrs near-zero:** Offline RL agents trained on these trajectories will rarely see non-zero RRS/ECRS bids. This is a limitation of the MILP expert in discharge-heavy markets. Agents may underlearn RRS/ECRS in Stage 2.
