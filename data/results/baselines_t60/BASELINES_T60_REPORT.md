# Baselines T-60 Report
**Date:** 2026-04-25
**Window:** 2026-01-01 → 2026-02-23 (54 days, CT-aligned)
**Session:** cc-baselines (Phase 1)

## Reference Ceilings

- MILP-replay continuous-SoC: **58.40 $/kW-yr** ($86,394.20 total, commit 92c5a49)
- Fleet median (T-60 window): 24.93 $/kW-yr
- Fleet top-quartile: 32.23 $/kW-yr

## Results

| Baseline | All-days $/kW-yr | Ex-Fern $/kW-yr | Fern-only $/kW-yr | vs Fleet Median | vs MILP-replay ceiling |
|----------|-----------------|-----------------|-------------------|-----------------|------------------------|
| tbx_energy_only | 10.96 | 9.62 | 82.07 | -56.0% | -81.2% |
| tbx_with_as | 12.52 | 11.20 | 82.08 | -49.8% | -78.6% |
| pf_t60 | 63.16 | 61.65 | 142.81 | +153.3% | +8.1% |

## Revenue Composition

### tbx_energy_only
- Energy revenue: $ 16,214.25
- AS revenue:     $      0.00  (0.0%)

### tbx_with_as
- Energy revenue: $ 16,214.25
- AS revenue:     $  2,304.57  (12.4%)

### pf_t60
- Energy revenue: $ 76,740.68
- AS revenue:     $ 16,695.03  (17.9%)

## PF Solve Info

- Approach: full_horizon
- Solve time: 4.0s
- LP revenue (pre-projection): $93,450.53

## Sanity Checks

Gate: PF > MILP-replay ceiling ($58.40/kW-yr): **PASS**
Gate: TBx_with_AS > TBx_energy_only: **PASS**

All sanity checks passed.

## Reward Convention Note

Baseline revenues are computed by running policies through the eval harness (`experiments/prepare_postbreak.py`), which computes:
  `energy_rev = p_energy_mw * rt_lmp * DT`
  `as_rev = c_as_mw * rt_mcpc * DT`
Physical $ throughout. The stored `rewards` field of the trajectory NPZ is never read by this code (TBx is purely rule-based; PF re-solves its own LP from market prices).
The mixed-unit convention in stored rewards does not affect these numbers.
