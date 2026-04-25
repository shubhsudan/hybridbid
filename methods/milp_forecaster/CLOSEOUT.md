# MILP+Forecaster Method Closeout (Method 1)

**Sprint:** sprint-offline-rl  
**Completed:** April 25, 2026  
**Session:** cc-baselines

---

## Result Summary

| Window       | Revenue       | $/kW-yr |
|---|---|---|
| All days     | $13,361.72    | 9.03    |
| Ex-Fern      | $11,147.64    | 7.68    |
| Fern-only    | $2,214.08     | —       |

**vs fleet median (+24.93 $/kW-yr):** −63.8%  
**vs MILP-replay ceiling (58.40 $/kW-yr):** −84.5%  
**vs PF oracle (63.16 $/kW-yr):** −85.7%  
**vs BC (29.16 $/kW-yr):** −69.0% — **worse than BC and both TBx baselines**  
**soc_ceiling_fraction:** 0.189  **soc_floor_fraction:** 0.290

---

## What Was Tried

**Forecaster architecture:** 4-layer transformer encoder (8 heads, 128-dim, ff=512, dropout=0.1). Input: log1p-transformed `price_history (32, 12)`. Output: `(288, 6)` forecast of next 24h RT prices [rt_lmp + 5 rt_mcpc]. 1.75M parameters.

**Training data:** 622,409 samples spanning Jan 2020 – Nov 2025 (6 years of ERCOT 5-min data). Val set: Dec 2025 (8,928 samples). T-60 window (Jan–Feb 2026) excluded from both.

**Training:** Adam, lr=5e-4, cosine schedule, MSE loss in log-space. Budget: 30k steps. Early-stopped at step 4000 (val improvement plateau): best val_loss=0.3898 at step 1000, no improvement for 3 consecutive 1k-step val checks.

**Policy:** At each CT midnight (54 times over T-60), the policy:
1. Passes `obs["price_history"]` through the transformer → forecasted (288, 6) RT prices
2. Applies inverse log1p transform → physical $/MWh
3. Solves 24h MILP LP (HiGHS, 10s timeout) with forecasted prices and current SoC
4. Buffers the 288-step action sequence, replays through the day

---

## What Failed

**Primary failure: regression to mean in the forecaster.**  
The train/val gap (train_loss≈0.11, val_loss≈0.39 at step 1000) indicates out-of-distribution generalization failure. The transformer learned average 2020–2024 price patterns but couldn't generalize to Dec 2025 patterns (likely different weather conditions, market structure evolution). At inference on T-60, the model outputs near-mean prices (~$20-30/MWh) for most days, giving the MILP no dispatch signal. With flat price forecasts and a terminal SoC penalty (λ=20), the MILP defaults to near-zero dispatch — hence 72% idle on p_energy.

**Consequence:** Most days see forecasted revenue of $120-240, which is roughly the "ambient" value of doing a single shallow cycle. The MILP executes this but misses high-value intraday spread opportunities.

**Secondary failure: one LP timeout (day 7, Jan 7 CT).**  
The HiGHS solver hit the 10-second timeout, returning zero actions for the entire day. The timeout was likely caused by extreme values in the inverse-transformed forecast (the transformer occasionally outputs large log-space values that invert to numerically challenging prices for LP). Zero actions for a full day cost ~$250 in expected revenue (∼2% of total).

**Structural cause: 32-step context window is insufficient for 24h lookahead.**  
A 32-step (2.67h) price history does not contain enough information to predict 24h ahead reliably. The transformer's val loss of 0.39 in log-space corresponds to an RMSE of ~0.63 in log-space, or approximately 2× multiplicative uncertainty in physical prices — far too noisy for MILP planning.

---

## Training Details

| | Value |
|---|---|
| Architecture | 4-layer TransformerEncoder (8 heads, 128-dim) + MLP head |
| Parameters | 1,747,264 |
| Training samples | 622,409 (Jan 2020 – Nov 2025) |
| Val samples | 8,928 (Dec 2025) |
| LR | 5e-4 cosine annealing |
| Batch size | 64 |
| Early stop | Step 1000 (patience=3 × 1000-step val intervals) |
| Best val_loss | 0.3898 (log-space MSE) |

---

## Artifacts

- `methods/milp_forecaster/checkpoints/forecaster_best.pt` — best checkpoint (step 1000)
- `methods/milp_forecaster/checkpoints/forecaster_training_log.json` — val loss log
- `data/results/eval_milpf/summary.json` — full T-60 metrics
- `data/results/eval_milpf/trajectory.parquet` — per-interval log

---

## Key Lesson for Results Write-Up

**MILP+forecaster is not a viable baseline with a naive 32-step transformer.**  
The method reveals a general principle: MILP optimization is only as good as its price forecast. With regression-to-mean forecasts, the MILP conservatively dispatches near-zero, earning less than a threshold policy that ignores optimization entirely (9.03 vs BC's 29.16 $/kW-yr).

This is a meaningful finding for the paper: it quantifies the "price of uncertainty" in the forecaster, and explains why BC (which learns dispatch patterns from a crystal-ball optimizer) can outperform a live-MILP approach with poor forecasts.

**What would be needed to make MILP+forecaster competitive:**
1. Longer context window (96+ steps = 8h history, or DAM prices as a 24h forward signal)
2. More expressive forecaster (seq2seq with separate decoder, or probabilistic model)
3. Training data closer in time to deployment (Dec 2025 prices better than 2020–2024 average)
4. Remove terminal SoC penalty at inference (MILP should not assume it needs to return to 50% under uncertainty)

None of these are within Day 2 sprint scope.

---

## Comparison Table Position

| Method | All $/kW-yr | vs Fleet Median | vs MILP-replay | vs PF Oracle |
|---|---|---|---|---|
| TBx energy-only | 10.96 | −56.0% | −81.2% | −82.6% |
| **MILP+forecaster** | **9.03** | **−63.8%** | **−84.5%** | **−85.7%** |
| TBx with AS | 12.52 | −49.8% | −78.6% | −80.2% |
| BC (Phase 2) | 29.16 | +17.0% | −50.1% | −53.8% |
| MILP-replay ceiling | 58.40 | +134.2% | 0% | −7.5% |
| PF oracle | 63.16 | +153.3% | +8.1% | 0% |

MILP+forecaster finishes last among all evaluated methods.
