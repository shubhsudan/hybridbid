# MILP+Forecaster Method Closeout (Method 1)

**Sprint:** sprint-offline-rl  
**Completed:** April 25, 2026 (Phase 3 with diagnosis and hybrid investigation)  
**Session:** cc-baselines

---

## Final Result: Clamped-Persistence + CLARABEL Fallback

| Window       | Revenue       | $/kW-yr |
|---|---|---|
| All days     | $33,794.93    | 22.84   |
| Ex-Fern      | $31,250.88    | 21.52   |
| Fern-only    | $2,544.05     | —       |

**vs fleet median (+24.93 $/kW-yr):** −8.4%  
**vs MILP-replay ceiling (58.40 $/kW-yr):** −60.9%  
**vs PF oracle (63.16 $/kW-yr):** −63.8%  
**vs BC (29.16 $/kW-yr):** −21.7% — below BC  
**soc_ceiling_fraction:** 0.160  **soc_floor_fraction:** 0.198

---

## How We Got Here (Four Variants)

| Variant | All $/kW-yr | Notes |
|---|---|---|
| Pure transformer | 9.03 | AS collapse + LMP compression — bugged |
| Persistence (raw) | 21.55 | Two LP failures (Fern + Feb 1) — both earn $0 |
| Hybrid (trans LMP + clim AS) | 18.59 | Below $22 bound — allocation bias |
| **Clamped-persistence + CLARABEL** | **22.84** | Production result, sanity OK |

See `DIAGNOSIS.md` for the full diagnostic chain.

---

## Original Transformer Failure Modes

### Bug 1 (Critical): AS Price Collapse

**What:** All 5 RT MCPC (AS clearing price) outputs are near-zero across all eval intervals — mean≈$0.00, std≈$0.01 vs actual mean $0.44–$1.35 with event spikes to $30+.

**Why:** AS clearing prices are sparse. The vast majority of intervals have $0 clearing (market doesn't clear AS at each 5-min interval). MSE minimization on sparse targets regresses to the mode (zero). The transformer learned to always predict zero to avoid incurring loss on the few non-zero observations. During Fern, actual regup reached $30/MWh; transformer predicted $0.009.

**Impact:** Eliminated the entire AS revenue stream from MILP planning. Actual AS revenue with good forecasts (persistence): ~$11,237.

### Bug 2 (Material): LMP Variance Compression

**What:** Forecast std = $22.88 vs actual std = $68.50 (3× compression). P95 cut from $122 to $51. Max forecast $437 vs actual max $1,350.

**Why:** MSE on heavy-tailed price distributions regresses toward the conditional mean. Fern-scale spikes (max $1,350) contribute disproportionately to actual variance; the model discounts them after a few training epochs (they're rare). After 4k training steps, the model had converged to predicting near-mean prices.

**Partially mitigating factor:** The transformer did correctly anticipate Fern severity on Jan 26 (forecast mean=$122 vs actual $141, P95=$303 vs actual=$305) because pre-storm prices were already visible in the 32-step context window. The model can respond to within-context signals; it cannot predict volatility not yet visible.

**Outstanding limitation:** LMP variance compression is NOT fixed by the persistence approach. Quantile loss or volatility-weighted training would address this; deferred to future work. The MILP with persistence compensates by using actual prior-day price variance (which naturally captures storm dynamics through temporal autocorrelation).

---

## Methodology Framing Update

**Original framing (incorrect):** "Price of forecaster uncertainty — MILP optimization with poor forecasts is dominated by rule-based methods."  
This was premature. Persistence (same MILP, prior-day actual prices) gives 21.55–22.84 $/kW-yr vs transformer's 9.03. The MILP formulation is sound.

**Corrected framing for paper:**  
"Single MSE-trained forecaster fails on heterogeneous price streams: dense LMP (benefits from transformer's temporal pattern learning) and sparse AS (MSE collapses to zero-prediction on near-zero-modal distributions). Task-appropriate per-stream forecasting recovers most of the value: persistence for both streams reaches 22.84 $/kW-yr (−8.4% vs fleet median) while the naive MSE transformer earns 9.03 $/kW-yr (−63.8% vs fleet median)."

**Additional finding from hybrid investigation:** The two bugs interact non-linearly. Fixing only Bug 1 (AS collapse) with climatological mean AS, while Bug 2 (LMP compression) remains, distorts the MILP's AS/energy allocation ratio. With mean-AS (always-on $1.40/h for nsrs) and compressed-LMP ($15-35 range), the MILP over-allocates to AS (85.7% of capacity), earning high AS revenue ($19,425) but negligible energy revenue ($8,083) — net 18.59 $/kW-yr, worse than persistence. Both bugs must be fixed simultaneously for the hybrid approach to work.

---

## Policy: Clamped-Persistence with CLARABEL Fallback

**Forecaster:** 1-day lag persistence. `predicted_price[t+k] = actual_price[t + k - 1 day]`

**Price clamping:** LMP clipped at ±$500/MWh, MCPC clipped at $100/MWh. Prevents LP unbounded status on extreme storm prices (Jan 25 CT had max LMP $938 due to pre-Fern conditions).

**Solver fallback:** HiGHS primary (≤10s), CLARABEL fallback on HiGHS failure/unbounded. CLARABEL handled 2/54 days (Jan 26 = Fern, Feb 1). Without fallback: 2 days earn $0 (−2.29 $/kW-yr shortfall).

**54 MILP solves, all optimal.** Average solve time 0.3s; Fern day 10s (CLARABEL).

---

## Training Details (Transformer — archived)

| | Value |
|---|---|
| Architecture | 4-layer TransformerEncoder (8 heads, 128-dim) + MLP head |
| Parameters | 1,747,264 |
| Training samples | 622,409 (Jan 2020 – Nov 2025) |
| Val samples | 8,928 (Dec 2025) |
| Early stop | Step 1000 (patience=3 × 1000-step intervals) |
| Best val_loss | 0.3898 (log-space MSE) |

---

## Artifacts

- `methods/milp_forecaster/checkpoints/forecaster_best.pt` — transformer checkpoint
- `methods/milp_forecaster/checkpoints/as_climate_hod.npy` — (24, 5) climatological AS table
- `methods/milp_forecaster/DIAGNOSIS.md` — full diagnostic chain and investigation
- `data/results/eval_milpf_persist_final/summary.json` — production result
- `data/results/eval_milpf_persist_final/trajectory.parquet` — per-interval log
- `data/results/eval_milpf/` — pure transformer result (archived, bugged)
- `data/results/eval_milpf_hybrid/` — hybrid result (archived, allocation-biased)
- `data/results/eval_milpf_persist/` — unclamped persistence (archived, 2 failures)

---

## Comparison Table Position

| Method | All $/kW-yr | vs Fleet Median | vs MILP-replay |
|---|---|---|---|
| MILP+forecaster (transformer, bugged) | 9.03 | −63.8% | −84.5% |
| TBx energy-only | 10.96 | −56.0% | −81.2% |
| TBx with AS | 12.52 | −49.8% | −78.6% |
| **MILP+forecaster (persistence+CLARABEL)** | **22.84** | **−8.4%** | **−60.9%** |
| BC (Phase 2) | 29.16 | +17.0% | −50.1% |
| MILP-replay ceiling | 58.40 | +134.2% | 0% |
| PF oracle | 63.16 | +153.3% | +8.1% |

Method 1 (MILP+forecaster) sits between TBx and BC — above rule-based thresholds but below BC's expert imitation.

**The RL methods (Methods 3/4/5) need to beat BC at 29.16 $/kW-yr to justify offline RL over BC, and MILP+forecaster at 22.84 to justify offline RL over classical MILP-based dispatch.**
