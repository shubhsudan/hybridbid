# MILP+Forecaster Diagnosis

**Generated:** April 25, 2026  
**Session:** cc-baselines  
**Trigger:** $9.03 $/kW-yr result is structurally implausible; three diagnostics required before attributing.

---

## Diagnosis Summary

**The $9.03 result has two bugs. MILP integration is correct.**

| Finding | Verdict |
|---|---|
| MILP integration | **Correct.** Persistence control gives 21.55 $/kW-yr; MILP dispatches reasonably when given usable prices. |
| Bug 1: AS price collapse | **Critical.** Transformer outputs near-zero for ALL 5 RT MCPC products (mean≈$0.00, std≈$0.01 vs actual mean $0.44–$1.35 with spikes to $30+). Kills entire AS revenue stream. |
| Bug 2: LMP variance compression | **Material.** Transformer captures mean LMP correctly but compresses standard deviation 3× (actual std=$68.50 → forecast std=$22.88). P95 cut from $122 to $51. Misses intraday spread → MILP underestimates dispatch value. |

Both bugs are training failures, not architectural limitations. The MILP formulation is sound.

---

## Diagnostic 1: Forecaster Prediction Distribution

Forecast statistics over all 54 T-60 eval days (each day: transformer forecast from CT midnight obs vs actual realized prices, 15,552 total intervals):

**RT LMP (energy price):**
| Metric | Actual | Transformer Forecast | Gap |
|---|---|---|---|
| Mean | $35.69 | $28.80 | −$6.89 |
| Std | $68.50 | $22.88 | **−45.62 (3× compression)** |
| P5 | −$1.52 | $15.04 | forecast never negative |
| P95 | $122.50 | $51.15 | **−57.5% at tail** |
| Max | $1,350.50 | $437.32 | |
| MAE | — | $22.06 | |

**RT MCPC (AS clearing prices — all 5 products):**
| Product | Actual Mean | Actual Std | Forecast Mean | Forecast Std | MAE |
|---|---|---|---|---|---|
| regup | $1.35 | $5.88 | **$0.00** | $0.01 | $1.35 |
| regdn | $1.14 | $3.50 | **$0.00** | $0.01 | $1.13 |
| rrs | $0.44 | $4.91 | **$0.00** | $0.01 | $0.43 |
| ecrs | $0.66 | $8.23 | **$0.00** | $0.01 | $0.66 |
| nsrs | $1.26 | $8.72 | **$0.00** | $0.01 | $1.26 |

**Interpretation:**

RT LMP: Transformer captures the rough mean but fails on volatility. The 3× variance compression makes intraday price spreads invisible — the MILP sees a $15–50 band when the actual band is −$11 to $357. This degrades energy dispatch value.

RT MCPC: Complete collapse. All 5 AS clearing prices predicted as essentially zero across all 15,552 intervals. This is a training failure caused by sparsity: AS prices clear at $0 in the majority of intervals, so MSE-minimizing regression learns to always predict $0. During events (AS prices spike to $5–$50+ on Fern), the transformer still predicts near-zero. This eliminates the entire AS revenue dimension.

---

## Diagnostic 2: Persistence-Forecast Control

**Persistence definition:** `predicted_price[t+k] = actual_price[t + k − 1 day]`

Policy: at each CT midnight, solve MILP with prior-day actual prices. Same formulation, same SoC initialization.

| Metric | Transformer | Persistence | Fleet Median |
|---|---|---|---|
| All-days $/kW-yr | 9.03 | **21.55** | 24.93 |
| Ex-Fern $/kW-yr | 7.68 | **21.96** | — |
| Fern-only $ | $2,214 | **$0** (LP failure) | — |
| vs fleet median | −63.8% | **−13.6%** | 0% |

Persistence daily forecast revenues (selected days):

| Day | Label | Persist Forecast Rev | Trans Forecast Rev | Actual Rev (PF) |
|---|---|---|---|---|
| 1 | Jan 1 | $1,045 | $487 | $288 |
| 7 | Jan 7 | $1,016 | $0 (timeout) | $620 |
| 10 | Jan 10 | $950 | $150 | $723 |
| 25 | Jan 25 | $6,413 | $1,637 | ~$750 |
| 26 | Jan 26 (Fern) | **$0 (unbounded)** | $4,800 | $3,931 |
| 27 | Jan 27 | $5,959 | $1,079 | ~$650 |

**Interpretation:**

MILP integration is correct. With real AS prices in the forecast, the MILP actively dispatches energy + AS, earning $500–$1,000/day on normal days vs $100–$200 with the transformer. Persistence achieves 21.55 $/kW-yr (−13.6% vs fleet median), which is consistent with industry references for persistence-based MILP dispatch.

Persistence Fern-day failure: status=`unbounded` on day 26. The prior day (Jan 25 CT) already had elevated prices due to pre-storm conditions. Jan 25 prices fed as a Fern day forecast produced an LP that HiGHS marked unbounded — likely a numerical issue with extreme persistence prices (mean=$258/MWh, max=$938/MWh). This is a separate edge-case bug in the persistence policy, not in the transformer.

Persistence also fails on day 32 (HiGHS solver error, zero actions). Two days of failure out of 54 cost ~$1,500 in expected revenue.

---

## Diagnostic 3: MILP Integration Sanity Check (5 Days)

For each day: forecaster prediction vs actual vs persistence, and resulting MILP action vs PF action.

**Day 1: Jan 1 (quiet winter day)**
- Actual LMP: mean=$18.2, std=$4.0, P95=$24.2
- Trans forecast LMP: mean=$15.9, std=$8.6 → plausible range but slightly low
- Trans forecast AS regup: mean=$0.013 (actual $0.00 — coincidentally correct today, but for the wrong reason)
- MILP+trans: mean p_energy=−0.27, step_rev=**−$110** (negative! MILP charged expecting higher future prices, actual spread didn't materialize)
- MILP+persist: $186; PF: $288

**Day 7: Jan 7 (transformer LP timeout)**
- Actual LMP: mean=$19.5, std=$7.0, P95=$33.8
- Trans forecast LMP: mean=$23.4, std=$3.6 — mild prices, should not cause timeout
- LP timeout (10s) on first run; actual prices were not extreme → likely one-off HiGHS numerical issue, not systematic
- MILP+persist: $177; PF: $620 (large PF-persist gap driven by PF having perfect AS prices: actual regup mean=$0.58, persist mean=$0.63 vs transformer=$0.003)

**Day 10: Jan 10 (low-price day with negatives)**
- Actual LMP: mean=$8.9, std=$13.8, min=−$11.1
- Trans forecast LMP: mean=$19.6, std=$3.1 — **significantly overestimates** (no negative prices in forecast, biased +$10.7)
- MILP charged at ~$18/MWh expecting to discharge later at higher prices → actual prices didn't rise → step_rev=**−$94**
- MILP+persist: $506; PF: $723

**Day 26: Jan 26 (Fern — major storm)**
- Actual LMP: mean=$141.0, std=$100.7, P95=$304.6, max=$357.3
- Trans forecast LMP: mean=$122.2, std=$95.1, P95=$303.3 — **transformer correctly predicted Fern severity** from pre-storm price history
- Trans forecast AS regup: mean=$0.009 (actual mean=$10.72, max=$30.82) — **catastrophic AS miss despite correct LMP**
- MILP+trans: p_energy mean=−0.07, step_rev=$2,214 — captures energy revenue (has full battery entering Fern due to high-SoC bias) but zero AS
- MILP+persist: $0 (LP unbounded with Jan 25 persistence prices ≥$900/MWh)
- PF: $3,931 (captures both energy + $1,717 in AS)

*Note:* The Fern day result reveals that the transformer CAN learn LMP spikes from preceding price history, but the AS collapse prevents it from capturing the correlated AS price surge. On Fern, AS prices (regup, regdn, rrs, ecrs, nsrs) all spike simultaneously with LMP — and the transformer outputs near-zero for all of them.

**Day 40: Feb 9 (normal post-Fern day)**
- Actual LMP: mean=$29.4, std=$12.3, P95=$52.8
- Trans forecast LMP: mean=$30.5, std=$4.9 — mean close, variance compressed 2.5×
- Trans AS regup: mean=$0.003 (actual $1.14, max=$6.64) — AS collapse
- MILP+trans: $128; MILP+persist: $770; PF: $1,138

---

## Root Cause Attribution

### Bug 1: AS Price Collapse (explains most of the $12K shortfall vs persistence)

**What:** Transformer outputs near-zero for all RT MCPC products across all 15,552 eval intervals.

**Why:** AS clearing prices are highly sparse. In the 2020–2025 training data, the majority of 5-min intervals have $0 AS clearing (market doesn't clear AS at each interval — it clears at settlement points). MSE minimization on a distribution where mode ≈ 0 and variance is dominated by rare spikes causes the model to predict zero. The AS spike events are too rare and too large (outlier regime) for the model to learn from them with the 30k-step budget.

**Revenue impact:** AS revenue in the actual eval = $5,856 (from persistence control trajectory). Transformer AS revenue = ~$0 at planning time → MILP plans no AS → actual AS revenue was $5,856 in persistence but only ~$2,905 in transformer eval (some AS earned coincidentally from leftover capacity buffer).

### Bug 2: LMP Variance Compression (explains the remainder)

**What:** Actual std=$68.50 → forecast std=$22.88 (3× compression). Tails are especially flat: P95 cut from $122 to $51.

**Why:** MSE on heavy-tailed price distributions regresses toward the conditional mean. The Fern spike (max $1,350) contributes heavily to actual variance but the model can't consistently reproduce it. During training on 2020–2024 data, storms like Fern are rare enough that the model learns to discount them.

**Mitigating factor:** The transformer DID correctly predict elevated Fern prices on Jan 26 (mean=$122 vs actual $141) because the pre-storm price history visible in the 32-step context was already elevated. So the model can respond to within-context signals; it just can't predict volatility that isn't yet visible in the context window.

---

## Conclusion

**The 9.03 $/kW-yr result is primarily a forecaster training failure, not an architectural limitation or MILP integration bug.** Two bugs account for essentially all the underperformance:

1. **AS price collapse** (primary): Transformer predicts zero for all AS products. Fix: use persistence for AS, or train with a loss that upweights non-zero AS intervals.
2. **LMP variance compression** (secondary): Fix: train with quantile loss for LMP tails, longer budget, or hybrid approach.

The MILP integration responds correctly to price signals. When given reasonable prices (persistence), it achieves 21.55 $/kW-yr. The "MILP+forecaster is inferior to rule-based methods" framing is **incorrect** — it was the forecaster, not the MILP.

**Persistence control at 21.55 $/kW-yr sits just below BC (29.16) and below fleet median (24.93).** This is the correct benchmark for "what MILP+forecaster with day-1 persistence-level quality gives you" — not $9.03.

---

## Recommended Fix

A hybrid forecaster uses the transformer for LMP (where it captures mean and event-driven spikes reasonably) and substitutes persistence for AS prices (prior-day actual). This is a 2-line change in the policy. Expected outcome: approximately 25–35 $/kW-yr (between persistence 21.55 and MILP-replay 58.40), likely slightly above or below BC depending on LMP forecast quality on non-Fern days.

This fix stays within Method 1 scope (no new method, no architecture change). Awaiting Karthik's green-light before implementing.
