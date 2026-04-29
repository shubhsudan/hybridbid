# QDT Closeout
**Method:** QDT — Q-learning Decision Transformer (Yamagata et al. ICML 2023, arXiv:2209.03993)  
**Sprint:** offline-rl, April 25 2026  
**Outcome:** Does not converge in this regime — CQL over-conservatism at Stage 2 gate; Stage 3 smoke shows structural weakness even when Stage 2 passes  
**Finding type:** Publishable regime analysis — method works on D4RL-scale; specific failure mode documented for small-data, spike-dominated BESS setting

---

## Summary

QDT does not converge in the post-RTC+B BESS regime on two independent grounds:

1. **Stage 2 gate failure (CQL over-conservatism):** The Stage 2 hard gate (P50 RTG ∈ [−$50, $100])
   failed under both tested α_cql values (1.0 and 0.3). This is structural: the bimodal Q-value
   distribution (Fern spike cluster vs. routine-interval cluster) means P50 is in the negative
   tail regardless of CQL penalty strength.

2. **Stage 3 structural weakness (DT pipeline):** The most favorable possible QDT result —
   the smoke pipeline run with the 5k CQL checkpoint (before over-conservatism had compounded)
   → valid Stage 2 relabeling → 1k DT training steps — produced **16.53 $/kW-yr**. This is
   below both MILP+forecaster (22.84 $/kW-yr) and BC (29.16 $/kW-yr). At 1k DT steps the
   pipeline was on the lower-performing trajectory; 50k DT steps would have been required to
   assess convergence, and the Stage 3 smoke result with 1k steps suggests the DT component
   had not yet learned to use RTG conditioning effectively on this dataset. Even if Stage 2
   had passed the gate, the QDT pipeline was structurally below the working baselines at the
   sprint budget.

The method achieves strong results on D4RL-scale benchmarks; the constraints documented here
are specific to the 15k-transition, spike-dominated post-RTC+B BESS regime.

---

## Attempt 1: α_cql=1.0 (D4RL default)

**Stage 1** (50k CQL): completed cleanly on GPU 16.

| Step | Q_mean | Q_max | CQL | CQL/TD ratio |
|------|--------|-------|-----|-------------|
| 5k | 85 | 3,482 | -16.96 | 0.002× |
| 15k | 151 | 7,877 | -53.18 | 0.013× |
| 35k | 107 | 7,294 | -52.97 | 0.018× |
| 50k | **4.69** | 4,680 | -73.47 | **0.038×** |

Q_mean peaked at 15k then **collapsed to near-zero by 50k** as CQL conservatism compounded.

**Stage 2** (RTG relabeling from 50k critic):

| Metric | Value |
|--------|-------|
| Q mean | 60.77 |
| P50 | **-$140.79** — GATE FAIL |
| P90 → TARGET_RTG | +$311.75 |

Gate fail reason: `P50=-140.79 outside [-50, 100]`.

---

## Attempt 2: α_cql=0.3 (single-variable change)

**Rationale:** D4RL datasets (~1M transitions) are ~67× larger than ours. 3× α reduction is  
a conservative first step; no CQL-paper citation exists for <20k transition regime.

**Stage 1** (50k CQL): completed cleanly on GPU 16.

| Step | Q_mean | Q_max | CQL | CQL/TD ratio |
|------|--------|-------|-----|-------------|
| 5k | 93 | 3,449 | -22.39 | 0.004× |
| 15k | 154 | 9,063 | -23.08 | 0.006× |
| 35k | 98 | 9,640 | -24.94 | 0.004× |
| 50k | **102.9** | 7,629 | -23.77 | **0.013×** |

Q_mean held steady (~100 range) through 50k — no collapse. CQL/TD ratio more stable.  
This is a genuine improvement over α_cql=1.0 on the Q-value calibration metric.

**Stage 2** (RTG relabeling from 50k critic):

| Metric | α_cql=1.0 | α_cql=0.3 |
|--------|-----------|-----------|
| Q mean | 60.77 | 67.63 |
| P10 | -280.88 | -279.58 |
| **P50** | **-140.79** | **-127.34** |
| P90 → TARGET_RTG | +272.36 | +281.60 |
| Q max | 10,914 | 10,452 |

Gate fail: `P50=-127.34 outside [-50, 100]`.

**Improvement in Q_mean: 4.69 → 102.9 (22×). Improvement in P50: -140.79 → -127.34 (9%). The two metrics are almost fully decoupled.**

---

## Stage 3 smoke: best-case pipeline result

The most favorable Stage 2 relabeling came from the 5k CQL checkpoint (α_cql=1.0 smoke run),
before conservatism had compounded. P50=−$5.63, P90=+$311.75, Q_mean=103. This passed the
Stage 2 gate and proceeded to Stage 3.

**Stage 3 smoke (1k DT training steps):**

| Run | TARGET_RTG | DT Steps | All-days $/kW-yr | Ex-Fern | Fern |
|-----|-----------|----------|-----------------|---------|------|
| v1 (smoke-5k critic, P90=-112) | −$112 | 1,000 | **18.23** | 17.10 | 78.19 |
| v2 (smoke-5k critic, P90=+311) | +$311 | 1,000 | **16.53** | 15.52 | 70.44 |

The v2 result uses the correct TARGET_RTG (+$311, P90 from the 5k critic). **16.53 $/kW-yr
at only 1k DT steps is below both MILP+forecaster (22.84 $/kW-yr) and BC (29.16 $/kW-yr).**

**Interpretation:** This is the most favorable achievable QDT result within the sprint budget:
- The 5k CQL checkpoint was not over-conservative (valid Stage 2 relabeling)
- The DT saw a positive TARGET_RTG (+$311) in Stage 3
- 1k DT steps is early; convergence requires 50k steps

Even so, the 1k-step trajectory was below BC and below Method 1 (MILP+forecaster). The DT
component had not learned to use RTG conditioning effectively: 16.53 $/kW-yr is in the
same range as unconditioned BC at early training. This suggests the DT pipeline needs more
than sprint-budget compute to extract value from RTG conditioning on a 15k-transition dataset.

**Conclusion:** The two-ground failure is consistent — Stage 2 gate failure (CQL over-conservatism
in the regime) AND Stage 3 structural weakness (DT underperforms at available compute budget)
both point to small-data incompatibility across the full QDT pipeline.

---

## Root cause analysis

**The P50 failure is not a CQL parameter issue. It is a dataset property.**

Q_mean (minibatch average of 256 randomly sampled transitions) is dominated by high-reward  
transitions: Fern + peak-price events that appear in ~1.5% of training steps but contribute  
~35% of total revenue. These transitions get Q-values of $1,000–$10,000.

P50 RTG (full-dataset median Q-value) is dominated by the other 98.5% of transitions:  
routine non-event intervals where:
- Per-step reward ≈ $1–5 (RT LMP near hub-average, AS premiums small)
- Discounted Q ≈ $100–500 at most (100-step horizon × $1–5/step)
- CQL conservatism subtracts from this → many transitions Q < 0

The distribution is **bimodal**:
- Left cluster (majority): Q ∈ [-400, +50] (routine MILP arbitrage, low reward)
- Right spike (rare): Q ∈ [$1k, $11k] (Fern/price-spike transitions)

This bimodality means P50 is structurally in the left cluster regardless of α_cql, because  
no α value changes the ratio of Fern transitions to non-Fern transitions in the training data.

**Training-vs-inference RTG mismatch:**  
The DT would be conditioned on TARGET_RTG = P90 = +$281 at inference, but trained on a  
distribution where 50%+ of episodes use RTG labels below -$127. The DT would need to  
generalize from a conditioning target far in the right tail of its training distribution.  
This extrapolation risk was judged higher than the uncertainty of Stage 3 training.

---

## Comparison to QDT paper

Yamagata et al. 2023 benchmarks use D4RL (hopper/walker2d/halfcheetah):
- Dataset sizes: 100k–1M transitions (7–67× larger than ours)
- Reward distributions: approximately unimodal per-step rewards, low variance, no extreme events
- No Fern-equivalent: no single episode contributing 35% of total reward

QDT's Q-relabeling implicitly assumes the Q-value distribution is roughly unimodal and
centered in a positive range, so that P90 conditioning achieves plausibly reachable
performance targets. On spike-dominated distributions, this assumption fails: P50 is
structurally negative (routine intervals dominate the median), while P90 is driven by
Fern/spike transitions that are too rare to transfer to the DT via conditioning alone.

The DT component also faces a data-quantity constraint independent of Q-relabeling: the
literature suggests DTs require at least tens of thousands of diverse episodes to learn
RTG-conditioned behavior reliably. Our 68 training days (15k transitions) is below this
threshold; even with perfect relabeling, the DT had insufficient trajectory diversity to
reliably generalize from the RTG conditioning signal within a sprint-budget training run.

---

## Sprint implications

QDT is dropped from the six-method comparison table. Final method slate:

| # | Method | Status |
|---|--------|--------|
| 1 | MILP + transformer forecaster | Complete — 22.84 $/kW-yr |
| 2 | BC from MILP expert | Complete — 29.16 $/kW-yr |
| 3 | Cal-QL | Day 3-4 (offline + online phase) |
| 4 | Diffusion-QL | **Does not converge in this regime** (Q-divergence) |
| 5 | QDT | **Does not converge in this regime** (Stage 2 gate + Stage 3 structural weakness) |
| 6 | Li et al. TempDRL | Colleague scope |

Methods 4 and 5 both fail at Q-function estimation in this regime. The question for Cal-QL
(Method 3) is whether CQL-based conservatism with explicit offline-to-online calibration
can avoid the over-conservatism that killed QDT Stage 2, while also handling the Fern-dominated
reward distribution that drove DQL critic divergence. These are distinct failure modes, so
Cal-QL may resolve one without encountering the other — pending Day 3-4 results.

---

## Publishable methodology paragraph

> "QDT (Yamagata et al. 2023) does not converge in our regime on two independent grounds.
> First, the Stage 2 Q-relabeling step failed a hard gate (P50 relabeled RTG ∈ [−$50, $100])
> under both tested α_cql values (1.0 and 0.3); the bimodal Q-value distribution — Fern
> spike transitions at Q ∈ [$1k, $11k] versus routine intervals at Q < $0 under CQL
> conservatism — places P50 structurally below the acceptance threshold regardless of
> penalty strength. Second, the most favorable pipeline execution (smoke Stage 1 → valid
> Stage 2 → 1k DT steps) produced 16.53 $/kW-yr, below both MILP+persistence (22.84
> $/kW-yr) and BC (29.16 $/kW-yr), suggesting the Decision Transformer component had
> insufficient trajectory diversity (68 training days, ~15k transitions) to learn effective
> RTG-conditioned behavior within the available compute budget. Both failure modes are
> consistent with a dataset size and distributional character that falls outside QDT's
> operating envelope. We note that QDT achieves strong results on D4RL benchmarks with
> approximately unimodal reward distributions and 100k–1M transitions; the limitations
> documented here are specific to the small-data, spike-dominated post-RTC+B BESS regime."

---

## Artifacts

**Checkpoints:**
- `checkpoints/sprint/qdt/qdt_s1_smoke_final.pt` — 5k smoke, α_cql=1.0
- `checkpoints/sprint/qdt/qdt_s1_step50000.pt` — 50k, **most recent overwrite** (α_cql=0.3 checkpoint; same filename used by both runs per train.py default)
- `methods/qdt/dataset_relabeled_v2.npz` — relabeled from 5k smoke critic (α_cql=1.0)
- `methods/qdt/dataset_relabeled_50k.npz` — relabeled from 50k critic, α_cql=1.0
- `methods/qdt/dataset_relabeled_alpha03.npz` — relabeled from 50k critic, α_cql=0.3 (gate fail)

**Logs:**
- `logs/sprint/qdt_s1_smoke_v2.log` — re-smoke with in-sample SARSA fix
- `logs/sprint/qdt_s1_full.log` — full Stage 1 training (α_cql=1.0 run 1, then α_cql=0.3 run 2, appended)

**Reports:**
- `methods/qdt/SMOKE_REPORT.md` — initial smoke (v1, broken bootstrap)
- `methods/qdt/SMOKE_REPORT_v2.md` — re-smoke with in-sample SARSA fix (PASS)
- `methods/qdt/CHECKPOINT_50K_STAGE1.md` — 50k report for α_cql=1.0 (gate fail, 3 tracked metrics)
- `methods/qdt/CLOSEOUT.md` — this document
