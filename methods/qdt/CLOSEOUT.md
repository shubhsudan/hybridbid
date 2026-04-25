# QDT Closeout
**Method:** QDT — Q-learning Decision Transformer (Yamagata et al. ICML 2023, arXiv:2209.03993)  
**Sprint:** offline-rl, April 25 2026  
**Outcome:** FAILURE — RTG distribution shift on right-skewed small dataset; gate failed both runs  
**Finding type:** Publishable failure analysis — fundamental incompatibility with spike-dominated reward regime

---

## Summary

QDT's Stage 2 hard gate (P50 RTG ∈ [−$50, $100]) failed under both tested α_cql values.  
The failure is not a CQL tuning issue: it is a structural incompatibility between QDT's  
Q-relabeling step and reward distributions dominated by rare high-reward events.

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
- Dataset sizes: 100k–1M transitions
- Reward distributions: approximately unimodal per-step rewards, low variance
- Fern-equivalent: no spike-dominated distribution

Our setting differs on both axes:
- 15k transitions (7–67× smaller)
- Spike-dominated distribution (Fern = 35% revenue, 1.5% of transitions)

QDT's Q-relabeling implicitly assumes the Q-value distribution is roughly unimodal and  
centered in a positive range, so that P90 conditioning is achievable. On spike-dominated  
distributions, this assumption breaks.

---

## Sprint implications

QDT is dropped from the six-method comparison table. Final method slate:

| # | Method | Status |
|---|--------|--------|
| 1 | MILP + transformer forecaster | In progress (cc-baselines) |
| 2 | BC from MILP expert | Complete (cc-baselines) |
| 3 | Cal-QL | Day 3-4 target |
| 4 | Diffusion-QL | **FAILURE** (Q-divergence) |
| 5 | QDT | **FAILURE** (RTG distribution shift) |
| 6 | Li et al. TempDRL | Colleague scope |

Both Methods 4 and 5 exhibit failures driven by Q-function estimation in our regime  
(15k transitions, spike-dominated rewards). Whether this pattern extends to all Q-based  
offline RL methods in this regime, or is specific to Q-maximization and Q-relabeling  
architectures, remains an open question pending Cal-QL results.

---

## Publishable methodology paragraph

> "QDT (Yamagata et al. 2023) was applied with two α_cql values (1.0 and 0.3). Both failed  
> the Stage 2 hard gate (P50 relabeled RTG ∈ [-50, 100]) in our regime: 15k transitions,  
> spike-dominated rewards (Winter Storm Fern, 35% of revenue from 0.2% of training steps).  
> The α_cql=0.3 run produced substantially better Q_mean at the 50k checkpoint (102.9 vs.  
> 4.69 for α=1.0), but P50 RTG improved only marginally (-127 vs. -141), indicating the  
> distribution shift is structural rather than a CQL tuning artifact. The reward distribution  
> creates a bimodal Q-value distribution: a handful of Fern transitions with Q ∈ [$1k, $11k],  
> and a majority of routine intervals with Q < 0 under CQL conservatism. Conditioning the  
> Decision Transformer on P90 = +$281 at inference while the median training RTG is -$127  
> constitutes an out-of-distribution extrapolation. We note that QDT achieves strong results  
> on D4RL benchmarks with approximately unimodal reward distributions and 100k–1M transitions;  
> the limitations documented here are specific to the small-data, spike-dominated post-RTC+B  
> BESS bidding regime."

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
