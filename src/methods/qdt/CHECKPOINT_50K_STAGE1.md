# QDT Stage 1 — 50k Checkpoint Report
**Date:** 2026-04-25  
**Machine:** Narnia GPU 16 (A16)  
**Wall time:** ~10.4 min (08:08:48 → 08:19:09, 12.5ms/step)  
**Checkpoint:** `checkpoints/sprint/qdt/qdt_s1_step50000.pt`

---

## Three tracked metrics (from 50k launch brief)

### 1. CQL/TD ratio at 50k checkpoint

| Step | CQL | TD | Ratio |
|------|-----|-----|-------|
| 5k (smoke) | -18.32 | 7,908 | 0.002× |
| 5k (re-smoke) | -18.32 | 7,908 | 0.002× |
| 50k | **-73.47** | **1,933** | **0.038×** |

**Result: IMPROVED.** 0.038× is in the OK range [0.01×, 10×]. CQL penalty grew from -18 to -73 (4×), while TD loss fell from 7,908 to 1,933 (4×). Both trends are healthy: Bellman fitting is converging, conservatism is strengthening. `α_cql=1.0` appears sufficient — no tuning needed to ~5.0.

### 2. P50 of relabeled RTG (Stage 2, from 50k critic)

| Step | mean | P50 | P90 → TARGET_RTG |
|------|------|-----|-----------------|
| 5k (smoke) | 103 | -5.63 | +311.75 |
| 50k | **60.77** | **-140.79** | **+272.36** |

**Result: WRONG DIRECTION. FLAG.**

P50 moved from -$5.63 → -$140.79 over 45k additional training steps. Direction: more negative, not less. More than half of the 19,584 training transitions receive negative Q-value labels from the 50k critic.

**Root cause**: CQL conservatism is compounding. Q_mean trajectory:

| Step | Q_mean | Q_max |
|------|--------|-------|
| 5k | 85 | 3,482 |
| 15k | 151 | 7,877 |
| 25k | 102 | 4,194 |
| 35k | 107 | 7,294 |
| 45k | 24 | 3,389 |
| 50k | **4.69** | 4,680 |

After peaking around 15k–35k, Q_mean declines to near-zero by 50k. The growing CQL penalty (-16 → -73) progressively pushes all Q-values down. At 50k the critic has learned to be conservative to the point of assigning negative value to the median (s,a) pair in the expert MILP dataset.

**Training-vs-inference RTG distribution shift:**
- Median training RTG label: -$140.79
- Inference TARGET_RTG (P90): +$272.36
- Gap: $413

The DT will be conditioned on $272 at inference, but the median K=20 window it trains on has RTGs far below zero. This is a known challenge for QDT/DT on small, right-skewed offline datasets.

### 3. C4 CQL penalty check (log correctness)

Fixed in commit `012d63e`: check now triggers WARN on `cql > 0` (conservatism not established) rather than FAIL on `cql ≤ 0`. At 50k, log shows:
```
[SMOKE Stage1 C4] CQL=-73.47  TD=1932.57  ratio=0.038×  direction=correct  engagement=OK
```
Correct behavior.

---

## Full Stage 1 snapshot at 50k

| Metric | Value |
|--------|-------|
| TD loss | 1,932 |
| CQL loss | -73.47 |
| Total loss | 1,859 |
| Q_mean (last batch) | 4.69 |
| Q_max (last batch) | 4,680 |
| CQL direction | correct (Q_data > Q_rand) |
| CQL/TD ratio | 0.038× (OK) |
| NaN | No |
| Checkpoint saved | ✓ |

---

## Decision gate: proceed to Stage 3 or re-tune?

**Option A — Proceed to Stage 3 as-is.**  
TARGET_RTG=$272 (P90) is above the $200 floor and positive. DT may still learn useful conditioning even with negative-median RTG labels — D4RL medium-expert benchmarks also have wide RTG distributions. The policy at 1k DT steps was $16.53/kW-yr on smoke; 50k DT steps may recover the gap. Risk: DT conditions on an out-of-distribution inference target and generalizes poorly.

**Option B — Re-run Stage 1 with α_cql=0.1.**  
Reduce conservatism to get Q_mean positive and P50 positive by 50k. Would require ~10 more minutes for a new Stage 1 run. One-variable rule: only α_cql changes. Risk: reduced conservatism might produce OOD-optimistic Q-values that inflate TARGET_RTG unrealistically.

**Option C — Replace Q-relabeling with MC returns.**  
Use per-episode discounted return instead of Q-values for RTG labels. MC returns are always non-negative for positive-reward trajectories. No critic needed. Much simpler, more interpretable. Fundamentally changes Stage 2 (not a one-variable change). Deferred to post-sprint if QDT fails.

**Recommendation (for Karthik's review):** Attempt Option A first (proceed to Stage 3 with current labels). The P50 issue is real but may not be fatal — the DT sees the full RTG distribution and can learn to distinguish high-RTG from low-RTG states. If Stage 3 50k eval is below $10/kW-yr, then consider Option B or C in a documented re-run.

**Awaiting your decision before launching Stage 3 (50k DT training).**

---

## DQL status (GPU 13, β_Q=0.5 restart)

Running. Early-monitor readings at steps 26.5k–28k:
- Q_max: 7,414, 3,524, 10,654, 9,435 (0.45×–1.37× restart value)
- All well below 31,136 kill threshold
- critic_loss: 1,672–5,838 (normal; compare to 101,931–287,717 during β_Q=1.0 divergence)
- β_Q=0.5 appears to have stabilized Q-values so far
