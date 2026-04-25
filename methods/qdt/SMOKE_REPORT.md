# QDT Smoke Report
**Date:** 2026-04-25  
**Machine:** Narnia GPU 16 (A16)  
**Smoke steps:** Stage 1 = 5k, Stage 3 = 1k  
**Smoke verdict:** BLOCKED — TD target bug requires fix before full training

---

## Stage 1: CQL Critic (5k steps)

| Metric | Value |
|--------|-------|
| Wall time | 67.7s (13.5ms/step) |
| Final TD loss | oscillatory (range 1,010–12,825) |
| Final CQL penalty | negative throughout |
| Q_mean at 5k | -234 |
| Q_max at 5k | 6,352 |
| NaN in Q-values | No |

**Issue:** CQL penalty was negative throughout, meaning dataset actions already receive lower Q
than random actions — the opposite of the intended conservatism direction. This is a side
effect of the TD target bug below (random actions → pessimistic targets → all Q-values
depressed → CQL pushes in the wrong direction).

---

## Stage 2: RTG Relabeling

| Metric | Value |
|--------|-------|
| Q mean | -277 |
| Q std | (not captured) |
| Q P90 | **-112.49** ← CRITICAL |
| Wall time | ~1s |

**CRITICAL FLAG:** P90 of Q-values is **negative (-112.49)**. This becomes `TARGET_RTG` at
inference, meaning the DT will be conditioned on "achieve a return of -$112" every step.
A policy conditioned on negative returns will learn to produce suboptimal (or adversarial)
actions. Stage 3 DT training proceeded on top of these broken RTG labels.

---

## Stage 3: Decision Transformer (1k steps)

| Metric | Value |
|--------|-------|
| Initial DT loss | 0.077 |
| Final DT loss | 0.049 |
| NaN | No |
| Wall time | ~30s for 1k steps |

Loss decreasing cleanly — DT architecture is sound. However, the RTG sequences it trained on
are contaminated by Stage 1's broken Q-values, so the Stage 3 smoke weights are not usable.

---

## Root Cause: TD Target Bug in Stage 1

**Location:** `methods/qdt/train.py`, `run_stage1()`, TD target block  
**Bug:** Next-state bootstrap used random uniform actions in the valid p.u. range:
```python
# BUGGY (was):
a_next = torch.cat([
    torch.rand(act.shape[0], 1, device=device) * 2 - 1,  # p_energy [-1,1]
    torch.rand(act.shape[0], 5, device=device),            # c_as [0,1]
], dim=1)
```
Random actions are OOD from the ERCOT post-break data distribution. The CQL critic assigns
pessimistic (low) Q-values to OOD actions. Using random actions for the TD bootstrap means
the target `r + γ * Q(s', a_rand)` is systematically too low, pushing all Q-values negative.

**Fix (committed):** Replace random actions with shuffled current-batch actions:
```python
# FIXED:
idx_shuffle = torch.randperm(act.shape[0], device=device)
a_next = act[idx_shuffle].detach()
```
Shuffled batch actions break temporal correlation while staying in-distribution, giving
well-calibrated TD targets. This is consistent with how CQL is implemented in standard
offline RL libraries (d3rlpy, CORL).

---

## Re-smoke Required

Before launching Stage 1 full (50k steps), a clean 5k re-smoke on Narnia GPU 16 must confirm:

1. Q_mean stays positive (expected >$100 given physical-$ rewards with mean ~$5/step × 288 steps/day)
2. P90 of RTG labels after Stage 2 is positive
3. CQL penalty is positive (random Q < dataset Q, confirming conservatism direction)
4. TD loss converges within 5k (expect 2–4 decades of decrease)

Estimated re-smoke time: ~70s for Stage 1 + ~1s Stage 2 + ~30s Stage 3 = <2 min total.

---

## Action Items (pending Karthik review)

- [x] Fix committed to `methods/qdt/train.py` on M4
- [ ] Push fix to Narnia (git pull on Narnia)
- [ ] Re-run 3-stage smoke on Narnia GPU 16 with fixed code
- [ ] Report re-smoke Q distribution (confirm positive P90)
- [ ] Karthik review of re-smoke before Stage 1 full (50k) launch
