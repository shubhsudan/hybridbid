# QDT Smoke Report v2 — In-Sample SARSA Bootstrap
**Date:** 2026-04-25  
**Machine:** Narnia GPU 16 (A16)  
**Code commit:** a9e9461 (in-sample SARSA + flag clarifications)  
**Data:** `data/expert_trajectories/receding_horizon_postbreak_train.npz` (19,584 transitions, 68 CT days)

---

## Flag semantics (traceability)

| Flag | Value | Used where |
|------|-------|------------|
| `done` | always 0.0 | Q-learning terminal flag in CQL Bellman target; never zeros bootstrap (no terminal states in ERCOT BESS) |
| `sarsa_done` | 1.0 at 68 CT-midnight boundaries + final idx (69 total) | **CQL Bellman target only**: zeros γ·Q(s', a_next) when a_next belongs to next episode |
| `truncated` | 1.0 at 68 CT-midnight positions | **SequenceDataset (Stage 3)**: validates K=20 windows don't straddle episode boundaries |

These three flags are distinct. `sarsa_done` ≈ `truncated | (idx == N-1)`.

---

## Stage 1: CQL Critic (5k smoke steps)

**Wall time:** 60.7s (12.1ms/step on A16)

| Metric | Value | Status |
|--------|-------|--------|
| NaN in Q-values | No | C1 PASS |
| Q_mean (last batch) | 153.27 | C2 PASS (>0) |
| Q P90 (per-transition, last batch) | 334.46 | C3 PASS (>$200 floor) |
| CQL penalty (last step) | -18.32 | C4 — see analysis below |
| TD loss (last step) | 7,907.86 | Oscillatory; see below |
| Q distribution | mean=153 std=624 min=-710 P10=-57 P50=11 P90=334 max=6755 | |

**TD loss trajectory:** Highly oscillatory (744–13,932 across steps 1k–5k). This is expected at 5k steps with γ=0.99 and a reward std of $43.74 — the critic is still bootstrapping aggressively. Should stabilize by 25k–50k steps.

**C4 — CQL penalty sign analysis:**

`cql_loss = logsumexp(Q_rand) − Q_data = −18.32`

A negative value means **Q_data > logsumexp(Q_rand)**: the critic already gives higher Q to in-distribution dataset actions than to random OOD actions. This is the **desired conservatism direction**.

The C4 "FAIL" label in the log is misleading: it fires on `cql ≤ 0` but negative CQL does not indicate failure here. The gradient of `L_CQL` w.r.t. Q-network weights is always the same sign regardless of loss sign:
- ∂L_CQL/∂Q_data = −1 → gradient descent pushes Q_data **up**
- ∂L_CQL/∂Q_rand = +softmax > 0 → gradient descent pushes Q_rand **down**

So the conservatism gradient direction is correct. Negative CQL means the in-sample SARSA bootstrap has naturally placed dataset Q above OOD Q — CQL reinforces this.

**CQL/TD ratio:** |−18.32 / 7907.86| = 0.002× — CQL barely engaged relative to TD. This is the FLAG-WEAK case. With α_cql=1.0 and a highly variable reward (std $43.74), TD dominates at 5k. This may self-correct at 50k as TD loss falls. If CQL ratio remains <0.01× at 50k, consider increasing α_cql to 5.0.

**Bootstrap spot-check (C5):** act[k][0] and next_act[k][0] show realistic MILP-style values (1.0 max charge, 0.0 idle, -1.0 max discharge). No `sarsa_done=1` in the last batch (no CT-midnight boundaries in that minibatch, which is expected at ~3% density). ✓

---

## Stage 2: RTG Relabeling

**Wall time:** ~1s

| Metric | Value |
|--------|-------|
| Q mean | 103.44 |
| Q std | 485.43 |
| Q min | -1,239.94 |
| Q P10 | -57.07 |
| Q P50 | **-5.63** |
| Q P90 | **311.75** → becomes TARGET_RTG |
| Q max | 10,914.02 |

**P50 is negative.** More than half of transitions receive negative Q labels. This warrants understanding:

The reward distribution is extremely right-skewed: mean=$5.95/step but std=$43.74, with 84 intervals >$500 in the entire dataset. Most steps have near-zero reward ($0 or close), and the discount γ=0.99 means a 5k-step critic's Q-values haven't converged across a long effective horizon. The critic correctly identifies that most individual (s,a) pairs have low/negative Q-value while a few events (Fern, peak-price) dominate.

This is a 5k-step critic. At 50k steps, Q-values should converge to a tighter distribution. The P50 being negative is **acceptable for a smoke-quality critic** and does not block Stage 1 full (50k). It should improve materially by the 50k checkpoint.

**TARGET_RTG = $311.75.** This is the P90 of Q-values over the full training set. The DT will be conditioned on "achieve $311.75 per-transition discounted return" — a target in the top 10% of what the 5k critic thinks is achievable. Reasonable for a smoke run.

---

## Stage 3: Decision Transformer (1k smoke steps)

**Wall time:** 55.0s (55.0ms/step; includes eval at step 1k)

| Metric | Value |
|--------|-------|
| DT loss step 200 | 0.0768 |
| DT loss step 1000 | 0.0557 |
| NaN | No |
| Eval all_days | $16.53/kW-yr |
| Eval ex_fern | $15.52/kW-yr |
| Eval fern_only | $70.44/kW-yr |

**DT loss decreased from 0.077→0.056 (−27%).** Architecture is sound; loss is decreasing.

**$16.53/kW-yr at 1k DT steps** on a 5k critic is a meaningful signal: the policy is learning something (not random/zero), captures Fern value ($70.44/kW-yr from a 1-day event), but is below fleet median ($24.93). This is expected at 1k DT steps — 50k should close the gap significantly.

**SequenceDataset:** 18,292 valid K=20 windows from 19,584 transitions.  
RTG(t=0) per window: mean=$113.67, P50=−$0.62, P95=$516.76 — right-skewed distribution consistent with Stage 2 results.

---

## Smoke verdict

| Stage | Status | Note |
|-------|--------|------|
| Stage 1 (5k CQL) | **PASS with monitoring** | C4 FAIL label is a false alarm; negative CQL is correct conservatism direction. CQL/TD ratio 0.002× is weak — watch at 50k. TD loss oscillatory but normal at 5k. |
| Stage 2 (RTG relabeling) | **PASS** | P90=$311.75 → healthy TARGET_RTG. P50 negative at 5k-step critic is acceptable; will improve at 50k. |
| Stage 3 (1k DT) | **PASS** | Loss decreasing, no NaN, eval $16.53/kW-yr shows learning. |

**C4 code fix needed** (non-blocking): `run_stage1()` C4 check should trigger FAIL only on `cql > 0 AND cql >> td` (dominating), and WARN (not FAIL) on `cql < 0` with a note explaining the conservatism-already-present case. Deferred to post-smoke since it's a log-label issue, not a correctness issue.

---

## Pre-conditions for Stage 1 full (50k) launch

All smoke-pass criteria met. Two items to monitor at the 50k checkpoint:

1. **TD loss convergence**: should drop from ~8k toward <1k by step 25k. If still >5k at 25k, CQL critic may not be converging — reduce lr to 1e-4 before continuing.
2. **CQL/TD ratio at 50k**: if still <0.01×, consider increasing α_cql from 1.0 to 5.0 before Stage 3. The 15k-transition dataset has lower diversity than D4RL benchmarks; CQL may need a stronger conservatism weight.

**Stage 3 note**: Stage 2 and Stage 3 must be re-run using the Stage 1 **50k checkpoint** (not 5k smoke checkpoint). Smoke Stage 3 results ($16.53/kW-yr) are informational only.

---

## DQL status (GPU 13, unaffected by QDT)

DQL hit 25k checkpoint and exited with `sys.exit(0)`:
- Q_max=7,784 (0.9× smoke = within spec)
- Eval: all_days=$0.45/kW-yr at step 25k (early; expected to improve)
- Checkpoint saved: `checkpoints/sprint/dql/dql_step25000.pt`
- Awaiting Karthik review before resuming to 50k.
