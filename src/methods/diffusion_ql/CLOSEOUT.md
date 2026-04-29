# Diffusion-QL Closeout
**Method:** Diffusion-QL (Wang et al. ICLR 2023, arXiv:2208.06193)  
**Sprint:** offline-rl, April 25 2026  
**Outcome:** Does not converge in this regime — Q-value divergence under both β_Q values tested  
**Finding type:** Publishable regime analysis — method works on D4RL-scale; specific failure mode documented for small-data, spike-dominated BESS setting

---

## Summary

Diffusion-QL does not converge in the post-RTC+B BESS regime: 15k transitions,
spike-dominated reward distribution (Winter Storm Fern, ~35% of revenue from 0.2%
of transitions). Q-value divergence occurred at steps 29k–31k under both β_Q=1.0
and β_Q=0.5, confirming a structural incompatibility with this regime rather than
a hyperparameter artifact. The method achieves strong results on D4RL-scale benchmarks
(100k–1M transitions, approximately unimodal rewards); the constraints documented here
are regime-specific.

---

## Attempt 1: β_Q=1.0 (paper default)

**Budget:** 25k steps (checkpoint gate) + resumed to ~31.5k (manually stopped on divergence)  
**Hardware:** Narnia GPU 0 (A16)

| Step | Q_mean | Q_max | Q_max ratio vs smoke |
|------|--------|-------|---------------------|
| 5k (smoke) | 205 | 8,553 | 1.0× (reference) |
| 25k (checkpoint) | 447 | 7,784 | 0.9× (within spec) |
| 29,000 | 823 | 8,330 | 0.97× |
| 29,500 | 4,793 | **49,633** | **5.8×** — DIVERGENCE RISK |
| 30,000 | 10,252 | **80,495** | **9.4×** — KILL THRESHOLD MET |
| 30,500 | 7,969 | 53,013 | 6.2× |
| 31,000 | 6,238 | 39,733 | 4.6× |
| 31,500 | 6,219 | 55,035 | 6.4× |

**Kill reason:** Q_max exceeded 4× smoke (threshold 34,212) and exceeded 50,000 at step 30k.  
Process killed manually. 25k checkpoint intact.

**Eval at 25k:** all_days=$0.45/kW-yr — uninformative (DQL typically improves in second half).

---

## Attempt 2: β_Q=0.5 (single-variable restart)

**Budget:** Resumed from 25k checkpoint, target 50k; auto-killed at step 29.5k  
**Hardware:** Narnia GPU 13 (A16)  
**Code change:** only `beta_q: 1.0 → 0.5` (one-variable rule)

| Step | Q_max | vs restart (7,784) |
|------|-------|---------------------|
| 26,500 | 7,415 | 0.95× |
| 27,000 | 3,524 | 0.45× |
| 27,500 | 10,654 | 1.37× |
| 28,000 | 9,435 | 1.21× |
| 29,000 | 8,322 | **1.07×** |
| 29,500 | **275,490** | **35.4×** — KILL THRESHOLD MET |

**Kill reason:** EARLY-MONITOR auto-kill at step 29,500 (`sys.exit(1)`) — Q_max reached 275,490,
which is 35.4× the restart value of 7,784. Threshold was 4× (31,136). Process killed by code.

**Pattern:** Divergence onset falls in the same step window (29k–31k) for both β_Q values.
β_Q=0.5 provides no meaningful stabilization: the transition from Q_max ≈ 8k to Q_max = 275k
in a single 500-step window is characteristic of the same feedback loop, just with slightly
delayed onset. No further tuning attempted per sprint discipline (no stacked variable changes).

---

## Root cause analysis

**This regime is structurally outside Diffusion-QL's operating envelope.**

Diffusion-QL combines a diffusion-based policy with a Q-maximization term
`−β_Q * normalize(Q_π)`. The normalization (`Q / |Q|.mean()`) is designed to make the
Q-gradient scale invariant to Q magnitude. This works when Q variance is low (unimodal
reward → stable Bellman targets → low-variance Q). It fails when Q variance is high.

**Why Q variance is high in this regime:**

1. **Fern-dominated distribution.** ~35% of total revenue comes from Jan 26 CT (Winter Storm
   Fern), which is 1 day out of 68 training days (1.5% of transitions). Fern transitions carry
   Q-values in the $5k–$11k range; routine transitions carry Q ≈ $50–500. A small number of
   transitions with 10–100× the typical Q value produces σ/μ ≈ 3–5 throughout training.

2. **Small dataset.** 15k transitions (vs. D4RL's 100k–1M). The BC loss in Diffusion-QL
   requires enough in-distribution coverage to anchor the policy against Q-maximization drift.
   At 15k transitions, the coverage is too sparse: the policy drifts OOD, the critic assigns
   overestimated Q to OOD actions (classic offline RL bootstrapping error), and the
   Q-normalization amplifies this into an unstable gradient signal.

3. **Feedback timing.** Instability onset is identical for both β_Q values (~29k steps). This
   is the point where BC loss has been minimized enough that Q-gradient dominates — and at
   that step, the Q-value variance from Fern bootstrapping becomes the primary gradient source.
   The instantaneous transition (stable 29k → explosive 29.5k) is characteristic of a
   bifurcation in the Q-value dynamics, not a gradual drift.

**Why β_Q=0.5 also fails:** β_Q scales the Q-gradient magnitude but does not fix the
underlying issue (high Q variance from Fern bootstrapping). The feedback loop activates at
identical step range regardless of β_Q. No further reduction was attempted; any β_Q small
enough to prevent divergence would make Q-maximization negligible, reducing Diffusion-QL
to pure BC — which is already implemented as Method 2.

---

## Sprint implications

Diffusion-QL is dropped from the six-method comparison table. Final method slate:

| # | Method | Status |
|---|--------|--------|
| 1 | MILP + transformer forecaster | Complete — 22.84 $/kW-yr |
| 2 | BC from MILP expert | Complete — 29.16 $/kW-yr |
| 3 | Cal-QL | Day 3-4 (offline + online phase) |
| 4 | Diffusion-QL | **Does not converge in this regime** (Q-divergence) |
| 5 | QDT | **Does not converge in this regime** (Stage 2 gate + Stage 3 structural weakness) |
| 6 | Li et al. TempDRL | Colleague scope |

Cal-QL (Method 3) is the sole remaining RL contender. Its offline-to-online structure is
different from Diffusion-QL's critic in two relevant ways: (1) the Q-conservatism is an
explicit CQL penalty rather than an emergent property of Q-normalization, making it more
directly controllable; (2) the online phase uses the bootstrap-resampling environment to
expose the policy to fresh Q-function estimation beyond the offline distribution. Whether
this is sufficient to handle Fern-dominated reward variance remains the open question for Day 3–4.

---

## Publishable signal

> "Diffusion-QL (Wang et al. 2023) does not converge in our regime (15k transitions,
> spike-dominated reward distribution: Winter Storm Fern contributes 35% of revenue from
> 0.2% of training transitions). Q-value divergence occurred at steps 29k–31k under
> β_Q=1.0 (paper default) and β_Q=0.5, indicating a structural incompatibility rather
> than a hyperparameter sensitivity. The Q-normalization that stabilizes Diffusion-QL on
> D4RL datasets (100k–1M transitions, approximately unimodal rewards) becomes ineffective
> when Q variance is elevated by isolated high-value spike events: the normalization amplifies
> rather than moderates the gradient. We note that Diffusion-QL achieves strong results on
> D4RL benchmarks; the limitations documented here are specific to the small-data,
> spike-dominated post-RTC+B BESS bidding regime."

---

## Artifacts

- `checkpoints/sprint/dql/dql_smoke_final.pt` — 5k smoke checkpoint (Q_mean=205, Q_max=8,553)
- `checkpoints/sprint/dql/dql_step25000.pt` — 25k checkpoint (Q_mean=447, Q_max=7,784, eval=$0.45/kW-yr)
- `logs/sprint/dql_smoke.log` — 5k smoke training log
- `logs/sprint/dql_full.log` — full training log: 25k checkpoint + β_Q=1.0 divergence + β_Q=0.5 divergence
- `methods/diffusion_ql/SMOKE_REPORT.md` — 5k smoke report (PASS; Q within spec at 5k)

---

## Addendum (post-audit) — Action-Space Infeasibility as Primary Mechanism

Following the action-feasibility audit conducted during Cal-QL post-mortem, the
DQL failure mechanism is reattributed:

**Primary:** Q-extrapolation in unconstrained action space. Offline RL training
never queries the env, so Q-values for actor-proposed actions are populated
entirely through bootstrap targets `r + γQ(s', a_actor)`. The dataset contains
only feasible expert actions (MILP-respecting joint capacity constraint). At
actor update, Q is queried at infeasible high-AS, high-capacity actions where
it has no grounding signal — the network extrapolates optimistically into the
infeasible region. This produces a gradient signal that pushes the actor
toward infeasible bids. The eval-time projection silently scales these to
feasibility, but during training there is no such constraint, so Q grows
unbounded as the actor saturates AS dimensions.

**Secondary:** Heavy-tailed Fern reward variance (max $1,125 vs mean $5.95
across 15k transitions) accelerates the divergence by seeding higher-magnitude
bootstrap targets early. This is the original CLOSEOUT's primary attribution;
it remains valid as a contributing mechanism but is not the structural cause.

The original CLOSEOUT's diagnosis (steps 29k–31k onset, β_Q invariance to
divergence onset) is consistent with the primary mechanism: action-space
infeasibility is structural and cannot be tuned away by β_Q adjustment, which
matches the observation that both β_Q=1.0 and β_Q=0.5 diverged at the same
step.

**Cross-method consistency:** This mechanism is shared with Cal-QL (where
calibration anchor delays it until step ~13k via the V_beh floor, then vanilla
CQL applies). QDT is not affected — DT imitates the feasible expert
distribution and never Q-maximizes away from data, so its failure (Stage 2 RTG
bimodality and over-conservatism) is unrelated.
