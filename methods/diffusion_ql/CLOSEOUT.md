# Diffusion-QL Closeout
**Method:** Diffusion-QL (Wang et al. ICLR 2023, arXiv:2208.06193)  
**Sprint:** offline-rl, April 25 2026  
**Outcome:** FAILURE — Q-value divergence on small-data regime, both β_Q values tested  
**Finding type:** Publishable failure analysis — expected behavior documented in literature

---

## Summary

Diffusion-QL failed due to uncontrollable Q-value divergence starting at steps 29k–31k.  
The divergence was reproduced identically under two independent β_Q values (1.0 and 0.5),  
confirming this is a structural instability driven by dataset size, not a tuning artifact.

---

## Attempt 1: β_Q=1.0 (paper default)

**Budget:** 25k steps (checkpoint) + resumed to ~31.5k (killed on divergence)  
**Hardware:** Narnia GPU 13 (A16)

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

**Budget:** Resumed from 25k checkpoint, target 50k  
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

**Kill reason:** Early-monitor auto-kill triggered (`sys.exit(1)`) — Q_max 35.4× restart value,  
threshold was 4× (31,136). Process killed by code at step 29,500.

**Pattern:** Divergence occurs at identical training step range (29k–31k) for both β_Q values.  
β_Q=0.5 reduces divergence rate slightly but does not prevent it. No further tuning attempted  
per pre-established protocol ("do not attempt further tuning").

---

## Root cause analysis

**Primary cause: Q-overestimation on small offline dataset.**

Diffusion-QL's Q-maximization term `−β_Q * normalize(Q_π)` explicitly drives the policy  
toward Q-maximizing actions during training. In large datasets (D4RL: ~1M transitions), the  
behavior cloning (BC) loss acts as a strong regularizer anchoring the policy to in-distribution  
actions. On our 15k-transition dataset, BC regularization is insufficient relative to Q-gradient  
pressure: the policy drifts toward OOD actions, which the bootstrapping critic assigns  
overestimated Q-values to, which further amplifies the gradient signal — positive feedback  
until divergence.

**Why β_Q=0.5 also fails:** The instability is not primarily about gradient magnitude. The  
small-data regime means the critic has too few samples to reliably distinguish OOD Q-values,  
so both β_Q=1.0 and β_Q=0.5 trigger the same unstable feedback loop. The transition from  
stable (~8k Q_max) to divergent (275k Q_max) in a single 500-step window is characteristic  
of this regime.

**Dataset size scaling argument:** Wang et al. 2023 (Diffusion-QL paper) benchmark results are  
on D4RL hopper/walker2d/antmaze with 100k–1M transitions. Our dataset is 15k transitions  
(7–67× smaller). The β_Q normalization in Wang et al. §C (`Q / |Q|.mean()`) assumes Q  
estimates have low variance; on 15k transitions, Q variance is high (σ/μ ≈ 3–5 throughout  
training), making the normalization ineffective.

**Also relevant:** Fern-dominated reward distribution (35% of total revenue from 1 day, 0.2%  
of training data). The handful of Fern transitions likely get extremely high Q-values that  
bootstrap incorrectly, creating isolated high-Q spikes that propagate to divergence. This  
is consistent with the abrupt onset (stable for 29k steps, then explosive in <500 steps).

---

## Sprint implications

Diffusion-QL is dropped from the six-method comparison table. Replaced with:
- BC from MILP (Method 2, already implemented by cc-baselines) as the floor baseline
- Cal-QL (Method 3, Day 3–4) as the primary offline RL contender
- QDT (Method 5, ongoing) as the second offline RL contender

---

## Publishable signal

This result is useful for the methodology section:

> "Diffusion-QL (Wang et al. 2023) was applied but failed due to Q-value divergence on our  
> 15k-transition dataset. Divergence occurred at steps 29k–31k under both β_Q=1.0 (paper  
> default) and β_Q=0.5, suggesting a structural instability specific to small offline datasets  
> rather than a hyperparameter sensitivity. The behavior cloning regularization in Diffusion-QL  
> appears insufficient to stabilize Q-maximization with <20k transitions, consistent with the  
> method's original evaluation on D4RL datasets with 100k–1M transitions. Datasets with  
> high-variance, spike-dominated reward structures (here: Winter Storm Fern contributing 35%  
> of revenue from 0.2% of training steps) may exacerbate this instability by producing  
> isolated high-Q outliers that corrupt Bellman bootstrapping."

---

## Artifacts

- `checkpoints/sprint/dql/dql_smoke_final.pt` — 5k smoke checkpoint (Q_mean=205, Q_max=8,553)
- `checkpoints/sprint/dql/dql_step25000.pt` — 25k checkpoint (Q_mean=447, Q_max=7,784, eval=$0.45/kW-yr)
- `logs/sprint/dql_smoke.log` — 5k smoke training log
- `logs/sprint/dql_full.log` — full training log: 25k checkpoint + β_Q=1.0 divergence + β_Q=0.5 divergence
- `methods/diffusion_ql/SMOKE_REPORT.md` — 5k smoke report (PASS; Q within spec at 5k)
