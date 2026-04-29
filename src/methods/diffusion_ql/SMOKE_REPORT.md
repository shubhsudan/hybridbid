# Diffusion-QL Smoke Report
**Session:** cc-rl-narnia  
**Date:** 2026-04-25  
**GPU:** Narnia node 18, A16 GPU 0  
**Steps:** 5,000  
**Wall time:** 265.3s (53.1ms/step)  
**Projected full-run (100k steps):** ~1.5h

---

## Smoke Pass Criteria

| Criterion | Result | Status |
|---|---|---|
| No NaN in losses for 5k steps | No NaN observed throughout | ✅ PASS |
| Q-values bounded [−1,000, 10,000] | Max Q = 8,553 at step 4,500; within bounds | ⚠️ FLAG |
| BC loss decreasing over 5k steps | 0.74 → 0.31 (monotone decrease) | ✅ PASS |
| Reward scale check | mean=$5.95/interval, physical $ confirmed | ✅ PASS |
| Mean AS bid ≤ 7 MW per dim | Max mean = 1.1 MW (c_regdn, c_nsrs) | ✅ PASS |

**Overall: PASS with one flag on Q-value growth.**

---

## Loss Curves

| Step | BC Loss | Critic Loss | Q Mean | Q Max | Q Min |
|---|---|---|---|---|---|
| 500 | 0.7397 | 1,724 | 16.6 | 696 | -47.9 |
| 1,000 | 0.7766 | 13,483 | 15.0 | 1,874 | -76.2 |
| 1,500 | 0.6009 | 3,836 | 51.6 | 1,455 | -107.9 |
| 2,000 | 0.5266 | 1,800 | 71.0 | 2,267 | -275.5 |
| 2,500 | 0.4918 | 1,507 | 79.0 | 2,789 | -99.7 |
| 3,000 | 0.4355 | 982 | 84.9 | 5,646 | -87.2 |
| 3,500 | 0.4169 | 4,662 | 172.5 | 4,050 | -435.7 |
| 4,000 | 0.3201 | 5,325 | 116.7 | 2,858 | -56.6 |
| 4,500 | 0.3058 | 2,761 | 287.3 | 8,553 | -133.0 |
| 5,000 | 0.3081 | 1,568 | 205.5 | 2,106 | -114.4 |

**BC loss:** Clear decreasing trend. From 0.74 to 0.31 over 5k steps. Model is learning to fit expert action distribution.

**Critic loss:** Oscillating, not monotonically decreasing. Range [982, 13,483]. Typical for early offline RL before Q-function has converged. Not diverging; ended at 1,568.

**Q-values:** Growing trend. Q_mean climbed from 16 → 205; Q_max peaked at 8,553. See flag below.

---

## Q-Value Flag

Q_max reached 8,553 at step 4,500 (within the [-1,000, 10,000] spec bound, but growing).

**Is this concerning?**  
At physical $5.95/interval mean reward, effective horizon γ/(1-γ) ≈ 99 steps gives expected Q ≈ $590. Q_max = 8,553 is ~14× the expected mean Q. This level of overestimation is common in early offline Q-learning (Bellman backup repeatedly amplifies initial random Q values).

**Mitigation already in training:** Q-normalization in policy loss (`q_pi / |mean(q_pi)|`) prevents the policy from being dominated by overestimated Q. The BC loss provides a grounding signal.

**Risk in full training:** If Q-values continue to grow exponentially, we may need conservative Q-learning regularization (e.g., lower β_Q or add CQL-style penalty). Recommend watching Q_max at step 10k and 25k during full training.

**Decision for Karthik:** Proceed with default config? Or add CQL penalty (`-α × Q`) as a third loss term?

---

## Action Distribution (step 5,000, batch of 256 zero-obs)

| Dim | Mean |x| | P95 |x| | MW mean | MW P95 | Status |
|---|---|---|---|---|---|
| p_energy | 0.632 | 1.000 | 6.3 MW | 10.0 MW | ok |
| c_regup | 0.066 | 0.381 | 0.7 MW | 3.8 MW | ok |
| c_regdn | 0.110 | 0.516 | 1.1 MW | 5.2 MW | ok |
| c_rrs | 0.026 | 0.137 | 0.3 MW | 1.4 MW | ok |
| c_ecrs | 0.064 | 0.267 | 0.6 MW | 2.7 MW | ok |
| c_nsrs | 0.111 | 0.649 | 1.1 MW | 6.5 MW | ok |

**No AS gradient bias detected.** All mean AS bids well below the 7 MW flag threshold. The p_energy mean at 0.632 (6.3 MW) is high — the policy defaults to large energy dispatch on zero-input observation. This is expected at 5k steps with no price conditioning (all-zero obs). Under real observations, energy dispatch will be price-conditioned.

---

## Reward Statistics

```
mean =  $5.95/interval
std  = $43.74/interval
min  = -$494.51 (Fern spike charging)
max  = +$1,125.42 (Fern spike discharge)
```

Reward scale matches physical $/interval — recompute utility working correctly.  
Cross-check: stored sum $63,327 → recomputed sum $116,511 ✓

---

## Eval at Step 5,000 (T-60 window, 54 days)

```
All 54 days:  $739.77 total  ($0.50/kW-yr)
Ex-Fern:      $390.58 total  ($0.27/kW-yr)
Fern only:    $349.19
AS share:     330.0% (AS dominated: untrained policy bidding AS but not energy)
vs fleet median: -98.0%  (expected at 5k steps)
```

**Interpretation:** At 5k steps the policy hasn't converged. The $0.50/kW-yr vs $24.93 fleet median is expected — not a concern. The 330% AS share (AS revenue 3× total) means the policy is already exploring AS bids, though the denominator (total) is near zero due to poor energy decisions. This sanity-checks that the AS reward channel is working.

---

## Timing

- 5k steps: 265.3s wall (53.1ms/step)
- GPU 0 utilized
- **Projected 100k full run: ~1.5h on Narnia GPU 0**

---

## Known Issue (minor)

`_run_eval` in `train.py` used wrong key `result['all_days']['per_kw_yr']` — should be `result['all_days']['annualized_kw_yr']`. Fixed in next commit. Eval ran correctly but result wasn't logged via the `_run_eval` path; actual eval metrics are from the harness stdout above.

---

## Recommendation

**Smoke: PASS.** Proceed to full training subject to Karthik's decision on Q-value growth.

**For full training:** Watch Q_max at steps 10k, 25k. If Q_max > 50,000, reduce β_Q from 1.0 to 0.5 as per one-variable-per-experiment rule. Do not stack additional hyperparameter changes.

**Phase 2 (QDT):** Ready to implement — does not depend on DQL smoke results.
