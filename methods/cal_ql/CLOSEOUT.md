# Cal-QL Closeout

**Method:** Cal-QL (Nakamoto et al. 2023, NeurIPS. "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning.")
**Sprint:** sprint-offline-rl, April 2026
**Outcome:** Closed at 25k steps. Q divergence confirmed. No continuation to 50k.
**Finding type:** Two-stage failure in small-data, spike-dominated, capacity-constrained regime.

---

## 1. Method Scope and Sprint Framing

Cal-QL augments conservative Q-learning (CQL) with a calibration anchor that prevents over-conservatism: the critic target includes `push_up = max(Q_data(s,a), V_beh(s))`, where `V_beh(s)` is the per-state Monte Carlo return under the behavior policy. The anchor ensures Q is never pushed below the behavior policy's value, which is the mechanism that addresses QDT's failure mode on this dataset.

The sprint scope covers the offline phase only. Cal-QL's full contribution in Nakamoto et al. is offline pre-training followed by online fine-tuning, where the calibrated Q serves as a warm-start that reduces the number of environment interactions needed for online convergence. Online fine-tuning requires a market simulator, which was scoped out on Day 1 as a sprint-level constraint. This run therefore characterizes Cal-QL's offline component on small-data, spike-dominated ERCOT -- not its full offline-to-online capability, and not its D4RL-benchmark behavior.

---

## 2. Setup

### Architecture

Flat MLP, no TTFE encoder. Twin Q-networks (2x256 ReLU) over concatenated (obs, action) input of dimension 404. Actor: shared 256-dim trunk plus separate mean and log-std heads; squashed Gaussian with mixed squash (tanh for p_energy, (tanh+1)/2 for c_as). OBS_DIM = 398 (32x12 price history + 14 static features). ACT_DIM = 6 (p_energy plus 5 AS products).

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| alpha_cql | 0.3 | Reduced from paper default 1.0; QDT lesson: D4RL default causes over-conservatism on 15k-transition data |
| alpha_entropy | 0.1 | Round 1 with 0.0 caused full mode collapse (log_pi=198, p_energy=1.000 +/- 0.000); 0.1 is the minimum to prevent log_std saturation |
| gamma | 0.99 | Standard |
| lr_actor, lr_critic | 3e-4 | Standard |
| n_random_actions | 10 | 10 uniform OOD + 10 policy OOD = 20 total per state for CQL penalty |
| batch_size | 256 | Standard |
| total_steps | 25,000 | Halted at mandatory review gate |

### Data

Training split: 19,584 transitions, 68 CT-days (Dec 5, 2025 to Feb 10, 2026, includes Winter Storm Fern on Jan 26 CT).
Validation split: 17,856 transitions, 62 CT-days (Feb 11 to Apr 15, 2026).
Post-RTC+B only; no pre-break data (action-space heterogeneity constraint, see CLAUDE.md).

### V_beh Distribution (training split)

V_behavior is the per-state discounted Monte Carlo return under the MILP behavior policy, computed backward within each CT-day episode. It provides the calibration floor.

| Statistic | Value |
|-----------|-------|
| Mean | $449 |
| P50 | $243 |
| Max | $11,952 (Fern spike states) |
| Fraction negative | 17.2% of training states |

States with negative V_beh have no calibration floor (the max() resolves to Q_data regardless of its sign), making them structurally equivalent to vanilla CQL from the start.

---

## 3. Training Trajectory

### Q and CQL Term by Checkpoint

| Step | Q_mean | Q_max | Q_p90 | CQL_mean | CQL_max | mean_log_std |
|------|--------|-------|-------|----------|---------|--------------|
| 5,000 | 306 | 3,274 | 487 | -193.8 | -92.1 | -0.584 |
| 10,000 | 679 | 2,722 | 1,030 | -59.5 | +5.6 | -0.636 |
| 15,000 | 1,451 | 4,118 | 2,096 | +34.6 | +77.1 | -0.657 |
| 20,000 | 2,525 | 7,763 | 3,663 | +87.8 | +113.3 | -0.703 |
| 25,000 | 3,631 | 8,561 | 4,990 | +117.3 | +140.2 | -0.743 |

Q_mean grew 12x across the 25k window. The CQL term mean crossed from negative to positive between steps 10k and 15k (CQL_max was already +5.6 at step 10k, indicating per-state flip was already underway). Entropy remained healthy throughout: mean_log_std drifted from -0.584 to -0.743, well clear of the -10 kill threshold.

### Actor Action Evolution (global mean, p.u., val split)

| Step | p_energy | c_regup | c_regdn | c_rrs | c_ecrs | c_nsrs |
|------|----------|---------|---------|-------|--------|--------|
| 5,000 | 0.839 | 0.932 | 0.246 | 0.077 | 0.072 | 0.924 |
| 10,000 | 0.864 | 0.990 | 0.617 | 0.224 | 0.188 | 0.971 |
| 15,000 | 0.706 | 0.996 | 0.600 | 0.488 | 0.407 | 0.988 |
| 20,000 | 0.484 | 0.997 | 0.388 | 0.286 | 0.147 | 0.987 |
| 25,000 | -0.039 | 0.994 | 0.429 | 0.148 | 0.163 | 0.998 |

Two saturation events are visible in this table. c_regup locked at approximately 0.990 p.u. by step 10k (std=0.037, down from 0.091 at step 5k). c_nsrs locked at approximately 0.988 by step 15k (std=0.079) and continued to 0.998 (std=0.003) by step 25k. The p_energy mean crossed zero between steps 20k and 25k: by step 25k the policy was on average net-charging (-0.039 p.u.), a reversal from the initially discharge-dominant behavior (0.839 at step 5k). This is consistent with the actor maximizing Q in a regime where AS bids have saturated the joint capacity budget and p_energy is being pushed around by the residual gradient signal.

---

## 4. Deployment Evaluation (T-60 Frozen Harness)

Checkpoint: `calql_step25000.pt`. Eval harness: `experiments/prepare_postbreak.py`. Continuous SoC across 54 CT-days (Jan 1 to Feb 23, 2026). Deterministic policy (squashed mean, no sampling noise). Harness applies `project_action(action_mw, soc)` at every step -- this is the designed behavior and is identical for all methods evaluated on this harness.

Canary check: MILP-replay result = $58.3961/kW-yr (target $58.40 +/- 2%). Harness validated.

### Revenue Summary

| Window | Days | Total USD | $/kW-yr |
|--------|------|-----------|---------|
| All 54 days | 54 | $34,689 | 23.45 |
| Ex-Fern | 53 | $32,975 | 22.71 |
| Fern only (Jan 26) | 1 | $1,714 | 62.57 |

### Revenue Composition

| Source | Total USD | Share |
|--------|-----------|-------|
| Energy | $26,532 | 76.5% |
| AS (regup) | $1,490 | 4.3% |
| AS (regdn) | $5,991 | 17.3% |
| AS (rrs) | $27 | 0.1% |
| AS (ecrs) | $28 | 0.1% |
| AS (nsrs) | $621 | 1.8% |

### Projection Statistics

| Metric | Value |
|--------|-------|
| Steps with joint cap binding | 15,552 / 15,552 (100%) |
| Mean pre-projection cap sum | 41.1 MW (4.1x the 10 MW limit) |
| Mean AS scale factor | 0.200 (80% cut applied at every step) |

Every step of the 54-day deployment required projection. The policy's raw output averaged 41.1 MW total capacity, 4.1x the physical limit. After projection, energy bids retained priority (p_energy survives the individual SoC clip), and AS received the remaining budget scaled proportionally. The regdn-heavy AS revenue ($5,991 of $8,157 AS total) reflects regdn's charge-direction SoC constraint being less restrictive at moderate SoC than the discharge-direction constraints on regup, rrs, ecrs, and nsrs.

### Slate Comparison

| Method | $/kW-yr |
|--------|---------|
| TBx energy-only | 10.96 |
| MILP + forecaster | 22.84 |
| **Cal-QL 25k** | **23.45** |
| Fleet median | 24.93 |
| BC from MILP expert | 29.16 |
| MILP-replay ceiling | 58.40 |

Cal-QL 25k sits between the MILP+forecaster baseline ($22.84) and the fleet median ($24.93), 5.9% below fleet median and 20% below BC.

---

## 5. Two-Stage Failure Mechanism

The 23.45 $/kW-yr result is not primarily a function of the actor's learned policy. It is the projector's output when fed the actor's saturated proposals. The two stages that produced this result are distinct and separately verifiable from the diagnostic dumps.

### Stage 1: Calibration Deactivation (step ~13k)

The calibration anchor `push_up = max(Q_data(s,a), V_beh(s))` binds only when `V_beh(s) > Q_data(s,a)`. At step 5k, Q_mean=306 and V_beh mean=$449, so the anchor was active for the majority of states. As training progressed and Q grew, the inequality flipped state-by-state. By step 10k (Q_mean=679, CQL_max already +5.6), the first states had flipped. By step 15k (Q_mean=1,451, CQL_mean=+34.6), the anchor was inactive for the majority of the dataset. After the flip, `max(Q_data, V_beh) = Q_data`, and the calibration mechanism provides no constraint. The subsequent training is operationally equivalent to vanilla CQL with a positive (reward-adding rather than penalty-adding) CQL term -- a configuration that has no analog in the method's design.

The flip was not triggered by data pathology or a hyperparameter error. It is a structural consequence of Q learning in a small-data high-reward regime: the behavior policy captured a significant fraction of the MILP oracle's revenue (V_beh max = $11,952, which is the expert's Fern-day return), and the Q function grew to match and then exceed it during normal training. The calibration anchor was designed to prevent under-estimation; it does not prevent over-estimation once Q has grown past V_beh.

### Stage 2: Q Extrapolation to Infeasible Action Space

The environment was never called during training. All transitions (s, a, r, s') in the training loop came from the MILP expert dataset. Rewards were computed at dataset load time from stored expert actions via `recompute_rewards()`, and are fixed throughout training. The actor gradient during `update_actor` is:

```
a_pi, log_pi = self.actor.sample(obs)       # raw squashed output, no projection
q_pi = self.twin_q.q_min(obs, a_pi)         # Q evaluated at a_pi directly
actor_loss = (-q_pi + alpha_ent * log_pi).mean()
```

There is no projection of `a_pi` before the Q call. The Q function was trained on feasible `(s, a_expert)` pairs drawn from the dataset, where the joint capacity constraint `|p_energy| + sum(c_as) <= P_max` always holds (MILP guarantees this). When the actor queries Q at a_pi with `sum = 41.1 MW`, it is extrapolating the Q function more than 4x outside its training distribution in the capacity dimension. Standard neural network extrapolation is optimistic in this direction: the Q function has seen only feasible states where high bids correlate with high revenue, so it assigns high values to even higher bids. The actor gradient points toward higher and higher capacity bids because Q says that is where maximum return lives.

The same mechanism applies to CQL's OOD action samples (both uniform-random and policy samples), which are also fed to Q without projection. This means the CQL penalty term is computed against Q values at infeasible points, further distorting the Q landscape.

### Combined Effect

The deployed policy is not the actor's policy; it is the projector's policy given the actor's proposals. After 25k steps of Q-inflation, the actor proposes maximum capacity on every step (mean 41.1 MW). The projector allocates energy first (up to SoC limits), then distributes the residual budget proportionally across the AS dimensions. The resulting behavior is roughly equivalent to a simple heuristic: discharge at near-maximum power, take proportional AS bids with whatever capacity remains. The $23.45/kW-yr result measures how well that heuristic performs on the T-60 window, not how well Cal-QL's offline RL learned.

---

## 6. Comparison to Other Sprint Methods

All three RL methods in this sprint share Stage 2 (Q extrapolation to infeasible action space). This is a structural property of offline Q-learning on datasets generated by a constrained optimizer: the training data is feasible by construction, but the Q function is queried at unconstrained actor outputs during gradient updates.

DQL failed at steps 29k to 31k under the same Stage 2 mechanism. The accelerating factor was Fern-dominated Q variance: the 35%-of-revenue Fern day created a high-return outlier that elevated Q_max faster and earlier. DQL reached the same saturation state as Cal-QL but needed more steps because it lacked a calibration floor to initially moderate growth.

QDT was protected from Stage 2 because it uses no Q-maximization during deployment: the decision transformer selects actions by imitating expert trajectories conditioned on a return target. Its failure was the opposite pathology (CQL over-conservatism causing Stage 3 smoke below BC), not Q-extrapolation. BC is protected by construction (no RL objective at all).

Cal-QL's calibration anchor delayed the onset of Q divergence relative to DQL and provided a cleaner training trajectory in the early steps (5k to 10k). It did not prevent divergence once Q crossed V_beh. The anchor mechanism helps when the dataset's behavior policy is better than Q's current estimate and the gap remains for a sustained period. On 15k training transitions with a behavior policy that captured $58/kW-yr-equivalent revenue, Q exceeded V_beh within the first half of the 25k budget.

---

## 7. Regime-Specificity and Limitations

The two failure stages described above are specific to the following conditions, all of which hold in this experiment:

**Stage 1 conditions:** Small dataset with high-quality behavior policy. The V_beh ceiling is high ($11,952 for Fern states) because the MILP expert is near-optimal. A lower-quality dataset (e.g., random or sub-optimal behavior policy) would have a lower V_beh ceiling that Q might never exceed, keeping the anchor active throughout. Alternatively, a larger dataset would slow Q growth per gradient step, giving the anchor more training steps before the flip.

**Stage 2 conditions:** Constrained action space enforced by the environment but absent from training. This condition is not specific to Cal-QL or even to offline RL: any gradient-based policy trained without the environment's constraint in the loss will face it at deployment. The silent projector in the eval harness makes this invisible during training and visible only at deployment. A projection-aware training objective (e.g., projecting a_actor before Q, or adding a feasibility penalty to actor_loss) would be the correct fix, but was not in scope for this sprint.

Cal-QL on D4RL benchmarks (hopper, halfcheetah, antmaze) does not exhibit these failure modes: the datasets are larger (100k to 1M transitions), the behavior policies are more varied (mixed-quality, not near-optimal), and the action spaces are unconstrained or lightly bounded by tanh. The failure observed here is regime-specific to small-data, spike-dominated, capacity-constrained ERCOT, not a property of the Cal-QL method generally.

---

## 8. Path B Disposition

A bootstrap-resampled online phase was approved on Day 1 and scoped as a follow-on to the offline phase if the 25k checkpoint showed calibrated Q values. The mechanism: resample training transitions by day (preserving CT-day episode structure), create a synthetic "online" buffer from the resampled data, and run Cal-QL online update steps to simulate the online fine-tuning phase without requiring a live simulator.

This path was closed on Day 3 after the 25k diagnostic confirmed that the calibration anchor had deactivated at step ~13k. The specific finding: with the anchor inactive and Q already 12x its smoke value, additional training steps from the same fixed transition support cannot restore the anchor condition (`V_beh > Q_data`). Resampling produces the same transitions with the same Q_data values that caused the flip; it does not introduce new states or new behavioral coverage. The online fine-tuning benefit in Nakamoto et al. comes from the online phase's ability to visit new states and update Q with real environment feedback, neither of which is available in the bootstrap-resample variant. Path B closure was a scope decision based on the verified mechanism, not a failure to execute the approach.

---

## Artifacts

| Artifact | Path |
|----------|------|
| 25k checkpoint | `checkpoints/sprint/cal_ql/calql_step25000.pt` |
| V_behavior cache | `data/cal_ql/V_behavior.npy` |
| Smoke results | `methods/cal_ql/SMOKE_RESULTS.md` |
| Diagnostic dumps | `methods/cal_ql/diagnostics_step{5000,10000,15000,20000,25000}.json` |
| Eval trajectory | `data/results/eval_cal_ql_25k/trajectory.parquet` |
| Eval summary | `data/results/eval_cal_ql_25k/summary.json` |
| Full eval report | `results/cal_ql_25k_eval.json` |
| MILP-replay canary | `data/results/eval_milp_replay_ct/summary.json` |
