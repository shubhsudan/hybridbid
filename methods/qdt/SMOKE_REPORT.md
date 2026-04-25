# QDT Smoke Report (v2 — in-sample SARSA fix)
**Date:** 2026-04-25  
**Machine:** Narnia GPU 16 (A16)  
**Status:** Fix implemented, re-smoke required on Narnia

---

## Background: Two bootstrap fixes, one discarded

### Smoke v1 (broken): random actions for TD bootstrap
Original Stage 1 used `a_next = random_uniform(valid_range)`. CQL's conservatism is designed
to depress Q-values on OOD actions. Using OOD actions for TD bootstrap guaranteed
pessimistic targets → Q_mean = -234 → P90 RTG = -112.49 → DT conditioned on negative returns.

### Fix attempt v1 (discarded): shuffled batch actions
Proposed `a_next = act[torch.randperm(B)]`. Rejected: this computes `Q(s', a_from_s'')` where
`s''` is unrelated to `s'`. Still OOD by a different distribution; would pass the negative-Q
symptom check without being methodologically sound.

### Fix v2 (current, committed): in-sample SARSA-style
`a_next = dataset_actions[i+1]` — the actual recorded action at the next timestep.

- No OOD risk by construction
- Consistent with IQL / ReBRAC reference implementations for continuous offline RL
- Cleanest implementation for small datasets (~15k transitions)
- At CT-midnight boundaries: `sarsa_done=1.0` zeros the γ·Q(s', a') term (next action
  belongs to a different episode after daily SoC reset)
- This is the only place `truncateds` zeros the bootstrap; `done` flag for Q-learning
  remains 0.0 throughout (no terminal states)

---

## Implementation changes

**`methods/qdt/data_loader.py` — `PostbreakDataset`:**
- Now returns 7-tuple: `(obs, act, rew, next_obs, done, next_act, sarsa_done)`
- `next_act[i] = actions[i+1]` for non-boundary; zero-padded at last index
- `sarsa_done[i] = 1.0` at 68 CT-midnight truncated positions + final dataset index

**`methods/qdt/train.py` — `run_stage1()`:**
- Unpacks 7-tuple from data iterator
- TD target: `r + γ * (1 - sarsa_done) * Q_target(s', next_act)`
- CQL penalty still uses random OOD actions (correct — conservatism penalty is supposed to
  push down Q on OOD, separate from the bootstrap)

**Local sanity checks (M4):**
```
next_act[5][0] = act[6][0] = 1.0000  ✓ (in-sample pairing correct)
sarsa_done[287] = 1.0  (last step of CT-day 1)  ✓
sarsa_done[288] = 0.0  (first step of CT-day 2)  ✓
sarsa_boundaries = 68  ✓ (one per CT training day)
```

---

## Strengthened smoke pass criteria for re-smoke

Standard (from smoke v1):
1. No NaN in Q-values
2. Q_mean > 0

New (added):
3. Q P90 > $200 (conservative floor for daily MILP revenue scale)
4. CQL penalty > 0 AND in range [0.01×, 10×] of Bellman loss
5. Bootstrap spot-check: 10 logged (act[k], next_act[k], sarsa_done[k]) pairs from last batch

All 5 checks are now embedded in `run_stage1()` smoke output.

---

## Re-smoke to run on Narnia GPU 16

```bash
# On Narnia, after git pull:
python -m methods.qdt.train --stage 1 --mode smoke --gpu 16 \
  --train-path data/expert_trajectories/receding_horizon_postbreak_train.npz \
  2>&1 | tee logs/sprint/qdt_s1_smoke_v2.log

python -m methods.qdt.train --stage 2 --gpu 16 \
  --train-path data/expert_trajectories/receding_horizon_postbreak_train.npz \
  --relabeled-path methods/qdt/dataset_relabeled_v2.npz

python -m methods.qdt.train --stage 3 --mode smoke --gpu 16 \
  --relabeled-path methods/qdt/dataset_relabeled_v2.npz \
  2>&1 | tee logs/sprint/qdt_s3_smoke_v2.log
```

Expected outcomes after fix:
- Stage 1: Q_mean positive, P90 > $200, CQL penalty positive
- Stage 2: P90 of RTG labels > $200
- Stage 3: DT loss decreasing from ~0.07, no NaN

---

## Pending (stop gate)

Results to be filled in after re-smoke completes. Do NOT launch Stage 1 full (50k) until
Karthik has reviewed re-smoke results and explicitly approved.
