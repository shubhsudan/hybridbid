# cc-rl-narnia Phase 0 Recon
**Session:** cc-rl-narnia  
**Date:** 2026-04-25 (Day 2)  
**Status:** COMPLETE — awaiting green-light before Phase 1

---

## 1. Trajectory Files (M4)

**Path discrepancy vs CLAUDE.md:** CLAUDE.md references `data/processed/receding_horizon_postbreak_*_option_d.npz`. These files do NOT exist. The actual postbreak trajectory files are at `data/expert_trajectories/receding_horizon_postbreak_{train,val}.npz`. The `_option_d`-suffixed files in `data/expert_trajectories/` are pre-break quantized trajectories (420k transitions, 1D int64 actions) — incompatible schema, do NOT use for Methods 4/5.

**Correct files (used for Methods 4 and 5):**

| File | Transitions | Days | Keys |
|---|---|---|---|
| `data/expert_trajectories/receding_horizon_postbreak_train.npz` | 19,584 | 68 (×288 intervals) | price_history(19584,32,12), static_features(19584,14), actions(19584,6), rewards(19584,), next_*, dones, truncateds, soc |
| `data/expert_trajectories/receding_horizon_postbreak_val.npz` | 17,856 | 62 (×288 intervals) | same schema |

**Schema validation:**
- `actions`: float32 (19584, 6), range [-1.0, 1.0] — **normalized p.u.**, NOT physical MW
- `rewards`: float32, mixed convention (energy term: p.u. via Li et al. Eq.26; AS term: physical MW × price × Δt). Total train reward = 63,327.
- `dones`: all False (0 episode terminations). **`truncateds` marks CT-midnight boundaries** — 68 truncations in train, 62 in val. Critical for Q-value bootstrapping (don't zero out Q at truncations).
- `soc`: MWh values, range [2.0, 18.0] MWh (SoC bounds respected).

**Action convention note:** Training data actions are p.u. [-1,1] × P_MAX=10 MW → eval harness policy must de-normalize (multiply by 10) when returning physical MW to harness.

---

## 2. Narnia Environment

| Library | Version |
|---|---|
| torch | 2.5.1+cu121 |
| CUDA available | True |
| numpy | 2.4.3 |
| pandas | 2.3.3 |

All required libraries confirmed.

---

## 3. Narnia GPU

32× NVIDIA A16 cards, 15,356 MiB each. Status at recon time:

- GPUs 0–5, 7–25: ~14,962 MiB free, 0% utilization — **essentially idle**
- GPU 6: 5,439 MiB free (light usage)
- GPU 26: 14,831 MiB free, 14% util (light background)

**Plan:** GPU 0 → Diffusion-QL, GPU 1 → QDT (parallel, split free memory).

---

## 4. Trajectory Files on Narnia

Files confirmed present and **byte-identical** to M4:

```
~/hybridbid/data/expert_trajectories/receding_horizon_postbreak_train.npz  1,600,481 bytes
~/hybridbid/data/expert_trajectories/receding_horizon_postbreak_val.npz    1,485,244 bytes
```

No SCP needed.

---

## 5. Git State

| Machine | Branch | Head commit |
|---|---|---|
| M4 | sprint-offline-rl | `92c5a49` (2 commits ahead of remote) |
| Narnia | sprint-offline-rl | `5cd0465` (= remote, behind M4 by 2 commits) |

**Action needed:** Push M4 commits (`92c5a49`, `aca17f8`) to remote before Narnia can pull new method code. Will do as part of method implementation commits.

M4 working tree has unstaged modifications: `CLAUDE.md`, `receding_horizon_postbreak_train.npz`, `receding_horizon_postbreak_val.npz`. NPZ modifications are byte-identical to committed versions (recomputed locally; content unchanged). CLAUDE.md modifications are Karthik's notes — will not touch.

---

## 6. Method Scaffolding

`methods/` directory does NOT exist on M4 or Narnia. Will create:
```
methods/
  diffusion_ql/
    __init__.py
    model.py      (encoder, diffusion net, twin Q)
    train.py      (5k smoke / 100k full)
    policy.py     (eval harness wrapper)
    SMOKE_REPORT.md  (after smoke)
  qdt/
    __init__.py
    model.py      (CQL critic, relabeler, DT)
    train.py      (stage 1 CQL + stage 2 relabel + stage 3 DT)
    policy.py     (eval harness wrapper)
    SMOKE_REPORT.md  (after smoke)
```

---

## 7. Critical Design Notes (from code inspection)

1. **Episode termination:** Use `truncateds` (not `dones`) to mask terminal Q-targets. At truncated steps, standard Bellman is still valid (use next-state bootstrap). At terminated steps (none here), zero the bootstrap. Both DQL and QDT critics should NOT zero Q at truncation boundaries.

2. **Action de-normalization for eval harness:** Policy wrapper must scale p.u. actions → physical MW via `× P_MAX = 10`. Eval harness expects physical MW.

3. **Reward convention:** Do not recompute or normalize stored rewards. Use as-is for Q-learning signal. $/kW-yr metrics come from the eval harness, not the stored rewards.

4. **dones vs truncateds for DT context:** In QDT's Stage 3, context window should NOT be cut at truncation boundaries (they're CT midnight resets in training, not real episode ends for eval). Harness runs continuous 54 days. Use sliding window, ignore truncateds in DT context.

---

## Phase 0: COMPLETE

Ready to proceed to Phase 1 (Diffusion-QL) on Karthik's green-light.
