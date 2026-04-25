# Cal-QL Closeout

**Method:** Cal-QL (Nakamoto et al. 2023, NeurIPS. "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning.")
**Sprint:** offline-rl, April 2026
**Outcome:** *In progress — Day 3*
**Finding type:** TBD

---

## Sprint context

Cal-QL is the sole remaining RL contender after:
- Diffusion-QL: Q-divergence at steps 29k–31k under both β_Q=1.0 and β_Q=0.5 (Fern-dominated Q variance)
- QDT: CQL over-conservatism (P50 RTG = −$127 under α_cql=1.0 and 0.3); Stage 3 smoke below BC

Cal-QL's two structural differences from QDT Stage 1 (which failed identically):
1. **Calibration floor:** `push_up_target = max(Q_data, V_behavior)` prevents over-conservatism by anchoring Q from below at the behavior policy's return.
2. **Squashed Gaussian actor:** Q-maximization via actor update (not critic-only as in QDT). The actor provides an in-distribution gradient signal to the critic, reducing the Q-function's sensitivity to OOD extrapolation.

The question: does the calibration floor prevent the QDT failure mode (CQL over-conservatism → Q_mean→0) while also avoiding the DQL failure mode (Q divergence → Q_max explosion)?

---

## Hyperparameters

See `config.yaml`. Key values:
- `alpha_cql`: 0.3 (QDT lesson — D4RL default 1.0 over-conservative on 15k-transition data)
- `alpha_entropy`: 0.0 (no SAC entropy; CQL provides regularization)
- `n_random_actions`: 10 (10 uniform OOD + 10 policy OOD = 20 total per state)

---

## Smoke results (5k steps)

*[Fill in after Narnia smoke run — see SMOKE_RESULTS.md]*

| Metric | Value | Status |
|--------|-------|--------|
| Q_max_smoke | TBD | TBD |
| Q_mean | TBD | TBD |
| CQL term range | TBD | TBD |
| Action dist. | TBD | TBD |
| Smoke result | TBD | TBD |

---

## 25k checkpoint

*[Fill in after 25k checkpoint — requires Karthik review before continuing]*

| Metric | Value | vs Smoke |
|--------|-------|----------|
| Q_max | TBD | TBD |
| Q_mean | TBD | TBD |
| CQL term | TBD | TBD |
| Val Q_p50 | TBD | TBD |

Continuation decision: TBD

---

## 50k results

*[Fill in after 50k completion — full eval via experiments/prepare_postbreak.py by Karthik]*

| Window | $/kW-yr | Fleet percentile |
|--------|---------|-----------------|
| all_days | TBD | TBD |
| ex_fern | TBD | TBD |
| fern_only | TBD | TBD |

---

## Root cause analysis

*[Fill in after results are known]*

---

## Sprint implications

*[Fill in after results are known]*

---

## Artifacts

- `checkpoints/sprint/cal_ql/calql_step5000.pt` — smoke checkpoint
- `checkpoints/sprint/cal_ql/calql_step25000.pt` — 25k checkpoint
- `checkpoints/sprint/cal_ql/calql_step50000.pt` — 50k final (if completed)
- `data/cal_ql/V_behavior.npy` — precomputed behavior policy returns
- `logs/sprint/calql_smoke.log` — smoke training log
- `logs/sprint/calql_full.log` — full training log
- `methods/cal_ql/SMOKE_RESULTS.md` — 5k smoke report
