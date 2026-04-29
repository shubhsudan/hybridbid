# Cal-QL: Offline Phase

**Reference:** Nakamoto et al. 2023, NeurIPS. "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning."

---

## Sprint scope

This implementation covers the **offline phase only**. The paper's online fine-tuning phase is
explicitly out of scope for this prompt. A bootstrap-resampled online phase (not true online
fine-tuning, since we lack a market simulator) is deferred to a follow-on prompt if the offline
smoke clears.

This framing must be documented in the methodology: "We evaluate Cal-QL in its offline phase.
The online fine-tuning described in Nakamoto et al. (2023) requires environment interaction;
our bootstrap-resampled analog does not introduce state-action coverage beyond the offline
support and is therefore distinct from the published algorithm."

---

## Architecture

- **Encoder:** None (flat MLP). Observation 398-dim (price_history 32×12 flattened + static_features 14).
- **Actor:** Squashed Gaussian, 2 hidden layers ×256 ReLU. Mixed action bounds:
  `p_energy ∈ (−1, 1)` via tanh; `c_as ∈ (0, 1)×5` via (tanh+1)/2.
- **Critics:** TwinQ, same 2-hidden-layer MLP structure, input `(obs, act)` = 404 dims.
  Target networks with Polyak τ=0.005.
- **No TTFE.** Pre-TTFE raw observations per sprint spec.

---

## Cal-QL calibration

Standard CQL pushes `Q(s, a_dataset)` down without a floor. In this regime (Fern-dominated
distribution, 15k transitions), this causes the same over-conservatism that killed QDT Stage 2
(P50 RTG = −$127). Cal-QL adds a calibration lower bound:

```
push_up_target = max(Q(s, a_dataset), V_behavior(s))
CQL penalty    = alpha_cql * (logsumexp(Q_OOD) − push_up_target)
```

`V_behavior(s)` is the Monte Carlo discounted return from state s under the expert policy,
precomputed from training trajectories and cached at `data/cal_ql/V_behavior.npy`.

---

## Key deviation from QDT Stage 1

QDT used `alpha_cql=1.0` (D4RL default) and collapsed to `Q_mean=4.69` by step 50k.
Even `alpha_cql=0.3` gave `P50=-$127` on the Stage 2 gate. Cal-QL starts at `alpha_cql=0.3`
(same QDT lesson) and adds the calibration floor to prevent the over-conservatism the lower
alpha alone could not prevent.

---

## Files

| File | Purpose |
|---|---|
| `networks.py` | Actor, TwinQ |
| `data_loader.py` | Dataset with V_behavior, SARSA done |
| `calql_agent.py` | Offline CQL+calibration update step |
| `train_offline.py` | Entry point; sys.exit(0) at 25k checkpoint |
| `eval_policy.py` | PolicyInterface adapter for eval harness |
| `smoke_test.py` | 5k step smoke runner |
| `config.yaml` | All hyperparameters explicit |
| `CLOSEOUT.md` | Method closeout (filled as runs complete) |
