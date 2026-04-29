# BC Method Closeout — Behavior Cloning from MILP Expert

**Sprint:** sprint-offline-rl  
**Completed:** April 25, 2026  
**Session:** cc-baselines

---

## Result Summary

| Window       | Revenue       | $/kW-yr |
|---|---|---|
| All days     | $43,146.48    | 29.16   |
| Ex-Fern      | $39,077.84    | 26.91   |
| Fern-only    | $4,068.64     | —       |

**vs fleet median (+24.93 $/kW-yr):** +17.0%  
**vs MILP-replay ceiling (58.40 $/kW-yr):** −50.1%  
**vs PF oracle (63.16 $/kW-yr):** −53.8%  
**Fleet percentile:** 25–50 (beats median, misses top quartile at $32.23/kW-yr)

---

## What Was Tried

**Architecture:** 3-layer MLP, 256 hidden units, ReLU activations.  
Output head: `tanh × P_max` for `p_energy` (signed energy), `sigmoid × P_max` for 5 AS dims.  
Input: flattened `price_history (32×12=384)` + `static_features (14)` = 398 dims.

**Loss:** MSE on physical MW actions. Stored NPZ `rewards` field never loaded — reward convention is mixed-unit (Li et al. Eq.26 artifact); discarding it is correct for BC.

**Training:** Adam, lr=3e-4, weight_decay=1e-4, batch_size=256. Early stopping, patience=5.  
Dataset: 15,552 training transitions (68 CT days), 14,208 val transitions (62 CT days). Actions converted from p.u. → physical MW at load time (× P_max=10).

**Trajectory:** 22 epochs total. Best val_loss = 8.6454 at epoch 17. Early stopped at epoch 22 (5 epochs without improvement: epochs 18–22). Train loss continued declining (8.6→4.8) while val loss plateaued — mild overfitting, expected given ~15k transitions.

**In-loop probe at epoch 20 (last):**
- `p_energy`: mean=−0.645, std=4.644, zero=2%  
- `c_regup`: mean=+2.29, std=2.59, zero=20%  
- `c_regdn`: mean=+2.10, std=2.73, zero=40%  
- `c_rrs`: mean=+0.11, std=0.20, zero=74%  
- `c_ecrs`: mean=+0.15, std=0.31, zero=70%  
- `c_nsrs`: mean=+1.25, std=1.85, zero=27%

No collapse (max std = 4.64 MW >> 0.5 MW threshold).

---

## What Worked

- BC trivially beats both TBx baselines (10.96 / 12.52 $/kW-yr) by a wide margin.
- BC beats fleet median (+17.0%) — demonstrates that imitating MILP actions alone is sufficient to outperform an average real ERCOT operator.
- No action collapse, no NaN, numerically stable throughout.
- AS products are learned: regup and nsrs both have non-trivial mean and std. RRS/ECRS near-zero matches MILP training distribution (MILP rarely clears these in training window).
- Sprint discipline held: smoke gate reviewed, early stop exit handled cleanly.

---

## What Failed / Fell Short

**Primary failure: conservative energy dispatch.**  
BC energy std = 3.70 MW vs PF std = 5.79 MW. BC charges at max (>9.5 MW) only 0.7% of intervals (PF: 15.5%) and discharges at max only 1.2% (PF: 14.3%). BC timidly follows the expert's average rather than learning sharp high/low thresholds.

**Consequence: systematic high-SoC bias.**  
BC spends 25.9% of intervals at SoC_MAX (18 MWh hard cap), P75 = 18.00 MWh. PF spends only 15.8% at cap, P75 = 15.37 MWh. BC hoards energy and misses discharge opportunities at high prices.

**Revenue gap vs MILP-replay ($58.40/kW-yr):** −50.1%. Despite imitating the same expert that generated the MILP-replay policy, BC recovers only ~50% of oracle value. This is characteristic of compounding error in BC (covariate shift) — BC never saw continuous-SoC states during training (training MILP resets SoC to 50% each CT midnight).

**Covariate shift is the main structural failure.** The training distribution has SoC starting at 50% each day by construction (daily MILP reset). The eval harness has continuous SoC. BC was never trained on states with SoC=18 MWh at day start, which occur 25.9% of the time in eval. This is a fundamental distribution mismatch that BC cannot fix without DAgger or an online correction phase.

---

## Fern Investigation (BC Fern-day > PF Fern-day)

**Triggered because:** BC Fern-day revenue ($4,068.64) > PF Fern-day ($3,912.54), +$156.09.

**Verdict: SoC entry-state artifact. NOT memorization.**

| | BC | PF |
|---|---|---|
| SoC at Jan 26 CT midnight | **18.00 MWh (SOC_MAX cap)** | 3.58 MWh (17.9%) |
| Fern energy revenue | $2,893.51 | $1,916.77 |
| Fern AS revenue | $1,175.14 | $1,995.78 |
| Fern total | $4,068.64 | $3,912.54 |

BC enters Fern at the hard SoC ceiling (18.00 MWh) — a consequence of its high-SoC bias over the prior 25 eval days, not deliberate positioning. With a full battery, BC can discharge more energy during Fern's extreme prices (+$976 energy) but cannot offer charge-direction AS (regdn, rrs, ecrs, nsrs), which requires SoC headroom. PF enters Fern nearly empty (3.58 MWh), enabling more AS capacity but less energy discharge.

**Memorization ruled out:** BC's obs contains no calendar date. The cyclical time features (sin/cos hour, sin/cos month) cannot uniquely identify Jan 26. BC has no mechanism to "know" it's Fern day. The full-battery entry is a statistical byproduct of BC's general charging bias, not a learned Fern-specific strategy.

**Context:** BC's advantage on Fern is not repeatable or reliable. If the eval period were extended, the SoC entry luck would average out. The −50.1% gap vs MILP-replay across all 54 days is the correct measure of BC quality.

---

## Hyperparameters (final)

| Param | Value |
|---|---|
| Architecture | [398, 256, 256, 256, 6] |
| Activation | ReLU (trunk), tanh/sigmoid (head) |
| Optimizer | Adam |
| LR | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Max epochs | 50 |
| Early stop patience | 5 |
| Best val_loss epoch | 17 |
| Final epoch | 22 |

---

## Artifacts

- `methods/bc/checkpoints/best.pt` — best checkpoint (epoch 17)
- `methods/bc/training_log.json` — per-epoch losses, probe stats, Fern slice
- `data/results/eval_bc/summary.json` — full T-60 metrics with ceiling ratios
- `data/results/eval_bc/trajectory.parquet` — per-interval log

---

## Recommendation for RL Methods

The 29.16 $/kW-yr BC result sets the RL floor. Any offline RL method (Cal-QL, Diffusion-QL, QDT) that does not beat BC is a failure. The bar is not just beating fleet median — it is beating BC at 29.16 $/kW-yr.

The primary failure modes to address in RL:
1. Covariate shift (continuous SoC): ensure RL's Q-function trains on states across the full SoC range.
2. Conservative energy dispatch: RL reward signal should pressure toward high-price discharge.
3. Stored reward convention: recompute rewards from actions × prices at training time (use `methods/_shared/reward_recompute.py`).
