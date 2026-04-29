# HybridBid — Offline RL for ERCOT Post-RTC+B BESS Bidding

Optimizes revenue for a 10 MW / 20 MWh battery storage system bidding in ERCOT's post-RTC+B co-optimized market (energy + 5 ancillary services, 5-minute SCED intervals). Compares offline RL methods against MILP-based baselines on a 54-day evaluation window (Jan 1 – Feb 23, 2026).

**FOML Spring 2026 — Final Project**  
Karthik Mattu (`km5503@g.rit.edu`)

---

## Results Summary

Evaluated on the T-60 window (54 days, hub-average LMP proxy, same basis as the ERCOT fleet benchmark).

| Method | All days ($/kW-yr) | ex-Fern ($/kW-yr) | Fern-only ($/kW) | Fleet percentile |
|---|---|---|---|---|
| TBx energy-only (baseline) | ~11 | — | — | <25 |
| TBx + AS (baseline) | ~18 | — | — | <25 |
| MILP+Forecaster (Method 1) | 22.84 | — | — | <25 |
| **BC from MILP expert (Method 2)** | **29.16** | — | — | **25–50** |
| Cal-QL (Method 3) | — | — | — | closed out |
| Diffusion-QL (Method 4) | — | — | — | closed out |
| QDT (Method 5) | — | — | — | closed out |
| Perfect Foresight MIP (oracle) | ~58 | — | — | — |

**Fleet benchmarks:** median $24.93/kW-yr, top-quartile $32.23/kW-yr.  
**MILP-replay ceiling:** $58.40/kW-yr (continuous-SoC 6D joint).  
BC is the winning method: beats fleet median by ~17%; offline RL methods (Cal-QL, Diffusion-QL, QDT) all failed on this small-data, spike-dominated dataset.

Winter Storm Fern (Jan 26 CT) accounts for ~35% of total fleet revenue. See `src/methods/*/CLOSEOUT.md` for per-method failure analysis.

---

## Setup

### Requirements

Python 3.11. GPU required for training (CUDA 12.1 tested on A16); CPU sufficient for eval-only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the CUDA-specific torch build instead:
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Note:** The Gurobi academic license is not required. HiGHS (open-source) is the default MILP solver; CLARABEL is used as a fallback for QP-stall edge cases.

### Data

Processed Parquet files are checked in under `data/processed/` (~24 MB). The MILP expert trajectories used for offline RL training are in `data/expert_trajectories/`. Raw ERCOT files are not committed; to re-fetch:

```bash
python -m src.data.pipeline --start 2024-01-01 --end 2026-04-15
```

---

## How to Run

All commands run from the repo root.

### Evaluate a trained method on T-60

The eval harness (`src/evaluation/eval_t60.py`) takes any policy that implements the contract below and writes results to `data/results/eval_{method_name}/`.

**Policy contract:**
```python
class MyPolicy:
    def reset(self) -> None: ...
    def __call__(self, obs: dict) -> np.ndarray:
        # obs = {"price_history": (32, 12), "static_features": (14,)}
        # returns shape (6,): [p_energy, c_regup, c_regdn, c_rrs, c_ecrs, c_nsrs] in MW
        # p_energy signed (+ discharge, − charge); c_as ≥ 0
```

### Run BC (best method)

```bash
# Smoke gate first (required)
python -m src.methods.bc.run_bc --phase smoke

# Full training (50 epochs, early stopping) — runs T-60 eval on completion
python -m src.methods.bc.run_bc --phase full
```

Trained checkpoint: `models/bc/best.pt`

### Run MILP+Forecaster (Method 1)

```bash
python -m src.methods.milp_forecaster.run_milp_forecaster --phase smoke
python -m src.methods.milp_forecaster.run_milp_forecaster --phase full
```

Trained checkpoint: `models/milp_forecaster/forecaster_best.pt`

### Run T-60 baselines (TBx, Perfect Foresight MIP)

```bash
python -m src.methods.baselines.run_t60_baselines
```

### Run the six-method comparison notebook

```bash
jupyter notebook notebooks/02_results_comparison.ipynb
```

Requires `data/results/eval_*/summary.json` to exist for each method.

### Stage 1 / Stage 2 SAC (pre-sprint, online RL — for reference)

```bash
python -m src.training.train_stage1 --config configs/battery.yaml
python -m src.training.train_stage2 --checkpoint models/sac_stage1/checkpoint_final.pt
```

---

## Project Structure

```
hybridbid/
├── README.md
├── AI_USAGE.md
├── requirements.txt
├── configs/
│   ├── battery.yaml          # Battery spec (10 MW / 20 MWh, η=0.95)
│   └── data_products.yaml    # ERCOT data product IDs
├── data/
│   ├── processed/            # Canonical 5-min Parquet files (tracked, ~24 MB)
│   ├── expert_trajectories/  # MILP receding-horizon trajectories for offline RL
│   ├── raw/                  # Downloaded ERCOT files (gitignored)
│   └── results/              # Eval outputs per method (gitignored)
├── models/                   # Final trained checkpoints
│   ├── bc/                   # BC best + last (2.7 MB each)
│   ├── cal_ql/               # Cal-QL 25k (7.9 MB, closed out)
│   ├── milp_forecaster/      # Transformer forecaster best + last (20 MB each)
│   ├── sac_stage1/           # Stage 1 SAC v5.9.2 final (5 MB)
│   └── sac_stage2/           # Stage 2 SAC v3a final (5 MB)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # ERCOT data audit and price distributions
│   └── 02_results_comparison.ipynb  # Six-method T-60 comparison (run this last)
├── src/
│   ├── data/                 # ERCOT data fetcher, pipeline, schema, preprocessing
│   ├── env/                  # ERCOT gym environment (ercot_env.py — frozen)
│   ├── models/               # Model architecture: SAC, TTFE, networks, replay buffer
│   ├── training/             # Stage 1 / Stage 2 SAC training entry points
│   ├── baselines/            # Pre-sprint TBx and energy-only perfect foresight
│   ├── evaluation/           # T-60 eval harness, aggregate, diagnostics
│   ├── utils/                # CT/UTC time utils, battery simulator
│   └── methods/              # Sprint offline RL methods
│       ├── _shared/          # Shared reward recomputation
│       ├── baselines/        # T-60 TBx and PF-MIP policies
│       ├── bc/               # Behavioral Cloning (winner, $29.16/kW-yr)
│       ├── cal_ql/           # Cal-QL offline-to-online (closed out)
│       ├── diffusion_ql/     # Diffusion-QL (closed out)
│       ├── qdt/              # Q-weighted Decision Transformer (closed out)
│       └── milp_forecaster/  # MILP + transformer price forecaster ($22.84/kW-yr)
├── tests/                    # 111 unit tests (pytest)
├── scripts/                  # Data inspection and preprocessing utilities
└── archive/                  # Historical artifacts (see archive/README.md)
```

---

## Key Design Decisions

**Action space (6D continuous, post-RTC+B only):**  
`[p_energy, c_regup, c_regdown, c_rrs, c_ecrs, c_nonspin]` in MW. `p_energy` signed (charge negative), AS dimensions non-negative. ERCOT pays AS for availability, not deployment — AS revenue is unconditional on energy dispatch. This differs from Li et al. (2024).

**Observation space (Dict schema):**  
`price_history: (32, 12)` — rolling 32-step window of [RT LMP, 5 RT MCPCs, DAM SPP, 5 DAM AS prices]; `static_features: (14,)` — 7 system features + 6 cyclical time encodings + 1 SoC. Raw pre-TTFE features stored to avoid locking trajectory files to a specific TTFE checkpoint.

**MILP expert trajectories:**  
Receding-horizon, 24h lookahead, per-interval (5-min) commit, daily SoC reset to 0.5 at CT midnight. Soft terminal SoC penalty (λ=20). Solver: HiGHS (CLARABEL fallback). Training split: Dec 5, 2025 – Feb 10, 2026 (~15k post-break transitions).

**Why offline RL methods failed:**  
The post-break dataset has ~15k transitions (68 training days) with a heavily skewed reward distribution — Winter Storm Fern (Jan 26) alone accounts for ~25–40% of total revenue. Cal-QL exhibited entropy collapse and OOD extrapolation failure. Diffusion-QL suffered Q-divergence from action-space infeasibility extrapolation. QDT failed on RTG distribution shift from the spike-dominated dataset. Full analyses in `src/methods/*/CLOSEOUT.md`.

**All timestamps are CT (US/Central).** A UTC/CT alignment bug affected pre-break pipelines; sprint code is fully CT-aligned. See `archive/sprint_docs/TIMEZONE_AUDIT.md`.

---

## References

1. Li et al. (2024). *TempDRL: Temporal-Aware Deep RL for BESS Bidding.* arXiv:2402.19110
2. Nakamoto et al. (2023). *Cal-QL: Calibrated Offline RL.* NeurIPS 2023
3. Wang et al. (2023). *Diffusion Policies as an Expressive Policy Class for Offline RL.* ICLR 2023
4. Yamagata et al. (2023). *Q-Transformer: Scalable Offline RL via Autoregressive Q-Functions.* ICML 2023
5. ERCOT RTC+B market documentation: ercot.com
