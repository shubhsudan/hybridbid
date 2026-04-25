# CLAUDE.md — Sprint: Offline RL for Post-RTC+B BESS Bidding
**Date:** April 25, 2026 (Day 2 of 7-day sprint)
**Branch:** `sprint-offline-rl`
**Status:** Day 1 complete. MILP trajectories generated, eval harness validated, gap explained.

---

## CRITICAL INSTRUCTIONS FOR CLAUDE CODE

1. **This file governs the sprint branch only.** For HybridBid v5.1 work, see `CLAUDE_hybridbid_v51.md` on `main`.
2. **Three sessions, three ownership domains.** Do not edit files outside your session's domain. See `## SESSION OWNERSHIP` below.
3. **Stop and report on data quality gate failures.** Never silently work around bugs.
4. **One variable per experiment.** If a method fails, replace it; don't pile on compensatory changes.
5. **Mandatory review gates.** No method progresses past 50k training steps without Karthik's explicit review.
6. **Process-level halts, not instruction-level pauses.** Every training run has a `sys.exit` at a checkpoint.
7. **Closeout docs for each method.** Even for failed methods. Document what failed and why.

---

## SPRINT GOAL

Build a demo-ready offline-RL system for ERCOT post-RTC+B BESS bidding. Compare 5 methods (3 RL + 2 baselines) plus a Li et al. transfer reproduction (colleague's task) on the T-60 window. Demo-ready by Wednesday April 30, 2026 for professor meeting.

**The deliverable to professor:** comparison table across 6 methods on the T-60 window, with three-way revenue split (all / ex-Fern / Fern-only) and fleet percentile.

**The benchmark to beat:** ERCOT fleet median $24.93/kW-yr (top-quartile $32.23/kW-yr), commit `bd07a9c`. Hub-average LMP proxy.

**The ceilings:**
- MILP daily-reset oracle: $62.62/kW-yr (~2.5× fleet)
- MILP continuous-SoC replay: $58.40/kW-yr (correct comparable upper bound; methods exceeding this would have learned SoC continuity beyond MILP)

---

## METHOD SLATE

| # | Method | Category | Owner | Session |
|---|---|---|---|---|
| 1 | MILP + transformer forecaster | Classical baseline | Karthik / Claude Code | `cc-baselines` |
| 2 | BC from MILP expert | RL floor baseline | Karthik / Claude Code | `cc-baselines` |
| 3 | Cal-QL (offline-to-online) | Primary RL contender | Karthik / Claude Code | `cc-rl-narnia` |
| 4 | Diffusion-QL | Primary RL contender | Karthik / Claude Code | `cc-rl-narnia` |
| 5 | QDT (DT + Q-relabel) | Primary RL contender | Karthik / Claude Code | `cc-rl-narnia` |
| 6 | Li et al. TempDRL (NEM→ERCOT transfer) | Scientific comparison | Colleague | (separate) |

---

## LOCKED DECISIONS (no re-litigation mid-sprint)

### Data and dates
- **Eval window (T-60)**: Jan 1 – Feb 23, 2026 (54 days, CT-aligned)
- **Training window**: Dec 5, 2025 – Feb 10, 2026 (68 days CT-aligned, includes Fern)
- **Validation window**: Feb 11 – Apr 15, 2026 (62 days CT-aligned)
- **Time-based split.** No K-fold. No random split.
- **Fern (Jan 26 CT) pinned to training set.**
- **All dates are CT (US/Central). Never UTC.** This is the bug class that took half of Day 1 to fix.

### Action and observation space
- **Action: 6D continuous, post-break only.** `[p_energy, c_regup, c_regdown, c_rrs, c_ecrs, c_nonspin]` in MW. p_energy signed (charge negative), AS dims non-negative.
- **No pre-break pretraining for any RL method.** Pre-break trajectories are 1D quantized (7-atom discrete). Action-space heterogeneity makes pretraining impractical for the sprint. Methods 3/4/5 train on post-break only (~15k transitions train split). Documented as a data-heterogeneity constraint in methodology.
- **Observation: split Dict schema (`_option_d`)**. `price_history (32, 12)` + `static_features (14,)`. Store raw pre-TTFE features. Never store post-TTFE 90-dim flat features (would lock to TTFE-at-generation-time weights).

### Battery
- **10 MW / 20 MWh, 2h duration.** 
- η_ch = η_dch = 0.95 (round-trip 0.9025). **Hardcode; do NOT read `configs/battery.yaml`** (stale 0.92 value).
- SoC limits: [0.1, 0.9] p.u. → [2.0, 18.0] MWh.

### MILP trajectory generation
- Receding horizon, 24h lookahead, per-interval (5-min) commit (NOT hourly — post-RTC+B SCED co-optimizes every 5 min).
- Joint 6D co-optimization. Energy + 5 AS solved simultaneously.
- Daily SoC reset to 0.5 at CT midnight (deliberate parallelization choice; eval is continuous-SoC, not reset).
- Soft terminal SoC penalty `λ=20 * (SoC[23:55] - 0.5)²`. Do NOT enforce as hard constraint.
- Rewards in p.u., NOT physical MW. Δt = 5/60. (Both v5 bugs fixed.)
- Solver: HiGHS (CLARABEL fallback for QP-stall edge cases like Dec 26).

### Reward structure (post-break)
- Energy revenue: `p_energy × ρ_RT × Δt` (Δt = 5/60)
- AS revenue: `sum(c_AS × ρ_AS_RT × Δt)` for each of 5 products
- **AS pays for availability, not deployment.** Do NOT condition AS revenue on `p_energy != 0`. ERCOT-specific; differs from Li et al.'s binary coupling.
- No degradation cost in step reward (paper-spec; v5 lesson).
- No reward scaling. No price normalization.

### Eval harness
- **Continuous SoC across 54-day eval window.** No midnight reset. This is the real deployment scenario.
- **Three-way revenue split required**: `all_days`, `ex_fern`, `fern_only`.
- **Method-agnostic interface:** `policy(obs) → np.ndarray (6,)`. Every method wraps its trained model in this.
- Eval harness lives at `experiments/prepare_postbreak.py` (committed).
- MILP-replay test is the canary: every method's eval should produce results that, when MILP actions are replayed through the harness, give $58.40/kW-yr ± 2%.

### What is NOT in this sprint
- T-1 live demo pipeline (cut Day 1 — recruiter artifact deferred to post-sprint)
- IQL, RLPD (dropped)
- DT, ReBRAC (dominated)
- Deep methodological exploration beyond the 3 RL picks
- Pre-break MILP UTC bug fix (deferred to `main`, files in audit `TIMEZONE_AUDIT.md`)
- Beating Li et al. reproduction (colleague's scope)

---

## SESSION OWNERSHIP (no overlapping file edits)

### `cc-baselines` (M4)
**Files owned:**
- `experiments/prepare_postbreak.py` (eval harness)
- `methods/milp_forecaster/` (Method 1)
- `methods/bc/` (Method 2)
- `data/results/eval_*` (all eval outputs)
- Recomputed baselines: TBx, Perfect Foresight MIP on T-60 (joint 6D)
- `data/processed/` (read-only access)

### `cc-rl-narnia` (Narnia)
**Files owned:**
- `methods/cal_ql/` (Method 3)
- `methods/diffusion_ql/` (Method 4)
- `methods/qdt/` (Method 5)
- `checkpoints/sprint/` (all training artifacts)
- `logs/sprint/` (training logs)
- `data/processed/` (read-only access)

### Shared (read-only for both sessions)
- `src/env/ercot_env.py` (do NOT edit; fork or wrap if needed)
- `src/data/postbreak_milp.py` (frozen; do not modify)
- `data/processed/receding_horizon_postbreak_*_option_d.npz` (frozen trajectory files)
- This `CLAUDE.md`

If a session needs to modify a shared file, stop and ask Karthik first.

---

## DATA STATE (verified, Day 2 morning)

### Trajectory files (committed, CT-aligned)
- `data/processed/receding_horizon_postbreak_train_option_d.npz`: 68 CT days, $116,669 / $62.62/kW-yr
- `data/processed/receding_horizon_postbreak_val_option_d.npz`: 62 CT days, $76,525 / $45.05/kW-yr

### Schema (Dict-style, `_option_d`)
- `price_history`: (N, 32, 12) — rolling 32-step window of [RT LMP, 5 RT MCPC, DAM SPP, 5 DAM AS]
- `static_features`: (N, 14) — 7 system + 6 cyclical time + 1 SoC
- `actions`: (N, 6) — 6D continuous joint action
- `rewards`: (N,) — per-interval revenue, **p.u.**
- `next_price_history`: (N, 32, 12)
- `next_static_features`: (N, 14)
- `dones`: (N,) — True at CT day boundaries (daily-reset training only)
- `soc`: (N,) — diagnostic SoC trace

### Price data
- All M4 parquets through Apr 15, 2026 ✓
- April data on M4 (transferred from Narnia)
- All datasets agree at UTC timestamps; CT-aligned everywhere in sprint code

### Known data caveats (carry into methodology)
- RT energy uses hub-average LMP proxy (per-RN SPP unavailable historically). Same proxy as fleet benchmark — comparison is self-consistent.
- Jan 26 CT (Winter Storm Fern) = 35% of fleet revenue, expected ~25-40% of any method's revenue.
- 84 intervals >$500 in entire post-break period — small-sample scarcity tail.
- Pre-break pipeline has UTC-vs-CT bugs (`ercot_env._build_day_index:339`, `perfect_foresight.py:291`); deferred to `main` issues.

---

## DAY-BY-DAY (sprint plan, slipped 1 day from original)

### Day 2 (Friday April 25) — current
**Status:** Day 1 took longer due to UTC/CT reconciliation. Day 2 work resumes today.

`cc-baselines`:
- Eval harness (DONE, validated)
- Method 1 (MILP + transformer forecaster) implementation
- Method 2 (BC from MILP expert) implementation
- Recompute baselines (TBx, Perfect Foresight MIP) on T-60 with joint 6D action

`cc-rl-narnia`:
- Method 4 (Diffusion-QL) smoke test (5k steps, no NaN)
- Method 5 (QDT) smoke test (5k steps, no NaN)
- Full training launches at end of Day 2 if smoke passes

### Day 3 (Saturday April 26)
- Method 3 (Cal-QL) implementation, offline-only (bootstrap from post-break trajectories; simulator path was descoped Day 1)
- Methods 4/5 mid-training evals (25k steps)
- Methods 1/2 first eval results

### Day 4 (Sunday April 27)
- Method 3 launched
- Methods 4/5 continuing
- First six-method comparison preview

### Day 5 (Monday April 28)
- All methods completing
- Colleague Li et al. reproduction integrated
- Six-method comparison table on T-60
- Executive summary draft

### Day 6 (Tuesday April 29)
- Final evals
- Demo construction
- Presentation 80% drafted
- **Hard gate**: methods that aren't presentable get cut, no rescue

### Day 7 (Wednesday April 30)
- Presentation review, demo rehearsal
- Professor meeting

---

## EVALUATION DIMENSIONS

Every method produces in `data/results/eval_<method_name>/`:

1. **`trajectory.parquet`**: per-interval log
2. **`summary.json`**: aggregate metrics
3. **`comparison_card.md`**: human-readable summary

Required metrics in `summary.json`:
- `all_days_$_per_kW_yr`, `ex_fern_$_per_kW_yr`, `fern_only_$_per_kW`
- `energy_revenue_$`, `as_revenue_$` (split by 5 AS products)
- `cycles_per_day`
- `soc_histogram`: P5, P25, P50, P75, P95
- `action_distribution`: per-dim mean, P95, fraction_zero
- `fleet_percentile`: <25 / 25-50 / 50-75 / 75-100
- `vs_fleet_median`: ($/kW-yr − $24.93) / $24.93
- `vs_milp_replay_ceiling`: ($/kW-yr − $58.40) / $58.40

---

## SPRINT DISCIPLINE RULES (from Day 1 lessons)

1. **Process-level halts.** Every training run has a `sys.exit` at a checkpoint. Mid-run "fixes" are forbidden.
2. **Mandatory review gates.** 5k smoke → 25k mid-eval → 50k checkpoint review → continue or stop.
3. **Stop-and-report on data quality gate failures.** Day 1's UTC/CT bug was caught because the harness's MILP-replay canary refused to silently produce a wrong number. Maintain this discipline everywhere.
4. **One variable per experiment.** Don't stack hyperparameter changes.
5. **Closeout docs.** Each method gets a `CLOSEOUT_<method>.md` regardless of success or failure. What was tried, what worked, what didn't, why.
6. **Daily check-in.** End of each day, each session reports: what completed, what's blocked, what's queued for tomorrow.

---

## MACHINES

- **M4 MacBook** (`karthikmattu@100.113.39.50`, `~/hybridbid`) — `cc-baselines` session, eval harness host
- **MacBook Air** (`karthikmattu@100.99.63.48`) — background tasks; M4→Air SSH does NOT work
- **Narnia** (`km5503@narnia.gccis.rit.edu`, GPU node 18, A16) — `cc-rl-narnia` session, RL training. CVXPY + HiGHS + CLARABEL all confirmed available in `hybridbid` conda env.
- Connected via Tailscale.
- Lab CPLEX machine: not used in sprint (RDP-only, sequential SoC dependency made setup cost too high).

---

## KEY REFERENCES

1. **Li et al. (2024)** — arXiv:2402.19110. TempDRL paper, primary reference for HybridBid v5.1 (parked).
2. **Cal-QL** — Nakamoto et al. 2023, NeurIPS.
3. **Diffusion-QL** — Wang et al. 2023, ICLR.
4. **QDT** — Yamagata et al. 2023, ICML.
5. **ERCOT RTC+B** — ercot.com.

---

## OPEN QUESTIONS (unblocked, not blocking)

- Two pre-break papers outside 5-year window (Howard & Ruder 2018, Ash & Adams 2020) — pending instructor confirmation.
- Final paper methodology updates needed for v5.1 architecture (post-sprint task).
- T-1 recruiter artifact (post-sprint, separate weekend project after winning method is known).
- Pre-break UTC-vs-CT bugs filed for `main` (separate from sprint).

---

## END-OF-WEEK DELIVERABLE

For Wednesday Apr 30 professor meeting:

1. Executive summary: one paragraph (what was tried, what won, what didn't)
2. Comparison table: 6 methods × 3 metrics (all_days, ex_fern, fern_only) × fleet percentile
3. Demo: T-60 panel only (T-1 cut Day 1)
4. Technical appendix: closeout docs per method
5. Forward-looking section: implications for offline RL on post-RTC+B (small-sample, Fern-dominated, AS-decoupled)
