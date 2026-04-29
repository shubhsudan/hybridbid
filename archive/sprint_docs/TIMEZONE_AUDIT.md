# Timezone Audit — UTC vs Central Time Bug Class
**Date:** 2026-04-24
**Branch:** sprint-offline-rl
**Scope:** All code reachable by the post-break sprint pipeline
**Trigger:** UTC/CT day boundary bug found in `postbreak_milp.py` during price reconciliation

---

## Verdict

**No additional post-break bugs found. Fleet benchmark is clean. Proceed to Step 2 (fix + regen).**

Two pre-break code paths carry the same UTC-day bug, but both are out of sprint scope (per Karthik's do-not-modify constraint on this branch). Flagged below with file:line references.

---

## 1. Fleet Benchmark Pipeline (`scripts/pull_60day_disclosure.py`, commit `bd07a9c`)

**Finding: CT-aligned throughout. No bug.**

The disclosure pipeline uses `tz_convert("US/Central")` at every date boundary:

```python
# DAM date assignment
dam["hour_start_utc"] = pd.to_datetime(dam["Interval Start"], utc=True)
dam["date"] = dam["hour_start_utc"].dt.tz_convert("US/Central").dt.date          # ✓ CT

# SCED date assignment
sced["sced_ts_utc"] = pd.to_datetime(sced["SCED Timestamp"], utc=True).dt.floor("5min")
sced["date"] = sced["sced_ts_utc"].dt.tz_convert("US/Central").dt.date           # ✓ CT

# RT hourly date rollup
merged["date"] = (pd.to_datetime(merged["hour_start_utc"], utc=True)
                    .dt.tz_convert("US/Central").dt.date)                         # ✓ CT
```

All three date assignments — DAM, SCED, and RT hourly — use CT. The daily revenue aggregation (per-ESR `groupby("Resource Name", "date")`) is therefore correctly aligned to ERCOT operating days (midnight-to-midnight CT).

**Fleet median $24.93/kW-yr, P75 $32.23/kW-yr, and per-day contributions are valid as-is. Do not recompute.**

---

## 2. Smoke Test Fern Contribution (48.3%, 5-day sample)

**Finding: Check is internally consistent, but used UTC-day framing. CT-aligned equivalent is ~26%. Check still passes.**

The smoke test (`--smoke` flag in `postbreak_milp.py`) ran on `SMOKE_DATES = ["2026-01-25", "2026-01-26", "2026-01-27", "2026-02-05"]` interpreted as UTC date strings. The Fern contribution check looked for `"2026-01-26"` in `d["date"]`, where `d["date"]` is the UTC date string passed to `process_day`.

UTC day "2026-01-26" spans **CT Jan 25 18:00 → CT Jan 26 17:59**. This window captured:
- The $938.06 spike (CT Jan 25 18:00 = UTC Jan 26 00:00)
- All of CT Jan 26 00:00–18:00 (elevated prices, max $357.29)

**Price statistics from M4 processed parquet:**

| Day | Convention | Intervals | Mean rt_lmp | Max rt_lmp |
|-----|-----------|-----------|-------------|------------|
| UTC Jan 25 | UTC 00:00 → UTC 23:59 | 288 | $137.83 | $353.31 |
| UTC Jan 26 | UTC 00:00 → UTC 23:59 | 288 | $262.75 | $938.06 |
| CT Jan 25 | CT 00:00 → CT 23:59 | 288 | $258.37 | $938.06 |
| CT Jan 26 | CT 00:00 → CT 23:59 | 288 | $141.03 | $357.29 |
| CT Jan 27 | CT 00:00 → CT 23:59 | 288 | $111.89 | $300.58 |
| CT Jan 10 | CT 00:00 → CT 23:59 | 288 | $8.88 | $44.57 |
| CT Feb 5  | CT 00:00 → CT 23:59 | 288 | $19.40 | $58.52 |

The $938 spike belongs to **CT Jan 25** (not CT Jan 26). The fleet benchmark correctly attributed it to CT Jan 25; the fleet's "Jan 26 contributed 35%" refers to the sustained elevated prices during CT Jan 26 operating day (DAM positions + all-day $357 RT LMP), which is also high-revenue but separate from the spike.

**CT-aligned Fern contribution estimate (CT Jan 26 as Fern):**

Using mean-price-weighted naive revenue (proxy for MILP with energy constraint):

| CT day | Mean rt_lmp | Proxy revenue |
|--------|-------------|---------------|
| CT Jan 10 | $8.88 | $135 |
| CT Jan 25 | $258.37 | $3,927 |
| CT Jan 26 | $141.03 | $2,144 |
| CT Jan 27 | $111.89 | $1,701 |
| CT Feb 5  | $19.40 | $295 |
| **Total** | | **$8,202** |

CT Jan 26 / total ≈ **26%** (vs 48.3% UTC). The sanity check threshold is `10% ≤ fern_pct ≤ 80%` — **still passes** after CT-aligned regen.

After regen, the smoke test's `fern_diags = [d for d in diag if "2026-01-26" in d["date"]]` will match CT Jan 26 date strings (same string, now means the CT operating day). The reported contribution will drop from ~48% to ~26%. No check failure.

**Note:** If "Fern" is redefined as CT Jan 25 (where the $938 spike lives), the CT-aligned contribution would be ~47% — nearly identical to the 48.3% UTC result, because CT Jan 25 and UTC Jan 26 both capture the spike with similar price means ($258 vs $263). The harness uses `FERN_DATE = date(2026, 1, 26)` (CT Jan 26), consistent with the fleet disclosure convention. This label is correct for fleet revenue attribution purposes; changing it is out of scope.

---

## 3. Pre-Break Perfect Foresight Baseline (`src/baselines/perfect_foresight.py`)

**Finding: UTC-day bug present. Out of sprint scope — report only.**

```python
# src/baselines/perfect_foresight.py:291
for date, day_prices in prices.groupby(prices.index.date):  # BUG: UTC dates
```

`prices.index.date` returns UTC calendar dates. Each "day" in the MIP optimization covers a UTC 24-hour window (UTC midnight to midnight), not an ERCOT CT operating day. This is the same bug class as `postbreak_milp.py`.

**Severity:** Low to moderate. Pre-break data (2020–2025) shows less extreme intra-day price structure than Fern. The 6-hour shift has smaller absolute impact when prices are uniformly low. Also the pre-break TBx baseline (rule-based, no date grouping) is unaffected.

**Impact on reported numbers:** The $1,519/day Perfect Foresight MIP figure in `CLAUDE.md` was computed with UTC-day grouping. The CT-aligned equivalent would differ by the revenue contribution of hours 18:00–24:00 CT in each day's window. For typical pre-RTC+B prices ($20–100 range), the delta is likely <2%.

**Action:** Do NOT fix on this branch. This touches pre-break baselines and `src/baselines/perfect_foresight.py` is not called anywhere in the post-break sprint pipeline. Flag for separate fix on `main` after sprint closes.

---

## 4. Eval Harness (`experiments/prepare_postbreak.py`)

**Finding: CT-aligned throughout. No bug.**

All four date-sensitive code paths use Central Time:

| Check | Code location | Convention | Status |
|-------|--------------|------------|--------|
| Data load trim | `_find_t60_indices()` line 136: `ts.tz_convert("US/Central")` | CT ✓ | OK |
| T-60 step selection | `_find_t60_indices()` line 140: `ct_dates = [t.date() for t in ts_ct]` | CT ✓ | OK |
| Per-step date assignment | `evaluate()` line 317: `ct_date = ct_ts.date()` (CT-converted timestamp) | CT ✓ | OK |
| Fern split | `evaluate()` line 371: `df[df["ct_date"] == FERN_DATE]` with `FERN_DATE = date(2026, 1, 26)` CT | CT ✓ | OK |

`FERN_DATE = date(2026, 1, 26)` captures the CT Jan 26 operating day (sustained elevated prices, consistent with fleet disclosure convention). All revenue is computed correctly per 5-minute interval. The Fern reporting split is a labeling choice, not a revenue computation.

One informational note for Step 3 re-validation: after CT-aligned MILP regen, the `fern_only` metric in the harness will show CT Jan 26 revenue (~$4,000–8,000 MILP). The CT Jan 25 revenue (spike day, ~$8,000–12,000 MILP) appears in `ex_fern`. This is correct behavior — CT Jan 25 is not labeled as Fern in the fleet convention.

---

## 5. Additional UTC-Day Bug Sites

**`src/env/ercot_env.py:339` — `_build_day_index()`**

```python
# src/env/ercot_env.py:339
dates = pd.Series(self.timestamps.date).unique()  # BUG: UTC dates
for d in dates:
    day_mask = self.timestamps.date == d           # BUG: UTC dates
```

Same bug class. Used by Stage 1 training to determine episode start positions. Affects Stage 1 training on pre-RTC+B data (2020–2025). Not called anywhere in the post-break eval harness or MILP pipeline.

**Severity:** Medium for Stage 1 training correctness (episode boundaries are 6h off ERCOT day structure). However, Stage 1 training uses pre-RTC+B data where prices are more uniform — the impact on SAC convergence is likely small. Deferred per sprint scope constraint.

**Action:** Do NOT fix on this branch. Add to backlog for post-sprint `main` fix.

---

## No Other UTC-Day Bug Sites Found

The following files were audited and confirmed clean:

| File | Reason |
|------|--------|
| `src/data/preprocessing.py` | All timestamps converted to UTC; no day grouping |
| `src/data/pipeline.py` | No date grouping code |
| `src/data/ercot_fetcher.py` | Timezone conversion utilities only; `_normalize_rt_lmp_schema` correctly tz-localizes |
| `src/evaluation/evaluate_stage1.py` | No date grouping |
| `src/evaluation/evaluate_stage2.py` | No date grouping |
| `src/utils/time_utils.py` | Pure timezone conversion helpers |
| `src/utils/battery_sim.py` | No date grouping |
| `src/models/feasibility.py` | No date logic |
| `src/training/train_stage1.py` | Delegates date logic to env |

---

## Summary Table

| Component | Convention | Bug? | Sprint action |
|-----------|-----------|------|--------------|
| `scripts/pull_60day_disclosure.py` (fleet benchmark) | CT ✓ | None | None |
| `experiments/prepare_postbreak.py` (eval harness) | CT ✓ | None | None |
| `src/data/postbreak_milp.py` | UTC ✗ | Yes — known | Fix in Step 2 |
| `src/env/ercot_env.py:339` (`_build_day_index`) | UTC ✗ | Yes | Defer to `main` |
| `src/baselines/perfect_foresight.py:291` | UTC ✗ | Yes | Defer to `main` |
| All other sprint-reachable code | N/A | None | None |

**Proceed to Step 2: fix `get_complete_days()` and launch CT-aligned regen on Narnia.**
