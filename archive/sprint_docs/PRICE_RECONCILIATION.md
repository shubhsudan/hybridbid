# Price Reconciliation — Fern Jan 26 RT LMP Discrepancy
**Date:** 2026-04-24
**Branch:** sprint-offline-rl
**Analyst:** cc-baselines session (M4 MacBook)

---

## Executive Summary

**There is no price data error.** The M4 parquet, the Narnia NPZ, and the raw ERCOT API cache
are all internally consistent and correct. The apparent "$249 vs $938" discrepancy was caused by
comparing prices at different timestamps: UTC midnight Jan 26 ($938) vs CT midnight Jan 26 ($249).

**Root cause:** `postbreak_milp.py` iterates over UTC calendar dates, while
`prepare_postbreak.py` (the eval harness) uses Central Time dates. This produces a 6-hour
offset between the MILP's "Fern day" and the harness's "Fern day."

**Fix required:** Change `postbreak_milp.py` to iterate over CT operating-day boundaries
(see §4). Re-generate NPZ trajectories. Estimated wall clock: ~6h on Narnia, ~20h on M4.

---

## 1. ERCOT Ground Truth

**Source:** ErcotAPI `get_lmp_by_settlement_point()`, product NP6-788-CD, HB_HUBAVG hub,
5-minute SCED intervals. Cached in `data/raw/rt_lmp/2026-01-25.parquet` (old schema,
CPT-aware `Interval Start`) and `data/raw/rt_lmp/2026-01-26.parquet` (new schema,
CT-naive `SCEDTimestamp`, normalized to UTC by `_normalize_rt_lmp_schema`).

**Note on expected scarcity window:** The task spec assumed the Fern peak was Jan 26 14:00–18:00 CT.
The actual ERCOT data shows the peak was **Jan 25 18:00–20:00 CT** (the scarcity ramp began at
18:00 CT Jan 25, abruptly jumped from $353 to $938/MWh in one 5-min interval). By Jan 26 14:00 CT,
prices had collapsed to near-zero (renewable generation recovered). Both windows are tabulated below.

### Peak scarcity: Jan 25 CT 17:45 – 19:00 (the actual Fern peak)

| Timestamp (CT) | ERCOT ground truth ($/MWh) | UTC equivalent |
|----------------|---------------------------|----------------|
| 2026-01-25 17:45 CT | 328.18 | 2026-01-25 23:45 UTC |
| 2026-01-25 17:50 CT | 340.35 | 2026-01-25 23:50 UTC |
| 2026-01-25 17:55 CT | 353.31 | 2026-01-25 23:55 UTC |
| **2026-01-25 18:00 CT** | **938.06** | **2026-01-26 00:00 UTC** |
| 2026-01-25 18:05 CT | 935.62 | 2026-01-26 00:05 UTC |
| 2026-01-25 18:10 CT | 928.30 | 2026-01-26 00:10 UTC |
| 2026-01-25 18:15 CT | 814.99 | 2026-01-26 00:15 UTC |
| 2026-01-25 18:20 CT | 854.75 | 2026-01-26 00:20 UTC |
| 2026-01-25 18:25 CT | 905.71 | 2026-01-26 00:25 UTC |
| 2026-01-25 18:30 CT | 664.06 | 2026-01-26 00:30 UTC |
| 2026-01-25 18:35 CT | 717.41 | 2026-01-26 00:35 UTC |
| 2026-01-25 18:40 CT | 679.01 | 2026-01-26 00:40 UTC |

### ERCOT operating day "Jan 26 CT": midnight to 06:00 CT

| Timestamp (CT) | ERCOT ground truth ($/MWh) | UTC equivalent |
|----------------|---------------------------|----------------|
| 2026-01-26 00:00 CT | 249.80 | 2026-01-26 06:00 UTC |
| 2026-01-26 00:05 CT | 277.18 | 2026-01-26 06:05 UTC |
| 2026-01-26 00:10 CT | 277.19 | 2026-01-26 06:10 UTC |
| 2026-01-26 00:15 CT | 264.47 | 2026-01-26 06:15 UTC |
| 2026-01-26 00:20 CT | 263.64 | 2026-01-26 06:20 UTC |

### Task-requested window: Jan 26 CT 14:00 – 18:00 (post-scarcity, not peak)

| Timestamp (CT) | ERCOT ground truth ($/MWh) | Note |
|----------------|---------------------------|------|
| 2026-01-26 14:00 CT | −0.07 | Renewable recovery |
| 2026-01-26 14:15 CT | −0.09 | |
| 2026-01-26 14:30 CT | −0.10 | |
| 2026-01-26 14:45 CT | −0.63 | |
| 2026-01-26 15:00 CT | −1.15 | Near-zero solar |
| 2026-01-26 16:00 CT | ~−1.84 | |

The task's assumed scarcity window (Jan 26 14:00–18:00 CT) was off-peak. The scarcity peak
was the previous evening (Jan 25 18:00 CT).

---

## 2. Three-Way Comparison

For 12 intervals spanning the actual scarcity peak (Jan 25 17:45 – 18:55 CT):

| Timestamp (CT) | Timestamp (UTC) | ERCOT ground truth | M4 parquet | Narnia NPZ | Match |
|----------------|-----------------|-------------------|-----------|-----------|-------|
| Jan 25 17:45 | Jan 25 23:45 UTC | $328.18 | $328.18 | $328.18 | ✓ all 3 |
| Jan 25 17:50 | Jan 25 23:50 UTC | $340.35 | $340.35 | $340.35 | ✓ all 3 |
| Jan 25 17:55 | Jan 25 23:55 UTC | $353.31 | $353.31 | $353.31 | ✓ all 3 |
| Jan 25 18:00 | **Jan 26 00:00 UTC** | **$938.06** | **$938.06** | **$938.06** | ✓ all 3 |
| Jan 25 18:05 | Jan 26 00:05 UTC | $935.62 | $935.62 | $935.62 | ✓ all 3 |
| Jan 25 18:10 | Jan 26 00:10 UTC | $928.30 | $928.30 | $928.30 | ✓ all 3 |
| Jan 25 18:15 | Jan 26 00:15 UTC | $814.99 | $814.99 | $814.99 | ✓ all 3 |
| Jan 25 18:20 | Jan 26 00:20 UTC | $854.75 | $854.75 | $854.75 | ✓ all 3 |
| Jan 25 18:25 | Jan 26 00:25 UTC | $905.71 | $905.71 | $905.71 | ✓ all 3 |
| Jan 25 18:30 | Jan 26 00:30 UTC | $664.06 | $664.06 | $664.06 | ✓ all 3 |
| Jan 25 18:45 | Jan 26 00:45 UTC | $798.13 | $798.13 | $798.13 | ✓ all 3 |
| Jan 25 19:00 | Jan 26 01:00 UTC | $629.37 | $629.37 | $629.37 | ✓ all 3 |

**All three datasets agree at every UTC timestamp.** The "NPZ" column is derived from
`train['price_history'][14976, -1, 0]` (the current rt_lmp at the start of the MILP's
"Fern day"). The full 32-step lookback was verified: 32/32 exact matches between NPZ
price_history and M4 parquet when aligned on UTC timestamps.

**The original "$249 vs $938" discrepancy** arose from comparing:
- NPZ price_history[-1, 0] at NPZ "Fern start" (UTC midnight Jan 26 = CT Jan 25 18:00): **$938**
- M4 parquet at CT midnight Jan 26 (UTC 06:00 Jan 26): **$249**

These are prices at different times, 6 hours apart. Both are correct.

---

## 3. Root Cause Diagnosis

**Diagnosis type: Timezone convention mismatch (Option 4 from task spec).**

### The bug in `postbreak_milp.py`

Function `get_complete_days()` (line ~493) iterates over calendar dates using:

```python
# src/data/postbreak_milp.py line ~500
all_dates = pd.Series(timestamps.date).unique()
```

`timestamps` is a UTC-aware `DatetimeIndex`. `timestamps.date` returns the **UTC calendar date**
for each timestamp. For a timestamp at `2026-01-25 20:00:00 UTC` (= CT 14:00 Jan 25),
`timestamps.date` returns `2026-01-25`, not `2026-01-26`.

As a result, the MILP defines each "day" as a 24-hour UTC window:
- **"Fern day" in MILP** = UTC Jan 26 00:00 to UTC Jan 26 23:55 = **CT Jan 25 18:00 to CT Jan 26 17:55**
- This window includes the peak scarcity prices ($938 at CT Jan 25 18:00 = UTC Jan 26 00:00)

### The harness `prepare_postbreak.py`

Function `_find_t60_indices()` uses Central Time:

```python
# experiments/prepare_postbreak.py
ts_ct = ts.tz_convert("US/Central")
ct_dates = np.array([t.date() for t in ts_ct])
```

- **"Fern day" in harness** = CT Jan 26 00:00 to CT Jan 26 23:55 = **UTC Jan 26 06:00 to UTC Jan 27 05:55**
- This window starts at $249/MWh (CT midnight Jan 26) and peaks at $357/MWh (CT 04:45 Jan 26)

### The 6-hour offset

| Feature | MILP (UTC days) | Harness (CT days) |
|---------|-----------------|-------------------|
| "Jan 26" start | UTC 00:00 Jan 26 = CT 18:00 Jan 25 | CT 00:00 Jan 26 = UTC 06:00 Jan 26 |
| "Jan 26" end | UTC 23:55 Jan 26 = CT 17:55 Jan 26 | CT 23:55 Jan 26 = UTC 05:55 Jan 27 |
| First step rt_lmp | $938.06 (scarcity peak) | $249.80 (post-scarcity) |
| Day max rt_lmp | $1,350.50 | $357.29 |

### Why this produces negative MILP replay revenue

The MILP replay policy serves 288 actions per "day" that were optimized against UTC-day
prices. The harness serves those actions against CT-day prices (offset by 6 hours). The MILP
discharges at times when its prices are high ($938+), but those times correspond to 6 hours
earlier in the harness's CT-day view — where prices are from the previous afternoon. At those
harness timestamps, the battery is supposed to be idle or charging, not discharging. The
result is revenue that looks like "discharge at low prices, charge at high prices" = large
negative revenue.

### Extent of the problem

The UTC/CT offset affects **all 132 MILP days** (68 train + 64 val), not just Fern day.
Every day in the NPZ has its episode defined 6 hours earlier than an ERCOT operating day.
The MILP trajectories are internally self-consistent (each UTC day is correctly optimized),
but they are misaligned with ERCOT's CT operating-day convention.

### Validation: UTC-aligned MILP revenue matches report

To confirm the diagnosis, the MILP replay revenue was computed against UTC-day aligned prices:
- T-60 UTC window: `2026-01-01 00:00 UTC` to `2026-02-23 23:55 UTC` (15,552 steps)
- MILP energy revenue: $74,710
- MILP AS revenue: $15,935
- **Total: $90,646** vs reported $90,814 = **−0.19% delta** (within tolerance) ✓

### `ercot_env.py` has the same issue

`ERCOTBatteryEnv._build_day_index()` (line ~337) uses:

```python
dates = pd.Series(self.timestamps.date).unique()
```

This is the same UTC-date pattern. Stage 1 pre-break training also defines episodes as UTC days.
The Stage 1 agent learns to optimize over UTC-day windows, not ERCOT CT-day windows. Since the
T-60 harness uses CT days, there is an implicit eval/train distribution mismatch for Stage 1
agents as well, though it's less acute (pre-break prices are lower and less peaky).

### Code references

| File | Line | Issue |
|------|------|-------|
| `src/data/postbreak_milp.py` | ~500 | `timestamps.date` → UTC date (should be CT) |
| `src/env/ercot_env.py` | ~339 | `self.timestamps.date` → UTC date (same bug) |
| `experiments/prepare_postbreak.py` | `_find_t60_indices` | Correctly uses CT dates ✓ |

---

## 4. Recommended Fix and Estimated Cost

### Option A — Recommended: Fix MILP + regenerate NPZ trajectories

**Change in `src/data/postbreak_milp.py`** in `get_complete_days()`:

```python
# BEFORE (UTC dates — wrong for ERCOT):
all_dates = pd.Series(timestamps.date).unique()
for d in sorted(all_dates):
    date_ts = pd.Timestamp(d, tz="UTC")
    mask = timestamps.date == d

# AFTER (CT dates — correct for ERCOT operating day):
timestamps_ct = timestamps.tz_convert("US/Central")
all_dates = pd.Series(timestamps_ct.date).unique()
for d in sorted(all_dates):
    mask = timestamps_ct.date == d
```

The `date_ts` comparison (used for range filtering) also needs updating:

```python
# BEFORE:
date_ts = pd.Timestamp(d, tz="UTC")
if date_ts < start_ts or date_ts > end_ts:
    continue

# AFTER: compare CT dates to start/end CT dates
start_ct = pd.Timestamp(start, tz="UTC").tz_convert("US/Central").date()
end_ct   = pd.Timestamp(end,   tz="UTC").tz_convert("US/Central").date()
if d < start_ct or d > end_ct:
    continue
```

**Change in `src/env/ercot_env.py`** (optional — affects Stage 1 training, risky to change mid-sprint):

```python
# BEFORE:
dates = pd.Series(self.timestamps.date).unique()
for d in dates:
    day_mask = self.timestamps.date == d

# AFTER:
timestamps_ct = self.timestamps.tz_convert("US/Central")
dates = pd.Series(timestamps_ct.date).unique()
for d in dates:
    day_mask = timestamps_ct.date == d
```

**Estimated cost:**

| Task | Wall clock | Machine |
|------|-----------|---------|
| Fix `postbreak_milp.py` (2-line change) | 15 min | M4 |
| Regenerate train NPZ (68 CT days) | ~6h | Narnia |
| Regenerate val NPZ (64 CT days) | ~6h | Narnia |
| Verify harness MILP replay (within 2% of new reference) | 30 min | M4 |
| Fix `ercot_env.py` + re-run Stage 1 stability smoke test | 2h | M4 |
| **Total (Narnia parallel)** | **~8h** | — |

### Option B — Quick validation only: UTC-day mode in harness

Add a `utc_days=True` parameter to `_find_t60_indices()` in `prepare_postbreak.py`.
When True, use `timestamps.date` (UTC dates) instead of CT dates for the T-60 window.
Use this only for MILP replay validation, not for evaluating learned policies.

**Tradeoff:** Does not fix the underlying wrong-convention in MILP expert data. All offline RL
agents trained on UTC-day episodes will see a 6-hour phase shift relative to the CT-day eval.
Revenue comparisons to the fleet benchmark (which is CT-day based) will be off by up to 6 hours
of phase.

**Estimated cost:** 30 min. No NPZ regeneration.

### Option C — Ignore (not recommended)

Do nothing. The harness is correct for CT-day evaluation. Learned policies that internalize
the correct CT-day ERCOT operating pattern can still outperform the UTC-day-biased MILP.
But: expert trajectories teach the wrong phase for daily dispatch cycles (charge in the
afternoon, discharge at CT 6PM instead of CT morning peak). This will degrade offline RL
quality in a measurable but hard-to-quantify way.

### Recommendation

**Option A.** Fix `postbreak_milp.py`, regenerate both NPZ splits on Narnia in parallel
(~6h wall clock each, launch together). The `ercot_env.py` fix should be deferred —
it requires a Stage 1 re-run (~12h) and Stage 1 is not blocked on this issue for CC evaluation.
Add a TODO in `ercot_env.py` noting the UTC-day convention and deferring the fix.

If Narnia is unavailable or time is tight, Option B unblocks Day 2 method validation
immediately (MILP replay becomes verifiable) while Option A runs in the background.

---

## 5. Impact Assessment

| Artifact | Affected by UTC/CT bug | Notes |
|----------|----------------------|-------|
| M4 parquet (processed prices) | No | Prices are correct UTC timestamps throughout |
| Narnia NPZ (post-break train/val) | Yes | Episodes are 6-hour phase-shifted vs ERCOT CT day |
| Fleet benchmark ($24.93/kW-yr) | No | Computed from M4 parquet with CT-day convention |
| Stage 1 SAC training (pre-break) | Yes (same bug in ercot_env.py) | Lower severity — pre-break prices are less peaky |
| Stage 2 offline RL (Cal-QL, IQL) | Yes (if trained on current NPZ) | Phase-shifted expert data biases daily dispatch cycle |
| Eval harness (CT-day, correct) | No | Harness correctly uses CT days |
| MILP replay validation | Yes | Revenue target ($90,814) applies to UTC-day alignment, not CT |

**Action before Day 2 methods:** Do NOT train Methods 4/5 on the current NPZ until Option A is
completed (or Option B is applied as a stopgap). The UTC-day phase shift is particularly
harmful for energy arbitrage agents learning when to charge vs discharge.
