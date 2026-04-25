"""
Phase 1 runner: recompute T-60 reference baselines with joint 6D action space.

Produces:
  data/results/eval_tbx_energy_only/   — TBx energy-only (1D, apples-to-apples with v5.1)
  data/results/eval_tbx_with_as/       — TBx + AS when idle (6D, realistic post-RTC+B)
  data/results/eval_pf_t60/            — Perfect Foresight MIP oracle (6D, T-60 ceiling)
  data/results/baselines_t60/BASELINES_T60_REPORT.md

Sanity gates (stop-and-report on failure):
  PF > MILP-replay ceiling ($58.40/kW-yr)
  TBx_with_AS > TBx_energy_only
  All baselines > fleet median ($24.93/kW-yr)  [soft; PF must, TBx_energy may not]

Usage:
  cd /path/to/hybridbid
  python -m methods.baselines.run_t60_baselines
  python -m methods.baselines.run_t60_baselines --skip-pf   # TBx only (fast)
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.prepare_postbreak import (
    load_data,
    _find_t60_indices,
    evaluate,
    PRICE_COLS,
    P_MAX,
    FLEET_MEDIAN_KW_YR,
    FLEET_TOP_Q_KW_YR,
)
from methods.baselines.tbx_policy import (
    TBxEnergyOnlyPolicy,
    TBxWithASPolicy,
    calibrate_thresholds,
)
from methods.baselines.pf_policy import (
    PrecomputedPolicy,
    solve_pf_milp,
)

TRAIN_NPZ = ROOT / "data/expert_trajectories/receding_horizon_postbreak_train.npz"
RESULTS_DIR = str(ROOT / "data/results")
BASELINES_OUT = ROOT / "data/results/baselines_t60"

# Reference values — do NOT recompute
MILP_REPLAY_CEILING = 58.40   # $/kW-yr, commit 92c5a49 / eval_milp_replay_ct
MILP_REPLAY_TOTAL   = 86_394.20


def load_training_lmp() -> np.ndarray:
    """
    Load training-window RT LMP from the train NPZ for TBx threshold calibration.
    price_history[:, -1, 0] = current-step RT LMP for each training transition.
    """
    data = np.load(TRAIN_NPZ)
    rt_lmp_train = data["price_history"][:, -1, 0].astype(np.float64)
    print(
        f"[run_t60] Training RT LMP: N={len(rt_lmp_train)} "
        f"mean={rt_lmp_train.mean():.2f} "
        f"P25={np.percentile(rt_lmp_train, 25):.2f} "
        f"P75={np.percentile(rt_lmp_train, 75):.2f} "
        f"max={rt_lmp_train.max():.2f} $/MWh"
    )
    return rt_lmp_train


def load_t60_prices() -> tuple[np.ndarray, np.ndarray]:
    """
    Load T-60 window prices for PF MILP solve.

    Returns
    -------
    rt_lmp  : (15552,) float64 [$/MWh]
    rt_mcpc : (15552, 5) float64 [$/MWh], columns = [regup, regdn, rrs, ecrs, nsrs]
    """
    merged = load_data(str(ROOT / "data/processed"))
    start_idx, end_idx, _ = _find_t60_indices(merged)
    price_arr = merged[PRICE_COLS].values[start_idx:end_idx]   # (N, 12)
    rt_lmp  = price_arr[:, 0].astype(np.float64)
    rt_mcpc = price_arr[:, 1:6].astype(np.float64)   # cols: [regup, regdn, rrs, ecrs, nsrs]
    print(
        f"[run_t60] T-60 prices loaded: {len(rt_lmp)} steps "
        f"({len(rt_lmp) // 288} days)"
    )
    return rt_lmp, rt_mcpc


def check_sanity(summaries: dict[str, dict]) -> list[str]:
    """
    Run Phase 1 sanity gates. Returns list of FAIL strings (empty = all pass).
    """
    failures = []

    if "pf_t60" in summaries and "tbx_energy_only" in summaries:
        pf_kw  = summaries["pf_t60"]["all_days"]["annualized_kw_yr"]
        tbx_kw = summaries["tbx_energy_only"]["all_days"]["annualized_kw_yr"]

        # PF must beat MILP-replay ceiling
        if pf_kw <= MILP_REPLAY_CEILING:
            failures.append(
                f"GATE FAIL: PF ${pf_kw:.2f}/kW-yr ≤ MILP-replay ceiling "
                f"${MILP_REPLAY_CEILING:.2f}/kW-yr — formulation wrong or solver timeout"
            )

    if "tbx_with_as" in summaries and "tbx_energy_only" in summaries:
        as_kw  = summaries["tbx_with_as"]["all_days"]["annualized_kw_yr"]
        en_kw  = summaries["tbx_energy_only"]["all_days"]["annualized_kw_yr"]
        if as_kw <= en_kw:
            failures.append(
                f"GATE FAIL: TBx_with_AS ${as_kw:.2f}/kW-yr ≤ TBx_energy_only "
                f"${en_kw:.2f}/kW-yr — AS adds no revenue (pricing data issue?)"
            )

    # PF must beat fleet median (soft gate — TBx may not)
    if "pf_t60" in summaries:
        pf_kw = summaries["pf_t60"]["all_days"]["annualized_kw_yr"]
        if pf_kw <= FLEET_MEDIAN_KW_YR:
            failures.append(
                f"GATE FAIL: PF ${pf_kw:.2f}/kW-yr ≤ fleet median "
                f"${FLEET_MEDIAN_KW_YR:.2f}/kW-yr — oracle should dominate fleet"
            )

    return failures


def write_report(summaries: dict[str, dict], sanity_failures: list[str], pf_meta: dict = None) -> None:
    BASELINES_OUT.mkdir(parents=True, exist_ok=True)
    report_path = BASELINES_OUT / "BASELINES_T60_REPORT.md"

    def _row(name: str) -> str:
        if name not in summaries:
            return f"| {name} | — | — | — | — | — |"
        s = summaries[name]
        a  = s["all_days"]
        ex = s["ex_fern"]
        fern = s["fern_only"]
        kw = a["annualized_kw_yr"]
        vs_med  = round((kw / FLEET_MEDIAN_KW_YR - 1) * 100, 1)
        vs_ceil = round((kw / MILP_REPLAY_CEILING - 1) * 100, 1)
        return (
            f"| {name} "
            f"| {kw:.2f} "
            f"| {ex['annualized_kw_yr']:.2f} "
            f"| {fern['annualized_kw_yr']:.2f} "
            f"| {vs_med:+.1f}% "
            f"| {vs_ceil:+.1f}% |"
        )

    lines = [
        "# Baselines T-60 Report",
        f"**Date:** {date.today()}",
        "**Window:** 2026-01-01 → 2026-02-23 (54 days, CT-aligned)",
        "**Session:** cc-baselines (Phase 1)",
        "",
        "## Reference Ceilings",
        "",
        f"- MILP-replay continuous-SoC: **{MILP_REPLAY_CEILING:.2f} $/kW-yr** (${MILP_REPLAY_TOTAL:,.2f} total, commit 92c5a49)",
        f"- Fleet median (T-60 window): {FLEET_MEDIAN_KW_YR:.2f} $/kW-yr",
        f"- Fleet top-quartile: {FLEET_TOP_Q_KW_YR:.2f} $/kW-yr",
        "",
        "## Results",
        "",
        "| Baseline | All-days $/kW-yr | Ex-Fern $/kW-yr | Fern-only $/kW-yr | vs Fleet Median | vs MILP-replay ceiling |",
        "|----------|-----------------|-----------------|-------------------|-----------------|------------------------|",
        _row("tbx_energy_only"),
        _row("tbx_with_as"),
        _row("pf_t60"),
        "",
        "## Revenue Composition",
        "",
    ]

    for name, s in summaries.items():
        total = s["energy_rev_total"] + s["as_rev_total"]
        as_share = f"{s['as_rev_total']/total*100:.1f}%" if total != 0 else "N/A"
        lines += [
            f"### {name}",
            f"- Energy revenue: ${s['energy_rev_total']:>10,.2f}",
            f"- AS revenue:     ${s['as_rev_total']:>10,.2f}  ({as_share})",
            "",
        ]

    if pf_meta:
        lines += [
            "## PF Solve Info",
            "",
            f"- Approach: {pf_meta.get('approach', 'unknown')}",
        ]
        if pf_meta.get("approach") == "full_horizon":
            lines.append(f"- Solve time: {pf_meta.get('solve_time', 0):.1f}s")
            lines.append(f"- LP revenue (pre-projection): ${pf_meta.get('revenue', 0):,.2f}")
        elif pf_meta.get("approach") == "weekly_fallback":
            lines.append(f"- Weeks: {pf_meta.get('n_weeks', '?')}")
            lines.append(f"- Statuses: {pf_meta.get('week_statuses', [])}")
            lines.append(f"- Total revenue: ${pf_meta.get('total_revenue', 0):,.2f}")
        lines.append("")

    lines += [
        "## Sanity Checks",
        "",
        "Gate: PF > MILP-replay ceiling ($58.40/kW-yr): " + (
            "**PASS**" if "pf_t60" in summaries and
            summaries["pf_t60"]["all_days"]["annualized_kw_yr"] > MILP_REPLAY_CEILING
            else "**FAIL** — see failures below"
        ),
        "Gate: TBx_with_AS > TBx_energy_only: " + (
            "**PASS**" if "tbx_with_as" in summaries and "tbx_energy_only" in summaries and
            summaries["tbx_with_as"]["all_days"]["annualized_kw_yr"] >
            summaries["tbx_energy_only"]["all_days"]["annualized_kw_yr"]
            else "**FAIL**"
        ),
        "",
    ]

    if sanity_failures:
        lines += ["## ⚠ SANITY FAILURES", ""]
        for f in sanity_failures:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("All sanity checks passed.")

    lines += [
        "",
        "## Reward Convention Note",
        "",
        "Baseline revenues are computed by running policies through the eval harness "
        "(`experiments/prepare_postbreak.py`), which computes:",
        "  `energy_rev = p_energy_mw * rt_lmp * DT`",
        "  `as_rev = c_as_mw * rt_mcpc * DT`",
        "Physical $ throughout. The stored `rewards` field of the trajectory NPZ is never "
        "read by this code (TBx is purely rule-based; PF re-solves its own LP from market prices).",
        "The mixed-unit convention in stored rewards does not affect these numbers.",
    ]

    report_path.write_text("\n".join(lines) + "\n")
    print(f"[run_t60] Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute T-60 reference baselines (Phase 1)")
    parser.add_argument("--skip-pf", action="store_true", help="Skip Perfect Foresight (fast TBx-only run)")
    args = parser.parse_args()

    print("[run_t60] === Phase 1: T-60 reference baselines ===")
    t0 = time.time()

    # ── TBx threshold calibration ──────────────────────────────────────────
    rt_lmp_train = load_training_lmp()
    p_low, p_high = calibrate_thresholds(rt_lmp_train)
    print(f"[run_t60] TBx thresholds: p_low={p_low:.2f} p_high={p_high:.2f} $/MWh")

    # ── Evaluate TBx policies ──────────────────────────────────────────────
    summaries = {}

    print("\n[run_t60] --- TBx energy-only ---")
    tbx_energy = TBxEnergyOnlyPolicy(p_low=p_low, p_high=p_high)
    summaries["tbx_energy_only"] = evaluate(
        tbx_energy, method_name="tbx_energy_only",
        data_dir=str(ROOT / "data/processed"),
        results_dir=RESULTS_DIR,
    )

    print("\n[run_t60] --- TBx with AS ---")
    tbx_as = TBxWithASPolicy(p_low=p_low, p_high=p_high)
    summaries["tbx_with_as"] = evaluate(
        tbx_as, method_name="tbx_with_as",
        data_dir=str(ROOT / "data/processed"),
        results_dir=RESULTS_DIR,
    )

    # ── Perfect Foresight MIP ──────────────────────────────────────────────
    pf_meta = {}
    if not args.skip_pf:
        print("\n[run_t60] --- Perfect Foresight MIP (T-60 oracle) ---")
        rt_lmp_t60, rt_mcpc_t60 = load_t60_prices()
        actions_mw, pf_meta = solve_pf_milp(rt_lmp_t60, rt_mcpc_t60)

        pf_policy = PrecomputedPolicy(actions_mw)
        summaries["pf_t60"] = evaluate(
            pf_policy, method_name="pf_t60",
            data_dir=str(ROOT / "data/processed"),
            results_dir=RESULTS_DIR,
        )
    else:
        print("[run_t60] Skipping PF (--skip-pf)")

    # ── Sanity gates ───────────────────────────────────────────────────────
    print("\n[run_t60] === Sanity checks ===")
    failures = check_sanity(summaries)
    if failures:
        for f in failures:
            print(f"  *** {f}")
        print("\n[run_t60] STOP: sanity gate failure. Do not proceed to Phase 2.")
        sys.exit(1)
    else:
        print("  All sanity gates passed.")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n[run_t60] === Summary ===")
    print(f"{'Baseline':<22} {'All $/kW-yr':>12} {'Ex-Fern':>10} {'Fern $/kW-yr':>14} {'vs Median':>10} {'vs Ceiling':>11}")
    print("-" * 82)
    for name, s in summaries.items():
        a  = s["all_days"]
        ex = s["ex_fern"]
        fern = s["fern_only"]
        kw = a["annualized_kw_yr"]
        vs_med  = round((kw / FLEET_MEDIAN_KW_YR - 1) * 100, 1)
        vs_ceil = round((kw / MILP_REPLAY_CEILING - 1) * 100, 1)
        print(f"  {name:<20} {kw:>12.2f} {ex['annualized_kw_yr']:>10.2f} {fern['annualized_kw_yr']:>14.2f} {vs_med:>+10.1f}% {vs_ceil:>+10.1f}%")

    # ── Write report ───────────────────────────────────────────────────────
    write_report(summaries, failures, pf_meta)

    elapsed = time.time() - t0
    print(f"\n[run_t60] Phase 1 complete in {elapsed:.1f}s")
    print("[run_t60] STOP: Awaiting Karthik's green-light for Phase 2 (BC).")


if __name__ == "__main__":
    main()
