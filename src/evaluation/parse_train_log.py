"""
Parse stage1_train_v591.log to extract alpha, critic loss, and grad_c
trajectories at specific step windows around each instability event.

Also scans for NaN detections and crash/recovery events.

Usage:
  python -m src.evaluation.parse_train_log
  python -m src.evaluation.parse_train_log --log logs/stage1_train_v591.log
"""

import argparse
import re
import sys
from pathlib import Path

# Step windows to inspect (centered on collapse events)
WINDOWS = [
    ("300k collapse",  290_000,  310_000),
    ("425k collapse",  415_000,  435_000),
    ("550k collapse",  540_000,  565_000),
    ("675k collapse",  660_000,  715_000),
    ("750k collapse",  740_000,  765_000),
    ("900k collapse",  885_000,  960_000),
]


def parse_log(log_path: str):
    path = Path(log_path)
    if not path.exists():
        print(f"ERROR: log not found at {log_path}", file=sys.stderr)
        sys.exit(1)

    # ─── regex patterns ────────────────────────────────────────────────────
    # Log format:
    #   Step  50000/1000000 | ep=576 | critic=1.6814 | actor=-22.9588 |
    #   alpha=0.1423 | avg_reward=-8.9 | avg_soc=3.92 |
    #   grad_c=13.092 [q1=...] | grad_a=0.642 | ...
    step_re = re.compile(r'Step\s+(\d+)/', re.IGNORECASE)
    alpha_re = re.compile(r'\balpha=([0-9eE+\-.]+)')
    critic_re = re.compile(r'\bcritic=([0-9eE+\-.]+)')
    grad_c_re = re.compile(r'\bgrad_c=([0-9eE+\-.]+)')
    nan_re = re.compile(r'nan|inf(?!o)|nan_detected|nan_guard|explod', re.IGNORECASE)
    reward_re = re.compile(r'\bavg_reward=([0-9eE+\-\.]+)')
    actor_re = re.compile(r'\bactor=([0-9eE+\-\.]+)')
    grad_a_re = re.compile(r'\bgrad_a=([0-9eE+\-\.]+)')
    soc_re = re.compile(r'\bavg_soc=([0-9eE+\-\.]+)')

    records = []   # list of dicts, one per logged step
    nan_events = []

    current = {}
    with open(path, "r", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip()

            # NaN/crash detection
            if nan_re.search(line):
                m = step_re.search(line)
                s = int(m.group(1)) if m else None
                nan_events.append((lineno, s, line.strip()))

            # Step extraction — flush previous record when step changes
            m = step_re.search(line)
            if m:
                step = int(m.group(1))
                if current and current.get("step") != step:
                    records.append(current)
                    current = {}
                current["step"] = step

            # Field extraction
            for pattern, key in [
                (alpha_re, "alpha"),
                (critic_re, "critic_loss"),
                (grad_c_re, "grad_c"),
                (reward_re, "reward"),
                (actor_re, "actor_loss"),
                (grad_a_re, "grad_a"),
                (soc_re, "avg_soc"),
            ]:
                m = pattern.search(line)
                if m:
                    try:
                        current[key] = float(m.group(1))
                    except ValueError:
                        pass

    if current:
        records.append(current)

    return records, nan_events


def print_window(label: str, lo: int, hi: int, records: list):
    # Records are logged every 1000 steps; match on range
    subset = [r for r in records if lo <= r.get("step", -1) <= hi]
    if not subset:
        print(f"\n  [no records found in {lo}–{hi}]")
        return

    print(f"\n{'─'*72}")
    print(f"  {label}  (steps {lo:,} – {hi:,})")
    print(f"{'─'*72}")
    hdr = f"  {'step':>8}  {'alpha':>8}  {'critic':>10}  {'grad_c':>8}  {'avg_rew':>9}  {'actor':>9}  {'grad_a':>8}  {'avg_soc':>7}"
    print(hdr)
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*7}")
    for r in subset:
        step  = r.get("step", "?")
        alpha = f"{r['alpha']:.5f}"        if "alpha"       in r else "       ?"
        cl    = f"{r['critic_loss']:.4f}"  if "critic_loss" in r else "         ?"
        gc    = f"{r['grad_c']:.4f}"       if "grad_c"      in r else "       ?"
        rw    = f"{r['reward']:.2f}"       if "reward"      in r else "        ?"
        al    = f"{r['actor_loss']:.4f}"   if "actor_loss"  in r else "        ?"
        ga    = f"{r['grad_a']:.4f}"       if "grad_a"      in r else "       ?"
        sc    = f"{r['avg_soc']:.2f}"      if "avg_soc"     in r else "      ?"
        print(f"  {step:>8}  {alpha:>8}  {cl:>10}  {gc:>8}  {rw:>9}  {al:>9}  {ga:>8}  {sc:>7}")


def main(log_path: str):
    print(f"\nParsing: {log_path}")
    records, nan_events = parse_log(log_path)
    print(f"  Parsed {len(records)} step records, {len(nan_events)} NaN/crash events")

    # ─── NaN / crash events ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  NaN / Crash / Guard Events ({len(nan_events)} total)")
    print(f"{'='*72}")
    if not nan_events:
        print("  None found.")
    else:
        for lineno, step, text in nan_events[:60]:
            tag = f"step={step}" if step else f"line={lineno}"
            print(f"  [{tag}] {text[:110]}")
        if len(nan_events) > 60:
            print(f"  ... and {len(nan_events)-60} more")

    # ─── Per-window trajectories ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Alpha / Critic Loss / Grad Trajectories at Collapse Windows")
    print(f"{'='*72}")
    for label, lo, hi in WINDOWS:
        print_window(label, lo, hi, records)

    # ─── Alpha overall trajectory (every 25k steps) ────────────────────────
    print(f"\n\n{'='*72}")
    print(f"  Alpha at 25k-step checkpoints")
    print(f"{'='*72}")
    milestones = list(range(25_000, 1_025_000, 25_000))
    print(f"  {'step':>8}  {'alpha':>10}  {'avg_reward':>10}  {'critic':>12}  {'avg_soc':>7}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*7}")
    for ms in milestones:
        # find closest record within ±1500 steps
        near = [r for r in records if abs(r.get("step", -1e9) - ms) <= 1500]
        if near:
            r = min(near, key=lambda x: abs(x.get("step", 0) - ms))
            a  = f"{r['alpha']:.6f}"      if "alpha"       in r else "         ?"
            rw = f"{r['reward']:.2f}"     if "reward"      in r else "         ?"
            cl = f"{r['critic_loss']:.4f}" if "critic_loss" in r else "           ?"
            sc = f"{r['avg_soc']:.2f}"    if "avg_soc"     in r else "    ?"
            print(f"  {ms:>8}  {a:>10}  {rw:>10}  {cl:>12}  {sc:>6}")
        else:
            print(f"  {ms:>8}  {'(no record)':>10}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/stage1_train_v591.log")
    args = parser.parse_args()
    main(args.log)
