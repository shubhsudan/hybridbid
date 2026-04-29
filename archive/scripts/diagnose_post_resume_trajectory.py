"""
Post-resume gradient trajectory diagnostic.

Reads the v2 (stop-gradient fix, resumed from 50k) training log and reports:
  1. grad_a_pre trajectory from step 50k to kill, bucketed every 5k steps
  2. Linear-regression slope of log(grad_a) vs step for spikes > 50
  3. Verdict: DECAYING / GROWING / FLAT
  4. Plot: experiments/diagnose_post_resume_trajectory.png

Regex adapted to our log format (grad_a=PRE→POST, critic=, grad_t=N [proj=X attn=Y]).
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

import sys
LOG_PATH = Path(sys.argv[1] if len(sys.argv) > 1
                else "experiments/logs/tier1_seed42_v2.log")
print(f"Analyzing: {LOG_PATH}\n")

rows = []
with open(LOG_PATH) as f:
    for line in f:
        if not line.startswith("Step"):
            continue
        m_step = re.search(r"Step\s+(\d+)", line)
        m_grad_a = re.search(r"grad_a=([0-9.]+)→", line)
        m_qmax = re.search(r"q_exp_maxabs=([0-9.]+)", line)
        m_attn = re.search(r"attn=([0-9.]+)\]", line)
        m_critic = re.search(r"critic=([0-9.]+)", line)
        m_bin = re.search(r"bin_ent=([0-9.]+)", line)
        m_grad_c = re.search(r"grad_c=([0-9.]+)→", line)
        if not all([m_step, m_grad_a, m_qmax, m_attn, m_critic, m_bin, m_grad_c]):
            continue
        rows.append({
            "step": int(m_step.group(1)),
            "grad_a_pre": float(m_grad_a.group(1)),
            "q_exp_maxabs": float(m_qmax.group(1)),
            "grad_ttfe_attn": float(m_attn.group(1)),
            "critic_loss": float(m_critic.group(1)),
            "bin_ent": float(m_bin.group(1)),
            "grad_c_pre": float(m_grad_c.group(1)),
        })

df = pd.DataFrame(rows)
df_post = df[df["step"] >= 50000].copy()
df_post["steps_since_resume"] = df_post["step"] - 50000
print(f"Parsed {len(df_post)} log entries from step 50k onward.")
print(f"Step range: {df_post['step'].min()} → {df_post['step'].max()}")

# 5k-step buckets
df_post["bucket"] = (df_post["steps_since_resume"] // 5000) * 5000
bucket_stats = df_post.groupby("bucket").agg(
    n_logs=("grad_a_pre", "count"),
    max_grad_a=("grad_a_pre", "max"),
    mean_grad_a=("grad_a_pre", "mean"),
    spike_count=("grad_a_pre", lambda x: (x > 100).sum()),
    mean_q_maxabs=("q_exp_maxabs", "mean"),
    mean_ttfe_attn=("grad_ttfe_attn", "mean"),
)
print("\nPost-resume trajectory by 5k-step bucket:")
print(bucket_stats.to_string())

# Spike trend regression
spikes = df_post[df_post["grad_a_pre"] > 50].copy()
print(f"\nSpikes > 50 count: {len(spikes)}")
slope = intercept = r = None
if len(spikes) >= 3:
    slope, intercept, r, p, stderr = linregress(
        spikes["steps_since_resume"].values, np.log(spikes["grad_a_pre"].values)
    )
    print(f"\nSpike magnitude trend (log grad_a ~ step):")
    print(f"  slope     = {slope:+.3e}")
    print(f"  intercept = {intercept:+.3f}")
    print(f"  R         = {r:+.3f}")
    print(f"  p-value   = {p:.4f}")
    print(f"  n spikes  = {len(spikes)}")

    # Half-life if decaying
    if slope < 0:
        half_life = -np.log(2) / slope
        print(f"  half-life = {half_life:.0f} steps")

    if slope < -1e-4:
        verdict = "DECAYING"
        detail = "consistent with stale Adam momentum washing out"
    elif slope > 1e-4:
        verdict = "GROWING"
        detail = "fix genuinely insufficient"
    else:
        verdict = "FLAT"
        detail = "ambiguous — could be either"
    print(f"\nVERDICT: {verdict} — {detail}")
else:
    print("\nInsufficient spikes (<3) for regression.")

# List every spike > 50
print(f"\nAll spikes (grad_a_pre > 50):")
print(f"   {'step':>7} | {'post-resume':>11} | {'grad_a':>8} | {'q_maxabs':>8} | {'ttfe_attn':>9}")
print(f"   {'-'*7}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")
for _, r_ in spikes.iterrows():
    print(f"   {int(r_['step']):>7} | {int(r_['steps_since_resume']):>11} | "
          f"{r_['grad_a_pre']:>8.1f} | {r_['q_exp_maxabs']:>8.1f} | {r_['grad_ttfe_attn']:>9.3f}")

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

ax = axes[0]
ax.semilogy(df_post["steps_since_resume"], df_post["grad_a_pre"].clip(lower=0.01),
            "o-", markersize=4, alpha=0.7, color="steelblue")
ax.axhline(100, color="red", linestyle="--", label="kill threshold (100)")
ax.axhline(50, color="orange", linestyle=":", label="spike threshold (50)")
# Overlay regression line if computed
if slope is not None and len(spikes) >= 3:
    x = np.array([0, df_post["steps_since_resume"].max()])
    y = np.exp(slope * x + intercept)
    ax.semilogy(x, y, "g-", linewidth=2, alpha=0.7,
                label=f"log-slope fit: {slope:+.2e} ({verdict})")
ax.set_ylabel("grad_a_pre (log scale)")
ax.set_title(f"Post-resume actor gradient trajectory (v2, stop-gradient fix, resumed from 50k)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(df_post["steps_since_resume"], df_post["q_exp_maxabs"], "b-", alpha=0.7)
ax.set_ylabel("q_exp_maxabs")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(df_post["steps_since_resume"], df_post["grad_ttfe_attn"], "m-", alpha=0.7)
ax.set_ylabel("grad_ttfe_attn")
ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(df_post["steps_since_resume"], df_post["critic_loss"], "c-", alpha=0.7, label="critic_loss")
ax2 = ax.twinx()
ax2.plot(df_post["steps_since_resume"], df_post["bin_ent"], "orange", alpha=0.6, label="bin_ent")
ax.set_ylabel("critic_loss", color="c")
ax2.set_ylabel("bin_ent", color="orange")
ax.set_xlabel("steps since resume (0 = step 50000)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path("experiments/diagnose_post_resume_trajectory.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=100)
print(f"\nPlot saved: {out_path}")
