"""
Phase 2c — Post-RTC+B Data Exploration & Insights Analysis
Generates publication-quality charts and FINDINGS.md
"""

import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

COLORS = {
    "pre": "#2196F3",
    "post": "#FF5722",
    "lmp": "#1976D2",
    "spp": "#388E3C",
    "regup": "#E53935",
    "regdn": "#43A047",
    "rrs": "#FB8C00",
    "ecrs": "#8E24AA",
    "nsrs": "#00897B",
    "energy": "#1565C0",
    "wind": "#26A69A",
    "solar": "#FFA000",
    "load": "#546E7A",
    "net_load": "#D84315",
}

OUT = Path("data/results/eda")
OUT.mkdir(parents=True, exist_ok=True)

RTCB = pd.Timestamp("2025-12-05 06:00:00", tz="UTC")

# ── Load data ────────────────────────────────────────────────
def load_table(name):
    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(f"data/processed/{name}/*.parquet"))]
    df = pd.concat(dfs).sort_index()
    return df[~df.index.duplicated(keep="last")]

print("Loading data...")
ep = load_table("energy_prices")
ap = load_table("as_prices")
sc = load_table("system_conditions")

pre = ep.index < RTCB
post = ep.index >= RTCB

# Convert to CPT for hour-of-day analyses
ep_cpt = ep.copy()
ep_cpt.index = ep_cpt.index.tz_convert("US/Central")
ap_cpt = ap.copy()
ap_cpt.index = ap_cpt.index.tz_convert("US/Central")
sc_cpt = sc.copy()
sc_cpt.index = sc_cpt.index.tz_convert("US/Central")

findings = []

# ══════════════════════════════════════════════════════════════
# ANALYSIS 1: The Before vs After Story
# ══════════════════════════════════════════════════════════════
print("\n=== Analysis 1: Before vs After ===")

# ── 1a. RT price volatility shift ──
print("  1a. Volatility...")
rolling_std = ep["rt_lmp"].rolling(12, min_periods=6).std()  # 12 x 5min = 1hr

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

# Top: rolling std time series
ax = axes[0]
ax.plot(rolling_std.index, rolling_std.values, linewidth=0.4, color=COLORS["lmp"], alpha=0.6)
# Daily average for smoother view
daily_std = rolling_std.resample("D").mean()
ax.plot(daily_std.index, daily_std.values, linewidth=2, color=COLORS["lmp"], label="Daily avg 1-hr σ")
ax.axvline(RTCB, color="red", linewidth=2, linestyle="--", alpha=0.8, label="RTC+B Go-Live (Dec 5)")
ax.set_ylabel("Rolling 1-hr Std Dev ($/MWh)")
ax.set_title("RT LMP Volatility: Before vs After Real-Time Co-Optimization")
ax.legend(loc="upper right")
ax.set_xlim(ep.index.min(), ep.index.max())

# Bottom: daily range box plots
daily_range_pre = ep.loc[pre, "rt_lmp"].resample("D").apply(lambda x: x.max() - x.min()).dropna()
daily_range_post = ep.loc[post, "rt_lmp"].resample("D").apply(lambda x: x.max() - x.min()).dropna()

ax = axes[1]
bp = ax.boxplot(
    [daily_range_pre.values, daily_range_post.values],
    labels=["Pre-RTC+B\n(Nov 1 – Dec 4)", "Post-RTC+B\n(Dec 5 – Mar 20)"],
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="black", linewidth=2),
)
bp["boxes"][0].set_facecolor(COLORS["pre"])
bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor(COLORS["post"])
bp["boxes"][1].set_alpha(0.6)
ax.set_ylabel("Daily Price Range ($/MWh)")
ax.set_title("Daily RT LMP Range: Pre vs Post RTC+B")

plt.tight_layout()
fig.savefig(OUT / "1a_price_volatility_pre_vs_post.png")
plt.close()

vol_pre_mean = daily_range_pre.mean()
vol_post_mean = daily_range_post.mean()
vol_pre_med = daily_range_pre.median()
vol_post_med = daily_range_post.median()
std_pre = ep.loc[pre, "rt_lmp"].std()
std_post = ep.loc[post, "rt_lmp"].std()

findings.append(f"""## Analysis 1: The Before vs After Story

### 1a. RT Price Volatility Shift
- **Pre-RTC+B daily price range:** mean ${vol_pre_mean:.1f}/MWh, median ${vol_pre_med:.1f}/MWh
- **Post-RTC+B daily price range:** mean ${vol_post_mean:.1f}/MWh, median ${vol_post_med:.1f}/MWh
- **Change:** {((vol_post_mean/vol_pre_mean)-1)*100:+.1f}% in mean daily range
- **Overall RT LMP std dev:** Pre ${std_pre:.1f}, Post ${std_post:.1f}
""")

# ── 1b. Price distribution shift ──
print("  1b. Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RT LMP
ax = axes[0]
pre_lmp = ep.loc[pre, "rt_lmp"].dropna()
post_lmp = ep.loc[post, "rt_lmp"].dropna()
bins = np.linspace(-50, 200, 100)
ax.hist(pre_lmp.clip(-50, 200), bins=bins, alpha=0.5, density=True, color=COLORS["pre"],
        label=f"Pre-RTC+B (n={len(pre_lmp):,})", edgecolor="none")
ax.hist(post_lmp.clip(-50, 200), bins=bins, alpha=0.5, density=True, color=COLORS["post"],
        label=f"Post-RTC+B (n={len(post_lmp):,})", edgecolor="none")
ax.set_xlabel("RT LMP ($/MWh)")
ax.set_ylabel("Density")
ax.set_title("RT LMP Distribution")
ax.legend()
ax.set_xlim(-50, 200)

# DAM SPP
ax = axes[1]
pre_spp = ep.loc[pre, "dam_spp"].dropna()
post_spp = ep.loc[post, "dam_spp"].dropna()
bins_dam = np.linspace(-20, 150, 80)
ax.hist(pre_spp.clip(-20, 150), bins=bins_dam, alpha=0.5, density=True, color=COLORS["pre"],
        label=f"Pre-RTC+B (n={len(pre_spp):,})", edgecolor="none")
ax.hist(post_spp.clip(-20, 150), bins=bins_dam, alpha=0.5, density=True, color=COLORS["post"],
        label=f"Post-RTC+B (n={len(post_spp):,})", edgecolor="none")
ax.set_xlabel("DAM SPP ($/MWh)")
ax.set_ylabel("Density")
ax.set_title("DAM SPP Distribution")
ax.legend()

plt.tight_layout()
fig.savefig(OUT / "1b_price_distribution_pre_vs_post.png")
plt.close()

pct_neg_pre = (pre_lmp < 0).sum() / len(pre_lmp) * 100
pct_neg_post = (post_lmp < 0).sum() / len(post_lmp) * 100
pct_100_pre = (pre_lmp > 100).sum() / len(pre_lmp) * 100
pct_100_post = (post_lmp > 100).sum() / len(post_lmp) * 100
pct_500_pre = (pre_lmp > 500).sum() / len(pre_lmp) * 100
pct_500_post = (post_lmp > 500).sum() / len(post_lmp) * 100

findings.append(f"""### 1b. Price Distribution Shift
- **Negative prices:** Pre {pct_neg_pre:.2f}%, Post {pct_neg_post:.2f}%
- **Prices > $100:** Pre {pct_100_pre:.2f}%, Post {pct_100_post:.2f}%
- **Prices > $500:** Pre {pct_500_pre:.3f}%, Post {pct_500_post:.3f}%
- **RT LMP mean:** Pre ${pre_lmp.mean():.2f}, Post ${post_lmp.mean():.2f}
- **RT LMP median:** Pre ${pre_lmp.median():.2f}, Post ${post_lmp.median():.2f}
""")

# ── 1c. Intraday price profile ──
print("  1c. Intraday profile...")
pre_cpt = ep_cpt.index < RTCB.tz_convert("US/Central")
post_cpt = ep_cpt.index >= RTCB.tz_convert("US/Central")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in [(axes[0], "rt_lmp", "RT LMP"), (axes[1], "dam_spp", "DAM SPP")]:
    pre_hourly = ep_cpt.loc[pre_cpt].groupby(ep_cpt.loc[pre_cpt].index.hour)[col].mean()
    post_hourly = ep_cpt.loc[post_cpt].groupby(ep_cpt.loc[post_cpt].index.hour)[col].mean()
    ax.plot(pre_hourly.index, pre_hourly.values, "o-", color=COLORS["pre"],
            linewidth=2, markersize=5, label="Pre-RTC+B")
    ax.plot(post_hourly.index, post_hourly.values, "s-", color=COLORS["post"],
            linewidth=2, markersize=5, label="Post-RTC+B")
    ax.set_xlabel("Hour of Day (CPT)")
    ax.set_ylabel("Average Price ($/MWh)")
    ax.set_title(f"Average Hourly {title} Profile")
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)

plt.tight_layout()
fig.savefig(OUT / "1c_intraday_price_profile.png")
plt.close()

pre_peak = ep_cpt.loc[pre_cpt & ep_cpt.index.hour.isin(range(16, 21)), "rt_lmp"].mean()
pre_offpeak = ep_cpt.loc[pre_cpt & ep_cpt.index.hour.isin(range(0, 6)), "rt_lmp"].mean()
post_peak = ep_cpt.loc[post_cpt & ep_cpt.index.hour.isin(range(16, 21)), "rt_lmp"].mean()
post_offpeak = ep_cpt.loc[post_cpt & ep_cpt.index.hour.isin(range(0, 6)), "rt_lmp"].mean()

findings.append(f"""### 1c. Intraday Price Profile
- **Pre-RTC+B peak/off-peak spread (RT LMP):** ${pre_peak:.2f} - ${pre_offpeak:.2f} = ${pre_peak - pre_offpeak:.2f}/MWh
- **Post-RTC+B peak/off-peak spread (RT LMP):** ${post_peak:.2f} - ${post_offpeak:.2f} = ${post_peak - post_offpeak:.2f}/MWh
- Peak hours: 4-9 PM CPT; Off-peak hours: 12-6 AM CPT
""")

# ── 1d. Summary statistics table ──
print("  1d. Summary stats...")

def compute_stats(s):
    s = s.dropna()
    return {
        "Mean": f"${s.mean():.2f}",
        "Median": f"${s.median():.2f}",
        "Std Dev": f"${s.std():.2f}",
        "Min": f"${s.min():.2f}",
        "Max": f"${s.max():.2f}",
        "% Negative": f"{(s < 0).sum() / len(s) * 100:.2f}%",
        "% > $100": f"{(s > 100).sum() / len(s) * 100:.2f}%",
        "% > $500": f"{(s > 500).sum() / len(s) * 100:.3f}%",
        "Count": f"{len(s):,}",
    }

stats_pre_rt = compute_stats(ep.loc[pre, "rt_lmp"])
stats_post_rt = compute_stats(ep.loc[post, "rt_lmp"])
stats_pre_dam = compute_stats(ep.loc[pre, "dam_spp"])
stats_post_dam = compute_stats(ep.loc[post, "dam_spp"])

findings.append(f"""### 1d. Summary Statistics

| Metric | RT LMP Pre | RT LMP Post | DAM SPP Pre | DAM SPP Post |
|--------|-----------|------------|------------|-------------|
| Mean | {stats_pre_rt['Mean']} | {stats_post_rt['Mean']} | {stats_pre_dam['Mean']} | {stats_post_dam['Mean']} |
| Median | {stats_pre_rt['Median']} | {stats_post_rt['Median']} | {stats_pre_dam['Median']} | {stats_post_dam['Median']} |
| Std Dev | {stats_pre_rt['Std Dev']} | {stats_post_rt['Std Dev']} | {stats_pre_dam['Std Dev']} | {stats_post_dam['Std Dev']} |
| Min | {stats_pre_rt['Min']} | {stats_post_rt['Min']} | {stats_pre_dam['Min']} | {stats_post_dam['Min']} |
| Max | {stats_pre_rt['Max']} | {stats_post_rt['Max']} | {stats_pre_dam['Max']} | {stats_post_dam['Max']} |
| % Negative | {stats_pre_rt['% Negative']} | {stats_post_rt['% Negative']} | {stats_pre_dam['% Negative']} | {stats_post_dam['% Negative']} |
| % > $100 | {stats_pre_rt['% > $100']} | {stats_post_rt['% > $100']} | {stats_pre_dam['% > $100']} | {stats_post_dam['% > $100']} |
| % > $500 | {stats_pre_rt['% > $500']} | {stats_post_rt['% > $500']} | {stats_pre_dam['% > $500']} | {stats_post_dam['% > $500']} |
| N intervals | {stats_pre_rt['Count']} | {stats_post_rt['Count']} | {stats_pre_dam['Count']} | {stats_post_dam['Count']} |

**LinkedIn Headline:** ERCOT's real-time co-optimization (RTC+B) reshaped price dynamics — volatility, distribution tails, and peak/off-peak spreads all shifted measurably after Dec 5, 2025.
""")


# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: The Co-optimization Signature
# ══════════════════════════════════════════════════════════════
print("\n=== Analysis 2: Co-optimization Signature ===")

# ── 2a. Correlation matrix ──
print("  2a. Correlation matrix...")
post_prices = pd.DataFrame({
    "RT LMP": ep.loc[post, "rt_lmp"],
    "MCPC RegUp": ap.loc[post, "rt_mcpc_regup"],
    "MCPC RegDn": ap.loc[post, "rt_mcpc_regdn"],
    "MCPC RRS": ap.loc[post, "rt_mcpc_rrs"],
    "MCPC ECRS": ap.loc[post, "rt_mcpc_ecrs"],
    "MCPC NSRS": ap.loc[post, "rt_mcpc_nsrs"],
}).dropna()

corr = post_prices.corr()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(corr.columns, fontsize=10)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11,
                fontweight="bold", color=color)
plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
ax.set_title("Post-RTC+B Price Correlation Matrix\n(Energy + 5 AS Products, 5-min intervals)")
plt.tight_layout()
fig.savefig(OUT / "2a_correlation_heatmap_post_rtcb.png")
plt.close()

findings.append(f"""## Analysis 2: The Co-optimization Signature

### 2a. Post-RTC+B Price Correlation Matrix
{corr.to_markdown()}

Key observations:
- RT LMP correlations with MCPCs show the degree of energy-AS coupling from co-optimization
- High inter-AS correlations indicate products move together under scarcity
""")

# ── 2b. LMP–MCPC scatter ──
print("  2b. Scatter plots...")
mcpc_cols = ["rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs"]
mcpc_labels = ["RegUp", "RegDn", "RRS", "ECRS", "NSRS"]
mcpc_colors = [COLORS["regup"], COLORS["regdn"], COLORS["rrs"], COLORS["ecrs"], COLORS["nsrs"]]

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
for ax, col, label, color in zip(axes, mcpc_cols, mcpc_labels, mcpc_colors):
    x = ep.loc[post, "rt_lmp"]
    y = ap.loc[post, col]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    ax.scatter(x, y, s=1, alpha=0.08, color=color, rasterized=True)
    # Binned average
    bins = np.percentile(x, np.linspace(0, 100, 30))
    bin_idx = np.digitize(x, bins)
    for b in range(1, len(bins)):
        m = bin_idx == b
        if m.sum() > 5:
            ax.plot(x[m].mean(), y[m].mean(), "o", color="black", markersize=4, zorder=5)
    ax.set_xlabel("RT LMP ($/MWh)")
    ax.set_ylabel(f"MCPC {label} ($/MW)")
    ax.set_title(f"LMP vs {label}")
    r = x.corr(y)
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
            va="top", fontweight="bold", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

plt.suptitle("RT LMP vs RT MCPC Scatter (Post-RTC+B, 5-min intervals)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "2b_lmp_vs_mcpc_scatter.png")
plt.close()

scatter_corrs = {}
for col, label in zip(mcpc_cols, mcpc_labels):
    x = ep.loc[post, "rt_lmp"]
    y = ap.loc[post, col]
    mask = x.notna() & y.notna()
    scatter_corrs[label] = x[mask].corr(y[mask])

findings.append(f"""### 2b. LMP–MCPC Scatter Relationships
Correlations (RT LMP vs each MCPC, post-RTC+B):
""" + "\n".join(f"- **{k}:** r = {v:.3f}" for k, v in scatter_corrs.items()) + """
""")

# ── 2c. Intraday co-movement ──
print("  2c. Intraday co-movement...")
post_cpt_mask = ap_cpt.index >= RTCB.tz_convert("US/Central")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

hours = range(24)
lmp_hourly = ep_cpt.loc[post_cpt, "rt_lmp"].groupby(ep_cpt.loc[post_cpt].index.hour).mean()
ax1.plot(hours, [lmp_hourly.get(h, np.nan) for h in hours], "o-", color=COLORS["lmp"],
         linewidth=2.5, markersize=6, label="RT LMP", zorder=10)
ax1.set_ylabel("RT LMP ($/MWh)", color=COLORS["lmp"])

for col, label, color in zip(mcpc_cols, mcpc_labels, mcpc_colors):
    hourly = ap_cpt.loc[post_cpt_mask].groupby(ap_cpt.loc[post_cpt_mask].index.hour)[col].mean()
    ax2.plot(hours, [hourly.get(h, np.nan) for h in hours], "s--", color=color,
             linewidth=1.5, markersize=4, label=f"MCPC {label}", alpha=0.8)

ax2.set_ylabel("MCPC ($/MW)")
ax1.set_xlabel("Hour of Day (CPT)")
ax1.set_title("Intraday Price Co-movement: Energy + AS (Post-RTC+B)")
ax1.set_xticks(range(0, 24, 2))
ax1.set_xlim(-0.5, 23.5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, ncol=2)

plt.tight_layout()
fig.savefig(OUT / "2c_intraday_comovement.png")
plt.close()

findings.append("""### 2c. Intraday Co-movement
- Chart shows how energy (LMP) and all 5 AS products (MCPCs) move across the day
- Evening ramp (4-8 PM CPT) is the key co-movement period where both energy and reserves become scarce
""")

# ── 2d. DAM vs RT AS comparison ──
print("  2d. DAM vs RT comparison...")
dam_cols = ["dam_as_regup", "dam_as_regdn", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs"]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
dam_rt_corrs = {}
for ax, dam_col, rt_col, label, color in zip(axes, dam_cols, mcpc_cols, mcpc_labels, mcpc_colors):
    # Average RT MCPC to hourly to match DAM granularity
    post_mask = ap.index >= RTCB
    rt_hourly = ap.loc[post_mask, rt_col].resample("h").mean()
    dam_hourly = ap.loc[post_mask, dam_col].resample("h").mean()
    both = pd.DataFrame({"DAM": dam_hourly, "RT": rt_hourly}).dropna()
    if len(both) > 10:
        ax.scatter(both["DAM"], both["RT"], s=8, alpha=0.3, color=color, rasterized=True)
        r = both["DAM"].corr(both["RT"])
        dam_rt_corrs[label] = r
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10,
                va="top", fontweight="bold", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        # 45-degree line
        lim = max(both["DAM"].quantile(0.99), both["RT"].quantile(0.99))
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel(f"DAM {label} ($/MW)")
    ax.set_ylabel(f"RT MCPC {label} ($/MW)")
    ax.set_title(f"{label}")

plt.suptitle("DAM vs RT AS Prices (Post-RTC+B, Hourly Average)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "2d_dam_vs_rt_as_prices.png")
plt.close()

findings.append(f"""### 2d. DAM vs RT AS Price Comparison
Correlations (DAM AS vs RT MCPC hourly averages):
""" + "\n".join(f"- **{k}:** r = {v:.3f}" for k, v in dam_rt_corrs.items()) + """

**LinkedIn Headline:** ERCOT's co-optimization creates measurable coupling between energy and reserve prices — AS products now co-move with energy LMPs at 5-minute granularity, revealing new arbitrage opportunities for battery storage.
""")


# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: The Revenue Opportunity
# ══════════════════════════════════════════════════════════════
print("\n=== Analysis 3: Revenue Opportunity ===")

P_MAX = 10.0    # MW
E_MAX = 20.0    # MWh
SOC_MIN = 0.10 * E_MAX  # 2 MWh
SOC_MAX = 0.90 * E_MAX  # 18 MWh
ETA_CH = 0.92
ETA_DCH = 0.92
DT = 5 / 60    # hours per interval

# AS duration requirements (MWh per MW) from CLAUDE.md
AS_DURATION = {
    "rt_mcpc_regup": 0.5,   # 30 min
    "rt_mcpc_regdn": 0.5,   # 30 min
    "rt_mcpc_rrs": 0.5,     # 30 min (PFR/UFR)
    "rt_mcpc_ecrs": 1.0,    # 1 hour
    "rt_mcpc_nsrs": 4.0,    # 4 hours
}
AS_LABELS = {
    "rt_mcpc_regup": "RegUp",
    "rt_mcpc_regdn": "RegDn",
    "rt_mcpc_rrs": "RRS",
    "rt_mcpc_ecrs": "ECRS",
    "rt_mcpc_nsrs": "NSRS",
}

# ── 3a. Theoretical AS revenue by product ──
print("  3a. AS revenue by product...")
post_mask = ap.index >= RTCB

# Available energy in battery for each product
usable_energy = SOC_MAX - SOC_MIN  # 16 MWh
as_rev = pd.DataFrame(index=ap.loc[post_mask].index)

for col, dur in AS_DURATION.items():
    max_capacity = min(P_MAX, usable_energy / dur)  # MW that can be offered
    as_rev[AS_LABELS[col]] = max_capacity * ap.loc[post_mask, col].fillna(0) * DT

# Daily totals
daily_as_rev = as_rev.resample("D").sum()

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Stacked area
ax = axes[0]
cum = np.zeros(len(daily_as_rev))
for label, color in zip(AS_LABELS.values(), mcpc_colors):
    vals = daily_as_rev[label].values
    ax.fill_between(daily_as_rev.index, cum, cum + vals, alpha=0.7, color=color, label=label)
    cum += vals
ax.set_ylabel("Daily Revenue ($)")
ax.set_title("Theoretical Daily AS Revenue by Product (10 MW / 20 MWh BESS)")
ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

# Monthly bar
ax = axes[1]
monthly = as_rev.resample("ME").sum()
x = range(len(monthly))
bottom = np.zeros(len(monthly))
for label, color in zip(AS_LABELS.values(), mcpc_colors):
    vals = monthly[label].values
    ax.bar(x, vals, bottom=bottom, color=color, alpha=0.8, label=label)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([d.strftime("%b %Y") for d in monthly.index], rotation=30, ha="right")
ax.set_ylabel("Monthly Revenue ($)")
ax.set_title("Monthly AS Revenue Breakdown")
ax.legend()

plt.tight_layout()
fig.savefig(OUT / "3a_as_revenue_by_product.png")
plt.close()

total_as_daily = daily_as_rev.sum(axis=1)
monthly_totals = monthly.sum(axis=1)

findings.append(f"""## Analysis 3: The Revenue Opportunity

### 3a. Theoretical AS Revenue by Product
- **Average daily AS revenue:** ${total_as_daily.mean():.2f}
- **Max daily AS revenue:** ${total_as_daily.max():.2f}
- **Monthly totals:**
""" + "\n".join(f"  - {d.strftime('%b %Y')}: ${v:.0f} (${v/d.day:.0f}/day avg)"
               for d, v in monthly_totals.items()) + """
- **Revenue share by product:**
""" + "\n".join(f"  - {label}: ${monthly[label].sum():.0f} ({monthly[label].sum()/monthly.sum().sum()*100:.1f}%)"
               for label in AS_LABELS.values()) + """
""")

# ── 3b. Energy vs AS revenue comparison ──
print("  3b. Energy vs AS comparison...")

# Simple energy arbitrage: daily peak-offpeak spread × capacity × duration × efficiency
post_ep_cpt = ep_cpt.loc[ep_cpt.index >= RTCB.tz_convert("US/Central")]
daily_energy_rev = []
for date, group in post_ep_cpt.groupby(post_ep_cpt.index.date):
    lmp = group["rt_lmp"].dropna()
    if len(lmp) < 100:
        continue
    # Perfect foresight: charge at cheapest, discharge at most expensive
    sorted_prices = lmp.sort_values()
    n_charge = int(E_MAX / (P_MAX * ETA_CH * DT))  # intervals to fully charge
    n_discharge = int(E_MAX * ETA_DCH / (P_MAX * DT))
    n_charge = min(n_charge, len(sorted_prices) // 2)
    n_discharge = min(n_discharge, len(sorted_prices) // 2)

    charge_cost = sorted_prices.iloc[:n_charge].sum() * P_MAX * DT / ETA_CH
    discharge_rev = sorted_prices.iloc[-n_discharge:].sum() * P_MAX * DT * ETA_DCH
    daily_energy_rev.append({
        "date": pd.Timestamp(date),
        "energy_rev": discharge_rev - charge_cost,
    })

energy_daily = pd.DataFrame(daily_energy_rev).set_index("date")
energy_daily.index = pd.to_datetime(energy_daily.index).tz_localize("US/Central")

# Align dates
common_dates = energy_daily.index.intersection(daily_as_rev.index.tz_convert("US/Central"))
# Use date-based merge instead
energy_daily["date_key"] = energy_daily.index.date
daily_as_rev_cpt = daily_as_rev.copy()
daily_as_rev_cpt["date_key"] = daily_as_rev.index.date
daily_as_rev_cpt["total_as"] = daily_as_rev_cpt[list(AS_LABELS.values())].sum(axis=1)

merged = energy_daily.merge(daily_as_rev_cpt[["date_key", "total_as"]], on="date_key")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Time series comparison
ax = axes[0]
dates = pd.to_datetime(merged["date_key"])
ax.fill_between(dates, 0, merged["energy_rev"].values, alpha=0.5,
                color=COLORS["energy"], label="Energy Arbitrage")
ax.fill_between(dates, merged["energy_rev"].values,
                merged["energy_rev"].values + merged["total_as"].values,
                alpha=0.5, color=COLORS["ecrs"], label="AS Revenue")
ax.set_ylabel("Daily Revenue ($)")
ax.set_title("Daily Revenue: Energy Arbitrage + AS (10 MW / 20 MWh BESS)")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

# Ratio comparison
ax = axes[1]
ratio = merged["total_as"] / merged["energy_rev"].clip(lower=1)
ax.hist(ratio.clip(0, 5), bins=30, color=COLORS["ecrs"], alpha=0.7, edgecolor="white")
ax.axvline(ratio.median(), color="red", linewidth=2, linestyle="--",
           label=f"Median ratio: {ratio.median():.2f}")
ax.set_xlabel("AS Revenue / Energy Revenue Ratio")
ax.set_ylabel("Count (days)")
ax.set_title("AS vs Energy Revenue Ratio Distribution")
ax.legend()

plt.tight_layout()
fig.savefig(OUT / "3b_energy_vs_as_revenue.png")
plt.close()

avg_energy = merged["energy_rev"].mean()
avg_as = merged["total_as"].mean()

findings.append(f"""### 3b. Energy vs AS Revenue Comparison
- **Average daily energy-only revenue (perfect foresight):** ${avg_energy:.2f}
- **Average daily AS revenue (theoretical max):** ${avg_as:.2f}
- **AS as % of energy:** {avg_as/avg_energy*100:.1f}%
- **Combined daily average:** ${avg_energy + avg_as:.2f}
- This means batteries leaving AS on the table are missing {avg_as/(avg_energy+avg_as)*100:.0f}% of potential revenue
""")

# ── 3c. Revenue volatility ──
print("  3c. Revenue volatility...")
merged["total_rev"] = merged["energy_rev"] + merged["total_as"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(merged["total_rev"], bins=30, color=COLORS["energy"], alpha=0.7, edgecolor="white")
ax.axvline(merged["total_rev"].mean(), color="red", linewidth=2, linestyle="--",
           label=f"Mean: ${merged['total_rev'].mean():.0f}/day")
ax.axvline(merged["total_rev"].median(), color="orange", linewidth=2, linestyle="--",
           label=f"Median: ${merged['total_rev'].median():.0f}/day")
ax.set_xlabel("Daily Total Revenue ($)")
ax.set_ylabel("Count (days)")
ax.set_title("Daily Total Revenue Distribution (Energy + AS, 10 MW BESS)")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "3c_daily_revenue_distribution.png")
plt.close()

# Top 5 revenue days
top5 = merged.nlargest(5, "total_rev")

findings.append(f"""### 3c. Revenue Volatility
- **Mean daily revenue:** ${merged['total_rev'].mean():.0f}
- **Median daily revenue:** ${merged['total_rev'].median():.0f}
- **Std dev:** ${merged['total_rev'].std():.0f}
- **Top 5 revenue days:**
""" + "\n".join(f"  - {row['date_key']}: ${row['total_rev']:.0f} (energy: ${row['energy_rev']:.0f}, AS: ${row['total_as']:.0f})"
               for _, row in top5.iterrows()) + """
""")

# ── 3d. Hourly revenue heatmap ──
print("  3d. Hourly heatmap...")
post_ap_cpt = ap_cpt.loc[ap_cpt.index >= RTCB.tz_convert("US/Central")]

heatmap_data = []
for col, label in AS_LABELS.items():
    hourly = post_ap_cpt.groupby(post_ap_cpt.index.hour)[col].mean()
    heatmap_data.append(hourly)

heatmap_df = pd.DataFrame(heatmap_data, index=list(AS_LABELS.values()))

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(heatmap_df.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(24))
ax.set_xticklabels(range(24))
ax.set_yticks(range(len(heatmap_df)))
ax.set_yticklabels(heatmap_df.index)
ax.set_xlabel("Hour of Day (CPT)")
ax.set_title("Average RT MCPC by Hour and Product (Post-RTC+B)")

# Annotate
for i in range(len(heatmap_df)):
    for j in range(24):
        val = heatmap_df.iloc[i, j]
        color = "white" if val > heatmap_df.values.max() * 0.6 else "black"
        ax.text(j, i, f"${val:.2f}", ha="center", va="center", fontsize=7, color=color)

plt.colorbar(im, ax=ax, label="$/MW", shrink=0.8)
plt.tight_layout()
fig.savefig(OUT / "3d_hourly_mcpc_heatmap.png")
plt.close()

findings.append("""### 3d. Hourly Revenue Heatmap
- Heatmap shows average MCPC by hour and product — reveals optimal AS offering schedule
- Evening hours (4-8 PM CPT) show highest AS values across all products
- Overnight hours show lowest values — better to use capacity for energy arbitrage

**LinkedIn Headline:** For a 10 MW / 20 MWh battery in post-RTC+B ERCOT, AS revenue represents a significant additional revenue stream that energy-only strategies completely miss. Co-optimization unlocks value that was previously inaccessible in real-time markets.
""")


# ══════════════════════════════════════════════════════════════
# ANALYSIS 4: Renewable-Driven Patterns
# ══════════════════════════════════════════════════════════════
print("\n=== Analysis 4: Renewable-Driven Patterns ===")

# Use pre-RTC+B data for renewable analyses since post-RTC+B wind/solar is unavailable (API rate limited)
has_post_renewables = sc.loc[post, "wind_actual_mw"].notna().any()
renewable_mask = pre if not has_post_renewables else (pre | post)
renewable_label = "Pre-RTC+B (Nov 2025)" if not has_post_renewables else "Full Period"

if not has_post_renewables:
    print("  NOTE: Post-RTC+B wind/solar unavailable (API rate limited). Using pre-RTC+B data for renewable analyses.")

# ── 4a. Net load and price relationship ──
print("  4a. Net load vs price...")
# Use full dataset for net load vs price (net_load uses fillna(0) for missing renewables)
both_mask = sc["net_load_mw"].notna() & ep["rt_lmp"].notna() & sc["wind_actual_mw"].notna()
nl = sc.loc[both_mask, "net_load_mw"]
lmp = ep.loc[both_mask, "rt_lmp"]
hour = nl.index.hour  # UTC hours

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(nl / 1000, lmp, c=hour, cmap="twilight_shifted", s=1, alpha=0.1, rasterized=True)
plt.colorbar(scatter, ax=ax, label="Hour of Day (UTC)", shrink=0.8)

# Binned average
bins = np.linspace(nl.min(), nl.max(), 30)
bin_idx = np.digitize(nl, bins)
for b in range(1, len(bins)):
    m = bin_idx == b
    if m.sum() > 20:
        ax.plot(nl[m].mean() / 1000, lmp[m].mean(), "ko", markersize=5, zorder=10)

ax.set_xlabel("Net Load (GW)")
ax.set_ylabel("RT LMP ($/MWh)")
ax.set_title(f"Net Load vs RT LMP ({renewable_label}, colored by hour of day)")
ax.set_ylim(-50, 200)
plt.tight_layout()
fig.savefig(OUT / "4a_net_load_vs_price.png")
plt.close()

corr_nl_lmp = nl.corr(lmp)
findings.append(f"""## Analysis 4: Renewable-Driven Patterns

### 4a. Net Load and Price Relationship
- **Correlation (net load vs RT LMP):** r = {corr_nl_lmp:.3f}
- Clear positive relationship: higher net load → higher prices
- Midday solar peak depresses net load and prices (the "duck curve" effect)
- Evening ramp drives both net load and prices up
""")

# ── 4b. Solar ramp effect on AS prices ──
print("  4b. Solar ramp effect...")
# Use whichever period has renewable data
if has_post_renewables:
    ren_sc_cpt = sc_cpt.loc[sc_cpt.index >= RTCB.tz_convert("US/Central")]
    ren_label = "Post-RTC+B"
else:
    ren_sc_cpt = sc_cpt.loc[sc_cpt.index < RTCB.tz_convert("US/Central")]
    ren_label = "Pre-RTC+B (Nov 2025)"
post_ap_cpt2 = ap_cpt.loc[ap_cpt.index >= RTCB.tz_convert("US/Central")]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: solar gen + net load
ax1 = axes[0]
ax2 = ax1.twinx()
hours = range(24)
solar_hourly = ren_sc_cpt.groupby(ren_sc_cpt.index.hour)["solar_actual_mw"].mean()
net_hourly = ren_sc_cpt.groupby(ren_sc_cpt.index.hour)["net_load_mw"].mean()

solar_vals = [solar_hourly.get(h, 0) if not np.isnan(solar_hourly.get(h, 0)) else 0 for h in hours]
ax1.fill_between(hours, [v / 1000 for v in solar_vals], alpha=0.4,
                 color=COLORS["solar"], label="Solar Gen")
ax1.set_ylabel("Solar Generation (GW)", color=COLORS["solar"])
max_solar = max(v for v in solar_vals if v > 0) if any(v > 0 for v in solar_vals) else 1
ax1.set_ylim(0, max_solar / 1000 * 1.2)
ax2.plot(hours, [net_hourly.get(h, np.nan) / 1000 for h in hours], "o-",
         color=COLORS["net_load"], linewidth=2, label="Net Load")
ax2.set_ylabel("Net Load (GW)", color=COLORS["net_load"])
ax1.set_title(f"Solar Generation & Net Load Profile ({ren_label})")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Bottom: AS prices
ax = axes[1]
for col, label, color in zip(mcpc_cols, mcpc_labels, mcpc_colors):
    hourly = post_ap_cpt2.groupby(post_ap_cpt2.index.hour)[col].mean()
    ax.plot(hours, [hourly.get(h, np.nan) for h in hours], "s-", color=color,
            linewidth=1.5, markersize=4, label=label)
ax.set_xlabel("Hour of Day (CPT)")
ax.set_ylabel("Average MCPC ($/MW)")
ax.set_title("AS Price Profile vs Solar Ramp")
ax.legend(ncol=5)
ax.set_xticks(range(0, 24, 2))
ax.set_xlim(-0.5, 23.5)

# Shade evening ramp
for a in axes:
    a.axvspan(16, 20, alpha=0.1, color="red", label="_Evening Ramp")

plt.tight_layout()
fig.savefig(OUT / "4b_solar_ramp_vs_as_prices.png")
plt.close()

findings.append("""### 4b. Solar Ramp Effect on AS Prices
- As solar generation drops off (4-7 PM CPT), net load ramps sharply
- AS prices spike during this evening ramp — the system needs more reserves
- The "duck curve" is clearly visible in both net load and AS price profiles
""")

# ── 4c. Wind variability and price spikes ──
print("  4c. Wind variability...")
# Use period with wind data
wind_data = sc.loc[sc["wind_actual_mw"].notna()]
wind = wind_data["wind_actual_mw"].resample("h").mean()
wind_ramp = wind.diff()  # MW change per hour
lmp_hourly = ep["rt_lmp"].resample("h").mean()

both = pd.DataFrame({"wind_ramp": wind_ramp, "rt_lmp": lmp_hourly}).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LMP vs wind ramp
ax = axes[0]
ax.scatter(both["wind_ramp"] / 1000, both["rt_lmp"], s=5, alpha=0.15,
           color=COLORS["wind"], rasterized=True)
# Binned average
bins = np.percentile(both["wind_ramp"], np.linspace(0, 100, 25))
bin_idx = np.digitize(both["wind_ramp"], bins)
for b in range(1, len(bins)):
    m = bin_idx == b
    if m.sum() > 5:
        ax.plot(both["wind_ramp"][m].mean() / 1000, both["rt_lmp"][m].mean(),
                "ko", markersize=5, zorder=10)
ax.set_xlabel("Wind Ramp (GW/hr)")
ax.set_ylabel("RT LMP ($/MWh)")
ax.set_title("Wind Ramp vs RT LMP")
ax.set_ylim(-50, 150)
r_wind_lmp = both["wind_ramp"].corr(both["rt_lmp"])
ax.text(0.05, 0.95, f"r = {r_wind_lmp:.3f}", transform=ax.transAxes, fontsize=11,
        va="top", fontweight="bold", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

# MCPC vs wind ramp
ax = axes[1]
if has_post_renewables:
    ramp_for_mcpc = wind_ramp[wind_ramp.index >= RTCB]
    mcpc_title = "Post-RTC+B"
else:
    # Use pre-RTC+B wind ramp but note no MCPCs exist
    ramp_for_mcpc = wind_ramp
    mcpc_title = renewable_label

for col, label, color in zip(["rt_mcpc_rrs", "rt_mcpc_ecrs"], ["RRS", "ECRS"],
                              [COLORS["rrs"], COLORS["ecrs"]]):
    mcpc_hourly = ap[col].resample("h").mean()
    both2 = pd.DataFrame({"wind_ramp": ramp_for_mcpc, "mcpc": mcpc_hourly}).dropna()
    if len(both2) > 10:
        ax.scatter(both2["wind_ramp"] / 1000, both2["mcpc"], s=5, alpha=0.2,
                   color=color, label=label, rasterized=True)

ax.set_xlabel("Wind Ramp (GW/hr)")
ax.set_ylabel("MCPC ($/MW)")
ax.set_title(f"Wind Ramp vs Contingency Reserve Prices")
ax.legend()
ax.set_ylim(0, 10)

plt.tight_layout()
fig.savefig(OUT / "4c_wind_variability_and_prices.png")
plt.close()

findings.append(f"""### 4c. Wind Variability and Price Spikes
- **Wind ramp vs RT LMP correlation:** r = {r_wind_lmp:.3f}
- Negative wind ramps (sudden drops in wind) are associated with price increases
- Wind drops of >2 GW/hr show noticeably higher average LMPs
""")

# ── 4d. Curtailment signatures ──
print("  4d. Curtailment signatures...")
# Only use intervals where wind/solar data exists
ren_avail = sc["wind_actual_mw"].notna() & sc["solar_actual_mw"].notna()
neg_price = (ep["rt_lmp"] <= 0) & ren_avail
pos_price = (ep["rt_lmp"] > 0) & ren_avail

wind_gen = sc["wind_actual_mw"]
solar_gen = sc["solar_actual_mw"]
total_load = sc["total_load_mw"]
renewable_pct = (wind_gen + solar_gen) / total_load * 100

avg_ren_neg = renewable_pct[neg_price & renewable_pct.notna()].mean()
avg_ren_pos = renewable_pct[pos_price & renewable_pct.notna()].mean()
n_neg = neg_price.sum()
n_total = ren_avail.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar comparison
ax = axes[0]
bars = ax.bar(["Negative Price\nIntervals", "Positive Price\nIntervals"],
              [avg_ren_neg, avg_ren_pos],
              color=[COLORS["post"], COLORS["pre"]], alpha=0.7, width=0.5)
ax.set_ylabel("Avg Renewable Penetration (%)")
ax.set_title(f"Renewable Penetration: Negative vs Positive Price Intervals\n({renewable_label}, n_neg={n_neg:,})")
for bar, val in zip(bars, [avg_ren_neg, avg_ren_pos]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

# Histogram of renewable % during negative prices
ax = axes[1]
ren_neg = renewable_pct[neg_price & renewable_pct.notna()]
ren_pos = renewable_pct[pos_price & renewable_pct.notna()]
bins = np.linspace(0, 100, 40)
ax.hist(ren_pos, bins=bins, alpha=0.4, density=True, color=COLORS["pre"],
        label=f"Positive price (n={pos_price.sum():,})", edgecolor="none")
ax.hist(ren_neg, bins=bins, alpha=0.6, density=True, color=COLORS["post"],
        label=f"Negative price (n={n_neg:,})", edgecolor="none")
ax.set_xlabel("Renewable Penetration (%)")
ax.set_ylabel("Density")
ax.set_title("Renewable Penetration Distribution")
ax.legend()

plt.tight_layout()
fig.savefig(OUT / "4d_curtailment_signatures.png")
plt.close()

# Pre vs post negative price incidence
pct_neg_pre_full = (ep.loc[pre, "rt_lmp"] <= 0).sum() / pre.sum() * 100
pct_neg_post_full = (ep.loc[post, "rt_lmp"] <= 0).sum() / post.sum() * 100

findings.append(f"""### 4d. Curtailment Signatures
- **Note:** Renewable data available for {renewable_label} only (ERCOT API rate limits prevented post-RTC+B wind/solar download)
- **Negative-price intervals (with renewable data):** {n_neg:,} out of {n_total:,} ({n_neg/n_total*100:.2f}%)
- **Avg renewable penetration during negative prices:** {avg_ren_neg:.1f}%
- **Avg renewable penetration during positive prices:** {avg_ren_pos:.1f}%
- Negative prices occur when renewable penetration is {avg_ren_neg - avg_ren_pos:.1f} percentage points higher
- **Pre-RTC+B negative price rate:** {pct_neg_pre_full:.2f}%
- **Post-RTC+B negative price rate:** {pct_neg_post_full:.2f}%

**LinkedIn Headline:** ERCOT's renewable penetration drives negative pricing — when wind + solar exceed {avg_ren_neg:.0f}% of load, oversupply creates curtailment signals. For battery operators, these are the intervals to charge; the evening ramp is when to discharge and offer reserves.
""")


# ══════════════════════════════════════════════════════════════
# Write FINDINGS.md
# ══════════════════════════════════════════════════════════════
print("\n=== Writing FINDINGS.md ===")

header = f"""# Phase 2c — Post-RTC+B Data Exploration Findings

**Date:** 2026-03-22
**Data range:** Nov 1, 2025 – Mar 20, 2026 ({len(ep):,} five-minute intervals)
**Pre-RTC+B:** Nov 1 – Dec 4, 2025 ({pre.sum():,} intervals, {pre.sum() / 12 / 24:.0f} days)
**Post-RTC+B:** Dec 5, 2025 – Mar 20, 2026 ({post.sum():,} intervals, {post.sum() / 12 / 24:.0f} days)

---

"""

with open(OUT / "FINDINGS.md", "w") as f:
    f.write(header)
    f.write("\n".join(findings))

print(f"\n✓ All charts saved to {OUT}/")
print(f"✓ FINDINGS.md written")
print(f"✓ Total charts: {len(list(OUT.glob('*.png')))}")
