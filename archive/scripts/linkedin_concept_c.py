"""
LinkedIn Concept C — "Two Days in the Life of a Post-RTC+B Battery"
Publication-quality visualizations for LinkedIn post.

Charts:
  1. Two-day comparison (calm vs stress) — energy + AS prices
  2. DAM vs RT AS price scatter (forecast gap)
  3. Revenue tail distribution

Uses existing Phase 2c processed data. Does NOT modify any source files.
"""

import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────
OUT = Path("data/results/eda/linkedin")
OUT.mkdir(parents=True, exist_ok=True)

RTCB = pd.Timestamp("2025-12-05 06:00:00", tz="UTC")

# Battery params
P_MAX = 10.0
E_MAX = 20.0
DT = 5 / 60  # hours per 5-min interval
ETA = 0.92
AS_DURATION = {
    "rt_mcpc_regup": 0.5,
    "rt_mcpc_regdn": 0.5,
    "rt_mcpc_rrs": 0.5,
    "rt_mcpc_ecrs": 1.0,
    "rt_mcpc_nsrs": 4.0,
}

# Color palette — professional, distinctive
C = {
    "regup":  "#E8645A",  # deep coral
    "regdn":  "#7CAE7A",  # sage green
    "rrs":    "#D4A03C",  # amber/gold
    "ecrs":   "#7B6D8E",  # slate purple
    "nsrs":   "#4A9B8E",  # teal
    "lmp":    "#2C3E50",  # dark slate for LMP line
    "neg":    "#F0F0F0",  # light gray for negative price shading
    "hist":   "#4A7FB5",  # histogram blue
    "median": "#E8645A",  # coral for median line
    "mean":   "#2C3E50",  # dark for mean line
}

AS_COLS = ["rt_mcpc_regup", "rt_mcpc_regdn", "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs"]
AS_LABELS = ["Reg Up", "Reg Down", "RRS", "ECRS", "NSRS"]
AS_COLORS = [C["regup"], C["regdn"], C["rrs"], C["ecrs"], C["nsrs"]]
DAM_COLS = ["dam_as_regup", "dam_as_regdn", "dam_as_rrs", "dam_as_ecrs", "dam_as_nsrs"]

# Selected days
CALM_DATE = "2026-01-07"
STRESS_DATE = "2026-01-25"

# ── Global style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load data ────────────────────────────────────────────────
def load_table(name):
    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(f"data/processed/{name}/*.parquet"))]
    df = pd.concat(dfs).sort_index()
    return df[~df.index.duplicated(keep="last")]

print("Loading data...")
ep = load_table("energy_prices")
ap = load_table("as_prices")

# Convert to CPT
ep_cpt = ep.copy()
ep_cpt.index = ep_cpt.index.tz_convert("US/Central")
ap_cpt = ap.copy()
ap_cpt.index = ap_cpt.index.tz_convert("US/Central")


# ══════════════════════════════════════════════════════════════
# CHART 1: Two Days Side-by-Side
# ══════════════════════════════════════════════════════════════
print("\n=== Chart 1: Two Days Comparison ===")

def get_day_data(date_str, ep_cpt, ap_cpt):
    """Extract one day of data in CPT, return hour-of-day x-axis."""
    d = pd.Timestamp(date_str).date()
    ep_day = ep_cpt.loc[ep_cpt.index.date == d].copy()
    ap_day = ap_cpt.loc[ap_cpt.index.date == d].copy()

    # Create fractional hour for x-axis
    ep_day["hour"] = ep_day.index.hour + ep_day.index.minute / 60
    ap_day["hour"] = ap_day.index.hour + ap_day.index.minute / 60

    return ep_day, ap_day


calm_ep, calm_ap = get_day_data(CALM_DATE, ep_cpt, ap_cpt)
stress_ep, stress_ap = get_day_data(STRESS_DATE, ep_cpt, ap_cpt)

# Compute daily revenues for annotation
def daily_revenue(ep_day, ap_day):
    lmp = ep_day["rt_lmp"].dropna()
    sorted_p = lmp.sort_values()
    n_ch = min(int(E_MAX / (P_MAX * ETA * DT)), len(sorted_p) // 2)
    n_dch = min(int(E_MAX * ETA / (P_MAX * DT)), len(sorted_p) // 2)
    e_rev = (sorted_p.iloc[-n_dch:].sum() * P_MAX * DT * ETA
             - sorted_p.iloc[:n_ch].sum() * P_MAX * DT / ETA)
    as_rev = sum(
        min(P_MAX, 16 / dur) * ap_day[col].fillna(0).sum() * DT
        for col, dur in AS_DURATION.items()
    )
    return e_rev, as_rev

calm_e, calm_as = daily_revenue(calm_ep, calm_ap)
stress_e, stress_as = daily_revenue(stress_ep, stress_ap)

print(f"  Calm day ({CALM_DATE}): energy=${calm_e:.0f}, AS=${calm_as:.0f}, total=${calm_e+calm_as:.0f}")
print(f"  Stress day ({STRESS_DATE}): energy=${stress_e:.0f}, AS=${stress_as:.0f}, total=${stress_e+stress_as:.0f}")

# Build figure: 2 rows × 2 cols
fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                          gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28, "wspace": 0.08})

# Shared y-limits
lmp_max = max(calm_ep["rt_lmp"].max(), stress_ep["rt_lmp"].max()) * 1.08
lmp_min = min(calm_ep["rt_lmp"].min(), stress_ep["rt_lmp"].min()) - 5

all_as_sums = []
for ap_day in [calm_ap, stress_ap]:
    stacked = np.zeros(len(ap_day))
    for col in AS_COLS:
        stacked += ap_day[col].fillna(0).values
    all_as_sums.append(stacked.max())
as_max = max(all_as_sums) * 1.08

days_data = [
    (calm_ep, calm_ap, CALM_DATE, "calm"),
    (stress_ep, stress_ap, STRESS_DATE, "stress"),
]

for col_idx, (ep_day, ap_day, date_str, day_type) in enumerate(days_data):
    h = ep_day["hour"].values
    h_as = ap_day["hour"].values

    # ── Panel 1: LMP ──
    ax = axes[0, col_idx]
    lmp_vals = ep_day["rt_lmp"].values

    # Shade negative price intervals only where prices are actually negative
    neg_mask = lmp_vals < 0
    if neg_mask.any():
        ax.fill_between(h, lmp_vals, 0, where=neg_mask, alpha=0.20, color="#BDBDBD",
                         linewidth=0, zorder=1, interpolate=True)

    ax.plot(h, lmp_vals, color=C["lmp"], linewidth=1.0, zorder=3)
    ax.fill_between(h, 0, np.clip(lmp_vals, 0, None), alpha=0.06, color=C["lmp"], zorder=1)
    ax.axhline(0, color="#999999", linewidth=0.5, zorder=0)
    ax.set_ylim(lmp_min, lmp_max)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))

    # Only show y-axis labels on left
    if col_idx == 0:
        ax.set_ylabel("RT LMP  ($/MWh)", fontsize=11)
    else:
        ax.tick_params(labelleft=False)

    # Light horizontal gridlines for LMP panel only
    ax.grid(axis="y", alpha=0.25, linewidth=0.5, color="#CCCCCC")
    ax.grid(axis="x", alpha=0)

    # Column title
    d_fmt = pd.Timestamp(date_str).strftime("%B %d, %Y")
    if day_type == "calm":
        ax.set_title(f"Typical Day — {d_fmt}", fontsize=12.5, pad=10)
    else:
        ax.set_title(f"Stress Event — {d_fmt}", fontsize=12.5, pad=10)

    # ── Panel 2: Stacked AS ──
    ax2 = axes[1, col_idx]
    bottom = np.zeros(len(ap_day))
    for i, (col, label, color) in enumerate(zip(AS_COLS, AS_LABELS, AS_COLORS)):
        vals = ap_day[col].fillna(0).values
        ax2.fill_between(h_as, bottom, bottom + vals, alpha=0.85, color=color,
                          linewidth=0.3, edgecolor="white", label=label if col_idx == 0 else None)
        bottom += vals

    ax2.set_ylim(0, as_max)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 4))
    ax2.set_xlabel("Hour of Day (CPT)", fontsize=10.5)

    if col_idx == 0:
        ax2.set_ylabel("RT MCPC  ($/MW)", fontsize=11)
    else:
        ax2.tick_params(labelleft=False)

    # No gridlines on stacked area
    ax2.grid(False)

# ── Annotations ──
# Calm day — place text in the empty upper-right area
calm_mid_price = calm_ep["rt_lmp"].median()
axes[0, 0].annotate(
    "Flat — heuristic strategies\nwork fine here",
    xy=(16, calm_mid_price),
    xytext=(15, lmp_max * 0.55),
    fontsize=9, color="#555555", fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.8, connectionstyle="arc3,rad=0.2"),
    ha="center",
)

# Stress day — find peak hour for annotation
stress_lmp = stress_ep["rt_lmp"].values
stress_hours = stress_ep["hour"].values
peak_idx = np.nanargmax(stress_lmp)
peak_hour = stress_hours[peak_idx]
peak_val = stress_lmp[peak_idx]

# Morning ramp annotation (if there's a notable early spike)
morning_mask = stress_hours < 10
morning_lmp = stress_lmp.copy()
morning_lmp[~morning_mask] = np.nan
morning_peak_idx = np.nanargmax(morning_lmp)
morning_peak_h = stress_hours[morning_peak_idx]
morning_peak_v = stress_lmp[morning_peak_idx]

if morning_peak_v > 100:
    axes[0, 1].annotate(
        "Morning\nramp",
        xy=(morning_peak_h, morning_peak_v),
        xytext=(morning_peak_h - 2.5, lmp_max * 0.75),
        fontsize=8.5, color="#555555", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8),
        ha="center",
    )

# Evening scarcity annotation
evening_mask = (stress_hours >= 16) & (stress_hours <= 21)
evening_lmp = stress_lmp.copy()
evening_lmp[~evening_mask] = np.nan
if not np.all(np.isnan(evening_lmp)):
    eve_peak_idx = np.nanargmax(evening_lmp)
    eve_peak_h = stress_hours[eve_peak_idx]
    eve_peak_v = stress_lmp[eve_peak_idx]
    if eve_peak_v > 50:
        axes[0, 1].annotate(
            "Evening\nscarcity",
            xy=(eve_peak_h, eve_peak_v),
            xytext=(eve_peak_h + 2, lmp_max * 0.65),
            fontsize=8.5, color="#555555", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8),
            ha="center",
        )

# Main stress day revenue annotation on AS panel
axes[1, 1].annotate(
    f"Volatile — ${stress_e + stress_as:,.0f} in revenue,\ndriven by 5-min dynamics",
    xy=(12, as_max * 0.5),
    xytext=(12, as_max * 0.82),
    fontsize=9, color="#333333", fontstyle="italic",
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
)

# ── Shared legend at bottom ──
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

# ── Titles ──
fig.suptitle(
    "Two Days, Same Battery, Same Market — The Optimization Challenge of ERCOT's RTC+B",
    fontsize=15, fontweight="bold", y=0.98,
)
fig.text(
    0.5, 0.935,
    f"Left: Typical day ({pd.Timestamp(CALM_DATE).strftime('%b %d, %Y')}).  "
    f"Right: Stress event ({pd.Timestamp(STRESS_DATE).strftime('%b %d, %Y')}).  "
    f"5-minute co-optimized energy + ancillary service prices.",
    ha="center", fontsize=10, color="#666666",
)

fig.savefig(OUT / "concept_c_two_days.png")
fig.savefig(OUT / "concept_c_two_days.svg")
plt.close()
print(f"  Saved concept_c_two_days.png/svg")


# ══════════════════════════════════════════════════════════════
# CHART 2: DAM vs RT — The Forecast Gap
# ══════════════════════════════════════════════════════════════
print("\n=== Chart 2: DAM vs RT Forecast Gap ===")

post_ap = ap_cpt.loc[ap_cpt.index >= RTCB.tz_convert("US/Central")].copy()

# Resample to hourly means for comparison (DAM is hourly, RT is 5-min)
hourly_rt = post_ap[AS_COLS].resample("h").mean()
hourly_dam = post_ap[DAM_COLS].resample("h").mean()

fig, axes = plt.subplots(1, 5, figsize=(16, 4.8), sharey=False)
fig.subplots_adjust(wspace=0.38)

correlations = {}
for i, (rt_col, dam_col, label, color) in enumerate(
    zip(AS_COLS, DAM_COLS, AS_LABELS, AS_COLORS)
):
    ax = axes[i]
    x = hourly_dam[dam_col].values
    y = hourly_rt[rt_col].values

    # Drop NaN pairs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    # Clip to 95th percentile for visualization (outliers compress the view)
    clip_x = np.percentile(x[x > 0], 97) if (x > 0).any() else 100
    clip_y = np.percentile(y[y > 0], 97) if (y > 0).any() else 100
    clip_val = max(clip_x, clip_y, 50)  # at least $50

    ax.scatter(x, y, alpha=0.18, s=12, color=color, edgecolors="none", rasterized=True)

    # Correlation
    r = np.corrcoef(x, y)[0, 1]
    correlations[label] = r

    # 1:1 line
    ax.plot([0, clip_val], [0, clip_val], "--", color="#BBBBBB", linewidth=0.8, zorder=0)

    # Regression line — only within the clipped range
    if len(x) > 10:
        m, b = np.polyfit(x, y, 1)
        x_line = np.array([0, clip_val])
        y_line = m * x_line + b
        ax.plot(x_line, np.clip(y_line, 0, clip_val * 1.2), color=color, linewidth=1.5, alpha=0.7)

    ax.set_title(label, fontsize=11, fontweight="bold", color=color)
    ax.text(
        0.05, 0.92, f"r = {r:.2f}",
        transform=ax.transAxes, fontsize=10.5, fontweight="bold",
        color=color, va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8),
    )

    ax.set_xlabel("DAM ($/MW)")
    if i == 0:
        ax.set_ylabel("RT MCPC ($/MW)")

    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_xlim(0, clip_val)
    ax.set_ylim(0, clip_val)

    # Count outliers beyond clip
    n_outliers = ((x > clip_val) | (y > clip_val)).sum()
    if n_outliers > 0:
        ax.text(0.95, 0.05, f"+{n_outliers} outliers\nbeyond axes",
                transform=ax.transAxes, fontsize=7, color="#999999",
                ha="right", va="bottom")

fig.suptitle(
    "Day-Ahead AS Prices vs Real-Time MCPCs — Weak Correlation Across All Products",
    fontsize=13, fontweight="bold", y=1.04,
)
fig.text(0.5, 0.98, f"Post-RTC+B (Dec 5, 2025 – Mar 20, 2026)  ·  Hourly averages  ·  "
         f"Dashed line = perfect forecast",
         ha="center", fontsize=9, color="#888888")

fig.savefig(OUT / "dam_vs_rt_forecast_gap.png")
plt.close()
print(f"  Saved dam_vs_rt_forecast_gap.png")
print(f"  Correlations: {correlations}")


# ══════════════════════════════════════════════════════════════
# CHART 3: Revenue is Tail-Driven
# ══════════════════════════════════════════════════════════════
print("\n=== Chart 3: Revenue Tail Distribution ===")

# Compute daily revenue for all post-RTC+B days
post_ep_cpt = ep_cpt.loc[ep_cpt.index >= RTCB.tz_convert("US/Central")]
post_ap_cpt = ap_cpt.loc[ap_cpt.index >= RTCB.tz_convert("US/Central")]

daily_revs = []
for d, grp in post_ep_cpt.groupby(post_ep_cpt.index.date):
    lmp = grp["rt_lmp"].dropna()
    # Energy revenue (perfect foresight)
    if len(lmp) >= 100:
        sorted_p = lmp.sort_values()
        n_ch = min(int(E_MAX / (P_MAX * ETA * DT)), len(sorted_p) // 2)
        n_dch = min(int(E_MAX * ETA / (P_MAX * DT)), len(sorted_p) // 2)
        e_rev = (sorted_p.iloc[-n_dch:].sum() * P_MAX * DT * ETA
                 - sorted_p.iloc[:n_ch].sum() * P_MAX * DT / ETA)
    else:
        # Use DAM SPP as proxy for days without RT LMP
        spp = grp["dam_spp"].dropna()
        if len(spp) < 50:
            continue
        sorted_p = spp.sort_values()
        n_ch = min(int(E_MAX / (P_MAX * ETA * DT)), len(sorted_p) // 2)
        n_dch = min(int(E_MAX * ETA / (P_MAX * DT)), len(sorted_p) // 2)
        e_rev = (sorted_p.iloc[-n_dch:].sum() * P_MAX * DT * ETA
                 - sorted_p.iloc[:n_ch].sum() * P_MAX * DT / ETA)

    # AS revenue
    ap_day = post_ap_cpt.loc[post_ap_cpt.index.date == d]
    as_rev = sum(
        min(P_MAX, 16 / dur) * ap_day[col].fillna(0).sum() * DT
        for col, dur in AS_DURATION.items()
    )

    daily_revs.append({"date": d, "energy": e_rev, "as": as_rev, "total": e_rev + as_rev})

rev_df = pd.DataFrame(daily_revs).sort_values("total", ascending=False)
total_cum = rev_df["total"].sum()
top5_rev = rev_df.head(5)["total"].sum()
top10_rev = rev_df.head(10)["total"].sum()
pct_top5 = top5_rev / total_cum * 100
pct_top10 = top10_rev / total_cum * 100
median_rev = rev_df["total"].median()
mean_rev = rev_df["total"].mean()
max_rev = rev_df["total"].max()

print(f"  Total days: {len(rev_df)}")
print(f"  Median: ${median_rev:.0f}, Mean: ${mean_rev:.0f}, Max: ${max_rev:.0f}")
print(f"  Top 5 days: ${top5_rev:.0f} = {pct_top5:.1f}% of total")
print(f"  Top 10 days: ${top10_rev:.0f} = {pct_top10:.1f}% of total")

fig, ax = plt.subplots(figsize=(12, 5.5))

# Histogram
bins = np.linspace(0, rev_df["total"].max() * 1.02, 40)
n, bin_edges, patches = ax.hist(
    rev_df["total"], bins=bins, color=C["hist"], alpha=0.75,
    edgecolor="white", linewidth=0.6,
)

# Color the tail bars differently
threshold = rev_df.nlargest(5, "total")["total"].min()
for patch, left_edge in zip(patches, bin_edges[:-1]):
    if left_edge >= threshold:
        patch.set_facecolor(C["regup"])
        patch.set_alpha(0.85)

# Median and mean lines
ax.axvline(median_rev, color=C["median"], linewidth=2, linestyle="--", zorder=5)
ax.axvline(mean_rev, color=C["mean"], linewidth=2, linestyle="-.", zorder=5)

# Annotations
y_top = ax.get_ylim()[1]

# Place median/mean labels above the dashed lines
ax.text(median_rev + 200, y_top * 0.92, f"Median: ${median_rev:,.0f}/day",
        fontsize=10, color=C["median"], fontweight="bold", va="top")
ax.text(mean_rev + 200, y_top * 0.80, f"Mean: ${mean_rev:,.0f}/day",
        fontsize=10, color=C["mean"], fontweight="bold", va="top")

# Tail annotation — point at the red bars on the right
tail_center = (threshold + max_rev) / 2
ax.annotate(
    f"Top 5 days = {pct_top5:.0f}% of total revenue",
    xy=(threshold * 1.1, 0.8),
    xytext=(max_rev * 0.45, y_top * 0.55),
    fontsize=10.5, color=C["regup"], fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=C["regup"], lw=1.5,
                     connectionstyle="arc3,rad=-0.15"),
    ha="center",
)

# Gap annotation — place in the middle empty area
ax.text(max_rev * 0.48, y_top * 0.40,
        f"Mean is {mean_rev/median_rev:.1f}× the median\n— revenue is extremely right-skewed",
        fontsize=9.5, color="#555555", fontstyle="italic", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FAFAFA", edgecolor="#DDDDDD"))

ax.set_xlabel("Daily Total Revenue  ($)", fontsize=11)
ax.set_ylabel("Number of Days", fontsize=11)
ax.set_title("Battery Revenue is Dominated by Tail Events", fontsize=14, fontweight="bold", pad=12)
ax.text(0.5, 1.02,
        f"10 MW / 20 MWh BESS  ·  {len(rev_df)} post-RTC+B days  ·  Perfect-foresight energy + theoretical AS max",
        transform=ax.transAxes, ha="center", fontsize=9, color="#888888")

ax.grid(axis="y", alpha=0.2, linewidth=0.5)
ax.grid(axis="x", alpha=0)
ax.set_xlim(left=0)

fig.savefig(OUT / "revenue_tail_driven.png")
plt.close()
print(f"  Saved revenue_tail_driven.png")


# ══════════════════════════════════════════════════════════════
# Summary stats
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KEY STATS FOR LINKEDIN POST")
print("=" * 60)
total_intervals = len(ap_cpt.loc[ap_cpt.index >= RTCB.tz_convert("US/Central")])
print(f"Total 5-min intervals analyzed (post-RTC+B): {total_intervals:,}")
print(f"DAM vs RT AS correlations:")
for label, r in correlations.items():
    print(f"  {label}: r = {r:.3f}")
print(f"Correlation range: {min(correlations.values()):.3f} – {max(correlations.values()):.3f}")
print(f"Top 5 days = {pct_top5:.1f}% of total revenue")
print(f"Top 10 days = {pct_top10:.1f}% of total revenue")
print(f"Max single-day revenue: ${max_rev:,.0f}")
print(f"Median daily revenue: ${median_rev:,.0f}")
print(f"Mean daily revenue: ${mean_rev:,.0f}")
print(f"Calm day: {CALM_DATE} — LMP range ${calm_ep['rt_lmp'].max()-calm_ep['rt_lmp'].min():.0f}, revenue ${calm_e+calm_as:,.0f}")
print(f"  Chosen because: lowest daily price range ($31), typical flat profile, weekday")
print(f"Stress day: {STRESS_DATE} — LMP range ${stress_ep['rt_lmp'].max()-stress_ep['rt_lmp'].min():.0f}, revenue ${stress_e+stress_as:,.0f}")
print(f"  Chosen because: highest revenue day, dramatic spikes, max LMP $938")
print(f"\nTop 5 revenue days:")
for _, row in rev_df.head(5).iterrows():
    print(f"  {row['date']}: ${row['total']:,.0f} (energy: ${row['energy']:,.0f}, AS: ${row['as']:,.0f})")
print("\nDone!")
