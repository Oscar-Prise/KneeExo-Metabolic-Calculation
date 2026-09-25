"""
Outdoor metabolic / cost-of-transport pipeline (after Slade et al., Nature 2022).

Method
------
1. Breath-by-breath VO2 / VCO2 (mL/min) from the COSMED K5 export are turned into a
   metabolic rate signal, using two metrics that share every downstream step:
     * energy : Brockway (1987) power, P [W] = (16.58 * VO2 + 4.51 * VCO2) / 60
     * O2     : oxygen uptake, VO2 [mL/s] = VO2 / 60
   The O2 metric is kept because RQ often exceeds 1 in these trials, where the
   Brockway (aerobic, substrate-based) assumptions no longer hold.
2. Breath data are linearly interpolated onto a 1 Hz grid.
3. Quiet-standing baseline = mean of the last 3 min of Standing.xlsx.
4. For each condition, the baseline-subtracted signal is integrated (trapezoid) from
   t = 0 to the end of the final 3-min quiet-standing lap in Bout.txt. Including the
   post-walking standing captures the excess gas exchange that lags muscular energy use.
5. Cost of transport is normalised by preferred walking speed rather than distance:
       CoT = net total / (body mass * mean walking speed)   [J kg^-1 (m/s)^-1 or mL kg^-1 (m/s)^-1]
   where mean walking speed = total walking distance / walking time.
6. Results are also expressed as % change relative to NoExo (Normal Shoes).

Bout.txt times are the reference clock (K5 t=0 is assumed to coincide with the
stopwatch start; the K5 is stopped a few seconds later and the tail is dropped).

Usage
-----
    python metabolic_pipeline.py                 # every day-folder next to this script
    python metabolic_pipeline.py Sep25_2026_Ilseung [more folders ...]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #
IN2M = 0.0254
FT2M = 0.3048

STAIR_RUN_IN = 988.5
STAIR_RISE_IN = 371.0
STAIR_DIST_M = np.hypot(STAIR_RUN_IN, STAIR_RISE_IN) * IN2M  # hypotenuse of one flight

# Bout.txt has 8 laps: 7 walking bouts followed by 3 min of quiet standing.
# RA = ramp, SA = stair ascent, SD = stair descent.
WALK_SEGMENTS = [
    ("RA1", 288 * FT2M),
    ("RA2", 533 * FT2M),
    ("SA1", STAIR_DIST_M),
    ("SD1", STAIR_DIST_M),
    ("SA2", STAIR_DIST_M),
    ("SD2", STAIR_DIST_M),
    ("SA3", STAIR_DIST_M),
]
STANDING_LABEL = "Stand"
SEG_LABELS = [n for n, _ in WALK_SEGMENTS] + [STANDING_LABEL]
SEG_DISTS = [d for _, d in WALK_SEGMENTS] + [0.0]

CONDITIONS = ["NoExo", "ExoOff", "ExoOn"]
REFERENCE = "NoExo"
BASELINE_WINDOW_S = 180.0

BROCKWAY_O2 = 16.58  # kJ/L O2  == J/mL
BROCKWAY_CO2 = 4.51  # kJ/L CO2 == J/mL

# Each metric: rate signal per second from the breath table, plus display units.
#   total   : unit of the integral (per kg)
#   rate    : unit of the displayed rate (per kg); rate_scale converts per-second -> displayed
METRICS = {
    "energy": dict(
        signal=lambda df: (BROCKWAY_O2 * df["VO2"].to_numpy()
                           + BROCKWAY_CO2 * df["VCO2"].to_numpy()) / 60.0,  # W
        label="Brockway energy", total="J/kg", rate="W/kg", rate_scale=1.0,
        cot="J·kg⁻¹·(m/s)⁻¹", key="J",
    ),
    "O2": dict(
        signal=lambda df: df["VO2"].to_numpy() / 60.0,  # mL O2 / s
        label="O₂ uptake", total="mL O₂/kg", rate="mL/kg/min", rate_scale=60.0,
        cot="mL·kg⁻¹·(m/s)⁻¹", key="mLO2",
    ),
}

# Validated categorical slots 1-3 (light surface) - fixed per condition
COLORS = {"NoExo": "#2a78d6", "ExoOff": "#eb6834", "ExoOn": "#1baf7a"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
BAND = "#efeeea"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_k5(path: Path) -> tuple[pd.DataFrame, float]:
    """Return breath table (t [s], VO2, VCO2 [mL/min]) and body mass [kg]."""
    raw = pd.read_excel(path, sheet_name="Data", header=None)

    mass_row = raw.index[raw[0].astype(str).str.strip() == "Weight (kg)"]
    mass = float(raw.loc[mass_row[0], 1])

    header = raw.iloc[0]
    col = {name: j for j, name in header.items() if isinstance(name, str)}
    data = raw.iloc[3:, [col["t"], col["VO2"], col["VCO2"]]]
    data.columns = ["t", "VO2", "VCO2"]
    data = data.dropna(subset=["t"])

    data["t"] = pd.to_timedelta(data["t"].astype(str)).dt.total_seconds()
    data["VO2"] = pd.to_numeric(data["VO2"], errors="coerce")
    data["VCO2"] = pd.to_numeric(data["VCO2"], errors="coerce")
    return data.reset_index(drop=True), mass


def clean_breaths(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drop non-physiological breaths (non-positive gas volumes or RQ outside 0.6-1.5)."""
    rq = df["VCO2"] / df["VO2"]
    ok = (df["VO2"] > 0) & (df["VCO2"] > 0) & rq.between(0.6, 1.5)
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"    {name}: removed {n_bad}/{len(df)} breaths (bad VO2/VCO2 or RQ)")
    out = df[ok].reset_index(drop=True)
    out["RQ"] = out["VCO2"] / out["VO2"]
    return out


def read_bouts(path: Path) -> dict[str, np.ndarray]:
    """Parse Bout.txt -> {condition: cumulative end time [s] of each of the 8 laps}."""
    def to_s(txt):
        m, s = txt.split(":")
        return 60 * float(m) + float(s)

    # Stopwatch exports pad with non-breaking spaces, so match the mm:ss.xx fields directly
    bouts, current = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        times = re.findall(r"\d+:\d+(?:\.\d+)?", line)
        if len(times) >= 2:
            bouts[current].append(to_s(times[1]))  # cumulative time column
        elif line.strip():
            current = line.strip()
            bouts[current] = []
    return {k: np.array(v) for k, v in bouts.items()}


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #
def to_1hz(t: np.ndarray, y: np.ndarray, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, np.floor(t_end) + 1.0)
    if grid[-1] < t_end:
        grid = np.append(grid, t_end)
    return grid, np.interp(grid, t, y)  # holds edge values outside the breath range


def integrate(t, y, t0, t1):
    m = (t >= t0) & (t <= t1)
    return float(np.trapezoid(y[m], t[m]))


def standing_baseline(df: pd.DataFrame, signal: np.ndarray) -> float:
    """Time-weighted mean of the signal over the last BASELINE_WINDOW_S seconds."""
    t, y = to_1hz(df["t"].to_numpy(), signal, df["t"].iloc[-1])
    t0 = max(t[0], t[-1] - BASELINE_WINDOW_S)
    return integrate(t, y, t0, t[-1]) / (t[-1] - t0)


def process_day(day: Path) -> None:
    data_dir, out_dir = day / "Data", day / "Output"
    out_dir.mkdir(exist_ok=True)
    print(f"\n=== {day.name} ===")

    stand_df, mass = read_k5(data_dir / "Standing.xlsx")
    stand_df = clean_breaths(stand_df, "Standing")
    base = {m: standing_baseline(stand_df, spec["signal"](stand_df)) for m, spec in METRICS.items()}
    print(f"    body mass {mass:.1f} kg | standing baseline {base['energy']:.1f} W "
          f"({base['energy'] / mass:.2f} W/kg), VO2 {base['O2'] * 60:.0f} mL/min "
          f"| last {BASELINE_WINDOW_S:.0f} s of {stand_df['t'].iloc[-1]:.0f} s")

    bouts = read_bouts(data_dir / "Bout.txt")
    walk_dist = sum(SEG_DISTS)

    summary, seg_rows = [], []
    series = {m: {} for m in METRICS}  # metric -> cond -> (t, net rate per kg per s, ends)
    rq_series, seg_speeds = {}, {}
    for cond in CONDITIONS:
        if cond not in bouts or not (data_dir / f"{cond}.xlsx").exists():
            print(f"    {cond}: missing data or bout times, skipped")
            continue
        df, _ = read_k5(data_dir / f"{cond}.xlsx")
        df = clean_breaths(df, cond)
        ends = bouts[cond]
        if len(ends) != len(SEG_LABELS):
            raise ValueError(f"{cond}: expected {len(SEG_LABELS)} laps, got {len(ends)}")
        t_end, t_walk = ends[-1], ends[-2]
        if df["t"].iloc[-1] < t_end - 5:
            print(f"    WARNING {cond}: K5 ends at {df['t'].iloc[-1]:.0f} s, before bout end {t_end:.0f} s")

        starts = np.concatenate([[0.0], ends[:-1]])
        durs = ends - starts
        speeds = np.array([d / dt if d else np.nan for d, dt in zip(SEG_DISTS, durs)])
        seg_speeds[cond] = speeds[:-1]
        speed = walk_dist / t_walk
        rq_series[cond] = (df["t"].to_numpy(), df["RQ"].to_numpy(), ends)

        row = {
            "condition": cond, "body_mass_kg": mass,
            "walk_time_s": t_walk, "total_time_s": t_end,
            "distance_m": walk_dist, "mean_walk_speed_m_s": speed,
            "mean_RQ_walk": float(df.loc[df["t"] <= t_walk, "RQ"].mean()),
        }
        seg_cols = {lab: {} for lab in SEG_LABELS}
        for m, spec in METRICS.items():
            k = spec["key"]
            t, y = to_1hz(df["t"].to_numpy(), spec["signal"](df), t_end)
            y_net = (y - base[m]) / mass  # per kg per second
            series[m][cond] = (t, y_net, ends)
            total = integrate(t, y_net, 0.0, t_end)  # walking + 3-min recovery
            row[f"standing_{k}_per_s"] = base[m]
            row[f"net_total_{k}_kg"] = total
            row[f"net_rate_walk_{spec['rate'].replace('/', '_')}"] = total / t_walk * spec["rate_scale"]
            row[f"CoT_{k}_kg_per_m_s"] = total / speed
            for lab, t0, t1 in zip(SEG_LABELS, starts, ends):
                seg_cols[lab][f"net_{k}_kg"] = integrate(t, y_net, t0, t1)
        summary.append(row)

        for lab, t0, t1, d, v in zip(SEG_LABELS, starts, ends, SEG_DISTS, speeds):
            seg_rows.append({"condition": cond, "segment": lab, "start_s": t0, "end_s": t1,
                             "duration_s": t1 - t0, "distance_m": d, "speed_m_s": v,
                             **seg_cols[lab]})

    summ = pd.DataFrame(summary)
    if REFERENCE in set(summ["condition"]):
        ref = summ.set_index("condition").loc[REFERENCE]
        for c in [c for c in summ.columns if c.startswith(("CoT_", "net_total_", "net_rate_"))]:
            summ[f"{c}_pct_vs_{REFERENCE}"] = 100 * (summ[c] / ref[c] - 1)
    segs = pd.DataFrame(seg_rows)

    summ.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    segs.to_csv(out_dir / "segments.csv", index=False, float_format="%.4f")
    show = ["condition", "walk_time_s", "mean_walk_speed_m_s", "mean_RQ_walk"] + \
        [c for c in summ.columns if c.startswith("CoT_")]
    print(summ[show].to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    for old in ("net_power_timeseries.png", "cumulative_energy.png", "cot_summary.png"):
        (out_dir / old).unlink(missing_ok=True)  # names from the previous version
    plot_rq(rq_series, seg_speeds, day.name, out_dir / "RQ_timeseries.png")
    for m, spec in METRICS.items():
        plot_cumulative(series[m], spec, day.name, out_dir / f"cumulative_{m}.png")
        plot_summary(summ, seg_speeds, spec, day.name, out_dir / f"summary_{m}.png")
    print(f"    outputs -> {out_dir}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.yaxis.label.set_color(TEXT2)
    ax.xaxis.label.set_color(TEXT2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(*args, **kw):
    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def shade_segments(ax, ends, speeds):
    """Alternate bands per lap, labelled with the lap name and its mean speed."""
    starts = np.concatenate([[0.0], ends[:-1]])
    for i, (t0, t1, lab) in enumerate(zip(starts, ends, SEG_LABELS)):
        if i % 2 == 0:
            ax.axvspan(t0, t1, color=BAND, zorder=0, linewidth=0)
        txt = lab if i >= len(speeds) else f"{lab}\n{speeds[i]:.2f} m/s"
        ax.text((t0 + t1) / 2, 1.0, txt, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.5, color=TEXT2, linespacing=1.1)


def plot_rq(rq_series, seg_speeds, title, path):
    conds = list(rq_series)
    fig, axes = new_fig(len(conds), 1, figsize=(10, 2.8 * len(conds)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        t, rq, ends = rq_series[cond]
        style(ax)
        shade_segments(ax, ends, seg_speeds[cond])
        ax.axhline(1.0, color=TEXT2, linewidth=0.8, linestyle="--")
        m = t <= ends[-1]
        ax.plot(t[m], rq[m], color=COLORS[cond], linewidth=2)
        ax.set_ylabel("RQ (VCO₂/VO₂)")
        ax.text(0.005, 0.95, cond, transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=TEXT, va="top")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title} - respiratory quotient per breath (dashed = 1.0; lap mean speed shown)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_cumulative(series, spec, title, path):
    fig, ax = new_fig(figsize=(8, 4.5))
    style(ax)
    for cond, (t, y, ends) in series.items():
        e = np.concatenate([[0.0], np.cumsum(np.diff(t) * (y[1:] + y[:-1]) / 2)])
        ax.plot(t, e, color=COLORS[cond], linewidth=2, label=cond)
        ax.plot(ends[-2], np.interp(ends[-2], t, e), "o", color=COLORS[cond], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{cond}  {e[-1]:.0f} {spec['total']}", (t[-1], e[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Cumulative net {spec['label']} ({spec['total']})")
    ax.set_title(f"{title} - cumulative net {spec['label']} (dot = end of walking)",
                 color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT2, loc="upper left")
    ax.margins(x=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def bar_panel(ax, summ, col, lab, fmt):
    style(ax)
    x = np.arange(len(summ))
    vals = summ[col].to_numpy()
    ax.bar(x, vals, width=0.6, color=[COLORS[c] for c in summ["condition"]],
           edgecolor=SURFACE, linewidth=2)
    pct = f"{col}_pct_vs_{REFERENCE}"
    for xi, (v, c) in enumerate(zip(vals, summ["condition"])):
        txt = fmt.format(v)
        if pct in summ and c != REFERENCE:
            txt += f"\n{summ[pct].iloc[xi]:+.1f}%"
        ax.text(xi, v, txt, ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_xticks(x, summ["condition"])
    ax.set_title(lab, fontsize=10, color=TEXT, loc="left")
    ax.set_ylim(0, vals.max() * 1.25)


def plot_summary(summ, seg_speeds, spec, title, path):
    k = spec["key"]
    fig, axes = new_fig(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 2.4, 1]})
    bar_panel(axes[0], summ, f"CoT_{k}_kg_per_m_s", f"Cost of transport ({spec['cot']})", "{:.0f}")
    bar_panel(axes[2], summ, f"net_rate_walk_{spec['rate'].replace('/', '_')}",
              f"Net rate over walking time ({spec['rate']})", "{:.2f}")

    # Lap mean speed, grouped by lap
    ax = axes[1]
    style(ax)
    conds = list(summ["condition"])
    n, w = len(conds), 0.8 / len(conds)
    x = np.arange(len(WALK_SEGMENTS))
    for i, cond in enumerate(conds):
        v = seg_speeds[cond]
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, v, width=w, color=COLORS[cond], edgecolor=SURFACE, linewidth=2, label=cond)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT)
    ax.set_xticks(x, [n for n, _ in WALK_SEGMENTS])
    ax.set_title("Mean speed per lap (m/s)", fontsize=10, color=TEXT, loc="left")
    ax.set_ylim(0, max(np.nanmax(v) for v in seg_speeds.values()) * 1.2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT2, ncol=n, loc="upper right")

    fig.suptitle(f"{title} - {spec['label']} (% change vs {REFERENCE})",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    days = [root / a for a in sys.argv[1:]] or sorted(
        d for d in root.iterdir() if (d / "Data" / "Bout.txt").exists())
    for d in days:
        process_day(d)
