import string
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import supplementary as style

INPUT = None
DEST = None
CASES = ["firm_frozen", "bundle_off"]
SEEDS = {"firm_frozen": [42, 43, 44, 45, 46], "bundle_off": [42, 43]}
ROW_LABELS = ["Fixed firm rule\n(5 seeds)", "Reservation-wage\nbundle off (2 seeds)"]
SEED_MARKERS = ["o", "s", "^", "D", "v"]
FILL = ["white", style.TEAL, "white", style.TEAL, "white"]


def read_inputs():
    annual = pd.read_csv(INPUT / "macro_annual_policy.csv")
    window = pd.read_csv(INPUT / "mechanism_policy.csv")
    baseline = pd.read_csv(style.CORE / "historical_reference.csv")
    for case, seeds in [(style.TRAIN, list(range(42, 47))), *SEEDS.items()]:
        subset = annual[annual["case"].eq(case)]
        assert set(subset["policy_seed"]) == set(seeds)
        assert not subset.duplicated(["policy_seed", "year"]).any()
        assert all(
            (set(g["year"]) == set(range(2006, 2017)) for _, g in subset.groupby("policy_seed"))
        )
        assert np.isfinite(subset[style.METRICS]).all().all()
        assert subset[style.METRICS].ge(0).all().all() and subset[style.METRICS].le(1).all().all()
    return (annual, window, baseline)


def contrast(values, case):
    required = SEEDS[case]
    d = values[values["case"].isin([style.TRAIN, case])].pivot(
        index="policy_seed", columns="case", values="value"
    )
    assert set(d[case].dropna().index) == set(required)
    paired = d.loc[required, style.TRAIN] - d.loc[required, case]
    assert paired.notna().all()
    return paired


def interval(values):
    return style.T4 * values.std(ddof=1) / np.sqrt(5)


def difference_rows(ax, values, row_labels):
    limits = [0.0]
    for row, case in enumerate(CASES):
        paired = contrast(values, case)
        center = float(paired.mean())
        limits.extend(paired.values)
        ordinate = 1.2 * row
        offsets = np.linspace(-0.2, 0.12, len(paired))
        for (seed, v), dy in zip(paired.items(), offsets):
            idx = int(seed) - 42
            ax.scatter(
                v,
                ordinate + dy,
                s=18,
                marker=SEED_MARKERS[idx],
                edgecolors=style.TEAL,
                facecolors=FILL[idx],
                linewidths=0.9,
                zorder=3,
            )
        if len(paired) == 5:
            error = interval(paired)
            ax.hlines(
                ordinate + 0.32,
                center - error,
                center + error,
                color=style.INK,
                linewidth=1.0,
                zorder=2,
            )
            limits.extend((center - error, center + error))
        ax.plot(
            center,
            ordinate + 0.32,
            marker="|",
            markersize=6.5,
            markeredgewidth=1.25,
            color=style.INK,
            linestyle="",
            zorder=4,
        )
    ax.axvline(0, color=style.MUTED, lw=0.7, zorder=1)
    ax.set_ylim(1.76, -0.43)
    ax.set_yticks([0.05, 1.25], ROW_LABELS if row_labels else ["", ""])
    ax.tick_params(axis="y", length=0)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=style.GRID, linewidth=0.5)
    return limits


def seed_legend(fig, y):
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=m,
            color=style.TEAL,
            markerfacecolor=fill,
            markersize=4,
            label=str(seed),
        )
        for seed, m, fill in zip(range(42, 47), SEED_MARKERS, FILL)
    ]
    handles.append(
        Line2D([], [], linestyle="", marker="|", color=style.INK, markersize=7, label="Mean")
    )
    handles.append(Line2D([], [], linewidth=1, color=style.INK, label="95% interval (5 seeds)"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.53, y),
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.3,
        handletextpad=0.35,
        frameon=False,
    )


def save(fig, filename):
    for ext in ["pdf", "svg"]:
        fig.savefig(DEST / f"{filename}.{ext}", facecolor="white")
    plt.close(fig)


def macro(annual, observed):
    fig, axs = style.figure(
        2, 3, 4.35, top=0.848, bottom=0.11, left=0.152, wspace=0.35, hspace=1.12
    )
    curve_cases = [
        (style.TRAIN, style.TEAL, "-", "Jointly trained"),
        ("firm_frozen", style.BLUE, "--", "Fixed firm rule"),
        ("bundle_off", style.ORANGE, ":", "Reservation-wage bundle off"),
    ]
    for j, (metric, title) in enumerate(zip(style.METRICS, style.NAMES)):
        ax = axs[0, j]
        for case, color, ls, label in curve_cases:
            matched = annual[annual["case"].eq(case) & annual.policy_seed.isin([42, 43])]
            m = matched.groupby("year")[metric].mean()
            ax.plot(m.index, m.values * 100, color=color, ls=ls, lw=1.35)
        ax.plot(observed["year"], 100 * observed[metric], color=style.INK, ls="-.", lw=1.05)
        style.panel(ax, string.ascii_lowercase[j], title, "%")
        ax.set_xlim(2006, 2016)
        ax.set_xticks([2006, 2011, 2016])
    style.keys(
        fig,
        [(c, ls, label) for _, c, ls, label in curve_cases]
        + [(style.INK, "-.", "Regional survey")],
    )
    fig.text(
        0.152,
        0.915,
        "Annual trajectories on the two seeds shared by all configurations",
        color=style.MUTED,
    )
    seed_legend(fig, 0.49)
    limits = []
    for j, (metric, title) in enumerate(zip(style.METRICS, style.NAMES)):
        ax = axs[1, j]
        mean = (
            annual.groupby(["case", "policy_seed"], as_index=False)[metric]
            .mean()
            .rename(columns={metric: "value"})
        )
        mean["value"] *= 100
        limits.extend(difference_rows(ax, mean, row_labels=j == 0))
        style.panel(
            ax, string.ascii_lowercase[j + 3], ["Employment", "Participation", "Unemployment"][j]
        )
        ax.set_xlabel("Trained minus intervention (pp)")
    maximum = np.ceil(2 * max((abs(v) for v in limits))) / 2
    for ax in axs[1]:
        ax.set_xlim(-maximum * 1.06, maximum * 1.06)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    save(fig, "two_policy_macro")


def mechanisms(window):
    fig, axs = style.figure(
        3, 3, 5.55, top=0.875, bottom=0.09, left=0.156, wspace=0.39, hspace=1.05
    )
    seed_legend(fig, 0.993)
    fig.text(
        0.156,
        0.928,
        "Paired difference in 2006–2016 window estimates (jointly trained minus intervention)",
        color=style.MUTED,
    )
    for i, (metric, title, unit) in enumerate(style.MECHANISMS):
        ax = axs.flat[i]
        data = window[window.metric.eq(metric)]
        limits = difference_rows(ax, data, row_labels=i % 3 == 0)
        low, high = (min(limits), max(limits))
        assert high > low, metric
        span = high - low
        ax.set_xlim(low - 0.12 * span, high + 0.12 * span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        style.panel(ax, string.ascii_lowercase[i], title)
        ax.set_xlabel("Percentage points" if metric.endswith("_pct") else unit)
    save(fig, "two_policy_matching")
