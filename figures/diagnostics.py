import csv
import json
import statistics
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPATIAL = None
DATA = None
OUT = None
TEAL, BLUE, RUST, INK, GRID = ("#006861", "#5082AE", "#AB691E", "#162d38", "#dce5e7")
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6.7,
    "axes.labelsize": 6.7,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9aa9af",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "market-journal-diagnostics",
}


def rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def title(ax, letter, text):
    ax.text(0, 1.06, letter, transform=ax.transAxes, weight="bold", size=8)
    ax.text(0.075, 1.06, text, transform=ax.transAxes, weight="bold", size=7)


def save(fig, name):
    for suffix in ("pdf", "svg"):
        fig.savefig(
            OUT / f"{name}.{suffix}",
            metadata={"Creator": "Metropolitan labour-market microsimulation"},
        )
    plt.close(fig)


def criteria():
    data = rows("registered_criteria.csv")
    fig, axes = plt.subplots(2, 2, figsize=(166 / 25.4, 91 / 25.4))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.14, top=0.82, hspace=0.76, wspace=0.26)
    metrics = [
        ("U_LF_native_pct", "Unemployment", (4, 8), (2, 16)),
        ("NILF_35_44_native_pct", "Non-participation, ages 35–44", (0, 20), (0, 25)),
        ("NILF_55_64_native_pct", "Non-participation, ages 55–64", (28, 42), (24, 45)),
        ("NILF_65_99_native_pct", "Non-participation, ages 65–99", (76, 92), (72, 96)),
    ]
    for ax, letter, (metric, name, band, limits) in zip(axes.flat, "abcd", metrics):
        subset = sorted(
            (r for r in data if r["metric"] == metric), key=lambda r: int(r["policy_seed"])
        )
        assert [int(r["policy_seed"]) for r in subset] == list(range(42, 47))
        title(ax, letter, name)
        background = ax.axhspan(*band, facecolor="#e4efea", label="Registered range")
        handles = []
        for prefix, dx, mark, color, label in [
            ("full_11", -0.1, "o", TEAL, "2006–2016"),
            ("last_3", 0.1, "s", RUST, "2014–2016"),
        ]:
            values = [float(r[prefix + "_mean_pct"]) for r in subset]
            errors = [float(r[prefix + "_ci_half_pp"]) for r in subset]
            h = ax.errorbar(
                np.arange(42, 47) + dx,
                values,
                yerr=errors,
                fmt=mark,
                ms=3.3,
                lw=0.7,
                capsize=1.5,
                color=color,
                label=label,
            )
            handles.append(h)
        ax.set(xlim=(41.55, 46.45), ylim=limits, xticks=range(42, 47), ylabel="Rate (%)")
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Training seed")
    fig.legend(
        [*handles, background],
        ["2006–2016", "2014–2016", "Registered range"],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    save(fig, "registered_criteria")


def history():
    data = rows("historical_capacity_and_exits.csv")
    fig, axes = plt.subplots(1, 2, figsize=(166 / 25.4, 62 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.19, wspace=0.32)
    specs = [
        (
            "Exits from unemployment",
            "Opening unemployed per month (%)",
            [
                ("flow_U_E", "To employment", TEAL, "-"),
                ("flow_U_N", "To non-participation", BLUE, "--"),
            ],
        ),
        (
            "Funded and occupied positions",
            "Positions per 1,000 initial residents",
            [
                ("funded_active", "Funded positions", BLUE, "-"),
                ("effective_claims", "Filled + protected", TEAL, "--"),
                ("resident_filled", "Resident filled jobs", RUST, ":"),
            ],
        ),
    ]
    for ax, letter, (name, unit, series) in zip(axes, "ab", specs):
        title(ax, letter, name)
        for metric, label, col, ls in series:
            subset = [r for r in data if r["metric"] == metric]
            assert len(subset) == 132 and all((int(r["n_seeds"]) == 5 for r in subset))
            x = [int(r["year"]) + (int(r["month"]) - 0.5) / 12 for r in subset]
            ax.fill_between(
                x,
                [float(r["lower95"]) for r in subset],
                [float(r["upper95"]) for r in subset],
                color=col,
                alpha=0.12,
                lw=0,
            )
            ax.plot(x, [float(r["mean"]) for r in subset], color=col, ls=ls, lw=1.05, label=label)
        ax.set(xlim=(2006, 2017), xticks=range(2006, 2017, 2), xlabel="Calendar year", ylabel=unit)
        ax.legend(loc="upper left", frameon=False, handlelength=2)
        ax.margins(y=0.13)
        ax.grid(axis="y", color=GRID, lw=0.42)
    save(fig, "historical_capacity_and_exits")


def spatial():
    data = rows("detailed_spatial.csv")
    source = SPATIAL / "spatial_2016_summary.json"
    summaries = json.loads(source.read_text(encoding="utf8"))
    for scale in (1000, 2500):
        values = {
            int(r["seed"]): r["mean_closing_vacancies"]
            for r in summaries["per_policy"]
            if int(r["scale"]) == scale
        }
        assert set(values) == set(range(42, 47))
        for r in data:
            r[f"vacancies_{scale}"] = (
                statistics.mean(values.values()) if r["seed"] == "mean" else values[int(r["seed"])]
            )
    specs = [
        ("commute", "Assigned commute", "One-way travel time (min)", (49, 68)),
        ("access", "Access to a vacancy", "Unemployed person-months (%)", (19, 71)),
        ("vacancies", "Open vacancies", "Mean at month end (count)", (0, 4.3)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(166 / 25.4, 60 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.113, right=0.977, bottom=0.2, top=0.77, wspace=0.22)
    ys = np.arange(6)[::-1]
    for ax, letter, (metric, name, unit, limits) in zip(axes, "abc", specs):
        title(ax, letter, name)
        for j, y in enumerate(ys):
            seed = str(42 + j) if j < 5 else "mean"
            row = next((r for r in data if r["seed"] == seed))
            lo, hi = [float(row[f"{metric}_{n}"]) for n in (1000, 2500)]
            ax.plot([lo, hi], [y, y], color="#a4b2b8", lw=1.15 if j == 5 else 0.75)
            for v, m, col in [(lo, "o", TEAL), (hi, "D", BLUE)]:
                ax.plot(v, y, m, color=col, ms=4.2 if j == 5 else 3.5)
                if j == 5:
                    ax.text(
                        v,
                        y + 0.27,
                        f"{v:.2f}" if metric == "vacancies" else f"{v:.1f}",
                        ha="center",
                        color=col,
                        weight="bold",
                        size=6.3,
                    )
        ax.set(
            yticks=ys,
            yticklabels=[f"Seed {s}" for s in range(42, 47)] + ["Five-seed mean"],
            xlim=limits,
            ylim=(-0.45, 5.6),
            xlabel=unit,
        )
        ax.grid(axis="x", color=GRID, lw=0.45)
        ax.set_axisbelow(True)
    fig.legend(
        [
            mpl.lines.Line2D([], [], ls="", marker=m, color=c, ms=4)
            for m, c in [("o", TEAL), ("D", BLUE)]
        ],
        ["1,000 initial residents / 80 firms", "2,500 initial residents / 200 firms"],
        loc="upper center",
        frameon=False,
        ncol=2,
    )
    save(fig, "spatial_scale_detailed")


def acceptance():
    source = SPATIAL / "registered_acceptance_inference.json"
    stats = json.loads(source.read_text(encoding="utf8"))
    endpoints = [
        ("commute_at_finalized_match", "Commute at hire", "Travel time (min)", 1),
        ("offered_wage_at_finalized_match", "Offered hourly wage", "Model hourly wage units", 1),
        ("match_quality_at_finalized_match", "Match quality", "Quality score", 1),
        ("equalweight_macro_MAE", "Holdout error", "Mean absolute error (percentage points)", 100),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(166 / 25.4, 93 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.134, right=0.972, top=0.91, bottom=0.12, hspace=0.76, wspace=0.23)
    for ax, letter, (metric, name, unit, scale) in zip(axs.flat, "abcd", endpoints):
        title(ax, letter, name)
        for j in range(6):
            summary = j == 5
            r = (
                stats["across_five_policies_df4"][metric]
                if summary
                else stats["per_policy_paired_df49"][str(42 + j)][metric]
            )
            y = 5 - j
            lo, mu, hi = [scale * r[k] for k in ("ci95_low", "mean", "ci95_high")]
            ax.hlines(y, lo, hi, color=TEAL if summary else INK, lw=1.45 if summary else 0.8)
            ax.plot(
                mu,
                y,
                "D" if summary else "o",
                mfc=TEAL if summary else "white",
                mec=TEAL if summary else INK,
                ms=4 if summary else 3.1,
            )
        ax.axvline(0, color=INK, lw=0.65, ls="--")
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set(
            yticks=range(5, -1, -1),
            yticklabels=[f"Seed {s}" for s in range(42, 47)] + ["Five-seed mean"],
            ylim=(-0.5, 5.5),
            xlabel=unit,
        )
        ax.locator_params(axis="x", nbins=5)
    save(fig, "acceptance_commute_detailed")
