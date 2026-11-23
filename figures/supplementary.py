import string
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = None
CORE = None
CONTROLS = None
OUT = None
WIDTH = 180 / 25.4
T4 = 2.7764451051977987
TEAL, BLUE, ORANGE, PURPLE = ("#006861", "#5082AE", "#AB691E", "#4D365F")
INK, MUTED, GRID = ("#233746", "#526878", "#DCE4E8")
SHADE, ANCHOR = ("#F2EDE2", "#946633")
METRICS = ["employment_rate", "participation_rate", "unemployment_rate"]
NAMES = ["Employment / population", "Labour-force participation", "Unemployment / labour force"]
TRAIN = "trained"
COLD = ["untrained_random", "untrained_heuristic"]
GROUP = [
    (TRAIN, TEAL, "-", "Jointly trained"),
    (COLD[0], PURPLE, "--", "Untrained random"),
    (COLD[1], ORANGE, ":", "Untrained + heuristic firms"),
]
REGIME = [("ON", TEAL, "-", "Pandemic regime on"), ("OFF", BLUE, ":", "Pandemic regime off")]
MECHANISMS = [
    ("U_matching_pool", "Matching pool", "People / 1,000 initial"),
    ("offer_reach_pct", "Offer reach", "Pool participants offered (%)"),
    ("precommit_conversion_pct", "Pre-commit conversion", "Recipients matched (%)"),
    ("U_committed_hires", "Hires from unemployment", "Hires / 1,000 initial / month"),
    ("flow_U_E_pct", "Unemployment to employment", "Opening unemployed (%)"),
    ("duration_raw_months", "Ongoing unemployment", "Mean elapsed time (months)"),
    ("RL_postings", "Vacancy additions", "Additions / 1,000 initial / month"),
    ("funded_headroom", "Funded headroom", "Positions / 1,000 initial"),
    ("funded_capacity_occupancy_pct", "Capacity occupancy", "Effective claims / capacity (%)"),
]


def read(folder, stem):
    return pd.read_csv(folder / f"{stem}.csv")


def select(data, case, metric):
    return data.loc[(data["case"] == case) & (data.metric == metric)].copy()


def xvalues(d):
    x = d.year.to_numpy(dtype=float)
    return x + (d.month.to_numpy() - 0.5) / 12 if "month" in d.columns else x


def line(ax, d, color, ls="-", factor=1):
    assert len(d) and (not d.duplicated(["year", "month"] if "month" in d else ["year"]).any())
    for field in ("n_policies", "n_seeds"):
        if field in d:
            assert d[field].eq(5).all()
    d = d.sort_values(["year", "month"] if "month" in d else ["year"])
    x = xvalues(d)
    ax.fill_between(x, d.lower95 * factor, d.upper95 * factor, color=color, linewidth=0, alpha=0.13)
    ax.plot(x, d["mean"] * factor, color=color, ls=ls, lw=1.35)


def observed(ax, data, metric, relative=False):
    values = data[metric]
    if relative:
        reference = data.loc[data.year == 2019].set_index("month")[metric]
        assert len(reference) == 12
        values = values - data.month.map(reference)
    ax.plot(xvalues(data), 100 * values, color=INK, ls="--", lw=1.15, zorder=4)


def figure(rows, cols, height, *, top=0.88, bottom=0.115, wspace=0.4, hspace=0.6, left=0.105):
    fig, axs = plt.subplots(rows, cols, figsize=(WIDTH, height), squeeze=False)
    fig.subplots_adjust(left=left, right=0.98, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    for ax in axs.flat:
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(length=2.4, pad=2.2)
    return (fig, axs)


def panel(ax, letter, title, ylabel=None):
    ax.set_title(f"{letter}   {title}", fontweight="bold", loc="left", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)


def calendar(ax, start, end, monthly=False, covid=False, label=False):
    ax.set_xlim(start, end + (1 if monthly else 0))
    ax.set_xticks(np.arange(start, end + 1, 2 if end - start < 14 else 4))
    if label:
        ax.set_xlabel("Calendar year")
    if covid:
        ax.axvspan(2020, 2022, color=SHADE, zorder=-2)


def keys(fig, specs, *, ncol=None):
    lines = [Line2D([0], [0], color=c, ls=ls, lw=1.5, label=l) for c, ls, l in specs]
    fig.legend(
        handles=lines,
        loc="upper center",
        ncol=ncol or len(lines),
        frameon=False,
        bbox_to_anchor=(0.5, 0.997),
        borderaxespad=0,
        handlelength=2.3,
        columnspacing=1.05,
    )


def save(fig, name):
    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", facecolor="white")
    plt.close(fig)


def age_criteria():
    age = read(CORE, "age_stocks")
    age = age[age.population.eq("native")]
    criteria = read(CORE, "registered_criteria")
    fig, axs = figure(2, 3, 4.48, top=0.8, bottom=0.105, hspace=0.78, wspace=0.46)
    order = ["15-24", "25-54", "55-64", "65+"]
    for j, (metric, name) in enumerate(zip(["A2", "A3", "A1"], NAMES)):
        ax = axs[0, j]
        for scale, col, off, sym in [(1000, TEAL, -0.12, "o"), (2500, BLUE, 0.12, "s")]:
            d = age[(age.scale == scale) & (age.code == metric)].set_index("group").reindex(order)
            defined = d.five_seed_mean.notna() & d.anchor_full.notna()
            if metric == "A1":
                assert list(d.index[~defined]) == ["65+"]
            else:
                assert defined.all()
            gap = d.five_seed_mean - d.anchor_full
            assert np.allclose(gap[defined], d.loc[defined, "five_seed_bias"])
            ax.errorbar(
                np.arange(4)[defined] + off,
                gap[defined],
                yerr=d.loc[defined, "five_seed_ci95_half"],
                fmt=sym,
                ms=3,
                color=col,
                capsize=2,
                lw=0.9,
            )
        ax.axhline(0, color=INK, lw=0.75, zorder=1)
        if metric == "A1":
            ax.text(3, -3.8, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
        panel(ax, string.ascii_lowercase[j], name, "Model minus survey (pp)" if j == 0 else None)
        ax.set_xticks(range(4), order)
        ax.set_xlim(-0.35, 3.35)
        ax.set_xlabel("Age (years)")
        ax.set_ylim(-12, 17)
        ax.set_yticks([-10, -5, 0, 5, 10, 15])
    for j, (metric, name, lo, hi) in enumerate(
        [
            ("U_LF_native_pct", "Unemployment", 4, 8),
            ("NILF_55_64_native_pct", "Non-participation, ages 55–64", 28, 42),
        ]
    ):
        ax = axs[1, j]
        ax.axhspan(lo, hi, color="#E4EEE9", zorder=-2)
        for case, col, off, sym in [
            ("historical1000", TEAL, -0.12, "o"),
            ("historical2500", BLUE, 0.12, "s"),
        ]:
            d = select(criteria, case, metric).sort_values("policy_seed")
            assert list(d.policy_seed) == list(range(42, 47))
            ax.errorbar(
                d.policy_seed + off,
                d.full_11_mean_pct,
                yerr=d.full_11_ci_half_pp,
                fmt=sym,
                ms=3,
                color=col,
                capsize=2,
                lw=0.9,
            )
        panel(ax, string.ascii_lowercase[3 + j], name, "%")
        ax.set_xticks(range(42, 47))
        ax.set_xlabel("Training seed")
    axs[1, 2].axis("off")
    axs[1, 2].text(0, 0.95, "Original criteria", fontweight="bold", transform=axs[1, 2].transAxes)
    labels = {
        "U_LF_native_pct": "Unemployment",
        "NILF_35_44_native_pct": "Ages 35–44",
        "NILF_55_64_native_pct": "Ages 55–64",
        "NILF_65_99_native_pct": "Ages 65–99",
    }
    passed = criteria.assign(passed=criteria.final_classification.eq("PASS")).pivot_table(
        index="metric", columns="case", values="passed", aggfunc="sum"
    )
    small = passed["historical1000"].reindex(labels)
    large = select(criteria, "historical2500", "NILF_55_64_native_pct")
    if (
        small.tolist() == [0, 5, 5, 5]
        and large.final_classification.nunique() > 1
        and large.full_11_verdict.ne(large.last_3_verdict).any()
    ):
        summary = "1,000 initial residents\nAll five fail unemployment;\nall five pass the three\nnon-participation bands.\n\n2,500 initial residents\nAges 55–64 vary by seed\nand scoring window."
    else:
        counts = ["Policies passing both windows", "1,000 and 2,500 residents"]
        for key, label in labels.items():
            counts.append(f"{label}: {small[key]}/5 and {passed.at[key, 'historical2500']}/5")
        summary = "\n".join(counts)
    axs[1, 2].text(0, 0.81, summary, va="top", fontsize=7.4, transform=axs[1, 2].transAxes)
    handles = [
        Line2D([], [], ls="", marker="o", color=TEAL, label="1,000 initial residents"),
        Line2D([], [], ls="", marker="s", color=BLUE, label="2,500 initial residents"),
        Line2D([], [], lw=0.85, color=INK, label="Zero difference"),
        Patch(facecolor="#E4EEE9", label="Registered band (d,e)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=4,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.05,
    )
    fig.text(
        0.105,
        0.904,
        "Non-immigrant model rate minus all-person regional survey rate · 2006–2016",
        color=MUTED,
    )
    fig.text(
        0.105,
        0.446,
        "Registered non-immigrant criteria · intervals across 50 histories within each policy",
        color=MUTED,
    )
    save(fig, "age_criteria")


def scale_comparison():
    fig, axs = figure(2, 3, 4.07, top=0.82, bottom=0.11, hspace=0.73)
    cohort = read(CORE, "historical")
    reference = read(CORE, "historical_reference")
    tables = [
        read(CORE, "historical_seeds"),
        read(ROOT / "evaluation", "HISTORICAL_RESIDENT_SEED_CURVES"),
    ]
    for j, metric in enumerate(METRICS):
        ax = axs[0, j]
        for case, color, ls in [("historical1000", TEAL, "-"), ("historical2500", BLUE, "--")]:
            line(ax, select(cohort, case, metric), color, ls, 100)
        observed(ax, reference, metric)
        panel(ax, string.ascii_lowercase[j], NAMES[j], "%")
        calendar(ax, 2006, 2016)
        ax = axs[1, j]
        for data, color, ls in zip(tables, [TEAL, ORANGE], ["-", "--"]):
            wide = data.pivot(index=["year", "policy_seed"], columns="case", values=metric)
            diff = (wide.historical2500 - wide.historical1000) * 100
            by_year = diff.groupby(level="year")
            mean = by_year.mean()
            assert len(mean) == 11 and by_year.count().eq(5).all()
            h = T4 * by_year.std() / np.sqrt(5)
            series = pd.DataFrame(
                {"mean": mean, "lower95": mean - h, "upper95": mean + h}
            ).reset_index()
            line(ax, series, color, ls)
        ax.axhline(0, color=INK, lw=0.75, zorder=1)
        panel(ax, string.ascii_lowercase[j + 3], "2,500 minus 1,000", "Percentage points")
        calendar(ax, 2006, 2016, label=True)
    keys(
        fig,
        [(TEAL, "-", "1,000 residents"), (BLUE, "--", "2,500 residents"), (INK, "--", "Observed")],
    )
    fig.text(0.105, 0.915, "Levels · cohort / search-adjusted rates", color=MUTED)
    fig.text(0.105, 0.46, "Paired scale effects across policies", color=MUTED)
    fig.legend(
        handles=[
            Line2D([], [], color=TEAL, label="Cohort / search-adjusted"),
            Line2D([], [], color=ORANGE, ls="--", label="Current-resident raw stocks"),
        ],
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.44),
        handlelength=1.8,
    )
    save(fig, "scale_sensitivity")


def recruitment_paths():
    d = read(CONTROLS, "MECHANISM_ANNUAL_CURVES")
    fig, axs = figure(3, 3, 5.9, top=0.9, bottom=0.085, hspace=0.65, wspace=0.46)
    for i, (metric, title, unit) in enumerate(MECHANISMS):
        ax = axs.flat[i]
        for case, c, ls, _ in GROUP:
            line(ax, select(d, case, metric), c, ls)
        panel(ax, string.ascii_lowercase[i], title, unit)
        calendar(ax, 2006, 2016, label=i >= 6)
    keys(fig, [(c, ls, l) for _, c, ls, l in GROUP])
    save(fig, "recruitment_paths")


def transition_paths():
    cases = ["forward_RZ_OFF", "forward_RZ_ON"]
    states = [("E", "U"), ("E", "N"), ("U", "E"), ("U", "N"), ("N", "E"), ("N", "U")]
    metric_order = [f"flow_{a}_{b}" for a, b in states]
    catalogue = pd.read_json(ROOT / "evaluation/paired_histories.json")
    registry = catalogue.loc[catalogue.case.isin(cases)]
    identifiers = ["policy_seed", "replication_index", "evaluation_seed", "checkpoint_sha256"]
    off_keys, on_keys = [
        set(registry.loc[registry.case.eq(case), identifiers].itertuples(index=False, name=None))
        for case in cases
    ]
    assert off_keys == on_keys and len(off_keys) == 250 and (len(registry) == 500)
    assert registry.groupby(["case", "policy_seed"]).size().eq(50).all()
    monthly = read(CORE, "mechanism_policy")
    monthly = monthly.loc[
        monthly.case.isin(cases)
        & monthly.metric.isin(metric_order)
        & monthly.year.between(2020, 2021)
        & monthly.dataset.eq("model")
    ]
    expected = pd.MultiIndex.from_product(
        [range(42, 47), metric_order, [2020, 2021], range(1, 13), cases],
        names=["policy_seed", "metric", "year", "month", "case"],
    )
    cells = monthly.set_index(expected.names)
    assert cells.index.is_unique and cells.index.sort_values().equals(expected.sort_values())
    assert (monthly.denominator_sum > 0).all()
    assert np.allclose(monthly.value, 100 * monthly.numerator_sum / monthly.denominator_sum)
    rates = cells.value.unstack("case").groupby(["policy_seed", "metric"]).mean()
    rates["difference"] = rates[cases[1]] - rates[cases[0]]
    policy_means = rates.groupby("metric").mean().loc[metric_order]
    halves = rates.difference.groupby("metric").std().loc[metric_order] * T4 / np.sqrt(5)
    fig, axs = figure(1, 2, 2.97, left=0.255, top=0.74, bottom=0.205, wspace=0.17)
    level, effect = axs[0]
    yy = np.arange(len(states))[::-1]
    display = {"E": "Employed", "U": "Unemployed", "N": "Outside labour force"}
    for ax in axs[0]:
        ax.set_yticks(yy)
        ax.set_ylim(-0.53, 5.52)
        ax.grid(axis="y", color=GRID, linewidth=0.65)
        ax.tick_params(axis="y", length=0)
    level.set_yticklabels([f"{display[a]} → {display[b].lower()}" for a, b in states])
    effect.tick_params(axis="y", labelleft=False)
    off, on = (policy_means[case].to_numpy() for case in cases)
    for j, y in enumerate(yy):
        level.plot([off[j], on[j]], [y - 0.08, y + 0.08], color="#84929A", linewidth=0.85)
    level.scatter(
        off,
        yy - 0.08,
        s=16,
        marker="s",
        facecolor="white",
        edgecolor=BLUE,
        linewidth=0.95,
        zorder=3,
        label="Regime off",
    )
    level.scatter(on, yy + 0.08, s=19, color=TEAL, linewidth=0, zorder=3, label="Regime on")
    level.set_xlim(0, 6)
    level.set_xticks([0, 1, 2, 3, 4, 5, 6])
    level.set_xlabel("Monthly rate (% of opening population)")
    effect.axvline(0, color=INK, linewidth=0.8)
    for j, k in enumerate(metric_order):
        sample = rates.xs(k, level="metric").difference.sort_index()
        effect.scatter(
            sample.values,
            yy[j] - 0.2 + np.linspace(-0.095, 0.095, 5),
            s=10.5,
            facecolor=BLUE,
            linewidth=0,
            zorder=3,
        )
    effect.errorbar(
        policy_means.difference,
        yy + 0.06,
        xerr=halves,
        fmt="D",
        markersize=4,
        color=INK,
        elinewidth=1.0,
        capsize=2,
        zorder=4,
    )
    effect.set_xlim(-0.24, 1.87)
    effect.set_xticks([0, 0.5, 1, 1.5])
    effect.set_xlabel("Regime on minus off (percentage points)")
    panel(level, "a", "Monthly transition rates, 2020–2021")
    panel(effect, "b", "Paired change over the same months")
    legend_handles = [
        Line2D([], [], ls="", marker="s", markersize=4, mfc="white", mec=BLUE, label="Regime off"),
        Line2D([], [], ls="", marker="o", markersize=4, mfc=TEAL, mec=TEAL, label="Regime on"),
        Line2D(
            [], [], ls="", marker="o", markersize=3, mfc=BLUE, mec=BLUE, label="Seed difference"
        ),
        Line2D([], [], lw=1.05, marker="D", markersize=4, color=INK, label="Mean and 95% interval"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.55, 0.97),
        handlelength=1.3,
        columnspacing=1.45,
    )
    save(fig, "six_transition_paths")


def pandemic_arms():
    levels, changes, obs = (read(CORE, k) for k in ["forward", "changes", "observed"])
    fig, axs = figure(5, 2, 7.28, top=0.916, bottom=0.065, hspace=0.58, wspace=0.26, left=0.11)
    fig.text(0.14, 0.954, "Realised separation inputs", fontweight="bold")
    fig.text(0.6, 0.954, "Frozen-2019 separation inputs", fontweight="bold")
    for i, (metric, title) in enumerate(
        zip(
            METRICS + ["temp_layoff_share_population", "marginal_share_population"],
            NAMES + ["Temporary layoff", "Outside labour force, wanting work"],
        )
    ):
        for j, route in enumerate(["RZ", "FN"]):
            ax = axs[i, j]
            for flag, c, ls, _ in REGIME:
                line(
                    ax,
                    select(levels if i < 3 else changes, f"forward_{route}_{flag}", metric),
                    c,
                    ls,
                    100,
                )
            observed(ax, obs, metric, i >= 3)
            if i >= 3:
                ax.axhline(0, color=MUTED, lw=0.65)
            panel(ax, string.ascii_lowercase[2 * i + j], title, "%" if i < 3 else "Change (pp)")
            ax.axvline(2020 + 3.5 / 12, color=ANCHOR, ls="-.", lw=0.8)
            calendar(ax, 2017, 2023, monthly=True, covid=True, label=i == 4)
        lo = min((ax.get_ylim()[0] for ax in axs[i]))
        hi = max((ax.get_ylim()[1] for ax in axs[i]))
        for ax in axs[i]:
            ax.set_ylim(lo, hi)
    keys(
        fig,
        [(c, ls, l) for _, c, ls, l in REGIME]
        + [(INK, "--", "Observed"), (ANCHOR, "-.", "April 2020 anchor")],
    )
    save(fig, "pandemic_four_conditions")


def pandemic_effects():
    d = read(CORE, "paired")
    fig, axs = figure(1, 3, 2.38, top=0.79, bottom=0.2)
    for j, metric in enumerate(METRICS):
        ax = axs[0, j]
        for case, c, ls in [("RZ_ON_minus_OFF", TEAL, "-"), ("FN_ON_minus_OFF", PURPLE, "--")]:
            line(ax, select(d, case, metric), c, ls, 100)
        ax.axhline(0, color=INK, lw=0.65)
        panel(ax, string.ascii_lowercase[j], NAMES[j], "On − off (pp)")
        calendar(ax, 2017, 2023, monthly=True, covid=True, label=True)
    keys(fig, [(TEAL, "-", "Realised separations"), (PURPLE, "--", "Frozen-2019 separations")])
    save(fig, "pandemic_paired_effects")


def error_contrasts():
    data = read(CONTROLS, "MACRO_PAIRED_POLICY_CONTRASTS")
    scores = read(CONTROLS, "MACRO_POLICY_SCORES")
    fig, axs = figure(2, 3, 3.85, top=0.86, bottom=0.13, left=0.15, wspace=0.33, hspace=0.74)
    selection = data[(data.estimand == "annual_path_MAE_pp") & data.comparison.isin(COLD)]
    for j, metric in enumerate(METRICS):
        ax = axs[0, j]
        blocks = scores[scores.metric.eq(metric) & scores["case"].isin([TRAIN, *COLD])]
        paired = blocks.pivot(index="policy_seed", columns="case", values="annual_path_MAE_pp")
        assert list(paired.index) == list(range(42, 47)) and (not paired.isna().any().any())
        for _, row in paired.iterrows():
            ax.plot(range(3), row.loc[[TRAIN, *COLD]], color="#A6B3BD", lw=0.75, zorder=1)
        for k, (case, color, _, _) in enumerate(GROUP):
            ax.scatter(np.repeat(k, 5), paired[case], s=10, color=color, zorder=2)
            ax.scatter(k, paired[case].mean(), marker="_", s=115, color=INK, zorder=3)
        ax.set_xlim(-0.28, 2.28)
        ax.set_xticks(range(3), ["Trained", "Random", "Heuristic"])
        ax.set_ylim(bottom=0)
        panel(ax, string.ascii_lowercase[j], NAMES[j], "MAE (pp)")
        ax = axs[1, j]
        for k, (family, color) in enumerate(zip(COLD, [PURPLE, ORANGE])):
            s = selection[
                (selection.metric == metric) & (selection.comparison == family)
            ].sort_values("policy_seed")
            assert list(s.policy_seed) == list(range(42, 47))
            assert np.allclose(s.value, paired[TRAIN] - paired[family])
            mean = s.value.mean()
            h = T4 * s.value.std() / np.sqrt(5)
            ax.scatter(s.value, k + 0.2 + np.linspace(-0.07, 0.07, 5), color=color, s=8, zorder=4)
            ax.errorbar(mean, k, xerr=h, fmt="D", ms=4, elinewidth=1.35, capsize=2, color=color)
        ax.axvline(0, color=INK, lw=0.75)
        ax.set_ylim(1.55, -0.45)
        ax.set_yticks([0, 1], ["Random", "Heuristic firms"] if j == 0 else [])
        panel(ax, string.ascii_lowercase[j + 3], "Paired error difference")
        ax.set_xlabel("MAE difference (pp)")
    fig.text(
        0.15,
        0.95,
        "Mean absolute error by training seed · each line joins the same seed",
        color=MUTED,
    )
    fig.text(
        0.15,
        0.455,
        "Jointly trained minus control · negative values indicate closer empirical agreement",
        color=MUTED,
    )
    save(fig, "control_fit_errors")


def flow_wages():
    annual = read(CONTROLS, "FLOW_WAGE_ANNUAL_CURVES")
    specs = [
        ("raw_E_to_U_pct", "Employment to unemployment", "% of opening employed"),
        ("raw_E_to_N_pct", "Employment to non-participation", "% of opening employed"),
        ("N_to_E_per1000", "NILF to employment", "Entries / 1,000 initial / month"),
        (
            "positive_hourly_wage",
            "Mean hourly wage among employed workers",
            "Hourly wage (model units)",
        ),
    ]
    fig = plt.figure(figsize=(WIDTH, 3.82))
    layout = fig.add_gridspec(
        2, 6, left=0.095, right=0.982, top=0.835, bottom=0.14, wspace=2.0, hspace=1.07
    )
    time_axes = [fig.add_subplot(layout[0, j : j + 2]) for j in (0, 2, 4)]
    time_axes.append(fig.add_subplot(layout[1, :3]))
    for i, (ax, (metric, title, unit)) in enumerate(zip(time_axes, specs)):
        for case, color, style, _ in GROUP:
            line(ax, select(annual, case, metric), color, style)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(length=2.4, pad=2.2)
        calendar(ax, 2006, 2016, label=True)
        ax.set_xticks([2006, 2011, 2016])
        panel(ax, string.ascii_lowercase[i], title, unit)
    wage_axis = fig.add_subplot(layout[1, 3:])
    wages = read(ROOT / "wages", "wages_summary")
    for y, (scale, color, marker) in enumerate([(2500, BLUE, "s"), (1000, TEAL, "o")]):
        worker = wages[wages.scale.eq(scale) & wages.group.eq("worker")].set_index("metric")["mean"]
        low, mid, high = worker[["p10", "median", "p90"]]
        assert 0 < low <= mid <= high
        wage_axis.plot([low, high], [y, y], lw=1.4, color=color, marker="|", ms=6)
        wage_axis.plot(mid, y, marker=marker, color=color, ms=4)
    wage_axis.set_yticks([0, 1], ["2,500", "1,000"])
    wage_axis.set_ylabel("Initial residents")
    wage_axis.set_ylim(-0.6, 1.6)
    wage_axis.set_xlim(12, 45)
    wage_axis.set_xticks([15, 20, 25, 30, 35, 40, 45])
    wage_axis.set_xlabel("Hourly wage (model units)")
    wage_axis.set_axisbelow(True)
    wage_axis.grid(axis="x", color=GRID, linewidth=0.55)
    wage_axis.tick_params(length=2.4, pad=2.2)
    panel(wage_axis, "e", "Realised wage median and 10th–90th percentiles")
    keys(fig, [(color, style, name) for _, color, style, name in GROUP])
    save(fig, "flows_wages")


def policy_paths():
    fig, axs = figure(2, 3, 3.95, top=0.82, bottom=0.13, hspace=0.8)
    specs = [
        ("historical_seeds", "historical1000", "historical_reference", 2006, 2016),
        ("forward_seeds", "forward_RZ_ON", "observed", 2017, 2023),
    ]
    for i, (file, case, reference, start, end) in enumerate(specs):
        d = read(CORE, file)
        d = d[d["case"].eq(case)]
        for j, metric in enumerate(METRICS):
            ax = axs[i, j]
            for _, g in d.groupby("policy_seed"):
                ax.plot(xvalues(g), 100 * g[metric], color="#889BA5", lw=0.7)
            group = ["year", "month"] if i else ["year"]
            avg = d.groupby(group)[metric].mean().reset_index()
            assert d.groupby(group).size().eq(5).all()
            ax.plot(xvalues(avg), 100 * avg[metric], color=TEAL, lw=1.55)
            observed(ax, read(CORE, reference), metric)
            panel(ax, string.ascii_lowercase[i * 3 + j], NAMES[j], "%")
            calendar(ax, start, end, monthly=bool(i), covid=bool(i), label=bool(i))
    keys(
        fig,
        [
            ("#889BA5", "-", "Five individual policies"),
            (TEAL, "-", "Equal-policy mean"),
            (INK, "--", "Observed"),
        ],
    )
    fig.text(
        0.105,
        0.904,
        "Annual historical cohort rates · Toronto, Oshawa and Hamilton reference",
        color=MUTED,
    )
    fig.text(
        0.105,
        0.458,
        "Monthly forward rates · realised separations, regime on · Toronto–Hamilton reference",
        color=MUTED,
    )
    save(fig, "policy_seed_trajectories")


def durations():
    d = read(CORE, "mechanism")
    fig, axs = figure(2, 2, 4.25, top=0.875, bottom=0.13, hspace=0.72, wspace=0.33)
    for j, (metric, title) in enumerate(
        [
            ("elapsed_U_stock_months", "All currently unemployed"),
            ("elapsed_active_U_stock_months", "Currently unemployed and searching"),
        ]
    ):
        ax = axs[0, j]
        for flag, c, ls, _ in REGIME:
            line(ax, select(d, "forward_RZ_" + flag, metric), c, ls, 4.33)
        panel(ax, string.ascii_lowercase[j], title, "Elapsed duration (weeks)")
        calendar(ax, 2006, 2023, monthly=True, covid=True)
    for j, (file, col, order, labels, title, unit) in enumerate(
        [
            (
                "age_duration",
                "age_group",
                ["15-24", "25-34", "35-44", "45-54", "55-64"],
                ["15–24", "25–34", "35–44", "45–54", "55–64"],
                "Active-search unemployment by age",
                "Elapsed duration (weeks)",
            ),
            (
                "duration_bins",
                "duration_bin",
                ["dur_0_6", "dur_6_12", "dur_12_inf"],
                ["0–6", "6–12", "12+"],
                "Pre-commit matches by spell age",
                "Matching participants (%)",
            ),
        ]
    ):
        ax = axs[1, j]
        frame = read(CORE, file)
        g = frame.groupby(col).value
        assert g.count().reindex(order).eq(5).all()
        mean = g.mean().reindex(order)
        h = T4 * g.std().reindex(order) / np.sqrt(5)
        ax.errorbar(range(len(order)), mean, yerr=h, fmt="o", color=TEAL, ms=4, capsize=3)
        ax.set_xticks(range(len(order)), labels)
        ax.set_xlabel("Worker age (years)" if j == 0 else "Elapsed duration bin (model months)")
        ax.set_ylim(bottom=0)
        panel(ax, string.ascii_lowercase[j + 2], title, unit)
    keys(fig, [(c, ls, l) for _, c, ls, l in REGIME])
    save(fig, "unemployment_duration")


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 7.9,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A9B6BB",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "labour-figures",
        }
    )
