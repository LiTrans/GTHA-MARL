import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import matplotlib

matplotlib.use("Agg")
from matplotlib import rc_context
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import criteria_panel
import diagnostics
import firms
import learning
import spatial_pandemic
import supplementary
import tariff
import training


def plot_history(path, output):
    data = pd.read_csv(path)
    ids = ["policy_seed", "replication_index", "evaluation_seed"]
    rates = ["employment_rate", "participation_rate", "unemployment_rate"]
    required = {"year", "month", *ids, *rates}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Monthly history is missing {', '.join(sorted(missing))}")
    identities = data[ids].drop_duplicates()
    if len(identities) != 1 or identities.isna().any().any():
        raise ValueError(
            "Supply one nonempty history with a single policy, replication and evaluation seed."
        )
    month = pd.to_datetime(data[["year", "month"]].assign(day=1))
    if month.isna().any() or month.duplicated().any():
        raise ValueError("Each row needs a unique calendar month.")
    values = data[rates].apply(pd.to_numeric).set_index(month).sort_index()
    if ((values < 0) | (values > 1)).any().any() or values.isna().all().any():
        raise ValueError(
            "Each rate needs at least one defined value, and all defined rates must be shares in [0, 1]."
        )
    values = values.asfreq("MS") * 100
    output.mkdir(parents=True, exist_ok=True)
    with rc_context(rc=diagnostics.STYLE):
        fig, ax = plt.subplots(figsize=(6.7, 3.2), layout="constrained")
        labels = [
            "Employment / population",
            "Labour-force participation",
            "Unemployment / labour force",
        ]
        for name, label, color, linestyle, marker in zip(
            rates, labels, ["#006861", "#A85112", "#41667B"], ["-", "--", ":"], ["o", "s", "^"]
        ):
            ax.plot(
                values.index,
                values[name],
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=2.5 if len(values) < 30 else 1.3,
            )
        seed, index, evaluation = identities.iloc[0]
        ax.set_title(
            f"Monthly model rates  ·  policy seed {seed}, history {index}, evaluation seed {evaluation}",
            loc="left",
        )
        ax.set(xlabel="Calendar month", ylabel="Rate (%)", ylim=(0, 100))
        locator = mdates.AutoDateLocator(minticks=2, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        if len(values) == 1:
            date = values.index[0]
            ax.set_xlim(date - pd.Timedelta(days=15), date + pd.Timedelta(days=15))
        ax.grid(axis="y", color="#DCE3E6")
        ax.legend(ncol=3, loc="upper left", frameon=False)
        for suffix in ("svg", "pdf"):
            fig.savefig(output / f"labour_market_rates.{suffix}")
        plt.close(fig)


def render(inputs, output, selected):
    inputs = inputs.resolve()
    output.mkdir(parents=True, exist_ok=True)
    learning.configure_fonts()
    supplementary.configure_style()
    supplementary.ROOT = inputs
    supplementary.CORE = inputs / "core"
    supplementary.CONTROLS = inputs / "controls"
    supplementary.OUT = output
    firms.INPUT = inputs / "firm"
    firms.DEST = output
    diagnostics.DATA = inputs / "diagnostics"
    diagnostics.SPATIAL = spatial_pandemic.INPUT = inputs / "spatial"
    diagnostics.OUT = spatial_pandemic.OUTPUT = output

    def comparison(stem, draw):
        folder = inputs / "learning"
        if (folder / f"{stem}_source.csv").is_file():
            data = learning.load_figure(folder, stem)
        else:
            with TemporaryDirectory() as temp:
                folder = Path(temp)
                learning.prepare_data(supplementary.CONTROLS, folder)
                data = learning.load_figure(folder, stem)
        draw(data, output)

    def historical():
        panel = criteria_panel.draw_panel(diagnostics.DATA / "registered_criteria.csv")
        stem = "historical_learning"
        comparison(stem, learning.historical)
        criteria_panel.append_pdf(panel, output / stem)
        criteria_panel.append_svg(panel, output / stem)

    def firm_macro():
        annual, _, observed = firms.read_inputs()
        firms.macro(annual, observed)

    def firm_mechanisms():
        _, window, _ = firms.read_inputs()
        firms.mechanisms(window)

    def tariff_plot(extended=False):
        with ZipFile(inputs / "tariff" / "scores.zip") as archive:
            window, annual, baseline = tariff.read_scores(archive)
            if extended:
                tariff.contrast_figure(output / "tariff_scenario_contrasts.pdf", window)
            else:
                product = tariff.product_scores(archive)
                tariff.main_figure(
                    output / "tariff_transmission.pdf", window, annual, baseline, product
                )

    figures = {
        "Fig1": spatial_pandemic.framework,
        "Fig2": historical,
        "Fig3": lambda: comparison("recruitment_learning", learning.recruitment),
        "Fig4": spatial_pandemic.spatial,
        "Fig5": spatial_pandemic.covid,
        "Fig6": tariff_plot,
        "ED1": diagnostics.criteria,
        "ED2": supplementary.age_criteria,
        "ED3": supplementary.scale_comparison,
        "ED4": supplementary.recruitment_paths,
        "ED5": diagnostics.acceptance,
        "ED6": diagnostics.spatial,
        "ED7": supplementary.transition_paths,
        "ED8": lambda: tariff_plot(extended=True),
        "ED9": lambda: training.render(inputs / "training" / training.CSV_NAME, output),
        "S1": supplementary.pandemic_arms,
        "S2": supplementary.pandemic_effects,
        "S3": supplementary.error_contrasts,
        "S4": firm_macro,
        "S5": supplementary.flow_wages,
        "S6": firm_mechanisms,
        "S7": supplementary.policy_paths,
        "S8": supplementary.durations,
        "S9": diagnostics.history,
    }
    for name in selected or figures:
        if name not in figures:
            raise ValueError(f"Unknown figure {name}. Available figures: {', '.join(figures)}")
        print(f"Rendering {name}", flush=True)
        recipe = figures[name]
        with rc_context(rc=diagnostics.STYLE if recipe.__module__ == diagnostics.__name__ else {}):
            recipe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render study figures in SVG and PDF.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, help="Directory of study summary tables")
    source.add_argument("--history", type=Path, help="One monthly evaluation CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", nargs="+", metavar="ID")
    args = parser.parse_args()
    if args.history:
        if args.figure:
            parser.error("--figure selects figures for --inputs")
        plot_history(args.history, args.output)
    else:
        render(args.inputs, args.output, args.figure)
