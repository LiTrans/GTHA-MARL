import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.renderbase import STATE_DEFAULTS
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, PolyLine, String
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

SEEDS = tuple(range(42, 47))
YEARS = tuple(range(2006, 2017))
TRAIN, RANDOM, HEUR = ("trained", "untrained_random", "untrained_heuristic")
FAMILIES = (TRAIN, RANDOM, HEUR)
METRICS = ("employment_rate", "participation_rate", "unemployment_rate")
RECRUIT = ("U_matching_pool", "offer_reach_pct", "precommit_conversion_pct", "U_committed_hires")
FIELDS = ("panel", "configuration", "policy_seed", "year", "metric", "unit", "estimate")
T4 = 2.7764451051977987
WIDTH = 180 * 72 / 25.4
INK, SOFT, GRID, ZERO = ("#1C2932", "#425863", "#DFE5E7", "#778A93")
COL = {TRAIN: "#006861", RANDOM: "#4D365F", HEUR: "#AB691E", "LFS": "#25333C"}
PALE = {TRAIN: "#D6EBE9", RANDOM: "#ECE4F0", HEUR: "#F4E8D8"}
DASH = {TRAIN: None, RANDOM: (4, 2), HEUR: (1.1, 1.7), "LFS": (4, 1.8, 1, 1.8)}


def interval(values):
    if len(values) != len(SEEDS) or not all(map(math.isfinite, values)):
        raise ValueError("Each estimate requires five finite policy values")
    mean = statistics.mean(values)
    half = T4 * statistics.stdev(values) / math.sqrt(len(values))
    return (mean, mean - half, mean + half)


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare_data(directory, output):
    rows = {"historical_learning": [], "recruitment_learning": []}

    def add(figure, panel, case, seed, year, metric, unit, value):
        rows[figure].append((panel, case, seed, year, metric, unit, value))

    for row in read_table(directory / "MACRO_ANNUAL_POLICY_CURVES.csv"):
        if row["case"] in FAMILIES:
            for metric in METRICS:
                add(
                    "historical_learning",
                    "annual",
                    row["case"],
                    row["policy_seed"],
                    row["year"],
                    metric,
                    "percent",
                    100 * float(row[metric]),
                )
    for row in read_table(directory / "HISTORICAL_REFERENCE.csv"):
        for metric in METRICS:
            add(
                "historical_learning",
                "annual",
                "LFS",
                "",
                row["year"],
                metric,
                "percent",
                100 * float(row[metric]),
            )
    for row in read_table(directory / "MACRO_PAIRED_POLICY_CONTRASTS.csv"):
        if row["comparison"] in (RANDOM, HEUR) and row["estimand"] in (
            "window_mean_pct",
            "annual_path_MAE_pp",
        ):
            add(
                "historical_learning",
                row["estimand"],
                row["comparison"],
                row["policy_seed"],
                "",
                row["metric"],
                "percentage_points",
                float(row["value"]),
            )
    for row in read_table(directory / "MECHANISM_WINDOW_POLICY_ESTIMATES.csv"):
        metric, case = (row["metric"], row["case"])
        if case in FAMILIES and metric in RECRUIT:
            unit = "percent" if metric.endswith("_pct") else "per_1000_initial_per_month"
            val = float(row["value"])
            if unit == "percent":
                check([val], [100 * float(row["numerator_sum"]) / float(row["denominator_sum"])])
            add("recruitment_learning", "level", case, row["policy_seed"], "", metric, unit, val)
    for row in read_table(directory / "MECHANISM_PAIRED_POLICY_CONTRASTS.csv"):
        metric = row["metric"]
        if row["comparison"] in (RANDOM, HEUR) and metric in RECRUIT:
            unit = "percentage_points" if metric.endswith("_pct") else "per_1000_initial_per_month"
            value = float(row["value"])
            check([value], [float(row["value_trained"]) - float(row["value_counterfactual"])])
            add(
                "recruitment_learning",
                "difference",
                row["comparison"],
                row["policy_seed"],
                "",
                metric,
                unit,
                value,
            )
    controls = (RANDOM, HEUR)
    summaries = [
        ("historical_learning", "MACRO_COMBINED_ANNUAL_CURVES", "case", FAMILIES, "annual", 100.0),
        ("historical_learning", "MACRO_PAIRED_CONTRAST_SUMMARY", "comparison", controls, None, 1.0),
        ("recruitment_learning", "MECHANISM_WINDOW_SUMMARY", "case", FAMILIES, "level", 1.0),
        (
            "recruitment_learning",
            "MECHANISM_PAIRED_CONTRAST_SUMMARY",
            "comparison",
            controls,
            "difference",
            1.0,
        ),
    ]
    for figure, name, case_key, cases, fixed_panel, factor in summaries:
        grouped = defaultdict(list)
        for panel, case, seed, year, metric, unit, value in rows[figure]:
            grouped[panel, case, year, metric].append(value)
        for row in read_table(directory / (name + ".csv")):
            key = (
                fixed_panel or row.get("estimand"),
                row[case_key],
                row.get("year", ""),
                row["metric"],
            )
            if key[1] in cases and key in grouped:
                check(
                    interval(grouped[key]),
                    [factor * float(row[k]) for k in ("mean", "lower95", "upper95")],
                )
    for figure, records in rows.items():
        with (output / (figure + "_source.csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(records)


def check(actual, expected):
    if len(actual) != len(expected) or any(
        (not math.isclose(a, b, abs_tol=1e-09) for a, b in zip(actual, expected))
    ):
        raise ValueError(f"Summary mismatch: {actual} and {expected}")


def load_figure(output, figure):
    data = {}
    for row in read_table(output / (figure + "_source.csv")):
        key = (
            row["panel"],
            row["configuration"],
            int(row["policy_seed"] or 0),
            int(row["year"] or 0),
            row["metric"],
        )
        if key in data:
            raise ValueError(f"Duplicate figure row {key}")
        data[key] = float(row["estimate"])
    return data


def configure_fonts():
    fonts = Path("C:/Windows/Fonts")
    for name, system, bundled in (
        ("Arial", "arial.ttf", "Vera.ttf"),
        ("Arial-Bold", "arialbd.ttf", "VeraBd.ttf"),
    ):
        source = fonts / system
        if not source.is_file():
            import reportlab

            source = Path(reportlab.__file__).parent / "fonts" / bundled
        pdfmetrics.registerFont(TTFont(name, str(source)))
    STATE_DEFAULTS["fontName"] = "Arial"


class Artwork:

    def __init__(self, height):
        self.height = height
        self.drawing = Drawing(WIDTH, height)

    def line(self, x1, y1, x2, y2, color=GRID, width=0.55, dash=None):
        self.drawing.add(
            Line(
                x1,
                self.height - y1,
                x2,
                self.height - y2,
                strokeColor=HexColor(color),
                strokeWidth=width,
                strokeDashArray=dash,
            )
        )

    def text(self, x, y, value, size=8, color=INK, bold=False, anchor="start"):
        self.drawing.add(
            String(
                x,
                self.height - y,
                value,
                fontName="Arial-Bold" if bold else "Arial",
                fontSize=size,
                fillColor=HexColor(color),
                textAnchor=anchor,
            )
        )

    def points(self, points, color, width=1.35, dash=None):
        flat = [z for x, y in points for z in (x, self.height - y)]
        self.drawing.add(
            PolyLine(
                flat,
                strokeColor=HexColor(color),
                strokeWidth=width,
                strokeDashArray=dash,
                fillColor=None,
            )
        )

    def polygon(self, points, color):
        flat = [z for x, y in points for z in (x, self.height - y)]
        self.drawing.add(Polygon(flat, fillColor=HexColor(color), strokeColor=None))

    def dot(self, x, y, radius, color):
        self.drawing.add(
            Circle(x, self.height - y, radius, fillColor=HexColor(color), strokeColor=None)
        )

    def mean(self, x, y, radius, color):
        self.polygon([(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)], color)

    def save(self, stem):
        renderPDF.drawToFile(self.drawing, str(stem.with_suffix(".pdf")), initialFontName="Arial")
        svg = renderSVG.drawToString(self.drawing)
        svg = svg.replace("font-family: Arial-Bold;", "font-family: Arial; font-weight: bold;")
        svg = svg.replace(f'width="{WIDTH}"', 'width="180mm"')
        svg = svg.replace(f'height="{self.height}"', f'height="{self.height}pt"', 1)
        svg = svg.replace(
            f'viewBox="0 0 {int(WIDTH)} {self.height}"',
            f'viewBox="0 0 {WIDTH:.6f} {self.height}"',
            1,
        )
        stem.with_suffix(".svg").write_text(svg, encoding="utf-8")


def mapping(lower, upper, start, end):
    return lambda v: start + (v - lower) * (end - start) / (upper - lower)


def x_axis(fig, ticks, scale, top, bottom, zero=False):
    for t in ticks:
        x = scale(t)
        fig.line(
            x, top, x, bottom, ZERO if zero and t == 0 else GRID, 0.85 if zero and t == 0 else 0.5
        )
        fig.text(
            x, bottom + 11, format(t, "g").replace("-", "−"), size=7.3, color=SOFT, anchor="middle"
        )
    fig.line(scale(ticks[0]), bottom, scale(ticks[-1]), bottom, SOFT)


def seed_values(data, panel, case, metric, year=0):
    return [data[panel, case, seed, year, metric] for seed in SEEDS]


def interval_mark(fig, scale, y, values, color, dots=True):
    mean, lower, upper = interval(values)
    if dots:
        for i, value in enumerate(values):
            fig.dot(scale(value), y + 4.2 + i * 1.9, 0.96, color)
    fig.line(scale(lower), y, scale(upper), y, color, 1.15)
    fig.line(scale(lower), y - 2, scale(lower), y + 2, color, 0.8)
    fig.line(scale(upper), y - 2, scale(upper), y + 2, color, 0.8)
    fig.mean(scale(mean), y, 2.55, color)


def historical(data, output):
    if len(data) != 588:
        raise ValueError(f"Unexpected historical data row count: {len(data)}")
    fig = Artwork(365)
    legend = (
        (TRAIN, "Jointly trained", 8),
        (RANDOM, "Untrained random", 115),
        (HEUR, "Untrained workers + heuristic firms", 236),
        ("LFS", "Observed (LFS)", 413),
    )
    for case, name, x in legend:
        fig.line(x, 10, x + 14, 10, COL[case], 1.45, DASH[case])
        fig.text(x + 18, 12.5, name, 7.7)
    names = ["Employment", "Participation", "Unemployment"]
    prop = ["Population (%)", "Population (%)", "Labour force (%)"]
    axis_starts = [52, 56, 0]
    axis_span = 16
    for i, metric in enumerate(METRICS):
        x = 4 + 170 * i
        fig.text(x, 33, chr(97 + i), 9.5, bold=True)
        fig.text(x + 13, 33, names[i], 10.3, bold=True)
        fig.text(x + 13, 46, prop[i], 7.7, SOFT)
        left, right, top, bottom = (x + 20, x + 156, 59, 141)
        xx = mapping(2006, 2016, left, right)
        lower = axis_starts[i]
        upper = lower + axis_span
        yy = mapping(lower, upper, bottom, top)
        for val in range(lower, upper + 1, 4):
            fig.line(left, yy(val), right, yy(val))
            fig.text(left - 4, yy(val) + 2.2, str(val), 7.2, SOFT, anchor="end")
        for case in FAMILIES:
            dates = [interval(seed_values(data, "annual", case, metric, y)) for y in YEARS]
            top_line = [(xx(y), yy(v[2])) for y, v in zip(YEARS, dates)]
            bot_line = [(xx(y), yy(v[1])) for y, v in zip(YEARS, dates)]
            fig.polygon(top_line + bot_line[::-1], PALE[case])
        for case in (HEUR, RANDOM, TRAIN, "LFS"):
            vals = [
                (
                    data["annual", "LFS", 0, y, metric]
                    if case == "LFS"
                    else interval(seed_values(data, "annual", case, metric, y))[0]
                )
                for y in YEARS
            ]
            pts = [(xx(y), yy(v)) for y, v in zip(YEARS, vals)]
            fig.points(pts, COL[case], 1.25, DASH[case])
            if case == "LFS":
                for px, py in pts:
                    fig.dot(px, py, 1.25, COL[case])
        fig.line(left, bottom, right, bottom, SOFT)
        for year in (2006, 2011, 2016):
            fig.line(xx(year), bottom, xx(year), bottom + 2.6, SOFT)
            fig.text(xx(year), bottom + 12, str(year), 7.4, SOFT, anchor="middle")
    observed = (
        data["annual", "LFS", 0, 2009, "unemployment_rate"]
        - data["annual", "LFS", 0, 2008, "unemployment_rate"]
    )
    trained = statistics.mean(
        seed_values(data, "annual", TRAIN, "unemployment_rate", 2009)
    ) - statistics.mean(seed_values(data, "annual", TRAIN, "unemployment_rate", 2008))
    fig.text(
        7,
        169,
        (
            "Unemployment change in 2008–09: survey %+.2f pp, jointly trained %+.2f pp."
            % (observed, trained)
        ).replace("-", "−"),
        7.7,
        SOFT,
    )
    fig.line(5, 178, WIDTH - 5, 178)
    fig.text(6, 195, "Trained minus control, 2006–2016", 9.5, bold=True)
    for xx, case, label in ((239, RANDOM, "Versus random"), (342, HEUR, "Versus heuristic")):
        fig.line(xx, 192, xx + 12, 192, COL[case], 1.3)
        fig.mean(xx + 6, 192, 2.3, COL[case])
        fig.text(xx + 16, 195, label, 7.6)
    for i, (panel, title) in enumerate(
        (
            ("window_mean_pct", "Change in mean rate"),
            ("annual_path_MAE_pp", "Change in mean absolute error"),
        )
    ):
        origin = 96 + i * 208
        fig.text(origin - 13, 215, "de"[i], 9.3, bold=True)
        fig.text(origin, 215, title, 8.7, bold=True)
        fig.text(
            origin,
            227,
            "Percentage points" if i == 0 else "Percentage points; negative = closer to survey",
            7.3,
            SOFT,
        )
        pos = mapping(-8, 12, origin, origin + 187)
        x_axis(fig, [-8, -4, 0, 4, 8, 12], pos, 233, 338, True)
        for j, metric in enumerate(METRICS):
            y = 240 + 34 * j
            if i == 0:
                fig.text(6, y + 10, names[j], 8, INK, bold=True)
            for delta, case in ((0, RANDOM), (13.8, HEUR)):
                interval_mark(
                    fig, pos, y + delta, seed_values(data, panel, case, metric), COL[case]
                )
    fig.text(
        6,
        362,
        "Dots: matched seeds 42–46, top to bottom. Diamonds and bars: five-seed means and 95% intervals.",
        7.4,
        SOFT,
    )
    fig.save(output / "historical_learning")


def recruitment(data, output):
    if len(data) != 100:
        raise ValueError(f"Unexpected recruitment data row count: {len(data)}")
    fig = Artwork(284)
    specs = [
        (
            "Matching pool",
            ("Unemployed participants", "per 1,000 initial / month"),
            (0, 60, 120),
            (0, 40, 80),
        ),
        (
            "Offer reach",
            ("Unique recipients /", "unemployed pool (%)"),
            (0, 50, 100),
            (-24, -12, 0),
        ),
        (
            "Match conversion",
            ("Pre-commit matches /", "unique recipients (%)"),
            (0, 50, 100),
            (0, 8, 16),
        ),
        (
            "Committed hires",
            ("Hires from unemployment", "per 1,000 initial / month"),
            (0, 5, 10),
            (-0.2, 0, 0.2, 0.4),
        ),
    ]
    names = ["Jointly trained", "Untrained random", "Untrained workers"]
    for j, case in enumerate(FAMILIES):
        y = 52 + 39 * j
        fig.text(3, y + 8, names[j], 7.6, COL[case], bold=True)
        if case == HEUR:
            fig.text(3, y + 18, "+ heuristic firms", 7.6, COL[case], bold=True)
    fig.text(
        4,
        172,
        "Dots: seeds 42–46, top to bottom. Diamonds: equally weighted five-seed means.",
        7.4,
        SOFT,
    )
    fig.line(3, 178, WIDTH - 3, 178)
    fig.text(3, 193, "Paired difference from each control", 9.2, bold=True)
    fig.text(211, 193, "Jointly trained minus control; mean and 95% interval", 7.6, SOFT)
    fig.text(3, 220, "Versus random", 7.6, COL[RANDOM], bold=True)
    fig.text(3, 248, "Versus heuristic", 7.6, COL[HEUR], bold=True)
    for i, (metric, spec) in enumerate(zip(RECRUIT, specs)):
        title, desc, ticks, diff_ticks = spec
        x = 83 + 107 * i
        fig.text(x, 12, chr(97 + i), 9.7, bold=True)
        fig.text(x + 9, 12, title, 8.9, bold=True)
        for j, label in enumerate(desc):
            fig.text(x, 25 + 9 * j, label, 7.4, SOFT)
        pos = mapping(ticks[0], ticks[-1], x + 2, x + 91)
        x_axis(fig, ticks, pos, 40, 151)
        for j, case in enumerate(FAMILIES):
            y = 52 + 39 * j
            values = seed_values(data, "level", case, metric)
            mean = statistics.mean(values)
            suffix = "%" if metric.endswith("_pct") else ""
            xp = pos(mean)
            for n, val in enumerate(values):
                fig.dot(pos(val), y + 4 + n * 2.9, 1.2, COL[case])
            fig.mean(xp, y, 2.65, COL[case])
            right = xp > x + 64
            fig.text(
                xp - 4 if right else xp + 4,
                y - 3,
                f"{mean:.2f}{suffix}",
                7.8,
                COL[case],
                bold=True,
                anchor="end" if right else "start",
            )
        diff = mapping(diff_ticks[0], diff_ticks[-1], x + 2, x + 91)
        x_axis(fig, diff_ticks, diff, 201, 254, True)
        for j, case in enumerate((RANDOM, HEUR)):
            values = seed_values(data, "difference", case, metric)
            check(
                values,
                [
                    a - b
                    for a, b in zip(
                        seed_values(data, "level", TRAIN, metric),
                        seed_values(data, "level", case, metric),
                    )
                ],
            )
            interval_mark(fig, diff, 217 + 28 * j, values, COL[case], False)
        unit = "Percentage points" if metric.endswith("_pct") else "Per 1,000 initial / month"
        fig.text(x + 47, 279, unit, 7.3, SOFT, anchor="middle")
    fig.save(output / "recruitment_learning")
