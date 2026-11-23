import csv
import io
import math
import statistics
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

SEEDS = range(42, 47)
T4 = 2.7764451051977987
CONTROL = "control"
CENTRAL = "policy_record_persistent"
WINDOW = "impact_2027_2028"
INK = "#162C3B"
TEXT = "#354C58"
MUTED = "#546974"
GRID = "#DCE3E6"
ZERO = "#76848A"
SHADE = "#EEF2F4"
TEAL = "#007B81"
RUST = "#A14924"
VIOLET = "#665888"
STEEL = "#41667B"
SEED = "#70818A"
RATE_METRICS = (
    ("employment_rate", "Employment", TEAL),
    ("unemployment_rate", "Unemployment", RUST),
    ("participation_rate", "Participation", VIOLET),
)
PRODUCT_LABELS = (
    ("steel_aluminum", "Steel and aluminium"),
    ("other_manufacturing", "Other manufacturing"),
    ("unexposed", "Unexposed"),
)
SCENARIOS = (
    (CENTRAL, "Persistent tariff"),
    ("policy_e_low", "Low elasticity"),
    ("policy_e_high", "High elasticity"),
    ("policy_c_metal_chapter_0_8", "Metal content 0.80"),
    ("policy_ca_no_relief_ust_default", "No Canadian relief"),
    ("policy_claim_share_pre_shock", "Pre-shock claim shares"),
    ("policy_vehicles_unapproved", "Vehicle content unapproved"),
)
F_REG = "Arial"
F_BOLD = "Arial-Bold"
WIDTH, HEIGHT = (460, 372)


def line(c, x0, y0, x1, y1, colour, width=0.5, dash=None):
    c.add(
        Line(x0, y0, x1, y1, strokeColor=HexColor(colour), strokeWidth=width, strokeDashArray=dash)
    )


def text(c, x, y, value, size=9, colour=TEXT, align="left", bold=False):
    c.add(
        String(
            x,
            y,
            value,
            fontName=F_BOLD if bold else F_REG,
            fontSize=size,
            fillColor=HexColor(colour),
            textAnchor={"left": "start", "right": "end", "centre": "middle"}[align],
        )
    )


def polygon(c, coords, colour):
    c.add(
        Polygon(
            [value for xy in coords for value in xy], fillColor=HexColor(colour), strokeColor=None
        )
    )


def diamond(c, x, y, size, colour):
    polygon(c, [(x - size, y), (x, y + size), (x + size, y), (x, y - size)], colour)


def dot(c, x, y, size, colour):
    c.add(Circle(x, y, size, fillColor=HexColor(colour), strokeColor=None))


def save_artwork(drawing, path, title):
    pdf = Canvas(str(path), pagesize=(WIDTH, HEIGHT), initialFontName=F_REG, invariant=True)
    pdf.setTitle(title)
    renderPDF.draw(drawing, pdf, 0, 0)
    pdf.showPage()
    pdf.save()
    svg = renderSVG.drawToString(drawing)
    svg = svg.replace("font-family: Arial-Bold;", "font-family: Arial; font-weight: bold;")
    svg = svg.replace(f'width="{WIDTH}"', f'width="{WIDTH}pt"', 1)
    svg = svg.replace(f'height="{HEIGHT}"', f'height="{HEIGHT}pt"', 1)
    svg = svg.replace("<title>...</title>", "<title>" + title + "</title>", 1)
    svg = svg.replace(
        "<desc>...</desc>", "<desc>Five-policy paired tariff scenario differences.</desc>", 1
    )
    path.with_suffix(".svg").write_text(svg, encoding="utf-8")


def mapping(lo, hi, a, b):
    return lambda x: a + (x - lo) * (b - a) / (hi - lo)


def tick(value):
    return f"{value:g}".replace("-", "−")


def csv_rows(archive, name):
    with archive.open(name) as file, io.TextIOWrapper(file, encoding="utf-8-sig") as decoded:
        return list(csv.DictReader(decoded))


def read_scores(archive):
    rows = csv_rows(archive, "PANEL_EFFECTS.csv")
    window = {}
    for r in rows:
        if r["window"] != WINDOW or r["cell"] != "all":
            continue
        key = (r["scenario"], r["reference_scenario"], r["metric"])
        if key in window or (r["n_pairs_per_policy"], r["n_policy_seeds"], r["df"]) != (
            "50",
            "5",
            "4",
        ):
            raise ValueError(f"Unexpected score identity or coverage: {key}")
        values = [float(r[f"seed_{s}"]) for s in SEEDS]
        average = statistics.mean(values)
        margin = T4 * statistics.stdev(values) / math.sqrt(5)
        for field, value in zip(
            ("mean_effect", "ci95_low", "ci95_high"), (average, average - margin, average + margin)
        ):
            if not math.isclose(float(r[field]), value, rel_tol=1e-08, abs_tol=1e-09):
                raise ValueError(f"Panel and seed summaries disagree: {key} {field}")
        window[key] = (average, average - margin, average + margin, values)
    monthly = {}
    for r in csv_rows(archive, "MONTHLY_MACRO_SERIES.csv"):
        if r["scenario"] in (CONTROL, CENTRAL) and 2024 <= int(r["year"]) <= 2030:
            k = (r["scenario"], r["metric"], int(r["year"]), int(r["month"]))
            if k in monthly or r["units"] != "fraction":
                raise ValueError(f"Unexpected monthly series: {k}")
            monthly[k] = [float(r[f"seed_{s}"]) for s in SEEDS]
    annual = {}
    for metric, _, _ in RATE_METRICS[:2]:
        years = {}
        for year in range(2024, 2031):
            paired_months = []
            for month in range(1, 13):
                a = monthly[CENTRAL, metric, year, month]
                b = monthly[CONTROL, metric, year, month]
                paired_months.append([100 * (x - y) for x, y in zip(a, b)])
            seed_values = [statistics.mean(months) for months in zip(*paired_months)]
            average = statistics.mean(seed_values)
            margin = T4 * statistics.stdev(seed_values) / math.sqrt(5)
            years[year] = (average, average - margin, average + margin, seed_values)
        for a, b in zip(window[CENTRAL, CONTROL, metric][3], zip(years[2027][3], years[2028][3])):
            if not math.isclose(a, statistics.mean(b), abs_tol=1e-09):
                raise ValueError("Annual paired time course disagrees with primary-window score")
        annual[metric] = years
    baseline = {
        metric: 100
        * statistics.mean(
            (
                statistics.mean(monthly[CONTROL, metric, year, month])
                for year in (2027, 2028)
                for month in range(1, 13)
            )
        )
        for metric in ("unemployment_rate", "job_finding_rate")
    }
    return (window, annual, baseline)


def product_scores(archive):

    def selected_rows(name):
        with archive.open(name) as file, io.TextIOWrapper(file, encoding="utf-8-sig") as decoded:
            for row in csv.DictReader(decoded):
                if (
                    row["scenario"],
                    row["reference_scenario"],
                    row["window"],
                    row["domain"],
                    row["metric"],
                    row["units"],
                ) == (CENTRAL, CONTROL, WINDOW, "product_group", "employment", "workers"):
                    yield row

    support = {
        row["cell"]: [int(row[f"n_pairs_s{s}"]) for s in SEEDS]
        for row in selected_rows("ROLLUP_SUPPORT.csv")
    }
    by_seed = {
        (r["cell"], int(r["policy_seed"])): r for r in selected_rows("ROLLUP_POLICY_EFFECTS.csv")
    }
    product = {}
    for r in selected_rows("ROLLUP_PANEL_EFFECTS.csv"):
        cell = r["cell"]
        values = [float(by_seed[cell, s]["mean_effect"]) for s in SEEDS]
        counts = [int(by_seed[cell, s]["n_supported_pairs"]) for s in SEEDS]
        if counts != support[cell] or any((n < 1 or n > 50 for n in counts)):
            raise ValueError(f"Mismatched product-group coverage: {cell}")
        mean = statistics.mean(values)
        half = T4 * statistics.stdev(values) / math.sqrt(len(values))
        observed = [float(r[k]) for k in ("mean_effect", "ci95_low", "ci95_high")]
        if not all(
            (
                math.isclose(a, b, abs_tol=1e-09)
                for a, b in zip(observed, [mean, mean - half, mean + half])
            )
        ):
            raise ValueError(f"Mismatched product-group summary: {cell}")
        product[cell] = (mean, mean - half, mean + half, values, sum(counts))
    automotive = [
        r
        for r in csv_rows(archive, "PRODUCT_GROUP_REPRESENTATION.csv")
        if r["product_group"] == "automotive"
    ]
    if len({(r["scenario"], r["policy_seed"]) for r in automotive}) != 40 or any(
        (
            r["n_histories"] != "50"
            or r["automotive_headline_minimum"] != "3"
            or r["headline_eligible"] != "False"
            or (int(r["n_firms_min"]) >= 3)
            for r in automotive
        )
    ):
        raise ValueError("Automotive eligibility has changed; reconsider the exclusion")
    return product


def heading(c, x, y, letter, title):
    text(c, x, y, letter, size=9, colour=INK, bold=True)
    text(c, x + 13, y + 0.4, title, size=8.5, colour=INK, bold=True)


def summary_axis(c, bounds, ticks, position, bottom, top):
    for n in ticks:
        x = position(n)
        line(c, x, bottom, x, top, ZERO if n == 0 else GRID, 0.8 if n == 0 else 0.4)
        text(c, x, bottom - 12, tick(n), 7, MUTED, "centre")
    line(c, position(bounds[0]), bottom, position(bounds[1]), bottom, GRID)


def window_panel(c, scores, items, x_label, x0, x1, y_rows, x_limits, ticks, xlabel):
    where = mapping(*x_limits, x0, x1)
    y_bottom, y_top = (37, 144)
    summary_axis(c, x_limits, ticks, where, y_bottom, y_top)
    text(c, (x0 + x1) / 2, 8, xlabel, 7, MUTED, "centre")
    for y, (metric, label, colour) in zip(y_rows, items):
        mean, low, high, seed_values = scores[CENTRAL, CONTROL, metric]
        text(c, x_label, y - 1, label, 7.1, TEXT, "right")
        line(c, where(low), y + 3, where(high), y + 3, colour, 1.55)
        for end in (low, high):
            line(c, where(end), y + 0.8, where(end), y + 5.3, colour, 0.8)
        diamond(c, where(mean), y + 3, 3.5, colour)
        occupied = []
        for estimate in sorted(seed_values):
            x = where(estimate)
            for level in (0, 1, -1, 2, -2):
                yy = y - 4.7 + level * 3.1
                if all(((x - xx) ** 2 + (yy - yv) ** 2 > 10 for xx, yv in occupied)):
                    break
            occupied.append((x, yy))
            dot(c, x, yy, 1.55, SEED)


def product_panel(c, product):
    position = mapping(-10, 12, 340, 453)
    summary_axis(c, (-10, 12), (-10, -5, 0, 5, 10), position, 59, 142)
    for (key, label), y in zip(PRODUCT_LABELS, (129, 101, 73)):
        mean, low, high, seeds, pairs = product[key]
        text(c, 334, y + 2, label, 7, TEXT, "right")
        text(c, 334, y - 7, f"{pairs}/250 pairs", 6.3, MUTED, "right")
        line(c, position(low), y + 3, position(high), y + 3, STEEL, 1.5)
        for v in (low, high):
            line(c, position(v), y + 0.6, position(v), y + 5.4, STEEL, 0.8)
        diamond(c, position(mean), y + 3, 3.2, STEEL)
        for i, v in enumerate(sorted(seeds)):
            dot(c, position(v), y - 5.1 + i % 2 * 2.9, 1.4, SEED)
    text(c, 394, 36, "Difference in workers", 7, MUTED, "centre")
    line(c, 257, 29, 453, 29, GRID)
    text(c, 257, 17, "Automotive", 7.1, TEXT, bold=True)
    text(c, 257, 7, f"{product['automotive'][4]}/250 pairs", 6.3, MUTED)
    text(c, 326, 13, "Firm-coverage criterion unmet", 7, TEXT)


def main_figure(path, scores, annual, baseline, product):
    c = Drawing(WIDTH, HEIGHT)
    heading(c, 3, 357, "a", "Annual labour-market response")
    text(c, 16, 341, "Persistent tariff minus control (percentage points)", 7, MUTED)
    x = mapping(2024, 2030, 49, 360)
    y = mapping(-0.45, 0.6, 205, 321)
    c.add(
        Rect(
            x(2026.5),
            y(-0.45),
            x(2028.5) - x(2026.5),
            y(0.6) - y(-0.45),
            fillColor=HexColor(SHADE),
            strokeColor=None,
        )
    )
    for v in (-0.4, -0.2, 0, 0.2, 0.4, 0.6):
        line(c, 45, y(v), 368, y(v), ZERO if v == 0 else GRID, 0.7 if v == 0 else 0.4)
        text(c, 38, y(v) - 2.3, tick(v), 6.9, MUTED, "right")
    text(c, x(2027.5), y(0.56), "Primary window", 6.9, MUTED, "centre")
    for year in range(2024, 2031):
        text(c, x(year), 192, str(year), 7, MUTED, "centre")
    for (metric, label, colour), shade in zip(RATE_METRICS[:2], ("#D1E8E6", "#F2E1D9")):
        values = annual[metric]
        upper = [(x(year), y(v[2])) for year, v in values.items()]
        lower = [(x(year), y(v[1])) for year, v in reversed(values.items())]
        polygon(c, upper + lower, shade)
        points = [(x(year), y(v[0])) for year, v in values.items()]
        for (a, b), (d, e) in zip(points, points[1:]):
            line(c, a, b, d, e, colour, 1.4, (4, 2) if metric == "unemployment_rate" else None)
        for a, b in points:
            if metric == "unemployment_rate":
                polygon(c, [(a, b + 2.5), (a - 2.3, b - 1.8), (a + 2.3, b - 1.8)], colour)
            else:
                dot(c, a, b, 2.0, colour)
        text(c, 371, points[-1][1] - 2.6, label, 7.4, colour, bold=True)
    line(c, 3, 178, 457, 178, GRID)
    heading(c, 3, 160, "b", "Labour-market rates, 2027–28")
    heading(c, 244, 160, "c", "Firm employment by product group, 2027–28")
    text(
        c,
        16,
        146,
        f"Control: U/LF {baseline['unemployment_rate']:.1f}%; monthly job finding {baseline['job_finding_rate']:.1f}%",
        7,
        MUTED,
    )
    window_panel(
        c,
        scores,
        RATE_METRICS,
        83,
        91,
        229,
        [127, 93, 59],
        (-0.45, 0.52),
        [-0.4, -0.2, 0, 0.2, 0.4],
        "Difference (percentage points)",
    )
    product_panel(c, product)
    save_artwork(
        c, path, "Persistent tariffs, firm employment by product group and labour-market rates"
    )


def contrast_figure(path, scores):
    c = Drawing(WIDTH, HEIGHT)
    x0s = (134, 245, 356)
    span = 96
    ticks = (-0.4, -0.2, 0, 0.2, 0.4)
    for i, (metric, title, colour) in enumerate(RATE_METRICS):
        start = x0s[i]
        text(c, start + span / 2, 355, title, 8.3, colour, "centre", bold=True)
    groups = [
        (CONTROL, SCENARIOS, "Compared with control", 326),
        (CENTRAL, SCENARIOS[1:], "Compared with persistent tariff", 157),
    ]
    for reference, names, label, header_y in groups:
        c.add(Rect(2, header_y - 3.5, 455, 15, fillColor=HexColor(SHADE), strokeColor=None))
        text(c, 6, header_y + 0.4, label, 7.6, INK, bold=True)
        ys = [header_y - 17 - j * 21 for j in range(len(names))]
        for (scenario, name), y in zip(names, ys):
            text(c, 6, y - 2.3, name, 7.1, TEXT)
        for i, (metric, _, colour) in enumerate(RATE_METRICS):
            start = x0s[i]
            where = mapping(-0.44, 0.53, start, start + span)
            for v in ticks:
                line(
                    c,
                    where(v),
                    ys[-1] - 9,
                    where(v),
                    ys[0] + 9,
                    ZERO if v == 0 else GRID,
                    0.7 if v == 0 else 0.37,
                )
                if reference == CENTRAL:
                    text(c, where(v), ys[-1] - 22, tick(v), 6.9, MUTED, "centre")
            for (scenario, _), yy in zip(names, ys):
                mean, low, high, _ = scores[scenario, reference, metric]
                if low < -0.44 or high > 0.53:
                    raise ValueError("An interval exceeds the display domain")
                line(c, where(low), yy, where(high), yy, colour, 1.25)
                for end in (low, high):
                    line(c, where(end), yy - 1.8, where(end), yy + 1.8, colour, 0.65)
                diamond(c, where(mean), yy, 2.65, colour)
    text(c, 289, 2.5, "Paired difference (percentage points)", 7, MUTED, "centre")
    save_artwork(c, path, "All 13 paired tariff contrasts across policy seeds")
