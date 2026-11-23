import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from lxml import etree as ET

INPUT = None
OUTPUT = None
SVG = "http://www.w3.org/2000/svg"
INK = "#172B38"
MUTED = "#405865"
GRID = "#D6DFE2"
TEAL = "#006861"
OCHRE = "#A85112"
BLUE = "#5082AE"
WIDTH_PT = 504
WIDTH_MM = WIDTH_PT * 25.4 / 72


def node(parent, element, **attrs):
    return ET.SubElement(
        parent,
        f"{{{SVG}}}{element}",
        {key.replace("_", "-"): str(value) for key, value in attrs.items()},
    )


def canvas(height, title, description):
    root = ET.Element(
        f"{{{SVG}}}svg",
        nsmap={None: SVG},
        width=f"{WIDTH_MM:.5f}mm",
        height=f"{height * 25.4 / 72:.5f}mm",
        viewBox=f"0 0 {WIDTH_PT} {height}",
    )
    node(root, "title").text = title
    node(root, "desc").text = description
    return root


def label(root, x, y, content, *, size=6.8, color=INK, bold=False, anchor="start", **attrs):
    el = node(
        root,
        "text",
        x=x,
        y=y,
        font_family="Arial, Helvetica, sans-serif",
        font_size=size,
        fill=color,
        text_anchor=anchor,
        font_weight="bold" if bold else "normal",
        **attrs,
    )
    el.text = str(content)
    return el


def line(root, x1, y1, x2, y2, color=GRID, width=0.45, dash=None):
    attrs = {"stroke_dasharray": dash} if dash else {}
    return node(root, "line", x1=x1, x2=x2, y1=y1, y2=y2, stroke=color, stroke_width=width, **attrs)


def marker(root, x, y, color, radius=2.15, diamond=False):
    if diamond:
        r = radius * 1.24
        node(root, "polygon", points=f"{x},{y - r} {x + r},{y} {x},{y + r} {x - r},{y}", fill=color)
    else:
        node(root, "circle", cx=x, cy=y, r=radius, fill=color)


def binary():
    installed = (
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Inkscape/bin/inkscape.com"
    )
    executable = shutil.which("inkscape") or (str(installed) if installed.exists() else "")
    if not executable:
        raise RuntimeError("Inkscape could not be found")
    return executable


def save(root, stem):
    svg = OUTPUT / (stem + ".svg")
    ET.ElementTree(root).write(str(svg), encoding="UTF-8", xml_declaration=True)
    subprocess.run([binary(), str(svg), f"--export-filename={svg.with_suffix('.pdf')}"], check=True)


def load_json(name):
    return json.loads((INPUT / name).read_text(encoding="utf8"))


def spatial():
    acc = load_json("registered_acceptance_inference.json")
    commute = load_json("spatial_2016_summary.json")
    access = load_json("vacancy_access_2016_summary.json")
    endpoint = "commute_at_finalized_match"
    seeds = range(42, 47)
    by_seed = [acc["per_policy_paired_df49"][str(s)][endpoint] for s in seeds]
    overall = acc["across_five_policies_df4"][endpoint]
    assert all((r["n"] == 50 for r in by_seed)) and overall["n"] == 5
    assert abs(mean((r["mean"] for r in by_seed)) - overall["mean"]) < 1e-10
    base, altered = [acc["means_by_arm"][arm] for arm in ("baseline", "recentred")]
    assert abs(altered[endpoint] - base[endpoint] - overall["mean"]) < 1e-08
    root = canvas(
        183,
        "Commute at hire and geographic access to vacancies",
        "Panel a shows paired effects of re-centring offer acceptance. Panels b–d show commute, vacancy access and supply by market size in the historical replay.",
    )
    label(root, 4, 11, "a", bold=True, size=8)
    label(root, 16, 11, "Offer-acceptance intervention, 2006–2016", bold=True, size=7)
    label(root, 16, 23, "Change in mean commute at hire (minutes)", color=MUTED)
    x1, x2 = (53, 207)
    to_x = lambda u: x1 + u / 6 * (x2 - x1)
    for tick in (0, 2, 4, 6):
        x = to_x(tick)
        line(
            root,
            x,
            32,
            x,
            114,
            color=MUTED if tick == 0 else GRID,
            width=0.55 if tick == 0 else 0.4,
            dash="2 2" if tick == 0 else None,
        )
        label(root, x, 124, tick, anchor="middle", size=6.25, color=MUTED)
    for i, row in enumerate(by_seed):
        y = 40 + 12 * i
        label(root, x1 - 8, y + 2, f"Seed {42 + i}", anchor="end", size=6.45)
        marker(root, to_x(row["mean"]), y, TEAL, radius=2.15)
    y = 108
    label(root, x1 - 8, y + 2, "Mean", bold=True, anchor="end", size=6.7)
    line(root, to_x(overall["ci95_low"]), y, to_x(overall["ci95_high"]), y, INK, 0.95)
    marker(root, to_x(overall["mean"]), y, INK, diamond=True, radius=2.5)
    label(
        root,
        16,
        139,
        f"Paired increase  +{overall['mean']:.2f} min (95% CI {overall['ci95_low']:.2f}–{overall['ci95_high']:.2f})",
        bold=True,
        size=6.7,
    )
    label(
        root,
        16,
        151,
        f"Mean commute at hire  {base[endpoint]:.2f} vs {altered[endpoint]:.2f} min",
        size=6.45,
    )
    label(
        root,
        16,
        162,
        f"Offers accepted  {100 * base['TH_realized_acceptance_rate']:.2f}% vs {100 * altered['TH_realized_acceptance_rate']:.2f}%",
        size=6.45,
    )
    label(root, 16, 173, "Full rule first, then re-centred rule", color=MUTED, size=6.35)
    line(root, 232, 5, 232, 174, width=0.45)
    scales = ("1000", "2500")

    def stock(key):
        return [commute["scales"][s]["means_and_policy_ranges"][key]["mean"] for s in scales]

    values = [
        stock("stock_time_all_valid_mean"),
        [100 * access["scales"][s]["fraction_with_access"][0] for s in scales],
        stock("mean_closing_vacancies"),
    ]
    label(root, 246, 11, "Historical market size, 2016", bold=True, size=7)
    marker(root, 249, 23, TEAL)
    label(root, 256, 25, "1,000 residents / 80 firms", size=6.65)
    marker(root, 374, 23, BLUE, diamond=True)
    label(root, 381, 25, "2,500 residents / 200 firms", size=6.65)
    panels = [
        ("b", "Assigned commute among employed residents (minutes)", (50, 65), (50, 55, 60, 65), 1),
        (
            "c",
            "Unemployed person-months within 60 min of a vacancy (%)",
            (0, 100),
            (0, 25, 50, 75, 100),
            1,
        ),
        ("d", "Open vacancies at month end (mean number)", (0, 4), (0, 1, 2, 3, 4), 2),
    ]
    for j, (letter, title, limits, ticks, precision) in enumerate(panels):
        y = 40 + j * 48
        label(root, 245, y, letter, size=8, bold=True)
        label(root, 257, y, title, size=6.8, bold=True)
        left, right = (260, 492)
        scale_x = lambda x: left + (x - limits[0]) / (limits[1] - limits[0]) * (right - left)
        line(root, left, y + 23, right, y + 23, GRID, 0.55)
        for tick in ticks:
            x = scale_x(tick)
            line(root, x, y + 20, x, y + 26, GRID, 0.55)
            label(root, x, y + 35, tick, anchor="middle", size=6.2, color=MUTED)
        xx = [scale_x(v) for v in values[j]]
        line(root, xx[0], y + 23, xx[1], y + 23, MUTED, 0.8)
        for k, (x, v, col) in enumerate(zip(xx, values[j], (TEAL, BLUE))):
            marker(root, x, y + 23, col, 2.2, diamond=k == 1)
            label(
                root,
                x,
                y + 15,
                f"{v:.{precision}f}" + ("%" if j == 1 else ""),
                color=col,
                bold=True,
                anchor="middle",
                size=6.65,
            )
    save(root, "commute_and_vacancy_access")


def covid():
    with (INPUT / "pandemic.csv").open(encoding="utf8") as f:
        model = list(csv.DictReader(f))
    with (INPUT / "observed.csv").open(encoding="utf8") as f:
        obs = list(csv.DictReader(f))
    assert len(obs) == 84 and len(model) == 2 * 2 * 3 * 84
    metrics = ("employment_rate", "participation_rate", "unemployment_rate")
    names = ("Employment / population", "Labour-force participation", "Unemployment / labour force")
    cases = [("forward_RZ_ON", TEAL, None), ("forward_RZ_OFF", BLUE, "1.2 1.7")]
    root = canvas(
        242,
        "Conditional pandemic trajectories and recovery",
        "Monthly model and observed outcomes under realised separation inputs. Upper panels show levels. Lower panels subtract each series’ own 2019 calendar-month value. The dashed vertical marker identifies April 2020, the calibration anchor.",
    )
    legend = [
        (6, TEAL, None, "Pandemic regime on"),
        (90, BLUE, "1.2 1.7", "Pandemic regime off"),
        (177, INK, "3 2", "LFS observed"),
    ]
    for x, col, dash, name in legend:
        line(root, x, 9, x + 11, 9, col, 1.05, dash)
        label(root, x + 14, 11, name, size=6.65)
    node(root, "rect", x=6, y=18, width=11, height=7, fill="#E9EFF3")
    label(root, 20, 24, "2017–19 holdout", size=6.65)
    node(root, "rect", x=90, y=18, width=11, height=7, fill="#F3F1E9")
    label(root, 104, 24, "2020–21 pandemic window", size=6.65)
    line(root, 207, 17, 207, 25, OCHRE, 0.8, "1.8 1.6")
    label(root, 212, 24, "April 2020 calibration anchor", size=6.65)
    change_scale = (-12, 12, [-12, -6, 0, 6, 12])
    ranges = [
        [(50, 66, [50, 55, 60, 65]), (58, 75, [60, 65, 70, 75]), (2, 24, [5, 10, 15, 20])],
        [change_scale] * 3,
    ]

    def points(pairs):
        return " ".join((f"{x:.3f},{y:.3f}" for x, y in pairs))

    for i, view in enumerate(("level", "same_calendar_2019_change")):
        for j, (metric, name) in enumerate(zip(metrics, names)):
            l = 24 + j * 167
            r = l + 135
            top = 50 + i * 100
            bottom = top + 66
            lo, hi, ticks = ranges[i][j]
            x = lambda month: l + (month + 0.5) / 84 * (r - l)
            y = lambda value: bottom - (value - lo) / (hi - lo) * (bottom - top)
            label(root, l - 19, top - 10, "abcdef"[3 * i + j], size=8, bold=True)
            heading = f"{name} (%)" if i == 0 else "Change from same month in 2019 (pp)"
            label(root, l - 8, top - 10, heading, size=6.9, bold=True)
            node(
                root,
                "rect",
                x=l,
                y=top,
                width=36 / 84 * (r - l),
                height=bottom - top,
                fill="#E9EFF3",
            )
            node(
                root,
                "rect",
                x=l + 36 / 84 * (r - l),
                y=top,
                width=24 / 84 * (r - l),
                height=bottom - top,
                fill="#F3F1E9",
            )
            for v in ticks:
                yy = y(v)
                line(
                    root,
                    l,
                    yy,
                    r,
                    yy,
                    MUTED if i and v == 0 else GRID,
                    0.6 if i and v == 0 else 0.4,
                )
                label(
                    root,
                    l - 3,
                    yy + 2,
                    str(v).replace("-", "−"),
                    anchor="end",
                    size=6.25,
                    color=MUTED,
                )
            line(root, x(39), top, x(39), bottom, OCHRE, 0.7, "1.8 1.6")
            for case, col, dash in cases:
                data = sorted(
                    (
                        z
                        for z in model
                        if z["case"] == case and z["view"] == view and (z["metric"] == metric)
                    ),
                    key=lambda z: (int(z["year"]), int(z["month"])),
                )
                assert len(data) == 84 and all((int(z["n_seeds"]) == 5 for z in data))
                for idx, z in enumerate(data):
                    assert int(z["year"]) == 2017 + idx // 12 and int(z["month"]) == 1 + idx % 12
                band = [(x(k), y(100 * float(z["upper95"]))) for k, z in enumerate(data)]
                band += list(
                    reversed([(x(k), y(100 * float(z["lower95"]))) for k, z in enumerate(data)])
                )
                node(
                    root, "polygon", points=points(band), fill=col, fill_opacity=0.13, stroke="none"
                )

            def curve(vals, col, dash):
                assert all((lo <= v <= hi for v in vals))
                el = node(
                    root,
                    "polyline",
                    points=points([(x(k), y(v)) for k, v in enumerate(vals)]),
                    fill="none",
                    stroke=col,
                    stroke_width=0.97,
                    stroke_linejoin="round",
                )
                if dash:
                    el.set("stroke-dasharray", dash)

            for case, col, dash in cases:
                track = sorted(
                    (
                        z
                        for z in model
                        if z["case"] == case and z["view"] == view and (z["metric"] == metric)
                    ),
                    key=lambda z: (int(z["year"]), int(z["month"])),
                )
                curve([100 * float(z["mean"]) for z in track], col, dash)
            base = {int(z["month"]): float(z[metric]) for z in obs if int(z["year"]) == 2019}
            curve(
                [100 * (float(z[metric]) - (base[int(z["month"])] if i else 0)) for z in obs],
                INK,
                "3 2",
            )
            line(root, l, bottom, r, bottom, color=MUTED)
            for year in (2017, 2019, 2021, 2023):
                xx = x(12 * (year - 2017))
                line(root, xx, bottom, xx, bottom + 2, MUTED, 0.45)
                label(root, xx, bottom + 10, year, anchor="middle", size=6.25, color=MUTED)
    label(root, WIDTH_PT / 2, 237, "Calendar year", anchor="middle", size=6.65)
    save(root, "pandemic_empirical")


def census_locator(root):
    areas = load_json("census_2006_cma_locator.json")["areas"]
    rings = {v["name"]: v["ring_lon_lat"] for v in areas.values()}
    coordinates = [p for ring in rings.values() for p in ring]
    xs, ys = zip(*coordinates)
    midpoint_x = (min(xs) + max(xs)) / 2
    midpoint_y = (min(ys) + max(ys)) / 2
    dx = math.cos(math.radians(midpoint_y))
    factor = min(203 / ((max(xs) - min(xs)) * dx), 155 / (max(ys) - min(ys)))

    def xy(x, y):
        return (942 + (x - midpoint_x) * factor * dx, 348 - (y - midpoint_y) * factor)

    loc = node(root, "g", id="census-2006-metropolitan-area-boundaries")
    node(loc, "title").text = "2006 Census metropolitan areas of Toronto, Oshawa and Hamilton"
    fills = {"Toronto": "#CBE4E4", "Hamilton": "#BED2E3", "Oshawa": "#E7D8B8"}
    for name, ring in rings.items():
        points = " ".join(("%.2f,%.2f" % xy(x, y) for x, y in ring))
        node(
            loc,
            "polygon",
            points=points,
            fill=fills[name],
            stroke="#44616E",
            stroke_width=1.55,
            stroke_linejoin="round",
            id="cma-" + name,
        )
    anchors = {
        "Toronto": (-79.68, 43.99, 827, 320),
        "Hamilton": (-80.0, 43.24, 823, 392),
        "Oshawa": (-78.82, 43.94, 1049, 327),
    }
    for name, (lon, lat, tx, ty) in anchors.items():
        x, y = xy(lon, lat)
        right = tx > x
        end = tx - 6 if right else tx + 6
        bend = end - 12 if right else end + 12
        node(
            loc,
            "polyline",
            points=f"{x:.2f},{y:.2f} {bend},{ty - 5} {end},{ty - 5}",
            stroke="#536A75",
            stroke_width=1.4,
            fill="none",
        )
        marker(loc, x, y, "#35505C", radius=2.15)
        label(loc, tx, ty, name, anchor="start" if right else "end", size=22.2)
    label(loc, 942, 451, "2006 Census metropolitan areas", size=21, anchor="middle", color=MUTED)


def framework():
    source = Path(__file__).with_name("framework.svg")
    root = ET.parse(str(source)).getroot()
    census_locator(root)
    save(root, "framework_renewal")
