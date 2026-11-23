import math
import statistics
import pandas as pd
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
from learning import SEEDS

WIDTH, HEIGHT = (460, 418)
METRICS = ("worker/mean_reward_raw", "firm/mean_reward_raw", "worker_kl", "firm_kl")
COUNTS = (195, 200, 100)
OFFSETS = (0, 195, 395)
CSV_NAME = "training_diagnostics_source.csv"
COLORS = ("#7896A5", "#AB95A3", "#879A85", "#8F97AD", "#AE947B")
DASHES = (None, (3.2, 1.3), (1.3, 1.2), (3.8, 1.1, 1, 1.1), (5, 1.5))
INK, SOFT, GRID, MEAN = ("#1C2932", "#465A65", "#DBE2E4", "#006A67")


def require(value, message):
    if not value:
        raise ValueError(message)


def render(path, output):
    source = pd.read_csv(path, float_precision="round_trip")
    grouped = {
        (seed, stage): part for (seed, stage), part in source.groupby(["policy_seed", "stage"])
    }
    require(
        set(grouped) == {(s, t) for s in SEEDS for t in (1, 2, 3)},
        "Training source has incorrect identities.",
    )
    for (seed, stage), part in grouped.items():
        phase_rows_check = list(range(1, COUNTS[stage - 1] + 1))
        require(
            part.stage_update.tolist() == phase_rows_check
            and part.cumulative_update.tolist()
            == [u + OFFSETS[stage - 1] for u in phase_rows_check],
            f"Update sequence differs in seed {seed}, stage {stage}",
        )
    drawing = Drawing(WIDTH, HEIGHT)

    def text(x, y, value, size=7.1, color=INK, bold=False, align="start"):
        drawing.add(
            String(
                x,
                y,
                value,
                fontSize=size,
                fontName="Arial-Bold" if bold else "Arial",
                fillColor=HexColor(color),
                textAnchor=align,
            )
        )

    def line(x, y, x1, y1, color, width=0.45, dash=None):
        drawing.add(
            Line(x, y, x1, y1, strokeColor=HexColor(color), strokeWidth=width, strokeDashArray=dash)
        )

    def curve(points, color, width, dash=None):
        drawing.add(
            PolyLine(
                [v for pair in points for v in pair],
                strokeColor=HexColor(color),
                strokeWidth=width,
                strokeDashArray=dash,
                fillColor=None,
            )
        )

    for j, seed in enumerate(SEEDS):
        x = 7 + j * 57
        line(x, 408, x + 13, 408, COLORS[j], 0.95, DASHES[j])
        text(x + 16, 405.5, f"Seed {seed}")
    line(307, 408, 322, 408, MEAN, 1.5)
    text(325, 405.5, "Five-seed mean", bold=True)
    columns = ((35, 184), (201, 354), (371, 452))
    for i, ((left, right), role) in enumerate(
        zip(columns, ("Workers learn", "Workers and firms learn", "Workers and firms learn")), 1
    ):
        center = (left + right) / 2
        text(center, 386, f"Stage {i}", 8.5, bold=True, align="middle")
        text(center, 375, role, 7, SOFT, align="middle")
    specs = [
        ("Worker reward", "mean raw reward", (2.4, 2.54), (2.4, 2.46, 2.52), 1, False),
        ("Firm reward", "mean raw reward", (0.615, 0.72), (0.62, 0.66, 0.7), 1, False),
        (
            "Worker policy update",
            "KL × 1,000 (log scale)",
            (0.4, 18),
            (0.5, 1, 2, 4, 8, 16),
            1000,
            True,
        ),
        ("Firm policy update", "KL × 1,000", (0, 9), (0, 3, 6, 9), 1000, False),
    ]
    for row, (metric, spec) in enumerate(zip(METRICS, specs)):
        title, unit, bounds, ticks, scale, log = spec
        top = 344 - row * 87
        bottom = top - 58
        text(6, top + 14, chr(97 + row), 9.5, bold=True)
        text(21, top + 14, f"{title}  ·  {unit}", 8, bold=True)
        transform = math.log if log else float
        low, high = map(transform, bounds)

        def y(value):
            return bottom + (top - bottom) * (transform(value) - low) / (high - low)

        for col, (left, right) in enumerate(columns):
            stage = col + 1
            count, offset = (COUNTS[col], OFFSETS[col])

            def x(index):
                return left + (right - left) * index / (count - 1)

            if metric == "firm_kl" and stage == 1:
                drawing.add(
                    Rect(
                        left,
                        bottom,
                        right - left,
                        top - bottom,
                        fillColor=HexColor("#F3F5F5"),
                        strokeColor=None,
                    )
                )
                text(
                    (left + right) / 2,
                    bottom + 32,
                    "Fixed firm action",
                    7.7,
                    SOFT,
                    bold=True,
                    align="middle",
                )
                text(
                    (left + right) / 2,
                    bottom + 20,
                    "Firm actor update skipped",
                    7,
                    SOFT,
                    align="middle",
                )
            else:
                for v in ticks:
                    line(left, y(v), right, y(v), GRID)
                    if col == 0 or (metric == "firm_kl" and col == 1):
                        val = f"{v:.2f}" if row < 2 else f"{v:g}"
                        text(left - 4, y(v) - 2, val, 6.1, SOFT, align="end")
                values = []
                for j, seed in enumerate(SEEDS):
                    series = [v * scale for v in grouped[seed, stage][metric]]
                    require(
                        all((math.isfinite(v) and bounds[0] <= v <= bounds[1] for v in series)),
                        f"Data outside {metric} limits.",
                    )
                    values.append(series)
                    curve([(x(i), y(v)) for i, v in enumerate(series)], COLORS[j], 0.44, DASHES[j])
                curve(
                    [(x(i), y(statistics.mean(v))) for i, v in enumerate(zip(*values))], MEAN, 1.1
                )
            line(left, bottom, right, bottom, SOFT)
            if row == 3:
                for index in (0, count // 2 - 1, count - 1):
                    xx = x(index)
                    line(xx, bottom, xx, bottom - 2.5, SOFT)
                    text(xx, bottom - 12, str(offset + index + 1), 6.8, SOFT, align="middle")
    text(WIDTH / 2, 1.7, "Cumulative PPO update", 7.5, SOFT, align="middle")
    title = "Worker and firm training diagnostics over three stages"
    pdf = Canvas(
        str(output / "training_diagnostics.pdf"),
        pagesize=(WIDTH, HEIGHT),
        initialFontName="Arial",
        invariant=True,
    )
    pdf.setTitle(title)
    renderPDF.draw(drawing, pdf, 0, 0)
    pdf.showPage()
    pdf.save()
    svg = renderSVG.drawToString(drawing).replace(
        "font-family: Arial-Bold;", "font-family: Arial; font-weight: bold;"
    )
    svg = svg.replace(f'width="{WIDTH}"', f'width="{WIDTH}pt"', 1).replace(
        f'height="{HEIGHT}"', f'height="{HEIGHT}pt"', 1
    )
    svg = svg.replace("<title>...</title>", f"<title>{title}</title>", 1)
    svg = svg.replace(
        "<desc>...</desc>",
        "<desc>Reward and policy update trajectories for five seeds over 495 updates. The firm action was fixed during Stage 1.</desc>",
        1,
    )
    (output / "training_diagnostics.svg").write_text(svg, encoding="utf-8")
