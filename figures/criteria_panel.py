import statistics
import xml.etree.ElementTree as ET
from io import BytesIO
from pypdf import PdfReader, PdfWriter
from reportlab.graphics import renderPDF, renderSVG
import learning as art

LABELS = {
    "U_LF_native_pct": "Unemployment (% of labour force)",
    "NILF_35_44_native_pct": "Non-participation, ages 35–44",
    "NILF_55_64_native_pct": "Non-participation, ages 55–64",
    "NILF_65_99_native_pct": "Non-participation, ages 65–99",
}


def read_criteria(path):
    scores = [r for r in art.read_table(path) if r["case"] == "historical1000"]
    targets = ("4–8", "≤20", "28–42", "76–92")
    for (key, label), target in zip(LABELS.items(), targets):
        rows = sorted(
            (r for r in scores if r["metric"] == key), key=lambda r: int(r["policy_seed"])
        )
        if tuple((int(r["policy_seed"]) for r in rows)) != art.SEEDS:
            raise ValueError(f"Incomplete policy scores for {key}")
        means = [
            statistics.mean((float(r[field]) for r in rows))
            for field in ("full_11_mean_pct", "last_3_mean_pct")
        ]
        passes = sum((r["final_classification"] == "PASS" for r in rows))
        yield (label, target, *means, f"{passes}/{len(rows)}")


def draw_panel(path):
    art.configure_fonts()
    panel = art.Artwork(116)
    panel.line(5, 6, art.WIDTH - 5, 6)
    panel.text(6, 21, "f", 9.5, bold=True)
    panel.text(19, 21, "Registered criteria for simulated non-immigrants", 9.5, bold=True)
    panel.text(
        art.WIDTH - 5, 21, "1,000 initial residents and 80 firms", 7.7, art.SOFT, anchor="end"
    )
    target_x, full_x, late_x, count_x = (212, 285, 365, 462)
    panel.text((full_x + late_x) / 2, 34, "Five-policy mean (%)", 7.7, art.SOFT, anchor="middle")
    panel.text(6, 47, "Criterion", 8, bold=True)
    for x, text in [(target_x, "Target (%)"), (full_x, "2006–2016"), (late_x, "2014–2016")]:
        panel.text(x, 47, text, 8, bold=True, anchor="middle")
    panel.text(count_x, 37, "Policies passing", 8, bold=True, anchor="middle")
    panel.text(count_x, 47, "both windows", 8, bold=True, anchor="middle")
    panel.line(5, 52, art.WIDTH - 5, 52, art.ZERO, 0.65)
    for row, (label, target, full, late, count) in enumerate(read_criteria(path)):
        y = 65 + row * 14
        panel.text(6, y, label, 8)
        for x, text in [(target_x, target), (full_x, f"{full:.2f}"), (late_x, f"{late:.2f}")]:
            panel.text(x, y, text, 8.2, anchor="middle")
        panel.text(count_x, y, count, 8.2, bold=True, anchor="middle")
    panel.line(5, 113, art.WIDTH - 5, 113)
    return panel


def append_pdf(panel, path):
    base = PdfReader(BytesIO(path.with_suffix(".pdf").read_bytes())).pages[0]
    vector = renderPDF.drawToString(panel.drawing, initialFontName="Arial")
    table = PdfReader(BytesIO(vector)).pages[0]
    writer = PdfWriter()
    page = writer.add_blank_page(base.mediabox.width, float(base.mediabox.height) + panel.height)
    page.merge_translated_page(base, 0, panel.height)
    page.merge_page(table)
    writer.write(path.with_suffix(".pdf"))


def append_svg(panel, path):
    svg = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg)
    base = ET.parse(path.with_suffix(".svg")).getroot()
    _, _, width, height = map(float, base.attrib["viewBox"].split())
    new_height = height + panel.height
    base.set("height", f"{new_height:g}pt")
    base.set("viewBox", f"0 0 {width:g} {new_height:g}")
    base.find(f"{{{svg}}}title").text = (
        "Historical labour-market rates, learning effects and registered criteria"
    )
    base.find(f"{{{svg}}}desc").text = (
        "Panels a to e show the historical rates and matched control comparisons. Panel f gives the four registered targets, the mean rates in each scoring window and the number of policies passing both."
    )
    table = renderSVG.drawToString(panel.drawing)
    table = table.replace("font-family: Arial-Bold;", "font-family: Arial; font-weight: bold;")
    group = ET.fromstring(table).find(f"{{{svg}}}g")
    del group.attrib["id"]
    del group.attrib["style"]
    placed = ET.SubElement(
        base, f"{{{svg}}}g", {"id": "criteria", "transform": f"translate(0,{height:g})"}
    )
    placed.append(group)
    ET.ElementTree(base).write(path.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)
