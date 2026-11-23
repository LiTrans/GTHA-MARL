import argparse
import csv
import math
from pathlib import Path
from learning import FAMILIES, FIELDS, RECRUIT, SEEDS


def write_table(path, fields, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(records)


def recruitment():
    levels = [(87, 24, 71, 6.8), (51, 31, 62, 6.7), (61, 40, 65, 6.6)]
    for seed in SEEDS:
        for index, metric in enumerate(RECRUIT):
            values = [row[index] + 0.04 * (seed - 44) * (i + 1) for i, row in enumerate(levels)]
            is_percent = metric.endswith("_pct")
            for family, value in zip(FAMILIES, values):
                unit = "percent" if is_percent else "per_1000_initial_per_month"
                yield ("level", family, seed, "", metric, unit, value)
            for family, value in zip(FAMILIES[1:], values[1:]):
                unit = "percentage_points" if is_percent else "per_1000_initial_per_month"
                yield ("difference", family, seed, "", metric, unit, values[0] - value)


def capacity():
    series = {
        "flow_U_E": 5,
        "flow_U_N": 3,
        "funded_active": 600,
        "effective_claims": 570,
        "resident_filled": 550,
    }
    for month in range(132):
        for metric, level in series.items():
            mean = level * (1 + 0.03 * math.sin(month / 12))
            yield (2006 + month // 12, 1 + month % 12, metric, 5, mean, mean * 0.99, mean * 1.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate illustrative inputs for Fig3 and S9.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_table(args.output / "learning/recruitment_learning_source.csv", FIELDS, recruitment())
    write_table(
        args.output / "diagnostics/historical_capacity_and_exits.csv",
        ("year", "month", "metric", "n_seeds", "mean", "lower95", "upper95"),
        capacity(),
    )
