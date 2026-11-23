import argparse
import csv
from pathlib import Path
from .tariff import SCENARIOS


def write_table(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def covid():
    for year in (2020, 2021):
        for month in range(1, 13):
            active = (year, month) >= (2020, 3)
            shock = 0.8 if year == 2020 and 3 <= month <= 6 else 0.25 if active else 0.0
            for industry in range(1, 19):
                yield dict(
                    year=year,
                    month=month,
                    naics_18=industry,
                    regime_active=active,
                    temporary_layoff_entry_hazard=0.03 * shock,
                    temporary_layoff_recall_hazard=0.22 if active else 0.0,
                    marginal_attachment_entry_hazard=0.01 * shock,
                    marginal_attachment_reentry_hazard=0.12 if active else 0.0,
                    telework_capacity=0.3,
                    layoff_exposure_multiplier=1.0,
                    stringency_index=100 * shock,
                    shock_intensity=shock,
                    target_excess_temp_layoff_share=0.0,
                    target_excess_marginal_share=0.0,
                )


def exposures():
    for scenario in SCENARIOS:
        for priority, scope, code, group in (
            (1, "naics18", "3", "manufacturing"),
            (99, "default", "*", "other"),
        ):
            yield dict(
                priority=priority,
                scope=scope,
                scenario=scenario,
                code=code,
                label=group,
                product_group=group,
                export_share_us=0.45 if group == "manufacturing" else 0.12,
                imported_input_share=0.2,
                export_demand_elasticity=1.1,
                tariff_pass_through=0.7,
                productivity_drag_full=0.05,
                employment_elasticity=0.7,
                imported_input_share_basis="US_inputs_over_total_cost_before_tariff_coverage",
                current_outlays_over_revenue=0.8,
            )


def tariff_schedule():
    for scenario_index, scenario in enumerate(SCENARIOS):
        for year in range(2025, 2031):
            for month in range(1, 13):
                level = 0.02 * scenario_index if year >= 2027 else 0.0
                for group in ("manufacturing", "other", "input::naics18:3", "input::default:*"):
                    input_basket = group.startswith("input::")
                    yield dict(
                        scenario=scenario,
                        year=year,
                        month=month,
                        product_group=group,
                        us_tariff_rate=(
                            0.0
                            if input_basket
                            else level * (1 if group == "manufacturing" else 0.5)
                        ),
                        canada_countertariff_rate=0.5 * level if input_basket else 0.0,
                        uncertainty_index=0.0 if input_basket else level,
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate illustrative pandemic and tariff input schedules."
    )
    parser.add_argument("output", type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "covid.csv", covid())
    write_table(out / "tariff.csv", tariff_schedule())
    write_table(out / "exposures.csv", exposures())


if __name__ == "__main__":
    main()
