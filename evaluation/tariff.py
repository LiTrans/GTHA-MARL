import math
import statistics
from typing import Any

SEEDS = (42, 43, 44, 45, 46)
EXPECTED_ROLLOUTS = 50
MIN_DA_PAIRS_PER_SEED = 25
T_CRIT_DF4 = 2.7764451051977987
PRIMARY_WINDOW = "impact_2027_2028"
WINDOWS = {
    "historical_2025": (2025, 2025),
    "transition_2026": (2026, 2026),
    "impact_2027_2028": (2027, 2028),
    "later_2029_2030": (2029, 2030),
    "postshock_2027_2030": (2027, 2030),
}
MACRO_FIELDS = {
    "unemployment_rate": "percentage_points",
    "employment_rate": "percentage_points",
    "participation_rate": "percentage_points",
    "vacancy_rate": "percentage_points",
    "aggregate_job_finding_rate": "percentage_points",
    "aggregate_separation_rate": "percentage_points",
}
FIRM_FIELDS = {
    "realized_employment": "workers",
    "realized_vacancies": "vacancies",
    "realized_hourly_payroll": "hourly_wage_sum",
    "realized_profit": "model_profit_units",
    "effective_output_worker_equivalents": "worker_equivalents",
    "realized_employment_weighted_productivity_multiplier": "index",
    "trade_layoffs": "workers_per_month",
    "target_jobs_lost": "target_jobs",
    "vacancies_withdrawn": "vacancies_per_month",
    "mean_margin_multiplier": "index",
    "mean_productivity_multiplier": "index",
}
ROLLUP_FIELDS = {
    "employment": "workers",
    "vacancies": "vacancies",
    "hourly_payroll": "hourly_wage_sum",
    "profit": "model_profit_units",
    "effective_output_worker_equivalents": "worker_equivalents",
}


def finite(value: Any, context: str) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a measurement")
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"missing or nonnumeric value at {context}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite value at {context}: {value!r}")
    return number


def rate(value: Any, context: str) -> float:
    number = finite(value, context)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"rate outside [0,1] at {context}: {value!r}")
    return number


def nonnegative(value: Any, context: str, *, count: bool = False) -> float:
    number = finite(value, context)
    if number < 0 or (count and number != int(number)):
        raise ValueError(f"invalid nonnegative {('count' if count else 'quantity')} at {context}")
    return number


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def annual_window(rollout: dict, field: str, lo: int, hi: int) -> float:
    years = [int(year) for year in rollout.get("years", [])]
    values = rollout.get(field, [])
    if len(years) != len(values):
        raise ValueError(f"annual series length mismatch for {field}")
    selected = [
        rate(value, f"{field}/{year}") for year, value in zip(years, values) if lo <= year <= hi
    ]
    expected = hi - lo + 1
    if len(selected) != expected:
        raise ValueError(f"{field} window {lo}-{hi} has {len(selected)} years, expected {expected}")
    return mean(selected)


def monthly_window(rollout: dict, field: str, lo: int, hi: int) -> float:
    rows = [row for row in rollout["trade_war"]["monthly"] if lo <= int(row["year"]) <= hi]
    expected = (hi - lo + 1) * 12
    if len(rows) != expected:
        raise ValueError(f"monthly window {lo}-{hi} has {len(rows)} rows, expected {expected}")
    return mean([finite(row[field], f"{field}/{row['year']}-{row['month']}") for row in rows])


def rollup_window(
    rollout: dict, rollup_name: str, cell: str, field: str, lo: int, hi: int
) -> float:
    rows = [row for row in rollout["trade_war"]["monthly"] if lo <= int(row["year"]) <= hi]
    values = []
    if len(rows) != (hi - lo + 1) * 12:
        raise ValueError("rollup window has incomplete monthly coverage")
    for row in rows:
        rollup = row[rollup_name]
        if str(cell) not in rollup:
            raise ValueError(f"unsupported rollup cell: {rollup_name}/{cell}")
        values.append(finite(rollup[str(cell)].get(field), f"{rollup_name}/{cell}/{field}"))
    return mean(values)


def rollup_cells(rollout: dict, rollup_name: str, lo: int, hi: int) -> set[str]:
    cells: set[str] = set()
    for row in rollout["trade_war"]["monthly"]:
        if lo <= int(row["year"]) <= hi:
            cells.update((row.get(rollup_name) or {}).keys())
    return cells


def paired_cell_coverage(control, shock, rollup_name, lo, hi) -> list[dict]:
    cells = rollup_cells(control, rollup_name, lo, hi) | rollup_cells(shock, rollup_name, lo, hi)
    result = []
    expected = (hi - lo + 1) * 12
    for cell in sorted(cells):
        counts = []
        for arm in (control, shock):
            counts.append(
                sum(
                    (
                        cell in row[rollup_name]
                        for row in arm["trade_war"]["monthly"]
                        if lo <= int(row["year"]) <= hi
                    )
                )
            )
        supported = counts == [expected, expected]
        result.append(
            {
                "cell": cell,
                "control_months": counts[0],
                "shock_months": counts[1],
                "required_months": expected,
                "supported_pair": supported,
                "exclusion_reason": "" if supported else "missing_cell_months_not_imputed",
            }
        )
    return result


def assert_preshock_parity(control: dict, shock: dict) -> None:
    c_rows = control["trade_war"]["monthly"]
    s_rows = shock["trade_war"]["monthly"]
    onset = next(
        (
            (int(row["year"]), int(row["month"]))
            for row in s_rows
            if int(row.get("exposed_firms", 0)) > 0
        ),
        None,
    )
    if onset is None:
        raise ValueError("shock arm never activates")
    if len(c_rows) != len(s_rows):
        raise ValueError("paired monthly histories have different lengths")
    for c_row, s_row in zip(c_rows, s_rows):
        key = (int(c_row["year"]), int(c_row["month"]))
        if key >= onset:
            break
        if key != (int(s_row["year"]), int(s_row["month"])):
            raise ValueError("paired monthly keys diverge before shock")
        c_comparable = {name: value for name, value in c_row.items() if name != "scenario"}
        s_comparable = {name: value for name, value in s_row.items() if name != "scenario"}
        if c_comparable != s_comparable:
            raise ValueError(f"paired monthly paths diverge before shock at {key}")
    control_years = [int(year) for year in control.get("years", [])]
    shock_years = [int(year) for year in shock.get("years", [])]
    if control_years != shock_years:
        raise ValueError("paired annual year grids differ")
    for field in MACRO_FIELDS:
        c_values = control.get(field, [])
        s_values = shock.get(field, [])
        if len(c_values) != len(control_years) or len(s_values) != len(shock_years):
            raise ValueError(f"paired annual series length mismatch for {field}")
        for year, c_value, s_value in zip(control_years, c_values, s_values):
            if year >= onset[0]:
                break
            if finite(c_value, f"control/{field}/{year}") != finite(
                s_value, f"shock/{field}/{year}"
            ):
                raise ValueError(f"paired annual paths diverge before shock at {year}: {field}")


def summarize(seed_values: dict[int, float]) -> dict:
    if tuple(sorted(seed_values)) != SEEDS:
        raise ValueError(f"expected seeds {SEEDS}, got {tuple(sorted(seed_values))}")
    values = [seed_values[seed] for seed in SEEDS]
    center = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    half = T_CRIT_DF4 * se
    row = {
        "mean_effect": center,
        "ci95_low": center - half,
        "ci95_high": center + half,
        "sd_across_policy_seeds": sd,
        "n_policy_seeds": len(values),
        "df": len(values) - 1,
    }
    row.update({f"seed_{seed}": seed_values[seed] for seed in SEEDS})
    return row


def spatial_report_support(domain: str, counts: dict[int, int]) -> dict:
    minimum = MIN_DA_PAIRS_PER_SEED if domain == "da" else 1
    if set(counts) - set(SEEDS):
        raise ValueError("The spatial support contains an unknown policy seed")
    for count in counts.values():
        if type(count) is not int or not 0 <= count <= EXPECTED_ROLLOUTS:
            raise ValueError("spatial support must be an integer between 0 and 50")
    represented = all((counts.get(seed, 0) > 0 for seed in SEEDS))
    eligible = all((counts.get(seed, 0) >= minimum for seed in SEEDS))
    return {
        "minimum_pairs_per_policy_seed": minimum,
        "all_policy_seeds_represented": represented,
        "eligible_for_effect_summary": eligible,
        "summary_exclusion_reason": "" if eligible else "insufficient_paired_rollout_support",
    }


def require(value, text):
    if not value:
        raise ValueError(text)


SCENARIOS = (
    "control",
    "policy_c_metal_chapter_0_8",
    "policy_ca_no_relief_ust_default",
    "policy_claim_share_pre_shock",
    "policy_e_high",
    "policy_e_low",
    "policy_record_persistent",
    "policy_vehicles_unapproved",
)
CENTRAL = "policy_record_persistent"
CONTRASTS = tuple(((s, "control") for s in SCENARIOS if s != "control")) + tuple(
    ((s, CENTRAL) for s in SCENARIOS if s not in ("control", CENTRAL))
)
MONTHLY_FIELDS = (
    "unemployment_rate",
    "employment_rate",
    "participation_rate",
    "vacancy_rate",
    "job_finding_rate",
)
TARGET_FIELDS = {
    "baseline_target_jobs": "funded_target_positions",
    "shocked_target_jobs": "effective_target_positions",
}
FIRM_MONTHLY_FIELDS = (
    "realized_profit",
    "realized_vacancies",
    "effective_output_worker_equivalents",
    "shocked_target_jobs",
    "realized_employment",
)
ROLLUPS = (
    ("industry", "outcomes_by_industry"),
    ("da", "outcomes_by_da"),
    ("product_group", "outcomes_by_product_group"),
)
ROLLUP_SPECS = tuple(((domain, name, ROLLUP_FIELDS) for domain, name in ROLLUPS)) + tuple(
    (
        (
            domain + "_targets",
            "by_" + ("product_group" if domain == "product_group" else domain),
            TARGET_FIELDS,
        )
        for domain, _ in ROLLUPS
    )
)


def validate_data(log, scenario):
    require(log.get("years") == list(range(2006, 2031)), "Annual calendar must be2006–2030")
    for field in MACRO_FIELDS:
        require(len(log.get(field, [])) == 25, "Incomplete annual macro " + field)
        for v in log[field]:
            rate(v, field)
    trade = log.get("trade_war", {})
    require(
        trade.get("provenance", {}).get("scenario") == scenario
        and trade.get("provenance", {}).get("enabled") is True,
        "Trade provenance differs",
    )
    rows = trade.get("monthly", [])
    require(
        [(r["year"], r["month"]) for r in rows]
        == [(y, m) for y in range(2006, 2031) for m in range(1, 13)],
        "Trade calendar differs",
    )
    for row in rows:
        require(row["scenario"] == scenario, "Trade monthly scenario differs")
        for f in FIRM_FIELDS:
            value = finite(row.get(f), f)
            require(f == "realized_profit" or value >= 0, "Negative firm quantity")
        for _, name in ROLLUPS:
            require(isinstance(row.get(name), dict), "Missing current rollup " + name)
            for cell, values in row[name].items():
                for f in (*ROLLUP_FIELDS, "active_firms"):
                    value = finite(values.get(f), f"{name}/{cell}/{f}")
                    require(f == "profit" or value >= 0, "Negative rollup quantity")
        for domain, name, fields in ROLLUP_SPECS:
            if domain.endswith("_targets"):
                require(isinstance(row.get(name), dict), "Missing target summary " + name)
                for cell, values in row[name].items():
                    for f in fields:
                        nonnegative(values.get(f), f"{name}/{cell}/{f}", count=True)
        require(
            isinstance(row.get("by_product_group"), dict), "No current product exposure coverage"
        )
        if scenario == "control":
            require(
                all(
                    (
                        row[k] == 0
                        for k in (
                            "exposed_firms",
                            "affected_firms",
                            "target_jobs_lost",
                            "trade_layoffs",
                            "vacancies_withdrawn",
                        )
                    )
                ),
                "Nonneutral tariff control",
            )


def assert_pair(reference, shock, reference_scenario):
    if reference_scenario == "control":
        assert_preshock_parity(reference, shock)
    else:
        onset = lambda r: next(
            (
                (int(x["year"]), int(x["month"]))
                for x in r["trade_war"]["monthly"]
                if int(x.get("exposed_firms", 0)) > 0
            ),
            None,
        )
        a, b = (onset(reference), onset(shock))
        require(a is not None and b is not None, "Both policies must activate")
        (
            assert_preshock_parity(shock, reference)
            if a < b
            else assert_preshock_parity(reference, shock)
        )


def effects(reference, shock):
    for window, (lo, hi) in WINDOWS.items():
        for field, unit in MACRO_FIELDS.items():
            yield (
                (window, "macro", "all", field, unit),
                100
                * (annual_window(shock, field, lo, hi) - annual_window(reference, field, lo, hi)),
            )
        for field, unit in FIRM_FIELDS.items():
            yield (
                (window, "firm_aggregate", "all", field, unit),
                monthly_window(shock, field, lo, hi) - monthly_window(reference, field, lo, hi),
            )
        for field, unit in TARGET_FIELDS.items():
            yield (
                (window, "firm_targets", "all", field, unit),
                monthly_window(shock, field, lo, hi) - monthly_window(reference, field, lo, hi),
            )


def coverage(log, seed, scenario, index, declared_groups=()):
    rows = [r for r in log["trade_war"]["monthly"] if 2027 <= r["year"] <= 2030]
    require(len(rows) == 48, "Coverage requires 48 monthly observations")
    groups = set(declared_groups).union(*(set(r["by_product_group"]) for r in rows))
    result = []
    for group in sorted(groups):
        values = [r["by_product_group"].get(group, {}).get("firms", 0) for r in rows]
        for v in values:
            nonnegative(v, "product active firms", count=True)
        result.append(
            {
                "policy_seed": seed,
                "scenario": scenario,
                "replication_index": index,
                "product_group": group,
                "n_firms_min_2027_2030": min(values),
                "represented_all48months": min(values) > 0,
                "automotive_headline_eligible": group != "automotive" or min(values) >= 3,
            }
        )
    return result


def macro_months(payload, log):
    require(
        payload["schema"] == "tariff_native_monthly_metrics_v1",
        "Wrong schema of the monthly record",
    )
    rows = payload["months"]
    require(len(rows) == 300, "Missing 300 monthly metrics")
    for i, row in enumerate(rows):
        require(
            row["calendar"] == [2006 + i // 12, 1 + i % 12],
            "The calendar of the monthly metrics differs",
        )
        for f in (*MONTHLY_FIELDS, "separation_rate"):
            rate(row[f], f)
    for y in range(25):
        for f in (*MONTHLY_FIELDS, "separation_rate"):
            key = {
                "job_finding_rate": "aggregate_job_finding_rate",
                "separation_rate": "aggregate_separation_rate",
            }.get(f, f)
            require(
                abs(mean([r[f] for r in rows[y * 12 : y * 12 + 12]]) - log[key][y]) < 1e-12,
                "Monthly and annual means differ",
            )
    return rows


def policy_summary(values, required=50):
    require(len(values) == required, "Policy summary requires50paired histories")
    return mean(values)


def panel_summary(values):
    return summarize(values)
