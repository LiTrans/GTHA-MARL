from __future__ import annotations
import hashlib
import math
from numbers import Real
from pathlib import Path

SCHEMA = 1
CMAS = (535, 537)
TIMING = "post_matching_and_rewards_before_calendar_advance_and_next_month_demographics"
RATES = ("unemployment_rate_pct", "participation_rate_pct", "employment_population_rate_pct")
COUNT_FIELDS = ("E", "U", "N", "P", "LF")
AGE_BANDS = (
    ("15-24", 15, 24),
    ("25-34", 25, 34),
    ("35-44", 35, 44),
    ("45-54", 45, 54),
    ("55-64", 55, 64),
    ("65+", 65, math.inf),
)


def module_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def integer(value, context, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or (not math.isfinite(value))
        or (value != int(value))
        or (value < minimum)
    ):
        raise ValueError(f"Invalid integer {context}: {value!r}")
    return int(value)


def empty_counts():
    return dict.fromkeys(COUNT_FIELDS, 0)


def rates(counts):
    if counts["P"] <= 0 or counts["LF"] <= 0:
        raise ValueError("Empty population or labour force")
    return dict(
        zip(
            RATES,
            (
                100 * counts["U"] / counts["LF"],
                100 * counts["LF"] / counts["P"],
                100 * counts["E"] / counts["P"],
            ),
        )
    )


def validate_row(row):
    year = integer(row.get("year"), "year", 1)
    month = integer(row.get("month"), "month", 1)
    if month > 12:
        raise ValueError("Month > 12")
    groups = [row["counts"], *row["by_cma"].values(), *row["by_age"].values()]
    for group in groups:
        if set(group) != set(COUNT_FIELDS):
            raise ValueError("Wrong count fields")
        for field, value in group.items():
            integer(value, field)
        if (
            group["E"] + group["U"] + group["N"] != group["P"]
            or group["E"] + group["U"] != group["LF"]
        ):
            raise ValueError("Count identity failure")
    if set(row["by_cma"]) != {str(c) for c in CMAS} or set(row["by_age"]) != {
        b[0] for b in AGE_BANDS
    }:
        raise ValueError("Geographic/age groups differ")
    for split in (row["by_cma"], row["by_age"]):
        for field in COUNT_FIELDS:
            if sum((g[field] for g in split.values())) != row["counts"][field]:
                raise ValueError("Aggregate/component mismatch")
    if set(row["rates"]) != set(RATES):
        raise ValueError("Rate fields differ")
    for name, expected in rates(row["counts"]).items():
        value = row["rates"][name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or (not math.isfinite(value))
            or (abs(value - expected) > 1e-10)
        ):
            raise ValueError("Rate/count mismatch")
    for key in (
        "excluded_other_cma",
        "excluded_under15",
        "workers_total",
        "temporary_layoff_U",
        "nonsearching_unemployed_N",
    ):
        integer(row[key], key)
    if (
        row["workers_total"]
        != row["counts"]["P"] + row["excluded_other_cma"] + row["excluded_under15"]
    ):
        raise ValueError("Inclusion/exclusion identity failure")
    if (
        row["temporary_layoff_U"] > row["counts"]["U"]
        or row["nonsearching_unemployed_N"] > row["counts"]["N"]
    ):
        raise ValueError("Diagnostic state count exceeds stock")
    return (year, month)


class MonthlyModelStocks:

    def __init__(self, start_year):
        self.start_year = integer(start_year, "start_year", 1)
        self.rows = []

    def capture(self, state_manager):
        offset = len(self.rows)
        expected = (self.start_year + offset // 12, 1 + offset % 12)
        actual = (
            integer(state_manager.current_year, "year", 1),
            integer(state_manager.current_month, "month", 1),
        )
        if actual != expected:
            raise ValueError(f"Calendar gap/duplicate: {actual}, expected {expected}")
        row = {
            "year": actual[0],
            "month": actual[1],
            "counts": empty_counts(),
            "by_cma": {str(c): empty_counts() for c in CMAS},
            "by_age": {b[0]: empty_counts() for b in AGE_BANDS},
            "workers_total": len(state_manager.workers),
            "excluded_other_cma": 0,
            "excluded_under15": 0,
            "temporary_layoff_U": 0,
            "nonsearching_unemployed_N": 0,
        }
        ids = set()
        for key, ws in state_manager.workers.items():
            worker = ws.worker
            if str(key) != str(worker.id) or str(worker.id) in ids:
                raise ValueError("Worker ID/registry mismatch")
            ids.add(str(worker.id))
            cma = integer(worker.cma, "worker.cma", 1)
            age = integer(worker.age, "worker.age")
            if cma not in CMAS:
                row["excluded_other_cma"] += 1
                continue
            if age < 15:
                row["excluded_under15"] += 1
                continue
            status = worker.employment_status
            covid = ws.covid_state
            if status not in {"Employed", "Unemployed", "NILF"} or covid not in {
                "none",
                "temporary_layoff",
                "marginally_attached",
            }:
                raise ValueError("Unknown labour-force/COVID state")
            if covid == "temporary_layoff":
                if status == "Employed":
                    raise ValueError("Employed and temporarily laid off")
                bucket = "U"
                row["temporary_layoff_U"] += 1
            elif status == "Employed":
                bucket = "E"
            elif status == "Unemployed":
                searching = worker.is_searching
                if (
                    isinstance(searching, str)
                    or searching is None
                    or searching not in (False, True)
                ):
                    raise ValueError("Invalid search flag")
                bucket = "U" if searching else "N"
                if bucket == "N":
                    row["nonsearching_unemployed_N"] += 1
            else:
                bucket = "N"
            band = next((name for name, lo, hi in AGE_BANDS if lo <= age <= hi))
            for counts in (row["counts"], row["by_cma"][str(cma)], row["by_age"][band]):
                counts[bucket] += 1
                counts["P"] += 1
                if bucket != "N":
                    counts["LF"] += 1
        row["rates"] = rates(row["counts"])
        validate_row(row)
        self.rows.append(row)

    def export(self, completed_months):
        if integer(completed_months, "completed_months", 1) != len(self.rows):
            raise ValueError("Missing monthly observations")
        return {
            "schema_version": SCHEMA,
            "collector_sha256": module_sha256(),
            "metadata": metadata(),
            "monthly": list(self.rows),
        }


def metadata():
    return {
        "geography": "Toronto_Hamilton",
        "model_cmas": list(CMAS),
        "age_min": 15,
        "timing": TIMING,
        "rate_units": "percent_0_to_100",
        "weight": "one_per_model_worker",
        "classification": "E=Employed; U=temporary_layoff_or_searching_Unemployed; N=NILF_or_nonsearching_Unemployed",
        "unknown_status": "refuse",
        "survey_definition": "proxy_not_full_LFS_questionnaire",
    }
