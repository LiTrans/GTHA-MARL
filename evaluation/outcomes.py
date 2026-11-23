from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import fmean
from . import cohorts, holdout, job_finding, tariff
from .score import interval, lines, read, write_csv

RATES = {
    "employment_rate": "employment_rate",
    "participation_rate": "participation_rate_lfs",
    "unemployment_rate": "unemployment_rate_lfs",
    "temp_layoff_share_population": "temp_layoff_share_population",
    "marginal_share_population": "marginal_share_population",
}


def score_forward(histories, output, targets=None, december_lfs=None):
    pooled = defaultdict(list)
    seen = set()
    for (family, arm, workers, firms, seed, h), folder in histories.items():
        if family != "forward" or (workers, firms) != (1000, 80):
            continue
        seen.add((arm, seed, h))
        data = read(folder / "FORWARD_OUTCOMES_MONTHLY.json")["months"]
        if [tuple(row["period"]) for row in data] != [
            (2006 + i // 12, i % 12 + 1, i) for i in range(216)
        ]:
            raise ValueError(f"Incomplete forward calendar in {folder}")
        for row in data[132:]:
            y, m, _ = row["period"]
            counts = row["after_market"]["aggregate"]
            for metric, name in RATES.items():
                pooled[arm, seed, y, m, metric].append(float(counts[name]))
    arms = ("RZ_ON", "FN_ON", "FN_OFF", "RZ_OFF")
    if seen != {(a, s, h) for a in arms for s in tariff.SEEDS for h in range(50)}:
        raise ValueError("The complete forward comparison requires four paired arms at 1,000/80")
    means = {k: fmean(v) for k, v in pooled.items()}
    write_csv(
        output / "forward_policy.csv",
        (
            dict(arm=a, seed=s, year=y, month=m, metric=metric, level=v)
            for (a, s, y, m, metric), v in sorted(means.items())
        ),
    )
    effects, ensemble = ([], [])
    curves = {}
    for y in range(2017, 2024):
        for m in range(1, 13):
            for metric in RATES:
                for a in arms:
                    values = [means[a, s, y, m, metric] for s in tariff.SEEDS]
                    curves[a, y, m, metric] = fmean(values)
                    ensemble.append(dict(arm=a, year=y, month=m, metric=metric, **interval(values)))
                for mode in ("RZ", "FN"):
                    differences = [
                        100
                        * (
                            means[mode + "_ON", s, y, m, metric]
                            - means[mode + "_OFF", s, y, m, metric]
                        )
                        for s in tariff.SEEDS
                    ]
                    effects.append(
                        dict(mode=mode, year=y, month=m, metric=metric, **interval(differences))
                    )
    write_csv(output / "forward_paired.csv", effects)
    write_csv(output / "forward_ensemble.csv", ensemble)
    if targets:
        with Path(targets).open(newline="", encoding="utf-8-sig") as stream:
            observed = {}
            for row in csv.DictReader(stream):
                key = (int(row["year"]), int(row["month"]))
                if key in observed:
                    raise ValueError("Duplicate survey month")
                values = {}
                for metric in RATES:
                    raw = row[metric].strip()
                    stock = metric in ("temp_layoff_share_population", "marginal_share_population")
                    if not raw and stock and (key[0] not in (2019, 2020, 2021) or key == (2020, 4)):
                        continue
                    value = float(raw)
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Survey rates must be fractions: {key}, {metric}")
                    values[metric] = value
                observed[key] = values
        if set(observed) != {(y, m) for y in range(2017, 2024) for m in range(1, 13)}:
            raise ValueError("Expected all 84 survey months, 2017–2023")
        windows = {"holdout": (2017, 2019), "pandemic": (2020, 2021), "recovery": (2022, 2023)}
        rows = []
        december = (
            dict(zip(holdout.METRICS, holdout.december(december_lfs))) if december_lfs else {}
        )
        for arm in (*arms, *(["december_2016"] if december else [])):
            for metric in RATES:
                stock = metric in ("temp_layoff_share_population", "marginal_share_population")
                for window, (lo, hi) in windows.items():
                    if (
                        stock
                        and window != "pandemic"
                        or (arm == "december_2016" and (stock or window != "holdout"))
                    ):
                        continue
                    errors = []
                    for (y, m), measured in sorted(observed.items()):
                        if not lo <= y <= hi or (y, m) == (2020, 4):
                            continue
                        prediction = (
                            december[metric]
                            if arm == "december_2016"
                            else curves[arm, y, m, metric]
                        )
                        target = measured[metric]
                        if stock:
                            prediction -= curves[arm, 2019, m, metric]
                            target -= observed[2019, m][metric]
                        errors.append(100 * (prediction - target))
                    rows.append(
                        dict(
                            arm=arm,
                            metric=metric,
                            window=window,
                            months=len(errors),
                            mae_pp=fmean(map(abs, errors)),
                            rmse_pp=math.sqrt(fmean((v * v for v in errors))),
                            bias_pp=fmean(errors),
                        )
                    )
        write_csv(output / "forward_fit.csv", rows)


def score_spatial(histories, output):
    import numpy as np

    rows = []
    for (family, _, size, firms, seed, h), folder in histories.items():
        if family != "spatial":
            continue
        row = dict(scale=size, seed=seed, history=h)
        with gzip.open(folder / "SPATIAL_DISTRIBUTIONS.json.gz", "rt", encoding="utf8") as stream:
            payload = json.load(stream)
        if "2016" not in payload["windows"]:
            raise ValueError("Spatial scoring needs complete histories through 2016")
        for name in (
            "stock_time_all_valid",
            "opening_U_to_E_time_all_valid",
            "stock_straight_line_km_all_valid",
        ):
            d = payload["windows"]["2016"][name]
            for metric in ("n", "mean", "median", "p90", "p95", "zero_count"):
                row[name + "_" + metric] = d[metric]
        wage_groups = payload["industry_worker_positive_hourly_wages"]["full132"]
        wages = np.sort(np.concatenate([np.asarray(d["values"]) for d in wage_groups.values()]))
        if not len(wages) or (wages <= 0).any():
            raise ValueError(f"Invalid positive-wage sample in {folder}")
        p10, median, p90 = np.quantile(wages, [0.1, 0.5, 0.9], method="linear")
        n = len(wages)
        row.update(
            wage_median=float(median),
            wage_p10=float(p10),
            wage_p90=float(p90),
            wage_p90_p10=float(p90 / p10),
            wage_gini=float(2 * np.dot(np.arange(1, n + 1) / n, wages) / wages.sum() - 1 - 1 / n),
        )
        months = [
            r for r in lines(folder / "SPATIAL_RENEWAL_MONTHS.jsonl.gz") if r["calendar"][0] == 2016
        ]
        if [r["calendar"][1] for r in months] != list(range(1, 13)):
            raise ValueError("Expected a complete 2016 spatial panel")
        eligible = reached = total_u = weighted_vacancies = zero = 0
        for r in months:
            if r["availability"]["unit"] != "min" or r["availability"]["threshold"] != 60:
                raise ValueError("The spatial access measure requires travel time in minutes")
            for cell in r["DA_measurements"]:
                u, v = (cell["U_raw"], cell["reachable_closing_vacancies"])
                total_u += u
                if v is not None:
                    eligible += u
                    reached += u * (v > 0)
                    weighted_vacancies += u * v
                    zero += u * (r["availability"]["closing_vacancies"] == 0)
        report = read(folder / "SPATIAL_OBSERVER_REPORT.json")
        closing = [r for r in report["monthly_coverage"] if r["calendar"][0] == 2016]
        if [r["calendar"][1] for r in closing] != list(range(1, 13)):
            raise ValueError("Missing 2016 market-close vacancy counts")
        row.update(
            fraction_with_access=reached / eligible,
            mean_reachable=weighted_vacancies / eligible,
            fraction_in_zero_market_vacancy_months=zero / eligible,
            known_access_u_person_months=eligible,
            u_person_months=total_u,
            mean_closing_vacancies=fmean((r["availability"]["closing_vacancies"] for r in closing)),
            fraction_months_zero_closing_vacancies=fmean(
                (r["availability"]["closing_vacancies"] == 0 for r in closing)
            ),
            snapshot_count=sum((r["employed_commute_snapshot_n"] for r in closing)),
        )
        rows.append(row)
    if {(r["scale"], r["seed"], r["history"]) for r in rows} != {
        (n, s, h) for n in (1000, 2500) for s in tariff.SEEDS for h in range(50)
    }:
        raise ValueError("Spatial scoring needs both scales, five seeds and 50 histories each")
    write_csv(output / "spatial_history.csv", rows)
    keys = [k for k in rows[0] if k not in ("scale", "seed", "history")]
    policies = []
    for size in (1000, 2500):
        for seed in tariff.SEEDS:
            group = [r for r in rows if (r["scale"], r["seed"]) == (size, seed)]
            policies.append(
                dict(scale=size, seed=seed, **{k: fmean((r[k] for r in group)) for k in keys})
            )
    write_csv(output / "spatial_policy.csv", policies)


def score_entrants(histories, output):
    by_seed = defaultdict(list)
    first_jobs, next_month = ({}, {})
    for (family, arm, n, m, seed, h), folder in histories.items():
        if (family, arm, n, m) != ("forward", "RZ_ON", 1000, 80):
            continue
        source = folder / "FORWARD_SOURCE_MONTHS.jsonl.gz"
        flows, _, people = cohorts.history_tables(source, seed, h)
        first_jobs[seed, h] = cohorts.first_job_counts(people)
        next_month[seed, h] = job_finding.linked_counts(flows)
        records, trajectory = cohorts.holdout_trajectories(
            source, folder / "FORWARD_OUTCOMES_MONTHLY.json", seed, h
        )
        values = defaultdict(list)
        for _, _, year, _, category, _, _, _, pp, share, *_ in records:
            window = "2006-2016" if year <= 2016 else "2017-2019"
            values[window, "entry_group", category, "contribution_pp"].append(pp)
            values[window, "entry_group", category, "unemployed_stock_pct"].append(share)
        for _, _, _, _, category, _, pp, share, _ in trajectory:
            values["2017-2019", "trajectory", category, "contribution_pp"].append(pp)
            values["2017-2019", "trajectory", category, "unemployed_stock_pct"].append(share)
        for metric in ("contribution_pp", "unemployed_stock_pct"):
            components = [
                values["2017-2019", "trajectory", category, metric]
                for category in ("left_active_U_without_employment", "continuous_observed_active_U")
            ]
            values["2017-2019", "trajectory", "never_observed_employed", metric] = [
                None if a is None or b is None else a + b for a, b in zip(*components)
            ]
        for (window, kind, category, metric), series in values.items():
            if len(series) != (132 if window == "2006-2016" else 36):
                raise ValueError("Incomplete monthly entrant series")
            supported = [v for v in series if v is not None]
            if supported:
                by_seed[window, kind, category, metric, seed].append(fmean(supported))
    if set(first_jobs) != {(s, h) for s in tariff.SEEDS for h in range(50)}:
        raise ValueError("Entrant analysis needs all 250 realised-driver, pandemic-on histories")
    rows = []
    for key, histories_supported in sorted(by_seed.items()):
        window, kind, cat, metric, seed = key
        rows.append(
            dict(
                window=window,
                group=kind,
                category=cat,
                metric=metric,
                seed=seed,
                mean=fmean(histories_supported),
                histories=len(histories_supported),
            )
        )
    write_csv(output / "entrant_policy.csv", rows)
    keys = ("window", "group", "category", "metric")
    panels = defaultdict(list)
    for row in rows:
        panels[tuple((row[k] for k in keys))].append(row["mean"])
    write_csv(
        output / "entrant_panel.csv",
        (dict(zip(keys, key)) | interval(values) for key, values in sorted(panels.items())),
    )
    risks = []
    for seed in tariff.SEEDS:
        group = [first_jobs[seed, h] for h in range(50)]
        for month in (1, 3, 6, 12):
            supported = [r for r in group if r["entered_unemployed"]]
            if not supported:
                continue
            rates = {
                key
                + "_pct": fmean(
                    (100 * r[f"{key}_{month}"] / r["entered_unemployed"] for r in supported)
                )
                for key in ("ever_employed", "exited_before_job")
            }
            risks.append(
                dict(
                    seed=seed,
                    month=month,
                    histories_with_entrants=len(supported),
                    total_entries=sum((r["all_entries"] for r in group)),
                    entered_unemployed=sum((r["entered_unemployed"] for r in group)),
                    entered_unemployed_searching=sum(
                        (r["entered_searching_unemployed"] for r in group)
                    ),
                    **rates,
                )
            )
    write_csv(output / "entrant_first_employment_policy.csv", risks)
    write_csv(
        output / "entrant_first_employment_panel.csv",
        (
            dict(
                month=month,
                metric=metric,
                **interval([r[metric] for r in risks if r["month"] == month]),
            )
            for month in (1, 3, 6, 12)
            for metric in ("ever_employed_pct", "exited_before_job_pct")
            if any((r["month"] == month for r in risks))
        ),
    )
    flow_rows = job_finding.linked_rates(next_month, tariff.SEEDS)
    write_csv(output / "consecutive_month_job_finding_policy.csv", flow_rows)
    rates = defaultdict(list)
    for row in flow_rows:
        rates[row["series"], row["period"]].append(row["rate_pct"])
    write_csv(
        output / "consecutive_month_job_finding_panel.csv",
        (
            dict(series=series, period=period, **interval(values))
            for (series, period), values in sorted(rates.items())
        ),
    )
