import csv
import io
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
from . import tariff


def indexed(rows, fields):
    return dict(zip(fields, rows, strict=True))


def add_history(folder, seed, history, scenario, groups, monthly, coverage, read):
    log = read(folder / "simulation_log.json")
    tariff.validate_data(log, scenario)
    tariff.require(
        log.get("_rollout_seed") in (None, seed * 100 + history), "Different rollout seed"
    )
    native = read(folder / "TARIFF_NATIVE_MONTHLY_METRICS.json")
    for t, row in enumerate(tariff.macro_months(native, log)):
        for field in tariff.MONTHLY_FIELDS:
            monthly[scenario, t, field, seed] += row[field]
        for field in tariff.FIRM_MONTHLY_FIELDS:
            value = log["trade_war"]["monthly"][t][field]
            monthly[scenario, t, field, seed] += tariff.finite(value, field)
    coverage.extend(tariff.coverage(log, seed, scenario, history, groups))
    return log


def add_rollups(views, scenario, reference, seed, totals, cache):

    def measure(arm, window, name, cell, field, lo, hi):
        key = (arm, window, name, cell, field)
        if key not in cache:
            cache[key] = tariff.rollup_window(views[arm, window], name, cell, field, lo, hi)
        return cache[key]

    for window, (lo, hi) in tariff.WINDOWS.items():
        for domain, name, fields in tariff.ROLLUP_SPECS:
            available = tariff.paired_cell_coverage(
                views[reference, window], views[scenario, window], name, lo, hi
            )
            for item in available:
                cell = item["cell"]
                for field, unit in fields.items():
                    key = (scenario, reference, window, domain, cell, field, unit)
                    cell_total = totals[key][seed]
                    if item["supported_pair"]:
                        estimate = measure(scenario, window, name, cell, field, lo, hi)
                        baseline = measure(reference, window, name, cell, field, lo, hi)
                        cell_total[0] += estimate - baseline
                        cell_total[1] += 1


IDENTITY = ("scenario", "reference_scenario", "window", "domain", "cell", "metric", "units")


def paired_tables(pairs):
    policy, panel = ([], [])
    for key, by_seed in sorted(pairs.items()):
        label = indexed(key, IDENTITY)
        means = {s: tariff.policy_summary(by_seed[s]) for s in tariff.SEEDS}
        for seed, value in means.items():
            policy.append(dict(label, policy_seed=seed, n_paired_histories=50, mean_effect=value))
        panel.append(
            dict(
                label,
                n_pairs_per_policy=50,
                interval_type="pointwise_t_df4_conditional_on_parameters",
                window_role=(
                    "primary" if key[2] == tariff.PRIMARY_WINDOW else "secondary_or_reconstruction"
                ),
                **tariff.panel_summary(means),
            )
        )
    return (policy, panel)


def rollup_tables(totals):
    policy, panel, support = ([], [], [])
    for key, by_seed in sorted(totals.items()):
        label = indexed(key, IDENTITY)
        counts = {s: count for s, (_, count) in by_seed.items()}
        means = {s: value / count for s, (value, count) in by_seed.items() if count}
        eligible = tariff.spatial_report_support(label["domain"].removesuffix("_targets"), counts)
        support.append(
            dict(label, **{f"n_pairs_s{s}": counts.get(s, 0) for s in tariff.SEEDS}, **eligible)
        )
        for seed, value in sorted(means.items()):
            policy.append(
                dict(label, policy_seed=seed, n_supported_pairs=counts[seed], mean_effect=value)
            )
        if eligible["eligible_for_effect_summary"]:
            panel.append(
                dict(
                    label,
                    coverage_rule="All five policies, 25 matched pairs per DA or one per industry or product group",
                    **tariff.panel_summary(means),
                )
            )
    return (policy, panel, support)


def monthly_table(monthly, fields):
    for scenario in tariff.SCENARIOS:
        for t in range(300):
            for field in fields:
                means = {s: monthly[scenario, t, field, s] / 50 for s in tariff.SEEDS}
                unit = tariff.TARGET_FIELDS.get(field, tariff.FIRM_FIELDS.get(field, "fraction"))
                yield dict(
                    scenario=scenario,
                    year=2006 + t // 12,
                    month=t % 12 + 1,
                    metric=field,
                    units=unit,
                    n_histories_per_policy=50,
                    **tariff.panel_summary(means),
                )


def exposure_table(coverage):
    grouped = defaultdict(list)
    for row in coverage:
        grouped[row["scenario"], row["policy_seed"], row["product_group"]].append(
            row["n_firms_min_2027_2030"]
        )
    for (scenario, seed, group), values in sorted(grouped.items()):
        values += [0] * (50 - len(values))
        yield dict(
            scenario=scenario,
            policy_seed=seed,
            product_group=group,
            n_histories=50,
            n_firms_min=min(values),
            n_firms_mean=tariff.mean(values),
            n_firms_max=max(values),
            coverage_basis="minimum_active_firms_in_any_2027_2030_month",
            automotive_headline_minimum=3,
            headline_eligible=group != "automotive" or min(values) >= 3,
        )


def write_table(archive, name, rows):
    rows = iter(rows)
    first = next(rows, None)
    with (
        archive.open(name, "w", force_zip64=True) as binary,
        io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream,
    ):
        if first is not None:
            writer = csv.DictWriter(stream, fieldnames=first.keys())
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(rows)


def export(histories, destination, exposures):
    from .score import read

    paths = {
        (s, h, arm): folder
        for (family, arm, n, m, s, h), folder in histories.items()
        if family == "tariff" and (n, m) == (1000, 80)
    }
    expected = {(s, h, a) for s in tariff.SEEDS for h in range(50) for a in tariff.SCENARIOS}
    if set(paths) != expected:
        raise ValueError("A full five-seed tariff panel requires all 2,000 matched histories.")
    with exposures.open(encoding="utf-8-sig", newline="") as stream:
        groups = sorted({row["product_group"] for row in csv.DictReader(stream)})
    if not groups:
        raise ValueError(
            "Supply an exposure table containing the product groups used for the scenarios."
        )
    pairs = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    monthly = defaultdict(float)
    coverage, observations = ([], [])
    for seed in tariff.SEEDS:
        for history in range(50):
            logs = {
                a: add_history(
                    paths[seed, history, a], seed, history, a, groups, monthly, coverage, read
                )
                for a in tariff.SCENARIOS
            }
            views = {
                (a, window): {
                    "trade_war": {
                        "monthly": [r for r in log["trade_war"]["monthly"] if lo <= r["year"] <= hi]
                    }
                }
                for a, log in logs.items()
                for window, (lo, hi) in tariff.WINDOWS.items()
            }
            cache = {}
            for scenario, reference in tariff.CONTRASTS:
                tariff.assert_pair(logs[reference], logs[scenario], reference)
                for result, value in tariff.effects(logs[reference], logs[scenario]):
                    key = (scenario, reference, *result)
                    pairs[key][seed].append(value)
                    observations.append(
                        dict(
                            indexed(key, IDENTITY),
                            policy_seed=seed,
                            replication_index=history,
                            evaluation_seed=100 * seed + history,
                            effect=value,
                        )
                    )
                add_rollups(views, scenario, reference, seed, totals, cache)
    policy, panel = paired_tables(pairs)
    rollup_policy, rollup_panel, support = rollup_tables(totals)
    tables = {
        "POLICY_EFFECTS.csv": policy,
        "PANEL_EFFECTS.csv": panel,
        "PAIRED_ROLLOUT_EFFECTS.csv": observations,
        "ROLLUP_POLICY_EFFECTS.csv": rollup_policy,
        "ROLLUP_PANEL_EFFECTS.csv": rollup_panel,
        "ROLLUP_SUPPORT.csv": support,
        "MONTHLY_MACRO_SERIES.csv": monthly_table(monthly, tariff.MONTHLY_FIELDS),
        "MONTHLY_FIRM_SERIES.csv": monthly_table(monthly, tariff.FIRM_MONTHLY_FIELDS),
        "PRODUCT_GROUP_REPRESENTATION.csv": exposure_table(coverage),
        "PRODUCT_HISTORY_REPRESENTATION.csv": coverage,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        staging = Path(tmp) / "scores.zip"
        with ZipFile(staging, "w", compression=ZIP_DEFLATED) as archive:
            for name, rows in tables.items():
                write_table(archive, name, rows)
        staging.replace(destination)
