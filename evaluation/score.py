import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics as stats
import sys
from . import acceptance, tariff
from .job_finding import prime_age


def read(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def lines(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        yield from map(json.loads, stream)


def inventory(root):
    result = {}
    checkpoints = {}
    for meta in root.rglob("history.json"):
        item = read(meta)
        seed, history = (item["policy_seed"], item["replication_index"])
        key = (item["family"], item.get("arm"), item["workers"], item["firms"], seed, history)
        if key in result or item["evaluation_seed"] != 100 * seed + history:
            raise ValueError(f"Duplicate or inconsistent history at {meta}")
        checkpoint = item["checkpoint_sha256"]
        role = ("initial" if item["family"].startswith("untrained") else "trained", seed)
        if len(checkpoint) != 64 or checkpoints.setdefault(role, checkpoint) != checkpoint:
            raise ValueError(f"Checkpoint identity differs for {role} at {meta}")
        result[key] = meta.parent
    return result


def write_csv(path, records):
    rows = iter(records)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"No observations for {path.name}")
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def interval(values):
    from scipy.stats import t

    n = len(values)
    center = stats.fmean(values)
    hw = float(t.ppf(0.975, n - 1)) * stats.stdev(values) / math.sqrt(n) if n > 1 else None
    return dict(
        mean=center,
        seeds=n,
        lower95=center - hw if hw is not None else None,
        upper95=center + hw if hw is not None else None,
    )


def fit(histories, output):
    data = defaultdict(dict)
    for (family, arm, _, _, seed, h), folder in histories.items():
        if family == "acceptance" and arm == "baseline":
            if h in data[seed]:
                raise ValueError("Multiple baseline scales")
            data[seed][h] = folder
    if set(data) != set(tariff.SEEDS) or any((set(d) != set(range(50)) for d in data.values())):
        raise ValueError("Fit requires five policies and 50 baseline histories per policy")
    for seed, directories in sorted(data.items()):
        retained, probabilities, rate_means = ([], [], [])
        for h, folder in sorted(directories.items()):
            contexts = list(lines(folder / "offer_context_records.jsonl.gz"))
            verification = acceptance.offer_rates(
                contexts,
                lines(folder / "realized_accept_log.jsonl.gz"),
                read(folder / "ACCEPTANCE_RECEIPT.json"),
                seed,
                seed * 100 + h,
                "baseline",
            )
            rate_means.append(verification["TH_mean_live_probability"])
            for r in contexts:
                if r["month_index"] < 132:
                    if abs(acceptance.sigmoid((r["a_i"] + r["r_i"]) / 1.2) - r["p_i_B"]) > 1e-06:
                        raise ValueError("Baseline offer probability differs from the logit")
                    retained.append(r["r_i"])
                    probabilities.append(r["p_i_B"])
        c, error, mean_probability = acceptance.solve(retained, probabilities)
        record = dict(
            schema="current_policy_acceptance_recentre_fit_v1",
            status="FROZEN_AFTER_50_BASELINE_HISTORIES",
            policy_seed=seed,
            baseline_seed_block=list(range(100 * seed, 100 * seed + 50)),
            source_arm="baseline",
            baseline_histories=50,
            tau=1.2,
            c_s=c,
            residual=error,
            mean_probability=mean_probability,
            balance_T_s=0.5 * stats.stdev(rate_means),
        )
        write_json(output / f"policy_{seed}.json", record)


def score_acceptance(histories, output, args):
    empirical = acceptance.targets(args.targets)
    rows = []
    for (family, arm, _, _, seed, h), folder in histories.items():
        if family != "acceptance":
            continue
        evaluation_seed = 100 * seed + h
        rows.append(
            dict(
                policy_seed=seed,
                evaluation_seed=evaluation_seed,
                arm=arm,
                **acceptance.match_means(
                    lines(folder / "FINALIZED_MATCHES.jsonl.gz"), seed, evaluation_seed, family
                ),
                **acceptance.offer_rates(
                    lines(folder / "offer_context_records.jsonl.gz"),
                    lines(folder / "realized_accept_log.jsonl.gz"),
                    read(folder / "ACCEPTANCE_RECEIPT.json"),
                    seed,
                    evaluation_seed,
                    arm,
                ),
                **acceptance.macro_mae(read(folder / "ACCEPTANCE_TH_MONTHLY.json"), empirical),
            )
        )
    result = acceptance.infer(
        rows, {s: read(args.fit_directory / f"policy_{s}.json") for s in tariff.SEEDS}
    )
    result["means_by_arm"] = {}
    for arm in ("baseline", "recentred"):
        own = [r for r in rows if r["arm"] == arm]
        result["means_by_arm"][arm] = {
            k: stats.fmean((r[k] for r in own))
            for k in (
                *acceptance.ENDPOINTS,
                "TH_realized_acceptance_rate",
                "TH_finalized_match_count",
            )
        }
        result["means_by_arm"][arm]["macro_component_MAEs"] = {
            k: stats.fmean((r["macro_MAE_components"][k] for r in own)) for k in acceptance.RATES
        }
    write_json(output / "acceptance.json", result)


def write_json(path, value):

    def numeric_bounds(item):
        if isinstance(item, dict):
            return {k: numeric_bounds(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [numeric_bounds(v) for v in item]
        return None if isinstance(item, float) and math.isinf(item) else item

    path.write_text(json.dumps(numeric_bounds(value), allow_nan=False, indent=2), encoding="utf8")


def historical(folder, record):
    log = read(folder / "simulation_log.json")
    data = read(folder / "monthly.json")["months"]
    if [r["calendar"] for r in data] != [[2006 + i // 12, i % 12 + 1] for i in range(132)]:
        raise ValueError("Historical scoring needs the complete 132-month calendar")
    for row in data:
        if row["age_resident_raw_and_aligned"]["under15"]["P"] != 0:
            raise ValueError("Historical headline rates require an adult resident population")
    funnel = log["matching_funnel"]
    all_flows = [r["endpoint_flows_by_opening_age"]["all"] for r in data]
    seekers = sum(
        (sum(cell["n_unemployed_seekers"]) for cell in funnel["duration_binned"].values())
    )
    offered, matched = (sum(funnel[k]) for k in ("n_unemployed_offered", "n_unemployed_matched"))
    results = {"prime_age_job_finding_pct": prime_age(data)}
    annual = []
    for y in range(2006, 2017):
        cells = [r["age_resident_raw_and_aligned"]["all"] for r in data if r["calendar"][0] == y]
        p, e, u = (sum((c[k] for c in cells)) for k in ("P", "E", "U_raw"))
        if not p or e + u == 0:
            raise ValueError("Undefined annual resident rate")
        annual.append(
            dict(
                **record,
                year=y,
                employment_rate=e / p,
                participation_rate=(e + u) / p,
                unemployment_rate=u / (e + u),
            )
        )
    for band in ("all", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"):
        cells = [r["age_resident_raw_and_aligned"][band] for r in data]
        for name, n, d in (
            ("employment", "E", "P"),
            ("participation", "LF_raw", "P"),
            ("unemployment", "U_raw", "LF_raw"),
            ("search_unemployment", "U_active", "LF_aligned"),
            ("search_participation", "LF_aligned", "P"),
        ):
            monthly = [c[n] / c[d] for c in cells if c[d]]
            if len(monthly) == len(cells):
                results[name + "_" + band + "_pct"] = 100 * stats.fmean(monthly)
    totals = dict(
        searchers=seekers,
        offered=offered,
        precommit_matched=matched,
        opening_unemployed=sum((r["opening_U_raw"] for r in all_flows)),
        endpoint_hires=sum((r["endpoint_U_to_E"] for r in all_flows)),
    )
    keys = [
        "cohort_macros_by_age_native",
        "cohort_macros_by_age",
        "cohort_macros",
        "unemployment_rate",
        "years",
    ]
    return (record | results, totals, {k: log[k] for k in keys if k in log}, annual)


def score_historical(histories, output):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gtha_labor_marl"))
    from windowed_metrics import METRIC_EXTRACTORS
    from eval_aggregation import emit_dual_window_verdict

    samples, scored, counts = (defaultdict(dict), defaultdict(dict), defaultdict(list))
    yearly = []
    for (family, _, size, firms, seed, h), folder in sorted(
        histories.items(), key=lambda x: str(x[0])
    ):
        if family not in (
            "historical",
            "firm_fixed",
            "untrained_heuristic",
            "untrained_random",
            "bundle_off",
            "spatial",
        ):
            continue
        group = (family, size, firms, seed)
        row, funnel, log, annual = historical(
            folder, dict(configuration=family, workers=size, firms=firms, seed=seed, history=h)
        )
        yearly.extend(annual)
        samples[group][h] = row
        scored[group][h] = log
        counts[group].append(funnel)
    if not samples:
        raise ValueError("No historical runs were found")
    overview, criteria, policy_means = ([], [], {})
    ids = {"configuration", "workers", "firms", "seed", "history"}
    for group, block in sorted(samples.items()):
        if set(block) != set(range(50)):
            raise ValueError(f"Incomplete history indices for {group}")
        family, size, firms, seed = group
        common = set.intersection(*(set(r) for r in block.values())) - ids
        offsets = {k: stats.fmean((r[k] for r in block.values())) for k in common}
        totals = {k: sum((item[k] for item in counts[group])) for k in counts[group][0]}
        for name, n, d in (
            ("offer_reach_pct", "offered", "searchers"),
            ("precommit_match_per_recipient_pct", "precommit_matched", "offered"),
            ("opening_unemployed_to_employed_pct", "endpoint_hires", "opening_unemployed"),
        ):
            if totals[d]:
                offsets[name] = 100 * totals[n] / totals[d]
        policy_means[group] = offsets
        for metric, value in sorted(offsets.items()):
            overview.append(
                dict(
                    configuration=family,
                    workers=size,
                    firms=firms,
                    seed=seed,
                    metric=metric,
                    value=value,
                )
            )
        for metric, extractor, (lo, hi) in METRIC_EXTRACTORS:
            series = [extractor(scored[group][h]) for h in range(50)]
            if any(
                (
                    len(seq) != 11 or not all((x is not None and math.isfinite(x) for x in seq))
                    for seq in series
                )
            ):
                raise ValueError(f"Missing historical endpoint {group}, {metric}")
            criteria.append(
                dict(
                    configuration=family,
                    workers=size,
                    firms=firms,
                    seed=seed,
                    metric=metric,
                    verdict=emit_dual_window_verdict(series, lo, hi),
                )
            )
    write_json(output / "criteria.json", criteria)
    write_csv(output / "historical_annual.csv", yearly)
    write_csv(output / "historical_policy.csv", overview)
    contrasts = []
    for family, size, firms in sorted({key[:3] for key in policy_means}):
        if family == "historical" and size == 1000:
            continue
        seeds = sorted(
            (
                s
                for f, n, m, s in policy_means
                if (f, n, m) == (family, size, firms)
                and ("historical", 1000, 80, s) in policy_means
            )
        )
        if not seeds:
            continue
        cells = [
            (policy_means["historical", 1000, 80, s], policy_means[family, size, firms, s])
            for s in seeds
        ]
        shared = set.intersection(*(set(x) for pair in cells for x in pair))
        for metric in sorted(shared):
            delta = [trained[metric] - alternative[metric] for trained, alternative in cells]
            contrasts.append(
                dict(
                    reference="historical_1000",
                    alternative=family,
                    alternative_workers=size,
                    metric=metric,
                    effect="reference_minus_alternative",
                    **interval(delta),
                )
            )
    if contrasts:
        write_csv(output / "historical_contrasts.csv", contrasts)


def main():
    parser = argparse.ArgumentParser(description="Score histories produced by evaluation.run.")
    parser.add_argument(
        "analysis",
        choices=(
            "historical",
            "acceptance-fit",
            "acceptance",
            "forward",
            "spatial",
            "tariff",
            "entrants",
        ),
    )
    parser.add_argument("histories", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--targets", type=Path, help="Observed monthly rates in fractions")
    parser.add_argument("--fit-directory", type=Path)
    parser.add_argument(
        "--december-lfs",
        type=Path,
        help="Labour Force Survey public-use file for December 2016, used by the forecast benchmark",
    )
    parser.add_argument("--exposures", type=Path, help="Exposure CSV used for the tariff runs")
    args = parser.parse_args()
    histories = inventory(args.histories)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.analysis == "acceptance-fit":
        fit(histories, args.output)
    elif args.analysis == "acceptance":
        if not args.targets or not args.fit_directory:
            parser.error("Acceptance scoring requires --targets and --fit-directory")
        score_acceptance(histories, args.output, args)
    elif args.analysis == "historical":
        score_historical(histories, args.output)
    elif args.analysis == "tariff":
        if args.exposures is None or not args.exposures.is_file():
            parser.error("Tariff scoring requires --exposures")
        from .tariff_panel import export

        export(histories, args.output / "scores.zip", args.exposures)
    else:
        from .outcomes import score_forward, score_spatial, score_entrants

        if args.analysis == "forward":
            score_forward(histories, args.output, args.targets, args.december_lfs)
        elif args.analysis == "spatial":
            score_spatial(histories, args.output)
        else:
            score_entrants(histories, args.output)


if __name__ == "__main__":
    main()
