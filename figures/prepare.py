import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from prepare_controls import METRICS, load_histories, save_controls, summary, write

SEEDS = tuple(range(42, 47))


def check_grid(frame, keys, expected):
    actual = pd.MultiIndex.from_frame(frame[keys])
    grid = pd.MultiIndex.from_product(expected, names=keys)
    if actual.has_duplicates or not actual.sort_values().equals(grid.sort_values()):
        raise ValueError(f"Incomplete or duplicated cells for {keys}")


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def publish_forward(data, observed, output):
    policy = data.rename(columns={"seed": "policy_seed", "level": "value"}).copy()
    metrics = [*METRICS, "temp_layoff_share_population", "marginal_share_population"]
    dates = ["year", "month"]
    check_grid(
        policy,
        ["arm", "policy_seed", *dates, "metric"],
        [["RZ_ON", "RZ_OFF", "FN_ON", "FN_OFF"], SEEDS, range(2017, 2024), range(1, 13), metrics],
    )
    check_grid(observed, dates, [range(2017, 2024), range(1, 13)])
    survey_rates = observed.loc[:, list(METRICS)]
    survey_stocks = observed.loc[observed.year.between(2017, 2021), metrics[3:]]
    if (
        not all((np.isfinite(v).all().all() for v in (survey_rates, survey_stocks)))
        or not np.isfinite(policy.value).all()
    ):
        raise ValueError("The forward comparison requires complete monthly rates.")
    policy["case"] = "forward_" + policy.arm
    keys = ["case", *dates, "metric"]
    wide = policy.pivot(
        index=["case", *dates, "policy_seed"], columns="metric", values="value"
    ).reset_index()
    write(wide, output / "core", "forward_seeds")
    base_keys = ["case", "policy_seed", "month", "metric"]
    base = policy.loc[policy.year.eq(2019), base_keys + ["value"]].rename(columns={"value": "base"})
    centered = policy.merge(base, on=base_keys, validate="many_to_one")
    centered["value"] -= centered.base
    levels = summary(policy, keys, seed_column="n_seeds")
    changes = summary(centered, keys, seed_column="n_seeds")
    write(levels, output / "core", "forward")
    write(changes, output / "core", "changes")
    pairs = []
    for mode in ("RZ", "FN"):
        arms = policy[policy.arm.str.startswith(mode)].pivot(
            index=["policy_seed", *dates, "metric"], columns="arm", values="value"
        )
        pairs.append(
            (arms[mode + "_ON"] - arms[mode + "_OFF"])
            .rename("value")
            .reset_index()
            .assign(case=mode + "_ON_minus_OFF")
        )
    write(summary(pd.concat(pairs), keys, seed_column="n_seeds"), output / "core", "paired")
    for name in ("core", "spatial"):
        write(observed[dates + metrics], output / name, "observed")
    model = pd.concat(
        [levels.assign(view="level"), changes.assign(view="same_calendar_2019_change")]
    )
    write(
        model[model.case.str.startswith("forward_RZ") & model.metric.isin(METRICS)],
        output / "spatial",
        "pandemic",
    )


def forward_mechanisms(histories, output):
    totals, identities = ({}, [])
    seen = set()
    transitions = {
        f"flow_{a}_{b}": (f"endpoint_{a}_to_{b}", "opening_" + a + ("_raw" if a != "E" else ""))
        for a in ("E", "U", "N")
        for b in ("E", "U", "N")
        if a != b
    }
    for path in histories.rglob("history.json"):
        meta = json.loads(path.read_text(encoding="utf8"))
        arm, s, h = (meta["arm"], meta["policy_seed"], meta["replication_index"])
        if (
            meta["family"] != "forward"
            or arm not in ("RZ_ON", "RZ_OFF")
            or (meta["workers"], meta["firms"]) != (1000, 80)
        ):
            continue
        key = (arm, s, h)
        if key in seen or meta["evaluation_seed"] != 100 * s + h:
            raise ValueError(f"Invalid forward identity in {path}")
        seen.add(key)
        identities.append(
            dict(
                case="forward_" + arm,
                policy_seed=s,
                replication_index=h,
                evaluation_seed=100 * s + h,
                checkpoint_sha256=meta["checkpoint_sha256"],
            )
        )
        months = json.loads((path.parent / "monthly.json").read_text(encoding="utf8"))["months"]
        if [r["calendar"] for r in months] != [[2006 + t // 12, 1 + t % 12] for t in range(216)]:
            raise ValueError(f"Incomplete forward calendar in {path}")
        for row in months:
            year, month = row["calendar"]
            flow = row["endpoint_flows_by_opening_age"]["all"]
            stock = row["age_resident_raw_and_aligned"]["all"]
            values = {metric: (flow[n], flow[d]) for metric, (n, d) in transitions.items()}
            values["elapsed_U_stock_months"] = (
                stock["unemployment_duration_months_sum"],
                stock["unemployment_duration_raw_U_n"],
            )
            values["elapsed_active_U_stock_months"] = (
                stock["unemployment_duration_active_months_sum"],
                stock["unemployment_duration_active_U_n"],
            )
            for metric, (num, den) in values.items():
                k = ("forward_" + arm, s, year, month, metric)
                old_num, old_den = totals.get(k, (0, 0))
                totals[k] = (old_num + num, old_den + den)
    if seen != {(arm, s, h) for arm in ("RZ_ON", "RZ_OFF") for s in SEEDS for h in range(50)}:
        raise ValueError(
            "Forward mechanism figures require 250 histories in each realised-driver arm."
        )
    registry = pd.DataFrame(identities)
    if (
        registry.groupby(["policy_seed", "replication_index"])
        .checkpoint_sha256.nunique()
        .ne(1)
        .any()
    ):
        raise ValueError("Paired forward arms use different checkpoints.")
    rows = []
    for (case, s, y, m, metric), (num, den) in sorted(totals.items()):
        if not den:
            raise ValueError(f"Undefined {metric} for {case}, seed {s}, {y}-{m}")
        value = num / den * (100 if metric in transitions else 1)
        rows.append(
            dict(
                dataset="model",
                case=case,
                policy_seed=s,
                year=y,
                month=m,
                metric=metric,
                numerator_sum=num,
                denominator_sum=den,
                value=value,
            )
        )
    policies = pd.DataFrame(rows)
    write(policies, output / "core", "mechanism_policy")
    write(
        summary(policies, ["case", "year", "month", "metric"], seed_column="n_seeds").assign(
            dataset="model"
        ),
        output / "core",
        "mechanism",
    )
    json_file(output / "evaluation/paired_histories.json", identities)


def publish_spatial(policies, output):
    check_grid(policies, ["scale", "seed"], [[1000, 2500], SEEDS])
    commute, access = ({}, {})
    detail, wages = ([], [])
    for size, cells in policies.groupby("scale"):
        keys = ["stock_time_all_valid_mean", "mean_closing_vacancies"]
        commute[str(size)] = {
            "means_and_policy_ranges": {
                k: dict(
                    mean=float(cells[k].mean()),
                    min_seed_mean=float(cells[k].min()),
                    max_seed_mean=float(cells[k].max()),
                )
                for k in keys
            }
        }
        series = cells["fraction_with_access"]
        access[str(size)] = {
            "fraction_with_access": [float(series.mean()), float(series.min()), float(series.max())]
        }
        for metric in ("p10", "median", "p90", "p90_p10", "gini"):
            values = cells["wage_" + metric]
            avg = values.mean()
            hw = 2.7764451051977987 * values.std() / np.sqrt(5)
            wages.append(
                dict(
                    scale=size,
                    group="worker",
                    metric=metric,
                    mean=avg,
                    lo95=avg - hw,
                    hi95=avg + hw,
                )
            )
    for s in [*sorted(SEEDS), "mean"]:
        row = {"seed": s}
        for size in (1000, 2500):
            cells = policies[policies.scale.eq(size)]
            if s != "mean":
                cells = cells[cells.seed.eq(s)]
            row["commute_" + str(size)] = cells.stock_time_all_valid_mean.mean()
            row["access_" + str(size)] = 100 * cells.fraction_with_access.mean()
        detail.append(row)
    json_file(
        output / "spatial/spatial_2016_summary.json",
        dict(
            scales=commute,
            per_policy=policies[["scale", "seed", "mean_closing_vacancies"]].to_dict("records"),
        ),
    )
    json_file(output / "spatial/vacancy_access_2016_summary.json", dict(scales=access))
    write(pd.DataFrame(detail), output / "diagnostics", "detailed_spatial")
    write(pd.DataFrame(wages), output / "wages", "wages_summary")


def publish_criteria(records, output):
    rows = []
    for record in records:
        if record["configuration"] != "historical":
            continue
        size = record["workers"]
        s = record["verdict"]
        row = dict(
            case=f"historical{size}",
            policy_seed=record["seed"],
            metric=record["metric"],
            final_classification=s["final_classification"],
        )
        for k in ("full_11", "last_3"):
            row[k + "_mean_pct"] = s[k]["mean"]
            row[k + "_ci_half_pp"] = s[k]["ci_half"]
            row[k + "_verdict"] = s["verdict_" + k]
        rows.append(row)
    scores = pd.DataFrame(rows)
    keys = ["case", "policy_seed", "metric"]
    metrics = [
        "U_LF_native_pct",
        "NILF_35_44_native_pct",
        "NILF_55_64_native_pct",
        "NILF_65_99_native_pct",
    ]
    check_grid(scores, keys, [scores.case.unique(), SEEDS, metrics])
    small = scores[scores.case.eq("historical1000")]
    if small.empty:
        raise ValueError("The primary historical criteria are required.")
    write(scores, output / "core", "registered_criteria")
    write(small, output / "diagnostics", "registered_criteria")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare publication figure tables from evaluation histories and scores."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("controls", "forward", "spatial", "criteria", "acceptance", "historical"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=True, help="Figure input directory")
        if name in ("controls", "forward", "historical"):
            sub.add_argument("--histories", type=Path, required=name != "forward")
        if name in ("forward", "spatial", "criteria", "acceptance"):
            sub.add_argument("--scores", type=Path, required=True)
        if name in ("controls", "forward"):
            sub.add_argument("--reference", type=Path, required=True)
        if name == "historical":
            sub.add_argument("--age-reference", type=Path)
    args = parser.parse_args()
    if args.command == "controls":
        annual, monthly = load_histories(args.histories)
        save_controls(annual, monthly, pd.read_csv(args.reference), args.output)
    elif args.command == "forward":
        publish_forward(
            pd.read_csv(args.scores / "forward_policy.csv"),
            pd.read_csv(args.reference),
            args.output,
        )
        if args.histories:
            forward_mechanisms(args.histories, args.output)
    elif args.command == "spatial":
        publish_spatial(pd.read_csv(args.scores / "spatial_policy.csv"), args.output)
    elif args.command == "criteria":
        publish_criteria(
            json.loads((args.scores / "criteria.json").read_text(encoding="utf8")), args.output
        )
    elif args.command == "acceptance":
        source = args.scores / "acceptance.json"
        result = json.loads(source.read_text(encoding="utf8"))
        if not {"per_policy_paired_df49", "across_five_policies_df4", "means_by_arm"}.issubset(
            result
        ):
            raise ValueError("Incomplete acceptance scores.")
        target = args.output / "spatial/registered_acceptance_inference.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    else:
        from prepare_historical import historical

        historical(args.histories, args.output, args.age_reference)


if __name__ == "__main__":
    main()
