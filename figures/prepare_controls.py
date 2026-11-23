from collections import defaultdict
import json
import numpy as np
import pandas as pd
from learning import METRICS, T4

CASES = {
    "historical": "trained",
    "untrained_random": "untrained_random",
    "untrained_heuristic": "untrained_heuristic",
    "firm_fixed": "firm_frozen",
    "bundle_off": "bundle_off",
}
RATIOS = {
    "offer_reach_pct": ("offered", "pool", 100),
    "precommit_conversion_pct": ("matched", "offered", 100),
    "flow_U_E_pct": ("u_e", "open_u", 100),
    "flow_U_N_pct": ("u_n", "open_u", 100),
    "duration_raw_months": ("elapsed_u", "measured_u", 1),
    "funded_capacity_occupancy_pct": ("claims", "funded", 100),
    "raw_E_to_U_pct": ("e_u", "open_e", 100),
    "raw_E_to_N_pct": ("e_n", "open_e", 100),
    "positive_hourly_wage": ("wage_total", "wage_count", 1),
}
LEVELS = {
    "U_matching_pool": "pool",
    "U_committed_hires": "committed",
    "RL_postings": "postings",
    "funded_headroom": "headroom",
    "N_to_E_per1000": "n_e",
    "funded_active": "funded",
    "effective_claims": "claims",
    "resident_filled": "filled",
}


def write(table, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / (name + ".csv"), index=False, float_format="%.15g")


def summary(frame, by, value="value", seed_column="n_policies"):
    s = (
        frame.groupby(by)[value]
        .agg(["mean", "std", "count"])
        .rename(columns={"std": "sd", "count": seed_column})
        .reset_index()
    )
    half = (T4 * s.sd / np.sqrt(5)).where(s[seed_column].eq(5))
    return s.assign(lower95=s["mean"] - half, upper95=s["mean"] + half)


def annual_cohort(log):
    cohorts = log["cohort_macros_by_age"]["annual"]
    pop, labour, unemployed = np.zeros((3, 11))
    if log["years"] != list(range(2006, 2017)):
        raise ValueError("Historical scoring requires the 2006–2016 annual series.")
    for group in cohorts.values():
        p = np.asarray(group["pop"], dtype=float)
        lf = p * np.asarray(group["lfpr_aligned"])
        pop += p
        labour += lf
        unemployed += lf * group["ulf_aligned"]
    if np.any(pop <= 0) or np.any(labour <= 0) or np.any(unemployed > labour + 1e-08):
        raise ValueError("Undefined annual cohort rate.")
    return pd.DataFrame(
        dict(
            year=log["years"],
            employment_rate=(labour - unemployed) / pop,
            participation_rate=labour / pop,
            unemployment_rate=unemployed / labour,
        )
    )


def month_counts(log, months):
    funnel = log["matching_funnel"]
    for index, row in enumerate(months):
        if row["calendar"] != [2006 + index // 12, 1 + index % 12]:
            raise ValueError("Non-consecutive historical months.")
        status = row["age_resident_raw_and_aligned"]["all"]
        flow = row["endpoint_flows_by_opening_age"]["all"]
        capacity = row["market_capacity"]
        wages = [industry["positive_hourly_wage"] for industry in row["industry_worker_wages"]]
        funded = capacity["funded_active_capacity"]
        claims = capacity["effective_filled_claims"]
        yield dict(
            pool=sum(
                (c["n_unemployed_seekers"][index] for c in funnel["duration_binned"].values())
            ),
            offered=funnel["n_unemployed_offered"][index],
            matched=funnel["n_unemployed_matched"][index],
            committed=funnel["committed_hires_u2e"][index],
            postings=funnel["vacancies_posted_rl"][index],
            funded=funded,
            claims=claims,
            filled=capacity["resident_filled_total"],
            headroom=funded - claims,
            elapsed_u=status["unemployment_duration_months_sum"],
            measured_u=status["unemployment_duration_raw_U_n"],
            e_u=flow["endpoint_E_to_U"],
            e_n=flow["endpoint_E_to_N"],
            u_e=flow["endpoint_U_to_E"],
            u_n=flow["endpoint_U_to_N"],
            n_e=flow["endpoint_N_to_E"],
            open_u=flow["opening_U_raw"],
            open_e=flow["opening_E"],
            wage_total=sum((cell["sum"] for cell in wages)),
            wage_count=sum((cell["n"] for cell in wages)),
            count=1,
        )


def load_histories(histories):
    annual, monthly, seen = ([], {}, defaultdict(set))
    for meta in sorted(histories.rglob("history.json")):
        identity = json.loads(meta.read_text(encoding="utf-8"))
        case = CASES.get(identity["family"])
        if not case or (identity["workers"], identity["firms"]) != (1000, 80):
            continue
        seed, history = (identity["policy_seed"], identity["replication_index"])
        if identity["evaluation_seed"] != 100 * seed + history or history in seen[case, seed]:
            raise ValueError(f"Duplicate or inconsistent history identity at {meta}")
        seen[case, seed].add(history)
        folder = meta.parent
        log = json.loads((folder / "simulation_log.json").read_text(encoding="utf-8"))
        annual.append(annual_cohort(log).assign(case=case, policy_seed=seed, history=history))
        months = json.loads((folder / "monthly.json").read_text(encoding="utf-8"))["months"]
        if len(months) != 132:
            raise ValueError(f"Incomplete historical series at {meta}")
        cells = pd.DataFrame(month_counts(log, months))
        key = (case, seed)
        monthly[key] = monthly[key] + cells if key in monthly else cells
    expected = {
        (case, seed)
        for case in CASES.values()
        for seed in (range(42, 44) if case == "bundle_off" else range(42, 47))
    }
    if set(seen) != expected or any((ids != set(range(50)) for ids in seen.values())):
        raise ValueError(
            "Historical controls need 50 matched histories in each study policy and configuration."
        )
    blocks = []
    for (case, seed), cells in monthly.items():
        blocks.append(
            cells.assign(
                case=case,
                policy_seed=seed,
                year=2006 + cells.index // 12,
                month=1 + cells.index % 12,
            )
        )
    return (pd.concat(annual, ignore_index=True), pd.concat(blocks, ignore_index=True))


def mechanism_scores(monthly, keys):
    sums = monthly.groupby(keys).sum(numeric_only=True)
    records = []
    for metric, (n, d, factor) in RATIOS.items():
        if sums[d].le(0).any():
            raise ValueError(f"Undefined denominator for {metric}")
        records.append(
            pd.DataFrame(
                dict(
                    metric=metric,
                    numerator_sum=sums[n],
                    denominator_sum=sums[d],
                    value=factor * sums[n] / sums[d],
                )
            )
        )
    for metric, column in LEVELS.items():
        records.append(pd.DataFrame(dict(metric=metric, value=sums[column] / sums["count"])))
    return pd.concat(records).reset_index()


def pair(frame, keys, values):
    other = frame[frame.case.ne("trained")].rename(columns={"case": "comparison"})
    trained = frame[frame.case.eq("trained")][keys + values]
    pairs = other.merge(
        trained, on=keys, suffixes=("_counterfactual", "_trained"), validate="many_to_one"
    )
    if len(pairs) != len(other):
        raise ValueError("Missing matched trained comparator.")
    for field in values:
        pairs[field] = pairs[field + "_trained"] - pairs[field + "_counterfactual"]
    return pairs


def save_controls(annual, monthly, reference, output):
    ctrl = output / "controls"
    metrics = list(METRICS)
    annual_policy = annual.groupby(["case", "policy_seed", "year"])[metrics].mean().reset_index()
    write(annual_policy, ctrl, "MACRO_ANNUAL_POLICY_CURVES")
    write(
        annual_policy[annual_policy.case.isin(["trained", "firm_frozen", "bundle_off"])],
        output / "firm",
        "macro_annual_policy",
    )
    annual_long = annual_policy.melt(id_vars=["case", "policy_seed", "year"], var_name="metric")
    write(summary(annual_long, ["case", "year", "metric"]), ctrl, "MACRO_COMBINED_ANNUAL_CURVES")
    if reference.year.duplicated().any() or set(reference.year) != set(range(2006, 2017)):
        raise ValueError(
            "Supply one empirical reference observation for each year from 2006 through 2016."
        )
    for folder, name in ((ctrl, "HISTORICAL_REFERENCE"), (output / "core", "historical_reference")):
        write(reference[["year", *metrics]], folder, name)
    keys = ["case", "policy_seed", "history", "metric"]
    scores = annual.melt(
        id_vars=["case", "policy_seed", "history", "year"], value_vars=metrics, var_name="metric"
    )
    observed = reference.melt(
        id_vars="year", value_vars=metrics, var_name="metric", value_name="observed"
    )
    scores = scores.merge(observed, on=["year", "metric"], validate="many_to_one")
    scores["window_mean_pct"] = 100 * scores.value
    scores["annual_path_MAE_pp"] = 100 * (scores.value - scores.observed).abs()
    cols = ["window_mean_pct", "annual_path_MAE_pp"]
    histories = scores.groupby(keys)[cols].mean().reset_index()
    policies = histories.groupby(["case", "policy_seed", "metric"])[cols].mean().reset_index()
    write(policies, ctrl, "MACRO_POLICY_SCORES")
    paired = pair(histories, ["policy_seed", "history", "metric"], cols)
    paired = (
        paired.groupby(["comparison", "policy_seed", "metric"])[cols]
        .mean()
        .reset_index()
        .melt(id_vars=["comparison", "policy_seed", "metric"], var_name="estimand")
    )
    write(paired, ctrl, "MACRO_PAIRED_POLICY_CONTRASTS")
    write(
        summary(paired, ["comparison", "metric", "estimand"]), ctrl, "MACRO_PAIRED_CONTRAST_SUMMARY"
    )
    window = mechanism_scores(monthly, ["case", "policy_seed"])
    annual_mech = mechanism_scores(monthly, ["case", "policy_seed", "year"])
    write(window, ctrl, "MECHANISM_WINDOW_POLICY_ESTIMATES")
    write(
        window[window.case.isin(["trained", "firm_frozen", "bundle_off"])],
        output / "firm",
        "mechanism_policy",
    )
    by_year = summary(annual_mech, ["case", "year", "metric"])
    write(by_year, ctrl, "MECHANISM_ANNUAL_CURVES")
    flow_metrics = ["raw_E_to_U_pct", "raw_E_to_N_pct", "N_to_E_per1000", "positive_hourly_wage"]
    write(by_year[by_year.metric.isin(flow_metrics)], ctrl, "FLOW_WAGE_ANNUAL_CURVES")
    write(summary(window, ["case", "metric"]), ctrl, "MECHANISM_WINDOW_SUMMARY")
    diff = pair(
        window[["case", "policy_seed", "metric", "value"]], ["policy_seed", "metric"], ["value"]
    )
    write(diff, ctrl, "MECHANISM_PAIRED_POLICY_CONTRASTS")
    write(summary(diff, ["comparison", "metric"]), ctrl, "MECHANISM_PAIRED_CONTRAST_SUMMARY")
    trained = monthly[monthly.case.eq("trained")]
    months = mechanism_scores(trained, ["policy_seed", "year", "month"])
    names = {
        "flow_U_E_pct": "flow_U_E",
        "flow_U_N_pct": "flow_U_N",
        "funded_active": "funded_active",
        "effective_claims": "effective_claims",
        "resident_filled": "resident_filled",
    }
    months = months[months.metric.isin(names)].copy()
    months["metric"] = months.metric.map(names)
    write(
        summary(months, ["year", "month", "metric"], seed_column="n_seeds"),
        output / "diagnostics",
        "historical_capacity_and_exits",
    )
