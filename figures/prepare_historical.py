import json
import numpy as np
import pandas as pd
from prepare_controls import METRICS, annual_cohort, summary, write

AGE_GROUPS = {
    "15-24": ("15-24",),
    "25-54": ("25-34", "35-44", "45-54"),
    "55-64": ("55-64",),
    "65+": ("65-99",),
}
AGE_METRICS = {"EPOP": "A2", "LFPR": "A3", "ULF": "A1"}


def native_age_rates(log):
    annual = log["cohort_macros_by_age_native"]["annual"]
    for group, bands in AGE_GROUPS.items():
        pop, labour, u = np.zeros((3, 11))
        for band in bands:
            block = annual[band]
            n = np.asarray(block["pop"])
            lf = n * block["lfpr_aligned"]
            pop += n
            labour += lf
            u += lf * block["ulf_aligned"]
        if np.any(pop <= 0) or (group != "65+" and np.any(labour <= 0)):
            raise ValueError(f"Undefined historical rate for {group}")
        for code, values in (("A2", (labour - u) / pop), ("A3", labour / pop)):
            yield dict(group=group, code=code, value=100 * values.mean())
        if group != "65+":
            yield dict(group=group, code="A1", value=100 * (u / labour).mean())


def resident_annual(months):
    if [r["calendar"] for r in months] != [[2006 + t // 12, 1 + t % 12] for t in range(132)]:
        raise ValueError("A complete 2006–2016 historical calendar is required.")
    data = pd.DataFrame([row["age_resident_raw_and_aligned"]["all"] for row in months])
    counts = data.groupby(2006 + data.index // 12)[["P", "E", "U_raw"]].sum()
    labour = counts.E + counts.U_raw
    return pd.DataFrame(
        dict(
            year=counts.index,
            employment_rate=counts.E / counts.P,
            participation_rate=labour / counts.P,
            unemployment_rate=counts.U_raw / labour,
        )
    )


def save_age_scores(rows, reference, output):
    targets = reference.loc[
        reference.metric.isin(AGE_METRICS) & reference.group.isin(AGE_GROUPS)
    ].copy()
    targets["year"] = pd.to_numeric(targets.year)
    targets = targets[targets.year.between(2006, 2016)]
    targets = targets[~(targets.metric.eq("ULF") & targets.group.eq("65+"))]
    if targets.duplicated(["metric", "group", "year"]).any() or len(targets) != 121:
        raise ValueError(
            "Age references must cover all 11 years and the 11 defined rate/age combinations."
        )
    targets["code"] = targets.metric.map(AGE_METRICS)
    anchors = targets.groupby(["group", "code"]).value_pct.mean().rename("anchor_full")
    means = rows.groupby(["scale", "seed", "code", "group"]).value.mean().reset_index()
    s = summary(means, ["scale", "code", "group"], seed_column="n_seeds").merge(
        anchors, on=["group", "code"]
    )
    s["five_seed_ci95_half"] = s.upper95 - s["mean"]
    s["five_seed_bias"] = s["mean"] - s.anchor_full
    s = s.rename(columns={"mean": "five_seed_mean"}).assign(population="native")
    write(s, output / "core", "age_stocks")


def historical(histories, output, age_reference):
    cohorts, residents, ages, durations, bins = ([], [], [], [], [])
    seen = set()
    for path in sorted(histories.rglob("history.json")):
        info = json.loads(path.read_text(encoding="utf8"))
        n, s, h = (info["workers"], info["policy_seed"], info["replication_index"])
        if info["family"] != "historical" or (n, info["firms"]) not in ((1000, 80), (2500, 200)):
            continue
        if (n, s, h) in seen or info["evaluation_seed"] != 100 * s + h:
            raise ValueError(f"Invalid historical identity at {path}")
        seen.add((n, s, h))
        case = f"historical{n}"
        log = json.loads((path.parent / "simulation_log.json").read_text(encoding="utf8"))
        months = json.loads((path.parent / "monthly.json").read_text(encoding="utf8"))["months"]
        cohorts.append(annual_cohort(log).assign(case=case, policy_seed=s))
        residents.append(resident_annual(months).assign(case=case, policy_seed=s))
        if age_reference:
            ages.extend((dict(scale=n, seed=s, **r) for r in native_age_rates(log)))
        if n == 1000:
            for age, val in log["unemployment_duration_by_age_active_search"].items():
                durations.append(
                    dict(policy_seed=s, age_group="65+" if age == "65-99" else age, value=val)
                )
            for band, counters in log["matching_funnel"]["duration_binned"].items():
                bins.append(
                    dict(
                        policy_seed=s,
                        duration_bin=band,
                        matches=sum(counters["n_workers_matched"]),
                        openings=sum(counters["n_unemployed_seekers"]),
                    )
                )
    if seen != {(n, s, h) for n in (1000, 2500) for s in range(42, 47) for h in range(50)}:
        raise ValueError("Scale figures require 250 historical histories at each market size.")
    keys = ["case", "policy_seed", "year"]
    for tables, folder, name in (
        (cohorts, "core", "historical_seeds"),
        (residents, "evaluation", "HISTORICAL_RESIDENT_SEED_CURVES"),
    ):
        seed = pd.concat(tables).groupby(keys)[list(METRICS)].mean().reset_index()
        write(seed, output / folder, name)
        if name == "historical_seeds":
            curves = summary(
                seed.melt(id_vars=keys, var_name="metric"),
                ["case", "year", "metric"],
                seed_column="n_seeds",
            )
            write(curves, output / "core", "historical")
    age = pd.DataFrame(durations).groupby(["policy_seed", "age_group"]).value.agg(["mean", "count"])
    age["value"] = age["mean"].where(age["count"].eq(50))
    write(age[["value"]].reset_index().assign(dataset="model"), output / "core", "age_duration")
    binned = (
        pd.DataFrame(bins).groupby(["policy_seed", "duration_bin"])[["matches", "openings"]].sum()
    )
    binned["value"] = 100 * binned.matches / binned.openings.where(binned.openings.gt(0))
    write(binned[["value"]].reset_index().assign(dataset="model"), output / "core", "duration_bins")
    if age_reference:
        save_age_scores(pd.DataFrame(ages), pd.read_csv(age_reference), output)
