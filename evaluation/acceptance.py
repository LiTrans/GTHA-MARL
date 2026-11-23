from pathlib import Path
from itertools import zip_longest
import csv
import math
import statistics

POLICIES = tuple(range(42, 47))
TH = (535, 537)
CMAS = (532, 535, 537)
ENDPOINTS = {
    "offered_wage_at_finalized_match": -1,
    "match_quality_at_finalized_match": -1,
    "commute_at_finalized_match": 1,
    "equalweight_macro_MAE": 1,
}
MATCH_FIELDS = {
    "offered_wage_at_finalized_match": "offered_wage",
    "match_quality_at_finalized_match": "match_quality",
    "commute_at_finalized_match": "commute_minutes",
}
RATES = {
    "unemployment_rate_lfs": "unemployment_rate",
    "participation_rate_lfs": "participation_rate",
    "employment_rate": "employment_share_population",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def targets(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 36, "Exactly 36 target months are required")
    result = {}
    for row in rows:
        key = (int(row["year"]), int(row["month"]))
        require(key not in result, "Duplicate empirical target month")
        result[key] = {native: float(row[empirical]) for native, empirical in RATES.items()}
        require(
            all((finite(value) and 0 <= value <= 1 for value in result[key].values())),
            "Invalid empirical target rate",
        )
    require(
        set(result) == {(y, m) for y in range(2017, 2020) for m in range(1, 13)},
        "The holdout target is incomplete",
    )
    return result


def match_means(rows, seed, evaluation_seed, family):
    values = {name: [] for name in MATCH_FIELDS}
    seen = set()
    for row in rows:
        mi = row["month_index"]
        require(
            type(mi) is int
            and mi in range(168)
            and (row["year"] == 2006 + mi // 12)
            and (row["calendar_month"] == mi % 12 + 1)
            and (row["policy_seed"] == seed)
            and (row["evaluation_seed"] == evaluation_seed)
            and (row["family"] == family)
            and (row["cma"] in CMAS),
            "Finalized-match identity/calendar/geography differs",
        )
        identity = (mi, row["worker_id"])
        require(
            identity not in seen
            and row["commit_source_ID_observed"] is True
            and (row["operational_accepted"] is True)
            and (row["successful_bargaining"] is True)
            and (row["source_route"] in ("matching_u2e_committed", "matching_j2j_committed")),
            "Duplicate or noncommitted match entered endpoint",
        )
        seen.add(identity)
        require(
            all((finite(row[name]) for name in MATCH_FIELDS.values()))
            and row["commute_minutes"] >= 0,
            "Undefined finalized-match endpoint",
        )
        if mi < 132 and row["cma"] in TH:
            for name, field in MATCH_FIELDS.items():
                values[name].append(row[field])
    count = len(values["offered_wage_at_finalized_match"])
    return {
        "TH_finalized_match_count": count,
        **{name: statistics.fmean(sample) if sample else None for name, sample in values.items()},
    }


def macro_mae(payload, empirical):
    require(
        payload["schema"] == "acceptance_native_TH_market_close_snapshots_v1"
        and payload["benchmark_cmas"] == list(TH)
        and (payload["native_get_covid_validation_snapshot"] is True)
        and (payload["separation_standard_driver_mode"] == "as_loaded")
        and (payload["COVID_regime_enabled"] is False)
        and (len(payload["months"]) == 168),
        "The scope of the complete 168-month snapshot differs",
    )
    errors = {name: [] for name in RATES}
    for mi, row in enumerate(payload["months"]):
        key = (row["year"], row["month"])
        require(key == (2006 + mi // 12, mi % 12 + 1), "The snapshot calendar is incomplete")
        if mi < 132:
            continue
        aggregate = row["aggregate"]
        e, u, n = (aggregate[name] for name in ("E", "U_lfs", "N_lfs"))
        require(
            all((type(value) is int and value >= 0 for value in (e, u, n)))
            and e + u > 0
            and (e + u + n > 0),
            "Undefined holdout denominator; no partial-window MAE allowed",
        )
        rates = {
            "unemployment_rate_lfs": u / (e + u),
            "participation_rate_lfs": (e + u) / (e + u + n),
            "employment_rate": e / (e + u + n),
        }
        require(
            aggregate["population"] == e + u + n
            and all(
                (
                    finite(aggregate[name]) and abs(aggregate[name] - value) <= 1e-12
                    for name, value in rates.items()
                )
            ),
            "Reported stock rates disagree with the source counts",
        )
        for name, value in rates.items():
            errors[name].append(abs(value - empirical[key][name]))
    require(
        all((len(values) == 36 for values in errors.values())), "All36 holdout months mandatory"
    )
    components = {name: statistics.fmean(values) for name, values in errors.items()}
    return {
        "equalweight_macro_MAE": statistics.fmean(components.values()),
        "macro_MAE_components": components,
    }


def offer_rates(contexts, decisions, receipt, seed, evaluation_seed, arm):
    all_probabilities, th_probabilities = ([], [])
    offered = accepted = th_offered = th_accepted = 0
    month_counts = [0] * 168
    for index, pair in enumerate(zip_longest(contexts, decisions)):
        require(None not in pair, "Actual offer/context stream length mismatch")
        context, decision = pair
        keys = (
            "policy_seed",
            "evaluation_seed",
            "offer_id",
            "year",
            "calendar_month",
            "month_index",
            "cma",
            "p_i_B",
            "arm",
        )
        require(
            all((context[key] == decision[key] for key in keys))
            and (
                context["policy_seed"],
                context["evaluation_seed"],
                context["offer_id"],
                context["arm"],
            )
            == (seed, evaluation_seed, index, arm)
            and (type(decision["accepted"]) is bool)
            and (context["cma"] in CMAS),
            "Actual offer/decision identity differs",
        )
        mi = context["month_index"]
        require(
            type(mi) is int
            and mi in range(168)
            and ((context["year"], context["calendar_month"]) == (2006 + mi // 12, mi % 12 + 1)),
            "Actual offer calendar differs",
        )
        probability = context["p_i_B"]
        require(
            finite(probability)
            and 0 <= probability <= 1
            and (context["tau"] == 1.2)
            and finite(context["job_security"])
            and (context["job_security"] == 0.5),
            "The offer probabilities do not reproduce from temperature and job security",
        )
        month_counts[mi] += 1
        if mi < 132:
            all_probabilities.append(probability)
            offered += 1
            accepted += decision["accepted"]
            if context["cma"] in TH:
                th_probabilities.append(probability)
                th_offered += 1
                th_accepted += decision["accepted"]
    require(
        sum(month_counts) == receipt["offer_count"] == receipt["bernoulli_offer_count"]
        and len(receipt["monthly_reconciliations"]) == 168,
        "Actual offer receipt count/horizon mismatch",
    )
    for mi, row in enumerate(receipt["monthly_reconciliations"]):
        require(
            row["month_index"] == mi
            and row["recorded_offers"] == row["native_offers"] == month_counts[mi]
            and (row["reconciliation_passed"] is True),
            "Actual offer monthly funnel mismatch",
        )
    require(th_offered > 0 and offered > 0, "Undefined TH baseline/treatment acceptance rate")
    return {
        "TH_mean_live_probability": statistics.fmean(th_probabilities),
        "TH_realized_acceptance_rate": th_accepted / th_offered,
        "TH_actual_offers": th_offered,
        "TH_actual_accepted": th_accepted,
        "all_CMA_mean_live_probability_descriptive": statistics.fmean(all_probabilities),
        "all_CMA_realized_acceptance_rate_descriptive": accepted / offered,
        "job_security_all_actual_offers_constant_half": True,
    }


def interval(values):
    require(
        len(values) >= 2 and all((finite(value) for value in values)),
        "Defined paired values required for t inference",
    )
    try:
        from scipy.stats import t as student_t
    except ImportError as error:
        raise RuntimeError(
            "Paired inference requires scipy.stats.t; no normal approximation"
        ) from error
    n = len(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    critical = float(student_t.ppf(0.975, n - 1))
    if se:
        statistic = mean / se
        probability = float(2 * student_t.sf(abs(statistic), n - 1))
        degenerate = None
    else:
        statistic = 0.0 if mean == 0 else None
        probability = 1.0 if mean == 0 else 0.0
        degenerate = "zero_constant" if mean == 0 else "nonzero_constant_limit"
    return {
        "n": n,
        "df": n - 1,
        "mean": mean,
        "se": se,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "t": statistic,
        "p_two_sided": probability,
        "degenerate_constant_case": degenerate,
    }


def holm(probabilities):
    require(
        set(probabilities) == set(ENDPOINTS)
        and all((finite(p) and 0 <= p <= 1 for p in probabilities.values())),
        "Exactly4 finite confirmatory p-values required",
    )
    result, running = ({}, 0.0)
    for rank, (name, probability) in enumerate(
        sorted(probabilities.items(), key=lambda item: item[1])
    ):
        running = max(running, min(1.0, (4 - rank) * probability))
        result[name] = running
    return result


def infer(rows, calibrations):
    indexed = {}
    for row in rows:
        key = (row["policy_seed"], row["evaluation_seed"], row["arm"])
        require(key not in indexed, "Duplicate/scientifically repeated result")
        indexed[key] = row
    expected = {
        (seed, seed * 100 + index, arm)
        for seed in POLICIES
        for index in range(50)
        for arm in ("baseline", "recentred")
    }
    require(
        set(indexed) == expected and set(calibrations) == set(POLICIES),
        "Confirmatory inference requires complete5x50 paired histories in both arms",
    )
    per_seed, effects, balance = ({}, {name: [] for name in ENDPOINTS}, [])
    paired_rows = []
    for seed in POLICIES:
        own = {name: [] for name in ENDPOINTS}
        expected_probabilities, baseline_rates, treatment_rates = ([], [], [])
        for index in range(50):
            evaluation_seed = seed * 100 + index
            baseline, treatment = (
                indexed[seed, evaluation_seed, arm] for arm in ("baseline", "recentred")
            )
            require(
                baseline["TH_finalized_match_count"] > 0
                and treatment["TH_finalized_match_count"] > 0,
                "Zero finalized matches invalidate mechanism endpoint pair",
            )
            delta = {
                name: treatment[name] - baseline[name]
                for name in ENDPOINTS
                if finite(baseline[name]) and finite(treatment[name])
            }
            require(
                set(delta) == set(ENDPOINTS),
                "Exactly50 defined deltas per endpoint and seed required",
            )
            for name, value in delta.items():
                own[name].append(value)
            paired_rows.append(
                {
                    "policy_seed": seed,
                    "evaluation_seed": evaluation_seed,
                    "recentred_minus_baseline": delta,
                }
            )
            expected_probabilities.append(baseline["TH_mean_live_probability"])
            baseline_rates.append(baseline["TH_realized_acceptance_rate"])
            treatment_rates.append(treatment["TH_realized_acceptance_rate"])
        per_seed[seed] = {name: interval(values) for name, values in own.items()}
        for name in ENDPOINTS:
            effects[name].append(per_seed[seed][name]["mean"])
        frozen = calibrations[seed]
        require(
            frozen["policy_seed"] == seed
            and frozen["tau"] == 1.2
            and (frozen["status"] == "FROZEN_AFTER_50_BASELINE_HISTORIES")
            and (frozen["baseline_seed_block"] == list(range(seed * 100, seed * 100 + 50))),
            "Wrong fit record for this policy",
        )
        threshold = frozen["balance_T_s"]
        recomputed = 0.5 * statistics.stdev(expected_probabilities)
        require(
            finite(threshold)
            and threshold >= 0
            and math.isclose(threshold, recomputed, rel_tol=1e-10, abs_tol=1e-12),
            "The fitted target differs from the mean live probabilities of the 50 baseline histories",
        )
        contrast = statistics.fmean(treatment_rates) - statistics.fmean(baseline_rates)
        sensitivity = 0.5 * statistics.stdev(baseline_rates)
        balance.append(
            {
                "policy_seed": seed,
                "absolute_arm_mean_realized_rate_difference": abs(contrast),
                "arm_mean_realized_rate_difference": contrast,
                "frozen_primary_T_s": threshold,
                "within_T_s": abs(contrast) <= threshold,
                "within_2T_s": abs(contrast) <= 2 * threshold,
                "descriptive_realized_SD_half": sensitivity,
                "descriptive_within_realized_SD_half": abs(contrast) <= sensitivity,
                "descriptive_within_twice_realized_SD_half": abs(contrast) <= 2 * sensitivity,
            }
        )
    across = {name: interval(values) for name, values in effects.items()}
    adjusted = holm({name: record["p_two_sided"] for name, record in across.items()})
    for name, wanted in ENDPOINTS.items():
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in effects[name]]
        forward = signs.count(wanted)
        opposite = signs.count(-wanted)
        verdict = (
            "DETECTED"
            if adjusted[name] < 0.05 and forward >= 4
            else (
                "DIRECTION_CONTRADICTED"
                if adjusted[name] < 0.05 and opposite >= 4
                else "WITHIN_NOISE"
            )
        )
        across[name].update(
            holm_adjusted_p=adjusted[name],
            registered_direction="positive" if wanted == 1 else "negative",
            policy_signs=signs,
            policies_in_registered_direction=forward,
            policies_in_opposite_direction=opposite,
            verdict=verdict,
        )
    balanced = sum((row["within_T_s"] for row in balance)) >= 4 and all(
        (row["within_2T_s"] for row in balance)
    )
    return {
        "per_policy_paired_df49": per_seed,
        "paired_rollout_deltas": paired_rows,
        "across_five_policies_df4": across,
        "primary_realized_rate_balance": {"pass": balanced, "per_policy": balance},
    }


TAU = 1.2
TOLERANCE = 1e-10


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def solve(retained, probabilities, tau=TAU):
    require(
        tau == TAU and len(retained) == len(probabilities) and (len(retained) > 0),
        "Exact nonempty paired baseline support and temperature 1.2 required",
    )
    require(
        all((math.isfinite(x) for x in [*retained, *probabilities]))
        and all((0 <= x <= 1 for x in probabilities)),
        "Finite baseline utilities/probabilities required",
    )
    target = statistics.fmean(probabilities)
    require(0 < target < 1, "Finite intercept cannot match a boundary pooled probability")

    def residual(value):
        return statistics.fmean((sigmoid((r + value) / tau) for r in retained)) - target

    lo, hi = (-80.0, 80.0)
    left, right = (residual(lo), residual(hi))
    require(left * right <= 0, "Initial numerical bracket does not contain the nuisance root")
    for _ in range(200):
        middle = (lo + hi) / 2
        value = residual(middle)
        if abs(value) <= TOLERANCE:
            return (middle, abs(value), target)
        if left * value < 0:
            hi = middle
        else:
            lo, left = (middle, value)
    raise ValueError("Bisection did not satisfy its specified numerical tolerance")
