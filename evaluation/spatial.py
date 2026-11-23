from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import pickle
import random
from pathlib import Path
import numpy as np
import torch
from . import commute

SCREEN_REASONS = ("commute_gate", "industry_mismatch", "match_quality_floor", "selectivity")
DECLINE_REASONS = (
    "wage_gap",
    "long_commute",
    "low_wage_utility",
    "poor_match_quality",
    "high_reservation",
    "age_friction",
    "random_rejection",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def create(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def ratio(n, d):
    return n / d if d else None


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def integer(value):
    value = number(value)
    return int(value) if value is not None and value == int(value) else None


def da(value):
    value = integer(value)
    return None if value is None else str(value)


def rng_fingerprint():
    return hashlib.sha256(
        pickle.dumps(
            (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()),
            protocol=5,
        )
    ).hexdigest()


def quantile(ordered, q):
    if not ordered:
        return None
    p = (len(ordered) - 1) * q
    i = int(p)
    return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (p - i)


def distribution(values):
    ordered = sorted(values)
    n = len(ordered)
    total = math.fsum(ordered)
    return {
        "n": n,
        "sum": total,
        "mean": ratio(total, n),
        "median": quantile(ordered, 0.5),
        "p90": quantile(ordered, 0.9),
        "p95": quantile(ordered, 0.95),
        "deciles": [quantile(ordered, i / 10) for i in range(1, 10)],
        "gini": (
            2 * math.fsum(((i + 1) * v for i, v in enumerate(ordered))) / (n * total) - (n + 1) / n
            if n and total > 0
            else None
        ),
        "minimum": ordered[0] if n else None,
        "maximum": ordered[-1] if n else None,
        "zero_count": sum((v == 0 for v in ordered)),
        "values": values,
    }


def firm_tiers(path):
    counts = Counter()
    rows = 0
    missing = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            key = da(row.get("da_id", row.get("da")))
            if key is None:
                missing += 1
            else:
                counts[key] += 1
    require(counts and missing == 0, "Full firm source has unavailable DA rows")
    ordered = sorted(counts.values())
    t1 = ordered[len(ordered) // 3]
    t2 = ordered[2 * len(ordered) // 3]
    tiers = {
        key: "low" if n < 5 or n <= t1 else "middle" if n <= t2 else "high"
        for key, n in counts.items()
    }
    return (
        tiers,
        {
            "firm_rows": rows,
            "classified_DAs": len(counts),
            "lower_tercile_count": t1,
            "upper_tercile_count": t2,
            "rule": "Native full2006 firm counts: <5 or <=lowertercile low; <=uppertercile middle; otherwise high. DAs absent from full firm rows use native low-tier fallback; no official urban classification.",
        },
    )


def _matrix(matcher):
    use_time = getattr(matcher, "_use_travel_time", None)
    require(type(use_time) is bool, "Matcher mode not explicit")
    matrix = getattr(matcher, "da_travel_time_matrix" if use_time else "da_distance_matrix", None)
    indices = getattr(matcher, "da_id_to_idx", None)
    require(
        matrix is not None and isinstance(indices, dict),
        "The distance matrix or its index is unavailable",
    )
    if torch.is_tensor(matrix):
        require(matrix.device.type == "cpu", "Passive spatial replay is CPU-only")
        array = matrix.detach().numpy()
    else:
        array = np.asarray(matrix)
    require(array.ndim == 2 and array.shape[0] == array.shape[1], "The area matrix is not square")
    return (
        array,
        {da(k): int(v) for k, v in indices.items()},
        60.0 if use_time else 30.0,
        "min" if use_time else "km",
    )


def opportunities(matcher, home_das, vacancies):
    matrix, indices, threshold, unit = _matrix(matcher)
    dest = []
    unknown = 0
    for vacancy in vacancies.values():
        key = da(getattr(vacancy, "da_id", None))
        idx = indices.get(key)
        if idx is None or not 0 <= idx < matrix.shape[1]:
            unknown += 1
        else:
            dest.append(idx)
    results = {}
    for home in sorted(home_das):
        idx = indices.get(home)
        reason = None
        value = None
        if idx is None or not 0 <= idx < matrix.shape[0]:
            reason = "home_DA_absent_from_native_matrix"
        elif unknown:
            reason = "one_or_more_remaining_vacancy_DAs_unresolved"
        elif not dest:
            value = 0
        else:
            values = matrix[idx, np.asarray(dest, dtype=np.int64)]
            if not np.isfinite(values).all() or (values < 0).any():
                reason = "invalid_native_matrix_cell"
            else:
                value = int(np.count_nonzero(values <= threshold))
        results[home] = {"reachable_closing_vacancies": value, "reachable_unknown_reason": reason}
    return (
        results,
        {
            "closing_vacancies": len(vacancies),
            "unknown_vacancy_DAs": unknown,
            "threshold": threshold,
            "unit": unit,
            "worker_specific_factors_applied": False,
            "native_lookup_or_cache_method_called": False,
        },
    )


def native_rejection_counts(env):
    stats = getattr(env, "last_matching_stats", None)
    require(isinstance(stats, dict), "The matching statistics are unavailable")
    screen = integer(stats.get("applicants_screened_out"))
    declines = integer(stats.get("rejections"))
    source = stats.get("rejection_reasons_by_da")
    known_empty = source is None and screen == declines == 0
    if known_empty:
        source = {}
    if source is None:
        return (
            {},
            {
                "source_present": False,
                "known_empty_from_explicit_zero_stage_totals": False,
                "screen_global": screen,
                "declines_global": declines,
                "all_stages_reconciled": False,
                "reason": "native_by_DA_reason_source_absent",
            },
        )
    require(isinstance(source, dict), "The rejection reasons by area are not a mapping")
    copied = {}
    unknown = []
    for key, reasons in source.items():
        home = da(key)
        require(
            home is not None and isinstance(reasons, dict),
            "The home area of a rejection is unavailable",
        )
        row = {}
        for reason, count in reasons.items():
            require(type(count) is int and count >= 0, "Invalid count of a rejection reason")
            row[str(reason)] = count
            if reason not in SCREEN_REASONS + DECLINE_REASONS:
                unknown.append(reason)
        copied[home] = row
    observed_screen = sum((r.get(k, 0) for r in copied.values() for k in SCREEN_REASONS))
    observed_declines = sum((r.get(k, 0) for r in copied.values() for k in DECLINE_REASONS))
    return (
        copied,
        {
            "source_present": True,
            "known_empty_from_explicit_zero_stage_totals": known_empty,
            "screen_global": screen,
            "declines_global": declines,
            "screen_by_DA": observed_screen,
            "declines_by_DA": observed_declines,
            "unknown_reason_keys": sorted(set(unknown)),
            "all_stages_reconciled": not unknown
            and screen == observed_screen
            and (declines == observed_declines),
            "reason": "Exact native sequential primary reasons; screen tag is co-occurrence, not a causal rejection mechanism.",
        },
    )


class SpatialObserver:

    def __init__(self, output_dir, provenance=None, firm_csv_path=None):
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        require(firm_csv_path is not None, "The full firm file is required")
        self.firm_csv = Path(firm_csv_path).resolve()
        self.tiers, self.tier_provenance = firm_tiers(self.firm_csv)
        self.provenance = dict(provenance or {})
        self.path = self.output / "SPATIAL_MONTHS.jsonl.gz"
        require(not self.path.exists(), "Preserve spatial observations")
        self.stream = gzip.open(self.path, "xt", encoding="utf-8", compresslevel=1)
        self.renewal_path = self.output / "SPATIAL_RENEWAL_MONTHS.jsonl.gz"
        require(not self.renewal_path.exists(), "Preserve spatial renewal observations")
        self.renewal_stream = gzip.open(self.renewal_path, "xt", encoding="utf-8", compresslevel=1)
        self.months = 0
        self.renewal_months = 0
        self.opening = None
        self.initial = None
        self.summary = []
        self.distributions = defaultdict(lambda: defaultdict(list))
        self.worker_wages = defaultdict(lambda: defaultdict(list))
        self.posted_wages = defaultdict(lambda: defaultdict(list))
        self.complete = False
        self.commute = commute
        self.source_pins = {"firm_csv": pin(self.firm_csv)}

    def before(self, env, action, month_index, calendar):
        state = rng_fingerprint()
        require(
            self.opening is None and month_index == self.months, "Spatial opening lifecycle differs"
        )
        require(
            list(calendar) == [2006 + month_index // 12, 1 + month_index % 12],
            "Spatial historical calendar differs",
        )
        self.identity = (month_index, list(calendar))
        self.opening = {
            str(k): {
                "status": str(ws.worker.employment_status),
                "home_DA": da(getattr(ws.worker, "da_id", None)),
            }
            for k, ws in env.state_manager.workers.items()
        }
        if self.initial is None:
            counts = Counter(
                (r["home_DA"] for r in self.opening.values() if r["home_DA"] is not None)
            )
            self.initial = set(counts)
            create(
                self.output / "INITIAL_SPATIAL_DOMAIN.json",
                {
                    "schema": "passive_spatial_initial_DA_domain_v1",
                    "initial_residents": len(self.opening),
                    "unknown_home_DA_residents": sum(
                        (r["home_DA"] is None for r in self.opening.values())
                    ),
                    "initial_DA_counts": dict(sorted(counts.items())),
                    "firm_tiers": {k: self.tiers.get(k, "low") for k in sorted(counts)},
                    "classification": self.tier_provenance,
                    "sources": self.source_pins,
                    "provenance": self.provenance,
                    "selection": "All initially occupied known home DAs; no selection on realized labour status, offers, outcomes or commutes.",
                },
            )
        require(state == rng_fingerprint(), "Spatial opening changed an RNG state")

    def after(self, env, month_index, calendar):
        state = rng_fingerprint()
        require(
            self.opening is not None and (month_index, list(calendar)) == self.identity,
            "Spatial closing lifecycle differs",
        )
        year, month = calendar
        workers = env.state_manager.workers
        cells = {}
        unknown_home = 0
        wage_coverage = Counter()

        def cell(home):
            return cells.setdefault(
                home,
                {
                    "home_DA": home,
                    "initial_domain": home in self.initial,
                    "firm_count_tier": self.tiers.get(home, "low"),
                    "P": 0,
                    "E": 0,
                    "U_raw": 0,
                    "N_raw": 0,
                    "opening_U_raw": 0,
                    "endpoint_U_to_E": 0,
                    "unemployed_duration_months_n": 0,
                    "unemployed_duration_months_sum": 0.0,
                    "stock_assigned_time_n": 0,
                    "stock_assigned_time_sum": 0.0,
                    "stock_positive_time_n": 0,
                    "stock_positive_time_sum": 0.0,
                    "UE_assigned_time_n": 0,
                    "UE_assigned_time_sum": 0.0,
                    "UE_positive_time_n": 0,
                    "UE_positive_time_sum": 0.0,
                    "raw_U_duration_month_values": [],
                    "stock_assigned_time_values": [],
                    "UE_assigned_time_values": [],
                },
            )

        for home in self.initial:
            cell(home)
        for key, ws in workers.items():
            w = ws.worker
            home = da(getattr(w, "da_id", None))
            status = str(w.employment_status)
            require(status in ("Employed", "Unemployed", "NILF"), "Unknown raw worker status")
            if home is None:
                unknown_home += 1
                continue
            row = cell(home)
            row["P"] += 1
            row[{"Employed": "E", "Unemployed": "U_raw", "NILF": "N_raw"}[status]] += 1
            if status == "Unemployed":
                duration = number(getattr(w, "unemployment_months", None))
                if duration is not None and duration >= 0:
                    row["unemployed_duration_months_n"] += 1
                    row["unemployed_duration_months_sum"] += duration
                    row["raw_U_duration_month_values"].append(duration)
            opening = self.opening.get(str(key))
            if opening and opening["status"] == "Unemployed":
                origin = opening["home_DA"]
                if origin is not None:
                    cell(origin)["opening_U_raw"] += 1
                    cell(origin)["endpoint_U_to_E"] += int(status == "Employed")
            if status == "Employed":
                employer = getattr(w, "current_employer_id", None)
                firm_state = env.state_manager.firms.get(employer)
                code = integer(getattr(getattr(firm_state, "firm", None), "industry", None))
                wage = number(getattr(w, "current_wage", None))
                wage_coverage["E_persons"] += 1
                wage_coverage["unknown_employer_industry"] += int(code is None)
                if wage is not None and wage > 0:
                    wage_coverage["positive_wage_n"] += 1
                    for window in ("full132", str(year)):
                        self.worker_wages[window]["UNKNOWN" if code is None else str(code)].append(
                            wage
                        )
                else:
                    wage_coverage["missing_or_nonpositive_wage_E"] += 1
        rejection, stage = native_rejection_counts(env)
        for home in rejection:
            cell(home)
        access, availability = opportunities(
            env.matching_mechanism, set(cells), env.state_manager.vacancies
        )
        opening = {key: r["status"] for key, r in self.opening.items()}
        commute = self.commute.snapshot_commutes(
            env,
            opening,
            phase="after_market_before_renewal",
            year=year,
            calendar_month=month,
            month_index=month_index,
            provenance=self.provenance,
        )
        commute_unknown = Counter()
        for sample in commute["workers"]:
            home = da(sample["home_da_id"])
            if home is None:
                continue
            row = cell(home)
            time = sample["model_one_way_time_min"]
            distance = sample["da_centroid_haversine_km"]
            ue = sample["endpoint_status_route"] == "opening_U_to_closing_E"
            if time is None:
                commute_unknown[str(sample["time_unknown_reason"])] += 1
            for field, measure in (("time", time), ("straight_line_km", distance)):
                if measure is None:
                    continue
                for window in ("full132", str(year)):
                    self.distributions[window]["stock_" + field + "_all_valid"].append(measure)
                    if measure > 0:
                        self.distributions[window]["stock_" + field + "_positive"].append(measure)
                    if ue:
                        self.distributions[window]["opening_U_to_E_" + field + "_all_valid"].append(
                            measure
                        )
                    if ue and measure > 0:
                        self.distributions[window]["opening_U_to_E_" + field + "_positive"].append(
                            measure
                        )
            if time is not None:
                row["stock_assigned_time_n"] += 1
                row["stock_assigned_time_sum"] += time
                row["stock_assigned_time_values"].append(time)
                if time > 0:
                    row["stock_positive_time_n"] += 1
                    row["stock_positive_time_sum"] += time
                if ue:
                    row["UE_assigned_time_n"] += 1
                    row["UE_assigned_time_sum"] += time
                    row["UE_assigned_time_values"].append(time)
                if ue and time > 0:
                    row["UE_positive_time_n"] += 1
                    row["UE_positive_time_sum"] += time
        active = getattr(env, "firm_active", None)
        codes = getattr(env, "firm_industries", None)
        posted = getattr(env, "firm_posted_wages", None)
        require(
            all((torch.is_tensor(x) for x in (active, codes, posted))),
            "The posted-wage tensors are unavailable",
        )
        require(
            all((x.ndim == 1 for x in (active, codes, posted)))
            and len(active) == len(codes) == len(posted),
            "The posted-wage tensors are not aligned",
        )
        for live, code, wage in zip(
            active.detach().cpu().tolist(),
            codes.detach().cpu().tolist(),
            posted.detach().cpu().tolist(),
        ):
            if live and number(wage) is not None and (wage > 0):
                for window in ("full132", str(year)):
                    self.posted_wages[window][str(int(code))].append(float(wage))
        native_values = getattr(env, "last_matching_stats", {}).get("commute_values")
        if native_values is not None:
            require(
                all((number(v) is not None and v >= 0 for v in native_values)),
                "Invalid commute array of the matching events",
            )
            for window in ("full132", str(year)):
                self.distributions[window][
                    "native_matching_event_commute_" + availability["unit"]
                ].extend(map(float, native_values))
        for home, row in cells.items():
            reason = rejection.get(home, {})
            screen = sum((reason.get(k, 0) for k in SCREEN_REASONS))
            declines = sum((reason.get(k, 0) for k in DECLINE_REASONS))
            count = access[home]["reachable_closing_vacancies"]
            row.update(access[home])
            row.update(
                {
                    "raw_native_reason_counts": reason,
                    "quality_screen_failures": screen if stage["source_present"] else None,
                    "quality_fail_long_commute_tag": (
                        reason.get("commute_gate", 0) if stage["source_present"] else None
                    ),
                    "native_primary_offer_declines": declines if stage["source_present"] else None,
                    "native_primary_long_commute_declines": (
                        reason.get("long_commute", 0) if stage["source_present"] else None
                    ),
                    "quality_fail_long_commute_share": (
                        ratio(reason.get("commute_gate", 0), screen)
                        if stage["source_present"]
                        else None
                    ),
                    "primary_long_commute_decline_share": (
                        ratio(reason.get("long_commute", 0), declines)
                        if stage["source_present"]
                        else None
                    ),
                    "access_U_person_months_including_zeroV": (
                        row["U_raw"] if count is not None else None
                    ),
                    "access_reachable_sum_including_zeroV": (
                        count * row["U_raw"] if count is not None else None
                    ),
                    "earlier_access_U_person_months_positive_V_only": (
                        row["U_raw"]
                        if count is not None and availability["closing_vacancies"] > 0
                        else 0 if count is not None else None
                    ),
                    "earlier_access_sum_positive_V_only": (
                        count * row["U_raw"]
                        if count is not None and availability["closing_vacancies"] > 0
                        else 0 if count is not None else None
                    ),
                }
            )
        unchanged = state == rng_fingerprint()
        require(unchanged, "Passive spatial closing changed an RNG state")
        require(
            sum((r["P"] for r in cells.values())) + unknown_home == len(workers),
            "DA resident stock coverage does not close",
        )
        require(
            len(commute["workers"])
            == sum((r["E"] for r in cells.values()))
            + sum(
                (
                    ws.worker.employment_status == "Employed"
                    and da(getattr(ws.worker, "da_id", None)) is None
                    for ws in workers.values()
                )
            ),
            "Complete employed commute snapshot count differs",
        )
        record = {
            "schema": "passive_spatial_market_month_v1",
            "month_index": month_index,
            "calendar": list(calendar),
            "timing": "Immediately after native env._step; before external population renewal",
            "availability": availability,
            "DA_measurements": [cells[key] for key in sorted(cells)],
            "stage_counter_reconciliation": stage,
            "commute_coverage": commute["counts"],
            "commute_unknown_reasons": dict(commute_unknown),
            "wage_sample_coverage": dict(wage_coverage),
            "unknown_home_DA_residents": unknown_home,
            "raw_residents": len(workers),
            "December_support_snapshot": (
                {key: row["P"] for key, row in cells.items()} if month == 12 else None
            ),
            "observer_actor_calls": 0,
            "observer_RNG_draws": 0,
            "observer_RNG_state_unchanged": unchanged,
        }
        self.stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.summary.append(
            {
                "month_index": month_index,
                "calendar": list(calendar),
                "stage_counter_reconciliation": stage,
                "availability": availability,
                "raw_residents": len(workers),
                "employed_commute_snapshot_n": len(commute["workers"]),
                "native_flow_sample_n": None if native_values is None else len(native_values),
                "RNG_unchanged": unchanged,
            }
        )
        self.months += 1
        self.opening = None
        return record

    def after_renewal(self, env, month_index, calendar):
        state = rng_fingerprint()
        require(
            self.opening is None
            and month_index == self.renewal_months == self.months - 1
            and (list(calendar) == self.summary[-1]["calendar"]),
            "Spatial renewal lifecycle differs",
        )
        cells = {
            home: {"home_DA": home, "P": 0, "E": 0, "U_raw": 0, "N_raw": 0} for home in self.initial
        }
        unknown = 0
        for ws in env.state_manager.workers.values():
            home = da(getattr(ws.worker, "da_id", None))
            status = ws.worker.employment_status
            require(status in ("Employed", "Unemployed", "NILF"), "Unknown renewal raw status")
            if home is None:
                unknown += 1
                continue
            row = cells.setdefault(home, {"home_DA": home, "P": 0, "E": 0, "U_raw": 0, "N_raw": 0})
            row["P"] += 1
            row[{"Employed": "E", "Unemployed": "U_raw", "NILF": "N_raw"}[status]] += 1
        reachable, availability = opportunities(
            env.matching_mechanism, set(cells), env.state_manager.vacancies
        )
        for home, row in cells.items():
            row.update(reachable[home])
            row["initial_domain"] = home in self.initial
            row["firm_count_tier"] = self.tiers.get(home, "low")
        require(state == rng_fingerprint(), "Spatial renewal observation changed an RNG state")
        record = {
            "schema": "passive_spatial_renewal_month_v1",
            "month_index": month_index,
            "calendar": list(calendar),
            "timing": "After unchanged external population finalize and original core boundary audit",
            "availability": availability,
            "DA_measurements": [cells[key] for key in sorted(cells)],
            "unknown_home_DA_residents": unknown,
            "raw_residents": len(env.state_manager.workers),
            "observer_RNG_state_unchanged": True,
        }
        self.renewal_stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.renewal_stream.flush()
        self.renewal_months += 1
        return record

    def finish(self, expected_months=132):
        require(
            not self.complete
            and self.opening is None
            and (self.months == self.renewal_months == expected_months),
            "Spatial observation lifecycle incomplete",
        )
        self.stream.close()
        self.renewal_stream.close()
        self.complete = True
        distributions = {
            window: {name: distribution(values) for name, values in arrays.items()}
            for window, arrays in self.distributions.items()
        }
        wages = {
            window: {code: distribution(values) for code, values in arrays.items()}
            for window, arrays in self.worker_wages.items()
        }
        posted = {
            window: {code: distribution(values) for code, values in arrays.items()}
            for window, arrays in self.posted_wages.items()
        }
        path = self.output / "SPATIAL_DISTRIBUTIONS.json.gz"
        with gzip.open(path, "xt", encoding="utf-8", compresslevel=1) as stream:
            json.dump(
                {
                    "schema": "complete_passive_stock_endpoint_and_wage_distributions_v1",
                    "windows": distributions,
                    "industry_worker_positive_hourly_wages": wages,
                    "industry_active_firm_positive_posted_wages": posted,
                    "stock_sampling": "All observed employed residents per month; no5000-record tail cap. Native source-joined local assignments; missing destinations/geometry retain explicit reasons.",
                    "endpoint_flow_sampling": "Opening raw U to closing E, including all valid zero times and separately positive-only values. Distinct from native matching-event arrays.",
                    "quantiles": "Exact linear-interpolated quantiles from all saved values in each calendar year/window; no average of monthly medians/p95.",
                    "individual_IDs_or_OD_records_exported": False,
                },
                stream,
                allow_nan=False,
            )
        failed = [
            r["calendar"]
            for r in self.summary
            if not r["stage_counter_reconciliation"]["all_stages_reconciled"]
        ]
        report = {
            "schema": "passive_spatial_observer_report_v1",
            "status": "COMPLETE_SPATIAL_COLLECTION",
            "months": self.months,
            "renewal_months": self.renewal_months,
            "provenance": self.provenance,
            "source_pins": self.source_pins,
            "monthly_coverage": self.summary,
            "stage_counter_unreconciled_months": failed,
            "observer_actor_calls": 0,
            "observer_RNG_draws": 0,
            "RNG_unchanged_all_calls": all((r["RNG_unchanged"] for r in self.summary)),
            "runtime_state_or_parameters_modified": False,
            "new_spatial_support_rule_evaluated": False,
            "scientific_acceptance": False,
        }
        create(self.output / "SPATIAL_OBSERVER_REPORT.json", report)
        return {
            "schema": "spatial_replay_observer_receipt_v1",
            "status": "COMPLETE_SPATIAL_COLLECTION",
            "artifacts": [
                pin(self.output / name)
                for name in (
                    "INITIAL_SPATIAL_DOMAIN.json",
                    "SPATIAL_MONTHS.jsonl.gz",
                    "SPATIAL_RENEWAL_MONTHS.jsonl.gz",
                    "SPATIAL_DISTRIBUTIONS.json.gz",
                    "SPATIAL_OBSERVER_REPORT.json",
                )
            ],
            "months": self.months,
            "renewal_months": self.renewal_months,
            "stage_counter_unreconciled_months": failed,
            "actor_calls": 0,
            "RNG_calls": 0,
            "parameters_changed": False,
            "scientific_acceptance": False,
        }
