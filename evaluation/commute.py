from collections import Counter
import math
import torch


def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value):
    number = _finite(value)
    return int(number) if number is not None and number == int(number) else None


def _immigrant(value):
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes", "immigrant"):
            return True
        if value.lower() in ("false", "0", "no", "nonimmigrant", "native"):
            return False
    return None


def _distribution(values, *, time_minutes=False):
    if not values:
        result = {"n": 0, "mean": None, "median": None, "p90": None, "zero_count": 0}
        if time_minutes:
            result.update(
                {
                    "share_over_60": None,
                    "share_at_least_60": None,
                    "share_over_90": None,
                    "share_at_least_90": None,
                    "census_five_duration_bins": {
                        key: {"count": 0, "share": None}
                        for key in (
                            "under_15",
                            "15_to_under_30",
                            "30_to_under_45",
                            "45_to_under_60",
                            "60_plus",
                        )
                    },
                }
            )
        return result
    ordered = sorted(values)

    def quantile(probability):
        index = (len(ordered) - 1) * probability
        low = int(math.floor(index))
        high = int(math.ceil(index))
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    result = {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "zero_count": sum((v == 0 for v in values)),
    }
    if time_minutes:
        bins = {
            "under_15": sum((v < 15 for v in values)),
            "15_to_under_30": sum((15 <= v < 30 for v in values)),
            "30_to_under_45": sum((30 <= v < 45 for v in values)),
            "45_to_under_60": sum((45 <= v < 60 for v in values)),
            "60_plus": sum((v >= 60 for v in values)),
        }
        result.update(
            {
                "share_over_60": sum((v > 60 for v in values)) / len(values),
                "share_at_least_60": sum((v >= 60 for v in values)) / len(values),
                "share_over_90": sum((v > 90 for v in values)) / len(values),
                "share_at_least_90": sum((v >= 90 for v in values)) / len(values),
                "census_five_duration_bins": {
                    key: {"count": count, "share": count / len(values)}
                    for key, count in bins.items()
                },
            }
        )
    return result


def _firm_activity(env):
    identifiers = getattr(env, "firm_ids", None)
    activity = getattr(env, "firm_active", None)
    if identifiers is None or activity is None:
        return ({}, "authoritative_firm_activity_unavailable")
    if torch.is_tensor(activity):
        if activity.ndim != 1:
            return ({}, "authoritative_firm_activity_alignment_invalid")
        activity = activity.detach().cpu().tolist()
    elif isinstance(activity, (list, tuple)):
        activity = list(activity)
    else:
        return ({}, "authoritative_firm_activity_type_invalid")
    if len(identifiers) != len(activity) or len(set(identifiers)) != len(identifiers):
        return ({}, "authoritative_firm_activity_alignment_invalid")
    if any((type(value) is not bool for value in activity)):
        return ({}, "authoritative_firm_activity_values_not_boolean")
    return (dict(zip(identifiers, activity)), None)


def _paired_matrix_values(matrix, indexes, records, field, reason_field, map_reason, factors=None):
    if not torch.is_tensor(matrix) or matrix.ndim != 2:
        return
    selected = []
    for record in records:
        if record[reason_field] is not None:
            continue
        home, work = (record["_home_da"], record["_work_da"])
        left, right = (indexes.get(home), indexes.get(work))
        if (
            type(left) is not int
            or type(right) is not int
            or left < 0
            or (right < 0)
            or (left >= matrix.shape[0])
            or (right >= matrix.shape[1])
        ):
            record[reason_field] = map_reason
            continue
        if factors is not None and factors:
            raw = factors.get(record["_raw_worker_id"], 1.0)
            factor = _finite(raw)
            if factor is None or factor <= 0:
                record[reason_field] = "invalid_native_worker_commute_factor"
                continue
            record["native_worker_factor"] = factor
            record["worker_factor_source"] = (
                "native_worker_map"
                if record["_raw_worker_id"] in factors
                else "native_missing_worker_unity"
            )
        elif factors is not None:
            record["native_worker_factor"] = 1.0
            record["worker_factor_source"] = "native_empty_factor_map_unity"
        selected.append((record, left, right))
    if not selected:
        return
    with torch.no_grad():
        left = torch.tensor([x[1] for x in selected], device=matrix.device, dtype=torch.long)
        right = torch.tensor([x[2] for x in selected], device=matrix.device, dtype=torch.long)
        values = matrix[left, right]
        if factors is not None:
            base_values = values.detach().cpu().tolist()
            for (record, _, _), base in zip(selected, base_values):
                number = _finite(base)
                if number is not None and number >= 0:
                    record["native_base_one_way_time_min"] = number
                    record["native_base_time_unknown_reason"] = None
        if factors is not None and factors:
            multipliers = torch.tensor(
                [x[0]["native_worker_factor"] for x in selected],
                device=matrix.device,
                dtype=torch.float32,
            )
            values = values * multipliers
        copied = values.detach().cpu().tolist()
    for (record, _, _), value in zip(selected, copied):
        number = _finite(value)
        if number is None or number < 0:
            record[reason_field] = "invalid_matrix_commute_value"
        else:
            record[field] = number


def snapshot_commutes(
    env, opening_status=None, *, phase, year, calendar_month, month_index, provenance=None
):
    manager, matcher = (env.state_manager, env.matching_mechanism)
    opening_status = {} if opening_status is None else opening_status
    boundary_jobs = getattr(getattr(env, "boundary", None), "jobs", {})
    activity, activity_reason = _firm_activity(env)
    provenance = {} if provenance is None else dict(provenance)
    for name in (
        "run_id",
        "environment_id",
        "observer_epoch",
        "renewal_epoch",
        "source_reset_epoch",
    ):
        provenance.setdefault(name, None)
    provenance.setdefault(
        "identity_binding_reason",
        "No observer run binding was supplied to this standalone snapshot.",
    )
    records = []
    seen = set()
    for raw_id, state in manager.workers.items():
        worker = state.worker
        if worker.employment_status != "Employed":
            continue
        worker_id = str(raw_id)
        if worker_id in seen:
            raise ValueError("Commute observer worker identifiers collide after string conversion")
        seen.add(worker_id)
        employer = getattr(worker, "current_employer_id", None)
        firm_state = manager.firms.get(employer) if employer is not None else None
        firm = getattr(firm_state, "firm", None)
        home_da = getattr(worker, "da_id", None)
        work_da = getattr(firm, "da_id", None) if firm is not None else None
        home_cma = _integer(getattr(worker, "cma", None))
        work_cma = _integer(getattr(firm, "cma", None)) if firm is not None else None
        prior = opening_status.get(worker_id)
        age = _integer(getattr(worker, "age", None))
        active = activity.get(employer) if employer is not None else None
        route = (
            "initial_assignment"
            if phase == "initial"
            else (
                "opening_E_to_closing_E"
                if prior == "Employed"
                else (
                    "opening_U_to_closing_E"
                    if prior == "Unemployed"
                    else (
                        "opening_NILF_to_closing_E"
                        if prior == "NILF"
                        else "opening_status_unavailable_to_E"
                    )
                )
            )
        )
        job = boundary_jobs.get(raw_id) if isinstance(boundary_jobs, dict) else None
        job_kind = job.get("kind") if isinstance(job, dict) else None
        reason = None
        if job_kind is not None and job_kind != "local_payroll":
            reason = "unobserved_nonlocal_or_nonpayroll_destination"
        elif employer is None:
            reason = "missing_assigned_employer"
        elif firm is None:
            reason = "assigned_employer_not_in_firm_registry"
        elif active is False:
            reason = "assigned_employer_inactive"
        elif active is None:
            reason = activity_reason or "assigned_employer_activity_unavailable"
        elif home_da is None:
            reason = "missing_home_da"
        elif work_da is None:
            reason = "missing_workplace_da"
        record = {
            "worker_id": worker_id,
            "status": "Employed",
            "age": age,
            "provenance_ref": "month_header.provenance",
            "age_15plus": None if age is None else age >= 15,
            "age_15plus_reason": None if age is not None else "native_age_missing_or_invalid",
            "home_cma": home_cma,
            "home_cma_reason": None if home_cma is not None else "missing_or_invalid_worker_cma",
            "workplace_cma": work_cma,
            "workplace_cma_reason": None if work_cma is not None else "missing_or_invalid_firm_cma",
            "immigrant_status": _immigrant(getattr(worker, "immigrant_status", None)),
            "employer_id": None if employer is None else str(employer),
            "assigned_employer_active": active,
            "employer_activity_source": (
                "env.firm_active_aligned_with_firm_ids" if active is not None else None
            ),
            "employer_activity_unknown_reason": (
                None
                if active is not None
                else activity_reason or "employer_not_in_activity_registry"
            ),
            "worker_instance_id": None,
            "worker_instance_unknown_reason": "No independent worker-generation ledger is read; worker_id is the native registry key.",
            "job_instance_id": None,
            "job_instance_unknown_reason": "No stable job-tenure identity is exposed to this observer.",
            "employer_lineage_id": None,
            "employer_lineage_unknown_reason": "Native firm slots can be recycled; no independent generation ledger is read here.",
            "home_da_id": _integer(home_da),
            "workplace_da_id": _integer(work_da),
            "resident_job_kind": (
                job_kind if job_kind is not None else "native_modeled_assigned_firm"
            ),
            "opening_status": prior,
            "endpoint_status_route": route,
            "direct_NILF_mechanism_inferred_from_status": False,
            "census_commuter_eligibility_identified": False,
            "census_commuter_eligibility": None,
            "census_commuter_eligibility_unknown_reason": "Native state does not identify Census usual-place/home-working eligibility.",
            "stored_current_job_time_min": _finite(getattr(worker, "commute_time_min", None)),
            "stored_time_current_route_validity": "unverified_native_stored_field",
            "model_one_way_time_min": None,
            "time_unknown_reason": reason,
            "native_base_one_way_time_min": None,
            "native_base_time_unknown_reason": "Not looked up for an uncomputed assignment.",
            "time_reconstructed_from_current_assignment": True,
            "da_centroid_haversine_km": None,
            "distance_unknown_reason": reason,
            "native_worker_factor": None,
            "worker_factor_source": "not_applied_uncomputed_time",
            "_raw_worker_id": raw_id,
            "_home_da": home_da,
            "_work_da": work_da,
        }
        records.append(record)
    indexes = getattr(matcher, "da_id_to_idx", {})
    time_matrix = getattr(matcher, "da_travel_time_matrix", None)
    distance_matrix = getattr(matcher, "da_distance_matrix", None)
    time_mode = getattr(matcher, "_use_travel_time", False) is True
    for record in records:
        if record["time_unknown_reason"] is None:
            if not time_mode:
                record["time_unknown_reason"] = "native_travel_time_mode_inactive"
            elif not torch.is_tensor(time_matrix) or time_matrix.ndim != 2:
                record["time_unknown_reason"] = "native_time_matrix_unavailable"
        if record["distance_unknown_reason"] is None and (
            not torch.is_tensor(distance_matrix) or distance_matrix.ndim != 2
        ):
            record["distance_unknown_reason"] = "unscaled_da_distance_matrix_unavailable"
    _paired_matrix_values(
        time_matrix,
        indexes,
        records,
        "model_one_way_time_min",
        "time_unknown_reason",
        "home_or_workplace_da_absent_from_time_matrix",
        factors=getattr(matcher, "worker_commute_factors", None) or {},
    )
    _paired_matrix_values(
        distance_matrix,
        indexes,
        records,
        "da_centroid_haversine_km",
        "distance_unknown_reason",
        "home_or_workplace_da_absent_from_distance_matrix",
    )
    for record in records:
        record.pop("_raw_worker_id")
        record.pop("_home_da")
        record.pop("_work_da")
    times = [
        r["model_one_way_time_min"] for r in records if r["model_one_way_time_min"] is not None
    ]
    distances = [
        r["da_centroid_haversine_km"] for r in records if r["da_centroid_haversine_km"] is not None
    ]
    time_unknown = Counter(
        (r["time_unknown_reason"] for r in records if r["model_one_way_time_min"] is None)
    )
    distance_unknown = Counter(
        (r["distance_unknown_reason"] for r in records if r["da_centroid_haversine_km"] is None)
    )
    if None in time_unknown or None in distance_unknown:
        raise ValueError("Unknown commute value lacks a reason")
    counts = {
        "total_residents": len(manager.workers),
        "employed_residents": len(records),
        "computed_time_count": len(times),
        "unknown_time_count": len(records) - len(times),
        "zero_time_count": sum((v == 0 for v in times)),
        "positive_time_count": sum((v > 0 for v in times)),
        "time_unknown_by_reason": dict(time_unknown),
        "computed_straight_line_distance_count": len(distances),
        "unknown_straight_line_distance_count": len(records) - len(distances),
        "zero_straight_line_distance_count": sum((v == 0 for v in distances)),
        "distance_unknown_by_reason": dict(distance_unknown),
        "stored_time_nonnull_count": sum(
            (r["stored_current_job_time_min"] is not None for r in records)
        ),
        "stored_time_null_count": sum((r["stored_current_job_time_min"] is None for r in records)),
        "endpoint_status_routes": dict(Counter((r["endpoint_status_route"] for r in records))),
        "home_cma_unknown_count": sum((r["home_cma"] is None for r in records)),
        "census_commuter_eligibility_known_count": 0,
        "census_eligible_true_count": 0,
        "census_eligible_false_count": 0,
        "census_eligible_unknown_count": len(records),
        "employed_age15_plus_count": sum((r["age_15plus"] is True for r in records)),
        "employed_under15_count": sum((r["age_15plus"] is False for r in records)),
        "employed_age_unknown_count": sum((r["age_15plus"] is None for r in records)),
        "age15_plus_census_eligibility": {
            "true": 0,
            "false": 0,
            "unknown": sum((r["age_15plus"] is True for r in records)),
        },
        "assigned_employer_inactive_count": sum(
            (r["assigned_employer_active"] is False for r in records)
        ),
        "assigned_employer_activity_unknown_count": sum(
            (r["assigned_employer_active"] is None for r in records)
        ),
    }
    if counts["employed_residents"] != counts["computed_time_count"] + counts["unknown_time_count"]:
        raise ValueError("Commute coverage accounting does not close")
    if counts["employed_residents"] != sum(
        (
            counts[key]
            for key in (
                "employed_age15_plus_count",
                "employed_under15_count",
                "employed_age_unknown_count",
            )
        )
    ):
        raise ValueError("Commute age eligibility partition does not close")
    public_provenance = {
        **provenance,
        "method": "paired native DA time-matrix lookup with unchanged native worker-factor convention",
        "time_matrix_source": "env.matching_mechanism.da_travel_time_matrix",
        "distance_matrix_source": "env.matching_mechanism.da_distance_matrix; no worker factor applied",
        "time_is_current_assignment_reconstruction": True,
        "time_empirically_observed": False,
        "stored_time_backfilled": False,
        "geometry_imputation_flag": None,
        "geometry_imputation_unknown_reason": "Native DA centroid matrices do not expose an independently verified imputation ledger.",
        "coordinate_reference_system": None,
        "coordinate_reference_system_unknown_reason": "No independently archived CRS declaration is read by this observer.",
        "coordinate_source_pin": None,
        "coordinate_source_pin_unknown_reason": "Matrix source is identified, but no separately verified source-coordinate artifact is read here.",
        "home_DA_vintage": None,
        "workplace_DA_vintage": None,
        "centroid_vintage": None,
        "geography_vintage_unknown_reason": "Native DA identifiers/matrix mapping expose no independently verified Census-vintage tag.",
        "travel_network_baseline_label": "2006 baseline as labeled in the pinned GTHATravelTimeCalculator source; not an observed individual route.",
    }
    return {
        "schema": "read_only_complete_assigned_commute_month_v1",
        "phase": phase,
        "year": int(year),
        "calendar_month": int(calendar_month),
        "month_index": int(month_index),
        **{
            name: public_provenance[name]
            for name in (
                "run_id",
                "environment_id",
                "observer_epoch",
                "renewal_epoch",
                "source_reset_epoch",
            )
        },
        "time_unit": "one_way_model_minutes",
        "distance_unit": "unscaled_DA_centroid_Haversine_km",
        "environment_scope": "primary",
        "time_empirically_observed": False,
        "distance_is_observed_route_length": False,
        "stored_worker_fields_changed": False,
        "provenance": public_provenance,
        "counts": counts,
        "time_distribution": _distribution(times, time_minutes=True),
        "age15_plus_time_distribution": _distribution(
            [
                r["model_one_way_time_min"]
                for r in records
                if r["age_15plus"] is True and r["model_one_way_time_min"] is not None
            ],
            time_minutes=True,
        ),
        "straight_line_distance_distribution": _distribution(distances),
        "workers": records,
    }
