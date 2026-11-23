from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

SEED_FLAG = "SEPARATION_DESTINATION_SPLIT_SEED"
DOMAIN = "ordinary_destination_25_54_v1"
N_SHARE = 1079900 / (1386200 + 1079900)
U_SHARE = 1.0 - N_SHARE
PARAMETER_FLAG = "ABSOLUTE_SEPARATION_PARAMETERS"
_ABSOLUTE_PARAMETERS = None


def absolute_parameters():
    global _ABSOLUTE_PARAMETERS
    raw_path = os.environ.get(PARAMETER_FLAG)
    if not raw_path:
        raise ValueError("Absolute calibration requires an explicit parameter file")
    path = Path(raw_path).resolve(strict=True)
    if _ABSOLUTE_PARAMETERS is not None:
        if _ABSOLUTE_PARAMETERS["path"] != str(path):
            raise ValueError("Absolute separation parameters cannot be hot-swapped")
        return dict(_ABSOLUTE_PARAMETERS)
    content = path.read_bytes()
    supplied = json.loads(content)
    if (
        type(supplied.get("age_min")) is not int
        or supplied["age_min"] != 25
        or type(supplied.get("age_max")) is not int
        or (supplied["age_max"] != 54)
    ):
        raise ValueError("Absolute separation parameters have unsupported age scope")
    coefficients = {}
    for key in ("alpha_u", "alpha_n"):
        value = supplied.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Absolute separation coefficients must be finite and nonnegative")
        coefficients[key] = float(value)
    total = coefficients["alpha_u"] + coefficients["alpha_n"]
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Absolute separation coefficients must have positive finite total")
    _ABSOLUTE_PARAMETERS = {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "age_min": 25,
        "age_max": 54,
        **coefficients,
    }
    return dict(_ABSOLUTE_PARAMETERS)


def absolute_rate_vectors(env, base_rates, ages):
    import torch

    parameters = absolute_parameters()
    if base_rates.shape != ages.shape:
        raise ValueError("Absolute separation rate/age alignment differs")
    scope = (ages >= 25) & (ages < 55)
    to_u = torch.where(scope, base_rates * parameters["alpha_u"], base_rates)
    to_n = torch.where(scope, base_rates * parameters["alpha_n"], torch.zeros_like(base_rates))
    total = torch.where(scope, to_u + to_n, base_rates)
    valid = (
        torch.isfinite(base_rates)
        & torch.isfinite(to_u)
        & torch.isfinite(to_n)
        & torch.isfinite(total)
        & (base_rates >= 0)
        & (base_rates <= 1)
        & (to_u >= 0)
        & (to_n >= 0)
        & (total <= 1)
    )
    if not bool(valid.all().item()):
        raise ValueError("Invalid calibrated separation probability; clipping is forbidden")
    env._absolute_separation_this_step = {
        "schema": "absolute_ordinary_separation_inputs_v1",
        "enabled": True,
        "period": [int(env.current_year), int(env.current_month), int(env.current_step)],
        "parameters": parameters,
        "n_share": parameters["alpha_n"] / (parameters["alpha_u"] + parameters["alpha_n"]),
        "probability_dtype": str(base_rates.dtype),
    }
    return {
        "base": base_rates,
        "to_u": to_u,
        "to_n": to_n,
        "total": total,
        "scope": scope,
        "parameters": parameters,
    }


def active():
    return True


def destination_draw(seed, step, worker_id, original_separation_type):
    if type(seed) is not int or type(step) is not int or step < 0:
        raise ValueError("Destination seed and step must be explicit integers")
    key = [DOMAIN, seed, step, str(worker_id), str(original_separation_type)]
    digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _number(value):
    return None if value is None else float(value)


def _pension_state(ws):
    result = {}
    for prefix, obj in (("worker", ws.worker), ("state", ws)):
        for name, value in vars(obj).items():
            if "pension" not in name.lower() and "retir" not in name.lower():
                continue
            if hasattr(value, "item"):
                value = value.item()
            if value is None or isinstance(value, (bool, str, int, float)):
                result[prefix + "." + name] = value
    return result


def snapshot_state(ws):
    worker = ws.worker
    employer = getattr(worker, "current_employer_id", None)
    marker = getattr(ws, "temporary_nilf_entry_period", None)
    return {
        "status": str(worker.employment_status),
        "is_searching": bool(worker.is_searching),
        "skill_level": _number(getattr(worker, "skill_level", None)),
        "current_wage": _number(getattr(worker, "current_wage", None)),
        "previous_wage": _number(getattr(ws, "previous_wage", None)),
        "current_employer_id": None if employer is None else str(employer),
        "unemployment_months": int(getattr(worker, "unemployment_months", 0)),
        "months_unemployed": int(getattr(ws, "months_unemployed", 0)),
        "months_nilf": int(getattr(ws, "months_nilf", 0)),
        "months_employed": int(getattr(ws, "months_employed", 0)),
        "nilf_subtype": int(getattr(ws, "nilf_subtype", 0)),
        "nilf_subtype_months": int(getattr(ws, "nilf_subtype_months", 0)),
        "separation_reason": getattr(ws, "separation_reason", None),
        "lwf_retention_draw": _number(getattr(ws, "lwf_retention_draw", None)),
        "ei_eligible": bool(getattr(ws, "ei_eligible", False)),
        "ei_weeks_remaining": int(getattr(ws, "ei_weeks_remaining", 0)),
        "temporary_nilf_entry_period": None if marker is None else list(marker),
        "pension_and_retirement_state": _pension_state(ws),
    }


def partition_selected_events(env, separations):
    raw_seed = os.environ.get(SEED_FLAG)
    if raw_seed is None:
        raise ValueError("Destination routing requires its explicit evaluation seed")
    seed = int(raw_seed)
    step = int(env.current_step)
    year, month = (int(env.current_year), int(env.current_month))
    report = {
        "schema": "ordinary_separation_destination_partition_v1",
        "enabled": True,
        "seed": seed,
        "step": step,
        "year": year,
        "calendar_month": month,
        "n_share": N_SHARE,
        "events": [],
        "blocked_same_month_reentry_ids": [],
    }
    parameters = absolute_parameters()
    report["n_share"] = parameters["alpha_n"] / (parameters["alpha_u"] + parameters["alpha_n"])
    temporary_ids = set()
    seen = set()
    for worker_id, firm_id, source_type in separations:
        if worker_id in seen or source_type not in ("layoff", "quit"):
            raise ValueError(
                "A separation source record is repeated or is neither a layoff nor a quit"
            )
        seen.add(worker_id)
        ws = env.state_manager.workers[worker_id]
        if ws.worker.employment_status != "Employed":
            raise ValueError("Selected ordinary event is not employed at its source")
        age = int(ws.worker.age)
        in_scope = 25 <= age < 55
        draw = destination_draw(seed, step, worker_id, source_type) if in_scope else None
        destination = "N" if in_scope and draw < N_SHARE else "U"
        destination = "N" if in_scope and draw < report["n_share"] else "U"
        if destination == "N":
            temporary_ids.add(worker_id)
        report["events"].append(
            {
                "worker_id": str(worker_id),
                "firm_id": str(firm_id),
                "age": age,
                "original_separation_type": source_type,
                "in_scope": in_scope,
                "destination": destination,
                "destination_draw": draw,
                "before": snapshot_state(ws),
                "after": None,
                "temporary_entry_period": [year, month] if destination == "N" else None,
            }
        )
    env._destination_split_this_step = report
    env.state_manager._ordinary_temporary_nilf_ids_this_step = temporary_ids
    env._employment_route_ids_this_step["ordinary_temporary_nilf_separations"] = set(temporary_ids)


def apply_temporary_nilf_separation(state_manager, ws):
    worker = ws.worker
    if not 25 <= int(worker.age) < 55 or worker.employment_status != "Employed":
        raise ValueError("Temporary separation destination has wrong scope or origin")
    worker.employment_status = "NILF"
    worker.is_searching = False
    worker.current_employer_id = None
    worker.current_wage = 0.0
    worker.unemployment_months = 0
    ws.months_unemployed = 0
    ws.months_nilf = 0
    ws.nilf_subtype = 0
    ws.nilf_subtype_months = 0
    ws.separation_reason = None
    ws.lwf_retention_draw = None
    ws.ei_eligible = False
    ws.ei_weeks_remaining = 0
    ws.temporary_nilf_entry_period = (
        int(state_manager.current_year),
        int(state_manager.current_month),
    )
    state_manager._audit_log_nilf_entry(ws, "ordinary_temporary_separation", 0)


def finish_application(env):
    report = env._destination_split_this_step
    for event in report["events"]:
        ws = env.state_manager.workers[event["worker_id"]]
        event["after"] = snapshot_state(ws)
        if event["destination"] == "N":
            before, after = (event["before"], event["after"])
            if (
                after["status"] != "NILF"
                or after["nilf_subtype"] != 0
                or after["skill_level"] != before["skill_level"]
                or (after["pension_and_retirement_state"] != before["pension_and_retirement_state"])
                or (after["current_employer_id"] is not None)
                or (after["current_wage"] != 0.0)
                or (after["months_employed"] != 0)
                or (
                    after["temporary_nilf_entry_period"]
                    != [report["year"], report["calendar_month"]]
                )
            ):
                raise ValueError(
                    "The records of temporary separations out of the labour force are inconsistent"
                )
            firm = env.state_manager.firms.get(event["firm_id"])
            if firm is not None and event["worker_id"] in firm.employees:
                raise ValueError("Temporary NILF worker remains on the origin firm's payroll")


def tag_destination_routes(env):
    ids = env.state_manager._ordinary_temporary_nilf_ids_this_step
    env._employment_route_ids_this_step["ordinary_job_loss_separations"].difference_update(ids)
    env._employment_route_ids_this_step["ordinary_temporary_nilf_separations"] = set(ids)


def block_same_month_reentry(env, ws, worker_id):
    period = (int(env.current_year), int(env.current_month))
    if getattr(ws, "temporary_nilf_entry_period", None) != period:
        return False
    env._destination_split_this_step["blocked_same_month_reentry_ids"].append(str(worker_id))
    return True
