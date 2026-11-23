import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging
import math
import os
import sys
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional
import torch

logger = logging.getLogger(__name__)
from config.default_config import Config
from environment.labor_market_env_gpu import LaborMarketEnv
from models.dt_mappo import DTMAPPO, BR_SKILL_COL


def _compute_employment_stock_flow(status_before, status_after):
    before_ids = set(status_before)
    after_ids = set(status_after)
    shared_ids = before_ids & after_ids
    opening_employment = sum((status_before[wid] == "Employed" for wid in before_ids))
    closing_employment = sum((status_after[wid] == "Employed" for wid in after_ids))
    nonemployed_to_employed = sum(
        (status_before[wid] != "Employed" and status_after[wid] == "Employed" for wid in shared_ids)
    )
    employed_to_unemployed = sum(
        (
            status_before[wid] == "Employed" and status_after[wid] == "Unemployed"
            for wid in shared_ids
        )
    )
    employed_to_nilf = sum(
        (
            status_before[wid] == "Employed" and status_after[wid] not in ("Employed", "Unemployed")
            for wid in shared_ids
        )
    )
    employed_exits = sum((status_before[wid] == "Employed" for wid in before_ids - after_ids))
    employed_entrants = sum((status_after[wid] == "Employed" for wid in after_ids - before_ids))
    implied_closing = (
        opening_employment
        + nonemployed_to_employed
        + employed_entrants
        - employed_to_unemployed
        - employed_to_nilf
        - employed_exits
    )
    return {
        "opening_employment": int(opening_employment),
        "closing_employment": int(closing_employment),
        "nonemployed_to_employed": int(nonemployed_to_employed),
        "employed_to_unemployed": int(employed_to_unemployed),
        "employed_to_nilf": int(employed_to_nilf),
        "employed_exits": int(employed_exits),
        "employed_entrants": int(employed_entrants),
        "implied_closing_employment": int(implied_closing),
        "identity_residual": int(closing_employment - implied_closing),
    }


def _reconcile_employment_mechanisms(flow, counters):
    temporary_nilf = counters["ordinary_separations_to_temporary_nilf"]
    comparisons = {
        "matching_hires_residual": flow["nonemployed_to_employed"]
        - counters["matching_hires_nonemployed_to_employed"]
        + counters["same_month_rehires"]
        - counters["direct_nilf_to_employment"],
        "separation_residual": flow["employed_to_unemployed"]
        - counters["ordinary_separations_to_unemployment"]
        - counters["firm_exit_displacements"]
        - counters["overstaff_purge_displacements"]
        + counters["same_month_rehires"],
        "nilf_entry_residual": flow["employed_to_nilf"]
        - counters["retirements_from_employment"]
        - counters["ordinary_retirement_separations"]
        - temporary_nilf,
        "retirement_component_residual": counters["logged_retirements"]
        - counters["retirements_from_employment"]
        - counters["retirements_from_unemployment"],
        "demographic_exit_residual": flow["employed_exits"]
        - counters["demographic_employed_exits"],
        "entrant_placement_residual": flow["employed_entrants"]
        - counters["employed_entrant_placements"],
    }
    comparisons["all_mechanisms_reconciled"] = all(
        (residual == 0 for residual in comparisons.values())
    )
    return comparisons


def _capacity_budget_boundary_sources(renewals, binding_capacity):
    if type(binding_capacity) is not bool:
        raise ValueError("Capacity boundary activation must be explicit")
    sources = {
        "vacancy_withdrawals": 0,
        "withdrawn_vacancy_ids": [],
        "contraction_layoffs": [],
        "layoff_worker_ids": [],
    }
    if not binding_capacity:
        return sources
    vacancy_ids = set()
    worker_ids = set()
    for renewal in renewals:
        receipt = renewal["capacity_indexing"]
        if not isinstance(receipt, dict):
            raise ValueError("Active capacity boundary has no index receipt")
        withdrawn = receipt["capacity_budget_vacancy_withdrawals"]
        layoffs = receipt["capacity_budget_contraction_layoffs"]
        if not isinstance(withdrawn, dict):
            raise ValueError("Capacity withdrawal source receipt must be present")
        count = withdrawn["count"]
        ids = withdrawn["vacancy_ids"]
        if (
            type(count) is not int
            or count < 0
            or type(ids) is not list
            or any((type(value) is not str or not value for value in ids))
            or (len(ids) != count)
            or (len(set(ids)) != len(ids))
        ):
            raise ValueError("Invalid capacity boundary vacancy source IDs/count")
        if vacancy_ids.intersection(ids):
            raise ValueError("Repeated capacity vacancy source event")
        vacancy_ids.update(ids)
        sources["vacancy_withdrawals"] += count
        sources["withdrawn_vacancy_ids"].extend(ids)
        if type(layoffs) is not list:
            raise ValueError("Capacity boundary layoff source list must be present")
        for event in layoffs:
            if (
                not isinstance(event, dict)
                or type(event["worker_id"]) is not str
                or (not event["worker_id"])
                or (type(event["firm_id"]) is not str)
                or (not event["firm_id"])
            ):
                raise ValueError("Invalid capacity boundary layoff source event")
            wid = event["worker_id"]
            if wid in worker_ids:
                raise ValueError("Repeated capacity boundary layoff worker")
            worker_ids.add(wid)
            sources["contraction_layoffs"].append(dict(event))
            sources["layoff_worker_ids"].append(wid)
    return sources


def _reconcile_renewal_boundary(
    flow,
    status_before,
    status_after,
    vacancy_ids_before,
    vacancy_ids_after,
    employed_exits,
    employed_placements,
    capacity_sources,
    binding_capacity,
    include_identity=False,
):
    shared = set(status_before) & set(status_after)
    changed = {wid for wid in shared if status_before[wid] != status_after[wid]}
    before_vacancies, after_vacancies = (set(vacancy_ids_before), set(vacancy_ids_after))
    residuals = {
        "employed_exit_residual": flow["employed_exits"] - employed_exits,
        "employed_placement_residual": flow["employed_entrants"] - employed_placements,
        "vacancy_placement_residual": len(after_vacancies)
        - len(before_vacancies)
        + employed_placements,
    }
    unexplained = set(changed)
    if binding_capacity:
        expected = set(capacity_sources["layoff_worker_ids"])
        wrong_status = {
            wid
            for wid in expected & changed
            if (status_before[wid], status_after[wid]) != ("Employed", "Unemployed")
        }
        unexplained = changed ^ expected | wrong_status
        residuals["vacancy_placement_residual"] += capacity_sources["vacancy_withdrawals"]
        residuals["capacity_budget_layoff_residual"] = flow["employed_to_unemployed"] - len(
            expected
        )
        removed = before_vacancies - after_vacancies
        declared = set(capacity_sources["withdrawn_vacancy_ids"])
        residuals["capacity_budget_vacancy_id_residual"] = len(declared - removed) + len(
            after_vacancies - before_vacancies
        )
    residuals["continuing_worker_status_changes"] = len(unexplained)
    if include_identity:
        residuals["identity_residual"] = flow["identity_residual"]
    return {
        "residuals": residuals,
        "continuing_worker_status_change_ids": sorted(changed),
        "actual_continuing_worker_status_changes": len(changed),
        "unattributed_status_change_ids": sorted(unexplained),
    }


def get_age_label(age: int, age_bins: List[int] = None) -> str:
    if age_bins is None:
        age_bins = [15, 25, 35, 45, 55, 65, 100]
    for i in range(len(age_bins) - 1):
        if age_bins[i] <= age < age_bins[i + 1]:
            return f"{age_bins[i]}-{age_bins[i + 1] - 1}"
    return None


NATIVE_ARCHETYPE_IDS = frozenset({0, 1, 2})


def is_native_worker(worker) -> bool:
    aid = getattr(worker, "archetype_id", None)
    if aid is None:
        return True
    try:
        return int(aid) in NATIVE_ARCHETYPE_IDS
    except (TypeError, ValueError):
        return True


def collect_metrics_by_age_group(
    workers: Dict, matched_worker_ids: set = None, age_bins: List[int] = None
) -> Dict[str, Any]:
    if age_bins is None:
        age_bins = [15, 25, 35, 45, 55, 65, 100]
    if matched_worker_ids is None:
        matched_worker_ids = set()
    age_labels = []
    for i in range(len(age_bins) - 1):
        age_labels.append(f"{age_bins[i]}-{age_bins[i + 1] - 1}")
    counts = {
        label: {
            "employed": 0,
            "unemployed": 0,
            "nilf": 0,
            "unemp_duration_sum": 0,
            "unemp_count": 0,
            "unemp_duration_sum_active_search": 0,
            "unemp_count_active_search": 0,
            "matches": 0,
            "searching": 0,
        }
        for label in age_labels
    }
    for w_id, ws in workers.items():
        age = ws.worker.age
        status = ws.worker.employment_status
        label = get_age_label(age, age_bins)
        if label is None:
            continue
        if status == "Employed":
            counts[label]["employed"] += 1
        elif status == "Unemployed":
            counts[label]["unemployed"] += 1
            counts[label]["unemp_duration_sum"] += ws.worker.unemployment_months
            counts[label]["unemp_count"] += 1
            if ws.worker.is_searching:
                counts[label]["unemp_duration_sum_active_search"] += ws.worker.unemployment_months
                counts[label]["unemp_count_active_search"] += 1
        else:
            counts[label]["nilf"] += 1
        if ws.worker.is_searching:
            counts[label]["searching"] += 1
        if w_id in matched_worker_ids:
            counts[label]["matches"] += 1
    return counts


def collect_metrics_by_nilf_subtype(
    workers: Dict, matched_worker_ids: set = None
) -> Dict[str, Any]:
    if matched_worker_ids is None:
        matched_worker_ids = set()
    from environment.nilf_subtype import SUBTYPE_NAME

    labels = list(SUBTYPE_NAME.values())
    counts = {
        label: {
            "employed": 0,
            "unemployed": 0,
            "nilf": 0,
            "unemp_duration_sum": 0,
            "unemp_count": 0,
            "unemp_duration_sum_active_search": 0,
            "unemp_count_active_search": 0,
            "matches": 0,
            "searching": 0,
        }
        for label in labels
    }
    for w_id, ws in workers.items():
        status = ws.worker.employment_status
        if status == "NILF":
            subtype_int = int(getattr(ws, "nilf_subtype", 0))
            label = SUBTYPE_NAME.get(subtype_int, "none")
            counts[label]["nilf"] += 1
        elif status == "Employed":
            counts["none"]["employed"] += 1
        elif status == "Unemployed":
            counts["none"]["unemployed"] += 1
            counts["none"]["unemp_duration_sum"] += ws.worker.unemployment_months
            counts["none"]["unemp_count"] += 1
            if ws.worker.is_searching:
                counts["none"]["unemp_duration_sum_active_search"] += ws.worker.unemployment_months
                counts["none"]["unemp_count_active_search"] += 1
        if ws.worker.is_searching:
            if status == "NILF":
                subtype_int = int(getattr(ws, "nilf_subtype", 0))
                counts[SUBTYPE_NAME.get(subtype_int, "none")]["searching"] += 1
            else:
                counts["none"]["searching"] += 1
        if w_id in matched_worker_ids:
            label = "none"
            if status == "NILF":
                label = SUBTYPE_NAME.get(int(getattr(ws, "nilf_subtype", 0)), "none")
            counts[label]["matches"] += 1
    return counts


def collect_attachment_by_status_x_age(
    workers: Dict, mnl_config=None, age_bins: List[int] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if age_bins is None:
        age_bins = [15, 25, 35, 45, 55, 65, 100]
    age_labels = [f"{age_bins[i]}-{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)]
    STATUS_LABELS = ("E", "U_search", "U_no_search", "NILF_eligible", "NILF_ineligible")
    out = {s: {a: {"sum": 0.0, "count": 0} for a in age_labels} for s in STATUS_LABELS}
    from environment.nilf_subtype import (
        SUBTYPE_NONE,
        SUBTYPE_RETIREMENT,
        SUBTYPE_DISCOURAGED,
        effective_subtype_windows,
        is_search_eligible,
    )

    for _w_id, ws in workers.items():
        w = ws.worker
        status = w.employment_status
        age_label = get_age_label(w.age, age_bins)
        if age_label is None:
            continue
        att = float(getattr(w, "attachment", 1.0))
        if status == "Employed":
            label = "E"
        elif status == "Unemployed":
            label = "U_search" if bool(getattr(w, "is_searching", False)) else "U_no_search"
        else:
            subtype = int(getattr(ws, "nilf_subtype", SUBTYPE_NONE))
            months_in = int(getattr(ws, "nilf_subtype_months", 0))
            if mnl_config is not None:
                school_w, family_w, illness_w = effective_subtype_windows(mnl_config, int(w.age))
                eligible = is_search_eligible(
                    subtype=subtype,
                    months_in_subtype=months_in,
                    school_window=school_w,
                    family_window=family_w,
                    illness_window=illness_w,
                )
            elif subtype == SUBTYPE_NONE or subtype == SUBTYPE_DISCOURAGED:
                eligible = True
            elif subtype == SUBTYPE_RETIREMENT:
                eligible = False
            else:
                eligible = months_in >= 12
            label = "NILF_eligible" if eligible else "NILF_ineligible"
        out[label][age_label]["sum"] += att
        out[label][age_label]["count"] += 1
    return out


def collect_nilf_subtype_by_age(
    workers: Dict, age_bins: List[int] = None
) -> Dict[str, Dict[str, int]]:
    if age_bins is None:
        age_bins = [15, 25, 35, 45, 55, 65, 100]
    from environment.nilf_subtype import SUBTYPE_NAME

    age_labels = [f"{age_bins[i]}-{age_bins[i + 1] - 1}" for i in range(len(age_bins) - 1)]
    subtype_labels = list(SUBTYPE_NAME.values())
    grid = {a: {s: 0 for s in subtype_labels} for a in age_labels}
    for _w_id, ws in workers.items():
        if ws.worker.employment_status != "NILF":
            continue
        age_label = get_age_label(ws.worker.age, age_bins)
        if age_label is None:
            continue
        subtype = SUBTYPE_NAME.get(int(getattr(ws, "nilf_subtype", 0)), "none")
        grid[age_label][subtype] += 1
    return grid


def _skill_label_for(skill_value) -> str:
    try:
        s = float(skill_value)
    except (TypeError, ValueError):
        return "med"
    if s < 2.0:
        return "low"
    if s < 4.0:
        return "med"
    return "high"


def collect_metrics_by_skill_group(workers: Dict, matched_worker_ids: set = None) -> Dict[str, Any]:
    if matched_worker_ids is None:
        matched_worker_ids = set()
    labels = ["low", "med", "high"]
    counts = {
        label: {
            "employed": 0,
            "unemployed": 0,
            "nilf": 0,
            "unemp_duration_sum": 0,
            "unemp_count": 0,
            "unemp_duration_sum_active_search": 0,
            "unemp_count_active_search": 0,
            "matches": 0,
            "searching": 0,
        }
        for label in labels
    }
    for w_id, ws in workers.items():
        label = _skill_label_for(getattr(ws.worker, "skill_level", 2))
        status = ws.worker.employment_status
        if status == "Employed":
            counts[label]["employed"] += 1
        elif status == "Unemployed":
            counts[label]["unemployed"] += 1
            counts[label]["unemp_duration_sum"] += ws.worker.unemployment_months
            counts[label]["unemp_count"] += 1
            if ws.worker.is_searching:
                counts[label]["unemp_duration_sum_active_search"] += ws.worker.unemployment_months
                counts[label]["unemp_count_active_search"] += 1
        else:
            counts[label]["nilf"] += 1
        if ws.worker.is_searching:
            counts[label]["searching"] += 1
        if w_id in matched_worker_ids:
            counts[label]["matches"] += 1
    return counts


def analyze_archetype_performance(workers: Dict) -> Dict[str, Any]:
    ARCHETYPE_MAP = {
        0: "Native - Low Skill",
        1: "Native - Medium Skill",
        2: "Native - High Skill",
        3: "Recent Immigrant - Low Skill",
        4: "Recent Immigrant - High Skill",
        5: "Mid Immigrant - Low Skill",
        6: "Mid Immigrant - High Skill",
        7: "Established Immigrant - Low Skill",
        8: "Established Immigrant - High Skill",
    }
    archetype_metrics = {}
    for arch_id in range(9):
        archetype_metrics[arch_id] = {
            "name": ARCHETYPE_MAP.get(arch_id, f"Unknown Archetype {arch_id}"),
            "employed": 0,
            "unemployed": 0,
            "nilf": 0,
            "total_workers": 0,
            "total_wage": 0.0,
            "total_tenure": 0,
            "employed_count": 0,
            "unemp_duration_sum": 0,
            "unemp_count": 0,
        }
    for w_id, ws in workers.items():
        arch_id = ws.worker.archetype_id
        status = ws.worker.employment_status
        if arch_id not in archetype_metrics:
            continue
        metrics = archetype_metrics[arch_id]
        metrics["total_workers"] += 1
        if status == "Employed":
            metrics["employed"] += 1
            metrics["employed_count"] += 1
            metrics["total_wage"] += ws.worker.current_wage
            metrics["total_tenure"] += ws.months_employed
        elif status == "Unemployed":
            metrics["unemployed"] += 1
            metrics["unemp_duration_sum"] += ws.worker.unemployment_months
            metrics["unemp_count"] += 1
        else:
            metrics["nilf"] += 1
    results = {}
    for arch_id, metrics in archetype_metrics.items():
        total = metrics["total_workers"]
        if total == 0:
            continue
        lf = metrics["employed"] + metrics["unemployed"]
        results[arch_id] = {
            "name": metrics["name"],
            "total_workers": total,
            "employment_rate": metrics["employed"] / total,
            "unemployment_rate": metrics["unemployed"] / lf if lf > 0 else 0,
            "nilf_rate": metrics["nilf"] / total,
            "avg_wage": (
                metrics["total_wage"] / metrics["employed_count"]
                if metrics["employed_count"] > 0
                else 0.0
            ),
            "avg_tenure": (
                metrics["total_tenure"] / metrics["employed_count"]
                if metrics["employed_count"] > 0
                else 0.0
            ),
            "avg_unemp_duration": (
                metrics["unemp_duration_sum"] / metrics["unemp_count"]
                if metrics["unemp_count"] > 0
                else 0.0
            ),
        }
    return results


def collect_metrics_by_industry(workers: Dict, firms: Dict) -> Dict[int, Dict[str, float]]:
    industry_counts = defaultdict(
        lambda: {
            "employed": 0,
            "unemployed": 0,
            "separations": 0,
            "total_employed": 0,
            "vacancies": 0,
        }
    )
    for w_id, ws in workers.items():
        ind = ws.worker.industry
        if ws.worker.employment_status == "Employed":
            industry_counts[ind]["employed"] += 1
            industry_counts[ind]["total_employed"] += 1
        elif ws.worker.employment_status == "Unemployed":
            industry_counts[ind]["unemployed"] += 1
    for f_id, fs in firms.items():
        ind = fs.firm.industry
        industry_counts[ind]["vacancies"] += fs.n_vacancies
    return dict(industry_counts)


def detect_action_dims_from_checkpoint(checkpoint_path: str) -> tuple:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    worker_state = checkpoint["worker_ac"]
    firm_state = checkpoint["firm_ac"]

    def get_clean_key(state_dict, key):
        if key in state_dict:
            return state_dict[key]
        alt_key = "_orig_mod." + key
        if alt_key in state_dict:
            return state_dict[alt_key]
        return None

    worker_actor_weight = get_clean_key(worker_state, "actor.mlp.4.weight")
    n_worker_actions = worker_actor_weight.shape[0] if worker_actor_weight is not None else 40
    firm_actor_weight = get_clean_key(firm_state, "actor.mlp.4.weight")
    n_firm_actions = firm_actor_weight.shape[0] if firm_actor_weight is not None else 125
    return (n_worker_actions, n_firm_actions)


def apply_eval_config_overrides(config, args) -> None:
    if args.enable_trade_war:
        driver_file = str(args.trade_war_driver or "")
        exposure_file = str(args.trade_war_exposures or "")
        scenario = str(args.trade_war_scenario or "").strip()
        if not driver_file or not exposure_file or (not scenario):
            raise RuntimeError("A tariff run needs a driver file, an exposure file and a scenario")
        config.trade_war.enabled = True
        config.trade_war.scenario = scenario
        config.trade_war.driver_file = driver_file
        config.trade_war.exposure_file = exposure_file
    if args.net_population_projection:
        from age_renewal import enabled as _age_renewal_active

        if not _age_renewal_active():
            raise SystemExit("A net population projection needs age-cohort renewal")
    config.training.renewal_net_projection = args.net_population_projection
    config.matching.mnl_config.acceptance_temperature = 1.2
    config.training.freeze_firm_policy = bool(args.freeze_firm_policy)
    config.training.worker_res_wage_frozen_level = 2
    _mnl = config.matching.mnl_config
    _mnl.beta_wage = float(_mnl.estimated_beta_wage)
    _mnl.beta_commute = float(_mnl.estimated_beta_commute)
    if args.enable_reservation_wage_bundle:
        config.training.enable_reservation_wage_bundle = True
        config.training.reservation_decay_per_year = 0.075
        config.training.reservation_decay_compounded = True
    from population_participation import configure_participation

    configure_participation(config)


MANIFEST_ASSERT_FIELDS = [
    ("training.freeze_firm_policy", "training.freeze_firm_policy"),
    ("simulation.n_workers", "simulation.n_workers"),
    ("simulation.n_firms", "simulation.n_firms"),
    ("matching.mnl_config.acceptance_temperature", "matching.mnl_config.acceptance_temperature"),
    ("matching.mnl_config.beta_wage", "matching.mnl_config.beta_wage"),
    ("matching.mnl_config.beta_commute", "matching.mnl_config.beta_commute"),
    ("matching.mnl_config.beta_skill_match", "matching.mnl_config.beta_skill_match"),
    ("training.worker_res_wage_frozen_level", "training.worker_res_wage_frozen_level"),
    (
        "matching.mnl_config.illness_search_eligible_months_primeage",
        "matching.mnl_config.illness_search_eligible_months_primeage",
    ),
    (
        "matching.mnl_config.family_search_eligible_months_primeage",
        "matching.mnl_config.family_search_eligible_months_primeage",
    ),
    ("training.enable_reservation_wage_bundle", "training.enable_reservation_wage_bundle"),
    ("training.reservation_decay_per_year", "training.reservation_decay_per_year"),
    ("training.reservation_decay_compounded", "training.reservation_decay_compounded"),
]
_MANIFEST_FLOAT_REL_TOL = 1e-06
_MANIFEST_FLOAT_ABS_TOL = 1e-09


def _lookup_dotpath(obj, dotpath: str, default=...):
    cur = obj
    for part in dotpath.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _values_equal(manifest_val, runtime_val) -> bool:
    if manifest_val is None and runtime_val is None:
        return True
    if isinstance(manifest_val, float) or isinstance(runtime_val, float):
        try:
            mv = float(manifest_val)
            rv = float(runtime_val)
        except (TypeError, ValueError):
            return False
        if math.isnan(mv) and math.isnan(rv):
            return True
        return math.isclose(
            mv, rv, rel_tol=_MANIFEST_FLOAT_REL_TOL, abs_tol=_MANIFEST_FLOAT_ABS_TOL
        )
    return manifest_val == runtime_val


def assert_eval_config_matches_manifest(config, checkpoint_path: str) -> Dict[str, Any]:
    _ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    _manifest_path = os.path.join(_ckpt_dir, "manifest.json")
    summary = {
        "checkpoint": checkpoint_path,
        "manifest_path": _manifest_path,
        "manifest_present": False,
        "rows_ok": [],
        "rows_missing": [],
        "rows_mismatch": [],
    }
    if not os.path.exists(_manifest_path):
        return summary
    try:
        with open(_manifest_path) as _mf:
            manifest = json.load(_mf)
    except Exception:
        return summary
    summary["manifest_present"] = True
    manifest_snapshot = manifest.get("config_snapshot", {})
    _MISSING = object()
    for manifest_path, runtime_path in MANIFEST_ASSERT_FIELDS:
        m_val = _lookup_dotpath(manifest_snapshot, manifest_path, default=_MISSING)
        r_val = _lookup_dotpath(config, runtime_path, default=_MISSING)
        if m_val is _MISSING or m_val is None:
            summary["rows_missing"].append(
                {
                    "field": manifest_path,
                    "reason": "absent_or_null_in_manifest",
                    "runtime_value": None if r_val is _MISSING else r_val,
                }
            )
            continue
        if r_val is _MISSING:
            summary["rows_mismatch"].append(
                {"field": manifest_path, "manifest_value": m_val, "runtime_value": "<unset>"}
            )
            continue
        if _values_equal(m_val, r_val):
            summary["rows_ok"].append(
                {"field": manifest_path, "manifest_value": m_val, "runtime_value": r_val}
            )
        else:
            summary["rows_mismatch"].append(
                {"field": manifest_path, "manifest_value": m_val, "runtime_value": r_val}
            )
    if summary["rows_mismatch"]:
        col_w = max((len(r["field"]) for r in summary["rows_mismatch"])) + 2
        lines = [
            "Eval config does not match training-time manifest.",
            f"Checkpoint: {checkpoint_path}",
            f"Manifest:   {_manifest_path}",
            "",
            f"{'Field'.ljust(col_w)}{'Manifest'.ljust(14)}{'Current'.ljust(14)}",
            "-" * (col_w + 14 + 14 + 30),
        ]
        for row in summary["rows_mismatch"]:
            field = row["field"].ljust(col_w)
            mv = str(row["manifest_value"]).ljust(14)
            rv = str(row["runtime_value"]).ljust(14)
            lines.append(f"{field}{mv}{rv}")
        lines.append("")
        lines.append(
            "The eval would have measured a DIFFERENT model than was trained. Refusing to proceed."
        )
        raise RuntimeError("\n".join(lines))
    return summary


def run_evaluation(
    checkpoint_path: str,
    n_workers: int = 500,
    n_firms: int = 100,
    enable_reservation_wage_bundle: bool = False,
    rollout_reseed=None,
    freeze_firm_policy: bool = False,
    end_year: Optional[int] = None,
    max_episode_years: Optional[int] = None,
    enable_covid_regime: bool = False,
    covid_driver_file: Optional[str] = None,
    covid_arm_id: str = "",
    standard_driver_mode: str = "as_loaded",
    covid_seed: Optional[int] = None,
    net_population_projection: Optional[str] = None,
    enable_trade_war: bool = False,
    trade_war_scenario: str = "control",
    trade_war_driver: Optional[str] = None,
    trade_war_exposures: Optional[str] = None,
) -> Dict[str, Any]:
    n_worker_actions, n_firm_actions = detect_action_dims_from_checkpoint(checkpoint_path)
    config = Config()
    config.simulation.n_workers = n_workers
    config.simulation.n_firms = n_firms
    config.training.device = "cpu"
    _valid_standard_modes = {"as_loaded", "realized", "frozen_2019"}
    if standard_driver_mode not in _valid_standard_modes:
        raise ValueError(
            f"standard_driver_mode must be one of {sorted(_valid_standard_modes)}; got {standard_driver_mode!r}"
        )
    if enable_covid_regime and standard_driver_mode == "as_loaded":
        raise ValueError("The pandemic runs need the driver mode realized or frozen_2019")
    config.covid.enabled = bool(enable_covid_regime)
    config.covid.standard_driver_mode = str(standard_driver_mode)
    config.covid.arm_id = str(covid_arm_id or "")
    config.covid.rng_seed = int(0 if covid_seed is None else covid_seed)
    if covid_driver_file:
        config.covid.driver_file = str(covid_driver_file)
    _covid_validation_requested = bool(
        enable_covid_regime or covid_arm_id or standard_driver_mode != "as_loaded"
    )
    config.simulation.start_year = 2006
    if end_year is not None:
        config.simulation.end_year = int(end_year)
    if max_episode_years is not None:
        config.simulation.max_episode_years = int(max_episode_years)
    from types import SimpleNamespace as _SN

    _overrides = _SN(
        enable_trade_war=enable_trade_war,
        trade_war_scenario=trade_war_scenario,
        trade_war_driver=trade_war_driver,
        trade_war_exposures=trade_war_exposures,
        enable_reservation_wage_bundle=enable_reservation_wage_bundle,
        freeze_firm_policy=freeze_firm_policy,
        net_population_projection=net_population_projection,
    )
    apply_eval_config_overrides(config, _overrides)
    assert_eval_config_matches_manifest(config, checkpoint_path)
    if n_worker_actions == 25:
        config.action.worker_search_decision_options = 1
        config.action.worker_search_channel_options = 5
        config.action.worker_search_intensity_options = 5
    elif n_worker_actions == 40:
        config.action.worker_search_decision_options = 2
        config.action.worker_search_channel_options = 4
        config.action.worker_search_intensity_options = 5
    else:
        print(f"Warning: Unexpected worker action size {n_worker_actions}, using defaults")
    if enable_trade_war:
        from trade_war_stress.evaluation_settings import assert_frozen_eval_config

        config._openpop_eval_provenance = assert_frozen_eval_config(
            config, checkpoint_path, skip_manifest_assert=False
        )
    env = LaborMarketEnv(config=config)
    obs = env.reset()
    _reset_audits = getattr(env, "_fixed_count_reset_audits", [])
    if not _reset_audits or _reset_audits[-1].get("checks") != "PASS":
        raise RuntimeError("The reset lacks its record of the initial state")
    _ge_arm_flags_set = sum([False, False, False])
    if _ge_arm_flags_set > 1:
        raise ValueError(
            f"The three reservation-wage test settings exclude one another. {_ge_arm_flags_set} are set."
        )
    try:
        env.matching_mechanism.reset_episode_acceptance_tracking()
    except AttributeError:
        pass
    actual_worker_dim = env.fast_obs.state_table.worker_obs_dim
    actual_firm_dim = env.fast_obs.state_table.firm_obs_dim
    if config.network.worker_state_dim != actual_worker_dim:
        config.network.worker_state_dim = actual_worker_dim
    if config.network.firm_state_dim != actual_firm_dim:
        config.network.firm_state_dim = actual_firm_dim
    trainer = DTMAPPO(config, env)
    trainer.load(checkpoint_path)
    simulation_log = {
        "years": [],
        "unemployment_rate": [],
        "employment_rate": [],
        "participation_rate": [],
        "vacancy_rate": [],
        "aggregate_job_finding_rate": [],
        "aggregate_separation_rate": [],
        "job_finding_by_age": {},
        "unemployment_duration_by_age": {},
        "separation_by_industry": {},
        "cohort_macros_by_age": {},
        "cohort_macros_by_skill": {},
        "cohort_macros_by_nilf_subtype": {},
        "nilf_subtype_by_age_avg": {},
        "attachment_by_status_x_age_avg": {},
        "attachment_by_status_avg": {},
        "nilf_transition_counts": {},
        "nilf_transition_rates": {},
        "at_risk_pop_by_age": {},
        "worker_trajectories": [],
        "rewards_by_status": {
            "employed_high": [],
            "employed_low": [],
            "unemployed_short": [],
            "unemployed_long": [],
        },
        "firm_trajectories": [],
        "firm_rewards_by_status": {"overstaffed": [], "at_target": [], "understaffed": []},
        "vacancy_by_industry": {},
        "hiring_by_industry": {},
        "job_creation_destruction": [],
        "employment_stock_flow": [],
        "search_channel_distribution": [],
        "search_intensity_distribution": [],
        "job_finding_by_channel": {},
        "job_finding_by_intensity": {},
        "wage_strategy_distribution": [],
        "wage_dispersion_by_strategy": {},
        "commute_distance_distribution": [],
        "reservation_by_duration": {},
        "commute_tolerance_by_duration": {},
        "matching_funnel": {
            "applications": [],
            "offers": [],
            "matches": [],
            "rejections": [],
            "acceptance_rate": [],
            "offers_per_total_seeker": [],
            "share_total_offered": [],
            "accept_rate_given_offer_total": [],
            "n_no_offer": [],
            "n_rejected_all": [],
            "offers_per_unemployed_seeker": [],
            "share_unemployed_offered": [],
            "matches_per_unemployed_seeker": [],
            "accept_rate_given_offer_unemployed": [],
            "n_unemployed_offered": [],
            "n_unemployed_matched": [],
            "offers_per_employed_seeker": [],
            "n_employed_offered": [],
            "n_employed_matched": [],
            "employed_searching": [],
            "unemployed_active": [],
            "j2j_transitions": [],
            "j2j_offered": [],
            "j2j_rejected_wage": [],
            "nilf_reentry_attempts": [],
            "total_seekers": [],
            "total_vacancies": [],
            "vacancies_posted_rl": [],
            "vacancies_injected_ensure": [],
            "vacancies_injected_replenish": [],
            "vacancies_filled": [],
            "vacancies_expired": [],
            "vac_posted_growth": [],
            "vac_posted_replacement_sep": [],
            "vac_posted_replacement_j2j": [],
            "applicants_screened_out": [],
            "screening_rate": [],
            "n_applicants_total": [],
            "n_applicants_after_filters": [],
            "avg_accept_prob": [],
            "screen_reject_commute_gate": [],
            "screen_reject_industry_mismatch": [],
            "screen_reject_selectivity": [],
            "screen_reject_match_quality_floor": [],
            "screen_quality_below_firm_threshold": [],
            "screen_quality_floor_binding_band": [],
            "screen_quality_below_structural_floor": [],
            "accept_reject_wage_gap": [],
            "accept_reject_long_commute": [],
            "accept_reject_low_wage_utility": [],
            "accept_reject_poor_match_quality": [],
            "accept_reject_high_reservation": [],
            "accept_reject_age_friction": [],
            "accept_reject_random_rejection": [],
            "reject_component_wage_gap": [],
            "reject_component_long_commute": [],
            "reject_component_low_wage_utility": [],
            "reject_component_poor_match_quality": [],
            "reject_component_high_reservation": [],
            "reject_dominant_wage_gap": [],
            "reject_dominant_long_commute": [],
            "reject_dominant_low_wage_utility": [],
            "reject_dominant_poor_match_quality": [],
            "reject_dominant_high_reservation": [],
            "reject_dominant_random": [],
            "utility_diff_random_band_0": [],
            "utility_diff_random_band_1": [],
            "utility_diff_random_band_2": [],
            "utility_diff_random_band_3": [],
            "utility_diff_random_band_4": [],
            "utility_diff_random_sum": [],
            "utility_diff_random_n": [],
            "utility_diff_deterministic_sum": [],
            "utility_diff_deterministic_n": [],
            "duration_binned": {
                _bin: {
                    "n_unemployed_seekers": [],
                    "n_workers_offered": [],
                    "n_workers_matched": [],
                    "n_workers_no_offer": [],
                    "n_workers_rejected_all": [],
                    "offer_events": [],
                    "match_events": [],
                    "reject_dominant_wage_gap": [],
                    "reject_dominant_long_commute": [],
                    "reject_dominant_low_wage_utility": [],
                    "reject_dominant_poor_match_quality": [],
                    "reject_dominant_high_reservation": [],
                    "reject_dominant_random": [],
                    "match_fail_vacancy_filled": [],
                    "match_fail_firm_cap": [],
                    "match_fail_bargaining": [],
                    "bf_ratio_band_close_easy": [],
                    "bf_ratio_band_close_mid": [],
                    "bf_ratio_band_close_hard": [],
                    "bf_ratio_band_close_no": [],
                    "bf_ratio_sum": [],
                    "bf_ratio_n": [],
                    "bf_cap_posted": [],
                    "bf_cap_profit": [],
                    "bf_cap_industry": [],
                    "bf_res_sum": [],
                    "bf_fem_sum": [],
                    "bf_posted_sum": [],
                    "bf_prevwage_sum": [],
                    "bf_tier_high_M": [],
                    "bf_tier_high_F": [],
                    "bf_tier_low_M": [],
                    "bf_tier_low_F": [],
                    **{
                        f"bf_tier_{_tier}_{_band}": []
                        for _tier in ("high_M", "high_F", "low_M", "low_F")
                        for _band in ("close_easy", "close_mid", "close_hard", "close_no")
                    },
                    "bf_commute_sum": [],
                    "bf_commute_n": [],
                }
                for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf")
            },
        },
        "nash_bargaining": {
            "avg_worker_power": [],
            "avg_offered_wage": [],
            "avg_negotiated_wage": [],
            "avg_wage_premium": [],
            "worker_power_samples": [],
            "wage_premium_samples": [],
            "immigrant_power_samples": [],
            "nonimmigrant_power_samples": [],
        },
    }
    if _covid_validation_requested:
        simulation_log["covid_validation"] = {
            "metadata": env.get_covid_validation_metadata(),
            "monthly": [],
        }
    monthly_metrics = []
    age_metrics_by_year = defaultdict(list)
    age_metrics_by_year_native = defaultdict(list)
    skill_metrics_by_year = defaultdict(list)
    nilf_subtype_metrics_by_year = defaultdict(list)
    nilf_subtype_by_age_acc = []
    attachment_acc = []
    industry_metrics_by_year = defaultdict(list)
    sample_worker_ids = list(env.state_manager.workers.keys())[:100]
    worker_trajectories = {w_id: [] for w_id in sample_worker_ids}
    sample_firm_ids = list(env.state_manager.firms.keys())[:10]
    firm_trajectories = {f_id: [] for f_id in sample_firm_ids}
    hiring_by_industry_monthly = defaultdict(lambda: defaultdict(int))
    channel_counts = defaultdict(lambda: {"total": 0, "matches": 0})
    channel_counts_by_immigrant = {
        "native": defaultdict(lambda: {"total": 0, "matches": 0}),
        "immigrant": defaultdict(lambda: {"total": 0, "matches": 0}),
    }

    def _worker_immigrant_group(ws) -> str:
        _imm = getattr(ws.worker, "immigrant_status", False) if ws is not None else False
        if isinstance(_imm, str):
            _is_imm = _imm.lower() in ("true", "1", "yes", "immigrant")
        else:
            _is_imm = bool(_imm)
        return "immigrant" if _is_imm else "native"

    intensity_counts = defaultdict(lambda: {"total": 0, "matches": 0})
    search_participation = {"no_search": 0, "active": 0}
    search_participation_by_year = defaultdict(lambda: {"no_search": 0, "active": 0})
    channel_probs_accumulator = []
    intensity_probs_accumulator = []
    wage_probs_accumulator = []
    wage_strategy_counts = defaultdict(int)
    wages_by_strategy = defaultdict(list)
    firm_pbs_monthly = []
    firm_pbs_pooled = {s: {"n": 0, "sum": 0.0, "sumsq": 0.0} for s in (0, 1, 2)}
    commute_distances = []
    commute_stock = []
    use_travel_time = None
    commute_unit = None
    wages_by_industry_samples = []
    reservation_by_duration = defaultdict(list)
    total_months = config.simulation.months_per_year * (
        config.simulation.end_year - config.simulation.start_year + 1
    )
    unemployment_spells_by_immigrant = {True: [], False: []}
    commute_by_immigrant = {True: [], False: []}
    wages_by_immigrant = {True: [], False: []}
    jfr_by_age_immigrant = defaultdict(
        lambda: {True: {"found": 0, "months": 0}, False: {"found": 0, "months": 0}}
    )
    imm_spell_start = {}
    imm_previous_status = {}
    imm_worker_trajectories = {}
    commute_at_match = []
    monthly_imm_curves = {
        "emp_rate_imm": [],
        "emp_rate_non": [],
        "avg_wage_imm": [],
        "avg_wage_non": [],
        "hiring_imm": [],
        "hiring_non": [],
        "year_month": [],
    }

    def _age_to_group(age):
        if age < 25:
            return "15-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"

    def _is_immigrant(worker):
        name = getattr(worker, "archetype_name", "")
        if "mmigrant" in name:
            return True
        if getattr(worker, "immigrant", None) is not None:
            return bool(worker.immigrant)
        if getattr(worker, "immigrant_status", None) is not None:
            return bool(worker.immigrant_status)
        return False

    imm_tracked_ids = set()
    got_imm, got_non = (False, False)
    for wid, ws in env.state_manager.workers.items():
        w = ws.worker
        if 30 <= getattr(w, "age", 0) <= 35:
            is_imm = _is_immigrant(w)
            if is_imm and (not got_imm):
                imm_tracked_ids.add(wid)
                got_imm = True
            elif not is_imm and (not got_non):
                imm_tracked_ids.add(wid)
                got_non = True
        if got_imm and got_non:
            break
    for wid, ws in env.state_manager.workers.items():
        imm_previous_status[wid] = ws.worker.employment_status
    if rollout_reseed is not None:
        import torch as _torch_rr
        import numpy as _np_rr
        import random as _py_random_rr

        _torch_rr.manual_seed(int(rollout_reseed))
        try:
            _torch_rr.cuda.manual_seed_all(int(rollout_reseed))
        except Exception:
            pass
        _np_rr.random.seed(int(rollout_reseed))
        _py_random_rr.seed(int(rollout_reseed))
        if env._net_population is not None:
            env._net_population.seed(int(rollout_reseed))
    current_year = config.simulation.start_year
    year_metrics = []
    _expected_next_opening_employment = None
    _expected_next_opening_status = None
    _worker_trace = None
    if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1":
        from person_month_trace import WorkerTraceWriter

        _worker_trace = WorkerTraceWriter(os.environ["WORKER_TRACE_PATH"])
    metrics = None
    for month_idx in range(total_months):
        if env._net_population is not None:
            _reward_source_ids = tuple(env.worker_ids)
        worker_obs = obs["worker_obs"]
        firm_obs = obs["firm_obs"]
        global_state = obs["global_state"]
        worker_context = obs.get("worker_context", None)
        worker_history = obs.get("worker_history", None)
        n_workers = worker_obs.shape[0]
        n_firms = firm_obs.shape[0]
        worker_global_state = global_state.unsqueeze(0).expand(n_workers, -1)
        firm_global_state = global_state.unsqueeze(0).expand(n_firms, -1)
        with torch.no_grad():
            _retire_mask = None
            if getattr(env, "_endogenous_retirement_enabled", lambda: False)():
                _retire_mask = env._compute_retire_eligible_mask()
            worker_result = trainer.worker_ac.get_action_and_value(
                worker_obs,
                worker_global_state,
                worker_context,
                worker_history,
                deterministic=False,
                retire_eligible_mask=_retire_mask,
                skill_param=worker_obs[:, BR_SKILL_COL],
            )
            worker_action = worker_result["action"]
            if getattr(trainer.config.training, "freeze_firm_policy", False):
                n_f = firm_obs.shape[0]
                firm_action = torch.full((n_f,), 22, dtype=torch.long, device=firm_obs.device)
                firm_result = {"action": firm_action}
            else:
                firm_result = trainer.firm_ac.get_action_and_value(
                    firm_obs, firm_global_state, deterministic=False
                )
                firm_action = firm_result["action"]
            _wout = trainer.worker_ac.actor.forward(worker_obs, worker_context, worker_history)
            worker_logits = _wout[0] if isinstance(_wout, tuple) else _wout
            worker_probs = torch.softmax(worker_logits, dim=-1)
            n_actions = worker_probs.shape[-1]
            if n_actions == 40:
                n_dec, n_ch, n_int = (2, 4, 5)
            elif n_actions == 25:
                n_dec, n_ch, n_int = (1, 5, 5)
            else:
                n_dec, n_ch, n_int = (0, 0, 0)
            if n_dec > 0:
                probs_3d = worker_probs.reshape(-1, n_dec, n_ch, n_int)
                channel_probs = probs_3d.sum(dim=(1, 3)).mean(dim=0)
                intensity_probs = probs_3d.sum(dim=(1, 2)).mean(dim=0)
                channel_probs_accumulator.append(channel_probs.cpu().numpy())
                intensity_probs_accumulator.append(intensity_probs.cpu().numpy())
            firm_logits = trainer.firm_ac.actor.forward(firm_obs)
            firm_probs = torch.softmax(firm_logits, dim=-1)
            wage_strategy_probs = firm_probs.mean(dim=0)
            wage_probs_accumulator.append(wage_strategy_probs.cpu().numpy())
        action_td = obs.clone()
        action_td["worker_action"] = worker_action
        if "intent_action" in worker_result:
            action_td["worker_intent_action"] = worker_result["intent_action"]
        if _retire_mask is not None:
            action_td["worker_retire_eligible_mask"] = _retire_mask
        action_td["firm_action"] = firm_action
        worker_actions_decoded = env._decode_worker_actions_gpu(worker_action)
        active_unemp_searchers = set()
        for w_id, action in worker_actions_decoded.items():
            ws = env.state_manager.workers.get(w_id)
            if not (ws and ws.worker.employment_status == "Unemployed"):
                continue
            intensity = action.get("search_intensity", 0)
            is_active_searcher = action.get("search_decision", 0) == 1 and intensity > 0
            _yr = env.state_manager.current_year
            if is_active_searcher:
                search_participation["active"] += 1
                search_participation_by_year[_yr]["active"] += 1
                channel = action.get("search_channel", 0)
                channel_counts[channel]["total"] += 1
                intensity_counts[intensity]["total"] += 1
                active_unemp_searchers.add(w_id)
                _grp = _worker_immigrant_group(ws)
                channel_counts_by_immigrant[_grp][channel]["total"] += 1
            else:
                search_participation["no_search"] += 1
                search_participation_by_year[_yr]["no_search"] += 1
        firm_actions_decoded = env._decode_firm_actions_gpu(firm_action)
        for f_id, action in firm_actions_decoded.items():
            wage_strategy = action.get("wage_strategy", 2)
            wage_strategy_counts[wage_strategy] += 1
        searching_before = set()
        status_before = {}
        industry_before = {}
        for w_id, ws in env.state_manager.workers.items():
            status_before[w_id] = ws.worker.employment_status
            industry_before[w_id] = ws.worker.industry
            if ws.worker.is_searching or ws.worker.employment_status == "Unemployed":
                searching_before.add(w_id)
        _opening_employment_now = sum((status == "Employed" for status in status_before.values()))
        if (
            _expected_next_opening_employment is not None
            and _opening_employment_now != _expected_next_opening_employment
        ):
            raise AssertionError(
                f"cross-month employment continuity failed: expected={_expected_next_opening_employment} actual={_opening_employment_now} month_index={month_idx}"
            )
        if (
            _expected_next_opening_status is not None
            and status_before != _expected_next_opening_status
        ):
            _expected_ids = set(_expected_next_opening_status)
            _actual_ids = set(status_before)
            _status_mismatches = [
                wid
                for wid in sorted(_expected_ids & _actual_ids)
                if _expected_next_opening_status[wid] != status_before[wid]
            ]
            raise AssertionError(
                f"cross-month worker-status continuity failed: missing={sorted(_expected_ids - _actual_ids)[:10]} added={sorted(_actual_ids - _expected_ids)[:10]} status_changed={_status_mismatches[:10]} month_index={month_idx}"
            )
        _period_year = int(env.state_manager.current_year)
        _period_month = int(env.state_manager.current_month)
        _retire_before = int(getattr(env, "_endogenous_retire_count_running", 0))
        _nilf_log_before = len(getattr(env.state_manager, "nilf_entry_log", []))
        _renewals_before = len(getattr(getattr(env, "_net_population", None), "renewals", []))
        if _worker_trace is not None:
            _worker_trace.begin_month(env, month_idx, _period_year, _period_month, worker_action)
        if _worker_trace is not None:
            obs = _worker_trace.step(env, action_td)
        else:
            obs = env.step(action_td)
        _status_after = {
            w_id: ws.worker.employment_status for w_id, ws in env.state_manager.workers.items()
        }
        if _worker_trace is not None:
            _worker_trace.after_market(env)
        _employment_flow = _compute_employment_stock_flow(status_before, _status_after)
        _employment_flow.update(
            {
                "month": int(month_idx),
                "year": int(_period_year),
                "calendar_month": int(_period_month),
            }
        )
        _matching_stats = getattr(env, "last_matching_stats", {}) or {}
        _state_stats = getattr(env.state_manager, "stats", {}) or {}
        _new_nilf_rows = list(getattr(env.state_manager, "nilf_entry_log", []))[_nilf_log_before:]
        _employed_nilf_sources = {
            "employed",
            "employed_retirement",
            "employed_retirement_gate",
            "endogenous_retirement_action",
        }
        _logged_e_to_nilf = sum(
            (row.get("source_state") in _employed_nilf_sources for row in _new_nilf_rows)
        )
        _logged_retirements = sum(
            (row.get("source_state") == "endogenous_retirement_action" for row in _new_nilf_rows)
        )
        _demographic_employed_exits = 0
        _employed_entrant_placements = 0
        _renewals = list(getattr(getattr(env, "_net_population", None), "renewals", []))
        for _renewal in _renewals[_renewals_before:]:
            _demographic_employed_exits += sum(
                (status_before.get(wid) == "Employed" for wid in _renewal.get("exits", []))
            )
            _placement = _renewal.get("employed_entrant_placement") or {}
            _employed_entrant_placements += int(_placement.get("placed", 0))
        _mechanism_counters = {
            "matching_hires_nonemployed_to_employed": int(
                _matching_stats.get("committed_hires_u2e", 0)
            ),
            "direct_nilf_to_employment": int(_matching_stats.get("direct_nilf_to_employment", 0)),
            "same_month_rehires": int(_matching_stats.get("same_month_rehires", 0)),
            "separations_employed_to_unemployed": int(
                _state_stats.get("separations_this_month", 0)
            ),
            "ordinary_separations_to_unemployment": int(
                _matching_stats.get("ordinary_separations_to_unemployment", 0)
            ),
            "ordinary_retirement_separations": int(
                _matching_stats.get("ordinary_retirement_separations", 0)
            ),
            "firm_exit_displacements": int(_matching_stats.get("firm_exit_displacements", 0)),
            "overstaff_purge_displacements": int(
                _matching_stats.get("overstaff_purge_displacements", 0)
            ),
            "logged_employed_to_nilf": int(_logged_e_to_nilf),
            "logged_retirements": int(_logged_retirements),
            "retirements_from_employment": int(
                _matching_stats.get("retirements_from_employment", 0)
            ),
            "retirements_from_unemployment": int(
                _matching_stats.get("retirements_from_unemployment", 0)
            ),
            "retirement_counter_delta": int(
                getattr(env, "_endogenous_retire_count_running", 0) - _retire_before
            ),
            "demographic_employed_exits": int(_demographic_employed_exits),
            "employed_entrant_placements": int(_employed_entrant_placements),
        }
        _temporary_nilf = _matching_stats.get("ordinary_separations_to_temporary_nilf")
        if type(_temporary_nilf) is not int or _temporary_nilf < 0:
            raise ValueError("Active destination split requires an observed temporary-NILF counter")
        _mechanism_counters["ordinary_separations_to_temporary_nilf"] = _temporary_nilf
        _reconciliation = _reconcile_employment_mechanisms(_employment_flow, _mechanism_counters)
        _employment_flow["mechanism_counters"] = _mechanism_counters
        _employment_flow["mechanism_reconciliation"] = _reconciliation
        if not _reconciliation["all_mechanisms_reconciled"]:
            raise AssertionError(f"employment mechanism reconciliation failed: {_employment_flow}")
        simulation_log["employment_stock_flow"].append(_employment_flow)
        if _covid_validation_requested and _period_year >= int(config.covid.report_start_year):
            _validation_snapshot = env.get_covid_validation_snapshot(_period_year, _period_month)
            simulation_log["covid_validation"]["monthly"].append(_validation_snapshot)
        if hasattr(env, "last_firm_profit_raw") and hasattr(env, "last_firm_wage_strategy"):
            _pbs_step = {s: {"n": 0, "sum": 0.0, "sumsq": 0.0} for s in (0, 1, 2)}
            for _fid, _s in env.last_firm_wage_strategy.items():
                _p = env.last_firm_profit_raw.get(_fid)
                if _p is None or _s not in _pbs_step:
                    continue
                _cell = _pbs_step[_s]
                _cell["n"] += 1
                _cell["sum"] += float(_p)
                _cell["sumsq"] += float(_p) * float(_p)
            _n_hi, _n_lo = (_pbs_step[2]["n"], _pbs_step[0]["n"])
            _gap = (
                _pbs_step[2]["sum"] / _n_hi - _pbs_step[0]["sum"] / _n_lo
                if _n_hi > 0 and _n_lo > 0
                else None
            )
            firm_pbs_monthly.append(
                {"month_idx": month_idx, "per_strategy": _pbs_step, "gap_high_low": _gap}
            )
            for _s in (0, 1, 2):
                firm_pbs_pooled[_s]["n"] += _pbs_step[_s]["n"]
                firm_pbs_pooled[_s]["sum"] += _pbs_step[_s]["sum"]
                firm_pbs_pooled[_s]["sumsq"] += _pbs_step[_s]["sumsq"]
        for worker_id, ws in env.state_manager.workers.items():
            w = ws.worker
            is_imm = _is_immigrant(w)
            ag = _age_to_group(getattr(w, "age", 35))
            current_status = w.employment_status
            prev = imm_previous_status.get(worker_id, current_status)
            if prev != "Unemployed" and current_status == "Unemployed":
                imm_spell_start[worker_id] = month_idx
            if prev == "Unemployed" and current_status != "Unemployed":
                if worker_id in imm_spell_start:
                    duration = month_idx - imm_spell_start[worker_id]
                    unemployment_spells_by_immigrant[is_imm].append(duration)
                    del imm_spell_start[worker_id]
            if current_status == "Employed":
                ct = getattr(w, "commute_time_min", None)
                if ct is not None and ct > 0:
                    commute_by_immigrant[is_imm].append(float(ct))
                wage = getattr(w, "current_wage", 0)
                if wage and wage > 0:
                    wages_by_immigrant[is_imm].append(float(wage))
            if prev == "Unemployed":
                jfr_by_age_immigrant[ag][is_imm]["months"] += 1
                if current_status == "Employed":
                    jfr_by_age_immigrant[ag][is_imm]["found"] += 1
                    ct = getattr(w, "commute_time_min", None)
                    if ct is not None and ct > 0:
                        if worker_id in imm_spell_start:
                            unemp_duration = month_idx - imm_spell_start[worker_id]
                        else:
                            unemp_duration = getattr(w, "unemployment_months", 0)
                        commute_at_match.append(
                            {
                                "commute": float(ct),
                                "unemp_months": max(0, unemp_duration),
                                "immigrant": is_imm,
                                "age": getattr(w, "age", 35),
                                "skill": getattr(w, "skill_level", 2),
                                "home_da": getattr(w, "da_id", None),
                            }
                        )
            if worker_id in imm_tracked_ids:
                if worker_id not in imm_worker_trajectories:
                    imm_worker_trajectories[worker_id] = {
                        "status": [],
                        "wage": [],
                        "immigrant": is_imm,
                        "age": getattr(w, "age", 0),
                        "credential": getattr(w, "foreign_credential_recognition", "N/A"),
                        "language": getattr(w, "language_proficiency_english", "N/A"),
                    }
                s_code = (
                    0
                    if current_status == "Employed"
                    else 1 if current_status == "Unemployed" else 2
                )
                imm_worker_trajectories[worker_id]["status"].append(s_code)
                imm_worker_trajectories[worker_id]["wage"].append(
                    float(w.current_wage)
                    if current_status == "Employed" and hasattr(w, "current_wage")
                    else 0
                )
            imm_previous_status[worker_id] = current_status
        imm_employed = 0
        imm_total = 0
        imm_wage_sum = 0
        imm_wage_n = 0
        non_employed = 0
        non_total = 0
        non_wage_sum = 0
        non_wage_n = 0
        imm_hired = 0
        non_hired = 0
        for wid, ws in env.state_manager.workers.items():
            w = ws.worker
            is_imm = _is_immigrant(w)
            if is_imm:
                imm_total += 1
                if w.employment_status == "Employed":
                    imm_employed += 1
                    wage = getattr(w, "current_wage", 0)
                    if wage and wage > 0:
                        imm_wage_sum += wage
                        imm_wage_n += 1
            else:
                non_total += 1
                if w.employment_status == "Employed":
                    non_employed += 1
                    wage = getattr(w, "current_wage", 0)
                    if wage and wage > 0:
                        non_wage_sum += wage
                        non_wage_n += 1
            prev = imm_previous_status.get(wid, w.employment_status)
            if prev != "Employed" and w.employment_status == "Employed":
                if is_imm:
                    imm_hired += 1
                else:
                    non_hired += 1
        monthly_imm_curves["emp_rate_imm"].append(imm_employed / max(imm_total, 1) * 100)
        monthly_imm_curves["emp_rate_non"].append(non_employed / max(non_total, 1) * 100)
        monthly_imm_curves["avg_wage_imm"].append(imm_wage_sum / max(imm_wage_n, 1))
        monthly_imm_curves["avg_wage_non"].append(non_wage_sum / max(non_wage_n, 1))
        monthly_imm_curves["hiring_imm"].append(imm_hired)
        monthly_imm_curves["hiring_non"].append(non_hired)
        _renewal_curve_i = len(monthly_imm_curves["year_month"])
        _renewal_curve_year = int(env.start_year) + _renewal_curve_i // 12
        _renewal_curve_month = 1 + _renewal_curve_i % 12
        monthly_imm_curves["year_month"].append(f"{_renewal_curve_year}-{_renewal_curve_month:02d}")
        if hasattr(env, "last_matching_stats") and env.last_matching_stats:
            step_commutes = env.last_matching_stats.get("commute_values", [])
            commute_distances.extend(step_commutes)
            if use_travel_time is None:
                use_travel_time = env.last_matching_stats.get("use_travel_time", False)
                commute_unit = env.last_matching_stats.get("commute_unit", "km")
        for w_id, ws in env.state_manager.workers.items():
            if ws.worker.employment_status == "Employed" and ws.worker.commute_time_min is not None:
                commute_stock.append(ws.worker.commute_time_min)
        if hasattr(env, "last_matching_stats") and env.last_matching_stats:
            stats = env.last_matching_stats
            simulation_log["matching_funnel"]["applications"].append(stats.get("applications"))
            simulation_log["matching_funnel"]["offers"].append(stats.get("offers"))
            simulation_log["matching_funnel"]["matches"].append(stats.get("matches"))
            simulation_log["matching_funnel"]["rejections"].append(stats.get("rejections", 0))
            simulation_log["matching_funnel"]["acceptance_rate"].append(
                stats.get("acceptance_rate", 0)
            )
            simulation_log["matching_funnel"].setdefault("avg_mnl_accept_prob", []).append(
                stats.get("avg_mnl_accept_prob", stats.get("avg_accept_prob", 0.0))
            )
            simulation_log["matching_funnel"].setdefault(
                "avg_attachment_adjusted_match_prob", []
            ).append(
                stats.get("avg_attachment_adjusted_match_prob", stats.get("avg_accept_prob", 0.0))
            )
            simulation_log["matching_funnel"].setdefault(
                "attachment_conversion_failures", []
            ).append(int(stats.get("attachment_conversion_failures", 0)))
            simulation_log["matching_funnel"]["offers_per_total_seeker"].append(
                stats.get("offers_per_job_seeker", 0.0)
            )
            simulation_log["matching_funnel"]["share_total_offered"].append(
                stats.get("share_offered", 0.0)
            )
            simulation_log["matching_funnel"]["accept_rate_given_offer_total"].append(
                stats.get("accept_rate_given_offer", 0.0)
            )
            simulation_log["matching_funnel"]["n_no_offer"].append(
                stats.get("n_workers_no_offer", 0)
            )
            simulation_log["matching_funnel"]["n_rejected_all"].append(
                stats.get("n_workers_rejected_all", 0)
            )
            simulation_log["matching_funnel"]["offers_per_unemployed_seeker"].append(
                stats.get("offers_per_unemployed_seeker", 0.0)
            )
            simulation_log["matching_funnel"]["share_unemployed_offered"].append(
                stats.get("share_unemployed_offered", 0.0)
            )
            simulation_log["matching_funnel"]["matches_per_unemployed_seeker"].append(
                stats.get("matches_per_unemployed_seeker", 0.0)
            )
            simulation_log["matching_funnel"]["accept_rate_given_offer_unemployed"].append(
                stats.get("accept_rate_given_offer_unemployed", 0.0)
            )
            simulation_log["matching_funnel"]["n_unemployed_offered"].append(
                stats.get("n_unemployed_offered", 0)
            )
            simulation_log["matching_funnel"]["n_unemployed_matched"].append(
                stats.get("n_unemployed_matched", 0)
            )
            simulation_log["matching_funnel"]["offers_per_employed_seeker"].append(
                stats.get("offers_per_employed_seeker", 0.0)
            )
            simulation_log["matching_funnel"]["n_employed_offered"].append(
                stats.get("n_employed_offered", 0)
            )
            simulation_log["matching_funnel"]["n_employed_matched"].append(
                stats.get("n_employed_matched", 0)
            )
            emp_searching = stats.get("employed_searching")
            unemp_active = stats.get("unemployed_active")
            j2j = stats.get("j2j_transitions")
            nilf_reentry = stats.get("nilf_reentry_attempts")
            simulation_log["matching_funnel"]["employed_searching"].append(emp_searching)
            simulation_log["matching_funnel"]["unemployed_active"].append(unemp_active)
            simulation_log["matching_funnel"]["j2j_transitions"].append(j2j)
            simulation_log["matching_funnel"]["j2j_offered"].append(stats.get("j2j_offered"))
            simulation_log["matching_funnel"]["j2j_rejected_wage"].append(
                stats.get("j2j_rejected_wage")
            )
            simulation_log["matching_funnel"]["nilf_reentry_attempts"].append(nilf_reentry)
            _total_seekers = (
                None
                if unemp_active is None or emp_searching is None or nilf_reentry is None
                else unemp_active + emp_searching + nilf_reentry
            )
            simulation_log["matching_funnel"]["total_seekers"].append(_total_seekers)
            simulation_log["matching_funnel"]["total_vacancies"].append(
                stats.get("total_vacancies")
            )
            simulation_log["matching_funnel"]["vacancies_posted_rl"].append(
                stats.get("vacancies_posted_rl")
            )
            simulation_log["matching_funnel"]["vacancies_injected_ensure"].append(
                stats.get("vacancies_injected_ensure")
            )
            simulation_log["matching_funnel"]["vacancies_injected_replenish"].append(
                stats.get("vacancies_injected_replenish")
            )
            simulation_log["matching_funnel"]["vacancies_filled"].append(
                stats.get("vacancies_filled")
            )
            simulation_log["matching_funnel"]["vacancies_expired"].append(
                stats.get("vacancies_expired")
            )
            simulation_log["matching_funnel"]["vac_posted_growth"].append(
                stats.get("vac_posted_growth")
            )
            simulation_log["matching_funnel"]["vac_posted_replacement_sep"].append(
                stats.get("vac_posted_replacement_sep")
            )
            simulation_log["matching_funnel"]["vac_posted_replacement_j2j"].append(
                stats.get("vac_posted_replacement_j2j")
            )
            simulation_log["matching_funnel"].setdefault("vacancy_opening_balance", []).append(
                stats.get("vacancy_opening_balance")
            )
            simulation_log["matching_funnel"].setdefault("vacancies_withdrawn", []).append(
                stats.get("vacancies_withdrawn")
            )
            simulation_log["matching_funnel"].setdefault(
                "capacity_budget_vacancies_withdrawn", []
            ).append(stats.get("capacity_budget_vacancies_withdrawn"))
            simulation_log["matching_funnel"].setdefault("vacancies_exit_purged", []).append(
                stats.get("vacancies_exit_purged")
            )
            simulation_log["matching_funnel"].setdefault("vacancies_filled_u2e", []).append(
                stats.get("vacancies_filled_u2e")
            )
            simulation_log["matching_funnel"].setdefault("vacancies_filled_j2j", []).append(
                stats.get("vacancies_filled_j2j")
            )
            simulation_log["matching_funnel"].setdefault(
                "vacancies_filled_direct_nilf_reentry", []
            ).append(stats.get("vacancies_filled_direct_nilf_reentry"))
            simulation_log["matching_funnel"].setdefault("vacancy_event_ledger", []).append(
                stats.get("vacancy_event_ledger")
            )
            simulation_log["matching_funnel"].setdefault("committed_hires", []).append(
                stats.get("committed_hires")
            )
            simulation_log["matching_funnel"].setdefault("committed_hires_u2e", []).append(
                stats.get("committed_hires_u2e")
            )
            simulation_log["matching_funnel"].setdefault("committed_hires_j2j", []).append(
                stats.get("committed_hires_j2j")
            )
            for _k in (
                "applicants_screened_out",
                "screening_rate",
                "n_applicants_total",
                "n_applicants_after_filters",
                "avg_accept_prob",
                "screen_reject_commute_gate",
                "screen_reject_industry_mismatch",
                "screen_reject_selectivity",
                "screen_reject_match_quality_floor",
                "screen_quality_below_firm_threshold",
                "screen_quality_floor_binding_band",
                "screen_quality_below_structural_floor",
                "accept_reject_wage_gap",
                "accept_reject_long_commute",
                "accept_reject_low_wage_utility",
                "accept_reject_poor_match_quality",
                "accept_reject_high_reservation",
                "accept_reject_age_friction",
                "accept_reject_random_rejection",
                "reject_component_wage_gap",
                "reject_component_long_commute",
                "reject_component_low_wage_utility",
                "reject_component_poor_match_quality",
                "reject_component_high_reservation",
                "reject_dominant_wage_gap",
                "reject_dominant_long_commute",
                "reject_dominant_low_wage_utility",
                "reject_dominant_poor_match_quality",
                "reject_dominant_high_reservation",
                "reject_dominant_random",
                "utility_diff_random_band_0",
                "utility_diff_random_band_1",
                "utility_diff_random_band_2",
                "utility_diff_random_band_3",
                "utility_diff_random_band_4",
                "utility_diff_random_sum",
                "utility_diff_random_n",
                "utility_diff_deterministic_sum",
                "utility_diff_deterministic_n",
                "app_same_industry",
                "app_same_sector_diff_industry",
                "app_diff_sector",
                "screen_pass_same_industry",
                "screen_pass_same_sector_diff_industry",
                "screen_pass_diff_sector",
                "screen_fail_same_industry",
                "screen_fail_same_sector_diff_industry",
                "screen_fail_diff_sector",
                "offer_same_industry",
                "offer_same_sector_diff_industry",
                "offer_diff_sector",
                "match_same_industry",
                "match_same_sector_diff_industry",
                "match_diff_sector",
                "app_same_industry_dur_0_6",
                "app_same_industry_dur_6_12",
                "app_same_industry_dur_12_inf",
                "app_same_sector_diff_industry_dur_0_6",
                "app_same_sector_diff_industry_dur_6_12",
                "app_same_sector_diff_industry_dur_12_inf",
                "app_diff_sector_dur_0_6",
                "app_diff_sector_dur_6_12",
                "app_diff_sector_dur_12_inf",
                "screen_pass_same_industry_dur_0_6",
                "screen_pass_same_industry_dur_6_12",
                "screen_pass_same_industry_dur_12_inf",
                "screen_pass_same_sector_diff_industry_dur_0_6",
                "screen_pass_same_sector_diff_industry_dur_6_12",
                "screen_pass_same_sector_diff_industry_dur_12_inf",
                "screen_pass_diff_sector_dur_0_6",
                "screen_pass_diff_sector_dur_6_12",
                "screen_pass_diff_sector_dur_12_inf",
                "screen_fail_same_industry_dur_0_6",
                "screen_fail_same_industry_dur_6_12",
                "screen_fail_same_industry_dur_12_inf",
                "screen_fail_same_sector_diff_industry_dur_0_6",
                "screen_fail_same_sector_diff_industry_dur_6_12",
                "screen_fail_same_sector_diff_industry_dur_12_inf",
                "screen_fail_diff_sector_dur_0_6",
                "screen_fail_diff_sector_dur_6_12",
                "screen_fail_diff_sector_dur_12_inf",
            ):
                simulation_log["matching_funnel"].setdefault(_k, []).append(stats.get(_k, 0))
            _dur_root = simulation_log["matching_funnel"].setdefault(
                "duration_binned", {b: {} for b in ("dur_0_6", "dur_6_12", "dur_12_inf")}
            )
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                _slot = _dur_root.setdefault(_bin, {})
                _slot.setdefault("n_unemployed_seekers", []).append(
                    int(stats.get(f"n_unemployed_seekers_{_bin}", 0))
                )
                _slot.setdefault("n_workers_offered", []).append(
                    int(stats.get(f"n_workers_offered_{_bin}", 0))
                )
                _slot.setdefault("n_workers_matched", []).append(
                    int(stats.get(f"n_workers_matched_{_bin}", 0))
                )
                _slot.setdefault("n_workers_no_offer", []).append(
                    int(stats.get(f"n_workers_no_offer_{_bin}", 0))
                )
                _slot.setdefault("n_workers_rejected_all", []).append(
                    int(stats.get(f"n_workers_rejected_all_{_bin}", 0))
                )
                _slot.setdefault("offer_events", []).append(int(stats.get(f"offer_{_bin}", 0)))
                _slot.setdefault("match_events", []).append(int(stats.get(f"match_{_bin}", 0)))
                for _reason in (
                    "wage_gap",
                    "long_commute",
                    "low_wage_utility",
                    "poor_match_quality",
                    "high_reservation",
                    "random",
                ):
                    _slot.setdefault(f"reject_dominant_{_reason}", []).append(
                        int(stats.get(f"reject_dominant_{_reason}_{_bin}", 0))
                    )
                for _fail in ("vacancy_filled", "firm_cap", "bargaining"):
                    _slot.setdefault(f"match_fail_{_fail}", []).append(
                        int(stats.get(f"match_fail_{_fail}_{_bin}", 0))
                    )
                for _band in ("close_easy", "close_mid", "close_hard", "close_no"):
                    _slot.setdefault(f"bf_ratio_band_{_band}", []).append(
                        int(stats.get(f"bf_ratio_band_{_band}_{_bin}", 0))
                    )
                _slot.setdefault("bf_ratio_sum", []).append(
                    float(stats.get(f"bf_ratio_sum_{_bin}", 0.0))
                )
                _slot.setdefault("bf_ratio_n", []).append(int(stats.get(f"bf_ratio_n_{_bin}", 0)))
                for _cap in ("posted", "profit", "industry"):
                    _slot.setdefault(f"bf_cap_{_cap}", []).append(
                        int(stats.get(f"bf_cap_{_cap}_{_bin}", 0))
                    )
                for _lvl in ("bf_res_sum", "bf_fem_sum", "bf_posted_sum", "bf_prevwage_sum"):
                    _slot.setdefault(_lvl, []).append(float(stats.get(f"{_lvl}_{_bin}", 0.0)))
                for _tier in ("high_M", "high_F", "low_M", "low_F"):
                    _slot.setdefault(f"bf_tier_{_tier}", []).append(
                        int(stats.get(f"bf_tier_{_tier}_{_bin}", 0))
                    )
                for _tier in ("high_M", "high_F", "low_M", "low_F"):
                    for _band in ("close_easy", "close_mid", "close_hard", "close_no"):
                        _slot.setdefault(f"bf_tier_{_tier}_{_band}", []).append(
                            int(stats.get(f"bf_tier_{_tier}_{_band}_{_bin}", 0))
                        )
                _slot.setdefault("bf_commute_sum", []).append(
                    float(stats.get(f"bf_commute_sum_{_bin}", 0.0))
                )
                _slot.setdefault("bf_commute_n", []).append(
                    int(stats.get(f"bf_commute_n_{_bin}", 0))
                )
            for _cap in ("posted", "profit", "industry"):
                simulation_log["matching_funnel"].setdefault(f"bf_cap_{_cap}_aggregate", []).append(
                    int(stats.get(f"bf_cap_{_cap}", 0))
                )
            simulation_log["matching_funnel"].setdefault(
                "bargaining_failures_aggregate", []
            ).append(int(stats.get("bargaining_failures", 0)))
            nash_stats = stats.get("nash_bargaining", {})
            if nash_stats:
                simulation_log["nash_bargaining"]["avg_worker_power"].append(
                    nash_stats.get("avg_worker_power", 0)
                )
                simulation_log["nash_bargaining"]["avg_offered_wage"].append(
                    nash_stats.get("avg_offered_wage", 0)
                )
                simulation_log["nash_bargaining"]["avg_negotiated_wage"].append(
                    nash_stats.get("avg_negotiated_wage", 0)
                )
                simulation_log["nash_bargaining"]["avg_wage_premium"].append(
                    nash_stats.get("avg_wage_premium", 0)
                )
                simulation_log["nash_bargaining"]["worker_power_samples"].extend(
                    nash_stats.get("worker_power_sample", [])
                )
                simulation_log["nash_bargaining"]["wage_premium_samples"].extend(
                    nash_stats.get("wage_premium_sample", [])
                )
        matched_worker_ids = set()
        for w_id in searching_before:
            ws = env.state_manager.workers.get(w_id)
            if ws and ws.worker.employment_status == "Employed":
                if status_before.get(w_id) != "Employed":
                    matched_worker_ids.add(w_id)
        unemp_rate = metrics.get("unemployment_rate", 0.06) if metrics is not None else 0.06
        for w_id in matched_worker_ids:
            ws = env.state_manager.workers.get(w_id)
            if ws:
                w = ws.worker
                is_imm = _is_immigrant(w)
                power = env.matching_mechanism.compute_bargaining_power(
                    skill_level=getattr(w, "skill_level", 2),
                    education=getattr(w, "education", 3),
                    unemployment_months=getattr(w, "unemployment_months", 0),
                    unemployment_rate=unemp_rate,
                )
                if is_imm:
                    simulation_log["nash_bargaining"]["immigrant_power_samples"].append(power)
                else:
                    simulation_log["nash_bargaining"]["nonimmigrant_power_samples"].append(power)
        for w_id in matched_worker_ids:
            if w_id not in active_unemp_searchers:
                continue
            action = worker_actions_decoded.get(w_id, {})
            channel = action.get("search_channel", 0)
            intensity = action.get("search_intensity", 2)
            channel_counts[channel]["matches"] += 1
            intensity_counts[intensity]["matches"] += 1
            _ws_match = env.state_manager.workers.get(w_id)
            _grp_match = _worker_immigrant_group(_ws_match)
            channel_counts_by_immigrant[_grp_match][channel]["matches"] += 1
        separations_by_industry = defaultdict(int)
        for w_id, ws in env.state_manager.workers.items():
            if (
                status_before.get(w_id) == "Employed"
                and ws.worker.employment_status == "Unemployed"
            ):
                ind = industry_before.get(w_id, 0)
                separations_by_industry[ind] += 1
        metrics = env.get_validation_metrics()
        metrics["year"] = env.state_manager.current_year
        metrics["month"] = env.state_manager.current_month
        metrics["separations_by_industry"] = dict(separations_by_industry)
        monthly_metrics.append(metrics)
        year_metrics.append(metrics)
        age_counts = collect_metrics_by_age_group(
            env.state_manager.workers, matched_worker_ids=matched_worker_ids
        )
        age_metrics_by_year[env.state_manager.current_year].append(age_counts)
        _native_workers = {
            wid: ws for wid, ws in env.state_manager.workers.items() if is_native_worker(ws.worker)
        }
        age_counts_native = collect_metrics_by_age_group(
            _native_workers, matched_worker_ids=matched_worker_ids
        )
        age_metrics_by_year_native[env.state_manager.current_year].append(age_counts_native)
        skill_counts = collect_metrics_by_skill_group(
            env.state_manager.workers, matched_worker_ids=matched_worker_ids
        )
        skill_metrics_by_year[env.state_manager.current_year].append(skill_counts)
        nilf_subtype_counts = collect_metrics_by_nilf_subtype(
            env.state_manager.workers, matched_worker_ids=matched_worker_ids
        )
        nilf_subtype_metrics_by_year[env.state_manager.current_year].append(nilf_subtype_counts)
        nilf_subtype_by_age_acc.append(collect_nilf_subtype_by_age(env.state_manager.workers))
        attachment_acc.append(
            collect_attachment_by_status_x_age(env.state_manager.workers, mnl_config=env.config.mnl)
        )
        industry_counts = collect_metrics_by_industry(
            env.state_manager.workers, env.state_manager.firms
        )
        industry_metrics_by_year[env.state_manager.current_year].append(industry_counts)
        for w_id in sample_worker_ids:
            ws = env.state_manager.workers.get(w_id)
            if ws:
                worker_trajectories[w_id].append(
                    {
                        "month": month_idx,
                        "year": env.state_manager.current_year,
                        "status": ws.worker.employment_status,
                        "wage": ws.worker.current_wage,
                        "unemployment_months": ws.worker.unemployment_months,
                        "industry": ws.worker.industry,
                        "age": ws.worker.age,
                    }
                )
                unemp_months = ws.worker.unemployment_months
                if ws.worker.employment_status == "Unemployed":
                    duration_bucket = unemp_months // 3 * 3
                    duration_bucket = min(duration_bucket, 12)
                    _cw = float(getattr(ws.worker, "current_wage", None) or 0.0)
                    res_w = env.matching_mechanism.compute_reservation_wage(
                        current_wage=_cw,
                        skill_level=int(getattr(ws.worker, "skill_level", 2)),
                        unemployment_months=int(unemp_months),
                        res_wage_action=int(getattr(ws, "last_res_wage_action", 2)),
                        previous_wage=float(getattr(ws, "previous_wage", _cw) or _cw),
                    )
                    reservation_by_duration[duration_bucket].append(float(res_w))
        for f_id in sample_firm_ids:
            fs = env.state_manager.firms.get(f_id)
            if fs:
                firm_trajectories[f_id].append(
                    {
                        "month": month_idx,
                        "year": env.state_manager.current_year,
                        "n_employees": len(fs.employees),
                        "target_size": fs.firm.target_size,
                        "n_vacancies": fs.n_vacancies,
                        "industry": fs.firm.industry,
                    }
                )
        for w_id in matched_worker_ids:
            ws = env.state_manager.workers.get(w_id)
            if ws:
                ind = ws.worker.industry
                hiring_by_industry_monthly[env.state_manager.current_year][ind] += 1
        for w_id in matched_worker_ids:
            ws = env.state_manager.workers.get(w_id)
            if ws:
                employer_id = ws.worker.current_employer_id
                if employer_id:
                    firm_action = firm_actions_decoded.get(employer_id, {})
                    wage_strategy = firm_action.get("wage_strategy", 2)
                    wages_by_strategy[wage_strategy].append(ws.worker.current_wage)
        industry_wages = defaultdict(list)
        for w_id, ws in env.state_manager.workers.items():
            if ws.worker.employment_status == "Employed" and ws.worker.current_wage > 0:
                ind = ws.worker.industry
                if ind is not None:
                    industry_wages[int(ind)].append(ws.worker.current_wage)
        step_wages = {
            str(ind): float(np.mean(wages)) for ind, wages in industry_wages.items() if wages
        }
        wages_by_industry_samples.append(step_wages)
        total_vacancies = sum((fs.n_vacancies for fs in env.state_manager.firms.values()))
        total_hires = len(matched_worker_ids)
        total_separations = sum(separations_by_industry.values())
        simulation_log["job_creation_destruction"].append(
            {
                "month": month_idx,
                "year": env.state_manager.current_year,
                "vacancies_posted": total_vacancies,
                "hires": total_hires,
                "separations": total_separations,
                "net_job_change": total_hires - total_separations,
            }
        )
        next_td = obs.get("next", obs)
        worker_rewards = next_td.get("worker_reward", None)
        if worker_rewards is not None:
            _reward_ids = _reward_source_ids if env._net_population is not None else env.worker_ids
            if env._net_population is not None and len(worker_rewards) != len(_reward_ids):
                raise RuntimeError("worker reward rows do not match pre-step worker IDs")
            for i, w_id in enumerate(_reward_ids[:100]):
                ws = env.state_manager.workers.get(w_id)
                if ws and i < len(worker_rewards):
                    reward = worker_rewards[i].item()
                    status = ws.worker.employment_status
                    wage = ws.worker.current_wage
                    unemp_months = ws.worker.unemployment_months
                    if status == "Employed":
                        if wage >= 25:
                            simulation_log["rewards_by_status"]["employed_high"].append(reward)
                        else:
                            simulation_log["rewards_by_status"]["employed_low"].append(reward)
                    elif status == "Unemployed":
                        if unemp_months < 6:
                            simulation_log["rewards_by_status"]["unemployed_short"].append(reward)
                        else:
                            simulation_log["rewards_by_status"]["unemployed_long"].append(reward)
        firm_rewards = next_td.get("firm_reward", None)
        if firm_rewards is not None:
            for i, f_id in enumerate(env.firm_ids[:20]):
                fs = env.state_manager.firms.get(f_id)
                if fs and i < len(firm_rewards):
                    reward = firm_rewards[i].item()
                    n_emp = len(fs.employees)
                    target = fs.firm.target_size
                    if n_emp > target * 1.1:
                        simulation_log["firm_rewards_by_status"]["overstaffed"].append(reward)
                    elif n_emp < target * 0.9:
                        simulation_log["firm_rewards_by_status"]["understaffed"].append(reward)
                    else:
                        simulation_log["firm_rewards_by_status"]["at_target"].append(reward)
        if env.state_manager.current_year > current_year:
            avg_metrics = aggregate_year_metrics(year_metrics)
            simulation_log["years"].append(current_year)
            simulation_log["unemployment_rate"].append(avg_metrics["unemployment_rate"])
            simulation_log["employment_rate"].append(avg_metrics["employment_rate"])
            simulation_log["participation_rate"].append(avg_metrics["participation_rate"])
            simulation_log["vacancy_rate"].append(avg_metrics["vacancy_rate"])
            simulation_log["aggregate_job_finding_rate"].append(avg_metrics["job_finding_rate"])
            simulation_log["aggregate_separation_rate"].append(avg_metrics["separation_rate"])
            current_year = env.state_manager.current_year
            year_metrics = []
        _vacancies_pre_renewal = len(env.state_manager.vacancies)
        _vacancy_ids_pre_renewal = set(env.state_manager.vacancies)
        if env._net_population is not None:
            obs = env._net_population.finalize(env, obs)
        _vacancies_post_renewal = len(env.state_manager.vacancies)
        _vacancy_ids_post_renewal = set(env.state_manager.vacancies)
        _status_post_renewal = {
            w_id: ws.worker.employment_status for w_id, ws in env.state_manager.workers.items()
        }
        _renewal_flow = _compute_employment_stock_flow(_status_after, _status_post_renewal)
        _renewal_records = list(getattr(getattr(env, "_net_population", None), "renewals", []))
        _renewal_employed_exits = 0
        _renewal_employed_placements = 0
        for _renewal in _renewal_records[_renewals_before:]:
            _renewal_employed_exits += sum(
                (_status_after.get(wid) == "Employed" for wid in _renewal.get("exits", []))
            )
            _placement = _renewal.get("employed_entrant_placement") or {}
            _renewal_employed_placements += int(_placement.get("placed", 0))
        _budget_boundary_sources = _capacity_budget_boundary_sources(
            _renewal_records[_renewals_before:], True
        )
        _boundary_check = _reconcile_renewal_boundary(
            _renewal_flow,
            _status_after,
            _status_post_renewal,
            _vacancy_ids_pre_renewal,
            _vacancy_ids_post_renewal,
            _renewal_employed_exits,
            _renewal_employed_placements,
            _budget_boundary_sources,
            True,
        )
        _renewal_reconciliation = _boundary_check["residuals"]
        _renewal_continuing_status_changes = _boundary_check["continuing_worker_status_change_ids"]
        _renewal_reconciliation["all_mechanisms_reconciled"] = all(
            (value == 0 for value in _renewal_reconciliation.values())
        )
        _renewal_flow["mechanism_counters"] = {
            "demographic_employed_exits": int(_renewal_employed_exits),
            "employed_entrant_placements": int(_renewal_employed_placements),
        }
        _renewal_flow["mechanism_counters"].update(
            {
                "capacity_budget_contraction_layoffs": len(
                    _budget_boundary_sources["layoff_worker_ids"]
                ),
                "capacity_budget_vacancy_withdrawals": _budget_boundary_sources[
                    "vacancy_withdrawals"
                ],
            }
        )
        _renewal_flow["capacity_budget_source_events"] = _budget_boundary_sources
        _renewal_flow["actual_continuing_worker_status_changes"] = _boundary_check[
            "actual_continuing_worker_status_changes"
        ]
        _renewal_flow["unattributed_status_change_ids"] = _boundary_check[
            "unattributed_status_change_ids"
        ]
        _renewal_flow["mechanism_reconciliation"] = _renewal_reconciliation
        _renewal_flow["continuing_worker_status_change_ids"] = _renewal_continuing_status_changes
        simulation_log["employment_stock_flow"][-1]["renewal_boundary"] = _renewal_flow
        simulation_log["matching_funnel"]["vacancy_event_ledger"][-1]["renewal_boundary"] = {
            "opening": int(_vacancies_pre_renewal),
            "closing": int(_vacancies_post_renewal),
            "employed_entrant_placement_fills": int(_renewal_employed_placements),
            "identity_residual": int(_renewal_reconciliation["vacancy_placement_residual"]),
        }
        simulation_log["matching_funnel"]["vacancy_event_ledger"][-1]["renewal_boundary"].update(
            {
                "capacity_budget_vacancy_withdrawals": _budget_boundary_sources[
                    "vacancy_withdrawals"
                ],
                "capacity_budget_withdrawn_vacancy_ids": _budget_boundary_sources[
                    "withdrawn_vacancy_ids"
                ],
            }
        )
        if not _renewal_reconciliation["all_mechanisms_reconciled"]:
            raise AssertionError(
                f"renewal employment mechanism reconciliation failed: {_renewal_flow}"
            )
        _expected_next_opening_employment = int(_renewal_flow["closing_employment"])
        _expected_next_opening_status = dict(_status_post_renewal)
        if _worker_trace is not None:
            _worker_trace.finish_month(env, simulation_log["employment_stock_flow"][-1])
        if obs["done"].item():
            break
    if _worker_trace is not None:
        simulation_log["worker_level_trace"] = _worker_trace.close()
    if year_metrics:
        avg_metrics = aggregate_year_metrics(year_metrics)
        simulation_log["years"].append(current_year)
        simulation_log["unemployment_rate"].append(avg_metrics["unemployment_rate"])
        simulation_log["employment_rate"].append(avg_metrics["employment_rate"])
        simulation_log["participation_rate"].append(avg_metrics["participation_rate"])
        simulation_log["vacancy_rate"].append(avg_metrics["vacancy_rate"])
        simulation_log["aggregate_job_finding_rate"].append(avg_metrics["job_finding_rate"])
        simulation_log["aggregate_separation_rate"].append(avg_metrics["separation_rate"])
        _flow_inst = getattr(env, "_demographic_flow_instrumenter", None)
        if _flow_inst is not None and _flow_inst.enabled:
            _final_end_stocks = env.state_manager._compute_cohort_stocks_for_fix2a()
            _flow_inst.emit_year_snapshot(year=current_year, end_of_year_stocks=_final_end_stocks)
    _flow_inst = getattr(env, "_demographic_flow_instrumenter", None)
    if _flow_inst is not None and _flow_inst.enabled:
        simulation_log["demographic_flow_snapshots"] = _flow_inst.annual_snapshots
    jfr_age_active = aggregate_job_finding_by_age(age_metrics_by_year)
    jfr_age_u_stock = aggregate_job_finding_by_age_u_stock(age_metrics_by_year)
    simulation_log["job_finding_by_age"] = jfr_age_active
    simulation_log["job_finding_by_age_active_search"] = jfr_age_active
    simulation_log["job_finding_by_age_u_stock"] = jfr_age_u_stock
    simulation_log["unemployment_duration_by_age"] = aggregate_unemp_duration_by_age(
        age_metrics_by_year
    )
    simulation_log["unemployment_duration_by_age_active_search"] = (
        aggregate_unemp_duration_by_age_active_search(age_metrics_by_year)
    )
    simulation_log["separation_by_industry"] = aggregate_separation_by_industry(
        industry_metrics_by_year, monthly_metrics
    )
    simulation_log["cohort_macros_by_age"] = aggregate_cohort_macros_time_series(
        age_metrics_by_year, simulation_log["years"]
    )
    simulation_log["cohort_macros_by_age_native"] = aggregate_cohort_macros_time_series(
        age_metrics_by_year_native, simulation_log["years"]
    )
    simulation_log["cohort_macros_by_skill"] = aggregate_cohort_macros_time_series(
        skill_metrics_by_year, simulation_log["years"]
    )
    simulation_log["cohort_macros_by_nilf_subtype"] = aggregate_cohort_macros_time_series(
        nilf_subtype_metrics_by_year, simulation_log["years"]
    )
    if nilf_subtype_by_age_acc:
        from environment.nilf_subtype import SUBTYPE_NAME

        age_labels = list(nilf_subtype_by_age_acc[0].keys())
        subtype_labels = list(SUBTYPE_NAME.values())
        avg = {a: {s: 0.0 for s in subtype_labels} for a in age_labels}
        for snap in nilf_subtype_by_age_acc:
            for a in age_labels:
                for s in subtype_labels:
                    avg[a][s] += snap[a].get(s, 0)
        n = len(nilf_subtype_by_age_acc)
        for a in age_labels:
            for s in subtype_labels:
                avg[a][s] /= n
        simulation_log["nilf_subtype_by_age_avg"] = avg
    if attachment_acc:
        STATUS_LABELS = ("E", "U_search", "U_no_search", "NILF_eligible", "NILF_ineligible")
        age_labels = list(attachment_acc[0]["E"].keys())
        cell_sum = {s: {a: 0.0 for a in age_labels} for s in STATUS_LABELS}
        cell_cnt = {s: {a: 0 for a in age_labels} for s in STATUS_LABELS}
        for snap in attachment_acc:
            for s in STATUS_LABELS:
                for a in age_labels:
                    cell_sum[s][a] += float(snap[s][a]["sum"])
                    cell_cnt[s][a] += int(snap[s][a]["count"])
        by_sx_age = {
            s: {
                a: {
                    "mean": cell_sum[s][a] / cell_cnt[s][a] if cell_cnt[s][a] > 0 else float("nan"),
                    "count": cell_cnt[s][a],
                }
                for a in age_labels
            }
            for s in STATUS_LABELS
        }
        simulation_log["attachment_by_status_x_age_avg"] = by_sx_age
        status_only = {}
        for s in STATUS_LABELS:
            total_s = sum(cell_sum[s].values())
            total_c = sum(cell_cnt[s].values())
            status_only[s] = {
                "mean": total_s / total_c if total_c > 0 else float("nan"),
                "count": total_c,
            }
        simulation_log["attachment_by_status_avg"] = status_only
    simulation_log["retirement_hazard_audit"] = None
    simulation_log["retirement_diagnostics"] = {
        "actor_retire_count": int(getattr(env, "_endogenous_retire_count_running", 0)),
        "actor_retire_count_rollout": int(getattr(env, "_endogenous_retire_count_rollout", 0)),
        "invalid_retire_action_count": int(getattr(env, "_invalid_retire_action_count_running", 0)),
        "invalid_retire_action_count_rollout": int(
            getattr(env, "_invalid_retire_action_count_rollout", 0)
        ),
        "env_conversion_attempted": int(getattr(env, "_env_conversion_attempted_running", 0)),
        "env_conversion_attempted_rollout": int(
            getattr(env, "_env_conversion_attempted_rollout", 0)
        ),
        "env_conversion_converted": int(getattr(env, "_env_conversion_converted_running", 0)),
        "env_conversion_converted_rollout": int(
            getattr(env, "_env_conversion_converted_rollout", 0)
        ),
        "env_conversion_by_age": dict(getattr(env, "_env_conversion_by_age_running", {}) or {}),
        "env_conversion_by_age_rollout": dict(
            getattr(env, "_env_conversion_by_age_rollout", {}) or {}
        ),
    }
    simulation_log["structural_diagnostic"] = {
        "enabled": False,
        "schema": "retirement_churn_and_prime_age_absorption_v1",
        "behavior_changed": False,
    }
    try:
        simulation_log["per_episode_acceptance"] = (
            env.matching_mechanism.get_episode_acceptance_stats()
        )
    except AttributeError:
        simulation_log["per_episode_acceptance"] = None
    from collections import defaultdict as _dd
    from environment.nilf_subtype import SUBTYPE_NAME as _SUB_NAME

    nilf_entry_log = getattr(env.state_manager, "nilf_entry_log", []) or []
    AGE_BINS = [
        (15, 25, "15-24"),
        (25, 35, "25-34"),
        (35, 45, "35-44"),
        (45, 55, "45-54"),
        (55, 65, "55-64"),
        (65, 200, "65-99"),
    ]

    def _age_to_band(a):
        for lo, hi, lab in AGE_BINS:
            if lo <= a < hi:
                return lab
        return None

    WITHIN_ROLLOUT_SOURCES = {
        "employed",
        "employed_retirement",
        "employed_retirement_gate",
        "unemployed_struct",
        "unemployed_discouraged",
    }
    counts = _dd(lambda: _dd(lambda: _dd(int)))
    for rec in nilf_entry_log:
        if rec.get("source_state") not in WITHIN_ROLLOUT_SOURCES:
            continue
        band = _age_to_band(rec.get("age", -1))
        if band is None:
            continue
        sub_name = _SUB_NAME.get(int(rec.get("subtype", 0)), "none")
        counts[rec["source_state"]][band][sub_name] += 1
    at_risk = {
        band: {"employed_mean": 0.0, "unemployed_mean": 0.0, "n_months": 0}
        for _, _, band in AGE_BINS
    }
    for year, monthly_list in age_metrics_by_year.items():
        for month_counts in monthly_list:
            for band, vals in month_counts.items():
                if band not in at_risk:
                    continue
                at_risk[band]["employed_mean"] += vals.get("employed", 0)
                at_risk[band]["unemployed_mean"] += vals.get("unemployed", 0)
                at_risk[band]["n_months"] += 1
    for band in at_risk:
        n_m = max(at_risk[band]["n_months"], 1)
        at_risk[band]["employed_mean"] /= n_m
        at_risk[band]["unemployed_mean"] /= n_m
    rates = {src: {band: {} for _, _, band in AGE_BINS} for src in WITHIN_ROLLOUT_SOURCES}
    for src in WITHIN_ROLLOUT_SOURCES:
        is_unemployed = src.startswith("unemployed")
        denom_key = "unemployed_mean" if is_unemployed else "employed_mean"
        for _, _, band in AGE_BINS:
            denom_pop = at_risk[band][denom_key]
            n_months = at_risk[band]["n_months"]
            if denom_pop > 0 and n_months > 0:
                for sub_name in _SUB_NAME.values():
                    c = counts[src][band].get(sub_name, 0)
                    rates[src][band][sub_name] = c / (denom_pop * n_months)
            else:
                for sub_name in _SUB_NAME.values():
                    rates[src][band][sub_name] = 0.0
    simulation_log["nilf_transition_counts"] = {
        src: {band: dict(counts[src][band]) for _, _, band in AGE_BINS}
        for src in WITHIN_ROLLOUT_SOURCES
    }
    simulation_log["nilf_transition_rates"] = rates
    simulation_log["at_risk_pop_by_age"] = at_risk
    simulation_log["worker_trajectories"] = [
        {"worker_id": w_id, "trajectory": traj}
        for w_id, traj in list(worker_trajectories.items())[:50]
    ]
    simulation_log["firm_trajectories"] = [
        {"firm_id": f_id, "trajectory": traj} for f_id, traj in list(firm_trajectories.items())[:10]
    ]
    simulation_log["hiring_by_industry"] = {
        str(ind): sum(
            (hiring_by_industry_monthly[year].get(ind, 0) for year in hiring_by_industry_monthly)
        )
        for ind in set(
            (ind for year_data in hiring_by_industry_monthly.values() for ind in year_data)
        )
    }
    ind_totals = defaultdict(lambda: {"employed_sum": 0, "vacancy_sum": 0, "months": 0})
    for year, monthly_list in industry_metrics_by_year.items():
        for month_counts in monthly_list:
            for ind_code, counts in month_counts.items():
                ind_totals[str(ind_code)]["employed_sum"] += counts.get("employed", 0)
                ind_totals[str(ind_code)]["vacancy_sum"] += counts.get("vacancies", 0)
                ind_totals[str(ind_code)]["months"] += 1
    simulation_log["employment_by_industry"] = {
        ind: totals["employed_sum"] / totals["months"] if totals["months"] > 0 else 0
        for ind, totals in ind_totals.items()
    }
    simulation_log["vacancy_by_industry"] = {
        ind: totals["vacancy_sum"] / totals["months"] if totals["months"] > 0 else 0
        for ind, totals in ind_totals.items()
    }
    for key in simulation_log["firm_rewards_by_status"]:
        rewards_list = simulation_log["firm_rewards_by_status"][key]
        if len(rewards_list) > 1000:
            simulation_log["firm_rewards_by_status"][key] = rewards_list[:1000]
    if len(simulation_log["job_creation_destruction"]) > 132:
        simulation_log["job_creation_destruction"] = simulation_log["job_creation_destruction"][
            :132
        ]
    for key in simulation_log["rewards_by_status"]:
        rewards_list = simulation_log["rewards_by_status"][key]
        if len(rewards_list) > 1000:
            simulation_log["rewards_by_status"][key] = rewards_list[:1000]
    simulation_log["search_participation_distribution"] = dict(search_participation)
    simulation_log["search_participation_distribution_by_year"] = {
        int(yr): dict(counts) for yr, counts in search_participation_by_year.items()
    }
    simulation_log["search_channel_distribution"] = {
        k: v["total"] for k, v in channel_counts.items()
    }
    simulation_log["job_finding_by_channel"] = {
        k: v["matches"] / v["total"] if v["total"] > 0 else 0 for k, v in channel_counts.items()
    }
    simulation_log["search_channel_distribution_by_immigrant"] = {
        grp: {k: v["total"] for k, v in group.items()}
        for grp, group in channel_counts_by_immigrant.items()
    }
    simulation_log["job_finding_by_channel_by_immigrant"] = {
        grp: {k: v["matches"] / v["total"] if v["total"] > 0 else 0 for k, v in group.items()}
        for grp, group in channel_counts_by_immigrant.items()
    }
    if channel_probs_accumulator:
        avg_channel_probs = np.mean(channel_probs_accumulator, axis=0)
        simulation_log["policy_channel_probabilities"] = {
            i: float(avg_channel_probs[i]) for i in range(len(avg_channel_probs))
        }
    if intensity_probs_accumulator:
        avg_intensity_probs = np.mean(intensity_probs_accumulator, axis=0)
        simulation_log["policy_intensity_probabilities"] = {
            i: float(avg_intensity_probs[i]) for i in range(len(avg_intensity_probs))
        }
    if wage_probs_accumulator:
        avg_wage_probs = np.mean(wage_probs_accumulator, axis=0)
        simulation_log["policy_wage_strategy_probabilities"] = {
            i: float(avg_wage_probs[i]) for i in range(len(avg_wage_probs))
        }
    simulation_log["search_intensity_distribution"] = {
        k: v["total"] for k, v in intensity_counts.items()
    }
    simulation_log["job_finding_by_intensity"] = {
        k: v["matches"] / v["total"] if v["total"] > 0 else 0 for k, v in intensity_counts.items()
    }
    simulation_log["firm_profit_by_strategy"] = {
        "monthly": firm_pbs_monthly,
        "pooled": {"per_strategy": firm_pbs_pooled, "n_months": len(firm_pbs_monthly)},
    }
    simulation_log["wage_strategy_distribution"] = dict(wage_strategy_counts)
    simulation_log["wage_dispersion_by_strategy"] = {
        k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
        for k, v in wages_by_strategy.items()
        if v
    }
    simulation_log["use_travel_time"] = use_travel_time if use_travel_time is not None else False
    simulation_log["commute_unit"] = commute_unit if commute_unit is not None else "km"
    stock_sample = commute_stock[-5000:] if commute_stock else []
    if use_travel_time:
        simulation_log["commute_time_distribution"] = stock_sample
    else:
        simulation_log["commute_distance_distribution"] = stock_sample
    simulation_log["commute_flow_distribution"] = (
        commute_distances[-2000:] if commute_distances else []
    )
    simulation_log["wages_by_industry"] = wages_by_industry_samples
    simulation_log["reservation_by_duration"] = {
        k: {"mean": np.mean(v), "std": np.std(v)} for k, v in reservation_by_duration.items() if v
    }
    _renewal_entry_audit = env._renewal_entry_audit_summary()
    simulation_log["renewal_entry_audit"] = _renewal_entry_audit
    if env._net_population is not None:
        simulation_log["net_population_audit"] = env._net_population.summary(env)
    _renewal_curve_labels = monthly_imm_curves["year_month"]
    _renewal_expected_curve_n = 12 * len(simulation_log["years"])
    _renewal_expected_first = f"{simulation_log['years'][0]}-01"
    _renewal_expected_last = f"{simulation_log['years'][-1]}-12"
    if (
        len(_renewal_curve_labels) != _renewal_expected_curve_n
        or not _renewal_curve_labels
        or _renewal_curve_labels[0] != _renewal_expected_first
        or (_renewal_curve_labels[-1] != _renewal_expected_last)
        or (len(set(_renewal_curve_labels)) != len(_renewal_curve_labels))
    ):
        raise RuntimeError(
            "The monthly calendar by immigrant status is wrong: n=%d expected=%d first=%r expected_first=%r last=%r expected_last=%r"
            % (
                len(_renewal_curve_labels),
                _renewal_expected_curve_n,
                _renewal_curve_labels[0] if _renewal_curve_labels else None,
                _renewal_expected_first,
                _renewal_curve_labels[-1] if _renewal_curve_labels else None,
                _renewal_expected_last,
            )
        )
    simulation_log["_immigrant_splits"] = {
        "unemployment_spells": {
            "immigrant": unemployment_spells_by_immigrant[True],
            "nonimmigrant": unemployment_spells_by_immigrant[False],
        },
        "commute_times": {
            "immigrant": commute_by_immigrant[True],
            "nonimmigrant": commute_by_immigrant[False],
        },
        "wages": {"immigrant": wages_by_immigrant[True], "nonimmigrant": wages_by_immigrant[False]},
        "jfr_by_age": {
            ag: {
                "immigrant_found": jfr_by_age_immigrant[ag][True]["found"],
                "immigrant_months": jfr_by_age_immigrant[ag][True]["months"],
                "nonimmigrant_found": jfr_by_age_immigrant[ag][False]["found"],
                "nonimmigrant_months": jfr_by_age_immigrant[ag][False]["months"],
            }
            for ag in sorted(jfr_by_age_immigrant.keys())
        },
        "trajectories": {wid: traj for wid, traj in imm_worker_trajectories.items()},
        "monthly_curves": monthly_imm_curves,
        "commute_at_match": commute_at_match,
    }
    _ge_audit_attached = getattr(env.matching_mechanism, "_ge_audit_log", None)
    if _ge_audit_attached is not None:
        simulation_log["_ge_audit_log"] = _ge_audit_attached
    _trade_regime = getattr(env, "_trade_war_regime", None)
    if _trade_regime is not None and _trade_regime.enabled:
        simulation_log["trade_war"] = {
            "provenance": _trade_regime.provenance,
            "open_population_provenance": getattr(config, "_openpop_eval_provenance", None),
            "monthly": _trade_regime.history,
            "observed_monthly_stocks": env._trade_observed_stocks.export(len(monthly_metrics)),
        }
        if len(_trade_regime.history) != len(monthly_metrics):
            raise RuntimeError(
                f"The tariff history and the model history differ in length: {len(_trade_regime.history)} and {len(monthly_metrics)} months"
            )
    from market_snapshots import export as _coverage_export

    simulation_log["market_snapshot"] = _coverage_export(env, len(monthly_metrics))
    return simulation_log


def aggregate_year_metrics(year_metrics: List[Dict]) -> Dict[str, float]:
    if not year_metrics:
        return {}
    keys = [
        "unemployment_rate",
        "employment_rate",
        "participation_rate",
        "vacancy_rate",
        "job_finding_rate",
        "separation_rate",
    ]
    result = {}
    for key in keys:
        values = [m.get(key, 0) for m in year_metrics]
        result[key] = np.mean(values) if values else 0
    return result


def aggregate_job_finding_by_age(age_metrics_by_year: Dict) -> Dict[str, float]:
    age_totals = defaultdict(lambda: {"matches": 0, "searching": 0})
    for year, monthly_list in age_metrics_by_year.items():
        for month_counts in monthly_list:
            for age_label, counts in month_counts.items():
                age_totals[age_label]["matches"] += counts.get("matches", 0)
                age_totals[age_label]["searching"] += counts.get("searching", 0)
    result = {}
    for age_label, totals in age_totals.items():
        if totals["searching"] > 0:
            result[age_label] = totals["matches"] / totals["searching"]
        else:
            result[age_label] = 0.0
    return result


def aggregate_job_finding_by_age_u_stock(age_metrics_by_year: Dict) -> Dict[str, float]:
    age_totals = defaultdict(lambda: {"matches": 0, "unemployed": 0})
    for year, monthly_list in age_metrics_by_year.items():
        for month_counts in monthly_list:
            for age_label, counts in month_counts.items():
                age_totals[age_label]["matches"] += counts.get("matches", 0)
                age_totals[age_label]["unemployed"] += counts.get("unemployed", 0)
    result = {}
    for age_label, totals in age_totals.items():
        if totals["unemployed"] > 0:
            result[age_label] = totals["matches"] / totals["unemployed"]
        else:
            result[age_label] = 0.0
    return result


def aggregate_unemp_duration_by_age(age_metrics_by_year: Dict) -> Dict[str, float]:
    age_totals = defaultdict(lambda: {"duration_sum": 0, "count": 0})
    for year, monthly_list in age_metrics_by_year.items():
        for month_counts in monthly_list:
            for age_label, counts in month_counts.items():
                age_totals[age_label]["duration_sum"] += counts.get("unemp_duration_sum", 0)
                age_totals[age_label]["count"] += counts.get("unemp_count", 0)
    result = {}
    for age_label, totals in age_totals.items():
        if totals["count"] > 0:
            avg_months = totals["duration_sum"] / totals["count"]
            result[age_label] = avg_months * 4.33
        else:
            result[age_label] = 0.0
    return result


def aggregate_unemp_duration_by_age_active_search(age_metrics_by_year: Dict) -> Dict[str, float]:
    age_totals = defaultdict(lambda: {"duration_sum": 0, "count": 0})
    for year, monthly_list in age_metrics_by_year.items():
        for month_counts in monthly_list:
            for age_label, counts in month_counts.items():
                age_totals[age_label]["duration_sum"] += counts.get(
                    "unemp_duration_sum_active_search", 0
                )
                age_totals[age_label]["count"] += counts.get("unemp_count_active_search", 0)
    result = {}
    for age_label, totals in age_totals.items():
        if totals["count"] > 0:
            avg_months = totals["duration_sum"] / totals["count"]
            result[age_label] = avg_months * 4.33
        else:
            result[age_label] = 0.0
    return result


def aggregate_cohort_macros_time_series(
    cohort_metrics_by_year: Dict, years: List[int]
) -> Dict[str, Any]:
    cohort_labels = set()
    for monthly_list in cohort_metrics_by_year.values():
        for month_counts in monthly_list:
            cohort_labels.update(month_counts.keys())
    annual_keys = [
        "nilf_rate_raw",
        "nilf_rate_aligned",
        "lfpr_raw",
        "lfpr_aligned",
        "ulf_raw",
        "ulf_aligned",
        "pop",
    ]
    annual = {c: {k: [] for k in annual_keys} for c in cohort_labels}
    monthly = {c: {"nilf_rate_raw": [], "nilf_rate_aligned": []} for c in cohort_labels}
    _ordered_months = []
    for _source_year in sorted(cohort_metrics_by_year):
        _ordered_months.extend(cohort_metrics_by_year[_source_year])
    _expected_months = 12 * len(years)
    if len(_ordered_months) != _expected_months:
        raise RuntimeError(
            "expected %d monthly cohort snapshots for %d full reporting years; found %d"
            % (_expected_months, len(years), len(_ordered_months))
        )
    for _year_idx, year in enumerate(years):
        year_acc = {
            c: {"employed": 0, "unemployed": 0, "nilf": 0, "u_active": 0, "months": 0}
            for c in cohort_labels
        }
        _month_start = 12 * _year_idx
        for month_counts in _ordered_months[_month_start : _month_start + 12]:
            for c, vals in month_counts.items():
                emp = vals.get("employed", 0)
                unemp = vals.get("unemployed", 0)
                nilf_m = vals.get("nilf", 0)
                u_act = vals.get("unemp_count_active_search", 0)
                pop_m = emp + unemp + nilf_m
                nilf_aligned_m = nilf_m + (unemp - u_act)
                monthly[c]["nilf_rate_raw"].append(nilf_m / pop_m if pop_m > 0 else 0.0)
                monthly[c]["nilf_rate_aligned"].append(nilf_aligned_m / pop_m if pop_m > 0 else 0.0)
                year_acc[c]["employed"] += emp
                year_acc[c]["unemployed"] += unemp
                year_acc[c]["nilf"] += nilf_m
                year_acc[c]["u_active"] += u_act
                year_acc[c]["months"] += 1
        for c, acc in year_acc.items():
            pop = acc["employed"] + acc["unemployed"] + acc["nilf"]
            lf_raw = acc["employed"] + acc["unemployed"]
            lf_aligned = acc["employed"] + acc["u_active"]
            nilf_aligned = acc["nilf"] + (acc["unemployed"] - acc["u_active"])
            annual[c]["nilf_rate_raw"].append(acc["nilf"] / pop if pop > 0 else 0.0)
            annual[c]["nilf_rate_aligned"].append(nilf_aligned / pop if pop > 0 else 0.0)
            annual[c]["lfpr_raw"].append(lf_raw / pop if pop > 0 else 0.0)
            annual[c]["lfpr_aligned"].append(lf_aligned / pop if pop > 0 else 0.0)
            annual[c]["ulf_raw"].append(acc["unemployed"] / lf_raw if lf_raw > 0 else 0.0)
            annual[c]["ulf_aligned"].append(acc["u_active"] / lf_aligned if lf_aligned > 0 else 0.0)
            annual[c]["pop"].append(pop / acc["months"] if acc["months"] > 0 else 0)
    return {"annual": annual, "monthly": monthly}


def aggregate_separation_by_industry(
    industry_metrics_by_year: Dict, monthly_metrics: List[Dict]
) -> Dict[str, float]:
    industry_totals = defaultdict(lambda: {"employed": 0, "separations": 0})
    for year, monthly_list in industry_metrics_by_year.items():
        for month_counts in monthly_list:
            for ind_code, counts in month_counts.items():
                industry_totals[str(ind_code)]["employed"] += counts.get("employed", 0)
    for m in monthly_metrics:
        sep_by_ind = m.get("separations_by_industry", {})
        for ind_code, sep_count in sep_by_ind.items():
            industry_totals[str(ind_code)]["separations"] += sep_count
    result = {}
    for ind_code, totals in industry_totals.items():
        if totals["employed"] > 0:
            monthly_rate = totals["separations"] / totals["employed"]
            annual_rate = monthly_rate * 12
            result[ind_code] = min(annual_rate, 1.0)
        else:
            result[ind_code] = 0.0
    return result
