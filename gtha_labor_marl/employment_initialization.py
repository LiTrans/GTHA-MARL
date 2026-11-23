from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path

MODE = "workplace_distribution_initial_only_v1"
from market_setup import data_root

HOME = data_root()
MODEL_CMAS = (532, 535, 537)
_NAICS2_TO_NAICS18 = {
    "11": 1,
    "21": 2,
    "22": 3,
    "23": 4,
    "31-33": 5,
    "41": 7,
    "44-45": 8,
    "48-49": 9,
    "51": 15,
    "52": 10,
    "53": 10,
    "54": 11,
    "55": 12,
    "56": 12,
    "61": 13,
    "62": 14,
    "71": 15,
    "72": 16,
    "81": 17,
    "91": 18,
}


def enabled():
    from market_snapshots import plan

    mode = plan().get("employment_initialization")
    if mode is None:
        return False
    if mode != MODE:
        raise ValueError("Unknown initial employment design")
    return True


def apportioned(weights, count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Invalid allocation count")
    if (
        not weights
        or any((not math.isfinite(v) or v < 0 for v in weights.values()))
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Allocation requires positive finite source weight")
    total = sum(weights.values())
    raw = {k: count * v / total for k, v in weights.items()}
    result = {k: math.floor(v) for k, v in raw.items()}
    order = sorted(raw, key=lambda k: (-(raw[k] - result[k]), str(k)))
    for k in order[: count - sum(result.values())]:
        result[k] += 1
    if sum(result.values()) != count:
        raise ValueError("Integer allocation failed")
    return result


def _rank(seed, kind, key):
    return hashlib.sha256(f"employment-init-v1|{seed}|{kind}|{key}".encode()).hexdigest()


def _rule_for(code, industry):
    for prefix in ("3361", "3362", "3363", "331", "332"):
        if code.startswith(prefix):
            return "naics6_prefix:" + prefix
    return "naics18:5" if industry == 5 else "default:*"


def _load_census_data(data_dir):
    data_dir = Path(data_dir)
    wp_weights: dict = {}
    _wp_raw = defaultdict(int)
    with (data_dir / "cma_workplace_industry_2006.csv").open(newline="", encoding="utf-8-sig") as f:
        rows_loaded = 0
        for row in csv.DictReader(f):
            ind2 = row["industry"]
            if ind2 == "TOTAL" or row["workplace_scope"] != "usual" or row["worker_class"] != "all":
                continue
            naics18 = _NAICS2_TO_NAICS18.get(ind2)
            if naics18 is None:
                raise ValueError(f"Unrecognised 2-digit NAICS in workplace file: {ind2!r}")
            _wp_raw[int(row["cma"]), naics18] += int(row["employed_people"])
            rows_loaded += 1
    if rows_loaded == 0:
        raise ValueError("Workplace file loaded no data rows (header-only or wrong filters)")
    wp_weights = dict(_wp_raw)
    commuting: dict = {}
    _comm_raw = defaultdict(int)
    with (data_dir / "commuting_cma_industry_2006.csv").open(newline="", encoding="utf-8-sig") as f:
        rows_loaded = 0
        for row in csv.DictReader(f):
            ind2 = row["industry"]
            if ind2 == "TOTAL":
                continue
            naics18 = _NAICS2_TO_NAICS18.get(ind2)
            if naics18 is None:
                raise ValueError(f"Unrecognised 2-digit NAICS in commuting file: {ind2!r}")
            res_raw, wp_raw = (row["residence_cma"], row["workplace_cma"])
            res = int(res_raw) if res_raw.isdigit() else res_raw
            wp = int(wp_raw) if wp_raw.isdigit() else wp_raw
            _comm_raw[res, wp, naics18] += int(row["employed_people"])
            rows_loaded += 1
    if rows_loaded == 0:
        raise ValueError("Commuting file loaded no data rows (header-only or wrong filters)")
    commuting = dict(_comm_raw)
    _res_raw = defaultdict(
        lambda: {"all": 0, "wage_earner": 0, "self_employed": 0, "unpaid_family": 0}
    )
    with (data_dir / "resident_industry_class_place_2006.csv").open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows_loaded = 0
        for row in csv.DictReader(f):
            ind2 = row["industry"]
            if ind2 == "TOTAL" or row["place_of_work"] != "all_employed":
                continue
            naics18 = _NAICS2_TO_NAICS18.get(ind2)
            if naics18 is None:
                raise ValueError(f"Unrecognised 2-digit NAICS in resident file: {ind2!r}")
            cls = row["worker_class"]
            n = int(row["employed_people"])
            cell = _res_raw[int(row["cma"]), naics18]
            if cls in ("wage_earner", "self_employed", "unpaid_family"):
                cell[cls] += n
            elif cls == "all":
                cell["all"] += n
            rows_loaded += 1
    if rows_loaded == 0:
        raise ValueError("Resident file loaded no data rows (header-only or wrong filters)")
    resident_class = {k: dict(v) for k, v in _res_raw.items()}
    return (wp_weights, commuting, resident_class)


def plan():
    from market_snapshots import plan as _plan

    return _plan()


def source_profiles():
    from market_mechanisms import source_profiles as _sp

    return _sp()


def _commuting_accounting(workers, commuting, wp_weights):
    res_counts = Counter(((w["cma"], w["industry"]) for w in workers))
    rows = []
    for (cma, naics18), n_residents in sorted(res_counts.items()):
        local = commuting.get((cma, cma, naics18), 0)
        outgoing_3 = sum(
            (commuting.get((cma, other, naics18), 0) for other in MODEL_CMAS if other != cma)
        )
        outside = commuting.get((cma, "OUTSIDE_3_CMA", naics18), 0)
        total_res_flow = local + outgoing_3 + outside
        incoming_3 = sum(
            (commuting.get((other, cma, naics18), 0) for other in MODEL_CMAS if other != cma)
        )
        incoming_outside = commuting.get(("OUTSIDE_3_CMA", cma, naics18), 0)
        rows.append(
            {
                "cma": cma,
                "naics18": naics18,
                "employed_residents_in_sim": n_residents,
                "wp_employment_census": wp_weights.get((cma, naics18), 0),
                "commuting_res_local": local,
                "commuting_res_outgoing_3cma": outgoing_3,
                "commuting_res_outside": outside,
                "commuting_res_total_flow": total_res_flow,
                "commuting_incoming_3cma": incoming_3,
                "commuting_incoming_outside": incoming_outside,
                "outside_share": outside / total_res_flow if total_res_flow else None,
                "assumption_A2_applies": outside > 0,
            }
        )
    return rows


def allocate(workers, firms, profiles, seed, commuting, wp_weights=None):
    if len({r["id"] for r in workers}) != len(workers) or len({r["id"] for r in firms}) != len(
        firms
    ):
        raise ValueError("Duplicate initialization ID")
    source = {r["source_firm_id"]: r for rows in profiles.values() for r in rows}
    if len(source) != sum(map(len, profiles.values())):
        raise ValueError("Duplicate source establishment")
    for f in firms:
        s = source.get(f["id"])
        if s is None or any((f[k] != s[k] for k in ("industry", "naics6", "cma", "da_id"))):
            raise ValueError("A selected firm differs from its source record")
        if s["estimated_employees"] <= 0:
            raise ValueError("Selected firm has no positive source-size weight")
    cma_industry_est = defaultdict(float)
    cma_industry_rule_est = defaultdict(float)
    for industry_profiles in profiles.values():
        for s in industry_profiles:
            cma_industry_est[s["cma"], s["industry"]] += s["estimated_employees"]
            cma_industry_rule_est[
                s["cma"], s["industry"], _rule_for(s["naics6"], s["industry"])
            ] += s["estimated_employees"]
    res_industry_workers: dict = defaultdict(list)
    for w in workers:
        res_industry_workers[w["cma"], w["industry"]].append(w)
    origin_quotas: dict = {}
    group_rows = []
    cma_industry_spillover = []
    for (res_cma, industry), ws in sorted(res_industry_workers.items()):
        if industry not in profiles:
            raise ValueError("Worker industry lacks a source pool")
        within_flows = {}
        for wp_cma in MODEL_CMAS:
            v = commuting.get((res_cma, wp_cma, industry))
            if v is None:
                raise ValueError(
                    f"No commuting data for (res={res_cma}, wp={wp_cma}, industry={industry}). Pass data loaded from the census CSVs, not a partial or header-only file."
                )
            within_flows[wp_cma] = v
        outside = commuting.get((res_cma, "OUTSIDE_3_CMA", industry))
        if outside is None:
            raise ValueError(
                f"No commuting data for (res={res_cma}, OUTSIDE, industry={industry})."
            )
        total_within = sum(within_flows.values())
        if total_within == 0 and outside == 0:
            raise ValueError(
                f"All commuting flows are zero for (res={res_cma}, industry={industry}). This indicates missing or suppressed census data, not observed zero employment."
            )
        if total_within > 0:
            effective = {
                cma: flow + outside * flow / total_within for cma, flow in within_flows.items()
            }
        else:
            effective = {res_cma: 1.0}
        available_cmas = {f["cma"] for f in firms if f["industry"] == industry}
        if not available_cmas:
            raise ValueError(f"No selected firm exists for industry {industry}")
        redirected_outside_available = 0.0
        final_effective = defaultdict(float)
        for wp_cma, share in effective.items():
            if wp_cma in available_cmas:
                final_effective[wp_cma] += share
            else:
                redirected_outside_available += share
        if redirected_outside_available > 0 and final_effective:
            total_available_share = sum(final_effective.values())
            if total_available_share == 0:
                raise ValueError(
                    f"Cannot redistribute the commuting flow for residence {res_cma} and industry {industry}: the selected firms are only in metropolitan areas with no observed flow from this origin. Add a firm in an area that has observed flow."
                )
            for wp_cma in list(final_effective):
                final_effective[wp_cma] += (
                    redirected_outside_available * final_effective[wp_cma] / total_available_share
                )
            cma_industry_spillover.append(
                {
                    "res_cma": res_cma,
                    "industry": industry,
                    "workers": len(ws),
                    "redirected_share": redirected_outside_available,
                    "reason": "no_selected_firm_in_target_cma",
                }
            )
        elif redirected_outside_available > 0:
            raise ValueError(
                f"All commuting destinations for (res={res_cma}, industry={industry}) have no selected firm, and no available CMA to redirect to."
            )
        wp_cma_dist = apportioned({cma: v for cma, v in final_effective.items() if v > 0}, len(ws))
        for wp_cma, n_workers in sorted(wp_cma_dist.items()):
            if n_workers == 0:
                continue
            rule_weights = Counter()
            for s in profiles[industry]:
                if s["cma"] == wp_cma:
                    rule = _rule_for(s["naics6"], industry)
                    rule_weights[rule] += s["estimated_employees"]
            if not rule_weights:
                raise ValueError(f"No source profiles for (wp={wp_cma}, industry={industry})")
            rule_dist = apportioned(rule_weights, n_workers)
            for rule, n_rule_workers in sorted(rule_dist.items()):
                if n_rule_workers == 0:
                    continue
                selected = [
                    f
                    for f in firms
                    if f["industry"] == industry
                    and f["cma"] == wp_cma
                    and (_rule_for(f["naics6"], industry) == rule)
                ]
                if not selected:
                    raise ValueError(
                        f"Positive job quota but no selected firm: industry={industry} cma={wp_cma} rule={rule}"
                    )
                for _fid, _n in apportioned(
                    {f["id"]: source[f["id"]]["estimated_employees"] for f in selected},
                    n_rule_workers,
                ).items():
                    origin_quotas[res_cma, _fid] = origin_quotas.get((res_cma, _fid), 0) + _n
                group_rows.append(
                    {
                        "res_cma": res_cma,
                        "industry": industry,
                        "wp_cma": wp_cma,
                        "group": rule,
                        "commuting_flow_within": within_flows.get(wp_cma, 0),
                        "commuting_outside_flow": outside,
                        "workers_allocated": n_rule_workers,
                        "selected_firms": len(selected),
                        "wp_employment_census": (wp_weights or {}).get((wp_cma, industry), None),
                    }
                )
    quotas: dict = {}
    for (_rc, _fid), _n in origin_quotas.items():
        quotas[_fid] = quotas.get(_fid, 0) + _n
    for f in firms:
        quotas.setdefault(f["id"], 0)
    origin_remaining = dict(origin_quotas)
    assignment: dict = {}
    for (res_cma, industry), ws in sorted(res_industry_workers.items()):
        ordered = sorted(ws, key=lambda w: _rank(seed, "worker", w["id"]))
        destinations = sorted(
            [f for f in firms if f["industry"] == industry],
            key=lambda f: _rank(seed, "firm", f["id"]),
        )
        for w in ordered:
            candidates = [
                f for f in destinations if origin_remaining.get((res_cma, f["id"]), 0) > 0
            ]
            if candidates:
                fid = candidates[0]["id"]
                assignment[w["id"]] = fid
                origin_remaining[res_cma, fid] -= 1
    if (
        len(assignment) != len(workers)
        or any(origin_remaining.values())
        or Counter(assignment.values()) != Counter({k: v for k, v in quotas.items() if v})
    ):
        raise ValueError("Staged assignment failed reconciliation")
    selected_cells = {
        (f["industry"], f["cma"], _rule_for(f["naics6"], f["industry"])) for f in firms
    }
    unsupported = Counter()
    for industry, rows in profiles.items():
        for s in rows:
            key = (industry, s["cma"], _rule_for(s["naics6"], industry))
            if key not in selected_cells:
                unsupported[key] += s["estimated_employees"]
    return (
        assignment,
        quotas,
        {
            "groups": group_rows,
            "cma_industry_spillover": cma_industry_spillover,
            "unsupported_source_cells": [
                {"industry": k[0], "cma": k[1], "group": k[2], "source_size_proxy": v}
                for k, v in sorted(unsupported.items())
            ],
        },
    )


def initialize(env, _data_dir=None):
    if not enabled():
        return
    if (env.current_year, env.current_month, env.current_step) != (2006, 1, 0):
        raise ValueError("Employment initialization is initial-only")
    sm = env.state_manager
    if sm.vacancies:
        raise ValueError("Employment assignment must precede vacancy generation")
    p = plan()
    if _data_dir is not None:
        data_dir = Path(_data_dir)
    else:
        wp_key = p.get("workplace_data_dir")
        if not wp_key:
            raise ValueError(
                "The settings lack workplace_data_dir, which the workplace distribution needs"
            )
        data_dir = HOME / wp_key
    wp_weights, commuting_flows, resident_class_data = _load_census_data(data_dir)
    workers = [
        {"id": wid, "industry": int(ws.worker.industry), "cma": int(ws.worker.cma)}
        for wid, ws in sm.workers.items()
        if ws.worker.employment_status == "Employed"
    ]
    firms = [
        {
            "id": fid,
            "industry": int(fs.firm.industry),
            "cma": int(fs.firm.cma),
            "naics6": str(fs.firm.naics_6digit).removesuffix(".0"),
            "da_id": int(fs.firm.da_id),
        }
        for fid, fs in sm.firms.items()
    ]
    assignments, quotas, report = allocate(
        workers,
        firms,
        source_profiles()[0],
        p["population_design_seed"],
        commuting_flows,
        wp_weights,
    )
    before = {fid: len(fs.employees) for fid, fs in sm.firms.items()}
    source_counts = {fid: fs.firm.n_employees for fid, fs in sm.firms.items()}
    rosters = {fid: [] for fid in sm.firms}
    for wid, fid in assignments.items():
        rosters[fid].append(wid)
    for fid, fs in sm.firms.items():
        fs.employees = sorted(rosters[fid])
        fs.firm.n_employees = quotas[fid]
    for wid, fid in assignments.items():
        sm.workers[wid].worker.current_employer_id = fid
    sm._update_aggregate_stats()
    env._initial_employment_audit = {
        "mode": MODE,
        "decision_bearing": False,
        "population_representative": False,
        "weights": "commuting_flows_drive_cma_distribution; establishment_size_intra_cma_intra_industry",
        "assignment": "commuting_fraction_origin_quota_then_establishment_size_firm",
        "declared_assumptions": [
            "A1_worker_class_destination_independence",
            "A2_outside_3cma_redistributed_proportionally",
            "A3_self_employed_assigned_to_payroll_firms",
            "A4_intra_cma_intra_industry_uses_establishment_size",
        ],
        "worker_employment_total": len(workers),
        "worker_industry_counts": dict(Counter((w["industry"] for w in workers))),
        "firms": [
            {
                "firm_id": f["id"],
                "industry": f["industry"],
                "cma": f["cma"],
                "naics6": f["naics6"],
                "source_headcount": source_counts[f["id"]],
                "old_roster": before[f["id"]],
                "initial_jobs": quotas[f["id"]],
            }
            for f in firms
        ],
        "cross_cma_assignments": sum(
            (
                sm.firms[fid].firm.cma != sm.workers[wid].worker.cma
                for wid, fid in assignments.items()
            )
        ),
        "commuting_accounting": _commuting_accounting(workers, commuting_flows, wp_weights),
        "resident_class_fractions": [
            {"cma": k[0], "naics18": k[1], **v}
            for k, v in sorted(resident_class_data.items())
            if (k[0], k[1]) in wp_weights
        ],
        **report,
    }


def capacity(env):
    if not enabled():
        return
    if (env.current_year, env.current_month, env.current_step) != (2006, 1, 0) or not hasattr(
        env, "_initial_employment_audit"
    ):
        raise ValueError("Capacity reconciliation requires the initial assignment")
    import torch

    rows = {r["firm_id"]: r for r in env._initial_employment_audit["firms"]}
    targets = []
    for fid in env.firm_ids:
        fs = env.state_manager.firms[fid]
        jobs, vacancies = (len(fs.employees), len(fs.vacancies))
        if jobs != rows[fid]["initial_jobs"] or fs.firm.n_employees != jobs:
            raise ValueError("Initial headcount changed before capacity reconciliation")
        target = max(1, jobs + vacancies)
        rows[fid].update(
            initial_vacancies=vacancies, target=target, empty_firm_floor=target - jobs - vacancies
        )
        targets.append(target)
    env.firm_target_size = torch.tensor(targets, dtype=torch.int32, device=env.device)
    for fid, target in zip(env.firm_ids, targets):
        env.state_manager.firms[fid].firm.target_size = target
    env._initial_employment_audit["capacity_definition"] = (
        "initial_roster_plus_inherited_vacancy_draw_one_slot_floor"
    )
    env._initial_employment_audit["aggregate_target"] = sum(targets)


def audit(env):
    if not enabled():
        return None
    report = getattr(env, "_initial_employment_audit", None)
    if report is None or "aggregate_target" not in report:
        raise ValueError("Incomplete initial employment reconciliation")
    return report
