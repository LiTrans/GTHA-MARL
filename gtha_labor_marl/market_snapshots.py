import math
from collections import Counter
from functools import lru_cache
from market_setup import data_root

HOME = data_root()
PERIODS = [(y, m) for y in range(2006, 2025) for m in range(1, 13)]


def enabled():
    return True


@lru_cache(maxsize=1)
def plan():
    from market_setup import settings

    return settings()


def periods():
    from population_participation import pre2025_scope

    scope = pre2025_scope()
    if scope is None:
        raise ValueError("The run record needs a simulation horizon")
    return [(y, m) for y in range(2006, scope["simulation.end_year"] + 1) for m in range(1, 13)]


def select_initial_firms(pool, config):
    from market_setup import select_firms

    return select_firms(pool, config)


def integer(value):
    if hasattr(value, "item"):
        value = value.item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value))
        or (value != int(value))
        or (value < 0)
    ):
        raise ValueError(f"Invalid count: {value!r}")
    return int(value)


def snapshot(env, phase):
    sm = env.state_manager
    ids = list(env.firm_ids)
    if len(set(ids)) != len(ids) or set(ids) != set(sm.firms):
        raise ValueError("Firm tensors and the firm register hold different identities")
    employment = env.firm_employment.detach().cpu().tolist()
    targets = env.firm_target_size.detach().cpu().tolist()
    active = env.firm_active.detach().cpu().tolist()
    if not len(ids) == len(employment) == len(targets) == len(active):
        raise ValueError("Firm tensors differ in size")
    workers, rows = (Counter(), [])
    for i, fid in enumerate(ids):
        fs = sm.firms[fid]
        firm = fs.firm
        if str(firm.id) != str(fid):
            raise ValueError("Firm object/registry ID mismatch")
        roster = sorted((str(w) for w in fs.employees))
        workers.update(roster)
        exposure = env._trade_war_regime.resolve_exposure(firm)
        code = str(getattr(firm, "naics_6digit", "")).removesuffix(".0")
        valid_code = len(code) == 6 and code.isdigit()
        rows.append(
            {
                "firm_id": str(fid),
                "naics_6digit": code,
                "naics6_present": valid_code,
                "industry": int(firm.industry),
                "cma": int(firm.cma),
                "da_id": str(firm.da_id),
                "rule": f"{exposure.scope}:{exposure.code}",
                "product_group": exposure.product_group,
                "active": bool(active[i]),
                "employee_ids": roster,
                "headcount": len(roster),
                "tensor_employment": integer(employment[i]),
                "target_size": integer(firm.target_size),
                "tensor_target_size": integer(targets[i]),
            }
        )
    rows.sort(key=lambda r: r["firm_id"])
    return {
        "phase": phase,
        "year": int(sm.current_year),
        "month": int(sm.current_month),
        "firms": rows,
        "duplicate_roster_workers": sorted((w for w, n in workers.items() if n > 1)),
        "missing_roster_workers": sorted(set(workers) - {str(w) for w in sm.workers}),
        "workers_total": len(sm.workers),
    }


def capture_initial(env):
    from market_mechanisms import begin_rollout

    begin_rollout(env)
    if env._net_population is not None and env._net_population.initial_cohorts is None:
        env._net_population.capture_initial(env)
    env._market_snapshot_data = {"initial": {}, "monthly": []}


def capture_monthly(env):
    data = getattr(env, "_market_snapshot_data", None)
    if data is None:
        raise ValueError("The monthly run record was not started after reset")
    n = len(data["monthly"])
    calendar = periods()
    if n >= len(calendar):
        raise ValueError("The monthly run record has a missing or repeated month")
    data["monthly"].append(
        {"year": int(env.state_manager.current_year), "month": int(env.state_manager.current_month)}
    )


def export(env, completed_months):
    data = env._market_snapshot_data
    if completed_months != len(periods()) or len(data["monthly"]) != len(periods()):
        raise ValueError("The run stopped before the end of its horizon")
    if len(data["monthly"]) != completed_months:
        raise ValueError("The monthly run record is incomplete")
    data["partial_structural_diagnostic"] = False
    data["market_mechanisms"] = {
        "replacement_events": env._market_replacements,
        "phase_flows": env._market_flows,
        "retirement_is_absorbing": True,
        "population_representative": False,
    }
    from population_participation import export as population_export

    data["population_participation"] = population_export(env, completed_months)
    return data
