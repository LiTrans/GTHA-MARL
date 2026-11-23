from collections import Counter
import math

CMAS = (532, 535, 537)
BANDS = (
    ("15-24", 15, 24),
    ("25-34", 25, 34),
    ("35-44", 35, 44),
    ("45-54", 45, 54),
    ("55-64", 55, 64),
    ("65+", 65, math.inf),
)
DETAIL_BANDS = tuple(((f"{a}-{a + 4}", a, a + 4) for a in range(15, 85, 5))) + (
    ("85+", 85, math.inf),
)
SLID_NILF_COMPETING_RISKS = {
    "16_24": (2096 / 41939, 1492 / 41939),
    "25_54": (838 / 60378, 864 / 60378),
    "55_59": (85 / 21414, 128 / 21414),
    "60_64": (69 / 31866, 86 / 31866),
    "65_69": (40 / 41291, 46 / 41291),
}


def active():
    return True


def band(age):
    if not isinstance(age, (int, float)) or not math.isfinite(age) or age != int(age) or (age < 0):
        raise ValueError("Invalid age")
    return next((name for name, lo, hi in BANDS if lo <= age <= hi), "under15")


def pre2025_scope():
    from market_snapshots import plan

    p = plan()
    end = int(p["end_year"])
    return {
        "simulation.start_year": 2006,
        "simulation.end_year": end,
        "simulation.max_episode_years": end - 2005,
        "training.fix_retirement_channel_dedup": True,
        "mnl.prime_age_e_to_n_rate": 0.0,
        "mnl.u_to_nilf_struct_rate_prime": 0.0,
        "mnl.u_to_nilf_struct_rate_older": 0.0,
    }


def configure_participation(config):
    from market_snapshots import plan

    expected = {
        "reentry": "slid_2006_2011_age_competing_risks_v1",
        "retirement": "csv_55_59_learned_60_plus",
    }
    if plan().get("learned_participation_rule") is not None:
        from learned_participation import configure

        expected["retirement"] = "learned_55_plus_absorbing_v2"
        configure(config, plan())
    if plan().get("participation_decisions") != expected:
        raise ValueError("The participation settings are missing or differ from the expected ones")
    if config.reward.retirement_age_min != 60:
        raise ValueError("Retirement de-duplication requires learned retirement from age 60")
    config.mnl.allow_retirement_reentry = False
    config.mnl.prime_age_e_to_n_rate = 0.0
    config.mnl.u_to_nilf_struct_rate_prime = 0.0
    config.mnl.u_to_nilf_struct_rate_older = 0.0


def reentry_destination_probabilities(ws, config, inherited=0.0):
    if ws.worker.employment_status != "NILF" or ws.worker.age < 20:
        raise ValueError("Ineligible worker submitted for NILF re-entry")
    if getattr(ws, "covid_state", "none") != "none":
        raise ValueError("COVID-restricted worker submitted for re-entry")
    if int(ws.nilf_subtype) == 1 or int(ws.worker.age) >= 70:
        return (0.0, 0.0)
    age = int(ws.worker.age)
    key = (
        "16_24"
        if age < 25
        else "25_54" if age < 55 else "55_59" if age < 60 else "60_64" if age < 65 else "65_69"
    )
    return SLID_NILF_COMPETING_RISKS[key]


def phase_stocks(states):
    cells = {
        (c, b): Counter(P=0, E=0, U=0, N=0, LF=0, retired=0)
        for c in CMAS
        for b, _, _ in DETAIL_BANDS
    }
    for age, cma, status, search, subtype, lf in states.values():
        if cma not in CMAS or age < 15 or lf not in (0, 1):
            raise ValueError("Phase stock outside declared adult population")
        age_group = next((b for b, lo, hi in DETAIL_BANDS if lo <= age <= hi))
        cell = cells[cma, age_group]
        outcome = "E" if status == "Employed" else "U" if lf else "N"
        cell["P"] += 1
        cell[outcome] += 1
        cell["LF"] += lf
        cell["retired"] += int(outcome == "N" and subtype == 1)
    return [{"cma": c, "age_group": b, **v} for (c, b), v in cells.items()]


def monthly_transitions(before, after):
    cells = {
        (c, b): Counter(
            {**{a + z: 0 for a in "EUN" for z in "EUN"}, **{a + "_departures": 0 for a in "EUN"}}
        )
        for c in CMAS
        for b, _, _ in DETAIL_BANDS
    }
    arrivals = 0
    for wid, origin in before.items():
        age, cma, status, search, subtype, lf = origin
        b = next((b for b, lo, hi in DETAIL_BANDS if lo <= age <= hi))
        initial = "E" if status == "Employed" else "U" if lf else "N"
        cell = cells[cma, b]
        if wid not in after:
            cell[initial + "_departures"] += 1
        else:
            target = after[wid]
            final = "E" if target[2] == "Employed" else "U" if target[5] else "N"
            cell[initial + final] += 1
    arrivals = len(set(after) - set(before))
    return {
        "definition": "previous_month_end_to_current_month_end; January 2006 origin is initialized state",
        "age_and_cma": "origin state; entrants/departures not labour-force transitions",
        "origin_stocks": phase_stocks(before),
        "destination_stocks": phase_stocks(after),
        "arrival_stocks": phase_stocks({w: s for w, s in after.items() if w not in before}),
        "arrivals": arrivals,
        "cells": [{"cma": c, "age_group": b, **v} for (c, b), v in cells.items()],
    }


def validate_monthly_transitions(record):
    keys = {(c, b) for c in CMAS for b, _, _ in DETAIL_BANDS}

    def indexed(rows):
        if len(rows) != len(keys) or {(r["cma"], r["age_group"]) for r in rows} != keys:
            raise ValueError("Incomplete monthly transition support")
        return {(r["cma"], r["age_group"]): r for r in rows}

    origin = indexed(record["origin_stocks"])
    destination = indexed(record["destination_stocks"])
    cells = indexed(record["cells"])
    arrival_stocks = indexed(record["arrival_stocks"])
    survivors = departures = 0

    def count(v):
        if type(v) is not int or v < 0:
            raise ValueError("Invalid monthly transition count")
        return v

    for k, cell in cells.items():
        for a in "EUN":
            stayed = sum((count(cell[a + z]) for z in "EUN"))
            left = count(cell[a + "_departures"])
            if stayed + left != origin[k][a]:
                raise ValueError("Monthly transition origin denominator mismatch")
            survivors += stayed
            departures += left
    if survivors + count(record["arrivals"]) != sum((r["P"] for r in destination.values())):
        raise ValueError("Monthly transition destination population mismatch")
    if survivors + departures != sum((r["P"] for r in origin.values())):
        raise ValueError("Monthly transition origin population mismatch")
    if sum((r["P"] for r in arrival_stocks.values())) != record["arrivals"]:
        raise ValueError("Monthly arrival stocks disagree")
    for z in "EUN":
        arrivals = sum((r[z] for r in arrival_stocks.values()))
        matched = sum((cell[a + z] for cell in cells.values() for a in "EUN"))
        if matched + arrivals != sum((r[z] for r in destination.values())):
            raise ValueError("Monthly transition destination status mismatch")
    return True


def bucket(ws):
    w = ws.worker
    status = w.employment_status
    if status not in ("Employed", "Unemployed", "NILF"):
        raise ValueError("Unknown employment status")
    covid = getattr(ws, "covid_state", "none")
    if covid not in ("none", "temporary_layoff", "marginally_attached"):
        raise ValueError("Unknown COVID status")
    if covid == "temporary_layoff":
        if status == "Employed":
            raise ValueError("Employed while temporarily laid off")
        return "U"
    if status == "Employed":
        return "E"
    if w.is_searching not in (True, False):
        raise ValueError("Invalid search flag")
    return "U" if status == "Unemployed" and w.is_searching else "N"


def begin(env):
    env._population_ledger = {
        "schema_version": 1,
        "decision_bearing": False,
        "monthly": [],
        "calendar": [],
        "replacement_events": [],
        "reentry_funnel": [],
    }
    env._population_before = {
        str(k): (int(v.worker.age), int(v.worker.cma)) for k, v in env.state_manager.workers.items()
    }
    env._population_period = (
        int(env.state_manager.current_year),
        int(env.state_manager.current_month),
    )


def ledger(env):
    if not hasattr(env, "_population_ledger"):
        raise ValueError("Population ledger missing after reset")
    return env._population_ledger


def calendar(env):
    current = (int(env.state_manager.current_year), int(env.state_manager.current_month))
    prev = env._population_period
    distance = 12 * (current[0] - prev[0]) + current[1] - prev[1]
    if distance not in (0, 1) or (distance == 0 and ledger(env)["calendar"]):
        raise ValueError("Repeated or skipped population calendar")
    now = {
        str(k): (int(v.worker.age), int(v.worker.cma)) for k, v in env.state_manager.workers.items()
    }
    before = env._population_before
    if any((now[k][0] - before[k][0] != current[0] - prev[0] for k in now.keys() & before.keys())):
        raise ValueError("Surviving worker aged more/less than one calendar tick")
    entered, exited = (sorted(now.keys() - before.keys()), sorted(before.keys() - now.keys()))
    if len(now) != len(before) + len(entered) - len(exited):
        raise ValueError("Population balance failure")
    ledger(env)["calendar"].append(
        {
            "year": current[0],
            "month": current[1],
            "population_before": len(before),
            "population_after": len(now),
            "entered_ids": entered,
            "exited_ids": exited,
            "survivors_checked": len(now.keys() & before.keys()),
            "age_increment": current[0] - prev[0],
            "interpretation": "registry entries/exits, not identified gross immigration/emigration",
        }
    )
    env._population_before, env._population_period = (now, current)


def remove_replaced_worker(env, old_id):
    ws = env.state_manager.workers[old_id]
    memberships = [fid for fid, fs in env.state_manager.firms.items() if old_id in fs.employees]
    employer = getattr(ws.worker, "current_employer_id", None)
    if (
        len(memberships) > 1
        or (ws.worker.employment_status == "Employed" and memberships != [employer])
        or (ws.worker.employment_status != "Employed" and memberships)
    ):
        raise ValueError("The firm of a replaced worker has inconsistent employment records")
    for fid in memberships:
        roster = env.state_manager.firms[fid].employees
        if roster.count(old_id) != 1:
            raise ValueError("Duplicate departing roster entry")
        roster.remove(old_id)
    env._worker_firm_map.pop(old_id, None)


def prepare_replacement(env, ws, old_id, source, donor=None):
    w = ws.worker
    status, wage = (w.employment_status, float(w.current_wage))
    if status not in ("Employed", "Unemployed", "NILF") or not math.isfinite(wage):
        raise ValueError("Invalid replacement donor state")
    if getattr(w, "current_employer_id", None) is not None:
        raise ValueError("Replacement unexpectedly has an employer")
    if any((w.id in fs.employees for fs in env.state_manager.firms.values())):
        raise ValueError("Replacement is already on a roster")
    if donor is not None:
        raw = donor.get("immigrant_status", donor.get("immigrant", False))
        if str(raw).lower() not in ("true", "false", "1", "0", "1.0", "0.0"):
            raise ValueError("Missing/invalid donor immigration status")
        w.immigrant_status = str(raw).lower() in ("true", "1", "1.0")
        w.archetype_id = int(donor["archetype_id"])
        w.archetype_name = str(donor["archetype_name"])
        years = donor.get("years_since_arrival")
        w.years_since_arrival = (
            float(years) if years is not None and math.isfinite(float(years)) else None
        )
    else:
        w.immigrant_status = False
        w.years_since_arrival = None
        w.archetype_id = 2 if w.skill_level >= 3 else 1 if w.skill_level == 2 else 0
        w.archetype_name = ("native_low_skill", "native_medium_skill", "native_high_skill")[
            w.archetype_id
        ]
    w.employment_status = "Unemployed" if status in ("Employed", "Unemployed") else "NILF"
    w.is_searching = w.employment_status == "Unemployed"
    w.current_wage = 0.0
    w.current_employer_id = None
    w.unemployment_months = 0
    ws.months_employed = ws.months_unemployed = ws.months_nilf = 0
    ws.previous_wage = 0.0
    ws.previous_tenure_months = 0
    ws.ei_eligible = False
    ws.ei_weeks_remaining = 0
    ws.separation_reason = "re_entrant"
    ws.lwf_retention_draw = None
    ledger(env)["replacement_events"].append(
        {
            "year": int(env.current_year),
            "month": int(env.current_month),
            "old_id": str(old_id),
            "new_id": str(w.id),
            "source": source,
            "donor_status": status,
            "donor_wage": wage,
            "entry_status": w.employment_status,
            "age": int(w.age),
            "cma": int(w.cma),
            "donor_immigrant": bool(w.immigrant_status),
            "interpretation": "net-neutral replacement, not observed immigrant or market hire",
        }
    )


def record_funnel(env, ws, search_decision):
    from environment.nilf_subtype import is_search_eligible, effective_subtype_windows

    windows = effective_subtype_windows(env.config.mnl, int(ws.worker.age))
    eligible = is_search_eligible(
        ws.nilf_subtype,
        ws.nilf_subtype_months,
        *windows,
        allow_retirement_reentry=getattr(env.config.mnl, "allow_retirement_reentry", False),
    )
    intent = int(getattr(ws, "last_intent_search_decision", search_decision))
    reason = (
        "age_under20"
        if ws.worker.age < 20
        else (
            "retired"
            if ws.nilf_subtype == 1 and (not eligible)
            else (
                "subtype_window"
                if not eligible
                else "no_intent" if intent != 1 else "eligible_intent"
            )
        )
    )
    ledger(env)["reentry_funnel"].append(
        {
            "year": int(env.current_year),
            "month": int(env.current_month),
            "cma": int(ws.worker.cma),
            "age_group": band(ws.worker.age),
            "subtype": int(ws.nilf_subtype),
            "intent": intent,
            "reason": reason,
        }
    )


def capture(env):
    cells = {
        (c, b): Counter(
            E=0, U=0, N=0, P=0, LF=0, retired=0, temporary_nilf=0, nonsearching_U=0, orphan_E=0
        )
        for c in CMAS
        for b, _, _ in BANDS
    }
    sm = env.state_manager
    excluded = 0
    for wid, ws in sm.workers.items():
        w = ws.worker
        if str(w.id) != str(wid) or int(w.cma) not in CMAS:
            raise ValueError("Worker identity or geography outside declared cohort")
        b = band(w.age)
        if b == "under15":
            excluded += 1
            continue
        cell = cells[int(w.cma), b]
        k = bucket(ws)
        cell[k] += 1
        cell["P"] += 1
        cell["LF"] += int(k != "N")
        cell["retired"] += int(
            k == "N" and w.employment_status == "NILF" and (ws.nilf_subtype == 1)
        )
        cell["temporary_nilf"] += int(
            k == "N" and w.employment_status == "NILF" and (ws.nilf_subtype != 1)
        )
        cell["nonsearching_U"] += int(w.employment_status == "Unemployed" and k == "N")
        employer = getattr(w, "current_employer_id", None)
        if k == "E" and (employer not in sm.firms or wid not in sm.firms[employer].employees):
            boundary = getattr(env, "boundary", None)
            if boundary is not None and wid in boundary.jobs:
                boundary.validate_job(env, wid)
            else:
                cell["orphan_E"] += 1
    if sum((c["P"] for c in cells.values())) + excluded != len(sm.workers):
        raise ValueError("Joint stock partition failure")
    if any((c["orphan_E"] for c in cells.values())):
        raise ValueError("Employed worker without a firm-backed job")
    rows = ledger(env)["monthly"]
    year, month = (int(sm.current_year), int(sm.current_month))
    if (year, month) != (2006 + len(rows) // 12, 1 + len(rows) % 12):
        raise ValueError("Population stock calendar gap")
    rows.append(
        {
            "year": year,
            "month": month,
            "excluded_under15": excluded,
            "cells": [{"cma": c, "age_group": b, **v} for (c, b), v in cells.items()],
        }
    )


def export(env, months):
    value = ledger(env)
    if len(value["monthly"]) != months or len(value["calendar"]) != months:
        raise ValueError("Population ledger incomplete")
    counts = Counter((tuple(sorted(r.items())) for r in value["reentry_funnel"]))
    from market_snapshots import plan

    decisions = dict(plan()["participation_decisions"])
    reversible = getattr(env.config.mnl, "allow_retirement_reentry", False) is True
    if decisions.get("retirement") == "learned_55_plus_reversible_v1":
        from learned_participation import validate

        validate(env.config)
    elif decisions.get("retirement") == "learned_55_plus_absorbing_v2":
        from learned_participation import validate

        validate(env.config, retirement_absorbing=True)
    elif reversible:
        raise ValueError("Reversible retirement without its declared participation design")
    return {
        **value,
        "reentry_funnel": [{**dict(k), "count": v} for k, v in sorted(counts.items())],
        "retirement_absorbing": not reversible,
        "gross_demographic_flows_identified": False,
        "population_representative": False,
        "participation_decisions": decisions,
        "added_outflow_hazards": "removed_by_design",
    }
