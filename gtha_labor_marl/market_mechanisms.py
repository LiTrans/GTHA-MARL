from collections import Counter, defaultdict
from copy import copy
import csv
from functools import lru_cache
import hashlib
import json
import math
from environment.trade_war_regime import TradeWarRegime
from market_setup import data_root

HOME = data_root()
POOL = HOME / "Data_Agents/firm_population/firms_2006.csv"


def active():
    return True


def number(value):
    if hasattr(value, "item"):
        value = value.item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value))
        or (value < 0)
        or (value != int(value))
    ):
        raise ValueError(f"Invalid nonnegative count: {value!r}")
    return int(value)


@lru_cache(maxsize=1)
def source_profiles():
    by_industry, code_industries, rule_employment = (defaultdict(list), defaultdict(set), Counter())
    with POOL.open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            industry = int(raw["naics_18"])
            industry = 5 if industry == 6 else industry
            code = str(raw["naics_6digit"]).removesuffix(".0")
            if len(code) != 6 or not code.isdigit():
                raise ValueError("Invalid source NAICS")
            row = {
                "source_firm_id": raw["firm_id"],
                "industry": industry,
                "naics6": code,
                "cma": int(raw["cma"]),
                "da_id": int(raw["da_id"]),
                "estimated_employees": float(raw["estimated_employees"]),
            }
            if not math.isfinite(row["estimated_employees"]) or row["estimated_employees"] < 0:
                raise ValueError("Invalid source employment")
            by_industry[industry].append(row)
            code_industries[code].add(industry)
            rule_employment[industry, rule_for(code, industry)] += row["estimated_employees"]
    for rows in by_industry.values():
        rows.sort(key=lambda r: r["source_firm_id"])
    if any((len(v) != 1 for v in code_industries.values())):
        raise ValueError("Source NAICS-to-broad-industry correspondence is ambiguous")
    return (
        dict(by_industry),
        {k: next(iter(v)) for k, v in code_industries.items()},
        dict(rule_employment),
    )


def rule_for(code, industry):
    for prefix in ("3361", "3362", "3363", "331", "332"):
        if code.startswith(prefix):
            return "naics6_prefix:" + prefix
    return "naics18:5" if industry == 5 else "default:*"


def validate_identity(firm, mapping=None):
    mapping = source_profiles()[1] if mapping is None else mapping
    code = str(getattr(firm, "naics_6digit", "")).removesuffix(".0")
    if code not in mapping or int(firm.industry) != mapping[code]:
        raise ValueError(
            f"Incoherent firm identity: {firm.id}, NAICS={code}, industry={firm.industry}"
        )


class ValidatedTradeWarRegime(TradeWarRegime):

    def resolve_exposure(self, firm):
        validate_identity(firm)
        return super().resolve_exposure(firm)


def choose_profile(rows, slot, generation, year, month, population_seed=42):
    if not rows:
        raise ValueError("No source establishment for replacement industry")
    payload = json.dumps(
        ["replacement-source-v1", population_seed, str(slot), generation, year, month],
        separators=(",", ":"),
    )
    index = int(hashlib.sha256(payload.encode()).hexdigest(), 16) % len(rows)
    return rows[index]


def replace_identity(env, slot, industry, old_da):
    fs = env.state_manager.firms[slot]
    if fs.employees or fs.vacancies:
        raise ValueError("Replacement slot still has employees or vacancies")
    generations = getattr(env, "_market_generations", None)
    if generations is None:
        raise ValueError("Replacement generation ledger not initialized")
    generation = generations.get(str(slot), 0) + 1
    rows = source_profiles()[0].get(int(industry), [])
    profile = choose_profile(rows, slot, generation, env.current_year, env.current_month)
    centroids = getattr(env.matching_mechanism, "da_centroids", None)
    if centroids is not None and profile["da_id"] not in centroids:
        raise ValueError("Replacement DA is absent from the travel matrix")
    old = fs.firm
    fresh = copy(old)
    fresh.industry = int(industry)
    fresh.naics_6digit = profile["naics6"]
    fresh.cma = profile["cma"]
    fresh.da_id = profile["da_id"]
    fresh.n_employees = 0
    fresh.trade_margin_multiplier = 1.0
    fresh.trade_demand_multiplier = 1.0
    fresh.trade_productivity_multiplier = 1.0
    fresh.trade_input_cost_rate = 0.0
    fs.firm = fresh
    validate_identity(fresh)
    generations[str(slot)] = generation
    env._market_replacements.append(
        {
            "year": int(env.current_year),
            "month": int(env.current_month),
            "slot": str(slot),
            "generation": generation,
            "old_industry": int(old.industry),
            "old_naics6": str(old.naics_6digit),
            "source_firm_id": profile["source_firm_id"],
            "industry": int(industry),
            "naics6": fresh.naics_6digit,
            "cma": fresh.cma,
            "da_id": fresh.da_id,
        }
    )
    return fresh.da_id


def refresh_turnover(env, exits, entries):
    if not (exits or entries):
        return
    st = env.fast_obs.state_table
    for fid, fs in env.state_manager.firms.items():
        idx = st.firm_id_to_idx[fid]
        st.f_industry[idx] = fs.firm.industry
        st.f_sector[idx] = fs.firm.sector
        st.f_cma[idx] = fs.firm.cma
    if not env._trade_war_regime.enabled:
        return
    regime = env._trade_war_regime
    previous = regime._current_record
    if previous is None or not regime._history or regime._history[-1] is not previous:
        raise ValueError("Missing current-month tariff ledger before turnover")
    if (previous["year"], previous["month"]) != (int(env.current_year), int(env.current_month)):
        raise ValueError("Turnover month disagrees with tariff ledger")
    regime._current_key = None
    updated = regime.apply_month(env)
    if len(regime._history) < 2 or regime._history[-2] is not previous:
        raise ValueError("Unexpected tariff history mutation")
    regime._history.pop(-2)
    updated["trade_layoffs"] += previous["trade_layoffs"]
    updated["vacancies_withdrawn"] += previous["vacancies_withdrawn"]


def reentry_age_allowed(age):
    return age >= 20


def entry_cma(firm):
    return int(firm.cma)


def worker_states(env):
    result = {}
    for wid, ws in env.state_manager.workers.items():
        w = ws.worker
        status = w.employment_status
        if status not in ("Employed", "Unemployed", "NILF"):
            raise ValueError("Unknown labour-force state")
        covid = getattr(ws, "covid_state", "none")
        lf = (
            covid == "temporary_layoff"
            or status == "Employed"
            or (status == "Unemployed" and bool(w.is_searching))
        )
        result[str(wid)] = (
            int(w.age),
            int(w.cma),
            status,
            int(bool(w.is_searching)),
            int(getattr(ws, "nilf_subtype", 0)),
            int(lf),
        )
    return result


def transition_record(before, after, phase):
    counts = Counter()
    for wid in sorted(set(before) | set(after)):
        a, b = (before.get(wid), after.get(wid))
        if a == b:
            continue

        def cell(x):
            return (
                "absent"
                if x is None
                else f"age={x[0]}|cma={x[1]}|{x[2]}|search={x[3]}|subtype={x[4]}|LF={x[5]}"
            )

        counts[cell(a), cell(b)] += 1
    lf_before = sum((x[5] for x in before.values()))
    lf_after = sum((x[5] for x in after.values()))
    lf_delta = sum(
        (
            (after[w][5] if w in after else 0) - (before[w][5] if w in before else 0)
            for w in set(before) | set(after)
        )
    )
    if lf_after - lf_before != lf_delta:
        raise ValueError("Labour-force flow identity failed")
    from population_participation import phase_stocks

    return {
        "phase": phase,
        "population_before": len(before),
        "population_after": len(after),
        "lf_before": lf_before,
        "lf_after": lf_after,
        "lf_delta": lf_delta,
        "age_stocks_before": phase_stocks(before),
        "age_stocks_after": phase_stocks(after),
        "flows": [{"from": a, "to": b, "count": n} for (a, b), n in sorted(counts.items())],
    }


def begin_rollout(env):
    env._market_generations = {}
    env._market_replacements = []
    env._market_flows = []
    env._market_before = worker_states(env)
    env._market_month_origin = dict(env._market_before)
    from population_participation import begin

    begin(env)


def phase(env, label):
    before = getattr(env, "_market_before", None)
    if before is None:
        raise ValueError("The monthly flow record was not started after reset")
    after = worker_states(env)
    record = transition_record(before, after, label)
    record.update(
        year=int(env.state_manager.current_year), month=int(env.state_manager.current_month)
    )
    env._market_flows.append(record)
    env._market_before = after
    from population_participation import calendar, capture

    if label == "start_month_after_calendar_and_demographics":
        calendar(env)
    elif label == "matching_and_rewards":
        from population_participation import monthly_transitions, validate_monthly_transitions

        monthly = monthly_transitions(env._market_month_origin, after)
        validate_monthly_transitions(monthly)
        record["monthly_transitions"] = monthly
        env._market_month_origin = dict(after)
        capture(env)


def apportion(weights, total):
    if not isinstance(total, int) or total < 0 or (not weights):
        raise ValueError("Invalid allocation request")
    if any((not math.isfinite(v) or v < 0 for v in weights.values())) or sum(weights.values()) <= 0:
        raise ValueError("Invalid employment weights")
    ideal = {k: total * v / sum(weights.values()) for k, v in weights.items()}
    q = {k: math.floor(v) for k, v in ideal.items()}
    for k in sorted(q, key=lambda k: (-(ideal[k] - q[k]), k))[: total - sum(q.values())]:
        q[k] += 1
    return q
