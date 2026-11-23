from copy import deepcopy
import weakref
from capacity_budget_math import allocate_funded, fund_birth, BudgetInfeasible

VERSION = "binding_capacity_budget_v1"


class BudgetViolation(ValueError):
    pass


def enabled():
    return True


def _int(value, label, minimum=0):
    if hasattr(value, "item"):
        value = value.item()
    if type(value) is not int or value < minimum:
        raise BudgetViolation("Invalid integer " + label)
    return value


def _vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return list(value)


def _context(env):
    return {
        "year": _int(env.current_year, "year", 1),
        "calendar_month": _int(env.current_month, "month", 1),
        "step": _int(env.current_step, "step"),
    }


def _raw(env):
    ids = [str(fid) for fid in env.firm_ids]
    targets = [_int(value, "funded target") for value in _vector(env.firm_target_size)]
    active = _vector(env.firm_active)
    firms = {str(fid): fs for fid, fs in env.state_manager.firms.items()}
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(ids) != set(firms)
        or (len(ids) != len(targets))
        or (len(active) != len(ids))
        or any((type(on) is not bool for on in active))
    ):
        raise BudgetViolation("Funded slot/activity/registry shape mismatch")
    filled = []
    for fid, target, on in zip(ids, targets, active):
        fs = firms[fid]
        if _int(fs.firm.target_size, "state-manager funded quota") != target:
            raise BudgetViolation("Funded tensor/state-manager target mismatch: " + fid)
        if len(set(fs.employees)) != len(fs.employees):
            raise BudgetViolation("Duplicate resident employee in funded slot: " + fid)
        count = len(fs.employees) + 0
        if count and (not on):
            raise BudgetViolation("Inactive reservation has occupied jobs: " + fid)
        filled.append(count)
    return (ids, targets, active, firms, filled)


def _state(env):
    state = getattr(env, "_capacity_budget_state", None)
    if not isinstance(state, dict) or state.get("version") != VERSION:
        raise BudgetViolation("The capacity budget from reset is missing")
    if state["reset_epoch"] != getattr(env, "_capacity_lineage_reset_epoch", None):
        raise BudgetViolation(
            "The capacity budget and the capacity lineage come from different resets"
        )
    if (
        getattr(env, "_capacity_reference_total", None) != state["reference_capacity"]
        or getattr(env, "_capacity_reference_population", None) != state["reference_population"]
    ):
        raise BudgetViolation("Within-reset funded capacity reference changed")
    return state


def _claim_components(env, ids, active, firms, filled):
    residents = [len(firms[fid].employees) for fid in ids]
    incoming = [count - resident for count, resident in zip(filled, residents)]
    reserve_fn = getattr(env, "_covid_reserved_positions", None)
    covid = [
        _int(reserve_fn(fid), "COVID reserved positions") if reserve_fn is not None else 0
        for fid in ids
    ]
    if any((reserve and (not on) for reserve, on in zip(covid, active))):
        raise BudgetViolation("Inactive reservation has a COVID recall claim")
    effective = [count + reserve for count, reserve in zip(filled, covid)]
    return {
        "residents": residents,
        "incoming": incoming,
        "covid_reserved": covid,
        "effective_floors": effective,
    }


def _event(env, kind, **values):
    state = _state(env)
    item = {
        "sequence": len(state["journal"]),
        "kind": kind,
        "source_context": _context(env),
        **deepcopy(values),
    }
    state["journal"].append(item)
    return item


def _lineage(env):
    import capacity_lineage

    return (capacity_lineage, capacity_lineage._state(env))


def _publish(env):
    lineage, _ = _lineage(env)
    lineage._publish(env, primary=bool(getattr(env, "_person_trace_active", False)))


def audit(env):
    state = _state(env)
    ids, targets, active, firms, filled = _raw(env)
    claims = _claim_components(env, ids, active, firms, filled)
    if ids != state["slot_ids"]:
        raise BudgetViolation("The order of funded slots changed")
    pending = state["pending_reserved"]
    if any((fid not in ids for fid in pending)):
        raise BudgetViolation("Pending birth reservation has unknown slot")
    reserved = sum(
        (_int(record["funded_target"], "pending birth quota") for record in pending.values())
    )
    for fid in pending:
        if targets[ids.index(fid)] != 0:
            raise BudgetViolation(
                "The quota of a prepared entry was written before the entry was committed"
            )
    if sum(targets) + reserved != state["budget"]:
        raise BudgetViolation("Funded quotas plus pending reservations do not equal budget")
    if any((count > target for count, target in zip(claims["effective_floors"], targets))):
        raise BudgetViolation("Occupied employment exceeds a funded firm quota")
    return {
        "status": "PASS_BINDING_CAPACITY_BUDGET",
        "version": VERSION,
        "global_budget": state["budget"],
        "reference_capacity": state["reference_capacity"],
        "reference_population": state["reference_population"],
        "funded_total": sum(targets),
        "funded_active_capacity": sum((target for target, on in zip(targets, active) if on)),
        "funded_inactive_capacity": sum((target for target, on in zip(targets, active) if not on)),
        "pending_birth_reservations": reserved,
        "pending_birth_count": len(pending),
        "occupied_total": sum(filled),
        "resident_filled_total": sum(claims["residents"]),
        "incoming_filled_total": sum(claims["incoming"]),
        "covid_reserved_total": sum(claims["covid_reserved"]),
        "effective_filled_claims": sum(claims["effective_floors"]),
        "unfunded_birth_request_count": len(state["eligibility_weights"]),
        "reset_epoch": state["reset_epoch"],
        "slot_count": len(ids),
    }


def initialize(env):
    ids, targets, active, firms, filled = _raw(env)
    claims = _claim_components(env, ids, active, firms, filled)
    lineage, lineage_state = _lineage(env)
    population = _int(env.n_workers, "reset resident population", 1)
    budget = sum(targets)
    if any((count > target for count, target in zip(claims["effective_floors"], targets))):
        raise BudgetViolation("Reset occupied jobs exceed initial funded quotas")
    env._capacity_reference_total = budget
    env._capacity_reference_population = population
    env._capacity_budget_state = {
        "version": VERSION,
        "reset_epoch": lineage_state["reset_epoch"],
        "slot_ids": ids,
        "reference_capacity": budget,
        "reference_population": population,
        "budget": budget,
        "pending_reserved": {},
        "eligibility_weights": {},
        "index_calls": 0,
        "journal": [],
    }
    env.state_manager._capacity_budget_env = weakref.ref(env)
    _event(
        env,
        "FUNDED_BUDGET_RESET",
        global_budget=budget,
        reference_population=population,
        reference_capacity=budget,
        inactive_reservation=sum((target for target, on in zip(targets, active) if not on)),
    )
    lineage._event(
        lineage_state,
        "BINDING_BUDGET_RESET",
        global_budget=budget,
        reference_capacity=budget,
        reference_population=population,
        source_context=_context(env),
    )
    _publish(env)
    return audit(env)


def remaining_slots(env, firm_id):
    _state(env)
    fid = str(firm_id)
    ids, targets, active, _, filled = _raw(env)
    if fid not in ids:
        raise BudgetViolation("Unknown funded firm slot: " + fid)
    index = ids.index(fid)
    if not active[index]:
        return 0
    reserved_fn = getattr(env, "_covid_reserved_positions", None)
    covid_reserved = (
        _int(reserved_fn(fid), "COVID reserved positions") if reserved_fn is not None else 0
    )
    return max(0, targets[index] - filled[index] - covid_reserved)


def trim_vacancies(env, reason="capacity_budget_withdrawal"):
    _state(env)
    if not isinstance(reason, str) or not reason:
        raise BudgetViolation("Vacancy withdrawal context must be explicit")
    ids, _, _, firms, _ = _raw(env)
    remove = []
    for fid in ids:
        vacancies = sorted(firms[fid].vacancies, key=str)
        remove.extend(vacancies[remaining_slots(env, fid) :])
    if remove:
        env.state_manager.remove_vacancies(remove, reason="capacity_budget_withdrawal")
        _event(
            env,
            "FUNDED_VACANCIES_WITHDRAWN",
            context=reason,
            vacancy_ids=[str(value) for value in remove],
            count=len(remove),
        )
    return {
        "count": len(remove),
        "vacancy_ids": [str(value) for value in remove],
        "reason": "capacity_budget_withdrawal",
    }


def _write_targets(env, desired, kind):
    state = _state(env)
    ids, old, active, firms, filled = _raw(env)
    floors = _claim_components(env, ids, active, firms, filled)["effective_floors"]
    if len(desired) != len(ids) or any((type(value) is not int or value < 0 for value in desired)):
        raise BudgetViolation("Invalid funded target write vector")
    if any((value < floor for value, floor in zip(desired, floors))):
        raise BudgetViolation("Funded target write would remove occupied jobs")
    lineage, lineage_state = _lineage(env)
    lineage._check_current(env)
    event_sequence = len(state["journal"])
    writes = []
    for index, fid in enumerate(ids):
        if old[index] == desired[index]:
            continue
        record = lineage_state["slots"][fid]
        write = {
            "slot_id": fid,
            "capacity_lineage_id": record["capacity_lineage_id"],
            "target_before": old[index],
            "target_after": desired[index],
            "budget_event_sequence": event_sequence,
            "source_kind": kind,
            "reference_capacity": state["reference_capacity"],
            "reference_population": state["reference_population"],
            "source_context": _context(env),
        }
        firms[fid].firm.target_size = desired[index]
        env.firm_target_size[index] = desired[index]
        if getattr(env, "firm_base_size", None) is not None:
            env.firm_base_size[index] = desired[index]
        record["current_target"] = desired[index]
        if desired[index] == 0 and old[index] > 0:
            record["target_origin"] = (
                "BUDGET_INDEX_ZERO" if kind == "ANNUAL_INDEX" else "BUDGET_FUNDING_ZERO"
            )
            record["official_index_zero_evidence"] = deepcopy(write)
        elif desired[index] > 0:
            record["target_origin"] = "BUDGET_FUNDED_POSITIVE"
            record["official_index_zero_evidence"] = None
        writes.append(write)
    event = _event(env, "FUNDED_TARGETS_WRITTEN", authority=kind, writes=writes)
    lineage._event(
        lineage_state,
        "BINDING_BUDGET_TARGETS_WRITTEN",
        budget_event_sequence=event["sequence"],
        authority=kind,
        source_context=_context(env),
        writes=deepcopy(writes),
    )
    return writes


def request_weight(env, firm_id):
    state = _state(env)
    return _int(state["eligibility_weights"].get(str(firm_id), 0), "pending request weight")


def authenticate_zero(env, firm_id, evidence):
    state = _state(env)
    fid = str(firm_id)
    if not isinstance(evidence, dict) or evidence.get("slot_id") != fid:
        return False
    sequence = evidence.get("budget_event_sequence")
    if type(sequence) is not int or sequence < 0 or sequence >= len(state["journal"]):
        return False
    event = state["journal"][sequence]
    return (
        event.get("kind") == "FUNDED_TARGETS_WRITTEN"
        and evidence in event.get("writes", [])
        and (evidence.get("target_before", 0) > 0)
        and (evidence.get("target_after") == 0)
    )


def prepare_entry(env, firm_id, request):
    state = _state(env)
    audit(env)
    fid = str(firm_id)
    ids, targets, active, firms, filled = _raw(env)
    effective = _claim_components(env, ids, active, firms, filled)["effective_floors"]
    if fid not in ids or fid in state["pending_reserved"]:
        raise BudgetViolation("Unknown or repeated funded birth request")
    index = ids.index(fid)
    requested = _int(request, "native capped birth request", 1)
    if requested > 20:
        raise BudgetViolation("A firm entry requests more than the unchanged cap of 20")
    old_request = request_weight(env, fid)
    if old_request and old_request != requested:
        raise BudgetViolation("Pending birth request cannot be redrawn or increased")
    result = fund_birth(targets, effective, active, index, requested, state["budget"])
    donor_targets = list(result["targets"])
    donor_targets[index] = 0
    state["pending_reserved"][fid] = {
        "funded_target": result["funded_target"],
        "requested_target": requested,
        "source_context": _context(env),
    }
    if result["shortfall"]:
        state["eligibility_weights"][fid] = requested
    else:
        state["eligibility_weights"].pop(fid, None)
    _write_targets(env, donor_targets, "BIRTH_TRANSFER")
    withdrawals = trim_vacancies(env, reason="birth_donor_quota_transfer")
    _event(
        env,
        "BIRTH_FUNDED_FROM_ACTIVE_SLACK",
        slot_id=fid,
        requested_target=requested,
        funded_target=result["funded_target"],
        shortfall=result["shortfall"],
        donor_debits=result["donor_debits"],
        pending_reserved=result["funded_target"],
        vacancy_withdrawals=withdrawals,
    )
    audit(env)
    _publish(env)
    return {**result, "capacity_budget_vacancy_withdrawals": withdrawals}


def commit_entry(env, firm_id, final_target):
    state = _state(env)
    fid = str(firm_id)
    ids, targets, active, _, filled = _raw(env)
    if fid not in ids:
        raise BudgetViolation("Unknown committed funded slot")
    index = ids.index(fid)
    final_target = _int(final_target, "committed funded quota")
    pending = state["pending_reserved"].get(fid)
    if targets[index] != final_target or not active[index] or filled[index]:
        raise BudgetViolation("A committed funded entry does not start from empty stocks")
    if pending is not None:
        if pending["source_context"] != _context(env) or final_target != pending["funded_target"]:
            raise BudgetViolation("The committed entry differs from its pending reservation")
        del state["pending_reserved"][fid]
    _event(
        env,
        "FUNDED_ENTRY_COMMITTED",
        slot_id=fid,
        final_target=final_target,
        new_budget_capacity_created=0,
        reservation_consumed=final_target if pending is not None else 0,
        pending_request_weight=request_weight(env, fid),
    )
    return audit(env)


def _minimum_resident_contraction(env, desired):
    state = _state(env)
    ids, _, active, firms, filled = _raw(env)
    claims = _claim_components(env, ids, active, firms, filled)
    protected = [
        incoming + reserve
        for incoming, reserve in zip(claims["incoming"], claims["covid_reserved"])
    ]
    effective_total = sum(claims["effective_floors"])
    if effective_total <= desired:
        return []
    if sum(protected) > desired:
        error = BudgetInfeasible(budget=desired, filled_total=effective_total)
        error.context = {
            "reason": "PROTECTED_INCOMING_AND_RECALL_CLAIMS_EXCEED_BUDGET",
            "incoming_filled": sum(claims["incoming"]),
            "covid_reserved": sum(claims["covid_reserved"]),
            "source_context": _context(env),
        }
        error.args = (str(error) + "; unsupported=" + repr(error.context),)
        raise error
    retained = allocate_funded(
        desired - sum(protected), [0] * len(ids), claims["residents"], active
    )
    separations = []
    records = []
    temporary = getattr(env.state_manager, "_ordinary_temporary_nilf_ids_this_step", set())
    for index, fid in enumerate(ids):
        excess = claims["residents"][index] - retained[index]
        if excess < 0:
            raise BudgetViolation("Resident retention exceeded the actual roster")
        for wid in list(firms[fid].employees)[-excess:] if excess else []:
            ws = env.state_manager.workers.get(wid)
            if (
                ws is None
                or ws.worker.employment_status != "Employed"
                or str(ws.worker.current_employer_id) != fid
                or (wid in temporary)
            ):
                raise BudgetViolation(
                    "Capacity contraction resident has inconsistent source status"
                )
            separations.append((wid, fid, "layoff"))
            records.append(
                {
                    "worker_id": str(wid),
                    "firm_id": fid,
                    "source_context": _context(env),
                    "reason": "binding_capacity_budget_contraction",
                }
            )
    if len(separations) != effective_total - desired or len({w for w, _, _ in separations}) != len(
        separations
    ):
        raise BudgetViolation("Minimum resident contraction count does not equal required deficit")
    sm = env.state_manager
    missing = object()
    prior_market_count = sm.stats.get("separations_this_month", missing)
    try:
        sm.apply_separations(separations)
    finally:
        if prior_market_count is missing:
            sm.stats.pop("separations_this_month", None)
        else:
            sm.stats["separations_this_month"] = prior_market_count
    for wid, fid, _ in separations:
        ws = sm.workers[wid]
        if ws.worker.employment_status != "Unemployed" or wid in sm.firms[fid].employees:
            raise BudgetViolation(
                "A layoff from capacity contraction did not update the worker status or the firm roster"
            )
        ws.worker.current_employer_id = None
        ws.worker.current_wage = 0.0
        ws.worker.is_searching = True
        ws.worker.unemployment_months = 0
        getattr(env, "_worker_firm_map", {}).pop(wid, None)
    _event(
        env,
        "MINIMUM_RESIDENT_CONTRACTION_LAYOFFS",
        global_budget_before=state["budget"],
        requested_budget=desired,
        layoffs=records,
        resident_filled_before=sum(claims["residents"]),
        incoming_filled_protected=sum(claims["incoming"]),
        covid_recall_protected=sum(claims["covid_reserved"]),
        effective_claims_before=effective_total,
        existing_layoff_bookkeeping_calls=len(separations),
    )
    return records


def index(env, before_population, after_population):
    state = _state(env)
    _int(before_population, "pre-renewal population", 1)
    population = _int(after_population, "post-renewal population", 1)
    audit(env)
    if state["pending_reserved"]:
        raise BudgetViolation("Annual funded index overlaps an uncommitted birth")
    lineage, lineage_state = _lineage(env)
    if lineage_state["pending_entries"]:
        raise BudgetViolation("The annual funded index was applied while an entry is pending")
    ids, targets, active, firms, filled = _raw(env)
    claims_before = _claim_components(env, ids, active, firms, filled)
    from regional_capacity import OFFICIAL_ELASTICITY

    desired = round(
        state["reference_capacity"]
        * (1 + OFFICIAL_ELASTICITY * (population / state["reference_population"] - 1))
    )
    contraction_layoffs = _minimum_resident_contraction(env, desired)
    ids, _, active, firms, filled = _raw(env)
    claims = _claim_components(env, ids, active, firms, filled)
    weights = [max(target, request_weight(env, fid)) for fid, target in zip(ids, targets)]
    try:
        updated = allocate_funded(desired, claims["effective_floors"], weights, active)
    except BudgetInfeasible as error:
        error.context = {
            "reference_capacity": state["reference_capacity"],
            "reference_population": state["reference_population"],
            "population_before": before_population,
            "population_after": population,
            "reset_epoch": state["reset_epoch"],
            "global_budget_before": state["budget"],
            "source_context": _context(env),
        }
        error.args = (str(error) + "; context=" + repr(error.context),)
        raise
    old_budget = state["budget"]
    _write_targets(env, updated, "ANNUAL_INDEX")
    state["budget"] = desired
    state["index_calls"] += 1
    lineage_state["index_calls"] += 1
    lineage_state["index_reference"] = (
        state["reference_capacity"],
        state["reference_population"],
        OFFICIAL_ELASTICITY,
    )
    for fid, target in zip(ids, updated):
        requested = request_weight(env, fid)
        if requested and target >= requested:
            del state["eligibility_weights"][fid]
    withdrawals = trim_vacancies(env, reason="annual_funded_budget_index")
    result = {
        "mode": "binding_population_capacity_budget_v1",
        "population_before": before_population,
        "population_after": population,
        "reference_population": state["reference_population"],
        "reference_capacity": state["reference_capacity"],
        "official_elasticity": OFFICIAL_ELASTICITY,
        "active_capacity_before": sum((target for target, on in zip(targets, active) if on)),
        "active_capacity_after": sum((target for target, on in zip(updated, active) if on)),
        "global_budget_before": old_budget,
        "global_budget": desired,
        "funded_active_capacity": sum((target for target, on in zip(updated, active) if on)),
        "funded_inactive_capacity": sum((target for target, on in zip(updated, active) if not on)),
        "pending_birth_reservations": 0,
        "capacity_budget_vacancy_withdrawals": withdrawals,
        "capacity_budget_contraction_layoffs": contraction_layoffs,
        "resident_filled_before_index": sum(claims_before["residents"]),
        "resident_filled_after_index": sum(claims["residents"]),
        "incoming_filled_protected": sum(claims["incoming"]),
        "covid_recall_protected": sum(claims["covid_reserved"]),
        "unemployed_search_pressure_by_industry": None,
    }
    _event(env, "FUNDED_ANNUAL_INDEX_COMMITTED", receipt=result)
    lineage._event(
        lineage_state,
        "BINDING_BUDGET_INDEX_APPLIED",
        receipt=deepcopy(result),
        source_context=_context(env),
    )
    audit(env)
    _publish(env)
    return result


def receipt(env):
    result = deepcopy(_state(env))
    result["audit"] = audit(env)
    result["RNG_draws_by_helper"] = 0
    result["existing_layoff_bookkeeping_calls"] = sum(
        (event.get("existing_layoff_bookkeeping_calls", 0) for event in result["journal"])
    )
    result["new_parameters_selected"] = False
    return result
