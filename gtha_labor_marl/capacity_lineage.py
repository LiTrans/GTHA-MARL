from copy import deepcopy
from functools import wraps
import sys

VERSION = "capacity_lineage_v1"
OFFICIAL_RATIO = 0.79815636011484
_captures = []


class LineageViolation(ValueError):
    pass


def enabled():
    return True


def _int(value, label, minimum=0):
    if hasattr(value, "item"):
        value = value.item()
    if type(value) is not int or value < minimum:
        raise LineageViolation("Invalid integer " + label)
    return value


def _vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return list(value)


def _context(env):
    month = _int(env.current_month, "source month", 1)
    if month > 12:
        raise LineageViolation("Source month is outside calendar")
    return {
        "step": _int(env.current_step, "source step"),
        "year": _int(env.current_year, "source year", 1),
        "calendar_month": month,
    }


def _stocks(env):
    ids = [str(fid) for fid in env.firm_ids]
    if len(ids) != len(set(ids)):
        raise LineageViolation("Firm slot IDs collide")
    targets = [_int(v, "target") for v in _vector(env.firm_target_size)]
    active = _vector(env.firm_active)
    if (
        len(ids) != len(targets)
        or len(ids) != len(active)
        or any((type(a) is not bool for a in active))
    ):
        raise LineageViolation("Target/activity vectors differ")
    states = {str(fid): fs for fid, fs in env.state_manager.firms.items()}
    if set(states) != set(ids):
        raise LineageViolation("Firm states do not cover the firm slots")
    for fid, target, on in zip(ids, targets, active):
        fs = states[fid]
        if _int(fs.firm.target_size, "state-manager target") != target:
            raise LineageViolation("Tensor/state-manager target mismatch")
        if not on and fs.employees:
            raise LineageViolation("Inactive firm has resident employees")
    return (ids, dict(zip(ids, targets)), dict(zip(ids, active)), states)


def _state(env):
    value = getattr(env, "_capacity_lineage_state", None)
    if not isinstance(value, dict) or value.get("version") != VERSION:
        raise LineageViolation("The capacity lineage state from reset is missing")
    return value


def _check_current(env):
    state = _state(env)
    ids, targets, active, firms = _stocks(env)
    if set(state["slots"]) != set(ids):
        raise LineageViolation("Unknown capacity predecessor slot")
    for fid in ids:
        record = state["slots"][fid]
        if record["current_target"] != targets[fid] or record["lineage_closed"]:
            raise LineageViolation(
                "A capacity target changed outside the lineage record, or the current lineage is closed: "
                + fid
            )
    return (state, targets, active, firms)


def _event(state, kind, **data):
    state["journal"].append({"sequence": len(state["journal"]), "kind": kind, **data})


def _publish(env, primary=False):
    state = _state(env)
    if primary:
        state["primary_observed"] = True
    index = state["capture_index"]
    _captures[index] = deepcopy(state)


def initialize_lineages(env):
    ids, targets, active, _ = _stocks(env)
    epoch = getattr(env, "_capacity_lineage_reset_epoch", -1) + 1
    env._capacity_lineage_reset_epoch = epoch
    state = {
        "version": VERSION,
        "reset_epoch": epoch,
        "capture_index": len(_captures),
        "primary_observed": False,
        "slots": {},
        "closed_lineages": [],
        "pending_entries": {},
        "journal": [],
        "index_reference": None,
        "index_calls": 0,
        "primary_month_contexts": [],
    }
    for fid in ids:
        state["slots"][fid] = {
            "slot_id": fid,
            "capacity_lineage_id": f"{VERSION}:capture{state['capture_index']}:reset{epoch}:{fid}:lineage0",
            "capacity_generation": 0,
            "business_generation": 0,
            "current_target": targets[fid],
            "lineage_closed": False,
            "target_origin": "RESET_POSITIVE" if targets[fid] > 0 else "RESET_KNOWN_ZERO",
            "official_index_zero_evidence": None,
            "last_entry_authority": None,
        }
    env._capacity_lineage_state = state
    _captures.append(None)
    _event(
        state,
        "AUTHORITATIVE_RESET",
        slot_count=len(ids),
        active_capacity=sum((targets[fid] for fid in ids if active[fid])),
        source_context=_context(env),
        known_zero_slots=[fid for fid in ids if targets[fid] == 0],
    )
    _publish(env)
    import capacity_budget

    capacity_budget.initialize(env)
    return {"version": VERSION, "reset_epoch": epoch, "slot_count": len(ids)}


def prepare_entry(env, firm_id, predecessor_target, capped_draw):
    predecessor_target = _int(predecessor_target, "predecessor")
    capped_draw = _int(capped_draw, "already-consumed capped draw", 1)
    if capped_draw > 20:
        raise LineageViolation("A capped entrant exceeds the unchanged cap of 20")
    return _prepare_funded_entry(env, firm_id, predecessor_target, capped_draw)


def commit_entry(env, firm_id, final_target, br_scaling_applied=False):
    return _commit_funded_entry(env, firm_id, final_target, br_scaling_applied)


def initial_vacancy_count(env, firm_id, earlier_count):
    fid = str(firm_id)
    count = _int(earlier_count, "earlier initial vacancy count")
    state, _, active, _ = _check_current(env)
    record = state["slots"][fid]
    if not active[fid] or record["last_entry_authority"] is None:
        raise LineageViolation("Initial vacancy issue lacks a committed entry")
    import capacity_budget

    return min(count, capacity_budget.remaining_slots(env, fid))


def _prepare_funded_entry(env, firm_id, predecessor_target, capped_draw):
    import capacity_budget

    fid = str(firm_id)
    state, targets, active, firms = _check_current(env)
    capacity_budget.audit(env)
    if fid not in state["slots"] or fid in state["pending_entries"]:
        raise LineageViolation("Unknown or repeated funded entry slot")
    if active[fid] or firms[fid].employees or targets[fid] != predecessor_target:
        raise LineageViolation("Funded entry predecessor differs from an empty inactive stock")
    record = state["slots"][fid]
    evidence = record["official_index_zero_evidence"]
    zero_origin = record["target_origin"] in ("BUDGET_INDEX_ZERO", "BUDGET_FUNDING_ZERO")
    pending_request = capacity_budget.request_weight(env, fid)
    retry = (
        predecessor_target == 0
        and pending_request > 0
        and (record["target_origin"] == "BUDGET_PENDING_REQUEST" or zero_origin)
    )
    fresh = predecessor_target == 0 and zero_origin and (not retry)
    if (
        predecessor_target == 0
        and zero_origin
        and (
            not capacity_budget.authenticate_zero(env, fid, evidence)
            or evidence.get("capacity_lineage_id") != record["capacity_lineage_id"]
        )
    ):
        raise LineageViolation(
            "Funded zero predecessor lacks authenticated source journal evidence"
        )
    if (
        predecessor_target == 0
        and (not fresh)
        and (not retry)
        and (record["target_origin"] != "RESET_KNOWN_ZERO")
    ):
        raise LineageViolation("A funded zero target has no recorded origin")
    generation = record["capacity_generation"] + int(fresh)
    new_id = (
        f"{VERSION}:capture{state['capture_index']}:reset{state['reset_epoch']}:{fid}:lineage{generation}"
        if fresh
        else record["capacity_lineage_id"]
    )
    funding = None
    requested = capped_draw if fresh else pending_request if retry else 0
    if fresh or retry:
        funding = capacity_budget.prepare_entry(env, fid, requested)
    target = funding["funded_target"] if funding is not None else predecessor_target
    plan = {
        "slot_id": fid,
        "predecessor_target": predecessor_target,
        "old_capacity_lineage_id": record["capacity_lineage_id"],
        "new_capacity_lineage_id": new_id,
        "old_capacity_generation": record["capacity_generation"],
        "new_capacity_generation": generation,
        "business_generation": record["business_generation"] + 1,
        "target": target,
        "fresh_target_assignment": fresh,
        "authority": (
            "BINDING_BUDGET_INDEX_ZERO_NEW_LINEAGE"
            if fresh
            else (
                "RETRY_PENDING_FUNDED_BIRTH"
                if retry
                else (
                    "CONSERVE_FUNDED_POSITIVE"
                    if predecessor_target > 0
                    else "CONSERVE_RESET_KNOWN_ZERO"
                )
            )
        ),
        "candidate_draw_consumed": True,
        "capped_candidate_draw": capped_draw,
        "candidate_draw_used": fresh,
        "requested_target": requested,
        "original_request_reused": retry,
        "funding_shortfall": funding["shortfall"] if funding is not None else 0,
        "funding_donor_debits": deepcopy(funding["donor_debits"]) if funding is not None else {},
        "old_capacity_claim_extinguished": 0 if fresh else None,
        "source_context": _context(env),
        "official_index_zero_evidence": deepcopy(evidence) if fresh else None,
    }
    state["pending_entries"][fid] = plan
    _event(state, "ENTRY_PREPARED", **deepcopy(plan))
    _publish(env, primary=bool(getattr(env, "_person_trace_active", False)))
    return deepcopy(plan)


def _commit_funded_entry(env, firm_id, final_target, br_scaling_applied):
    import capacity_budget

    fid = str(firm_id)
    state = _state(env)
    if fid not in state["pending_entries"]:
        raise LineageViolation("Funded entry commit lacks its source plan")
    plan = state["pending_entries"][fid]
    if plan["source_context"] != _context(env):
        raise LineageViolation("Funded entry crossed its source month")
    if type(br_scaling_applied) is not bool or br_scaling_applied:
        raise LineageViolation("Funded entry cannot apply BR scaling")
    target = _int(final_target, "funded final target")
    _, targets, active, firms = _stocks(env)
    if (
        target != plan["target"]
        or targets[fid] != target
        or (not active[fid])
        or firms[fid].employees
    ):
        raise LineageViolation(
            "The committed funded entry differs from its source record or from the counted stocks"
        )
    record = state["slots"][fid]
    if plan["fresh_target_assignment"]:
        if (
            record["current_target"] != 0
            or record["capacity_lineage_id"] != plan["old_capacity_lineage_id"]
        ):
            raise LineageViolation(
                "A lineage with zero funded capacity changed before it was closed"
            )
        state["closed_lineages"].append(
            {
                "capacity_lineage_id": record["capacity_lineage_id"],
                "slot_id": fid,
                "closing_target": 0,
                "reason": "FUNDED_ZERO_ENTRY_EVENT",
                "replaced_by": plan["new_capacity_lineage_id"],
                "official_index_zero_evidence": deepcopy(plan["official_index_zero_evidence"]),
            }
        )
    if target == 0 and capacity_budget.request_weight(env, fid):
        origin = "BUDGET_PENDING_REQUEST"
    elif target > 0 and (plan["fresh_target_assignment"] or plan["original_request_reused"]):
        origin = "BUDGET_FUNDED_POSITIVE"
    else:
        origin = record["target_origin"]
    record.update(
        {
            "capacity_lineage_id": plan["new_capacity_lineage_id"],
            "capacity_generation": plan["new_capacity_generation"],
            "business_generation": plan["business_generation"],
            "current_target": target,
            "target_origin": origin,
            "official_index_zero_evidence": (
                None
                if plan["fresh_target_assignment"] or plan["original_request_reused"]
                else record["official_index_zero_evidence"]
            ),
            "last_entry_authority": plan["authority"],
        }
    )
    capacity_budget.commit_entry(env, fid, target)
    del state["pending_entries"][fid]
    _event(
        state,
        "ENTRY_COMMITTED",
        **deepcopy(plan),
        final_target=target,
        br_scaling_applied=False,
        active_capacity_inflow=target,
        global_budget_capacity_created=0,
    )
    _check_current(env)
    _publish(env, primary=bool(getattr(env, "_person_trace_active", False)))
    return deepcopy(record)


def _tag_index(env, before, active_before, original_receipt):
    state = _state(env)
    if state["pending_entries"]:
        raise LineageViolation("The regional capacity index was applied during an unfinished entry")
    _, after, active_after, _ = _stocks(env)
    if active_before != active_after:
        raise LineageViolation("Index unexpectedly changed firm activity")
    r = original_receipt
    if (
        not isinstance(r, dict)
        or r.get("mode") != "official_population_elasticity_capacity_v1"
        or r.get("official_elasticity") != OFFICIAL_RATIO
    ):
        raise LineageViolation("A zero capacity target must come from the regional capacity index")
    reference = (r["reference_capacity"], r["reference_population"], r["official_elasticity"])
    if state["index_reference"] is not None and state["index_reference"] != reference:
        raise LineageViolation("The reference capacity and population changed within one reset")
    state["index_reference"] = reference
    expected = round(
        reference[0] * (1 + OFFICIAL_RATIO * (r["population_after"] / reference[1] - 1))
    )
    if (
        sum((before[f] for f in before if active_before[f])) != r["active_capacity_before"]
        or sum((after[f] for f in after if active_after[f])) != r["active_capacity_after"]
        or r["active_capacity_after"] != expected
    ):
        raise LineageViolation(
            "The recorded regional capacity index differs from the counted targets"
        )
    state["index_calls"] += 1
    writes = []
    for fid in before:
        record = state["slots"][fid]
        if not active_before[fid] and before[fid] != after[fid]:
            raise LineageViolation("Index rewrote an inactive sizing memory")
        if before[fid] == after[fid]:
            continue
        item = {
            "slot_id": fid,
            "capacity_lineage_id": record["capacity_lineage_id"],
            "target_before": before[fid],
            "target_after": after[fid],
            "effective_year": int(env.current_year),
            "effective_month": int(env.current_month),
            "index_call": state["index_calls"],
            "reference_capacity": reference[0],
            "reference_population": reference[1],
            "population_after": r["population_after"],
        }
        if before[fid] > 0 and after[fid] == 0:
            record["target_origin"] = "OFFICIAL_INDEX_ZERO"
            record["official_index_zero_evidence"] = deepcopy(item)
        elif after[fid] > 0:
            record["target_origin"] = "OFFICIAL_INDEX_POSITIVE"
            record["official_index_zero_evidence"] = None
        record["current_target"] = after[fid]
        writes.append(item)
    _event(state, "OFFICIAL_INDEX_APPLIED", receipt=deepcopy(r), writes=writes)
    _check_current(env)
    _publish(env, primary=bool(getattr(env, "_person_trace_active", False)))


def journal(env):
    return deepcopy(_state(env))


def observe_primary_month(env, context):
    state, targets, active, _ = _check_current(env)
    if state["pending_entries"]:
        raise LineageViolation("Month ends with an uncommitted entry")
    if context["month_index"] != len(state["primary_month_contexts"]):
        raise LineageViolation("Primary month contexts are missing or repeated")
    context = deepcopy(context)
    context["journal_length"] = len(state["journal"])
    context["source_clock_at_finish"] = _context(env)
    context["active_capacity"] = sum((targets[fid] for fid in targets if active[fid]))
    context["pending_entry_count"] = 0
    state["primary_month_contexts"].append(context)
    _publish(env, primary=True)


def receipt():
    records = deepcopy(_captures)
    primary = [r for r in records if r is not None and r["primary_observed"]]
    untraced = [r for r in records if r is not None and (not r["primary_observed"])]
    result = {
        "schema": "capacity_lineage_receipt_v1",
        "enabled": True,
        "primary_records": primary,
        "auxiliary_reset_records": untraced,
        "auxiliary_record_scope": "Untraced or auxiliary resets; no claim every untraced reset was an auxiliary economic rollout.",
        "primary_months": sum((len(r["primary_month_contexts"]) for r in primary)),
        "primary_index_calls": sum((r["index_calls"] for r in primary)),
        "RNG_draws_by_helper": 0,
        "target_writes_by_helper": 0,
        "global_capture_is_observation_copy_only": True,
    }
    result["binding_capacity_budget_enabled"] = True
    result["target_writes_by_helper"] = sum(
        (
            len(event.get("writes", []))
            for record in records
            if record is not None
            for event in record["journal"]
            if event["kind"] == "BINDING_BUDGET_TARGETS_WRITTEN"
        )
    )
    result["target_write_authority"] = "SOURCE_JOURNALLED_BINDING_BUDGET"
    return result


def install(age_renewal_module, worker_writer_class=None):
    if not getattr(age_renewal_module, "_regional_capacity_installed", False):
        raise LineageViolation("Install the capacity lineage after the regional capacity index")
    if not getattr(age_renewal_module, "_capacity_lineage_installed", False):
        original = age_renewal_module.index_firm_capacity

        @wraps(original)
        def indexed(env, *args, **kwargs):
            import capacity_budget

            return capacity_budget.index(env, *args, **kwargs)

        age_renewal_module.index_firm_capacity = indexed
        age_renewal_module._capacity_lineage_installed = True
    if worker_writer_class is None:
        trace_module = sys.modules.get("person_month_trace")
        worker_writer_class = getattr(trace_module, "WorkerTraceWriter", None)
    if worker_writer_class is not None and (
        not getattr(worker_writer_class, "_capacity_lineage_finish_installed", False)
    ):
        original_finish = worker_writer_class.finish_month

        @wraps(original_finish)
        def finish(writer, env, *args, **kwargs):
            primary = bool(getattr(env, "_person_trace_active", False))
            current = getattr(writer, "current", None)
            if primary:
                if not isinstance(current, dict):
                    raise LineageViolation("Primary month hook lacks worker-trace context")
                context = {key: current[key] for key in ("month_index", "year", "calendar_month")}
            result = original_finish(writer, env, *args, **kwargs)
            if primary:
                observe_primary_month(env, context)
            return result

        worker_writer_class.finish_month = finish
        worker_writer_class._capacity_lineage_finish_installed = True
    return True
