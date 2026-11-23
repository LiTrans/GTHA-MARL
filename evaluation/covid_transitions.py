from collections import Counter
import math

NONE = "none"
TEMP = "temporary_layoff"
MARGINAL = "marginally_attached"
TAGS = (NONE, TEMP, MARGINAL)
ROUTES = {
    "covid_temporary_layoff_entry": ("Employed", NONE, "Unemployed", TEMP),
    "covid_marginal_entry": ("Employed", NONE, "NILF", MARGINAL),
    "covid_same_employer_recall": ("Unemployed", TEMP, "Employed", NONE),
    "covid_temporary_release_to_search": ("Unemployed", TEMP, "Unemployed", NONE),
    "covid_marginal_reentry_to_search": ("NILF", MARGINAL, "Unemployed", NONE),
    "covid_marginal_release_to_nilf": ("NILF", MARGINAL, "NILF", NONE),
}
LEDGER_ROUTES = {
    "temp_entries": "covid_temporary_layoff_entry",
    "marginal_entries": "covid_marginal_entry",
    "same_employer_recalls": "covid_same_employer_recall",
    "temp_releases_to_search": "covid_temporary_release_to_search",
    "marginal_reentries_to_search": "covid_marginal_reentry_to_search",
    "marginal_releases_to_nilf": "covid_marginal_release_to_nilf",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def state(row):
    for key in ("status", "covid_state", "covid_attached_employer_id", "employer_id"):
        require(key in row, "The pandemic stock reconciliation lacks the worker field:" + key)
    tag = row["covid_state"]
    require(tag in TAGS, "Unknown COVID tag")
    attached = row["covid_attached_employer_id"]
    if tag == TEMP:
        require(
            row["status"] == "Unemployed"
            and row["employer_id"] is None
            and isinstance(attached, str)
            and attached,
            "Temporary-layoff tag/attachment/status inconsistent",
        )
    elif tag == MARGINAL:
        require(
            row["status"] == "NILF" and row["employer_id"] is None and (attached is None),
            "Marginal tag/attachment/status inconsistent",
        )
    else:
        require(attached is None, "COVID_NONE has a residual attachment")
    return {"covid_state": tag, "covid_attached_employer_id": attached}


def stocks(population):
    counts = Counter((row["covid_state"] for row in population.values()))
    return {TEMP: counts[TEMP], MARGINAL: counts[MARGINAL]}


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0)


def committed_transaction(worker, market):
    routes = worker.get("routes")
    require(
        isinstance(routes, list)
        and "matching_u2e_committed" in routes
        and ("matching_j2j_committed" not in routes),
        "COVID clear lacks an ordinary committedU2E source route",
    )
    require(
        market["status"] == "Employed"
        and market["covid_state"] == NONE
        and (market["covid_attached_employer_id"] is None)
        and isinstance(market["employer_id"], str)
        and market["employer_id"],
        "COVID clear does not end in real employment without attachment",
    )
    matching = worker.get("matching")
    require(isinstance(matching, dict), "COVID clear lacks matching trace")
    pre = matching.get("pre_match", {})
    events = matching.get("events", {})
    require(
        pre.get("status") == "Unemployed"
        and pre.get("in_matching_pool") is True
        and (pre.get("is_matching_seeker") is True),
        "COVID clear was not a raw-U matching participant",
    )
    duration = pre.get("unemployment_months")
    require(type(duration) is int and duration >= 0, "COVID clear lacks valid pre-match duration")
    require(
        events.get("matches") == 1
        and type(events.get("bargaining_successes")) is int
        and (events["bargaining_successes"] >= 1),
        "COVID clear lacks one match/successful bargaining event",
    )
    firm = market["employer_id"]
    offers = matching.get("offer_events")
    bargains = matching.get("bargaining_attempt_details")
    screening = matching.get("screening_events")
    require(
        isinstance(offers, list) and isinstance(bargains, list) and isinstance(screening, list),
        "COVID clear lacks transaction-level screening/offer/bargaining evidence",
    )
    success = []
    for bargain in bargains:
        if bargain.get("firm_id") != firm or bargain.get("failed") is not False:
            continue
        vacancy = bargain.get("vacancy_id")
        round_index = bargain.get("round_index")
        if (
            not isinstance(vacancy, str)
            or not vacancy
            or (not positive(bargain.get("negotiated_wage")))
        ):
            continue
        if not any(
            (
                offer.get("firm_id") == firm
                and offer.get("vacancy_id") == vacancy
                and (offer.get("round_index") == round_index)
                and (offer.get("operational_accepted") is True)
                for offer in offers
            )
        ):
            continue
        if not any(
            (
                row.get("firm_id") == firm
                and row.get("vacancy_id") == vacancy
                and (row.get("passed") is True)
                for row in screening
            )
        ):
            continue
        success.append((firm, vacancy, round_index))
    require(
        bool(success),
        "COVID clear employer lacks a corresponding accepted offer and successful transaction",
    )
    return {"unemployment_months": duration, "transaction_evidence_count": len(success)}


def covid_market_stocks(worker, opening, market):
    require(set(opening) == set(market), "Within-market worker universe changed")
    population = {wid: state(row) for wid, row in opening.items()}
    market_tags = {wid: state(row) for wid, row in market.items()}
    records = {row["worker_id"]: row for row in worker["workers"]}
    require(
        len(records) == len(worker["workers"]) and set(opening) <= set(records),
        "Worker reconciliation records are missing or repeated",
    )
    audit = worker.get("covid_source_audit")
    require(
        isinstance(audit, dict) and audit.get("schema") == "covid_actual_method_source_audit_v1",
        "The pandemic source record is absent or unknown",
    )
    period = [worker["year"], worker["calendar_month"], worker["month_index"]]
    require(
        audit.get("projection", {}).get("period") == period,
        "The pandemic source record is out of date",
    )
    ledger = audit.get("source_ledger")
    events = audit.get("source_events")
    require(isinstance(events, list), "COVID source event list missing")
    opening_counts = stocks(population)
    counts = Counter()
    entered_temp = set()
    early = []
    for event in events:
        require(
            event.get("execution_phase") in (None, "after_firm_exits_before_firm_entries"),
            "Unknown COVID source execution phase",
        )
        if event.get("execution_phase") == "after_firm_exits_before_firm_entries":
            require(
                event.get("route") == "covid_temporary_release_to_search",
                "Nonrelease source in employer-death phase",
            )
            require(
                len(early) == event.get("sequence"),
                "Early employer-death releases must be a contiguous event prefix",
            )
            wid = event.get("worker_id")
            require(
                wid in opening
                and event.get("former_attached_employer_id")
                == opening[wid]["covid_attached_employer_id"],
                "Early release former employer differs from opening attachment",
            )
            early.append(event)
    pre_module_counts = dict(opening_counts)
    pre_module_counts[TEMP] -= len(early)
    require(pre_module_counts[TEMP] >= 0, "More early releases than opening temporary claims")
    module_events = events[len(early) :]
    early_counts = Counter((event["route"] for event in early))
    if ledger is None:
        require(
            not module_events and pre_module_counts == {TEMP: 0, MARGINAL: 0},
            "Pandemic stocks or events without a module ledger",
        )
    else:
        for prefix, tag in (("temp", TEMP), ("marginal", MARGINAL)):
            require(
                type(ledger.get(prefix + "_stock_start")) is int
                and ledger[prefix + "_stock_start"] == pre_module_counts[tag],
                "The opening of the module ledger differs after the observed releases from closed employers",
            )
    for sequence, event in enumerate(events):
        require(
            event.get("sequence") == sequence and event.get("route") in ROUTES,
            "Invalid COVID source event order/route",
        )
        wid = event.get("worker_id")
        require(wid in population, "COVID source event worker absent")
        before, after = (event.get("before"), event.get("after"))
        require(
            isinstance(before, dict) and isinstance(after, dict), "COVID event snapshots missing"
        )
        source_status, source_tag, dest_status, dest_tag = ROUTES[event["route"]]
        require(
            (
                before.get("status"),
                before.get("covid_state"),
                after.get("status"),
                after.get("covid_state"),
            )
            == (source_status, source_tag, dest_status, dest_tag),
            "A pandemic event has the wrong origin or destination",
        )
        current = population[wid]
        require(
            current["covid_state"] == source_tag,
            "COVID event tag chain differs from prior source state",
        )
        if source_status == "Employed":
            require(
                isinstance(before.get("employer_id"), str) and before["employer_id"],
                "COVID employed entry has no source employer",
            )
        else:
            require(
                before.get("employer_id") is None, "Detached COVID source has a current employer"
            )
        if event["route"] == "covid_same_employer_recall":
            require(
                after.get("employer_id") == current["covid_attached_employer_id"],
                "A recall went to a different employer",
            )
        else:
            require(
                after.get("employer_id") is None,
                "A detached or released worker still has an employer",
            )
        attached = before["employer_id"] if dest_tag == TEMP else None
        population[wid] = {"covid_state": dest_tag, "covid_attached_employer_id": attached}
        counts[event["route"]] += 1
        if dest_tag == TEMP:
            entered_temp.add(wid)
    module_close = stocks(population)
    if ledger is not None:
        for prefix, tag in (("temp", TEMP), ("marginal", MARGINAL)):
            require(
                type(ledger.get(prefix + "_stock_end")) is int
                and ledger[prefix + "_stock_end"] == module_close[tag],
                "The close of the module ledger differs from the replay of source events",
            )
        for field, route in LEDGER_ROUTES.items():
            require(
                type(ledger.get(field)) is int
                and ledger[field] == counts[route] - early_counts[route],
                "The pandemic source count differs:" + field,
            )
        require(
            ledger.get("temp_identity_residual") == ledger.get("marginal_identity_residual") == 0,
            "The pandemic module ledger does not balance",
        )
        require(
            pre_module_counts[TEMP]
            + ledger["temp_entries"]
            - ledger["same_employer_recalls"]
            - ledger["temp_releases_to_search"]
            == module_close[TEMP],
            "The temporary-layoff stock does not balance",
        )
        require(
            pre_module_counts[MARGINAL]
            + ledger["marginal_entries"]
            - ledger["marginal_reentries_to_search"]
            - ledger["marginal_releases_to_nilf"]
            == module_close[MARGINAL],
            "The marginal-attachment stock does not balance",
        )
    clears = []
    for wid, source in population.items():
        closing = market_tags[wid]
        if source == closing:
            continue
        require(
            source["covid_state"] == TEMP and closing["covid_state"] == NONE,
            "Unattributed post-COVID tag/attachment change; marginal changes/new tags are forbidden",
        )
        transaction = committed_transaction(records[wid], market[wid])
        clears.append(
            {
                "worker_id": wid,
                "prior_covid_state": TEMP,
                "new_temp_entry_this_month": wid in entered_temp,
                "pre_match_unemployment_months": transaction["unemployment_months"],
            }
        )
        population[wid] = {"covid_state": NONE, "covid_attached_employer_id": None}
    require(
        population == market_tags,
        "Source module and actual market commitments do not reproduce closing flags",
    )
    closing_counts = stocks(population)
    require(
        closing_counts[TEMP] == module_close[TEMP] - len(clears)
        and closing_counts[MARGINAL] == module_close[MARGINAL],
        "Pandemic-module stocks do not reconcile with the market stocks",
    )
    return {
        "schema": "native_covid_module_to_market_stock_bridge_v1",
        "month_index": worker["month_index"],
        "year": worker["year"],
        "month": worker["calendar_month"],
        "opening_stocks": opening_counts,
        "pre_native_module_stocks": pre_module_counts,
        "post_firm_exit_native_releases": early,
        "post_firm_exit_release_count": len(early),
        "post_covid_module_stocks": module_close,
        "market_close_stocks": closing_counts,
        "ordinary_hire_clears": clears,
        "ordinary_hire_clear_count": len(clears),
        "ordinary_hire_clears_of_new_temp_entries": sum(
            (item["new_temp_entry_this_month"] for item in clears)
        ),
        "ordinary_hire_clears_at_duration_zero": sum(
            (item["pre_match_unemployment_months"] == 0 for item in clears)
        ),
        "marginal_post_module_clear_count": 0,
        "temporary_stock_bridge_residual": 0,
        "marginal_stock_bridge_residual": 0,
        "retained_tags_and_attachments_exact": True,
        "all_post_module_changes_attributed": True,
        "source_ledger_rewritten": False,
        "native_runtime_files_changed": False,
        "evaluation_lifecycle_order_changed": bool(early),
        "new_hazard_or_distribution": False,
        "scientific_target_changed": False,
    }
