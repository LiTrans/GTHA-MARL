def require(ok, message):
    if not ok:
        raise ValueError(message)


def release_inactive_attachments(env, source, events, period):
    regime = env._covid_regime
    if regime is None:
        return []
    manager = env.state_manager
    candidates = []
    for wid in sorted(manager.workers):
        ws = manager.workers[wid]
        if ws.covid_state != "temporary_layoff":
            continue
        employer = ws.covid_attached_employer_id
        if employer and env._covid_firm_is_active(employer):
            continue
        require(
            ws.worker.employment_status == "Unemployed" and ws.worker.current_employer_id is None,
            "Dead-employer temporary attachment has inconsistent employment",
        )
        candidates.append((wid, ws, employer))
    if not candidates:
        return []
    require(events == [], "Firm-exit releases must precede the pandemic module")
    receipts = []
    with source.observed_mutations(regime, manager, events):
        for wid, ws, employer in candidates:
            start = len(events)
            months = ws.covid_months_in_state
            previous_wage = ws.previous_wage
            previous_tenure = ws.previous_tenure_months
            regime._release_temp_to_search(ws)
            require(len(events) == start + 1, "One release must emit exactly one source event")
            event = events[-1]
            require(
                event["route"] == "covid_temporary_release_to_search"
                and event["worker_id"] == str(wid),
                "Wrong identity of a release event",
            )
            event["execution_phase"] = "after_firm_exits_before_firm_entries"
            event["former_attached_employer_id"] = None if employer is None else str(employer)
            require(
                ws.covid_state == "none"
                and ws.covid_attached_employer_id is None
                and (ws.worker.employment_status == "Unemployed")
                and (ws.worker.is_searching is True),
                "A release did not move the worker of a closed employer into searching unemployment",
            )
            require(
                ws.previous_wage == previous_wage and ws.previous_tenure_months == previous_tenure,
                "A release changed the preserved employment history",
            )
            receipts.append(
                {
                    "period": list(period),
                    "worker_id": str(wid),
                    "former_attached_employer_id": None if employer is None else str(employer),
                    "former_employer_active_before_release": False,
                    "native_method": "CovidTransitionRegime._release_temp_to_search",
                    "months_in_temporary_layoff_before_release": months,
                    "status_after": ws.worker.employment_status,
                    "is_searching_after": ws.worker.is_searching,
                    "covid_state_after": ws.covid_state,
                    "attachment_after": ws.covid_attached_employer_id,
                    "native_retention_draw_after": ws.lwf_retention_draw,
                    "source_event_sequence": event["sequence"],
                    "native_retention_uniform_draw_relocated": True,
                    "extra_recall_Bernoulli_draws": 0,
                    "new_hazard_or_distribution": False,
                }
            )
    return receipts
