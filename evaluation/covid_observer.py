from collections import Counter
from contextlib import contextmanager
from functools import wraps

METHODS = {
    "_enter_temp_layoff": ("covid_temporary_layoff_entry", "Employed", "Unemployed", "manager_id"),
    "_enter_marginal": ("covid_marginal_entry", "Employed", "NILF", "manager_id"),
    "_recall_same_employer": ("covid_same_employer_recall", "Unemployed", "Employed", "manager_id"),
    "_release_temp_to_search": (
        "covid_temporary_release_to_search",
        "Unemployed",
        "Unemployed",
        "state",
    ),
    "_release_marginal_to_search": (
        "covid_marginal_reentry_to_search",
        "NILF",
        "Unemployed",
        "manager_state",
    ),
}
LEDGER_ROUTES = {
    "temp_entries": "covid_temporary_layoff_entry",
    "marginal_entries": "covid_marginal_entry",
    "same_employer_recalls": "covid_same_employer_recall",
    "temp_releases_to_search": "covid_temporary_release_to_search",
    "marginal_reentries_to_search": "covid_marginal_reentry_to_search",
    "marginal_releases_to_nilf": "covid_marginal_release_to_nilf",
}
SOURCE_ENDPOINTS = {
    route: (origin, destination) for route, origin, destination, _ in METHODS.values()
}
SOURCE_ENDPOINTS["covid_marginal_release_to_nilf"] = ("NILF", "NILF")


def _snapshot(ws):
    return {
        "status": str(ws.worker.employment_status),
        "covid_state": str(ws.covid_state),
        "employer_id": (
            None if ws.worker.current_employer_id is None else str(ws.worker.current_employer_id)
        ),
    }


@contextmanager
def observed_mutations(regime, manager, events):
    cls = type(regime)
    originals = {}
    state_ids = {id(ws): str(wid) for wid, ws in manager.workers.items()}
    if len(state_ids) != len(manager.workers):
        raise ValueError("COVID observer state identities are not unique")

    def emit(route, wid, before, after):
        events.append(
            {
                "sequence": len(events),
                "route": route,
                "worker_id": str(wid),
                "before": before,
                "after": after,
            }
        )

    try:
        for name, (route, origin, destination, binding) in METHODS.items():
            descriptor = cls.__dict__[name]
            originals[name] = descriptor
            static = isinstance(descriptor, staticmethod)
            function = descriptor.__func__ if static else descriptor

            def make_wrapper(function, static, binding, route, origin, destination):

                @wraps(function)
                def wrapped(*args, **kwargs):
                    payload = args if static else args[1:]
                    if kwargs:
                        raise ValueError(
                            "The keywords of a pandemic transition call differ from the expected ones"
                        )
                    if binding == "manager_id":
                        source_manager, wid = payload[:2]
                        if source_manager is not manager:
                            raise ValueError("COVID source manager differs")
                        ws = manager.workers[wid]
                    elif binding == "state":
                        ws = payload[0]
                        wid = state_ids[id(ws)]
                    else:
                        source_manager, ws = payload[:2]
                        if source_manager is not manager:
                            raise ValueError("COVID source manager differs")
                        wid = state_ids[id(ws)]
                    before = _snapshot(ws)
                    if before["status"] != origin:
                        raise ValueError("COVID source has unexpected origin: " + route)
                    result = function(*args, **kwargs)
                    after = _snapshot(ws)
                    if after["status"] != destination:
                        raise ValueError("COVID source has unexpected destination: " + route)
                    emit(route, wid, before, after)
                    return result

                return staticmethod(wrapped) if static else wrapped

            setattr(cls, name, make_wrapper(function, static, binding, route, origin, destination))
        name = "_release_carryover"
        descriptor = cls.__dict__[name]
        originals[name] = descriptor

        @wraps(descriptor)
        def release(self, source_manager, ledger):
            if source_manager is not manager:
                raise ValueError("COVID release manager differs")
            marginal = {
                str(wid): (ws, _snapshot(ws))
                for wid, ws in manager.workers.items()
                if ws.covid_state == "marginally_attached"
            }
            result = descriptor(self, source_manager, ledger)
            for wid, (ws, before) in sorted(marginal.items()):
                after = _snapshot(ws)
                if (
                    before["status"] != "NILF"
                    or after["status"] != "NILF"
                    or after["covid_state"] != "none"
                ):
                    raise ValueError("COVID marginal carryover release differs")
                emit("covid_marginal_release_to_nilf", wid, before, after)
            return result

        setattr(cls, name, release)
        yield
    finally:
        for name, descriptor in originals.items():
            setattr(cls, name, descriptor)


def project_employment(opening, market, routes, covid_events):
    if set(opening) != set(market):
        raise ValueError("Worker IDs changed within the market step")
    predicted = {str(wid): row["status"] for wid, row in opening.items()}
    e_affected, movements, gross = (set(), [], Counter())

    def move(wid, origin, destination, route):
        wid = str(wid)
        if wid not in predicted:
            raise ValueError("Employment mechanism contains an unknown worker")
        before = predicted[wid]
        if (before == "Employed") != (origin == "Employed"):
            raise ValueError("Employment source order/origin conflict: " + route)
        if origin == "Employed" or destination == "Employed":
            e_affected.add(wid)
        if before == "Employed" and destination != "Employed":
            gross["employment_exits"] += 1
        elif before != "Employed" and destination == "Employed":
            gross["employment_entries"] += 1
        predicted[wid] = destination
        movements.append(
            {"worker_id": wid, "route": route, "origin": origin, "destination": destination}
        )

    for route, origin in (
        ("retirements_from_employment", "Employed"),
        ("retirements_from_unemployment", "Unemployed"),
    ):
        for wid in sorted(routes[route]):
            move(wid, origin, "NILF", route)
    for route, destination in (
        ("ordinary_job_loss_separations", "Unemployed"),
        ("firm_exit_displacements", "Unemployed"),
        ("overstaff_purge_displacements", "Unemployed"),
        ("ordinary_retirement_separations", "NILF"),
        ("ordinary_temporary_nilf_separations", "NILF"),
    ):
        for wid in sorted(routes.get(route, set())):
            move(wid, "Employed", destination, route)
    for index, event in enumerate(covid_events):
        if event.get("sequence") != index or event.get("route") not in set(LEDGER_ROUTES.values()):
            raise ValueError("COVID source event sequence/type differs")
        if (event["before"]["status"], event["after"]["status"]) != SOURCE_ENDPOINTS[
            event["route"]
        ]:
            raise ValueError("COVID route has incorrect source/destination statuses")
        move(
            event["worker_id"], event["before"]["status"], event["after"]["status"], event["route"]
        )
    for route in ("direct_nilf_to_employment", "matching_u2e_committed"):
        for wid in sorted(routes[route]):
            move(wid, "nonemployed", "Employed", route)
    for wid in predicted:
        if (predicted[wid] == "Employed") != (market[wid]["status"] == "Employed"):
            raise ValueError("Independent source employment endpoint differs for worker " + wid)
        if (opening[wid]["status"] == "Employed" or wid in e_affected) and predicted[wid] != market[
            wid
        ]["status"]:
            raise ValueError(
                "Independent source nonemployment destination differs for worker " + wid
            )
    endpoint = {
        "nonemployed_to_employed": sum(
            (
                opening[wid]["status"] != "Employed" and status == "Employed"
                for wid, status in predicted.items()
            )
        ),
        "employed_to_unemployed": sum(
            (
                opening[wid]["status"] == "Employed" and status == "Unemployed"
                for wid, status in predicted.items()
            )
        ),
        "employed_to_nilf": sum(
            (
                opening[wid]["status"] == "Employed" and status == "NILF"
                for wid, status in predicted.items()
            )
        ),
    }
    return {
        "schema": "covid_source_composed_employment_projection_v1",
        "enabled": True,
        "source_event_counts": dict(Counter((event["route"] for event in covid_events))),
        "gross_employment_entries": gross["employment_entries"],
        "gross_employment_exits": gross["employment_exits"],
        "predicted_endpoint_counts": endpoint,
        "worker_count": len(predicted),
        "employment_endpoints_independently_verified": True,
        "E_involved_full_destination_check_count": len(
            e_affected | {wid for wid in opening if opening[wid]["status"] == "Employed"}
        ),
        "all_nonemployment_participation_routes_claimed_complete": False,
        "mechanism_order": "learned_retirement,ordinary_sources,COVID_source_calls,direct_reentry,matching",
    }
