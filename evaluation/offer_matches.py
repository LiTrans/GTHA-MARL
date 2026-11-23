from contextlib import contextmanager
import os
from .acceptance import require


@contextmanager
def trace_scope(env):
    owners = (env, env.matching_mechanism)
    require(
        not any((getattr(owner, "_person_trace_active", False) for owner in owners)),
        "Competing person-trace owner",
    )
    old = [
        (
            getattr(owner, "_person_trace_active", False),
            getattr(owner, "_person_trace_owner_token", None),
        )
        for owner in owners
    ]
    flag = os.environ.get("WORKER_LEVEL_TRACE")
    os.environ["WORKER_LEVEL_TRACE"] = "1"
    for owner in owners:
        owner._person_trace_active, owner._person_trace_owner_token = (True, id(env))
    for name in (
        "_person_trace_direct_nilf_ids",
        "_person_trace_committed_u2e_ids",
        "_person_trace_committed_j2j_ids",
    ):
        setattr(env, name, set())
    env._person_trace_retirement_origins = {}
    env._person_trace_separation_draw = None
    env.matching_mechanism.begin_person_month_trace()
    try:
        yield
    finally:
        for owner, (active, token) in zip(owners, old):
            owner._person_trace_active, owner._person_trace_owner_token = (active, token)
        if flag is None:
            os.environ.pop("WORKER_LEVEL_TRACE", None)
        else:
            os.environ["WORKER_LEVEL_TRACE"] = flag


def finalized_matches(env, opening, month_index, job):
    u, j = (set(env._person_trace_committed_u2e_ids), set(env._person_trace_committed_j2j_ids))
    require(not u & j, "One worker has simultaneousU2E/J2J commitment")
    trace = env.matching_mechanism.get_person_month_trace()["workers"]
    result = []
    for worker_id in sorted(u | j):
        ws = env.state_manager.workers[worker_id]
        actual = trace[str(worker_id)]
        firm = str(ws.worker.current_employer_id)
        require(
            ws.worker.employment_status == "Employed" and worker_id in opening,
            "Commit does not end in resident employment",
        )
        candidates = []
        for bargain in actual["bargaining_attempt_details"]:
            if bargain["firm_id"] != firm or bargain["failed"] is not False:
                continue
            offers = [
                row
                for row in actual["offer_events"]
                if row["firm_id"] == firm
                and row["vacancy_id"] == bargain["vacancy_id"]
                and (row["round_index"] == bargain["round_index"])
                and (row["operational_accepted"] is True)
            ]
            if len(offers) == 1:
                candidates.append((bargain, offers[0]))
        require(
            len(candidates) == 1,
            "Committed worker lacks exactly one successful accepted source transaction",
        )
        bargain, offer = candidates[0]
        require(
            offer["commute_units"] == "minutes" and bargain["commute_units"] == "minutes",
            "Actual finalized commute is not minutes",
        )
        result.append(
            {
                "schema": "actual_committed_acceptance_match_v1",
                "policy_seed": job["policy_seed"],
                "evaluation_seed": job["evaluation_seed"],
                "family": job["family"],
                "month_index": month_index,
                "year": 2006 + month_index // 12,
                "calendar_month": month_index % 12 + 1,
                "worker_id": str(worker_id),
                "cma": opening[worker_id]["cma"],
                "origin_status": opening[worker_id]["status"],
                "source_route": (
                    "matching_u2e_committed" if worker_id in u else "matching_j2j_committed"
                ),
                "firm_id": firm,
                "vacancy_id": offer["vacancy_id"],
                "round_index": offer["round_index"],
                "offered_wage": offer["posted_wage"],
                "match_quality": offer["match_quality"],
                "negotiated_wage": bargain["negotiated_wage"],
                "commute_minutes": offer["commute_value"],
                "operational_accepted": True,
                "successful_bargaining": True,
                "commit_source_ID_observed": True,
            }
        )
    return result
