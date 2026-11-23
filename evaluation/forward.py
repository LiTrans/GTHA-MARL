from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path
import os
import gzip
import json
from . import covid_lifecycle, covid_transitions, covid_observer


def require(ok, message):
    if not ok:
        raise ValueError(message)


def snapshot(env):
    return {
        str(wid): {
            "status": str(ws.worker.employment_status),
            "covid_state": str(ws.covid_state),
            "age": int(ws.worker.age),
            "cma": int(ws.worker.cma),
            "searching": bool(ws.worker.is_searching),
            "employer_id": (
                None
                if ws.worker.current_employer_id is None
                else str(ws.worker.current_employer_id)
            ),
            "covid_attached_employer_id": (
                None
                if ws.covid_attached_employer_id is None
                else str(ws.covid_attached_employer_id)
            ),
        }
        for wid, ws in env.state_manager.workers.items()
    }


class ForwardObserver:

    def __init__(self, env, job):
        self.env, self.job = (env, job)
        self.months, self.current, self.closed = (0, None, False)
        self.lifecycle_releases = []
        self.lifecycle_months = []
        self.boundaries, self.outcomes, self.last_market = ([], [], None)
        self.boundary_installed = False
        directory = Path(job["output"])
        self.directory = directory
        self.path = directory / "FORWARD_SOURCE_MONTHS.jsonl.gz"
        with self.path.open("xb"):
            pass
        self.stream = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=1)
        self.sources = {
            "covid_lifecycle": covid_lifecycle,
            "covid_source": covid_observer,
            "covid_phase_bridge": covid_transitions,
        }

    def release_after_firm_exits(self):
        require(
            self.current is not None
            and (not self.current.get("firm_exit_release_phase_checked", False)),
            "Missing or repeated firm-exit release phase",
        )
        records = self.sources["covid_lifecycle"].release_inactive_attachments(
            self.env,
            self.sources["covid_source"],
            self.current["source_events"],
            self.current["period"],
        )
        self.current["firm_exit_release_phase_checked"] = True
        self.lifecycle_releases.extend(records)
        self.lifecycle_months.append(
            {
                "period": list(self.current["period"]),
                "native_releases": len(records),
                "worker_ids": [row["worker_id"] for row in records],
            }
        )
        return records

    def before(self):
        require(
            self.current is None and self.months < len(self.job["calendar"]),
            "Repeated/excess forward market phase",
        )
        if not self.boundary_installed:
            population = self.env._net_population
            require(population is not None, "Forward renewal controller is absent")
            original = population.finalize

            def finalize(env, obs):
                result = original(env, obs)
                require(
                    env is self.env
                    and self.last_market is not None
                    and (len(self.boundaries) + 1 == self.months),
                    "Repeated or unowned forward renewal phase",
                )
                period, prior = self.last_market
                closing = snapshot(env)
                retained = set(prior) & set(closing)
                require(
                    all(
                        (
                            (prior[w]["covid_state"], prior[w]["covid_attached_employer_id"])
                            == (closing[w]["covid_state"], closing[w]["covid_attached_employer_id"])
                            for w in retained
                        )
                    ),
                    "Continuing worker COVID tag/attachment changed during population renewal",
                )
                require(
                    all(
                        (
                            closing[w]["covid_state"] == "none"
                            and closing[w]["covid_attached_employer_id"] is None
                            for w in set(closing) - set(prior)
                        )
                    ),
                    "New renewal entrant has an unattributed COVID tag",
                )
                self.boundaries.append(
                    {
                        "period": period,
                        "continuing_workers_checked": len(retained),
                        "tagged_population_exits": [
                            w
                            for w in sorted(set(prior) - set(closing))
                            if prior[w]["covid_state"] != "none"
                        ],
                        "new_entrants_checked": len(set(closing) - set(prior)),
                        "continuing_tags_and_attachments_exact": True,
                        "unattributed_tag_changes": 0,
                    }
                )
                y, m, _ = period
                self.outcomes[-1]["after_renewal"] = env.get_covid_validation_snapshot(y, m)
                return result

            population.finalize = finalize
            self.boundary_installed = True
        year, month = self.job["calendar"][self.months]
        require(
            (int(self.env.state_manager.current_year), int(self.env.state_manager.current_month))
            == (year, month),
            "Forward actual market calendar differs",
        )
        self.current = {
            "period": [year, month, self.months],
            "opening": snapshot(self.env),
            "source_events": [],
        }
        env = self.env
        env._person_trace_separation_draw = None
        env._person_trace_direct_nilf_ids = set()
        env._person_trace_committed_u2e_ids = set()
        env._person_trace_committed_j2j_ids = set()
        env._person_trace_retirement_origins = {}

    @contextmanager
    def native_trace_scope(self):
        env = self.env
        prior_flag = os.environ.get("WORKER_LEVEL_TRACE")
        owners = (env, env.matching_mechanism)
        require(
            not any((getattr(owner, "_person_trace_active", False) for owner in owners)),
            "Forward source observer cannot compete with another trace writer",
        )
        old = [
            (
                getattr(owner, "_person_trace_active", False),
                getattr(owner, "_person_trace_owner_token", None),
            )
            for owner in owners
        ]
        os.environ["WORKER_LEVEL_TRACE"] = "1"
        for owner in owners:
            owner._person_trace_active, owner._person_trace_owner_token = (True, id(self))
        try:
            env.matching_mechanism.begin_person_month_trace()
            yield
        finally:
            for owner, (active, token) in zip(owners, old):
                owner._person_trace_active, owner._person_trace_owner_token = (active, token)
            if prior_flag is None:
                os.environ.pop("WORKER_LEVEL_TRACE", None)
            else:
                os.environ["WORKER_LEVEL_TRACE"] = prior_flag

    def covid_phase(self, native):
        env = self.env
        regime = env._covid_regime
        require(self.current is not None, "COVID phase has no owning market")
        y, m, _ = self.current["period"]
        self.current["before_covid_module"] = env.get_covid_validation_snapshot(y, m)
        if regime is None:
            result = native()
            self.current["after_covid_module"] = env.get_covid_validation_snapshot(y, m)
            return result
        cls, source = (type(regime), self.sources["covid_source"])
        original = cls.apply_month

        @wraps(original)
        def observed(actual, manager, year, month, active):
            require(
                actual is regime and manager is env.state_manager, "Foreign COVID source manager"
            )
            events = self.current["source_events"]
            require(
                all(
                    (
                        e.get("execution_phase") == "after_firm_exits_before_firm_entries"
                        for e in events
                    )
                ),
                "COVID module has prior non-lifecycle events or was applied twice",
            )
            offset = len(events)
            with source.observed_mutations(actual, manager, events):
                ledger = original(actual, manager, year, month, active)
            from collections import Counter

            counts = Counter((event["route"] for event in events[offset:]))
            require(
                all(
                    (
                        getattr(ledger, field) == counts[route]
                        for field, route in source.LEDGER_ROUTES.items()
                    )
                ),
                "The observed pandemic method calls differ from the source ledger",
            )
            return ledger

        cls.apply_month = observed
        try:
            result = native()
            self.current["after_covid_module"] = env.get_covid_validation_snapshot(y, m)
            return result
        finally:
            cls.apply_month = original

    def after(self):
        env, current = (self.env, self.current)
        require(current is not None, "Forward market close has no opening")
        market = snapshot(env)
        routes = {
            name: {str(w) for w in ids} for name, ids in env._employment_route_ids_this_step.items()
        }
        routes["direct_nilf_to_employment"] = {str(w) for w in env._person_trace_direct_nilf_ids}
        routes["matching_u2e_committed"] = {str(w) for w in env._person_trace_committed_u2e_ids}
        routes["matching_j2j_committed"] = {str(w) for w in env._person_trace_committed_j2j_ids}
        for origin, name in (
            ("Employed", "retirements_from_employment"),
            ("Unemployed", "retirements_from_unemployment"),
        ):
            routes[name] = {
                str(w)
                for w, status in env._person_trace_retirement_origins.items()
                if status == origin
            }
        source = self.sources["covid_source"]
        projection = source.project_employment(
            current["opening"], market, routes, current["source_events"]
        )
        projection["period"] = current["period"]
        trace = env.matching_mechanism.get_person_month_trace()
        candidate = {
            w for w, r in current["opening"].items() if r["covid_state"] == "temporary_layoff"
        }
        candidate.update(
            (
                e["worker_id"]
                for e in current["source_events"]
                if e["route"] == "covid_temporary_layoff_entry"
            )
        )
        workers = [
            {
                "worker_id": w,
                "routes": [name for name, ids in routes.items() if w in ids],
                "matching": trace["workers"].get(w) if w in candidate else None,
            }
            for w in sorted(current["opening"])
        ]
        y, m, index = current["period"]
        native_ledger = env._last_covid_ledger
        audit = {
            "schema": "covid_actual_method_source_audit_v1",
            "source_events": current["source_events"],
            "projection": projection,
            "source_ledger": native_ledger.to_dict() if native_ledger is not None else None,
        }
        worker = {
            "year": y,
            "calendar_month": m,
            "month_index": index,
            "workers": workers,
            "covid_source_audit": audit,
        }
        bridge = self.sources["covid_phase_bridge"].covid_market_stocks(
            worker, current["opening"], market
        )
        self.outcomes.append(
            {
                "period": list(current["period"]),
                "before_covid_module": current["before_covid_module"],
                "after_covid_module": current["after_covid_module"],
                "after_market": env.get_covid_validation_snapshot(y, m),
                "after_renewal": None,
            }
        )
        self.current = {
            "period": current["period"],
            "opening": current["opening"],
            "after_market": market,
            "routes": routes,
            "worker": worker,
            "projection": projection,
            "bridge": bridge,
        }

    def reconcile(self, original, flow, counters):
        current = self.current
        require(
            current is not None
            and [flow["year"], flow["calendar_month"], flow["month"]] == current["period"],
            "Forward source projection is missing or stale",
        )
        mapping = {
            "ordinary_job_loss_separations": "ordinary_separations_to_unemployment",
            "ordinary_retirement_separations": "ordinary_retirement_separations",
            "ordinary_temporary_nilf_separations": "ordinary_separations_to_temporary_nilf",
            "firm_exit_displacements": "firm_exit_displacements",
            "overstaff_purge_displacements": "overstaff_purge_displacements",
            "same_month_rehires": "same_month_rehires",
            "matching_u2e_committed": "matching_hires_nonemployed_to_employed",
            "direct_nilf_to_employment": "direct_nilf_to_employment",
            "retirements_from_employment": "retirements_from_employment",
            "retirements_from_unemployment": "retirements_from_unemployment",
        }
        require(
            all(
                (
                    key in counters and len(current["routes"][route]) == counters[key]
                    for route, key in mapping.items()
                )
            ),
            "The observed sets of source IDs differ from the mechanism counters",
        )
        result = original(flow, counters)
        if self.job["scenario"].endswith("_ON"):
            projection = current["projection"]
            counters["covid_source_projection"] = deepcopy(projection)
            for residual, endpoint in (
                ("matching_hires_residual", "nonemployed_to_employed"),
                ("separation_residual", "employed_to_unemployed"),
                ("nilf_entry_residual", "employed_to_nilf"),
            ):
                result[residual] = (
                    flow[endpoint] - projection["predicted_endpoint_counts"][endpoint]
                )
            result["covid_composed_gross_employment_residual"] = (
                flow["closing_employment"]
                - flow["opening_employment"]
                - projection["gross_employment_entries"]
                + projection["gross_employment_exits"]
            )
            result["all_mechanisms_reconciled"] = all(
                (v == 0 for k, v in result.items() if k != "all_mechanisms_reconciled")
            )
        require(
            result["all_mechanisms_reconciled"] is True,
            "COVID-composed source employment reconciliation failed",
        )
        record = {
            "schema": "current_native_forward_source_market_month_v1",
            "period": current["period"],
            "opening": current["opening"],
            "after_market": current["after_market"],
            "routes": {k: sorted(v) for k, v in current["routes"].items()},
            "worker_evidence": current["worker"],
            "phase_bridge": current["bridge"],
            "native_mechanism_counters": deepcopy(counters),
            "mechanism_reconciliation": deepcopy(result),
        }
        self.stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.last_market = (list(current["period"]), current["after_market"])
        self.current = None
        self.months += 1
        return result

    def finish(self):
        require(
            self.months == len(self.boundaries) == len(self.outcomes) == len(self.job["calendar"])
            and all((row["after_renewal"] is not None for row in self.outcomes))
            and (self.current is None)
            and (not self.closed),
            "The source record of market and renewal events is incomplete",
        )
        self.stream.close()
        self.closed = True


def wrap_environment(base, job):

    class ObservedForwardEnvironment(base):

        def _step(self, action):
            audit = self._forward_source_observer
            audit.before()
            with audit.native_trace_scope():
                result = super()._step(action)
                audit.after()
            return result

        def process_firm_exits(self):
            result = super().process_firm_exits()
            self._forward_source_observer.release_after_firm_exits()
            return result

        def _apply_covid_month(self):
            return self._forward_source_observer.covid_phase(super()._apply_covid_month)

    return ObservedForwardEnvironment
