from __future__ import annotations
import gzip
import hashlib
import json
import math
from pathlib import Path
from functools import wraps


def _abort_on_trace_error(method):

    @wraps(method)
    def checked(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.abort()
            raise

    return checked


def _number(value):
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite worker trace value")
    return result


def committed_j2j_ids(committed_records, committed_count):
    ids = [row["worker_id"] for row in committed_records]
    if len(set(ids)) != len(ids) or len(ids) != committed_count:
        raise ValueError("Trace committed J2J records differ from transaction count")
    return set(ids)


def snapshot_workers(env):
    result = {}
    boundary_jobs = getattr(getattr(env, "boundary", None), "jobs", {})
    for wid, ws in env.state_manager.workers.items():
        worker = ws.worker
        key = str(wid)
        if key in result:
            raise ValueError("Worker IDs collide after string conversion")
        status = str(worker.employment_status)
        if status not in ("Employed", "Unemployed", "NILF"):
            status = str(worker.employment_status)
        job = boundary_jobs.get(wid)
        employer = getattr(worker, "current_employer_id", None)
        result[key] = {
            "age": int(worker.age),
            "status": status,
            "is_searching": bool(worker.is_searching),
            "unemployment_months": int(worker.unemployment_months),
            "separation_reason": getattr(ws, "separation_reason", None),
            "industry": int(worker.industry),
            "skill_level": _number(getattr(worker, "skill_level", None)),
            "previous_wage": _number(getattr(ws, "previous_wage", None)),
            "current_wage": _number(getattr(worker, "current_wage", None)),
            "employer_id": None if employer is None else str(employer),
            "nilf_subtype": (
                None if getattr(ws, "nilf_subtype", None) is None else int(ws.nilf_subtype)
            ),
            "resident_job_kind": (
                job["kind"]
                if job
                else "local_payroll" if status == "Employed" and employer is not None else None
            ),
        }
        if hasattr(ws, "temporary_nilf_entry_period"):
            period = ws.temporary_nilf_entry_period
            result[key]["temporary_nilf_entry_period"] = None if period is None else list(period)
    return result


def capture_separation_draw(env, employed_mask, effective_rates, layoff_shares):
    mask = employed_mask.detach().cpu().tolist()
    rates = effective_rates.detach().cpu().tolist()
    shares = layoff_shares.detach().cpu().tolist()
    ids = list(env.worker_ids)
    if not len(mask) == len(rates) == len(shares) == len(ids):
        raise ValueError("Separation-draw worker alignment differs")
    states = snapshot_workers(env)
    rows = {}
    for wid, employed, rate, share in zip(ids, mask, rates, shares):
        key = str(wid)
        if key not in states:
            raise ValueError("Separation draw worker missing from live population")
        rows[key] = {
            **states[key],
            "raw_employed_draw_risk": bool(employed),
            "effective_separation_probability": _number(rate),
            "layoff_share": _number(share),
        }
        if bool(employed) != (states[key]["status"] == "Employed"):
            raise ValueError("Separation tensor and worker status differ")
    env._person_trace_separation_draw = {
        "period": [int(env.current_year), int(env.current_month), int(env.current_step)],
        "workers": rows,
    }


def capture_absolute_random_draws(env, separation_draws, cause_draws, inputs):
    observed = env._person_trace_separation_draw
    metadata = env._absolute_separation_this_step
    period = [int(env.current_year), int(env.current_month), int(env.current_step)]
    if observed is None or observed.get("period") != period or metadata.get("period") != period:
        raise ValueError("Absolute separation observation has a missing/stale source phase")
    arrays = {
        key: value.detach().cpu().tolist()
        for key, value in {
            "base_separation_probability": inputs["base"],
            "probability_to_unemployment": inputs["to_u"],
            "probability_to_temporary_nilf": inputs["to_n"],
            "actual_total": inputs["total"],
            "absolute_calibration_in_scope": inputs["scope"],
            "separation_uniform_draw": separation_draws,
            "cause_uniform_draw": cause_draws,
        }.items()
    }
    ids = list(env.worker_ids)
    if any((len(values) != len(ids) for values in arrays.values())):
        raise ValueError("Absolute separation uniform/rate worker alignment differs")
    for index, wid in enumerate(ids):
        row = observed["workers"][str(wid)]
        if arrays["actual_total"][index] != row["effective_separation_probability"]:
            raise ValueError("Observed absolute total is not the probability used by the draw")
        for key, values in arrays.items():
            if key == "actual_total":
                continue
            value = values[index]
            if key == "absolute_calibration_in_scope":
                if type(value) is not bool or value != (25 <= row["age"] < 55):
                    raise ValueError("Absolute age scope differs between tensors and source worker")
                row[key] = value
            else:
                number = _number(value)
                if not 0 <= number <= 1:
                    raise ValueError("Invalid probability or uniform draw")
                if key.endswith("uniform_draw") and number == 1:
                    raise ValueError("A uniform draw must be strictly below one")
                row[key] = number


class WorkerTraceWriter:

    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, "xt", encoding="utf-8", compresslevel=1)
        self.months = 0
        self.worker_month_rows = 0
        self.current = None
        self._env = None
        self._closed = False

    def _bind_environment(self, env):
        if self._closed:
            raise ValueError("A closed trace writer cannot be reused")
        if self._env is not None and self._env is not env:
            raise ValueError("A trace writer cannot observe multiple environments")
        matcher = env.matching_mechanism
        token = id(self)
        for owner in (env, matcher):
            if (
                getattr(owner, "_person_trace_active", False)
                and getattr(owner, "_person_trace_owner_token", None) != token
            ):
                raise ValueError("Environment already has a different trace writer")
        self._env = env
        for owner in (env, matcher):
            owner._person_trace_owner_token = token
            owner._person_trace_active = True

    def _release_environment(self):
        if self._env is None:
            return
        for owner in (self._env, self._env.matching_mechanism):
            if getattr(owner, "_person_trace_owner_token", None) == id(self):
                owner._person_trace_active = False
                owner._person_trace_owner_token = None
        self._env = None

    def abort(self):
        self._release_environment()
        self.stream.close()
        self._closed = True

    @_abort_on_trace_error
    def step(self, env, action_td):
        if self._env is not env or not getattr(env, "_person_trace_active", False):
            raise ValueError("Trace step requires its owning environment")
        return env.step(action_td)

    @_abort_on_trace_error
    def begin_month(self, env, month_index, year, calendar_month, worker_action):
        if self.current is not None or month_index != self.months:
            raise ValueError("Trace month continuity differs")
        flat = worker_action[..., 0] if worker_action.ndim > 1 else worker_action
        actions = flat.detach().cpu().tolist()
        if len(actions) != len(env.worker_ids):
            raise ValueError("Trace actor-action worker alignment differs")
        opening = snapshot_workers(env)
        self._bind_environment(env)
        self.current = {
            "schema": "person_month_trace_v1",
            "month_index": int(month_index),
            "year": int(year),
            "calendar_month": int(calendar_month),
            "opening": opening,
            "retirement_actions": {
                str(wid) for wid, action in zip(env.worker_ids, actions) if int(action) == 40
            },
        }
        env.matching_mechanism.begin_person_month_trace()
        env._person_trace_separation_draw = None
        env._person_trace_direct_nilf_ids = set()
        env._person_trace_committed_u2e_ids = set()
        env._person_trace_committed_j2j_ids = set()
        env._person_trace_retirement_origins = {}

    @_abort_on_trace_error
    def after_market(self, env):
        if self.current is None:
            raise ValueError("Trace market close without month opening")
        self.current["after_market"] = snapshot_workers(env)
        routes = {
            name: {str(wid) for wid in ids}
            for name, ids in env._employment_route_ids_this_step.items()
        }
        routes["direct_nilf_to_employment"] = {
            str(wid) for wid in env._person_trace_direct_nilf_ids
        }
        routes["matching_u2e_committed"] = {str(wid) for wid in env._person_trace_committed_u2e_ids}
        routes["matching_j2j_committed"] = {str(wid) for wid in env._person_trace_committed_j2j_ids}
        for status, name in (
            ("Employed", "retirements_from_employment"),
            ("Unemployed", "retirements_from_unemployment"),
        ):
            routes[name] = {
                str(wid)
                for wid, origin in env._person_trace_retirement_origins.items()
                if origin == status
            }
        self.current["routes"] = routes
        self.current["matching"] = env.matching_mechanism.get_person_month_trace()
        draw = env._person_trace_separation_draw
        if draw is None:
            raise ValueError("Ordinary separation draw was not observed")
        expected_period = [self.current["year"], self.current["calendar_month"], self.months]
        if draw["period"] != expected_period:
            raise ValueError("Separation draw period differs from market month")
        regular_risk = {
            wid for wid, row in draw["workers"].items() if row["raw_employed_draw_risk"]
        }
        target_risk = set()
        target_selected = set()
        for wid in regular_risk:
            if wid not in draw["workers"] or not draw["workers"][wid]["raw_employed_draw_risk"]:
                raise ValueError("A local risk is absent from the drawn risks")
        for wid, row in draw["workers"].items():
            row["local_resident_ordinary_risk"] = wid in regular_risk
            row["target_reduction_risk"] = wid in target_risk
        self.current["separation_phase"] = {
            "raw_draw": draw,
            "regular_local_risk_ids": sorted(regular_risk),
            "regular_local_events": None,
            "excluded_boundary_draws": None,
            "target_resident_risk_ids": sorted(target_risk),
            "target_selected_resident_ids": sorted(target_selected),
            "report_period": expected_period,
        }
        if getattr(env, "_absolute_separation_this_step", None) is not None:
            absolute = env._absolute_separation_this_step
            if absolute.get("enabled") is not True or absolute.get("period") != expected_period:
                raise ValueError("Absolute separation report is missing or stale")
            self.current["separation_phase"]["absolute_calibration"] = absolute
        destination_split = getattr(env, "_destination_split_this_step", None)
        if destination_split is not None:
            if (
                destination_split.get("enabled") is not True
                or [
                    destination_split.get("year"),
                    destination_split.get("calendar_month"),
                    destination_split.get("step"),
                ]
                != expected_period
            ):
                raise ValueError("Destination split report is missing or stale")
            self.current["separation_phase"]["destination_split"] = destination_split

    @_abort_on_trace_error
    def finish_month(self, env, employment_flow):
        current = self.current
        if current is None or "after_market" not in current:
            raise ValueError("Trace finish without market close")
        post = snapshot_workers(env)
        opening, market = (current["opening"], current["after_market"])
        routes = current["routes"]
        counters = employment_flow["mechanism_counters"]
        expected = {
            "ordinary_job_loss_separations": "ordinary_separations_to_unemployment",
            "ordinary_retirement_separations": "ordinary_retirement_separations",
            "firm_exit_displacements": "firm_exit_displacements",
            "overstaff_purge_displacements": "overstaff_purge_displacements",
            "same_month_rehires": "same_month_rehires",
            "matching_u2e_committed": "matching_hires_nonemployed_to_employed",
            "direct_nilf_to_employment": "direct_nilf_to_employment",
            "retirements_from_employment": "retirements_from_employment",
            "retirements_from_unemployment": "retirements_from_unemployment",
        }
        if "ordinary_temporary_nilf_separations" in routes:
            expected["ordinary_temporary_nilf_separations"] = (
                "ordinary_separations_to_temporary_nilf"
            )
            temporary = routes["ordinary_temporary_nilf_separations"]
            if temporary & set().union(
                *(
                    routes[name]
                    for name in (
                        "ordinary_job_loss_separations",
                        "ordinary_retirement_separations",
                        "firm_exit_displacements",
                        "overstaff_purge_displacements",
                    )
                )
            ):
                raise ValueError("Temporary NILF source overlaps another separation route")
            for wid in temporary:
                if (
                    wid not in opening
                    or opening[wid]["status"] != "Employed"
                    or wid not in market
                    or (market[wid]["status"] != "NILF")
                    or (market[wid]["nilf_subtype"] != 0)
                    or (not 25 <= opening[wid]["age"] < 55)
                ):
                    raise ValueError("Temporary NILF route has invalid age or endpoint")
            if temporary & (
                routes["direct_nilf_to_employment"]
                | routes["matching_u2e_committed"]
                | routes["matching_j2j_committed"]
            ):
                raise ValueError(
                    "New temporary NILF worker returned to employment in the same month"
                )
        for route, counter in expected.items():
            if len(routes[route]) != counters[counter]:
                raise ValueError(f"Worker trace route count differs: {route}")
        matcher = current["matching"]
        all_ids = set(opening) | set(market) | set(post)
        for name, ids in routes.items():
            if not ids <= all_ids:
                raise ValueError(f"Trace source route contains unknown IDs: {name}")
        loss_sources = [
            routes[name]
            for name in (
                "ordinary_job_loss_separations",
                "firm_exit_displacements",
                "overstaff_purge_displacements",
            )
        ]
        if any((loss_sources[i] & loss_sources[j] for i in range(3) for j in range(i + 1, 3))):
            raise ValueError("Job-loss source ID sets overlap")
        losses = set().union(*loss_sources)
        if any((wid not in opening or opening[wid]["status"] != "Employed" for wid in losses)):
            raise ValueError("Job-loss source ID was not employed at month opening")
        if routes["same_month_rehires"] != losses & routes["matching_u2e_committed"]:
            raise ValueError("Same-month rehire IDs differ from loss/commit intersection")
        matching_ids = set(matcher["workers"])
        if not matching_ids <= all_ids:
            raise ValueError("Matcher trace contains unknown worker IDs")
        events = {}
        for worker in matcher["workers"].values():
            for key, count in worker["events"].items():
                events[key] = events.get(key, 0) + count
        stats = env.last_matching_stats
        for event, counter in (
            ("applications", "applications"),
            ("offers", "offers"),
            ("matches", "matches"),
            ("screen_fail", "applicants_screened_out"),
        ):
            if stats.get(counter) is None:
                raise ValueError(f"Trace reconciliation counter missing: {counter}")
            if events.get(event, 0) != stats[counter]:
                raise ValueError(
                    f"Person-matching event total differs: {event}={events.get(event, 0)}, {counter}={stats[counter]}"
                )
        workers = []
        for wid in sorted(all_ids):
            workers.append(
                {
                    "worker_id": wid,
                    "opening": opening.get(wid),
                    "after_market": market.get(wid),
                    "after_renewal": post.get(wid),
                    "routes": sorted((name for name, ids in routes.items() if wid in ids)),
                    "matching": matcher["workers"].get(wid),
                    "separation_prephase": current["separation_phase"]["raw_draw"]["workers"].get(
                        wid
                    ),
                    "retirement_action_selected": wid in current["retirement_actions"],
                }
            )
        phase = dict(current["separation_phase"])
        phase["raw_draw"] = {
            key: value for key, value in phase["raw_draw"].items() if key != "workers"
        }
        record = {key: current[key] for key in ("schema", "month_index", "year", "calendar_month")}
        record.update(
            {
                "workers": workers,
                "matcher": {key: value for key, value in matcher.items() if key != "workers"},
                "separation_phase": phase,
                "employment_flow": employment_flow,
                "aggregate_reconciliation": {
                    "route_counts_pass": True,
                    "matching_event_counts_pass": True,
                    "matching_event_totals": events,
                },
            }
        )
        self.stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        self.months += 1
        self.worker_month_rows += len(workers)
        self.current = None

    def close(self):
        if self.current is not None:
            self.abort()
            raise ValueError("Trace ended during a partial month")
        self.stream.close()
        self._release_environment()
        self._closed = True
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {
            "schema": "worker_level_trace_receipt_v1",
            "enabled": True,
            "path": str(self.path),
            "sha256": digest,
            "months": self.months,
            "worker_month_rows": self.worker_month_rows,
            "format": "gzip JSONL; one population-complete record per month",
            "behavior_changed": False,
        }
