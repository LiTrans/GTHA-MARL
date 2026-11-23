from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Mapping

STATUS = "computed"
SELECTION_DOMAIN = "trade-war-net-removal-candidate-v1"


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric data")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return value


@dataclass(frozen=True)
class AnnualPlan:
    year: int
    net_people: float
    record_weight: float
    carry_in: float
    signed_records: int
    carry_out: float
    monthly_batches: tuple[int, ...]


def plan_signed_net(year, net_people, record_weight, carry_in):
    year = _integer(year, "year", 1)
    net = _finite(net_people, "net_people")
    weight = _finite(record_weight, "record_weight")
    carry = _finite(carry_in, "carry_in")
    if weight <= 0 or not 0 <= carry < 1:
        raise ValueError("record_weight must be positive and carry_in in [0, 1)")
    exact = net / weight + carry
    if not math.isfinite(exact):
        raise ValueError("net/weight overflow")
    records = math.floor(exact)
    remainder = exact - records
    if not 0 <= remainder < 1:
        raise ValueError("invalid fractional carry after rounding")
    base, extra = divmod(abs(records), 12)
    sign = -1 if records < 0 else 1
    batches = tuple((sign * (base + (month < extra)) for month in range(12)))
    assert sum(batches) == records
    return AnnualPlan(year, net, weight, carry, records, remainder, batches)


def terminal_stub(year, registered_end_year, net_people, pending_batches=()):
    year = _integer(year, "year", 1)
    end = _integer(registered_end_year, "registered_end_year", 1)
    if year != end + 1:
        raise ValueError("Expected exactly the year after the final year")
    if _finite(net_people, "terminal net_people") != 0 or tuple(pending_batches):
        raise ValueError("terminal stub must be zero with no pending schedule")
    return {"year": year, "scheduled": 0, "removed": 0, "terminal_stub": True, "status": STATUS}


def _ids(worker_ids):
    ids = tuple(worker_ids)
    if not ids or any((not isinstance(wid, str) or not wid for wid in ids)):
        raise ValueError("worker IDs must be nonempty strings in a nonempty cohort")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate worker IDs")
    return ids


@dataclass(frozen=True)
class RemovalPlan:
    original_ids: tuple[str, ...]
    departed_ids: tuple[str, ...]
    survivor_ids: tuple[str, ...]
    survivor_indices: tuple[int, ...]
    rollout_seed: int
    year: int
    month: int
    eligible_ids_sha256: str


def select_departures(worker_ids, count, rollout_seed, year, month):
    ids = _ids(worker_ids)
    count = _integer(count, "departure count")
    seed = _integer(rollout_seed, "rollout_seed")
    year = _integer(year, "year", 1)
    month = _integer(month, "month", 1)
    if month > 12:
        raise ValueError("month must be in 1..12")
    if count >= len(ids):
        raise ValueError("net removal cannot exhaust the computational cohort")
    encoded_ids = json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded_ids).hexdigest()

    def priority(wid):
        key = [SELECTION_DOMAIN, seed, year, month, wid]
        blob = json.dumps(key, separators=(",", ":")).encode("utf-8")
        return (hashlib.sha256(blob).digest(), wid)

    chosen = set(sorted(ids, key=priority)[:count]) if count else set()
    keep = tuple((i for i, wid in enumerate(ids) if wid not in chosen))
    return RemovalPlan(
        ids,
        tuple((wid for wid in ids if wid in chosen)),
        tuple((ids[i] for i in keep)),
        keep,
        seed,
        year,
        month,
        digest,
    )


@dataclass(frozen=True)
class RegistryView:
    worker_ids: tuple[str, ...]
    workers: Mapping[str, Any]
    sampled_worker_ids: frozenset[str]
    firm_rosters: Mapping[str, tuple[str, ...]]
    live_worker_maps: Mapping[str, Mapping[str, Any]]
    live_worker_sets: Mapping[str, frozenset[str]]
    worker_index_maps: Mapping[str, Mapping[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class StagedRegistryRemoval:
    view: RegistryView
    archived_departures: Mapping[str, Any]
    roster_departures_by_firm: Mapping[str, int]
    before: int
    after: int
    added: int
    removed: int
    event_type: str = "projected_net_population_exit"


def stage_registry_removal(view, plan):
    ids = _ids(view.worker_ids)
    if ids != plan.original_ids:
        raise ValueError("stale removal plan: worker slot order changed")
    from .age_cohort_core import CohortRemovalPlan, validate_cohort_plan

    expected_plan = (
        validate_cohort_plan(view, plan)
        if isinstance(plan, CohortRemovalPlan)
        else select_departures(
            ids, len(plan.departed_ids), plan.rollout_seed, plan.year, plan.month
        )
    )
    if plan != expected_plan:
        raise ValueError("The removal plan differs from the recorded selection of residents")
    live = set(ids)
    if set(view.workers) != live or set(view.sampled_worker_ids) != live:
        raise ValueError("worker / sampled registry membership differs")
    roster_seen = set()
    for fid, roster in view.firm_rosters.items():
        for wid in roster:
            if wid not in live or wid in roster_seen:
                raise ValueError(f"unknown or duplicate roster member: {fid}, {wid}")
            roster_seen.add(wid)
    for name, mapping in view.live_worker_maps.items():
        if not set(mapping) <= live:
            raise ValueError(f"live map {name} has non-live IDs; do not silently repair")
    for name, values in view.live_worker_sets.items():
        if not set(values) <= live:
            raise ValueError(f"live set {name} has non-live IDs")
    old_indices = {w: i for i, w in enumerate(ids)}
    new_indices = {w: i for i, w in enumerate(plan.survivor_ids)}
    for name, mapping in view.worker_index_maps.items():
        for wid, index in mapping.items():
            if (
                wid not in live
                or isinstance(index, bool)
                or (not isinstance(index, Integral))
                or (index != old_indices[wid])
            ):
                raise ValueError(f"index map {name} has stale or unmapped worker rows")
    departed = set(plan.departed_ids)
    rosters = {
        fid: tuple((w for w in roster if w not in departed))
        for fid, roster in view.firm_rosters.items()
    }
    staged = RegistryView(
        plan.survivor_ids,
        {wid: view.workers[wid] for wid in plan.survivor_ids},
        frozenset(plan.survivor_ids),
        rosters,
        {
            name: {w: value for w, value in mapping.items() if w not in departed}
            for name, mapping in view.live_worker_maps.items()
        },
        {name: frozenset(set(values) - departed) for name, values in view.live_worker_sets.items()},
        {
            name: {w: new_indices[w] for w in mapping if w not in departed}
            for name, mapping in view.worker_index_maps.items()
        },
    )
    return StagedRegistryRemoval(
        staged,
        {w: view.workers[w] for w in plan.departed_ids},
        {fid: len(view.firm_rosters[fid]) - len(roster) for fid, roster in rosters.items()},
        len(ids),
        len(plan.survivor_ids),
        0,
        len(departed),
    )


WORKER_TENSORS = (
    "w_age",
    "w_skill",
    "w_education",
    "w_experience",
    "w_status",
    "w_wage",
    "w_industry",
    "w_tenure",
    "w_unemployment_months",
    "w_archetype_id",
    "w_sector_id",
    "w_job_apps",
    "w_is_searching",
    "w_ei_eligible",
    "w_ei_weeks",
    "w_cma",
    "w_da",
    "w_mean_field",
    "w_br_skill",
    "w_nilf_subtype",
    "w_nilf_subtype_months",
    "w_attachment",
    "_worker_obs",
)
GLOBAL_WORKER_TENSORS = ("w_cross_type", "w_age_skill_mf")


@dataclass(frozen=True)
class CompactedObservationRows:
    n_workers: int
    worker_id_to_idx: Mapping[str, int]
    worker_tensors: Mapping[str, Any]
    history_buffer: Any
    history_pointer: Any
    vacancy_context: Any


def stage_observation_compaction(fast_obs, plan):
    import torch

    st = fast_obs.state_table
    hb = fast_obs.history_buffer
    vs = fast_obs.vacancy_selector
    n = len(plan.original_ids)
    for name, obj in (("state_table", st), ("history_buffer", hb), ("vacancy_selector", vs)):
        if obj.n_workers != n:
            raise ValueError(f"{name} worker count mismatch")
    for i, wid in enumerate(plan.original_ids):
        if st.worker_id_to_idx.get(wid) != i:
            raise ValueError(
                "unmapped or permuted slot; wait for normal sync, do not weaken invariant"
            )
    expected = set(WORKER_TENSORS) | set(GLOBAL_WORKER_TENSORS)
    actual = {
        name
        for name, value in vars(st).items()
        if (name.startswith("w_") or name == "_worker_obs") and torch.is_tensor(value)
    }
    if actual != expected:
        raise ValueError(
            f"state-table tensor inventory changed: missing={expected - actual}, extra={actual - expected}"
        )

    def gather(value, name):
        if not torch.is_tensor(value) or value.ndim == 0 or value.shape[0] != n:
            raise ValueError(f"invalid worker-row tensor {name}")
        rows = torch.tensor(plan.survivor_indices, dtype=torch.long, device=value.device)
        return value.index_select(0, rows).clone()

    return CompactedObservationRows(
        len(plan.survivor_ids),
        {wid: i for i, wid in enumerate(plan.survivor_ids)},
        {name: gather(getattr(st, name), name) for name in WORKER_TENSORS},
        gather(hb._buffer, "history_buffer"),
        gather(hb._ptr, "history_pointer"),
        gather(vs._context, "vacancy_context"),
    )


def stage_candidate_removal(view, fast_obs, count, rollout_seed, year, month):
    count = _integer(count, "departure count")
    if count == 0:
        return None
    plan = select_departures(view.worker_ids, count, rollout_seed, year, month)
    registries = stage_registry_removal(view, plan)
    observations = stage_observation_compaction(fast_obs, plan)
    for i, wid in enumerate(plan.original_ids):
        worker = view.workers[wid].worker
        if worker.id != wid or not math.isclose(
            float(worker.age),
            float(fast_obs.state_table.w_age[i].item()),
            rel_tol=0,
            abs_tol=0.0001,
        ):
            raise ValueError("worker identity/age differs from its tensor row before compaction")
    if registries.after != observations.n_workers or registries.before - count != registries.after:
        raise ValueError("population identity failed")
    return (plan, registries, observations)
