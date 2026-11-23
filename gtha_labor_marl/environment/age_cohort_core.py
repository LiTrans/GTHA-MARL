from dataclasses import dataclass
import hashlib
import json
import math
from .net_population_core import RemovalPlan, _ids, _integer

DOMAIN = "age-cohort-renewal-v1"
TERMINAL_AGE = 85


def uniform(*key):
    b = json.dumps([DOMAIN, *key], separators=(",", ":")).encode()
    return (int.from_bytes(hashlib.sha256(b).digest()[:8], "big") >> 11) / float(1 << 53)


def cohort(age, cma):
    if isinstance(age, bool) or not math.isfinite(float(age)) or int(age) != age or (age < 15):
        raise ValueError("Age-15+ integral cohort required")
    if cma not in (532, 535, 537):
        raise ValueError("Unknown CMA")
    return (int(cma), min(int(age), TERMINAL_AGE))


def components(previous_aged, current):
    if any(
        (
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or (not math.isfinite(v))
            or (v < 0)
            for v in (previous_aged, current)
        )
    ):
        raise ValueError("Invalid source population")
    add = max(0.0, current - previous_aged)
    remove = max(0.0, previous_aged - current) / previous_aged if previous_aged else 0.0
    if not math.isclose(previous_aged * (1 - remove) + add, current, rel_tol=1e-12, abs_tol=1e-08):
        raise ValueError("Net-cohort expectation identity failure")
    return (add, remove)


def rounded_entries(expected, seed, year, cma, age):
    if not math.isfinite(expected) or expected < 0:
        raise ValueError("Invalid expected entries")
    base = math.floor(expected)
    return base + int(uniform("entry-rounding", seed, year, cma, age) < expected - base)


@dataclass(frozen=True)
class CohortRemovalPlan(RemovalPlan):
    cohort_keys: tuple[tuple[str, int, int], ...]
    exit_probabilities: tuple[tuple[int, int, float], ...]


def select_cohort_departures(worker_ids, keys, probabilities, seed, year, month=7):
    ids = _ids(worker_ids)
    seed = _integer(seed, "seed")
    year = _integer(year, "year", 1)
    if month != 7:
        raise ValueError("Cohort source is July-dated")
    if set(keys) != set(ids):
        raise ValueError("Cohort keys do not cover live IDs")
    for key, p in probabilities.items():
        if key != cohort(key[1], key[0]) or not math.isfinite(p) or (not 0 <= p <= 1):
            raise ValueError("Invalid cohort net-exit probability")
    if any((k not in probabilities for k in keys.values())):
        raise ValueError("Missing cohort source")
    chosen = {
        wid
        for wid in ids
        if uniform("net-exit", seed, year, *keys[wid], wid) < probabilities[keys[wid]]
    }
    if len(chosen) >= len(ids):
        raise ValueError("Renewal would exhaust the cohort")
    keep = tuple((i for i, w in enumerate(ids) if w not in chosen))
    digest = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    return CohortRemovalPlan(
        ids,
        tuple((w for w in ids if w in chosen)),
        tuple((ids[i] for i in keep)),
        keep,
        seed,
        year,
        month,
        digest,
        tuple(sorted(((w, *k) for w, k in keys.items()))),
        tuple(sorted(((*k, float(p)) for k, p in probabilities.items()))),
    )


def validate_cohort_plan(view, plan):
    keys = {w: (c, a) for w, c, a in plan.cohort_keys}
    probabilities = {(c, a): p for c, a, p in plan.exit_probabilities}
    actual = {w: cohort(ws.worker.age, ws.worker.cma) for w, ws in view.workers.items()}
    if keys != actual:
        raise ValueError("Stale cohort plan: worker age or CMA changed")
    expected = select_cohort_departures(
        view.worker_ids, keys, probabilities, plan.rollout_seed, plan.year, plan.month
    )
    if plan != expected:
        raise ValueError(
            "The repeated selection of departing cohort members gives a different result"
        )
    return expected
