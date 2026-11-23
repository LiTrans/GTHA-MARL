from collections import Counter
import csv
from functools import lru_cache
import hashlib
import math
from market_setup import data_root

HOME = data_root()
SOURCE = HOME / "cohort_reconciliation/single_age_population.csv"
CMAS = (532, 535, 537)
TERMINAL_AGE = 85


def capacity_indexing_enabled():
    if not enabled():
        raise ValueError("Capacity indexation requires cohort renewal")
    return True


def industry_alignment_enabled():
    return False


def employed_entrant_placement_enabled():
    return False


def choose_entrant_firm(candidates, entrant_industry, entrant_cma):
    eligible = [row for row in candidates if row["active"] and row["open_positions"] > 0]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            row["industry"] != entrant_industry,
            row["cma"] != entrant_cma,
            str(row["firm_id"]),
        ),
    )["firm_id"]


def indexed_capacity_targets(
    targets,
    active,
    before_population,
    after_population,
    filled=None,
    reference_population=None,
    reference_capacity=None,
    allocation_weights=None,
):
    if (
        type(before_population) is not int
        or type(after_population) is not int
        or before_population < 1
        or (after_population < 1)
    ):
        raise ValueError("Invalid population for capacity indexation")
    if len(targets) != len(active) or not targets:
        raise ValueError("Capacity target/activity shape mismatch")
    clean = []
    for target, is_active in zip(targets, active):
        if type(target) is not int or target < 0 or type(is_active) is not bool:
            raise ValueError("Invalid capacity target/activity value")
        clean.append(target)
    if filled is None:
        filled = [0] * len(targets)
    if len(filled) != len(targets) or any(
        (type(value) is not int or value < 0 for value in filled)
    ):
        raise ValueError("Invalid filled-employment floors")
    active_indexes = [i for i, value in enumerate(active) if value]
    current = sum((clean[i] for i in active_indexes))
    if not active_indexes or current < 1:
        raise ValueError("No positive active capacity to index")
    if reference_population is None:
        reference_population = before_population
    if reference_capacity is None:
        reference_capacity = current
    if (
        type(reference_population) is not int
        or reference_population < 1
        or type(reference_capacity) is not int
        or (reference_capacity < 1)
    ):
        raise ValueError("Invalid capacity reference")
    desired = round(reference_capacity * after_population / reference_population)
    floor_total = sum((filled[i] for i in active_indexes))
    if floor_total > desired:
        raise ValueError("Filled employment exceeds indexed aggregate capacity")
    distributable = desired - floor_total
    if allocation_weights is None:
        allocation_weights = clean
    if len(allocation_weights) != len(targets) or any(
        (type(value) not in (int, float) or value < 0 for value in allocation_weights)
    ):
        raise ValueError("Invalid capacity-allocation weights")
    weight_total = sum((allocation_weights[i] for i in active_indexes))
    if weight_total <= 0:
        raise ValueError("No positive active capacity-allocation weight")
    exact = {
        i: filled[i] + allocation_weights[i] * distributable / weight_total for i in active_indexes
    }
    result = list(clean)
    for i in active_indexes:
        result[i] = math.floor(exact[i])
    remainder = desired - sum((result[i] for i in active_indexes))
    order = sorted(active_indexes, key=lambda i: (-(exact[i] - result[i]), i))
    for i in order[:remainder]:
        result[i] += 1
    if sum((result[i] for i in active_indexes)) != desired:
        raise ValueError("Indexed capacity allocation failed")
    return result


def index_firm_capacity(env, before_population, after_population):
    if not capacity_indexing_enabled():
        return None
    ids = list(env.firm_ids)
    targets = [int(value) for value in env.firm_target_size.detach().cpu().tolist()]
    active = [bool(value) for value in env.firm_active.detach().cpu().tolist()]
    if not hasattr(env, "_capacity_reference_population"):
        env._capacity_reference_population = before_population
        env._capacity_reference_total = sum((t for t, a in zip(targets, active) if a))
    filled = []
    for fid in ids:
        staff = len(env.state_manager.firms[fid].employees)
        filled.append(staff)
    updated = indexed_capacity_targets(
        targets,
        active,
        before_population,
        after_population,
        filled,
        env._capacity_reference_population,
        env._capacity_reference_total,
        None,
    )
    for i, (fid, target) in enumerate(zip(ids, updated)):
        fs = env.state_manager.firms[fid]
        staff = filled[i]
        if active[i] and target < staff:
            raise ValueError("Indexed target below filled employment")
        fs.firm.target_size = target
        env.firm_target_size[i] = target
        if getattr(env, "firm_base_size", None) is not None:
            env.firm_base_size[i] = target
    return {
        "mode": "population_indexed_capacity_v1",
        "population_before": before_population,
        "population_after": after_population,
        "reference_population": env._capacity_reference_population,
        "reference_capacity": env._capacity_reference_total,
        "active_capacity_before": sum((t for t, a in zip(targets, active) if a)),
        "active_capacity_after": sum((t for t, a in zip(updated, active) if a)),
        "unemployed_search_pressure_by_industry": None,
    }


def source_scope():
    from market_snapshots import plan

    key = "Data/Simulation/Parameters/cohort_population.csv"
    return (HOME / key, key, int(plan()["end_year"]))


def enabled():
    from market_snapshots import plan

    if plan().get("demographic_mode") != "july_net_cohort_renewal_v1":
        raise ValueError("Age renewal requires its own development manifest")
    return True


@lru_cache(maxsize=1)
def source_stocks():
    source, key, end = source_scope()
    if not enabled():
        raise ValueError("Cohort renewal is disabled")
    values = {}
    with source.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = (int(r["year"]), int(r["cma"]), int(r["age"]))
            if k[0] > end:
                continue
            if k in values or r["reference_date"] != "July 1":
                raise ValueError("Duplicate/wrong-dated age source")
            v = float(r["population"]) if k[0] > 2025 and end == 2030 else int(r["population"])
            if not math.isfinite(v) or v < 0:
                raise ValueError("Invalid source population")
            values[k] = v
    expected = {(y, c, a) for y in range(2006, end + 1) for c in CMAS for a in range(91)}
    if set(values) != expected:
        raise ValueError("Age source coverage differs")
    return values


def cohort_stocks():
    raw = source_stocks()
    return {
        (y, c, a): sum((raw[y, c, x] for x in range(a, 91))) if a == TERMINAL_AGE else raw[y, c, a]
        for y in range(2006, source_scope()[2] + 1)
        for c in CMAS
        for a in range(15, TERMINAL_AGE + 1)
    }


def quotas(stocks, n):
    total = sum(stocks.values())
    if n < 1 or total <= 0:
        raise ValueError("Invalid initial population size")
    exact = {k: n * v / total for k, v in stocks.items()}
    result = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - result[k]), k))[: n - sum(result.values())]:
        result[k] += 1
    if sum(result.values()) != n:
        raise ValueError("Initial quota balance failure")
    return result


def donor_groups(pool):
    import pandas as pd

    required = {
        "worker_id",
        "age",
        "cma",
        "weight",
        "da_id",
        "employment_status",
        "archetype_id",
        "immigrant_status",
    }
    if not required <= set(pool.columns) or not pool.worker_id.is_unique:
        raise ValueError("Invalid donor pool")
    selected = pool[pool.cma.isin(CMAS) & (pool.age >= 15)].copy()
    for name in ("age", "cma", "weight", "da_id"):
        values = pd.to_numeric(selected[name], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError("Missing donor " + name)
    if (selected.weight <= 0).any() or (selected.age % 1 != 0).any():
        raise ValueError("Invalid donor age/weight")
    selected["_cohort_age"] = selected.age.clip(upper=TERMINAL_AGE).astype(int)
    return {
        (int(c), int(a)): g.sort_values("worker_id")
        .drop(columns=["_cohort_age"])
        .reset_index(drop=True)
        for (c, a), g in selected.groupby(["cma", "_cohort_age"])
    }


def initial_population(pool, n, seed):
    import numpy as np
    import pandas as pd
    from environment.age_cohort_core import uniform

    s = cohort_stocks()
    allocation = quotas(
        {(c, a): s[2006, c, a] for c in CMAS for a in range(15, TERMINAL_AGE + 1)}, n
    )
    groups = donor_groups(pool)
    parts = []
    for key, count in sorted(allocation.items()):
        if not count:
            continue
        if key not in groups or len(groups[key]) < count:
            raise ValueError(f"Insufficient donor support for initial cohort {key}")
        group = groups[key]
        rng = np.random.default_rng(int(uniform("initial", 42, *key) * (1 << 53)))
        selected = rng.choice(
            len(group), size=count, replace=False, p=group.weight.to_numpy() / group.weight.sum()
        )
        parts.append(group.iloc[selected])
    result = pd.concat(parts, ignore_index=True)
    if len(result) != n or not result.worker_id.is_unique:
        raise ValueError("Initial sample duplicates/size")
    return result


def annual_rule(year, record_weight):
    from environment.age_cohort_core import components

    s = cohort_stocks()
    adds = {}
    probs = {}
    if (
        year not in range(2007, source_scope()[2] + 1)
        or not math.isfinite(record_weight)
        or record_weight <= 0
    ):
        raise ValueError("Renewal year/weight outside scope")
    for c in CMAS:
        for a in range(15, TERMINAL_AGE + 1):
            current = s[year, c, a]
            if a == 15:
                adds[c, a] = current / record_weight
                probs[c, a] = 0.0
            else:
                aged = s[year - 1, c, a - 1] + (
                    s[year - 1, c, TERMINAL_AGE] if a == TERMINAL_AGE else 0
                )
                add, remove = components(aged, current)
                adds[c, a] = add / record_weight
                probs[c, a] = remove
    return (adds, probs)


def clamp_disabled(env):
    n = len(env.state_manager.workers)
    return {
        "theoretical": {"deaths": 0, "emigrants": 0, "immigrants": 0, "new_entrants": 0},
        "actual": {"replaced": 0, "population": n},
        "theoretical_population": n,
        "status": "EARLIER_CLAMP_DISABLED_NOT_ZERO_OBSERVED_FLOWS",
    }


def worker_from_donor(env, row, wid, cma, age, year, seed):
    import copy
    import numpy as np
    from environment.matching_mechanism_kround import Worker
    from environment.state_manager import WorkerState
    from environment.nilf_subtype import normalize_sex_value
    from environment.age_cohort_core import uniform, cohort

    if cohort(int(row.age), int(row.cma)) != (cma, age):
        raise ValueError("Donor age/geography mismatch")

    def number(key, default):
        value = row.get(key, default)
        return float(value) if value is not None and math.isfinite(float(value)) else default

    status = str(row.employment_status)
    if status not in ("Employed", "E", "Unemployed", "U", "NILF"):
        raise ValueError("Unknown donor employment label")
    searching = status != "NILF"
    industry = int(number("naics_18", 8))
    industry = 5 if industry == 6 else industry
    raw = str(row.immigrant_status).lower()
    if raw not in ("true", "false", "1", "0", "1.0", "0.0"):
        raise ValueError("Invalid donor immigrant flag")
    worker = Worker(
        id=wid,
        age=int(row.age),
        skill_level=number("skill_level", 2),
        education=int(number("education", 2)),
        industry=industry,
        sector=int(number("sector_id", 0)),
        experience_years=number("experience_total", 0),
        cma=cma,
        da_id=int(row.da_id),
        employment_status="Unemployed" if searching else "NILF",
        current_wage=0.0,
        is_searching=searching,
        unemployment_months=0,
        hours_per_week=number("hours_per_week", 40),
        sex=normalize_sex_value(row.sex),
        archetype_id=int(row.archetype_id),
        archetype_name=str(row.archetype_name),
        sector_id=int(number("sector_id", 0)),
        sector_name=str(row.get("sector_name", "services")),
    )
    worker.immigrant_status = raw in ("true", "1", "1.0")
    worker.years_since_arrival = number("years_since_arrival", None)
    worker.renewal_arrival_year = year
    worker.renewal_arrival_month = 7
    worker.renewal_entrant_type = "net_cohort_not_identified_migrant"
    worker.renewal_profile_status = status
    ws = WorkerState(worker=worker, separation_reason="re_entrant")
    rng = np.random.default_rng(
        int(uniform("entry-attributes", seed, year, cma, age, wid) * (1 << 53))
    )
    if not searching:
        sampler = copy.deepcopy(env.state_manager.nilf_subtype_sampler)
        if sampler is None:
            raise ValueError("Entrant NILF subtype sampler missing")
        sampler._rng = rng
        ws.nilf_subtype = sampler.sample(year, worker.age, worker.sex)
        ws.nilf_subtype_months = 0
    mean, std = env.config.mean_field.skill_distributions.get(worker.archetype_id, (0.5, 0.1))
    skill = float(np.clip(rng.normal(mean, std), 0.05, 0.95))
    return (ws, skill)


def reconcile_boundary_firm_cache(env, *, resident_only_observations=False):
    sm = env.state_manager
    expected = {fid: set() for fid in sm.firms}
    boundary = getattr(env, "boundary", None)
    if boundary is not None:
        boundary.reconcile(env)
    for wid, ws in sm.workers.items():
        if ws.worker.employment_status == "Employed":
            if boundary is not None and wid in boundary.jobs:
                boundary.validate_job(env, wid)
                continue
            fid = ws.worker.current_employer_id
            if fid not in expected:
                raise RuntimeError("Cohort boundary has an employed worker without a live employer")
            expected[fid].add(wid)
    indexes = env._firm_id_to_idx
    if set(indexes) != set(sm.firms) or set(indexes.values()) != set(
        range(len(env.firm_employment))
    ):
        raise RuntimeError("Cohort boundary firm cache index is incomplete or duplicated")
    st = env.fast_obs.state_table
    staged = env.firm_employment.clone()
    changes = []
    for fid, fs in sm.firms.items():
        roster = list(fs.employees)
        if len(set(roster)) != len(roster) or set(roster) != expected[fid]:
            raise RuntimeError("Cohort boundary employer/roster disagreement: " + str(fid))
        idx = indexes[fid]
        count = len(roster) + 0
        if not bool(env.firm_active[idx]) and count:
            raise RuntimeError("Inactive firm has employees at cohort boundary")
        obs_idx = st.firm_id_to_idx.get(fid)
        observation_count = len(roster) if resident_only_observations else count
        if obs_idx is None or int(st.f_n_employees[obs_idx]) != observation_count:
            raise RuntimeError("Cohort boundary post-market observation disagreement: " + str(fid))
        before = int(staged[idx])
        if before != count:
            changes.append(
                {"firm_id": str(fid), "earlier_cache": before, "post_market_count": count}
            )
        staged[idx] = count
    env.firm_employment = staged
    for fid, fs in sm.firms.items():
        fs.firm.n_employees = len(expected[fid]) + 0
    return changes


def controller_class():
    from environment.net_population_runtime import NetPopulationController
    from environment.net_population_core import stage_registry_removal, stage_observation_compaction
    from environment.age_cohort_core import (
        select_cohort_departures,
        cohort,
        rounded_entries,
        uniform,
    )

    class AgeRenewalController(NetPopulationController):

        def reset(self):
            super().reset()
            self.renewals = []
            self.finished = set()
            self.next_check = None
            self._cohort_plan = None
            self._donors = None
            self.initial_cohorts = None

        def capture_initial(self, env):
            if self.initial_cohorts is not None or self.renewals:
                raise RuntimeError("Repeated initial cohort capture")
            self.initial_cohorts = {
                w: cohort(ws.worker.age, ws.worker.cma)
                for w, ws in env.state_manager.workers.items()
            }
            s = cohort_stocks()
            expected = quotas(
                {(c, a): s[2006, c, a] for c in CMAS for a in range(15, TERMINAL_AGE + 1)},
                env.n_workers_fixed,
            )
            actual = Counter(self.initial_cohorts.values())
            if any((actual[k] != n for k, n in expected.items())):
                raise ValueError("Reset changed initial cohort quotas")

        def seed(self, seed):
            if self.renewals or self.next_check is not None:
                raise RuntimeError("Cannot reseed active cohort renewal")
            super().seed(seed)

        def schedule(self, *args, **kwargs):
            raise RuntimeError("Old net-growth scheduler cannot run with cohort renewal")

        def queue(self, *args, **kwargs):
            raise RuntimeError("Old net-removal queue cannot run with cohort renewal")

        def assert_step_ready(self):
            super().assert_step_ready()
            if self.next_check is not None:
                raise RuntimeError("Cohort boundary was not finalized before next actor step")

        def note_calendar(self, year, month):
            period = (int(year), int(month))
            if period[1] == 7 and 2007 <= period[0] <= self.end_year:
                if period[0] in self.finished or self.next_check is not None:
                    raise RuntimeError("Repeated cohort boundary")
                self.next_check = period

        def _stage_removal(self, view, fast_obs, count, year, month):
            plan = self._cohort_plan
            if plan is None or len(plan.departed_ids) != count:
                raise ValueError("Missing cohort removal plan")
            staged = stage_registry_removal(view, plan)
            compact = stage_observation_compaction(fast_obs, plan)
            for i, w in enumerate(plan.original_ids):
                if int(fast_obs.state_table.w_age[i].item()) != int(view.workers[w].worker.age):
                    raise ValueError("Stale age tensor at renewal")
            return (plan, staged, compact)

        def finalize(self, env, obs):
            if self.failed:
                raise RuntimeError("Failed cohort transaction cannot continue")
            if self.next_check is None:
                return obs
            if self.rollout_seed is None:
                raise RuntimeError("Cohort renewal requires an explicit rollout seed")
            if self.initial_cohorts is None:
                raise RuntimeError("Missing initial cohort capture")
            year, month = self.next_check
            if (env.current_year, env.current_month) != (year, month):
                raise ValueError("Cohort boundary/calendar mismatch")
            before = len(env.worker_ids)
            weight = sum(
                (v for (y, c, a), v in source_stocks().items() if y == 2006 and a >= 15)
            ) / float(env.n_workers_fixed)
            adds, probs = annual_rule(year, weight)
            keys = {
                w: cohort(ws.worker.age, ws.worker.cma)
                for w, ws in env.state_manager.workers.items()
            }
            plan = select_cohort_departures(env.worker_ids, keys, probs, self.rollout_seed, year)
            if any((a == 15 for c, a in keys.values())):
                raise ValueError(
                    "Age15 incumbent after January ageing would double-count adult entry"
                )
            if self._donors is None:
                self._donors = donor_groups(env.full_population_df)
            entries = []
            cell_records = []
            for (c, a), expected in sorted(adds.items()):
                count = rounded_entries(expected, self.rollout_seed, year, c, a)
                if expected > 0 and (c, a) not in self._donors:
                    raise ValueError(f"Missing entrant donor support {(c, a)}")
                if count:
                    group = self._donors[c, a]
                    weights = group.weight.to_numpy(dtype=float)
                    cdf = (weights / weights.sum()).cumsum()
                    for i in range(count):
                        import numpy as np

                        wid = f"r6x_cohort_{year}_{c}_{a}_{i}"
                        if wid in env._renewal_audit_created or wid in env.state_manager.workers:
                            raise ValueError("Entrant ID collision")
                        idx = min(
                            int(
                                np.searchsorted(
                                    cdf,
                                    uniform("donor", self.rollout_seed, year, c, a, i),
                                    side="right",
                                )
                            ),
                            len(group) - 1,
                        )
                        row = group.iloc[idx]
                        ws, skill = worker_from_donor(env, row, wid, c, a, year, self.rollout_seed)
                        entries.append((wid, ws, skill, str(row.worker_id)))
                cell_records.append(
                    {
                        "cma": c,
                        "age": a,
                        "expected_entries": expected,
                        "entries": count,
                        "net_exit_probability": probs[c, a],
                        "eligible": sum((k == (c, a) for k in keys.values())),
                        "exits": sum((keys[w] == (c, a) for w in plan.departed_ids)),
                    }
                )
            self._cohort_plan = plan
            try:
                self.next_check = None
                if plan.departed_ids:
                    self.due = (year, month, len(plan.departed_ids))
                    obs = super().finalize(env, obs)
                sm = env.state_manager
                for wid, ws, skill, donor in entries:
                    sm.workers[wid] = ws
                    env.worker_ids.append(wid)
                    env.sampled_worker_ids.add(wid)
                    env._renewal_audit_created.add(wid)
                    sm.worker_types[wid] = ws.worker.archetype_id
                    sm.worker_skills[wid] = skill
                env.n_workers = len(env.worker_ids)
                sm._update_aggregate_stats()
                sm.compute_worker_type_mean_fields()
                capacity_indexing = index_firm_capacity(env, before, env.n_workers)
                sm._update_aggregate_stats()
                sm.compute_worker_type_mean_fields()
                _budget_layoffs = capacity_indexing["capacity_budget_contraction_layoffs"]
                if entries or _budget_layoffs:
                    env._renewal_resize_obs()
                sm._state_table_ref = env.fast_obs.state_table
                sm._state_table_worker_idx = dict(env.fast_obs.state_table.worker_id_to_idx)
                cache_refresh = reconcile_boundary_firm_cache(
                    env, resident_only_observations=bool(entries)
                )
                env._renewal_assert_population_registry()
                env._renewal_assert_firm_employment()
                refreshed = env._get_observation_tensordict()
                for key in ("done", "terminated", "truncated"):
                    refreshed[key] = obs[key].clone()
                if env.n_workers != before - len(plan.departed_ids) + len(entries):
                    raise RuntimeError("Cohort balance failure")
                self.finished.add(year)
                self.renewals.append(
                    {
                        "year": year,
                        "month": 7,
                        "before": before,
                        "after": env.n_workers,
                        "before_cohorts": [[w, *k] for w, k in sorted(keys.items())],
                        "record_weight": weight,
                        "exits": list(plan.departed_ids),
                        "entries": [
                            {
                                "id": w,
                                "donor_id": d,
                                "age": ws.worker.age,
                                "cma": ws.worker.cma,
                                "entry_status": ws.worker.employment_status,
                                "wage": ws.worker.current_wage,
                            }
                            for w, ws, sk, d in entries
                        ],
                        "cells": cell_records,
                        "gross_flow_causes_identified": False,
                        "capacity_indexing": capacity_indexing,
                        "employed_entrant_placement": None,
                        "post_market_firm_cache_refresh": cache_refresh,
                    }
                )
                return refreshed
            except BaseException:
                self.failed = True
                raise

        def summary(self, env):
            self.assert_step_ready()
            env._renewal_assert_population_registry()
            if self.finished != set(range(2007, self.end_year + 1)):
                raise ValueError("Incomplete July renewal calendar")
            return {
                "enabled": True,
                "schema": "july_net_cohort_renewal_v1",
                "decision_bearing": False,
                "verdict": "PASS",
                "rollout_seed": self.rollout_seed,
                "source_sha256": hashlib.sha256(source_scope()[0].read_bytes()).hexdigest(),
                "end_year": self.end_year,
                "projected_expectations_not_observed": self.end_year > 2024,
                "earlier_clamp_applied": False,
                "earlier_net_growth_applied": False,
                "gross_flow_causes_identified": False,
                "terminal_age_group": "85+",
                "initial_reference_convention": "July 2006 composition used at January 2006 reset",
                "initial_cohorts": [[w, *k] for w, k in sorted(self.initial_cohorts.items())],
                "renewals": self.renewals,
                "initial_population": env._renewal_audit_base_n,
                "final_population": env.n_workers,
                "cumulative_entries": len(env._renewal_audit_created),
                "cumulative_exits": len(self.removed),
                "capacity_indexing_enabled": capacity_indexing_enabled(),
                "industry_alignment_enabled": False,
                "employed_entrant_placement_enabled": False,
                "no_employment_or_participation_targets": True,
            }

    return AgeRenewalController
