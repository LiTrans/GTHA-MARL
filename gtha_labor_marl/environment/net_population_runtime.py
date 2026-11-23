import copy
from .net_population_core import RegistryView, plan_signed_net, stage_candidate_removal
from trade_war_stress.net_projection_schema import load_projection


class NetPopulationController:

    def __init__(self, path, end_year):
        from population_participation import pre2025_scope

        scope = pre2025_scope()
        if scope is not None and scope["simulation.end_year"] == 2030:
            from age_renewal import enabled, source_stocks

            if end_year != 2030 or not enabled():
                raise ValueError("The forward projection requires its age-cohort controller")
            source_stocks()
            self.rows = {}
            self.metadata = {
                "unused_earlier_net_projection": True,
                "source": "age_cohort_candidate",
            }
            self.source_end_year = self.end_year = 2030
            self.reset()
            return
        source_end_year = 2030 if scope is not None else end_year
        self.rows, self.metadata = load_projection(path, source_end_year)
        self.source_end_year = source_end_year
        if scope is not None and (
            end_year != scope["simulation.end_year"] or any((y <= end_year for y in self.rows))
        ):
            raise ValueError("Future projection overlaps the pre-2025 development horizon")
        self.end_year = end_year
        self.reset()

    def reset(self):
        self.rollout_seed = None
        self.due = None
        self.removed = {}
        self.annual_plans = []
        self.events = []
        self.failed = False

    def seed(self, seed):
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError("net projection requires a nonnegative integral rollout seed")
        if self.annual_plans or self.events or self.due:
            raise RuntimeError("cannot reseed an active net projection rollout")
        self.rollout_seed = int(seed)

    def schedule(self, env, year, weight):
        if self.rollout_seed is None:
            raise RuntimeError("net projection has no explicit rollout seed")
        if self.due or any((r["year"] == year for r in self.annual_plans)):
            raise RuntimeError("unfinalized or duplicate projection year")
        plan = plan_signed_net(year, self.rows[year], weight, env._renewal_carry)
        env._renewal_carry = plan.carry_out
        self.annual_plans.append(
            {
                "year": year,
                "net_people": plan.net_people,
                "record_weight": weight,
                "carry_in": plan.carry_in,
                "carry_out": plan.carry_out,
                "signed_records": plan.signed_records,
            }
        )
        return {
            "year": year,
            "built": 0,
            "statuses": [],
            "removed_scheduled": 0,
            "net_projection": True,
            "batches": list(plan.monthly_batches),
        }

    def queue(self, env, year, month, count):
        if self.failed or self.due is not None or self.rollout_seed is None:
            raise RuntimeError("invalid net-exit transaction boundary")
        if count <= 0 or count >= len(env.worker_ids):
            raise RuntimeError("net exits must leave a nonempty cohort")
        self.due = (int(year), int(month), int(count))

    def assert_step_ready(self):
        if self.failed or self.due is not None:
            raise RuntimeError("net exits not finalized after previous evaluation step")

    def _stage_removal(self, view, fast_obs, count, year, month):
        return stage_candidate_removal(view, fast_obs, count, self.rollout_seed, year, month)

    def finalize(self, env, obs):
        if self.failed:
            raise RuntimeError("net population controller poisoned by prior failed transaction")
        if self.due is None:
            return obs
        year, month, count = self.due
        sm = env.state_manager
        live = set(env.worker_ids)
        view = RegistryView(
            tuple(env.worker_ids),
            sm.workers,
            frozenset(env.sampled_worker_ids),
            {fid: tuple(fs.employees) for fid, fs in sm.firms.items()},
            {
                "skills": {w: v for w, v in sm.worker_skills.items() if w in live},
                "types": {w: v for w, v in sm.worker_types.items() if w in live},
                "firm_map": {w: v for w, v in env._worker_firm_map.items() if w in live},
            },
            {},
            {
                "state_table_worker_idx": {
                    w: v for w, v in sm._state_table_worker_idx.items() if w in live
                }
            },
        )
        plan, staged, compact = self._stage_removal(view, env.fast_obs, count, year, month)
        if set(plan.departed_ids) & set(self.removed):
            raise RuntimeError("worker exited twice")
        records = [
            {
                "worker_id": w,
                "status_at_exit": sm.workers[w].worker.employment_status,
                "age_at_exit": float(sm.workers[w].worker.age),
                "industry_at_exit": int(sm.workers[w].worker.industry),
                "archetype_at_exit": int(sm.workers[w].worker.archetype_id),
                "employer_at_exit": getattr(sm.workers[w].worker, "current_employer_id", None),
            }
            for w in plan.departed_ids
        ]
        st, hb, vs = (
            env.fast_obs.state_table,
            env.fast_obs.history_buffer,
            env.fast_obs.vacancy_selector,
        )
        saved = []

        def put(owner, key, value):
            saved.append((owner, key, getattr(owner, key)))
            setattr(owner, key, value)

        old_removed = dict(self.removed)
        try:
            put(env, "worker_ids", list(staged.view.worker_ids))
            put(env, "sampled_worker_ids", set(staged.view.sampled_worker_ids))
            put(env, "n_workers", staged.after)
            put(env, "_worker_firm_map", dict(staged.view.live_worker_maps["firm_map"]))
            put(sm, "workers", dict(staged.view.workers))
            put(sm, "worker_skills", dict(staged.view.live_worker_maps["skills"]))
            put(sm, "worker_types", dict(staged.view.live_worker_maps["types"]))
            put(
                sm,
                "_state_table_worker_idx",
                dict(staged.view.worker_index_maps["state_table_worker_idx"]),
            )
            put(sm, "stats", dict(sm.stats))
            put(sm, "worker_type_mean_fields", copy.deepcopy(sm.worker_type_mean_fields))
            if hasattr(sm, "age_skill_mean_fields"):
                put(sm, "age_skill_mean_fields", copy.deepcopy(sm.age_skill_mean_fields))
            for fid, roster in staged.view.firm_rosters.items():
                fs = sm.firms[fid]
                put(fs, "employees", list(roster))
                workplace_count = len(roster)
                put(fs.firm, "n_employees", workplace_count)
            put(env, "firm_employment", env.firm_employment.clone())
            for fid, roster in staged.view.firm_rosters.items():
                env.firm_employment[env._firm_id_to_idx[fid]] = sm.firms[fid].firm.n_employees
            import torch

            for key, value in list(vars(st).items()):
                if torch.is_tensor(value):
                    put(
                        st,
                        key,
                        (
                            compact.worker_tensors[key]
                            if key in compact.worker_tensors
                            else value.clone()
                        ),
                    )
            put(st, "worker_id_to_idx", dict(compact.worker_id_to_idx))
            for obj in (st, hb, vs):
                put(obj, "n_workers", staged.after)
            put(hb, "_buffer", compact.history_buffer)
            put(hb, "_ptr", compact.history_pointer)
            put(vs, "_context", compact.vacancy_context)
            self.removed.update(
                {r["worker_id"]: {**r, "year": year, "month": month} for r in records}
            )
            sm._update_aggregate_stats()
            sm.compute_worker_type_mean_fields()
            st.sync_from_state_manager(sm, env.worker_ids)
            env._renewal_assert_worker_slot_order()
            env._renewal_assert_population_registry()
            env._renewal_assert_firm_employment()
            env._make_specs()
            refreshed = env._get_observation_tensordict()
            for key in ("done", "terminated", "truncated"):
                refreshed[key] = obs[key].clone()
        except BaseException:
            for owner, key, old in reversed(saved):
                setattr(owner, key, old)
            self.removed = old_removed
            self.failed = True
            raise
        self.events.append(
            {
                "event_type": "projected_net_population_exit",
                "year": year,
                "month": month,
                "rollout_seed": self.rollout_seed,
                "before": staged.before,
                "after": staged.after,
                "removed": count,
                "eligible_ids_sha256": plan.eligible_ids_sha256,
                "departed": records,
                "roster_departures": staged.roster_departures_by_firm,
                "selection": "employment_blind_hash_priority_v1",
            }
        )
        self.due = None
        return refreshed

    def summary(self, env):
        self.assert_step_ready()
        for row in self.annual_plans:
            expected = max(0, -row["signed_records"])
            actual = sum((e["removed"] for e in self.events if e["year"] == row["year"]))
            if actual != expected:
                raise RuntimeError("net exit schedule incomplete")
        env._renewal_assert_population_registry()
        return {
            "enabled": True,
            "schema": "net_population_audit_v1",
            "verdict": "PASS",
            "rollout_seed": self.rollout_seed,
            "source": self.metadata,
            "annual_plans": self.annual_plans,
            "exit_events": self.events,
            "removed_total": len(self.removed),
            "final_population": env.n_workers,
            "initial_population": env._renewal_audit_base_n,
            "cumulative_entries": len(env._renewal_audit_created),
            "boundary": "after_completed_month_logging_before_next_policy",
            "selection": "employment_blind_hash_priority_v1",
        }
