from dataclasses import replace
from functools import wraps

VERSION = "tariff_funded_effective_budget_view_v1"


def install(budget, lineage, trade_class):
    if getattr(budget, "_tariff_budget_view_installed", False):
        raise ValueError("Tariff funded/effective budget view already installed")
    if not getattr(trade_class, "_market_tariff_bridge_installed", False):
        raise ValueError("Install the tariff target link first")
    original = {
        name: getattr(budget, name)
        for name in (
            "_raw",
            "_write_targets",
            "audit",
            "remaining_slots",
            "index",
            "commit_entry",
            "initialize",
        )
    }
    native_after_capacity_update = trade_class.after_capacity_update

    @wraps(native_after_capacity_update)
    def after_capacity_update(self, env):
        result = native_after_capacity_update(self, env)
        if not self.enabled or not hasattr(env, "_capacity_lineage_state"):
            return result
        state = lineage._state(env)
        refreshed = []
        for fid, firm_state in env.state_manager.firms.items():
            key = str(fid)
            shock = self._current_shocks.get(key)
            if (
                shock is None
                or shock.employment_multiplier != 1.0
                or key in self._tariff_bridge_pending_first_shock
                or (not self._is_active_firm(env, fid))
                or (self._baseline_firm_objects.get(key) != id(firm_state.firm))
            ):
                continue
            baseline = budget._int(self._baseline_targets[key], "native inert tariff baseline")
            physical = budget._int(firm_state.firm.target_size, "native inert effective target")
            funded = state["slots"][key]["unshocked_capacity_target"]
            if physical != baseline or baseline != funded:
                raise budget.BudgetViolation(
                    "Inert tariff metadata cannot hide an unexplained target write"
                )
            if shock.baseline_target != baseline or shock.shocked_target != baseline:
                self._current_shocks[key] = replace(
                    shock, baseline_target=baseline, shocked_target=baseline
                )
                refreshed.append(
                    {
                        "slot_id": key,
                        "old_baseline": shock.baseline_target,
                        "native_baseline": baseline,
                        "physical_target": physical,
                        "employment_multiplier": 1.0,
                        "target_writes": 0,
                    }
                )
        if refreshed:
            lineage._event(
                state,
                "TARIFF_INERT_BASELINE_METADATA_REFRESH",
                version=VERSION,
                source_context=lineage._context(env),
                rows=refreshed,
                economic_values_changed=False,
            )
        return result

    @wraps(original["initialize"])
    def initialize(env):
        state, targets, _, _ = lineage._check_current(env)
        if "tariff_overlay_authority" in state:
            raise budget.BudgetViolation("The tariff target record was set twice during reset")
        state["tariff_overlay_authority"] = "tariff_capacity_overlay_authority_v1"
        for fid, record in state["slots"].items():
            record["unshocked_capacity_target"] = targets[fid]
            record["tariff_effective_target"] = targets[fid]
        return original["initialize"](env)

    def dual(env):
        ids, physical, active, firms, filled = original["_raw"](env)
        state = lineage._state(env)
        if state.get("tariff_overlay_authority") != "tariff_capacity_overlay_authority_v1":
            raise budget.BudgetViolation("Unknown tariff target record")
        claims = [
            budget._int(state["slots"][fid]["unshocked_capacity_target"], "unshocked funded quota")
            for fid in ids
        ]
        for fid, effective, funded in zip(ids, physical, claims):
            row = state["slots"][fid]
            if effective != row["current_target"] or effective != row["tariff_effective_target"]:
                raise budget.BudgetViolation("The tariff target and its capacity lineage disagree")
            if effective > funded:
                raise budget.BudgetViolation("Tariff schedule cannot mint funded quota")
        return (ids, physical, claims, active, firms, filled)

    def raw(env):
        ids, _, claims, active, firms, filled = dual(env)
        return (ids, claims, active, firms, filled)

    def sync_after_write(env, source, before):
        state = lineage._state(env)
        ids, physical, active, firms, _ = original["_raw"](env)
        regime = env._trade_war_regime
        writes = []
        for fid, target, on in zip(ids, physical, active):
            row = state["slots"][fid]
            funded = row["unshocked_capacity_target"]
            shock = regime._current_shocks.get(fid)
            pending = fid in regime._tariff_bridge_pending_first_shock
            if on and (not pending) and (shock is not None):
                if regime._baseline_targets.get(fid) != funded or shock.baseline_target != funded:
                    raise budget.BudgetViolation(
                        "The tariff baseline differs from the funded quota"
                    )
                expected = 0 if funded == 0 else max(1, round(funded * shock.employment_multiplier))
            else:
                expected = funded
            if target != expected:
                raise budget.BudgetViolation(
                    "A tariff change is not the single multiplier of its target"
                )
            row["current_target"] = row["tariff_effective_target"] = target
            if before.get(fid) != target:
                writes.append(
                    {
                        "slot_id": fid,
                        "funded_quota": funded,
                        "effective_before": before.get(fid),
                        "effective_after": target,
                        "source": source,
                        "formula_residual": target - expected,
                    }
                )
        if writes:
            lineage._event(
                state,
                "TARIFF_FUNDED_EFFECTIVE_WRITE_RECONCILIATION",
                version=VERSION,
                source_context=lineage._context(env),
                writes=writes,
            )
        trade_class._tariff_bridge_validate_environment(env)
        return writes

    @wraps(original["_write_targets"])
    def write(env, desired, kind):
        ids, physical, _, _, _, _ = dual(env)
        before = dict(zip(ids, physical))
        indexing = getattr(env, "_tariff_funded_index_in_progress", False)
        if not indexing:
            env._trade_war_regime.before_capacity_update(env)
            state = lineage._state(env)
            _, restored, _, _, _ = original["_raw"](env)
            for fid, target in zip(ids, restored):
                row = state["slots"][fid]
                if target != row["unshocked_capacity_target"]:
                    raise budget.BudgetViolation(
                        "Restoring the funding did not recover the quota without the shock"
                    )
                row["current_target"] = row["tariff_effective_target"] = target
            lineage._event(
                state,
                "TARIFF_RESTORE_UNSHOCKED_FOR_BINDING_FUNDING",
                version=VERSION,
                authority=kind,
                source_context=lineage._context(env),
            )
        result = original["_write_targets"](env, desired, kind)
        state = lineage._state(env)
        for fid, quota in zip(ids, desired):
            state["slots"][fid]["unshocked_capacity_target"] = quota
            state["slots"][fid]["tariff_effective_target"] = state["slots"][fid]["current_target"]
        if not indexing:
            env._trade_war_regime.after_capacity_update(env)
            sync_after_write(env, kind, before)
        return result

    @wraps(original["index"])
    def index(env, *args, **kwargs):
        if getattr(env, "_tariff_funded_index_in_progress", False):
            raise budget.BudgetViolation("Nested funded population index")
        env._tariff_funded_index_in_progress = True
        try:
            return original["index"](env, *args, **kwargs)
        finally:
            del env._tariff_funded_index_in_progress

    @wraps(original["commit_entry"])
    def commit(env, firm_id, final_target):
        row = lineage._state(env)["slots"][str(firm_id)]
        row["unshocked_capacity_target"] = budget._int(
            final_target, "native committed funded birth"
        )
        row["tariff_effective_target"] = row["current_target"]
        return original["commit_entry"](env, firm_id, final_target)

    @wraps(original["remaining_slots"])
    def remaining(env, firm_id):
        budget._state(env)
        fid = str(firm_id)
        ids, targets, active, _, filled = original["_raw"](env)
        if fid not in ids:
            raise budget.BudgetViolation("Unknown effective tariff slot: " + fid)
        i = ids.index(fid)
        if not active[i]:
            return 0
        fn = getattr(env, "_covid_reserved_positions", None)
        reserved = budget._int(fn(fid), "COVID reserved positions") if fn else 0
        return max(0, targets[i] - filled[i] - reserved)

    @wraps(original["audit"])
    def audit(env):
        result = original["audit"](env)
        ids, physical, funded, active, firms, filled = dual(env)
        claims = budget._claim_components(env, ids, active, firms, filled)
        suspended = [quota - target for quota, target in zip(funded, physical)]
        return {
            **result,
            "tariff_budget_view": VERSION,
            "funded_quantity_interpretation": "unshocked_native_budget_quota",
            "tariff_effective_total": sum(physical),
            "tariff_suspended_funded_quota": sum(suspended),
            "tariff_funded_effective_residual": sum(funded) - sum(physical) - sum(suspended),
            "effective_target_occupied_overhang": sum(
                (max(0, n - t) for n, t in zip(claims["effective_floors"], physical))
            ),
            "overhang_interpretation": "native_adjustment_timing_recorded_not_new_funded_jobs",
            "effective_hiring_ceilings_active": True,
        }

    budget._raw, budget._write_targets = (raw, write)
    trade_class.after_capacity_update = after_capacity_update
    budget.initialize = initialize
    budget.remaining_slots, budget.audit = (remaining, audit)
    budget.index, budget.commit_entry = (index, commit)
    budget._tariff_budget_view_installed = True
    return {
        "schema": VERSION,
        "native_funded_allocator_and_index_body_unchanged": True,
        "funded_accounting_uses_unshocked_claims": True,
        "hiring_uses_physical_effective_targets": True,
        "physical_tariff_formula_and_timing_changed": False,
        "new_coefficients": False,
        "extra_RNG_or_actor_calls": 0,
    }
