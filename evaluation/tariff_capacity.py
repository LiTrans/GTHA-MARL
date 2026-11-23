from __future__ import annotations
from dataclasses import replace
from functools import wraps

VERSION = "tariff_capacity_overlay_authority_v1"


def install(trade_class, environment_class, age_renewal, lineage, allowed_trade_classes=()):
    if getattr(trade_class, "_market_tariff_bridge_installed", False):
        raise ValueError("The tariff capacity link is already installed")
    if not getattr(age_renewal, "_capacity_lineage_installed", False):
        raise ValueError("Install the tariff link after the capacity ledger")
    allowed_types = {trade_class, *allowed_trade_classes}
    for candidate in allowed_types - {trade_class}:
        if not issubclass(candidate, trade_class) or set(candidate.__dict__) - {
            "__module__",
            "__doc__",
            "__firstlineno__",
            "__static_attributes__",
            "resolve_exposure",
        }:
            raise ValueError("Only the exposure-validating regime class may share tariff changes")
    original = {
        "reset": trade_class.reset,
        "apply": trade_class.apply_month,
        "capture": trade_class._capture_live_baseline,
        "shock": trade_class._shock_for,
        "after": trade_class.after_capacity_update,
        "index": age_renewal.index_firm_capacity,
        "initialize": lineage.initialize_lineages,
        "commit": lineage.commit_entry,
        "tag_index": lineage._tag_index,
        "exits": environment_class._process_firm_exits,
    }

    def regime_for(env):
        regime = getattr(env, "_trade_war_regime", None)
        if regime is None or not regime.enabled:
            return None
        if type(regime) not in allowed_types:
            raise lineage.LineageViolation("Unexpected trade regime class in the tariff link")
        return regime

    def state_for(env):
        state = lineage._state(env)
        if state.get("tariff_overlay_authority") != VERSION:
            raise lineage.LineageViolation("The tariff overlay has no record from reset")
        return state

    def publish(env):
        lineage._publish(env, primary=bool(getattr(env, "_person_trace_active", False)))

    def validate_dual(env):
        state, targets, active, firms = lineage._check_current(env)
        if state.get("tariff_overlay_authority") != VERSION:
            raise lineage.LineageViolation("The tariff overlay has no target record")
        regime = regime_for(env)
        for fid, record in state["slots"].items():
            claim = lineage._int(record.get("unshocked_capacity_target"), "unshocked claim")
            effective = lineage._int(record.get("tariff_effective_target"), "effective target")
            if effective != targets[fid]:
                raise lineage.LineageViolation(
                    "Effective tariff target differs from physical stocks"
                )
            if not active[fid] and claim != effective:
                raise lineage.LineageViolation(
                    "Inactive predecessor retains an effective tariff overlay"
                )
            if claim == effective:
                continue
            if regime is None or not active[fid]:
                raise lineage.LineageViolation("An effective target has no active tariff record")
            shock = regime._current_shocks.get(fid)
            if (
                shock is None
                or shock.baseline_target != claim
                or regime._baseline_firm_objects.get(fid) != id(firms[fid].firm)
                or (regime._baseline_targets.get(fid) != claim)
                or (
                    effective
                    != (
                        0 if claim == 0 else max(1, int(round(claim * shock.employment_multiplier)))
                    )
                )
            ):
                raise lineage.LineageViolation(
                    "The effective target differs from the tariff overlay"
                )
        return (state, targets, active, firms)

    def initialize(env):
        result = original["initialize"](env)
        if result is None:
            return result
        state = lineage._state(env)
        state["tariff_overlay_authority"] = VERSION
        for record in state["slots"].values():
            record["unshocked_capacity_target"] = record["current_target"]
            record["tariff_effective_target"] = record["current_target"]
        publish(env)
        return result

    def capture(self, firm_id, firm):
        target = lineage._int(getattr(firm, "target_size"), "native unshocked target")
        if target == 0:
            self._baseline_targets[firm_id] = 0
            self._baseline_firm_objects[firm_id] = id(firm)
            return 0
        return original["capture"](self, firm_id, firm)

    def shock_for(self, firm_id, firm, year, month, baseline):
        shock, active = original["shock"](self, firm_id, firm, year, month, baseline)
        return (replace(shock, shocked_target=0), active) if baseline == 0 else (shock, active)

    def native_after(self, env):
        if not self.enabled:
            return
        for firm_id, firm_state in env.state_manager.firms.items():
            key = str(firm_id)
            same_object = self._baseline_firm_objects.get(key) == id(firm_state.firm)
            baseline = self._capture_live_baseline(key, firm_state.firm)
            shock = self._current_shocks.get(key)
            if not same_object and self._is_active_firm(env, firm_id):
                self._tariff_bridge_pending_first_shock.add(key)
            if (
                same_object
                and shock is not None
                and self._is_active_firm(env, firm_id)
                and (shock.employment_multiplier != 1.0)
            ):
                target = (
                    0
                    if baseline == 0
                    else max(1, int(round(baseline * shock.employment_multiplier)))
                )
                firm_state.firm.target_size = target
                self._set_target_tensors(env, firm_id, key, target)
                self._current_shocks[key] = replace(
                    shock, baseline_target=baseline, shocked_target=target
                )

    def adopt(env, before, source_kind, *, restored=False):
        state = state_for(env)
        _, after, active, firms = lineage._stocks(env)
        regime = regime_for(env)
        writes = []
        for fid, target in after.items():
            record = state["slots"][fid]
            previous_claim = record["unshocked_capacity_target"]
            shock = regime._current_shocks.get(fid) if regime is not None else None
            plain_claim = restored
            if restored:
                claim = previous_claim
                expected = claim
                plain_claim = True
            elif (
                active[fid]
                and regime is not None
                and (fid in getattr(regime, "_tariff_bridge_pending_first_shock", set()))
            ):
                claim = previous_claim
                expected = claim
                plain_claim = True
            elif active[fid] and regime is not None:
                if (
                    regime._baseline_firm_objects.get(fid) != id(firms[fid].firm)
                    or fid not in regime._baseline_targets
                ):
                    raise lineage.LineageViolation(
                        "The tariff overlay lacks the baseline of the current firm"
                    )
                claim = lineage._int(regime._baseline_targets[fid], "native tariff baseline")
                if shock is None or shock.baseline_target != claim:
                    raise lineage.LineageViolation("Tariff baseline and shock disagree")
                expected = (
                    0 if claim == 0 else max(1, int(round(claim * shock.employment_multiplier)))
                )
            else:
                claim = previous_claim
                expected = before[fid]
            if target != expected:
                raise lineage.LineageViolation(
                    "Unexplained effective target write during " + source_kind
                )
            if before[fid] != target or previous_claim != claim:
                writes.append(
                    {
                        "slot_id": fid,
                        "capacity_lineage_id": record["capacity_lineage_id"],
                        "business_generation": record["business_generation"],
                        "effective_target_before": before[fid],
                        "effective_target_after": target,
                        "unshocked_claim_before": previous_claim,
                        "unshocked_claim_after": claim,
                        "employment_multiplier": (
                            None if plain_claim or shock is None else shock.employment_multiplier
                        ),
                        "authority": (
                            "trade_shock_restore_or_pending" if plain_claim else "trade_shock"
                        ),
                        "target_formula": (
                            "claim"
                            if plain_claim
                            else "zero_if_claim_zero_else_max1_round_claim_times_native_multiplier"
                        ),
                        "target_residual": target - expected,
                    }
                )
            record["current_target"] = target
            record["tariff_effective_target"] = target
            record["unshocked_capacity_target"] = claim
        if writes:
            lineage._event(
                state,
                "TRADE_OVERLAY_TARGET_AUTHORITY",
                source_kind=source_kind,
                source_context=lineage._context(env),
                writes=writes,
                capacity_index_or_entry_authority=False,
            )
        lineage._check_current(env)
        publish(env)
        return writes

    @wraps(original["reset"])
    def reset(self, env):
        env._tariff_bridge_reset_in_progress = True
        self._tariff_bridge_pending_first_shock = set()
        try:
            return original["reset"](self, env)
        finally:
            del env._tariff_bridge_reset_in_progress

    @wraps(original["apply"])
    def apply(self, env):
        if (
            not self.enabled
            or getattr(env, "_tariff_bridge_reset_in_progress", False)
            or (not hasattr(env, "_capacity_lineage_state"))
        ):
            return original["apply"](self, env)
        _, before, _, _ = validate_dual(env)
        previous_key = self._current_key
        sm = env.state_manager
        had_instance_method = "remove_vacancies" in vars(sm)
        previous_remove = sm.remove_vacancies

        def tagged_remove(ids, reason="unattributed_removal"):
            if reason != "unattributed_removal":
                raise lineage.LineageViolation(
                    "Unexpected additional vacancy removal while the tariff is applied"
                )
            return previous_remove(ids, reason="trade_policy_withdrawal")

        sm.remove_vacancies = tagged_remove
        try:
            result = original["apply"](self, env)
        finally:
            if had_instance_method:
                sm.remove_vacancies = previous_remove
            else:
                del sm.remove_vacancies
        if self._current_key != previous_key:
            self._tariff_bridge_pending_first_shock.clear()
        adopt(env, before, "NATIVE_TRADE_APPLY_MONTH")
        validate_dual(env)
        return result

    @wraps(original["tag_index"])
    def tag_index(env, *args, **kwargs):
        result = original["tag_index"](env, *args, **kwargs)
        state = state_for(env)
        for record in state["slots"].values():
            record["unshocked_capacity_target"] = record["current_target"]
            record["tariff_effective_target"] = record["current_target"]
        publish(env)
        return result

    @wraps(original["index"])
    def index(env, *args, **kwargs):
        regime = regime_for(env)
        if regime is None:
            return original["index"](env, *args, **kwargs)
        _, before, _, _ = validate_dual(env)
        regime.before_capacity_update(env)
        adopt(env, before, "RESTORE_UNSHOCKED_CLAIMS_BEFORE_REGIONAL_INDEX", restored=True)
        result = original["index"](env, *args, **kwargs)
        _, before_reapply, _, _ = lineage._stocks(env)
        regime.after_capacity_update(env)
        adopt(env, before_reapply, "REAPPLY_NATIVE_TRADE_AFTER_REGIONAL_INDEX")
        validate_dual(env)
        return result

    @wraps(original["exits"])
    def exits(env, *args, **kwargs):
        regime = regime_for(env)
        if regime is None:
            return original["exits"](env, *args, **kwargs)
        state, _, active_before, _ = validate_dual(env)
        result = original["exits"](env, *args, **kwargs)
        _, before_restore, active_after, firms = lineage._stocks(env)
        writes = []
        for fid, was_active in active_before.items():
            if not was_active or active_after[fid]:
                continue
            regime._tariff_bridge_pending_first_shock.discard(fid)
            record = state["slots"][fid]
            claim = record["unshocked_capacity_target"]
            if before_restore[fid] == claim:
                continue
            firms[fid].firm.target_size = claim
            regime._set_target_tensors(env, fid, fid, claim)
            record["current_target"] = claim
            record["tariff_effective_target"] = claim
            writes.append(
                {
                    "slot_id": fid,
                    "capacity_lineage_id": record["capacity_lineage_id"],
                    "business_generation": record["business_generation"],
                    "effective_target_before": before_restore[fid],
                    "effective_target_after": claim,
                    "unshocked_claim_before": claim,
                    "unshocked_claim_after": claim,
                    "employment_multiplier": None,
                    "authority": "trade_shock_restore_on_firm_exit",
                    "target_formula": "claim",
                    "target_residual": 0,
                }
            )
        if writes:
            lineage._event(
                state,
                "TRADE_OVERLAY_TARGET_AUTHORITY",
                source_kind="REMOVE_OVERLAY_FROM_EXITED_PREDECESSOR",
                source_context=lineage._context(env),
                writes=writes,
                capacity_index_or_entry_authority=False,
            )
        validate_dual(env)
        publish(env)
        return result

    @wraps(original["commit"])
    def commit(env, firm_id, *args, **kwargs):
        result = original["commit"](env, firm_id, *args, **kwargs)
        record = state_for(env)["slots"][str(firm_id)]
        record["unshocked_capacity_target"] = record["current_target"]
        record["tariff_effective_target"] = record["current_target"]
        regime = regime_for(env)
        if regime is not None:
            regime._tariff_bridge_pending_first_shock.add(str(firm_id))
        validate_dual(env)
        publish(env)
        return result

    trade_class._capture_live_baseline = capture
    trade_class._shock_for = shock_for
    trade_class.after_capacity_update = native_after
    trade_class.apply_month = apply
    trade_class.reset = reset
    environment_class._process_firm_exits = exits
    age_renewal.index_firm_capacity = index
    lineage.initialize_lineages = initialize
    lineage._tag_index = tag_index
    lineage.commit_entry = commit
    trade_class._market_tariff_bridge_installed = True
    trade_class._tariff_bridge_validate_environment = staticmethod(validate_dual)
    return {
        "schema": "tariff_capacity_bridge_install_receipt_v1",
        "version": VERSION,
        "native_positive_tariff_formula_retained": True,
        "known_zero_claim_stays_zero": True,
        "unshocked_regional_reference_preserved": True,
        "entry_rng_and_draw_law_changed": False,
        "inactive_predecessor_semantics": "unshocked_capacity_claim",
        "allowed_native_trade_types": sorted((cls.__name__ for cls in allowed_types)),
        "vacancy_removal_reason": "trade_policy_withdrawal",
        "no_model_steps": True,
        "old_BR_scaling_applied": False,
        "runtime_validation_method": "TradeWarRegime._tariff_bridge_validate_environment(env)",
    }
