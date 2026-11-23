from __future__ import annotations
from typing import Optional, Dict, Any
from collections import defaultdict

COHORTS = ["15-24", "25-34", "35-44", "45-54", "55-64", "65-99"]


def _age_to_cohort(age: int) -> str:
    if age < 15:
        return "15-24"
    if age < 25:
        return "15-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65-99"


def _age_bucket_key(age: int) -> Any:
    return age if age <= 99 else "100_plus"


def _basis_keys(is_immigrant: bool) -> tuple[str, ...]:
    return ("all_pop", "native_only") if not is_immigrant else ("all_pop",)


class DemographicFlowInstrumenter:

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._counters: Dict[str, Any] = {}
        self._annual_snapshots: list = []
        self._last_year_emitted: Optional[int] = None
        self._stocks_at_year_start: Dict[str, Dict] = {}
        self._reset_year_counters()

    def _reset_year_counters(self) -> None:

        def _zero_basis() -> Dict[str, int]:
            return {"all_pop": 0, "native_only": 0}

        self._counters = {
            "aged_into": {c: _zero_basis() for c in COHORTS},
            "aging_events_by_age": defaultdict(lambda: _zero_basis()),
            "aged_into_65_by_status": defaultdict(lambda: _zero_basis()),
            "aged_into_by_status": {c: defaultdict(lambda: _zero_basis()) for c in COHORTS},
            "retirement_by_pathway": {
                "csv_hazard": _zero_basis(),
                "gate": _zero_basis(),
                "retirement_action": _zero_basis(),
                "separation": _zero_basis(),
            },
            "retirement_by_age": defaultdict(lambda: _zero_basis()),
            "retirement_by_pathway_x_age": defaultdict(lambda: _zero_basis()),
            "prime_age_e_to_n_by_age": defaultdict(lambda: _zero_basis()),
            "nilf_reentry_by_age": defaultdict(lambda: _zero_basis()),
            "nilf_reentry_by_dest_x_age": defaultdict(lambda: _zero_basis()),
            "discouragement_by_age": defaultdict(lambda: _zero_basis()),
            "subtype_assignments": defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: _zero_basis()))
            ),
            "subtype_assignments_unknown_source": _zero_basis(),
            "actual_deaths_by_age": defaultdict(lambda: _zero_basis()),
            "actual_youth_entrants_by_age": defaultdict(lambda: _zero_basis()),
            "actual_immigrants_by_age": defaultdict(lambda: _zero_basis()),
            "actual_emigrants_by_age": defaultdict(lambda: _zero_basis()),
            "theoretical_deaths_by_age": defaultdict(lambda: _zero_basis()),
            "theoretical_emigrants_by_age": defaultdict(lambda: _zero_basis()),
            "theoretical_immigrants_count": 0,
            "theoretical_youth_entrants_count": 0,
            "nilf_reentry_life_event_by_age": defaultdict(
                lambda: defaultdict(lambda: _zero_basis())
            ),
            "nilf_reentry_rl_gated_by_age": defaultdict(lambda: defaultdict(lambda: _zero_basis())),
            "u_to_nilf_structural_by_age": defaultdict(lambda: _zero_basis()),
            "u_to_nilf_rl_discouraged_by_age": defaultdict(lambda: _zero_basis()),
            "u_to_nilf_chronic_65plus_by_age": defaultdict(lambda: _zero_basis()),
            "u_to_nilf_structural_by_age_x_priorsrch": defaultdict(
                lambda: defaultdict(lambda: _zero_basis())
            ),
            "u_to_nilf_rl_discouraged_by_age_x_priorsrch": defaultdict(
                lambda: defaultdict(lambda: _zero_basis())
            ),
            "u_to_nilf_chronic_65plus_by_age_x_priorsrch": defaultdict(
                lambda: defaultdict(lambda: _zero_basis())
            ),
            "e_to_u_no_search_separation_by_age": defaultdict(lambda: _zero_basis()),
            "u_no_search_to_u_search_by_age": defaultdict(lambda: _zero_basis()),
            "u_search_to_u_no_search_by_age": defaultdict(lambda: _zero_basis()),
            "matching_hire_from_non_lf_by_age": defaultdict(
                lambda: defaultdict(lambda: _zero_basis())
            ),
            "actual_replacements_removed_by_age": defaultdict(lambda: _zero_basis()),
            "actual_replacements_added_by_age": defaultdict(lambda: _zero_basis()),
            "actual_replacements_removed_by_age_x_status": defaultdict(
                lambda: defaultdict(lambda: _zero_basis())
            ),
            "actual_replacements_added_by_age_x_status": defaultdict(
                lambda: defaultdict(lambda: _zero_basis())
            ),
            "actual_replacements_by_source": {
                "mortality": 0,
                "emigration": 0,
                "collision": 0,
                "other": 0,
            },
            "demographics_clamped_invocations": 0,
        }

    def record_aging(
        self, prior_age: int, new_age: int, prior_status: str, is_immigrant: bool
    ) -> None:
        if not self.enabled:
            return
        for k in _basis_keys(is_immigrant):
            self._counters["aging_events_by_age"][_age_bucket_key(new_age)][k] += 1
        prior_cohort = _age_to_cohort(prior_age)
        new_cohort = _age_to_cohort(new_age)
        if new_cohort != prior_cohort:
            for k in _basis_keys(is_immigrant):
                self._counters["aged_into"][new_cohort][k] += 1
                self._counters["aged_into_by_status"][new_cohort][prior_status][k] += 1
        if new_age == 65 and prior_age < 65:
            for k in _basis_keys(is_immigrant):
                self._counters["aged_into_65_by_status"][prior_status][k] += 1

    def record_retirement(self, age: int, pathway: str, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        if pathway not in self._counters["retirement_by_pathway"]:
            pathway = "separation"
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["retirement_by_pathway"][pathway][k] += 1
            self._counters["retirement_by_age"][ab][k] += 1
            pkey = f"{pathway}_x_age_{ab}"
            self._counters["retirement_by_pathway_x_age"][pkey][k] += 1

    def record_prime_age_e_to_n(self, age: int, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["prime_age_e_to_n_by_age"][ab][k] += 1

    def record_nilf_reentry_life_event(self, age: int, dest: str, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["nilf_reentry_life_event_by_age"][ab][dest][k] += 1

    def record_nilf_reentry_rl_gated(self, age: int, dest: str, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["nilf_reentry_rl_gated_by_age"][ab][dest][k] += 1

    def record_u_to_nilf_structural(
        self, age: int, is_immigrant: bool, prior_is_searching: bool = True
    ) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        subkey = "from_search" if prior_is_searching else "from_no_search"
        for k in _basis_keys(is_immigrant):
            self._counters["u_to_nilf_structural_by_age"][ab][k] += 1
            self._counters["u_to_nilf_structural_by_age_x_priorsrch"][ab][subkey][k] += 1

    def record_u_to_nilf_rl_discouraged(
        self, age: int, is_immigrant: bool, prior_is_searching: bool = True
    ) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        subkey = "from_search" if prior_is_searching else "from_no_search"
        for k in _basis_keys(is_immigrant):
            self._counters["u_to_nilf_rl_discouraged_by_age"][ab][k] += 1
            self._counters["u_to_nilf_rl_discouraged_by_age_x_priorsrch"][ab][subkey][k] += 1

    def record_u_to_nilf_chronic_65plus(
        self, age: int, is_immigrant: bool, prior_is_searching: bool = True
    ) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        subkey = "from_search" if prior_is_searching else "from_no_search"
        for k in _basis_keys(is_immigrant):
            self._counters["u_to_nilf_chronic_65plus_by_age"][ab][k] += 1
            self._counters["u_to_nilf_chronic_65plus_by_age_x_priorsrch"][ab][subkey][k] += 1

    def record_e_to_u_no_search_separation(self, age: int, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["e_to_u_no_search_separation_by_age"][ab][k] += 1

    def record_u_no_search_to_u_search(self, age: int, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["u_no_search_to_u_search_by_age"][ab][k] += 1

    def record_u_search_to_u_no_search(self, age: int, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["u_search_to_u_no_search_by_age"][ab][k] += 1

    def record_matching_hire_from_non_lf(self, age: int, is_immigrant: bool, source: str) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["matching_hire_from_non_lf_by_age"][ab][source][k] += 1

    def record_theoretical_death(self, age: int, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["theoretical_deaths_by_age"][ab][k] += 1

    def record_theoretical_emigrant(self, age: int, is_immigrant: bool) -> None:
        if not self.enabled:
            return
        ab = _age_bucket_key(age)
        for k in _basis_keys(is_immigrant):
            self._counters["theoretical_emigrants_by_age"][ab][k] += 1

    def record_theoretical_immigrant(self, n: int) -> None:
        if not self.enabled:
            return
        self._counters["theoretical_immigrants_count"] += int(n)

    def record_theoretical_youth_entrant(self, n: int) -> None:
        if not self.enabled:
            return
        self._counters["theoretical_youth_entrants_count"] += int(n)

    def record_actual_worker_replacement(
        self,
        old_age: int,
        old_is_immigrant: bool,
        new_age: int,
        new_is_immigrant: bool,
        source: str,
        old_status: str = "unknown",
        new_status: str = "unknown",
    ) -> None:
        if not self.enabled:
            return
        old_ab = _age_bucket_key(old_age)
        new_ab = _age_bucket_key(new_age)
        for k in _basis_keys(old_is_immigrant):
            self._counters["actual_replacements_removed_by_age"][old_ab][k] += 1
            self._counters["actual_replacements_removed_by_age_x_status"][old_ab][old_status][
                k
            ] += 1
        for k in _basis_keys(new_is_immigrant):
            self._counters["actual_replacements_added_by_age"][new_ab][k] += 1
            self._counters["actual_replacements_added_by_age_x_status"][new_ab][new_status][k] += 1
        if source in self._counters["actual_replacements_by_source"]:
            self._counters["actual_replacements_by_source"][source] += 1
        else:
            self._counters["actual_replacements_by_source"]["other"] += 1

    def record_demographics_clamped_call(self) -> None:
        if not self.enabled:
            return
        self._counters["demographics_clamped_invocations"] += 1

    def record_stocks_snapshot(self, year: int, stocks: Dict[str, Dict]) -> None:
        if not self.enabled:
            return
        self._stocks_at_year_start[year] = stocks

    def record_worker_lf_event(
        self,
        worker_id: Optional[str] = None,
        month: Optional[int] = None,
        year: Optional[int] = None,
        channel_tag: Optional[str] = None,
        lf_delta: int = 0,
        age: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return

    def emit_year_snapshot(self, year: int, end_of_year_stocks: Optional[Dict] = None) -> Dict:
        if not self.enabled:
            return {}
        snap = {
            "year": year,
            "aged_into": self._counters["aged_into"],
            "aging_events_by_age": dict(self._counters["aging_events_by_age"]),
            "aged_into_65_by_status": dict(self._counters["aged_into_65_by_status"]),
            "aged_into_by_status": {
                c: dict(self._counters["aged_into_by_status"][c]) for c in COHORTS
            },
            "retirement_by_pathway": self._counters["retirement_by_pathway"],
            "retirement_by_age": dict(self._counters["retirement_by_age"]),
            "retirement_by_pathway_x_age": dict(self._counters["retirement_by_pathway_x_age"]),
            "prime_age_e_to_n_by_age": dict(self._counters["prime_age_e_to_n_by_age"]),
            "nilf_reentry_by_age": dict(self._counters["nilf_reentry_by_age"]),
            "nilf_reentry_by_dest_x_age": dict(self._counters["nilf_reentry_by_dest_x_age"]),
            "discouragement_by_age": dict(self._counters["discouragement_by_age"]),
            "nilf_reentry_life_event_by_age": _convert_nested_defaultdict(
                self._counters["nilf_reentry_life_event_by_age"]
            ),
            "nilf_reentry_rl_gated_by_age": _convert_nested_defaultdict(
                self._counters["nilf_reentry_rl_gated_by_age"]
            ),
            "u_to_nilf_structural_by_age": dict(self._counters["u_to_nilf_structural_by_age"]),
            "u_to_nilf_rl_discouraged_by_age": dict(
                self._counters["u_to_nilf_rl_discouraged_by_age"]
            ),
            "u_to_nilf_chronic_65plus_by_age": dict(
                self._counters["u_to_nilf_chronic_65plus_by_age"]
            ),
            "u_to_nilf_structural_by_age_x_priorsrch": {
                k: dict(v)
                for k, v in self._counters["u_to_nilf_structural_by_age_x_priorsrch"].items()
            },
            "u_to_nilf_rl_discouraged_by_age_x_priorsrch": {
                k: dict(v)
                for k, v in self._counters["u_to_nilf_rl_discouraged_by_age_x_priorsrch"].items()
            },
            "u_to_nilf_chronic_65plus_by_age_x_priorsrch": {
                k: dict(v)
                for k, v in self._counters["u_to_nilf_chronic_65plus_by_age_x_priorsrch"].items()
            },
            "e_to_u_no_search_separation_by_age": dict(
                self._counters["e_to_u_no_search_separation_by_age"]
            ),
            "u_no_search_to_u_search_by_age": dict(
                self._counters["u_no_search_to_u_search_by_age"]
            ),
            "u_search_to_u_no_search_by_age": dict(
                self._counters["u_search_to_u_no_search_by_age"]
            ),
            "matching_hire_from_non_lf_by_age": _convert_nested_defaultdict(
                self._counters["matching_hire_from_non_lf_by_age"]
            ),
            "subtype_assignments": _convert_nested_defaultdict(
                self._counters["subtype_assignments"]
            ),
            "subtype_assignments_unknown_source": self._counters[
                "subtype_assignments_unknown_source"
            ],
            "theoretical_deaths_by_age": dict(self._counters["theoretical_deaths_by_age"]),
            "theoretical_emigrants_by_age": dict(self._counters["theoretical_emigrants_by_age"]),
            "theoretical_immigrants_count": self._counters["theoretical_immigrants_count"],
            "theoretical_youth_entrants_count": self._counters["theoretical_youth_entrants_count"],
            "actual_replacements_removed_by_age": dict(
                self._counters["actual_replacements_removed_by_age"]
            ),
            "actual_replacements_added_by_age": dict(
                self._counters["actual_replacements_added_by_age"]
            ),
            "actual_replacements_removed_by_age_x_status": _convert_nested_defaultdict(
                self._counters["actual_replacements_removed_by_age_x_status"]
            ),
            "actual_replacements_added_by_age_x_status": _convert_nested_defaultdict(
                self._counters["actual_replacements_added_by_age_x_status"]
            ),
            "actual_replacements_by_source": self._counters["actual_replacements_by_source"],
            "actual_deaths_by_age": dict(self._counters["actual_deaths_by_age"]),
            "actual_youth_entrants_by_age": dict(self._counters["actual_youth_entrants_by_age"]),
            "actual_immigrants_by_age": dict(self._counters["actual_immigrants_by_age"]),
            "actual_emigrants_by_age": dict(self._counters["actual_emigrants_by_age"]),
            "demographics_clamped_invocations": self._counters["demographics_clamped_invocations"],
            "stocks_start": self._stocks_at_year_start.get(year),
            "stocks_end": end_of_year_stocks,
        }
        snap["stock_flow_residuals"] = self._compute_residuals(snap)
        snap["lf_flow_residuals"] = self._compute_lf_residuals(snap)
        if hasattr(self, "_fix2e_apply_matches_log"):
            _log = self._fix2e_apply_matches_log
            snap["fix2e_apply_matches_log"] = [e for e in _log if e.get("year") == year]
            self._fix2e_apply_matches_log = [e for e in _log if e.get("year") != year]
        if hasattr(self, "_per_worker_lf_events"):
            _elog = self._per_worker_lf_events
            snap["fix2e_per_worker_lf_events"] = [e for e in _elog if e.get("year") == year]
            self._per_worker_lf_events = [e for e in _elog if e.get("year") != year]
        self._annual_snapshots.append(snap)
        self._last_year_emitted = year
        self._reset_year_counters()
        return snap

    def _compute_lf_residuals(self, snap: Dict) -> Dict:
        if not snap.get("stocks_start") or not snap.get("stocks_end"):
            return None
        residuals: Dict[str, Dict[str, float]] = {}
        LF_STATUSES = ("E", "U_search")
        for c in COHORTS:
            for basis in ("all_pop", "native_only"):
                lf_end = sum(
                    (snap["stocks_end"].get(c, {}).get(s, {}).get(basis, 0) for s in LF_STATUSES)
                )
                lf_start = sum(
                    (snap["stocks_start"].get(c, {}).get(s, {}).get(basis, 0) for s in LF_STATUSES)
                )
                actual = lf_end - lf_start
                aged_in_LF = sum(
                    (
                        snap.get("aged_into_by_status", {}).get(c, {}).get(s, {}).get(basis, 0)
                        for s in LF_STATUSES
                    )
                )
                added_LF = 0
                for age_key, status_dict in snap.get(
                    "actual_replacements_added_by_age_x_status", {}
                ).items():
                    if _cohort_of_age_key(age_key) != c:
                        continue
                    for s in LF_STATUSES:
                        added_LF += status_dict.get(s, {}).get(basis, 0)
                reentry_C = 0
                for age_key, dest_dict in snap.get("nilf_reentry_life_event_by_age", {}).items():
                    if _cohort_of_age_key(age_key) != c:
                        continue
                    for s in LF_STATUSES:
                        reentry_C += dest_dict.get(s, {}).get(basis, 0)
                reentry_D = 0
                for age_key, dest_dict in snap.get("nilf_reentry_rl_gated_by_age", {}).items():
                    if _cohort_of_age_key(age_key) != c:
                        continue
                    for s in LF_STATUSES:
                        reentry_D += dest_dict.get(s, {}).get(basis, 0)
                next_idx = COHORTS.index(c) + 1 if c != "65-99" else None
                if next_idx is not None and next_idx < len(COHORTS):
                    next_c = COHORTS[next_idx]
                    aged_out_LF = sum(
                        (
                            snap.get("aged_into_by_status", {})
                            .get(next_c, {})
                            .get(s, {})
                            .get(basis, 0)
                            for s in LF_STATUSES
                        )
                    )
                else:
                    aged_out_LF = 0
                removed_LF = 0
                for age_key, status_dict in snap.get(
                    "actual_replacements_removed_by_age_x_status", {}
                ).items():
                    if _cohort_of_age_key(age_key) != c:
                        continue
                    for s in LF_STATUSES:
                        removed_LF += status_dict.get(s, {}).get(basis, 0)
                retire_c = sum(
                    (
                        v.get(basis, 0)
                        for age_key, v in snap.get("retirement_by_age", {}).items()
                        if _cohort_of_age_key(age_key) == c
                    )
                )
                prime_e2n_c = sum(
                    (
                        v.get(basis, 0)
                        for age_key, v in snap.get("prime_age_e_to_n_by_age", {}).items()
                        if _cohort_of_age_key(age_key) == c
                    )
                )

                def _u2n_from_search(counter_key: str) -> int:
                    x = snap.get(counter_key + "_x_priorsrch", {})
                    if not x:
                        return sum(
                            (
                                v.get(basis, 0)
                                for age_key, v in snap.get(counter_key, {}).items()
                                if _cohort_of_age_key(age_key) == c
                            )
                        )
                    return sum(
                        (
                            subd.get("from_search", {}).get(basis, 0)
                            for age_key, subd in x.items()
                            if _cohort_of_age_key(age_key) == c
                        )
                    )

                u2n_E = _u2n_from_search("u_to_nilf_structural_by_age")
                u2n_F = _u2n_from_search("u_to_nilf_rl_discouraged_by_age")
                u2n_G = _u2n_from_search("u_to_nilf_chronic_65plus_by_age")
                ch_H = sum(
                    (
                        v.get(basis, 0)
                        for age_key, v in snap.get("e_to_u_no_search_separation_by_age", {}).items()
                        if _cohort_of_age_key(age_key) == c
                    )
                )
                ch_I_in = sum(
                    (
                        v.get(basis, 0)
                        for age_key, v in snap.get("u_no_search_to_u_search_by_age", {}).items()
                        if _cohort_of_age_key(age_key) == c
                    )
                )
                ch_I_out = sum(
                    (
                        v.get(basis, 0)
                        for age_key, v in snap.get("u_search_to_u_no_search_by_age", {}).items()
                        if _cohort_of_age_key(age_key) == c
                    )
                )
                ch_J = 0
                for age_key, src_dict in snap.get("matching_hire_from_non_lf_by_age", {}).items():
                    if _cohort_of_age_key(age_key) != c:
                        continue
                    for src in ("U_no_search", "NILF"):
                        ch_J += src_dict.get(src, {}).get(basis, 0)
                pred = (
                    aged_in_LF
                    + added_LF
                    + reentry_C
                    + reentry_D
                    + ch_I_in
                    + ch_J
                    - aged_out_LF
                    - removed_LF
                    - retire_c
                    - prime_e2n_c
                    - u2n_E
                    - u2n_F
                    - u2n_G
                    - ch_H
                    - ch_I_out
                )
                residuals.setdefault(c, {})[basis] = float(actual - pred)
        return residuals

    def _compute_residuals(self, snap: Dict) -> Dict:
        if not snap.get("stocks_start") or not snap.get("stocks_end"):
            return None
        residuals: Dict[str, Dict[str, float]] = {}
        for c in COHORTS:
            for basis in ("all_pop", "native_only"):
                aged_in = snap["aged_into"].get(c, {}).get(basis, 0)
                next_idx = COHORTS.index(c) + 1 if c != "65-99" else None
                aged_out = (
                    snap["aged_into"].get(COHORTS[next_idx], {}).get(basis, 0)
                    if next_idx is not None and next_idx < len(COHORTS)
                    else 0
                )
                added_c = sum(
                    (
                        v.get(basis, 0)
                        for age_key, v in snap.get("actual_replacements_added_by_age", {}).items()
                        if _cohort_of_age_key(age_key) == c
                    )
                )
                removed_c = sum(
                    (
                        v.get(basis, 0)
                        for age_key, v in snap.get("actual_replacements_removed_by_age", {}).items()
                        if _cohort_of_age_key(age_key) == c
                    )
                )
                yth_c = (
                    sum(
                        (
                            v.get(basis, 0)
                            for age_key, v in snap.get("actual_youth_entrants_by_age", {}).items()
                            if _cohort_of_age_key(age_key) == c
                        )
                    )
                    if c == "15-24"
                    else 0
                )
                pred = aged_in - aged_out + added_c - removed_c + yth_c
                actual = snap["stocks_end"].get(c, {}).get("total", {}).get(basis, 0) - snap[
                    "stocks_start"
                ].get(c, {}).get("total", {}).get(basis, 0)
                residuals.setdefault(c, {})[basis] = float(actual - pred)
        return residuals

    @property
    def annual_snapshots(self) -> list:
        return list(self._annual_snapshots)


def _convert_nested_defaultdict(d):
    if isinstance(d, defaultdict) or isinstance(d, dict):
        return {k: _convert_nested_defaultdict(v) for k, v in d.items()}
    return d


def _cohort_of_age_key(age_key) -> str:
    if age_key == "100_plus":
        return "65-99"
    return _age_to_cohort(int(age_key))
