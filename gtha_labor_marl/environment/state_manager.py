import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import logging
from .matching_mechanism_kround import Worker, Vacancy, Firm, MatchResult, INDUSTRY_WAGE_CEILING


def _naics_resampling_seed(worker_id: str) -> int:
    return int(hashlib.sha256(str(worker_id).encode()).hexdigest()[:8], 16) % 2**31


def grant_ei_if_eligible(worker_state, ei_cfg) -> bool:
    tenure = worker_state.previous_tenure_months
    if tenure >= ei_cfg.ei_min_tenure_months:
        worker_state.ei_eligible = True
        worker_state.ei_weeks_remaining = min(
            ei_cfg.ei_max_weeks, ei_cfg.ei_min_weeks + tenure // 2
        )
        return True
    else:
        worker_state.ei_eligible = False
        worker_state.ei_weeks_remaining = 0
        return False


from .assign_archetypes import assign_worker_archetype, WORKER_ARCHETYPES
from .nilf_subtype import (
    NILFSubtypeSampler,
    normalize_sex_value,
    SUBTYPE_NONE,
    SUBTYPE_RETIREMENT,
    SUBTYPE_SCHOOL,
    SUBTYPE_ILLNESS,
    SUBTYPE_FAMILY,
    SUBTYPE_DISCOURAGED,
)

try:
    from ..data.industry_crosswalk import VALID_NAICS_18_CODES
except ImportError:
    from data.industry_crosswalk import VALID_NAICS_18_CODES
logger = logging.getLogger(__name__)
VALID_INDUSTRIES = set(VALID_NAICS_18_CODES)
LWF_RETENTION = {
    ("high_seniority", "M"): (0.65, 0.82),
    ("high_seniority", "F"): (0.65, 0.76),
    ("average", "M"): (0.78, 0.91),
    ("average", "F"): (0.69, 0.88),
}
HIGH_SENIORITY_THRESHOLD_MONTHS = 60


def lwf_draw_retention(previous_tenure_months: int, sex: int) -> float:
    tier = (
        "high_seniority" if previous_tenure_months >= HIGH_SENIORITY_THRESHOLD_MONTHS else "average"
    )
    sex_label = "F" if sex == 2 else "M"
    lo, hi = LWF_RETENTION[tier, sex_label]
    return float(np.random.uniform(lo, hi))


LWF_BACKFILL_MIX_25PLUS = {"job_loser": 0.341, "job_leaver": 0.09, "re_entrant": 0.569}
LWF_BACKFILL_MIX_15_24 = {"job_loser": 0.123, "job_leaver": 0.087, "re_entrant": 0.79}
LWF_BACKFILL_MIX = LWF_BACKFILL_MIX_25PLUS


def lwf_backfill_mix_for_age(age: int) -> Dict[str, float]:
    return LWF_BACKFILL_MIX_15_24 if age < 25 else LWF_BACKFILL_MIX_25PLUS


def impute_separation_state(worker_state) -> Optional[str]:
    if worker_state.worker.employment_status != "Unemployed":
        return None
    if worker_state.separation_reason != "unknown":
        return None
    _age = int(getattr(worker_state.worker, "age", 30))
    _mix = lwf_backfill_mix_for_age(_age)
    _r = float(np.random.random())
    _cum = 0.0
    _reason = "re_entrant"
    for _name, _p in _mix.items():
        _cum += _p
        if _r < _cum:
            _reason = _name
            break
    worker_state.separation_reason = _reason
    if _reason == "job_loser":
        _sex = int(getattr(worker_state.worker, "sex", 1) or 1)
        worker_state.lwf_retention_draw = lwf_draw_retention(
            previous_tenure_months=worker_state.previous_tenure_months, sex=_sex
        )
    return _reason


@dataclass
class WorkerState:
    worker: Worker
    history: List[Dict] = field(default_factory=list)
    ei_eligible: bool = False
    ei_weeks_remaining: int = 0
    previous_tenure_months: int = 0
    previous_wage: float = 0.0
    match_quality: float = 0.5
    learned_v0: float = 6.5
    last_search_action: int = 0
    last_search_decision: int = 0
    last_search_channel: int = 0
    last_search_intensity: int = 0
    last_intent_search_decision: int = 0
    last_intent_search_channel: int = 0
    last_intent_search_intensity: int = 0
    months_employed: int = 0
    months_unemployed: int = 0
    months_nilf: int = 0
    total_job_transitions: int = 0
    j2j_this_year: int = 0
    nilf_subtype: int = 0
    nilf_subtype_months: int = 0
    separation_reason: str = "unknown"
    lwf_retention_draw: Optional[float] = None
    covid_state: str = "none"
    covid_attached_employer_id: Optional[str] = None
    covid_months_in_state: int = 0
    covid_pre_shock_wage: float = 0.0
    covid_pre_shock_hours_per_week: float = 0.0
    covid_pre_shock_tenure_months: int = 0
    covid_pre_shock_commute_time_min: Optional[float] = None

    def add_to_history(self, month: int, year: int):
        self.history.append(
            {
                "month": month,
                "year": year,
                "status": self.worker.employment_status,
                "industry": self.worker.industry,
                "wage": self.worker.current_wage,
                "tenure": self.months_employed,
                "is_searching": self.worker.is_searching,
                "covid_state": self.covid_state,
            }
        )
        max_history = 36
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]


@dataclass
class FirmState:
    firm: Firm
    vacancies: Dict[str, Vacancy] = field(default_factory=dict)
    employees: List[str] = field(default_factory=list)
    total_hires: int = 0
    total_separations: int = 0
    total_vacancies_posted: int = 0
    total_vacancies_filled: int = 0
    current_wage_offer: float = 0.0
    wage_premium: float = 1.0
    previous_wage_premium: float = 1.0
    last_vacancy_action: int = 2
    last_wage_strategy: int = 2

    @property
    def n_vacancies(self) -> int:
        return len(self.vacancies)

    @property
    def vacancy_rate(self) -> float:
        total = self.firm.n_employees + self.n_vacancies
        return self.n_vacancies / total if total > 0 else 0


class StateManager:

    def __init__(self, config):
        self.config = config
        self._ei_period_new_separations = 0
        self._ei_period_new_grants = 0
        self._u_to_nilf_discouragement_period_count = 0
        self._u_to_nilf_discouragement_period_eligible = 0
        self._u_to_nilf_discouragement_period_durations = []
        self.workers: Dict[str, WorkerState] = {}
        self.firms: Dict[str, FirmState] = {}
        self.vacancies: Dict[str, Vacancy] = {}
        self._vacancy_ledger_active = False
        self._vacancy_ledger_events = []
        self._vacancy_ledger_opening = 0
        self.current_year: int = config.simulation.start_year
        self.current_month: int = 1
        self.total_months: int = 0
        self.stats = {
            "employed": 0,
            "unemployed": 0,
            "nilf": 0,
            "total_vacancies": 0,
            "matches_this_month": 0,
            "u2e_matches_this_month": 0,
            "separations_this_month": 0,
        }
        self.frozen_theta = None
        self.theta_is_frozen = False
        self.worker_types = {}
        self.worker_type_mean_fields = {}
        self.worker_skills = {}
        self.device = config.training.device
        self.nilf_subtype_sampler: Optional[NILFSubtypeSampler] = None
        self._demographic_flow_instrumenter = None
        self._state_table_ref = None
        self._state_table_worker_idx: Dict[str, int] = {}
        self.nilf_entry_log: list = []

    def initialize_from_data(
        self,
        population_df,
        firms_df,
        initial_vacancies: Optional[List[Vacancy]] = None,
        full_firms_df: Optional[pd.DataFrame] = None,
    ):
        self._full_firms_df = full_firms_df
        from config.default_config import NAICS_TO_SECTOR

        for _, row in population_df.iterrows():
            worker_id = row["worker_id"]
            naics_raw = row.get("naics_18", 0)
            naics_raw = int(naics_raw) if not np.isnan(naics_raw) else 0
            naics = naics_raw
            VALID_INDUSTRIES = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
            if naics not in VALID_INDUSTRIES:
                INDUSTRY_DIST = {
                    5: 0.12,
                    8: 0.11,
                    12: 0.1,
                    16: 0.08,
                    11: 0.08,
                    10: 0.07,
                    14: 0.07,
                    4: 0.06,
                    9: 0.06,
                    13: 0.06,
                    15: 0.05,
                    7: 0.04,
                    17: 0.04,
                    18: 0.03,
                    3: 0.01,
                    1: 0.01,
                    2: 0.01,
                }
                industries = list(INDUSTRY_DIST.keys())
                probs = [p / sum(INDUSTRY_DIST.values()) for p in INDUSTRY_DIST.values()]
                _rng = np.random.default_rng(_naics_resampling_seed(worker_id))
                new_naics = int(_rng.choice(industries, p=probs))
                naics = new_naics
            sector = NAICS_TO_SECTOR.get(naics, 0)
            emp_status = row["employment_status"]
            is_searching = emp_status in ["Unemployed", "U"]
            _arch_id = assign_worker_archetype(row)
            _sex_int = normalize_sex_value(row.get("sex"))
            worker = Worker(
                id=worker_id,
                age=int(row["age"]),
                skill_level=int(row.get("skill_level", 2)),
                education=int(row.get("education", 3)),
                industry=naics,
                sector=sector,
                experience_years=float(row.get("experience_total", 0)),
                cma=int(row.get("cma", 535)),
                da_id=int(row["da_id"]) if pd.notna(row.get("da_id")) else None,
                employment_status=(
                    "Employed"
                    if emp_status in ["Employed", "E"]
                    else "Unemployed" if emp_status in ["Unemployed", "U"] else "NILF"
                ),
                current_wage=min(
                    (
                        float(row.get("hourly_wage", 20.0))
                        if not np.isnan(row.get("hourly_wage", 20.0))
                        else 20.0
                    ),
                    INDUSTRY_WAGE_CEILING.get(naics, 80.0),
                ),
                is_searching=is_searching,
                unemployment_months=(
                    int(row.get("unemployment_duration_months", 0))
                    if not np.isnan(row.get("unemployment_duration_months", 0))
                    else 0
                ),
                hours_per_week=(
                    float(row.get("hours_per_week", 40.0))
                    if not np.isnan(row.get("hours_per_week", 40.0))
                    else 40.0
                ),
                archetype_id=_arch_id,
                archetype_name=WORKER_ARCHETYPES.get(_arch_id, "native_low_skill"),
                sector_id=int(row.get("sector_id", 0)),
                sector_name=str(row.get("sector_name", "services")),
                sex=_sex_int,
            )
            worker_state = WorkerState(
                worker=worker,
                previous_wage=worker.current_wage,
                previous_tenure_months=(
                    int(row.get("tenure_months", 0))
                    if not np.isnan(row.get("tenure_months", 0))
                    else 0
                ),
            )
            if worker.employment_status == "Unemployed":
                grant_ei_if_eligible(worker_state, self.config.ei)
            self.workers[worker_id] = worker_state
        VALID_INDUSTRIES = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
        for _, row in firms_df.iterrows():
            firm_id = row["firm_id"]
            naics = int(row.get("naics_18", 0))
            if naics not in VALID_INDUSTRIES:
                if naics == 6:
                    naics = 5
                elif naics == 0 or naics < 1:
                    naics = 8
                elif naics > 18:
                    naics = 18
            sector = NAICS_TO_SECTOR.get(naics, 0)
            n_emp = row.get("estimated_employees", row.get("n_employees", 10))
            target_size = row.get("target_size", None)
            if target_size is not None:
                target_size = int(target_size)
            firm = Firm(
                id=firm_id,
                industry=naics,
                sector=sector,
                n_employees=int(n_emp),
                cma=int(row.get("cma", 535)),
                da_id=(
                    int(row["da_id"])
                    if pd.notna(row.get("da_id"))
                    else int(row["da"]) if pd.notna(row.get("da")) else None
                ),
                target_size=target_size,
                naics_6digit=(
                    str(row.get("naics_6digit")).removesuffix(".0")
                    if pd.notna(row.get("naics_6digit"))
                    else None
                ),
            )
            self.firms[firm_id] = FirmState(firm=firm)
        if initial_vacancies:
            for vacancy in initial_vacancies:
                self.vacancies[vacancy.id] = vacancy
                if vacancy.firm_id in self.firms:
                    self.firms[vacancy.firm_id].vacancies[vacancy.id] = vacancy
        firms_by_industry_cma = {}
        for firm_id, fs in self.firms.items():
            key = (fs.firm.industry, fs.firm.cma)
            if key not in firms_by_industry_cma:
                firms_by_industry_cma[key] = []
            firms_by_industry_cma[key].append(firm_id)
        firms_by_industry = {}
        for firm_id, fs in self.firms.items():
            ind = fs.firm.industry
            if ind not in firms_by_industry:
                firms_by_industry[ind] = []
            firms_by_industry[ind].append(firm_id)
        MIN_FIRMS_PER_INDUSTRY = 2
        industries_needing_firms = [
            ind
            for ind in VALID_INDUSTRIES
            if ind not in firms_by_industry or len(firms_by_industry[ind]) < MIN_FIRMS_PER_INDUSTRY
        ]
        if industries_needing_firms:
            logger.warning(
                f"Industries with fewer than {MIN_FIRMS_PER_INDUSTRY} firms: {industries_needing_firms}"
            )
            sampled_firm_ids = set(self.firms.keys())
            for ind in industries_needing_firms:
                current_count = len(firms_by_industry.get(ind, []))
                needed = MIN_FIRMS_PER_INDUSTRY - current_count
                sampled_from_pool = False
                if self._full_firms_df is not None and "naics_18" in self._full_firms_df.columns:
                    pool_firms = self._full_firms_df[
                        (self._full_firms_df["naics_18"] == ind)
                        & ~self._full_firms_df["firm_id"].isin(sampled_firm_ids)
                    ]
                    if len(pool_firms) >= needed:
                        additional_firms = pool_firms.sample(
                            n=needed, random_state=self.config.seed + ind
                        )
                        for _, row in additional_firms.iterrows():
                            firm_id = row["firm_id"]
                            new_firm = Firm(
                                id=firm_id,
                                industry=int(row.get("naics_18", ind)),
                                sector=NAICS_TO_SECTOR.get(int(row.get("naics_18", ind)), 0),
                                n_employees=int(row.get("estimated_employees", 10)),
                                cma=int(row.get("cma", 535)),
                                da_id=row.get("da_id", None),
                                target_size=int(row.get("estimated_employees", 10)) + 5,
                                naics_6digit=(
                                    str(row.get("naics_6digit")).removesuffix(".0")
                                    if pd.notna(row.get("naics_6digit"))
                                    else None
                                ),
                            )
                            self.firms[firm_id] = FirmState(firm=new_firm)
                            if ind not in firms_by_industry:
                                firms_by_industry[ind] = []
                            firms_by_industry[ind].append(firm_id)
                            firm_cma = new_firm.cma or 535
                            if (ind, firm_cma) not in firms_by_industry_cma:
                                firms_by_industry_cma[ind, firm_cma] = []
                            firms_by_industry_cma[ind, firm_cma].append(firm_id)
                            sampled_firm_ids.add(firm_id)
                        sampled_from_pool = True
                if not sampled_from_pool:
                    logger.warning(
                        f"  The firm file has no firms for industry {ind}. Synthetic firms are created."
                    )
                    for i in range(needed):
                        synth_firm_id = f"synth_firm_{ind}_{i + 1:03d}"
                        avg_size = max(
                            1,
                            self.config.simulation.n_workers
                            // max(1, self.config.simulation.n_firms),
                        )
                        synth_target = max(3, int(avg_size * 0.8))
                        synth_firm = Firm(
                            id=synth_firm_id,
                            industry=ind,
                            sector=NAICS_TO_SECTOR.get(ind, 0),
                            n_employees=min(synth_target, avg_size),
                            cma=535,
                            da_id=None,
                            target_size=synth_target,
                        )
                        self.firms[synth_firm_id] = FirmState(firm=synth_firm)
                        if ind not in firms_by_industry:
                            firms_by_industry[ind] = []
                        firms_by_industry[ind].append(synth_firm_id)
                        if (ind, 535) not in firms_by_industry_cma:
                            firms_by_industry_cma[ind, 535] = []
                        firms_by_industry_cma[ind, 535].append(synth_firm_id)
        all_firm_ids = list(self.firms.keys())
        assignment_counters = {key: 0 for key in firms_by_industry_cma}
        industry_counters = {ind: 0 for ind in firms_by_industry}
        global_counter = 0
        for worker_id, ws in self.workers.items():
            if ws.worker.employment_status == "Employed":
                key = (ws.worker.industry, ws.worker.cma)
                if key in firms_by_industry_cma and firms_by_industry_cma[key]:
                    matching_firms = firms_by_industry_cma[key]
                    idx = assignment_counters[key] % len(matching_firms)
                    firm_id = matching_firms[idx]
                    assignment_counters[key] += 1
                elif (
                    ws.worker.industry in firms_by_industry
                    and firms_by_industry[ws.worker.industry]
                ):
                    matching_firms = firms_by_industry[ws.worker.industry]
                    idx = industry_counters[ws.worker.industry] % len(matching_firms)
                    firm_id = matching_firms[idx]
                    industry_counters[ws.worker.industry] += 1
                else:
                    firm_id = all_firm_ids[global_counter % len(all_firm_ids)]
                    global_counter += 1
                self.firms[firm_id].employees.append(worker_id)
                ws.worker.current_employer_id = firm_id
                ws.months_employed = getattr(ws, "previous_tenure_months", 0)
        self._update_aggregate_stats()

    def initialize_firm_wages(self, wage_data: Dict[int, float]):
        for firm_id, fs in self.firms.items():
            industry = fs.firm.industry
            base_wage = wage_data.get(industry, 20.0)
            fs.current_wage_offer = base_wage
            fs.wage_premium = 1.0
            fs.previous_wage_premium = 1.0

    def setup_nilf_subtype_sampler(self, nilf_composition_by_sex_df, rng=None) -> None:
        self.nilf_subtype_sampler = NILFSubtypeSampler(nilf_composition_by_sex_df, rng=rng)

    def sample_nilf_subtype_for_worker(self, worker: Worker) -> int:
        if self.nilf_subtype_sampler is None:
            return SUBTYPE_NONE
        return self.nilf_subtype_sampler.sample(
            year=self.current_year, age=worker.age, sex=worker.sex
        )

    def _subtype_window_months(self, subtype: int, age: int = 30) -> int:
        from environment.nilf_subtype import effective_subtype_windows

        school_w, family_w, illness_w = effective_subtype_windows(self.config.mnl, age)
        if subtype == SUBTYPE_SCHOOL:
            return school_w
        if subtype == SUBTYPE_FAMILY:
            return family_w
        if subtype == SUBTYPE_ILLNESS:
            return illness_w
        return 0

    def _audit_log_nilf_entry(
        self, ws: "WorkerState", source_state: Optional[str], subtype: int
    ) -> None:
        if source_state is None:
            return
        try:
            self.nilf_entry_log.append(
                {
                    "year": int(getattr(self, "current_year", 0)),
                    "month": int(getattr(self, "current_month", 0)),
                    "source_state": str(source_state),
                    "subtype": int(subtype),
                    "age": int(getattr(ws.worker, "age", 30)),
                    "sex": int(getattr(ws.worker, "sex", 0)),
                }
            )
        except Exception:
            pass

    def assign_exogenous_nilf_subtype(
        self,
        ws: "WorkerState",
        randomize_initial_months: bool = False,
        source_state: Optional[str] = None,
    ) -> int:
        subtype = self.sample_nilf_subtype_for_worker(ws.worker)
        ws.nilf_subtype = subtype
        if randomize_initial_months and subtype in (
            SUBTYPE_SCHOOL,
            SUBTYPE_FAMILY,
            SUBTYPE_ILLNESS,
        ):
            window = self._subtype_window_months(subtype, int(getattr(ws.worker, "age", 30)))
            ws.nilf_subtype_months = int(np.random.randint(0, max(2 * window, 1)))
        else:
            ws.nilf_subtype_months = 0
        self._audit_log_nilf_entry(ws, source_state, subtype)
        return subtype

    def set_discouraged_nilf_subtype(
        self, ws: "WorkerState", source_state: Optional[str] = None
    ) -> None:
        ws.nilf_subtype = SUBTYPE_DISCOURAGED
        ws.nilf_subtype_months = 0
        self._audit_log_nilf_entry(ws, source_state, SUBTYPE_DISCOURAGED)

    def set_retirement_nilf_subtype(
        self, ws: "WorkerState", source_state: Optional[str] = None
    ) -> None:
        ws.nilf_subtype = SUBTYPE_RETIREMENT
        ws.nilf_subtype_months = 0
        self._audit_log_nilf_entry(ws, source_state, SUBTYPE_RETIREMENT)

    def clear_nilf_subtype(self, ws: "WorkerState") -> None:
        ws.nilf_subtype = SUBTYPE_NONE
        ws.nilf_subtype_months = 0

    def assign_initial_nilf_subtypes(self) -> Dict[int, int]:
        if self.nilf_subtype_sampler is None:
            return {}
        counts: Dict[int, int] = {}
        for ws in self.workers.values():
            if ws.worker.employment_status == "NILF":
                st = self.assign_exogenous_nilf_subtype(ws, randomize_initial_months=True)
                counts[st] = counts.get(st, 0) + 1
        return counts

    def assign_worker_types(self):
        for w_id, ws in self.workers.items():
            self.worker_types[w_id] = ws.worker.archetype_id

    def assign_bounded_rationality_skills(self, skill_distributions: Dict):
        for w_id, type_name in self.worker_types.items():
            mean, std = skill_distributions.get(type_name, (0.5, 0.1))
            skill = np.clip(np.random.normal(mean, std), 0.05, 0.95)
            self.worker_skills[w_id] = float(skill)

    def compute_worker_type_mean_fields(self) -> Dict[int, Dict]:
        type_stats = defaultdict(
            lambda: {
                "wages": [],
                "employed": 0,
                "unemployed": 0,
                "nilf": 0,
                "search_durations": [],
                "tenures": [],
                "total": 0,
            }
        )
        for w_id, ws in self.workers.items():
            t = self.worker_types.get(w_id, 0)
            w = ws.worker
            type_stats[t]["total"] += 1
            if w.employment_status == "Employed":
                type_stats[t]["employed"] += 1
                type_stats[t]["wages"].append(w.current_wage)
                type_stats[t]["tenures"].append(ws.months_employed)
            elif w.employment_status == "Unemployed":
                type_stats[t]["unemployed"] += 1
                type_stats[t]["search_durations"].append(w.unemployment_months)
            else:
                type_stats[t]["nilf"] += 1
        total_workers = len(self.workers)
        mean_fields = {}
        for t, s in type_stats.items():
            n = max(s["total"], 1)
            mean_fields[t] = {
                "employment_rate": s["employed"] / n,
                "avg_wage": np.mean(s["wages"]) if s["wages"] else 0.0,
                "avg_search_duration": (
                    np.mean(s["search_durations"]) if s["search_durations"] else 0.0
                ),
                "avg_tenure": np.mean(s["tenures"]) if s["tenures"] else 0.0,
                "population_share": n / total_workers,
            }
        self.worker_type_mean_fields = mean_fields
        return mean_fields

    def compute_firm_mean_field(self) -> Dict:
        industry_stats = defaultdict(
            lambda: {
                "vacancies": 0,
                "unemployed": 0,
                "employed": 0,
                "wages": [],
                "vacancies_filled": 0,
                "vacancies_total": 0,
            }
        )
        for w_id, ws in self.workers.items():
            w = ws.worker
            ind = w.industry
            if w.employment_status == "Employed":
                industry_stats[ind]["employed"] += 1
                if w_id not in {}:
                    industry_stats[ind]["wages"].append(w.current_wage)
            elif w.employment_status == "Unemployed":
                industry_stats[ind]["unemployed"] += 1
        for v_id, vac in self.vacancies.items():
            ind = vac.industry
            industry_stats[ind]["vacancies"] += 1
        theta = {
            "market_tightness_by_industry": {},
            "avg_wage_by_industry": {},
            "vacancy_fill_rate_by_industry": {},
        }
        for ind, s in industry_stats.items():
            u = max(s["unemployed"], 1)
            theta["market_tightness_by_industry"][ind] = min(s["vacancies"] / u, 3.0)
            theta["avg_wage_by_industry"][ind] = np.mean(s["wages"]) if s["wages"] else 20.0
            theta["vacancy_fill_rate_by_industry"][ind] = self.stats["matches_this_month"] / max(
                self.stats["total_vacancies"], 1
            )
        total = len(self.workers)
        lf = self.stats["employed"] + self.stats["unemployed"]
        theta["aggregate_unemployment_rate"] = self.stats["unemployed"] / max(lf, 1)
        theta["aggregate_employment_rate"] = self.stats["employed"] / max(total, 1)
        theta["aggregate_vacancy_rate"] = self.stats["total_vacancies"] / max(total, 1)
        return theta

    def freeze_theta(self, theta: Dict):
        self.frozen_theta = theta
        self.theta_is_frozen = True

    def unfreeze_theta(self):
        self.theta_is_frozen = False

    def _update_aggregate_stats(self):
        self.stats["employed"] = sum(
            (1 for ws in self.workers.values() if ws.worker.employment_status == "Employed")
        )
        self.stats["unemployed"] = sum(
            (1 for ws in self.workers.values() if ws.worker.employment_status == "Unemployed")
        )
        self.stats["nilf"] = sum(
            (1 for ws in self.workers.values() if ws.worker.employment_status == "NILF")
        )
        self.stats["total_vacancies"] = len(self.vacancies)

    def _compute_cohort_stocks_for_fix2a(self):
        from .demographic_flow_instrumenter import _age_to_cohort, COHORTS

        stocks = {c: {} for c in COHORTS}
        for ws in self.workers.values():
            w = ws.worker
            age = int(w.age)
            cohort = _age_to_cohort(age)
            es = getattr(w, "employment_status", None)
            if es == "Employed":
                status = "E"
            elif es == "Unemployed":
                status = "U_search" if getattr(w, "is_searching", False) else "U_no_search"
            elif es == "NILF":
                status = (
                    "NILF_ineligible" if getattr(ws, "nilf_subtype", 0) == 1 else "NILF_eligible"
                )
            else:
                status = "unknown"
            imm_raw = getattr(w, "immigrant_status", False)
            is_imm = (
                imm_raw.lower() in ("true", "1", "yes", "immigrant")
                if isinstance(imm_raw, str)
                else bool(imm_raw)
            )
            basis_keys = ("all_pop",) if is_imm else ("all_pop", "native_only")
            for tag in (status, "total"):
                cell = stocks[cohort].setdefault(tag, {"all_pop": 0, "native_only": 0})
                for bk in basis_keys:
                    cell[bk] += 1
        return stocks

    def advance_time(self):
        self.current_month += 1
        if self.current_month > 12:
            _prior_year = self.current_year
            _inst = self._demographic_flow_instrumenter
            _instrumented = _inst is not None and getattr(_inst, "enabled", False)
            if _instrumented:
                _end_stocks = self._compute_cohort_stocks_for_fix2a()
                _inst.emit_year_snapshot(year=_prior_year, end_of_year_stocks=_end_stocks)
            self.current_month = 1
            self.current_year += 1
            if _instrumented:
                _start_stocks = self._compute_cohort_stocks_for_fix2a()
                _inst.record_stocks_snapshot(year=self.current_year, stocks=_start_stocks)
        self.total_months += 1
        _flow_inst = self._demographic_flow_instrumenter
        _flow_enabled = _flow_inst is not None and getattr(_flow_inst, "enabled", False)
        _st_ref = getattr(self, "_state_table_ref", None)
        _st_id_to_idx = getattr(_st_ref, "worker_id_to_idx", None) if _st_ref is not None else None
        for worker_id, worker_state in self.workers.items():
            worker_state.add_to_history(self.current_month, self.current_year)
            if self.current_month == 1:
                if _flow_enabled:
                    _prior_age = int(worker_state.worker.age)
                    _w = worker_state.worker
                    _es = getattr(_w, "employment_status", None)
                    if _es == "Employed":
                        _prior_status = "E"
                    elif _es == "Unemployed":
                        _prior_status = (
                            "U_search" if getattr(_w, "is_searching", False) else "U_no_search"
                        )
                    elif _es == "NILF":
                        _prior_status = (
                            "NILF_ineligible"
                            if getattr(worker_state, "nilf_subtype", 0) == 1
                            else "NILF_eligible"
                        )
                    else:
                        _prior_status = "unknown"
                    _imm_raw = getattr(_w, "immigrant_status", False)
                    if isinstance(_imm_raw, str):
                        _is_imm = _imm_raw.lower() in ("true", "1", "yes", "immigrant")
                    else:
                        _is_imm = bool(_imm_raw)
                worker_state.worker.age += 1
                if _st_ref is not None and _st_id_to_idx is not None:
                    _idx = _st_id_to_idx.get(worker_id)
                    if _idx is None:
                        raise AssertionError(
                            f"Worker {worker_id!r} is in the state manager but has no row in the state table when ages advance (month {self.current_month}, year {self.current_year}). A worker was added without updating the state table, or the table was rebuilt without its ID index."
                        )
                    _st_ref.w_age[_idx] = worker_state.worker.age
                if _flow_enabled:
                    _flow_inst.record_aging(
                        prior_age=_prior_age,
                        new_age=int(worker_state.worker.age),
                        prior_status=_prior_status,
                        is_immigrant=_is_imm,
                    )
            if worker_state.worker.employment_status == "Employed":
                worker_state.months_employed += 1
            elif worker_state.worker.employment_status == "Unemployed":
                worker_state.worker.unemployment_months += 1
                worker_state.months_unemployed = worker_state.worker.unemployment_months
                if worker_state.ei_weeks_remaining > 0:
                    worker_state.ei_weeks_remaining = max(
                        0,
                        worker_state.ei_weeks_remaining
                        - self.config.ei.ei_weeks_decrement_per_month,
                    )
            else:
                worker_state.months_nilf += 1
                worker_state.nilf_subtype_months += 1
            if worker_state.covid_state != "none":
                worker_state.covid_months_in_state += 1
        self.process_skill_accumulation()
        self.process_skill_depreciation()
        expired_ids = []
        vc = self.config.vacancy
        industry_survival = vc.vacancy_industry_survival
        base_survival = vc.vacancy_base_survival_rate
        duration_hazard_rate = vc.vacancy_duration_hazard_rate
        min_duration_factor = vc.vacancy_min_duration_factor
        skill_base = vc.vacancy_skill_survival_base
        skill_bonus = vc.vacancy_skill_survival_bonus
        max_age = vc.max_vacancy_duration
        for vac_id, vac in self.vacancies.items():
            vac.months_posted = getattr(vac, "months_posted", 0) + 1
            months_open = vac.months_posted
            if months_open > max_age:
                expired_ids.append(vac_id)
                continue
            industry = getattr(vac, "industry", 8)
            ind_survival = industry_survival.get(industry, base_survival)
            duration_factor = max(min_duration_factor, 1.0 - duration_hazard_rate * months_open)
            skill_level = getattr(vac, "skill_required", 2)
            skill_factor = min(1.0, skill_base + skill_bonus * skill_level)
            survival_prob = ind_survival * duration_factor * skill_factor
            survival_prob = min(0.95, max(0.5, survival_prob))
            if np.random.random() > survival_prob:
                expired_ids.append(vac_id)
        for vac_id in expired_ids:
            vac = self.vacancies.pop(vac_id)
            if vac.firm_id in self.firms:
                self.firms[vac.firm_id].vacancies.pop(vac_id, None)
        n_expired = len(expired_ids)
        self._record_vacancy_event("expiry", -n_expired)
        self.stats["total_vacancies"] = len(self.vacancies)
        self.stats["vacancies_expired_this_month"] = n_expired
        return n_expired

    def capacity_commit_allowed(
        self, worker_id, firm_id, vacancy_id, route, allow_same_employer=False
    ):
        import capacity_budget
        from weakref import ReferenceType

        reference = getattr(self, "_capacity_budget_env", None)
        if not isinstance(reference, ReferenceType):
            raise ValueError("Binding capacity commitment lacks environment weak reference")
        env = reference()
        if env is None or env.state_manager is not self:
            raise ValueError("Binding capacity commitment has a stale environment owner")
        if route not in ("u2e", "j2j", "direct_nilf"):
            raise ValueError("Unknown binding capacity commitment route")
        vacancy = self.vacancies.get(vacancy_id)
        reason = None
        if vacancy is None:
            reason = "missing_vacancy"
        elif getattr(vacancy, "firm_id", None) != firm_id:
            raise ValueError("Binding capacity commitment destination differs from vacancy")
        else:
            ws = self.workers.get(worker_id)
            same_employer = (
                allow_same_employer
                and ws is not None
                and (ws.worker.employment_status == "Employed")
                and (ws.worker.current_employer_id == firm_id)
            )
            if same_employer or capacity_budget.remaining_slots(env, firm_id) > 0:
                return True
            reason = "no_funded_headroom"
        if not hasattr(env, "_capacity_commit_rejections_this_step"):
            env._capacity_commit_rejections_this_step = {"u2e": 0, "j2j": 0, "direct_nilf": 0}
            env._capacity_commit_rejection_events_this_step = []
        env._capacity_commit_rejections_this_step[route] += 1
        env._capacity_commit_rejection_events_this_step.append(
            {
                "route": route,
                "worker_id": str(worker_id),
                "firm_id": str(firm_id),
                "vacancy_id": str(vacancy_id),
                "reason": reason,
            }
        )
        return False

    def apply_matches(self, matches: List[MatchResult]):
        _committed_matches = []
        self.stats["unemployed_pre"] = self.stats.get("unemployed", 0)
        self.stats["employed_pre"] = self.stats.get("employed", 0)
        u2e_count = 0
        for match in matches:
            if isinstance(match, dict):
                worker_id = match["worker_id"]
                vacancy_id = match["vacancy_id"]
                final_wage = match.get(
                    "negotiated_wage", match.get("offered_wage", match.get("final_wage", 20.0))
                )
            else:
                worker_id = match.worker_id
                vacancy_id = match.vacancy_id
                final_wage = match.final_wage
            worker_state = self.workers.get(worker_id)
            if worker_state is None:
                continue
            _capacity_vacancy = self.vacancies.get(vacancy_id)
            _capacity_firm = (
                getattr(_capacity_vacancy, "firm_id", None)
                if _capacity_vacancy is not None
                else (
                    match.get("firm_id")
                    if isinstance(match, dict)
                    else getattr(match, "firm_id", None)
                )
            )
            if not self.capacity_commit_allowed(worker_id, _capacity_firm, vacancy_id, "u2e"):
                continue
            worker = worker_state.worker
            old_status = worker.employment_status
            _j_pre_is_searching = bool(getattr(worker, "is_searching", False))
            if old_status == "Unemployed":
                _j_pre_aligned_status = "U_search" if _j_pre_is_searching else "U_no_search"
            elif old_status == "NILF":
                _j_pre_aligned_status = "NILF"
            elif old_status == "Employed":
                _j_pre_aligned_status = "E"
            else:
                _j_pre_aligned_status = "unknown"
            if old_status == "Unemployed":
                u2e_count += 1
                self.apply_reemployment_boost(worker_id)
            prev_firm_id = getattr(worker, "current_employer_id", None)
            if prev_firm_id and prev_firm_id in self.firms:
                prev_fs = self.firms[prev_firm_id]
                if worker_id in prev_fs.employees:
                    prev_fs.employees.remove(worker_id)
            self.clear_nilf_subtype(worker_state)
            worker_state.separation_reason = "unknown"
            worker_state.lwf_retention_draw = None
            worker.employment_status = "Employed"
            worker.is_searching = False
            if self._demographic_flow_instrumenter is not None and getattr(
                self._demographic_flow_instrumenter, "enabled", False
            ):
                _age_j = int(getattr(worker, "age", -1))
                _imm_raw_j = getattr(worker, "immigrant_status", False)
                _is_imm_j = (
                    _imm_raw_j.lower() in ("true", "1", "yes", "immigrant")
                    if isinstance(_imm_raw_j, str)
                    else bool(_imm_raw_j)
                )
                if _j_pre_aligned_status in ("U_no_search", "NILF"):
                    self._demographic_flow_instrumenter.record_matching_hire_from_non_lf(
                        age=_age_j, is_immigrant=_is_imm_j, source=_j_pre_aligned_status
                    )
                    self._demographic_flow_instrumenter.record_worker_lf_event(
                        worker_id=worker_id,
                        month=self.current_month,
                        year=self.current_year,
                        channel_tag="J",
                        lf_delta=+1,
                        age=_age_j,
                    )
            _v = self.vacancies.get(vacancy_id)
            _hire_industry = _v.industry if _v else getattr(worker, "industry", 8)
            _ceiling = INDUSTRY_WAGE_CEILING.get(_hire_industry, 80.0)
            final_wage = min(final_wage, _ceiling)
            worker.current_wage = final_wage
            worker.unemployment_months = 0
            worker.job_application_attempts = 0
            mq = (
                match.get("match_quality", 0.5)
                if isinstance(match, dict)
                else getattr(match, "match_quality", 0.5)
            )
            worker_state.match_quality = mq
            if isinstance(match, dict):
                worker.commute_time_min = match.get("commute_time_min", None)
            else:
                worker.commute_time_min = getattr(match, "commute_time_min", None)
            vacancy = self.vacancies.get(vacancy_id)
            if vacancy:
                new_industry = vacancy.industry
                if new_industry not in VALID_INDUSTRIES:
                    new_industry = 5 if new_industry == 6 else 8
                    logger.warning(
                        f"Hire with invalid industry {vacancy.industry} recorded under industry {new_industry}"
                    )
                worker.industry = new_industry
                worker.sector = vacancy.sector
            worker_state.ei_eligible = False
            worker_state.ei_weeks_remaining = 0
            worker_state.previous_wage = final_wage
            worker_state.previous_tenure_months = 0
            worker_state.total_job_transitions += 1
            worker_state.months_employed = 0
            if vacancy_id in self.vacancies:
                vacancy = self.vacancies.pop(vacancy_id)
                self._record_vacancy_event("fill_u2e", -1)
                firm_state = self.firms.get(vacancy.firm_id)
                if firm_state:
                    firm_state.vacancies.pop(vacancy.id, None)
                    if worker_id not in firm_state.employees:
                        firm_state.employees.append(worker_id)
                    firm_state.total_hires += 1
                    firm_state.total_vacancies_filled += 1
                    worker.current_employer_id = vacancy.firm_id
            else:
                for fs in self.firms.values():
                    if vacancy_id in fs.vacancies:
                        fs.vacancies.pop(vacancy_id)
                        break
                if vacancy and hasattr(vacancy, "firm_id"):
                    firm_state = self.firms.get(vacancy.firm_id)
                    if firm_state:
                        if worker_id not in firm_state.employees:
                            firm_state.employees.append(worker_id)
                        firm_state.total_hires += 1
                        firm_state.total_vacancies_filled += 1
                        worker.current_employer_id = vacancy.firm_id
            if worker_state.covid_state != "none":
                worker_state.covid_state = "none"
                worker_state.covid_attached_employer_id = None
                worker_state.covid_months_in_state = 0
                worker_state.covid_pre_shock_wage = 0.0
                worker_state.covid_pre_shock_hours_per_week = 0.0
                worker_state.covid_pre_shock_tenure_months = 0
                worker_state.covid_pre_shock_commute_time_min = None
            _committed_matches.append(match)
        self.stats["matches_this_month"] = len(_committed_matches)
        self.stats["u2e_matches_this_month"] = u2e_count
        self._update_aggregate_stats()
        return _committed_matches

    def backfill_reservation_bundle_state(self) -> Dict[str, int]:
        counts = {
            "job_loser": 0,
            "job_leaver": 0,
            "re_entrant": 0,
            "already_classified": 0,
            "not_unemployed": 0,
            "youth_15_24": 0,
            "prime_25plus": 0,
        }
        for wid, ws in self.workers.items():
            if ws.worker.employment_status != "Unemployed":
                counts["not_unemployed"] += 1
                continue
            if ws.separation_reason != "unknown":
                counts["already_classified"] += 1
                continue
            _age = int(getattr(ws.worker, "age", 30))
            counts["youth_15_24" if _age < 25 else "prime_25plus"] += 1
            _reason = impute_separation_state(ws)
            if _reason is not None:
                counts[_reason] += 1
        _total_imputed = counts["job_loser"] + counts["job_leaver"] + counts["re_entrant"]
        if _total_imputed > 0:
            _w_youth = counts["youth_15_24"] / _total_imputed
            _w_prime = counts["prime_25plus"] / _total_imputed
            _expected = {}
            for _r in ("job_loser", "job_leaver", "re_entrant"):
                _expected[_r] = (
                    _w_youth * LWF_BACKFILL_MIX_15_24[_r] + _w_prime * LWF_BACKFILL_MIX_25PLUS[_r]
                )
            _realized = {
                _r: counts[_r] / _total_imputed for _r in ("job_loser", "job_leaver", "re_entrant")
            }
            _AUDIT_TOL = 0.05
            _miscals = []
            for _r in ("job_loser", "job_leaver", "re_entrant"):
                _d = abs(_realized[_r] - _expected[_r])
                if _d > _AUDIT_TOL:
                    _miscals.append(
                        f"{_r}: realized={_realized[_r]:.3f} vs expected={_expected[_r]:.3f} (Δ={_d:.3f})"
                    )
            counts["_audit_expected"] = _expected
            counts["_audit_realized"] = _realized
            if _miscals:
                logger.warning(
                    f"Separation reasons assigned to the initial unemployed differ from the expected shares by more than {_AUDIT_TOL:.2f}: "
                    + "; ".join(_miscals)
                    + f" ({_total_imputed} workers assigned)."
                )
        return counts

    def apply_separations(self, separations: List[Tuple[str, str, str]]):
        for worker_id, firm_id, sep_type in separations:
            worker_state = self.workers.get(worker_id)
            if worker_state is None:
                continue
            worker = worker_state.worker
            worker_state.previous_tenure_months = worker_state.months_employed
            worker_state.previous_wage = worker.current_wage
            if sep_type == "retirement":
                worker.employment_status = "NILF"
                worker.is_searching = False
                if getattr(self.config.mnl, "nilf_subtype_enabled", False):
                    self.set_retirement_nilf_subtype(worker_state)
                if self.config.skill_dynamics.enabled:
                    current_skill = float(worker.skill_level)
                    skill_loss = current_skill * self.config.skill_dynamics.retirement_skill_decay
                    worker.skill_level = max(
                        current_skill - skill_loss, self.config.skill_dynamics.skill_floor
                    )
            elif worker_id in getattr(self, "_ordinary_temporary_nilf_ids_this_step", set()):
                from separation_destination_split import apply_temporary_nilf_separation

                apply_temporary_nilf_separation(self, worker_state)
            else:
                worker.employment_status = "Unemployed"
                worker.is_searching = False
                worker.unemployment_months = 0
                if self._demographic_flow_instrumenter is not None and getattr(
                    self._demographic_flow_instrumenter, "enabled", False
                ):
                    _imm_raw_h = getattr(worker, "immigrant_status", False)
                    _is_imm_h = (
                        _imm_raw_h.lower() in ("true", "1", "yes", "immigrant")
                        if isinstance(_imm_raw_h, str)
                        else bool(_imm_raw_h)
                    )
                    _age_h = int(getattr(worker, "age", -1))
                    self._demographic_flow_instrumenter.record_e_to_u_no_search_separation(
                        age=_age_h, is_immigrant=_is_imm_h
                    )
                    self._demographic_flow_instrumenter.record_worker_lf_event(
                        worker_id=worker_id,
                        month=self.current_month,
                        year=self.current_year,
                        channel_tag="H",
                        lf_delta=-1,
                        age=_age_h,
                    )
                if sep_type == "layoff":
                    worker_state.separation_reason = "job_loser"
                    _sex = int(getattr(worker, "sex", 1) or 1)
                    worker_state.lwf_retention_draw = lwf_draw_retention(
                        previous_tenure_months=worker_state.previous_tenure_months, sex=_sex
                    )
                else:
                    worker_state.separation_reason = "job_leaver"
                    worker_state.lwf_retention_draw = None
                if sep_type == "layoff":
                    self.apply_layoff_scarring(worker_id)
                self._ei_period_new_separations += 1
                if grant_ei_if_eligible(worker_state, self.config.ei):
                    self._ei_period_new_grants += 1
            worker.commute_time_min = None
            worker_state.months_employed = 0
            firm_state = self.firms.get(firm_id)
            if firm_state:
                if worker_id in firm_state.employees:
                    firm_state.employees.remove(worker_id)
                firm_state.total_separations += 1
        self.stats["separations_this_month"] = len(separations)
        self._update_aggregate_stats()

    def start_vacancy_ledger_month(self):
        self._vacancy_ledger_active = True
        self._vacancy_ledger_opening = len(self.vacancies)
        self._vacancy_ledger_events = []

    def _record_vacancy_event(self, reason: str, delta: int):
        if self._vacancy_ledger_active and delta:
            self._vacancy_ledger_events.append({"reason": str(reason), "delta": int(delta)})

    def finish_vacancy_ledger_month(self):
        closing = len(self.vacancies)
        event_delta = sum((row["delta"] for row in self._vacancy_ledger_events))
        result = {
            "opening_balance": int(self._vacancy_ledger_opening),
            "closing_balance": int(closing),
            "events": list(self._vacancy_ledger_events),
            "event_delta": int(event_delta),
            "identity_residual": int(closing - self._vacancy_ledger_opening - event_delta),
        }
        self._vacancy_ledger_active = False
        return result

    def add_vacancies(self, new_vacancies: List[Vacancy], reason: str = "unattributed_addition"):
        for vacancy in new_vacancies:
            self.vacancies[vacancy.id] = vacancy
            firm_state = self.firms.get(vacancy.firm_id)
            if firm_state:
                firm_state.vacancies[vacancy.id] = vacancy
                firm_state.total_vacancies_posted += 1
        self._record_vacancy_event(reason, len(new_vacancies))
        self.stats["total_vacancies"] = len(self.vacancies)

    def sync_vacancy_dicts(self):
        n_ghosts_purged = 0
        n_orphans_fixed = 0
        for firm_id, firm_state in self.firms.items():
            ghost_ids = [v_id for v_id in firm_state.vacancies if v_id not in self.vacancies]
            for v_id in ghost_ids:
                firm_state.vacancies.pop(v_id)
                n_ghosts_purged += 1
        for v_id, vacancy in self.vacancies.items():
            firm_state = self.firms.get(vacancy.firm_id)
            if firm_state and v_id not in firm_state.vacancies:
                firm_state.vacancies[v_id] = vacancy
                n_orphans_fixed += 1
        if n_ghosts_purged > 0 or n_orphans_fixed > 0:
            logger.warning(
                f"Removed {n_ghosts_purged} stale vacancies from firm records and repaired {n_orphans_fixed} entries of the vacancy register"
            )
        self.stats["total_vacancies"] = len(self.vacancies)

    def export_state(self, filepath: str):
        import pickle
        import dataclasses

        state_dict = {
            "workers": {
                wid: {
                    "worker": dataclasses.asdict(ws.worker),
                    "ei_eligible": ws.ei_eligible,
                    "ei_weeks_remaining": ws.ei_weeks_remaining,
                    "previous_tenure_months": ws.previous_tenure_months,
                    "previous_wage": ws.previous_wage,
                    "months_employed": ws.months_employed,
                    "months_unemployed": ws.months_unemployed,
                    "months_nilf": ws.months_nilf,
                    "total_job_transitions": ws.total_job_transitions,
                    "nilf_subtype": ws.nilf_subtype,
                    "nilf_subtype_months": ws.nilf_subtype_months,
                    "last_intent_search_decision": ws.last_intent_search_decision,
                    "last_intent_search_channel": ws.last_intent_search_channel,
                    "last_intent_search_intensity": ws.last_intent_search_intensity,
                    "separation_reason": ws.separation_reason,
                    "lwf_retention_draw": ws.lwf_retention_draw,
                    "covid_state": ws.covid_state,
                    "covid_attached_employer_id": ws.covid_attached_employer_id,
                    "covid_months_in_state": ws.covid_months_in_state,
                    "covid_pre_shock_wage": ws.covid_pre_shock_wage,
                    "covid_pre_shock_hours_per_week": ws.covid_pre_shock_hours_per_week,
                    "covid_pre_shock_tenure_months": ws.covid_pre_shock_tenure_months,
                    "covid_pre_shock_commute_time_min": ws.covid_pre_shock_commute_time_min,
                }
                for wid, ws in self.workers.items()
            },
            "firms": {
                fid: {
                    "firm": dataclasses.asdict(fs.firm),
                    "employees": list(fs.employees),
                    "total_separations": fs.total_separations,
                    "total_hires": fs.total_hires,
                }
                for fid, fs in self.firms.items()
            },
            "current_year": self.current_year,
            "current_month": self.current_month,
            "stats": self.stats.copy(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(state_dict, f)

    def remove_vacancies(self, vacancy_ids: List[str], reason: str = "unattributed_removal"):
        removed = 0
        for v_id in vacancy_ids:
            vacancy = self.vacancies.pop(v_id, None)
            if vacancy:
                removed += 1
                firm_state = self.firms.get(vacancy.firm_id)
                if firm_state:
                    firm_state.vacancies.pop(v_id, None)
            else:
                for firm_state in self.firms.values():
                    if v_id in firm_state.vacancies:
                        firm_state.vacancies.pop(v_id)
                        break
        self._record_vacancy_event(reason, -removed)
        self.stats["total_vacancies"] = len(self.vacancies)

    def get_all_firms(self) -> Dict[str, Firm]:
        return {f_id: fs.firm for f_id, fs in self.firms.items()}

    def get_all_vacancies(self) -> Dict[str, Vacancy]:
        return self.vacancies.copy()

    def get_worker_observations(self) -> Tuple[torch.Tensor, List[str]]:
        worker_ids = list(self.workers.keys())
        observations = []
        for w_id in worker_ids:
            ws = self.workers[w_id]
            obs = self._encode_worker_observation(ws, w_id)
            observations.append(obs)
        obs_tensor = torch.stack(observations).to(self.device)
        return (obs_tensor, worker_ids)

    def get_firm_observations(self) -> Tuple[torch.Tensor, List[str]]:
        firm_ids = list(self.firms.keys())
        observations = []
        for f_id in firm_ids:
            fs = self.firms[f_id]
            obs = self._encode_firm_observation(fs)
            observations.append(obs)
        obs_tensor = torch.stack(observations).to(self.device)
        return (obs_tensor, firm_ids)

    def _encode_worker_observation(self, ws: WorkerState, worker_id: str) -> torch.Tensor:
        w = ws.worker
        obs = []
        obs.append(w.age / 100.0)
        obs.append(w.skill_level / 4.0)
        obs.append(w.education / 6.0)
        obs.append(w.experience_years / 40.0)
        obs.extend(
            [
                1.0 if w.employment_status == "Employed" else 0.0,
                1.0 if w.employment_status == "Unemployed" else 0.0,
                1.0 if w.employment_status == "NILF" else 0.0,
            ]
        )
        obs.append(w.current_wage / 50.0)
        obs.append(w.industry / 18.0)
        obs.append(w.unemployment_months / 24.0)
        obs.append(w.job_application_attempts / 20.0)
        obs.append(1.0 if ws.ei_eligible else 0.0)
        obs.append(ws.ei_weeks_remaining / 45.0)
        obs.append(self.stats["unemployed"] / len(self.workers))
        obs.append(self.stats["total_vacancies"] / max(len(self.workers), 1))
        w_type = self.worker_types.get(worker_id, 0)
        mf = self.worker_type_mean_fields.get(w_type, {})
        obs.append(mf.get("employment_rate", 0.0))
        obs.append(mf.get("avg_wage", 20.0) / 50.0)
        obs.append(mf.get("avg_search_duration", 0.0) / 24.0)
        obs.append(mf.get("avg_tenure", 0.0) / 60.0)
        obs.append(mf.get("population_share", 0.2))
        obs.append(self.stats["total_vacancies"] / max(self.stats["unemployed"], 1) / 3.0)
        obs.append(
            self.stats["separations_this_month"]
            / max(self.stats.get("employed_pre", self.stats["employed"]), 1)
        )
        obs.append(
            self.stats["u2e_matches_this_month"]
            / max(self.stats.get("unemployed_pre", self.stats["unemployed"]), 1)
        )
        skill = self.worker_skills.get(worker_id, 0.5)
        obs.append(skill)
        target_dim = self.config.environment.worker_state_dim
        while len(obs) < target_dim:
            obs.append(0.0)
        return torch.tensor(obs[:target_dim], dtype=torch.float32)

    def _encode_firm_observation(self, fs: FirmState) -> torch.Tensor:
        f = fs.firm
        obs = []
        obs.append(f.n_employees / 500.0)
        obs.append(f.industry / 18.0)
        obs.append(f.sector / 6.0)
        obs.append(fs.n_vacancies / 10.0)
        obs.append(fs.vacancy_rate)
        obs.append(fs.total_hires / max(fs.total_vacancies_posted, 1))
        obs.append(fs.total_separations / max(f.n_employees, 1))
        if self.theta_is_frozen and self.frozen_theta is not None:
            theta = self.frozen_theta
            firm_industry = f.industry
            tightness = theta["market_tightness_by_industry"].get(firm_industry, 1.0)
            obs.append(min(tightness / 3.0, 1.0))
            obs.append(theta["aggregate_unemployment_rate"])
            obs.append(theta["aggregate_vacancy_rate"])
            obs.append(theta["avg_wage_by_industry"].get(firm_industry, 20.0) / 50.0)
            obs.append(theta["vacancy_fill_rate_by_industry"].get(firm_industry, 0.3))
            obs.append(theta["aggregate_employment_rate"])
        else:
            total = len(self.workers)
            lf = self.stats["employed"] + self.stats["unemployed"]
            tightness = self.stats["total_vacancies"] / max(self.stats["unemployed"], 1)
            obs.append(min(tightness / 3.0, 1.0))
            obs.append(self.stats["unemployed"] / max(lf, 1))
            obs.append(self.stats["total_vacancies"] / max(total, 1))
            ind_wages = [
                ws.worker.current_wage
                for ws in self.workers.values()
                if ws.worker.employment_status == "Employed" and ws.worker.industry == f.industry
            ]
            obs.append((np.mean(ind_wages) if ind_wages else 20.0) / 50.0)
            obs.append(self.stats["matches_this_month"] / max(self.stats["total_vacancies"], 1))
            obs.append(self.stats["employed"] / max(total, 1))
        target_dim = self.config.environment.firm_state_dim
        while len(obs) < target_dim:
            obs.append(0.0)
        return torch.tensor(obs[:target_dim], dtype=torch.float32)

    def get_global_state(self) -> torch.Tensor:
        total_workers = len(self.workers)
        global_obs = [
            self.current_month / 12.0,
            (self.current_year - self.config.simulation.start_year) / 10.0,
            self.stats["employed"] / total_workers,
            self.stats["unemployed"] / total_workers,
            self.stats["nilf"] / total_workers,
            self.stats["total_vacancies"] / total_workers,
            self.stats["u2e_matches_this_month"]
            / max(self.stats.get("unemployed_pre", self.stats["unemployed"]), 1),
            self.stats["separations_this_month"]
            / max(self.stats.get("employed_pre", self.stats["employed"]), 1),
        ]
        for archetype_id in range(9):
            mf = self.worker_type_mean_fields.get(archetype_id, {})
            global_obs.append(mf.get("employment_rate", 0.0))
            global_obs.append(mf.get("avg_wage", 20.0) / 50.0)
        return torch.tensor(global_obs, dtype=torch.float32, device=self.device)

    def get_metrics(self) -> Dict[str, float]:
        total = len(self.workers)
        labor_force = self.stats["employed"] + self.stats["unemployed"]
        return {
            "unemployment_rate": self.stats["unemployed"] / labor_force if labor_force > 0 else 0,
            "employment_rate": self.stats["employed"] / total if total > 0 else 0,
            "participation_rate": labor_force / total if total > 0 else 0,
            "vacancy_rate": (
                self.stats["total_vacancies"]
                / (self.stats["employed"] + self.stats["total_vacancies"])
                if self.stats["employed"] + self.stats["total_vacancies"] > 0
                else 0
            ),
            "job_finding_rate": self.stats["u2e_matches_this_month"]
            / max(self.stats.get("unemployed_pre", self.stats["unemployed"]), 1),
            "separation_rate": self.stats["separations_this_month"]
            / max(self.stats.get("employed_pre", self.stats["employed"]), 1),
            "n_workers": total,
            "n_employed": self.stats["employed"],
            "n_unemployed": self.stats["unemployed"],
            "n_vacancies": self.stats["total_vacancies"],
        }

    def _get_accumulation_rate(self, age: int, education: int, tenure_months: int) -> float:
        skill_cfg = self.config.skill_dynamics
        if age <= skill_cfg.age_young_max:
            base_rate = skill_cfg.accumulation_young
        elif age <= skill_cfg.age_prime_max:
            base_rate = skill_cfg.accumulation_prime
        elif age <= skill_cfg.age_mature_max:
            base_rate = skill_cfg.accumulation_mature
        else:
            base_rate = skill_cfg.accumulation_senior
        edu_modifiers = {
            0: skill_cfg.edu_modifier_0,
            1: skill_cfg.edu_modifier_1,
            2: skill_cfg.edu_modifier_2,
            3: skill_cfg.edu_modifier_3,
            4: skill_cfg.edu_modifier_4,
        }
        edu_mod = edu_modifiers.get(education, 1.0)
        new_job_mod = (
            skill_cfg.new_job_bonus if tenure_months < skill_cfg.new_job_threshold_months else 1.0
        )
        monthly_rate = base_rate * edu_mod * new_job_mod / 12.0
        return monthly_rate

    def _get_skill_ceiling(self, education: int) -> float:
        skill_cfg = self.config.skill_dynamics
        ceilings = {
            0: skill_cfg.skill_ceiling_0,
            1: skill_cfg.skill_ceiling_1,
            2: skill_cfg.skill_ceiling_2,
            3: skill_cfg.skill_ceiling_3,
            4: skill_cfg.skill_ceiling_4,
        }
        return ceilings.get(education, 4.0)

    def _get_depreciation_rate(self, unemployment_months: int) -> float:
        skill_cfg = self.config.skill_dynamics
        if unemployment_months < skill_cfg.grace_period_months:
            return 0.0
        base_rate = skill_cfg.depreciation_annual
        if unemployment_months >= skill_cfg.vltu_threshold_months:
            multiplier = skill_cfg.vltu_multiplier
        elif unemployment_months >= skill_cfg.ltu_threshold_months:
            multiplier = skill_cfg.ltu_multiplier
        else:
            multiplier = 1.0
        monthly_rate = base_rate * multiplier / 12.0
        return monthly_rate

    def process_skill_accumulation(self):
        if not self.config.skill_dynamics.enabled:
            return
        for worker_id, ws in self.workers.items():
            worker = ws.worker
            if worker.employment_status != "Employed":
                continue
            current_skill = float(worker.skill_level)
            ceiling = self._get_skill_ceiling(worker.education)
            if current_skill >= ceiling:
                continue
            monthly_rate = self._get_accumulation_rate(
                age=worker.age, education=worker.education, tenure_months=ws.months_employed
            )
            new_skill = current_skill + monthly_rate
            new_skill = min(new_skill, ceiling)
            worker.skill_level = new_skill

    def process_skill_depreciation(self):
        if not self.config.skill_dynamics.enabled:
            return
        skill_cfg = self.config.skill_dynamics
        for worker_id, ws in self.workers.items():
            worker = ws.worker
            if worker.employment_status == "Employed":
                continue
            current_skill = float(worker.skill_level)
            if current_skill <= skill_cfg.skill_floor:
                continue
            if worker.employment_status == "Unemployed":
                monthly_rate = self._get_depreciation_rate(worker.unemployment_months)
            else:
                monthly_rate = skill_cfg.nilf_depreciation_annual / 12.0
            if monthly_rate <= 0:
                continue
            new_skill = current_skill * (1.0 - monthly_rate)
            new_skill = max(new_skill, skill_cfg.skill_floor)
            worker.skill_level = new_skill

    def apply_layoff_scarring(self, worker_id: str):
        if not self.config.skill_dynamics.enabled:
            return
        skill_cfg = self.config.skill_dynamics
        ws = self.workers.get(worker_id)
        if ws is None:
            return
        worker = ws.worker
        current_skill = float(worker.skill_level)
        skill_loss = current_skill * skill_cfg.layoff_skill_loss
        new_skill = current_skill - skill_loss
        new_skill = max(new_skill, skill_cfg.skill_floor)
        worker.skill_level = new_skill

    def apply_reemployment_boost(self, worker_id: str):
        if not self.config.skill_dynamics.enabled:
            return
        skill_cfg = self.config.skill_dynamics
        ws = self.workers.get(worker_id)
        if ws is None:
            return
        worker = ws.worker
        current_skill = float(worker.skill_level)
        boost = current_skill * skill_cfg.reemployment_boost
        new_skill = current_skill + boost
        ceiling = self._get_skill_ceiling(worker.education)
        new_skill = min(new_skill, ceiling)
        worker.skill_level = new_skill
