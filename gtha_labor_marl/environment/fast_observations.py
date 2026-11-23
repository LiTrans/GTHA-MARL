import torch
from typing import Dict, List, Optional, Any


class VectorizedStateTable:
    NORM_AGE = 100.0
    NORM_SKILL = 4.0
    NORM_EDUCATION = 6.0
    NORM_EXPERIENCE = 40.0
    NORM_WAGE = 50.0
    NORM_INDUSTRY = 18.0
    NORM_UNEMP_MONTHS = 24.0
    NORM_TENURE = 120.0
    NORM_EI_WEEKS = 45.0
    NORM_FIRM_EMP_COARSE = 500.0
    NORM_FIRM_EMP_FINE = 100.0
    NORM_FIRM_VACANCIES = 10.0
    NORM_FIRM_TENURE = 60.0

    def __init__(
        self,
        n_workers: int,
        n_firms: int,
        worker_obs_dim: int = 24,
        firm_obs_dim: int = 22,
        device: torch.device = None,
        nilf_subtype_enabled: bool = False,
        school_window: int = 36,
        family_window: int = 24,
        illness_window: int = 12,
        primeage_windows_active: bool = False,
        primeage_age_lo: int = 35,
        primeage_age_hi: int = 55,
        family_window_primeage: int = 24,
        illness_window_primeage: int = 12,
        age_skill_mean_field_enabled: bool = False,
        attachment_enabled: bool = False,
        allow_retirement_reentry: bool = False,
    ):
        self.n_workers = n_workers
        self.n_firms = n_firms
        self.worker_obs_dim = worker_obs_dim
        self.firm_obs_dim = firm_obs_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nilf_subtype_enabled = bool(nilf_subtype_enabled)
        self.allow_retirement_reentry = allow_retirement_reentry is True
        self.school_window = int(school_window)
        self.family_window = int(family_window)
        self.illness_window = int(illness_window)
        self.primeage_windows_active = bool(primeage_windows_active)
        self.primeage_age_lo = int(primeage_age_lo)
        self.primeage_age_hi = int(primeage_age_hi)
        self.family_window_primeage = int(family_window_primeage)
        self.illness_window_primeage = int(illness_window_primeage)
        self.attachment_enabled = bool(attachment_enabled)
        self.age_skill_mean_field_enabled = bool(age_skill_mean_field_enabled)
        self.w_age = torch.zeros(n_workers, device=self.device)
        self.w_skill = torch.zeros(n_workers, device=self.device)
        self.w_education = torch.zeros(n_workers, device=self.device)
        self.w_experience = torch.zeros(n_workers, device=self.device)
        self.w_status = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self.w_wage = torch.zeros(n_workers, device=self.device)
        self.w_industry = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self.w_tenure = torch.zeros(n_workers, device=self.device)
        self.w_unemployment_months = torch.zeros(n_workers, device=self.device)
        self.w_archetype_id = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self.w_sector_id = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self.w_job_apps = torch.zeros(n_workers, device=self.device)
        self.w_is_searching = torch.zeros(n_workers, dtype=torch.bool, device=self.device)
        self.w_ei_eligible = torch.zeros(n_workers, dtype=torch.bool, device=self.device)
        self.w_ei_weeks = torch.zeros(n_workers, device=self.device)
        self.w_cma = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self.w_da = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self.f_n_employees = torch.zeros(n_firms, device=self.device)
        self.f_industry = torch.zeros(n_firms, dtype=torch.long, device=self.device)
        self.f_sector = torch.zeros(n_firms, dtype=torch.long, device=self.device)
        self.f_n_vacancies = torch.zeros(n_firms, device=self.device)
        self.f_total_hires = torch.zeros(n_firms, device=self.device)
        self.f_total_separations = torch.zeros(n_firms, device=self.device)
        self.f_vacancies_posted = torch.zeros(n_firms, device=self.device)
        self.f_cma = torch.zeros(n_firms, dtype=torch.long, device=self.device)
        self.f_target_size = torch.zeros(n_firms, device=self.device)
        self.f_wage_premium = torch.ones(n_firms, device=self.device)
        self.f_avg_tenure = torch.zeros(n_firms, device=self.device)
        self.f_avg_match_quality = torch.zeros(n_firms, device=self.device)
        self.f_months_since_last_hire = torch.zeros(n_firms, device=self.device)
        self.f_industry_avg_wage = torch.zeros(n_firms, device=self.device)
        self.f_hire_rate_recent = torch.zeros(n_firms, device=self.device)
        self.w_mean_field = torch.zeros(n_workers, 5, device=self.device)
        self.w_cross_type = torch.zeros(3, device=self.device)
        self.w_br_skill = torch.zeros(n_workers, device=self.device)
        self.w_age_skill_mf = torch.zeros(90, device=self.device)
        self.w_nilf_subtype = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self.w_nilf_subtype_months = torch.zeros(n_workers, device=self.device)
        self.w_attachment = torch.ones(n_workers, device=self.device)
        self.f_frozen_tightness = torch.zeros(n_firms, device=self.device)
        self.f_frozen_ind_avg_wage = torch.zeros(n_firms, device=self.device)
        self.f_frozen_fill_rate = torch.zeros(n_firms, device=self.device)
        self.f_frozen_unemp_rate = torch.tensor(0.0, device=self.device)
        self.f_frozen_vac_rate = torch.tensor(0.0, device=self.device)
        self.f_frozen_emp_rate = torch.tensor(0.0, device=self.device)
        self.unemployment_rate = torch.tensor(0.0, device=self.device)
        self.vacancy_rate = torch.tensor(0.0, device=self.device)
        self.n_vacancies_total = torch.tensor(0, device=self.device)
        self._worker_obs = torch.zeros(n_workers, worker_obs_dim, device=self.device)
        self._firm_obs = torch.zeros(n_firms, firm_obs_dim, device=self.device)
        self.worker_id_to_idx: Dict[str, int] = {}
        self.firm_id_to_idx: Dict[str, int] = {}

    def resize_workers(self, new_n_workers: int):
        if new_n_workers <= self.n_workers:
            return
        old_n = self.n_workers
        self.n_workers = new_n_workers

        def extend_tensor(t: torch.Tensor, dtype=None) -> torch.Tensor:
            extra = new_n_workers - old_n
            if dtype is None:
                dtype = t.dtype
            extension = torch.zeros(extra, dtype=dtype, device=self.device)
            return torch.cat([t, extension])

        self.w_age = extend_tensor(self.w_age)
        self.w_skill = extend_tensor(self.w_skill)
        self.w_education = extend_tensor(self.w_education)
        self.w_experience = extend_tensor(self.w_experience)
        self.w_status = extend_tensor(self.w_status, dtype=torch.long)
        self.w_wage = extend_tensor(self.w_wage)
        self.w_industry = extend_tensor(self.w_industry, dtype=torch.long)
        self.w_tenure = extend_tensor(self.w_tenure)
        self.w_unemployment_months = extend_tensor(self.w_unemployment_months)
        self.w_archetype_id = extend_tensor(self.w_archetype_id, dtype=torch.long)
        self.w_sector_id = extend_tensor(self.w_sector_id, dtype=torch.long)
        self.w_job_apps = extend_tensor(self.w_job_apps)
        self.w_ei_eligible = extend_tensor(self.w_ei_eligible, dtype=torch.bool)
        self.w_ei_weeks = extend_tensor(self.w_ei_weeks)
        self.w_cma = extend_tensor(self.w_cma, dtype=torch.long)
        self.w_da = extend_tensor(self.w_da, dtype=torch.long)
        extra_mf = torch.zeros(new_n_workers - old_n, 5, device=self.device)
        self.w_mean_field = torch.cat([self.w_mean_field, extra_mf])
        self.w_br_skill = extend_tensor(self.w_br_skill)
        self.w_nilf_subtype = extend_tensor(self.w_nilf_subtype, dtype=torch.long)
        self.w_nilf_subtype_months = extend_tensor(self.w_nilf_subtype_months)
        self.w_is_searching = extend_tensor(self.w_is_searching, dtype=torch.bool)
        if self.w_attachment.shape[0] < new_n_workers:
            _old_n = self.w_attachment.shape[0]
            self.w_attachment = extend_tensor(self.w_attachment)
            self.w_attachment[_old_n:] = 1.0
        extra_obs = torch.zeros(new_n_workers - old_n, self.worker_obs_dim, device=self.device)
        self._worker_obs = torch.cat([self._worker_obs, extra_obs])

    def initialize_from_state_manager(
        self, state_manager, worker_ids: List[str], firm_ids: List[str]
    ):
        self.worker_id_to_idx = {wid: i for i, wid in enumerate(worker_ids)}
        self.firm_id_to_idx = {fid: i for i, fid in enumerate(firm_ids)}
        for i, wid in enumerate(worker_ids):
            ws = state_manager.workers.get(wid)
            if ws is None:
                continue
            w = ws.worker
            self.w_age[i] = w.age
            self.w_skill[i] = w.skill_level
            self.w_education[i] = w.education
            self.w_experience[i] = getattr(w, "experience_years", 0)
            self.w_status[i] = {"Employed": 0, "Unemployed": 1, "NILF": 2}.get(
                w.employment_status, 2
            )
            self.w_wage[i] = w.current_wage if w.current_wage else 0.0
            self.w_industry[i] = w.industry if w.industry else 0
            self.w_tenure[i] = getattr(ws, "months_employed", 0)
            self.w_unemployment_months[i] = w.unemployment_months
            self.w_archetype_id[i] = getattr(w, "archetype_id", 0)
            self.w_sector_id[i] = getattr(w, "sector_id", 0)
            self.w_job_apps[i] = w.job_application_attempts
            self.w_ei_eligible[i] = ws.ei_eligible
            self.w_ei_weeks[i] = ws.ei_weeks_remaining
            self.w_cma[i] = w.cma if w.cma else 0
            self.w_nilf_subtype[i] = int(getattr(ws, "nilf_subtype", 0))
            self.w_nilf_subtype_months[i] = int(getattr(ws, "nilf_subtype_months", 0))
            self.w_attachment[i] = float(getattr(ws.worker, "attachment", 1.0))
        for i, fid in enumerate(firm_ids):
            fs = state_manager.firms.get(fid)
            if fs is None:
                continue
            f = fs.firm
            self.f_n_employees[i] = len(fs.employees)
            self.f_industry[i] = f.industry
            self.f_sector[i] = f.sector
            self.f_n_vacancies[i] = fs.n_vacancies
            self.f_total_hires[i] = fs.total_hires
            self.f_total_separations[i] = fs.total_separations
            self.f_vacancies_posted[i] = fs.total_vacancies_posted
            self.f_cma[i] = f.cma if hasattr(f, "cma") else 0
            self.f_target_size[i] = getattr(f, "target_size", max(1, len(fs.employees)))
            self.f_wage_premium[i] = fs.wage_premium
            if fs.employees:
                tenures = []
                match_quals = []
                for emp_id in fs.employees:
                    ws = state_manager.workers.get(emp_id)
                    if ws:
                        tenures.append(ws.months_employed)
                        match_quals.append(getattr(ws, "match_quality", 0.5))
                self.f_avg_tenure[i] = sum(tenures) / len(tenures) if tenures else 0.0
                self.f_avg_match_quality[i] = (
                    sum(match_quals) / len(match_quals) if match_quals else 0.5
                )
        self._sync_worker_mean_fields(state_manager, worker_ids)
        self._sync_frozen_theta(state_manager)
        self._update_global_state()

    def _update_global_state(self):
        n_unemployed = (self.w_status == 1).sum().float()
        n_in_lf = (self.w_status != 2).sum().float()
        self.unemployment_rate = n_unemployed / (n_in_lf + 1e-06)
        self.n_vacancies_total = self.f_n_vacancies.sum()
        self.vacancy_rate = self.n_vacancies_total / self.n_workers

    def sync_from_state_manager(self, state_manager, worker_ids: List[str]):
        n_workers = len(worker_ids)
        if n_workers > self.n_workers:
            self.resize_workers(n_workers)
        _mismatched: list = []
        for i, wid in enumerate(worker_ids):
            ws = state_manager.workers.get(wid)
            if ws is None:
                continue
            if wid not in self.worker_id_to_idx:
                continue
            _t = float(self.w_age[i].item())
            _w = float(ws.worker.age)
            if _t == 0.0 and _w > 0.0:
                continue
            if _t != _w:
                _mismatched.append((i, wid, _t, _w))
        if _mismatched:
            _n = len(_mismatched)
            _head = _mismatched[:3]
            raise AssertionError(
                f"Worker ages in the state table differ from the worker records for {_n} workers. First three: {_head}. An age changed without the state table being updated."
            )
        for i, wid in enumerate(worker_ids):
            ws = state_manager.workers.get(wid)
            if ws is None:
                continue
            w = ws.worker
            is_new_worker = wid not in self.worker_id_to_idx
            if is_new_worker:
                self.worker_id_to_idx[wid] = i
                self.w_skill[i] = w.skill_level
                self.w_education[i] = w.education
                self.w_experience[i] = getattr(w, "experience_years", 0)
                self.w_archetype_id[i] = getattr(w, "archetype_id", 0)
                self.w_sector_id[i] = getattr(w, "sector_id", 0)
                self.w_job_apps[i] = w.job_application_attempts
                self.w_ei_eligible[i] = getattr(ws, "ei_eligible", False)
                self.w_ei_weeks[i] = getattr(ws, "ei_weeks_remaining", 0)
                self.w_cma[i] = w.cma if w.cma else 0
            self.w_age[i] = w.age
            self.w_status[i] = {"Employed": 0, "Unemployed": 1, "NILF": 2}.get(
                w.employment_status, 2
            )
            self.w_wage[i] = w.current_wage if w.current_wage else 0.0
            self.w_industry[i] = w.industry if w.industry else 0
            self.w_tenure[i] = getattr(ws, "months_employed", 0)
            if w.employment_status == "NILF":
                self.w_unemployment_months[i] = getattr(ws, "months_nilf", 0)
            else:
                self.w_unemployment_months[i] = w.unemployment_months
            self.w_is_searching[i] = getattr(w, "is_searching", False)
            self.w_nilf_subtype[i] = int(getattr(ws, "nilf_subtype", 0))
            self.w_nilf_subtype_months[i] = int(getattr(ws, "nilf_subtype_months", 0))
        for fid, idx in self.firm_id_to_idx.items():
            fs = state_manager.firms.get(fid)
            if fs is None:
                continue
            f = fs.firm
            self.f_n_employees[idx] = len(fs.employees)
            self.f_n_vacancies[idx] = fs.n_vacancies
            self.f_total_hires[idx] = fs.total_hires
            self.f_total_separations[idx] = fs.total_separations
            self.f_vacancies_posted[idx] = fs.total_vacancies_posted
            self.f_target_size[idx] = getattr(f, "target_size", max(1, len(fs.employees)))
            self.f_wage_premium[idx] = fs.wage_premium
            if fs.employees:
                tenures = []
                match_quals = []
                for emp_id in fs.employees:
                    ws = state_manager.workers.get(emp_id)
                    if ws:
                        tenures.append(ws.months_employed)
                        match_quals.append(getattr(ws, "match_quality", 0.5))
                self.f_avg_tenure[idx] = sum(tenures) / len(tenures) if tenures else 0.0
                self.f_avg_match_quality[idx] = (
                    sum(match_quals) / len(match_quals) if match_quals else 0.5
                )
            else:
                self.f_avg_tenure[idx] = 0.0
                self.f_avg_match_quality[idx] = 0.0
        self._sync_worker_mean_fields(state_manager, worker_ids)
        self._sync_frozen_theta(state_manager)

    def _sync_worker_mean_fields(self, state_manager, worker_ids):
        for type_id, mf in state_manager.worker_type_mean_fields.items():
            type_mask = self.w_archetype_id == type_id
            if type_mask.any():
                self.w_mean_field[type_mask, 0] = mf.get("employment_rate", 0.0)
                self.w_mean_field[type_mask, 1] = mf.get("avg_wage", 20.0) / 50.0
                self.w_mean_field[type_mask, 2] = mf.get("avg_search_duration", 0.0) / 24.0
                self.w_mean_field[type_mask, 3] = mf.get("avg_tenure", 0.0) / 60.0
                self.w_mean_field[type_mask, 4] = mf.get("population_share", 0.2)
        stats = state_manager.stats
        n_unemployed = max(stats.get("unemployed", 1), 1)
        n_employed_pre = max(stats.get("employed_pre", stats.get("employed", 1)), 1)
        n_unemployed_pre = max(stats.get("unemployed_pre", n_unemployed), 1)
        self.w_cross_type[0] = stats.get("total_vacancies", 0) / n_unemployed / 3.0
        self.w_cross_type[1] = stats.get("separations_this_month", 0) / n_employed_pre
        self.w_cross_type[2] = stats.get("u2e_matches_this_month", 0) / n_unemployed_pre
        for i, wid in enumerate(worker_ids):
            if i >= self.n_workers:
                break
            self.w_br_skill[i] = state_manager.worker_skills.get(wid, 0.5)

    def _sync_frozen_theta(self, state_manager):
        if state_manager.theta_is_frozen and state_manager.frozen_theta is not None:
            theta = state_manager.frozen_theta
            for fid, idx in self.firm_id_to_idx.items():
                fs = state_manager.firms.get(fid)
                if fs is None:
                    continue
                ind = fs.firm.industry
                self.f_frozen_tightness[idx] = min(
                    theta["market_tightness_by_industry"].get(ind, 1.0) / 3.0, 1.0
                )
                self.f_frozen_ind_avg_wage[idx] = (
                    theta["avg_wage_by_industry"].get(ind, 20.0) / 50.0
                )
                self.f_frozen_fill_rate[idx] = theta["vacancy_fill_rate_by_industry"].get(ind, 0.3)
            self.f_frozen_unemp_rate = torch.tensor(
                theta["aggregate_unemployment_rate"], device=self.device
            )
            self.f_frozen_vac_rate = torch.tensor(
                theta["aggregate_vacancy_rate"], device=self.device
            )
            self.f_frozen_emp_rate = torch.tensor(
                theta["aggregate_employment_rate"], device=self.device
            )
        else:
            stats = state_manager.stats
            total = max(len(state_manager.workers), 1)
            lf = max(stats.get("employed", 0) + stats.get("unemployed", 0), 1)
            n_unemployed = max(stats.get("unemployed", 1), 1)
            for fid, idx in self.firm_id_to_idx.items():
                fs = state_manager.firms.get(fid)
                if fs is None:
                    continue
                tightness = stats.get("total_vacancies", 0) / n_unemployed
                self.f_frozen_tightness[idx] = min(tightness / 3.0, 1.0)
                ind = fs.firm.industry
                ind_wages = [
                    ws.worker.current_wage
                    for wid, ws in state_manager.workers.items()
                    if ws.worker.employment_status == "Employed"
                    and wid not in {}
                    and (ws.worker.industry == ind)
                ]
                self.f_frozen_ind_avg_wage[idx] = (
                    sum(ind_wages) / len(ind_wages) if ind_wages else 20.0
                ) / 50.0
                self.f_frozen_fill_rate[idx] = stats.get("matches_this_month", 0) / max(
                    stats.get("total_vacancies", 1), 1
                )
            self.f_frozen_unemp_rate = torch.tensor(
                stats.get("unemployed", 0) / lf, device=self.device
            )
            self.f_frozen_vac_rate = torch.tensor(
                stats.get("total_vacancies", 0) / total, device=self.device
            )
            self.f_frozen_emp_rate = torch.tensor(
                stats.get("employed", 0) / total, device=self.device
            )

    def compute_search_eligibility_mask(self) -> torch.Tensor:
        st = self.w_nilf_subtype
        m = self.w_nilf_subtype_months
        if self.primeage_windows_active:
            primeage = (self.w_age >= self.primeage_age_lo) & (self.w_age < self.primeage_age_hi)
            family_w = torch.where(
                primeage,
                torch.full_like(m, self.family_window_primeage),
                torch.full_like(m, self.family_window),
            )
            illness_w = torch.where(
                primeage,
                torch.full_like(m, self.illness_window_primeage),
                torch.full_like(m, self.illness_window),
            )
            return (
                (st == 0)
                | (st == 5)
                | (st == 1) & self.allow_retirement_reentry
                | (st == 2) & (m >= self.school_window)
                | (st == 4) & (m >= family_w)
                | (st == 3) & (m >= illness_w)
            )
        return (
            (st == 0)
            | (st == 5)
            | (st == 1) & self.allow_retirement_reentry
            | (st == 2) & (m >= self.school_window)
            | (st == 4) & (m >= self.family_window)
            | (st == 3) & (m >= self.illness_window)
        )

    def get_worker_observations(self) -> torch.Tensor:
        obs = self._worker_obs
        obs[:, 0] = self.w_age / self.NORM_AGE
        obs[:, 1] = self.w_skill / self.NORM_SKILL
        obs[:, 2] = self.w_education / self.NORM_EDUCATION
        obs[:, 3] = self.w_experience / self.NORM_EXPERIENCE
        obs[:, 4] = (self.w_status == 0).float()
        obs[:, 5] = (self.w_status == 1).float()
        obs[:, 6] = (self.w_status == 2).float()
        obs[:, 7] = self.w_wage / self.NORM_WAGE
        obs[:, 8] = self.w_industry.float() / self.NORM_INDUSTRY
        obs[:, 9] = self.w_unemployment_months / self.NORM_UNEMP_MONTHS
        obs[:, 10] = self.w_tenure / self.NORM_TENURE
        obs[:, 11] = self.w_ei_eligible.float()
        obs[:, 12] = self.w_ei_weeks / self.NORM_EI_WEEKS
        obs[:, 13] = self.unemployment_rate
        obs[:, 14] = self.vacancy_rate
        obs[:, 15:20] = self.w_mean_field
        obs[:, 20] = self.w_cross_type[0]
        obs[:, 21] = self.w_cross_type[1]
        obs[:, 22] = self.w_cross_type[2]
        if not self.nilf_subtype_enabled:
            obs[:, 23] = self.w_br_skill
            return obs
        st = self.w_nilf_subtype
        obs[:, 23] = (st == 0).float()
        obs[:, 24] = (st == 1).float()
        obs[:, 25] = (st == 2).float()
        obs[:, 26] = (st == 3).float()
        obs[:, 27] = (st == 4).float()
        obs[:, 28] = (st == 5).float()
        elig = self.compute_search_eligibility_mask()
        obs[:, 29] = elig.float()
        obs[:, 30] = self.w_br_skill
        if self.age_skill_mean_field_enabled:
            obs[:, 31:121] = self.w_age_skill_mf.unsqueeze(0).expand(obs.shape[0], -1)
        if self.attachment_enabled:
            obs[:, -1] = self.w_attachment
        return obs

    def get_firm_observations(self) -> torch.Tensor:
        obs = self._firm_obs
        obs[:, 0] = self.f_n_employees / self.NORM_FIRM_EMP_COARSE
        obs[:, 1] = self.f_industry.float() / self.NORM_INDUSTRY
        obs[:, 2] = self.f_sector.float() / 6.0
        obs[:, 3] = self.f_n_vacancies / self.NORM_FIRM_VACANCIES
        vacancy_rate = self.f_n_vacancies / (self.f_n_employees + self.f_n_vacancies + 1e-06)
        obs[:, 4] = vacancy_rate
        fill_rate = self.f_total_hires / (self.f_vacancies_posted + 1e-06)
        obs[:, 5] = fill_rate.clamp(0, 1)
        sep_rate = self.f_total_separations / (self.f_n_employees + 1e-06)
        obs[:, 6] = sep_rate.clamp(0, 1)
        mean_fill = fill_rate.mean()
        obs[:, 7] = (fill_rate - mean_fill).clamp(-1, 1)
        mean_vac_rate = vacancy_rate.mean()
        obs[:, 8] = (vacancy_rate - mean_vac_rate).clamp(-1, 1)
        staffing_gap = self.f_target_size - self.f_n_employees
        obs[:, 9] = staffing_gap / (self.f_target_size + 1e-06)
        obs[:, 10] = (self.f_wage_premium - 1.0) * 5.0
        obs[:, 11] = self.f_avg_tenure / self.NORM_FIRM_TENURE
        obs[:, 12] = self.f_avg_match_quality
        obs[:, 13] = self.f_n_employees / self.NORM_FIRM_EMP_FINE
        obs[:, 14] = self.unemployment_rate
        gap_ratio = staffing_gap / (self.f_target_size + 1e-06)
        obs[:, 15] = gap_ratio.abs() * gap_ratio
        obs[:, 16] = self.f_frozen_tightness
        obs[:, 17] = self.f_frozen_ind_avg_wage
        obs[:, 18] = self.f_frozen_fill_rate
        obs[:, 19] = self.f_frozen_unemp_rate
        obs[:, 20] = self.f_frozen_vac_rate
        obs[:, 21] = self.f_frozen_emp_rate
        return obs


class VectorizedVacancySelector:

    def __init__(
        self, n_workers: int, n_visible: int = 15, vacancy_dim: int = 7, device: torch.device = None
    ):
        self.n_workers = n_workers
        self.n_visible = n_visible
        self.vacancy_dim = vacancy_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._context = torch.zeros(n_workers, n_visible, vacancy_dim, device=self.device)
        self._vacancy_features: Optional[torch.Tensor] = None
        self._vacancy_cma: Optional[torch.Tensor] = None
        self._vacancy_industry: Optional[torch.Tensor] = None
        self._n_vacancies = 0

    def resize(self, new_n_workers: int):
        if new_n_workers <= self.n_workers:
            return
        old_n = self.n_workers
        self.n_workers = new_n_workers
        extra = new_n_workers - old_n
        extra_context = torch.zeros(extra, self.n_visible, self.vacancy_dim, device=self.device)
        self._context = torch.cat([self._context, extra_context], dim=0)

    def update_vacancies(self, vacancies: List[Any]):
        if not vacancies:
            self._vacancy_features = None
            self._n_vacancies = 0
            return
        n_vac = len(vacancies)
        self._n_vacancies = n_vac
        self._vacancy_features = torch.zeros(n_vac, self.vacancy_dim, device=self.device)
        self._vacancy_cma = torch.zeros(n_vac, dtype=torch.long, device=self.device)
        self._vacancy_industry = torch.zeros(n_vac, dtype=torch.long, device=self.device)
        for j, v in enumerate(vacancies):
            self._vacancy_features[j, 0] = v.offered_wage / 100.0
            self._vacancy_features[j, 1] = v.skill_required / 4.0
            self._vacancy_features[j, 2] = v.education_required / 6.0
            self._vacancy_features[j, 3] = v.industry / 18.0
            self._vacancy_features[j, 4] = v.sector / 6.0
            self._vacancy_features[j, 5] = v.cma / 1000.0
            self._vacancy_features[j, 6] = getattr(v, "job_security", 0.5)
            self._vacancy_cma[j] = v.cma
            self._vacancy_industry[j] = v.industry

    def get_visible_vacancies_topk(
        self, worker_cma: torch.Tensor, worker_industry: torch.Tensor, worker_skill: torch.Tensor
    ) -> torch.Tensor:
        self._context.zero_()
        if self._vacancy_features is None or self._n_vacancies == 0:
            return self._context
        n_vac = self._n_vacancies
        n_visible = min(self.n_visible, n_vac)
        cma_match = (worker_cma.unsqueeze(1) == self._vacancy_cma.unsqueeze(0)).float() * 2.0
        industry_match = (
            worker_industry.unsqueeze(1) == self._vacancy_industry.unsqueeze(0)
        ).float() * 1.5
        vac_skill = self._vacancy_features[:, 1] * 4.0
        skill_diff = torch.abs(worker_skill.unsqueeze(1) - vac_skill.unsqueeze(0))
        skill_match = torch.exp(-skill_diff * 0.5)
        wage_score = self._vacancy_features[:, 0].unsqueeze(0)
        relevance = cma_match + industry_match + skill_match + wage_score
        _, top_indices = torch.topk(relevance, k=n_visible, dim=1)
        flat_indices = top_indices.reshape(-1)
        gathered = self._vacancy_features[flat_indices]
        self._context[:, :n_visible, :] = gathered.reshape(
            self.n_workers, n_visible, self.vacancy_dim
        )
        return self._context


class CircularHistoryBuffer:

    def __init__(
        self,
        n_workers: int,
        history_length: int = 18,
        history_dim: int = 5,
        device: torch.device = None,
    ):
        self.n_workers = n_workers
        self.history_length = history_length
        self.history_dim = history_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._buffer = torch.zeros(n_workers, history_length, history_dim, device=self.device)
        self._ptr = torch.zeros(n_workers, dtype=torch.long, device=self.device)
        self._status_map = {"Employed": 1.0, "Unemployed": 0.0, "NILF": -1.0}

    def resize(self, new_n_workers: int):
        if new_n_workers <= self.n_workers:
            return
        old_n = self.n_workers
        self.n_workers = new_n_workers
        extra = new_n_workers - old_n
        extra_buffer = torch.zeros(extra, self.history_length, self.history_dim, device=self.device)
        self._buffer = torch.cat([self._buffer, extra_buffer], dim=0)
        extra_ptr = torch.zeros(extra, dtype=torch.long, device=self.device)
        self._ptr = torch.cat([self._ptr, extra_ptr])

    def initialize_from_state_manager(self, state_manager, worker_ids: List[str]):
        for i, wid in enumerate(worker_ids):
            ws = state_manager.workers.get(wid)
            if ws is None:
                continue
            hist = ws.history[-self.history_length :]
            for j, entry in enumerate(hist):
                self._buffer[i, j, 0] = self._status_map.get(entry.get("status", "Unemployed"), 0.0)
                self._buffer[i, j, 1] = entry.get("wage", 0.0) / 100.0
                self._buffer[i, j, 2] = entry.get("industry", 0) / 18.0
                self._buffer[i, j, 3] = entry.get("tenure", 0) / 120.0
                self._buffer[i, j, 4] = 1.0 if entry.get("is_searching", False) else 0.0
            self._ptr[i] = len(hist) % self.history_length

    def push_step(
        self,
        status: torch.Tensor,
        wage: torch.Tensor,
        industry: torch.Tensor,
        tenure: torch.Tensor,
        is_searching: torch.Tensor,
    ):
        status_encoded = torch.zeros(self.n_workers, device=self.device)
        status_encoded[status == 0] = 1.0
        status_encoded[status == 1] = 0.0
        status_encoded[status == 2] = -1.0
        positions = self._ptr.clone()
        worker_indices = torch.arange(self.n_workers, device=self.device)
        self._buffer[worker_indices, positions, 0] = status_encoded
        self._buffer[worker_indices, positions, 1] = wage / 100.0
        self._buffer[worker_indices, positions, 2] = industry.float() / 18.0
        self._buffer[worker_indices, positions, 3] = tenure / 120.0
        self._buffer[worker_indices, positions, 4] = is_searching.float()
        self._ptr = (self._ptr + 1) % self.history_length

    def get_histories(self) -> torch.Tensor:
        H = self.history_length
        D = self.history_dim
        idx = (self._ptr.unsqueeze(1) + torch.arange(H, device=self._buffer.device)) % H
        return self._buffer.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))


class FastObservationGenerator:

    def __init__(
        self,
        n_workers: int,
        n_firms: int,
        worker_obs_dim: int = 24,
        firm_obs_dim: int = 22,
        n_visible_vacancies: int = 15,
        vacancy_dim: int = 7,
        history_length: int = 18,
        history_dim: int = 5,
        device: torch.device = None,
        nilf_subtype_enabled: bool = False,
        school_window: int = 36,
        family_window: int = 24,
        illness_window: int = 12,
        primeage_windows_active: bool = False,
        primeage_age_lo: int = 35,
        primeage_age_hi: int = 55,
        family_window_primeage: int = 24,
        illness_window_primeage: int = 12,
        age_skill_mean_field_enabled: bool = False,
        attachment_enabled: bool = False,
        allow_retirement_reentry: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_table = VectorizedStateTable(
            n_workers=n_workers,
            n_firms=n_firms,
            worker_obs_dim=worker_obs_dim,
            firm_obs_dim=firm_obs_dim,
            device=self.device,
            nilf_subtype_enabled=nilf_subtype_enabled,
            school_window=school_window,
            family_window=family_window,
            illness_window=illness_window,
            primeage_windows_active=primeage_windows_active,
            primeage_age_lo=primeage_age_lo,
            primeage_age_hi=primeage_age_hi,
            family_window_primeage=family_window_primeage,
            illness_window_primeage=illness_window_primeage,
            age_skill_mean_field_enabled=age_skill_mean_field_enabled,
            attachment_enabled=attachment_enabled,
            allow_retirement_reentry=allow_retirement_reentry,
        )
        self.vacancy_selector = VectorizedVacancySelector(
            n_workers=n_workers,
            n_visible=n_visible_vacancies,
            vacancy_dim=vacancy_dim,
            device=self.device,
        )
        self.history_buffer = CircularHistoryBuffer(
            n_workers=n_workers,
            history_length=history_length,
            history_dim=history_dim,
            device=self.device,
        )

    def initialize(self, state_manager, worker_ids: List[str], firm_ids: List[str]):
        self.state_table.initialize_from_state_manager(state_manager, worker_ids, firm_ids)
        self.history_buffer.initialize_from_state_manager(state_manager, worker_ids)
        vacancies = list(state_manager.vacancies.values())
        self.vacancy_selector.update_vacancies(vacancies)

    def get_observations(self) -> Dict[str, torch.Tensor]:
        self.state_table._update_global_state()
        worker_obs = self.state_table.get_worker_observations()
        firm_obs = self.state_table.get_firm_observations()
        worker_context = self.vacancy_selector.get_visible_vacancies_topk(
            worker_cma=self.state_table.w_cma,
            worker_industry=self.state_table.w_industry,
            worker_skill=self.state_table.w_skill,
        )
        worker_history = self.history_buffer.get_histories()
        return {
            "worker_obs": worker_obs,
            "firm_obs": firm_obs,
            "worker_context": worker_context,
            "worker_history": worker_history,
        }

    def update_vacancies(self, vacancies: List[Any]):
        self.vacancy_selector.update_vacancies(vacancies)

    def resize_all(self, new_n_workers: int):
        self.state_table.resize_workers(new_n_workers)
        self.vacancy_selector.resize(new_n_workers)
        self.history_buffer.resize(new_n_workers)

    def push_history_step(self):
        n_state = self.state_table.n_workers
        if self.history_buffer.n_workers < n_state:
            self.history_buffer.resize(n_state)
        if self.vacancy_selector.n_workers < n_state:
            self.vacancy_selector.resize(n_state)
        self.history_buffer.push_step(
            status=self.state_table.w_status,
            wage=self.state_table.w_wage,
            industry=self.state_table.w_industry,
            tenure=self.state_table.w_tenure,
            is_searching=self.state_table.w_is_searching,
        )
