import math
import torch
from trade_war_stress.observed_monthly_stocks import MonthlyModelStocks
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from tensordict import TensorDict
from torchrl.envs import EnvBase

try:
    from torchrl.data import (
        Composite as CompositeSpec,
        Bounded as BoundedTensorSpec,
        Categorical as DiscreteTensorSpec,
        Unbounded as UnboundedContinuousTensorSpec,
    )
except ImportError:
    from torchrl.data.tensor_specs import (
        CompositeSpec,
        BoundedTensorSpec,
        DiscreteTensorSpec,
        UnboundedContinuousTensorSpec,
    )
import logging
from collections import defaultdict
import os
from posting_intensity import growth_coefficient as posting_growth_coefficient
from .state_manager import StateManager, VALID_INDUSTRIES
from .covid_transition_regime import (
    COVID_MARGINAL_ATTACHED,
    COVID_NONE,
    COVID_TEMP_LAYOFF,
    CovidDriverTable,
    CovidTransitionRegime,
)
from .fast_observations import FastObservationGenerator
from .nilf_subtype import (
    is_search_eligible,
    normalize_sex_value,
    sample_sex_for_age,
    SUBTYPE_RETIREMENT,
)
from .matching_mechanism_kround import MatchingMechanism, Vacancy, INDUSTRY_WAGE_CEILING
from rewards.firm_rewards import compute_firm_reward, FirmRewardConfig

logger = logging.getLogger(__name__)
YOUTH_INDUSTRY_DISTRIBUTION = {
    8: 0.33,
    16: 0.245,
    15: 0.098,
    12: 0.044,
    17: 0.042,
    5: 0.036,
    13: 0.034,
    10: 0.029,
    14: 0.028,
    4: 0.026,
    11: 0.021,
    18: 0.02,
    7: 0.019,
    1: 0.013,
    9: 0.012,
    3: 0.001,
    2: 0.0,
}
YOUTH_INDUSTRIES = list(YOUTH_INDUSTRY_DISTRIBUTION.keys())
_youth_probs_raw = list(YOUTH_INDUSTRY_DISTRIBUTION.values())
_youth_probs_sum = sum(_youth_probs_raw)
YOUTH_INDUSTRY_PROBS = [p / _youth_probs_sum for p in _youth_probs_raw]
ANNUAL_IMMIGRATION_RATE = 0.025
ANNUAL_EMIGRATION_RATE = 0.008
NEW_ENTRANT_RATE = 0.012
MORTALITY_RATES_BY_AGE = {(0, 49): 0.001, (50, 64): 0.005, (65, 74): 0.015, (75, 100): 0.04}
WAGE_GROWTH_BY_INDUSTRY = {
    1: 0.003,
    2: 0.004,
    3: 0.004,
    4: 0.004,
    5: 0.003,
    7: 0.003,
    8: 0.002,
    9: 0.003,
    10: 0.004,
    11: 0.005,
    12: 0.002,
    13: 0.003,
    14: 0.004,
    15: 0.002,
    16: 0.002,
    17: 0.002,
    18: 0.003,
}
TENURE_PREMIUM_RATE = 0.002
TENURE_PREMIUM_MAX = 0.008
INDUSTRY_SKILL_DISTRIBUTION = {
    8: [0.5, 0.35, 0.1, 0.05],
    12: [0.4, 0.4, 0.15, 0.05],
    16: [0.55, 0.3, 0.1, 0.05],
    17: [0.45, 0.35, 0.15, 0.05],
    1: [0.3, 0.4, 0.25, 0.05],
    4: [0.15, 0.4, 0.35, 0.1],
    5: [0.15, 0.35, 0.35, 0.15],
    9: [0.2, 0.35, 0.35, 0.1],
    2: [0.1, 0.25, 0.4, 0.25],
    3: [0.1, 0.25, 0.4, 0.25],
    7: [0.15, 0.3, 0.4, 0.15],
    10: [0.05, 0.2, 0.45, 0.3],
    11: [0.05, 0.15, 0.45, 0.35],
    13: [0.05, 0.15, 0.4, 0.4],
    14: [0.15, 0.3, 0.35, 0.2],
    15: [0.1, 0.25, 0.4, 0.25],
    18: [0.1, 0.25, 0.45, 0.2],
}
DEFAULT_SKILL_DISTRIBUTION = [0.25, 0.25, 0.25, 0.25]
INDUSTRY_VACANCY_RATE_ANNUAL = {
    1: 7.09,
    16: 5.48,
    12: 5.19,
    15: 5.14,
    17: 4.29,
    14: 3.81,
    4: 3.8,
    11: 3.78,
    9: 3.43,
    8: 3.43,
    10: 3.36,
    5: 2.83,
    7: 2.81,
    2: 2.59,
    18: 1.83,
    3: 1.56,
    13: 1.25,
}


def get_industry_skill_requirement(industry: int) -> int:
    probs = INDUSTRY_SKILL_DISTRIBUTION.get(industry, DEFAULT_SKILL_DISTRIBUTION)
    return np.random.choice([1, 2, 3, 4], p=probs)


FIRM_EXIT_PARAMS = {
    "empty_threshold_months": 6,
    "empty_exit_prob": 0.12,
    "struggling_threshold_months": 12,
    "struggling_ratio": 0.5,
    "struggling_exit_prob": 0.04,
    "base_exit_small": 0.004,
    "base_exit_medium": 0.0015,
    "base_exit_large": 0.0008,
    "small_threshold": 20,
    "medium_threshold": 100,
    "high_turnover_industries": [8, 16, 17],
    "high_turnover_multiplier": 1.3,
}
FIRM_ENTRY_PARAMS = {
    "replacement_rate": 1.5,
    "growth_rate": 0.015,
    "new_firm_size_distribution": {
        "micro": (1, 4, 0.4),
        "small": (5, 19, 0.35),
        "medium": (20, 99, 0.2),
        "large": (100, 500, 0.05),
    },
    "entry_size_multipliers": {
        "micro": (None, 0.8),
        "small": (1.0, 4.0),
        "medium": (4.0, 10.0),
        "large": (10.0, 20.0),
    },
    "initial_vacancies_per_firm": 3,
}
MIN_FIRMS_PER_INDUSTRY = 2


class DemographicTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        self.theoretical_deaths = 0
        self.theoretical_emigrants = 0
        self.theoretical_immigrants = 0
        self.theoretical_new_entrants = 0
        self.actual_replacements = 0
        self.replacement_sources = {"mortality": 0, "emigration": 0, "rebalance": 0}
        self.history = []

    def record_step(
        self,
        deaths: int,
        emigrants: int,
        immigrants: int,
        new_entrants: int,
        actual_replaced: int,
        source: str,
    ):
        self.theoretical_deaths += deaths
        self.theoretical_emigrants += emigrants
        self.theoretical_immigrants += immigrants
        self.theoretical_new_entrants += new_entrants
        self.actual_replacements += actual_replaced
        if source in self.replacement_sources:
            self.replacement_sources[source] += actual_replaced
        self.history.append(
            {
                "deaths": deaths,
                "emigrants": emigrants,
                "immigrants": immigrants,
                "new_entrants": new_entrants,
                "replaced": actual_replaced,
                "source": source,
            }
        )
        if len(self.history) > 100:
            self.history = self.history[-100:]


class LaborMarketEnv(EnvBase):

    def __init__(
        self, config, data_manager=None, device: str = None, batch_size: torch.Size = None
    ):
        from market_setup import configure, bind_environment

        configure(config)
        bind_environment(type(self))
        self.config = config
        self.device = device or config.training.device
        self._batch_size = batch_size or torch.Size([])
        super().__init__(device=self.device, batch_size=self._batch_size)
        self.state_manager = StateManager(config)
        self.matching_mechanism = MatchingMechanism(config.matching, device=self.device)
        from environment.demographic_flow_instrumenter import DemographicFlowInstrumenter

        self._net_population = None
        _net_projection_path = getattr(config.training, "renewal_net_projection", None)
        if _net_projection_path:
            from age_renewal import enabled as _age_enabled, controller_class

            if not _age_enabled():
                raise RuntimeError(
                    "net population projection requires historical demographic flows"
                )
            from .net_population_runtime import NetPopulationController

            _population_controller = (
                controller_class() if _age_enabled() else NetPopulationController
            )
            self._net_population = _population_controller(
                _net_projection_path, int(config.simulation.end_year)
            )
        self._renewal_carry = 0.0
        self._renewal_firm_baseline = None
        self._renewal_pending = None
        self._renewal_br_current_index = 1.0
        self._renewal_audit_created = set()
        self._renewal_audit_market_hired = {}
        self._renewal_audit_base_n = None
        self._renewal_audit_earlier_j2j_empty_registry = 0
        self._renewal_audit_earlier_j2j_workers = set()
        self._demographic_flow_instrumenter = DemographicFlowInstrumenter(enabled=False)
        self.state_manager._demographic_flow_instrumenter = self._demographic_flow_instrumenter
        self.retire_action_index = (
            2
            * config.action.worker_search_channel_options
            * config.action.worker_search_intensity_options
        )
        self._endogenous_retire_count_running = 0
        self._endogenous_retire_count_rollout = 0
        self._invalid_retire_action_count_running = 0
        self._invalid_retire_action_count_rollout = 0
        self._env_conversion_attempted_running = 0
        self._env_conversion_attempted_rollout = 0
        self._env_conversion_converted_running = 0
        self._env_conversion_converted_rollout = 0
        self._env_conversion_by_age_running = {}
        self._env_conversion_by_age_rollout = {}
        self.matching_mechanism.attachment_enabled = False
        self.matching_mechanism.attachment_alpha = float(
            getattr(config.training, "attachment_penalty_alpha", 0.3)
        )
        self.matching_mechanism.reservation_bundle_enabled = bool(
            getattr(config.training, "enable_reservation_wage_bundle", False)
        )
        self.matching_mechanism.reservation_decay_per_year = float(
            getattr(config.training, "reservation_decay_per_year", 0.0)
        )
        self.matching_mechanism.reservation_decay_compounded = bool(
            getattr(config.training, "reservation_decay_compounded", False)
        )
        if self.matching_mechanism.reservation_bundle_enabled:
            self.state_manager.backfill_reservation_bundle_state()
        self.data_manager = data_manager
        self._vacancy_id_counter = 0
        self.cma_to_das: Dict[int, List[int]] = {}
        if (
            data_manager
            and hasattr(data_manager, "da_centroids")
            and (data_manager.da_centroids is not None)
        ):
            self.matching_mechanism.load_da_centroids(data_manager.da_centroids)
            for _, row in data_manager.da_centroids.iterrows():
                cma = int(row["cma"])
                da_id = int(row["da_id"])
                if cma not in self.cma_to_das:
                    self.cma_to_das[cma] = []
                self.cma_to_das[cma].append(da_id)
        self.worker_ids: List[str] = []
        self.firm_ids: List[str] = []
        self.n_workers: int = config.simulation.n_workers
        self.n_firms: int = config.simulation.n_firms
        _max_eps_yrs = int(
            getattr(
                config.simulation,
                "max_episode_years",
                config.simulation.end_year - config.simulation.start_year + 1,
            )
        )
        self.episode_length = config.simulation.months_per_year * _max_eps_yrs
        self.current_step = 0
        self.episode_rewards = {"workers": [], "firms": []}
        self.last_matching_stats = {}
        self._firms_with_j2j_departure = set()
        self.last_firm_profit_raw = {}
        self.last_firm_wage_strategy = {}
        self._vac_baseline_accum = {
            "sep_total": 0,
            "baseline_sum": 0.0,
            "eff_lambda_sum": 0.0,
            "new_vac": 0,
        }
        self.start_year = config.simulation.start_year
        self.end_year = config.simulation.end_year
        self.current_year = self.start_year
        self.current_month = 1
        self._total_episodes = 0
        self._first_reset = True
        self._covid_regime = None
        self._last_covid_ledger = None
        self._covid_standard_driver_report = {}
        self._initialize_covid_validation()
        self._nilf_subtype_enabled = bool(getattr(config.mnl, "nilf_subtype_enabled", False))
        self._subtype_obs_dims = 7 if self._nilf_subtype_enabled else 0
        self._age_skill_mf_extra_dims = 0
        self._attachment_extra_dims = 0
        self._actual_worker_obs_dim = (
            config.environment.worker_state_dim
            + self._subtype_obs_dims
            + self._age_skill_mf_extra_dims
            + self._attachment_extra_dims
        )
        self.fast_obs = FastObservationGenerator(
            n_workers=self.n_workers,
            n_firms=self.n_firms,
            worker_obs_dim=self._actual_worker_obs_dim,
            firm_obs_dim=config.environment.firm_state_dim,
            n_visible_vacancies=config.environment.n_visible_vacancies,
            vacancy_dim=config.environment.vacancy_feature_dim,
            history_length=config.environment.history_length,
            history_dim=config.environment.history_dim,
            device=self.device,
            nilf_subtype_enabled=self._nilf_subtype_enabled,
            school_window=int(getattr(config.mnl, "school_search_eligible_months", 36)),
            allow_retirement_reentry=getattr(config.mnl, "allow_retirement_reentry", False),
            family_window=int(getattr(config.mnl, "family_search_eligible_months", 24)),
            illness_window=int(getattr(config.mnl, "illness_search_eligible_months", 6)),
            primeage_windows_active=False,
            primeage_age_lo=int(getattr(config.mnl, "nilf_window_primeage_age_lo", 35)),
            primeage_age_hi=int(getattr(config.mnl, "nilf_window_primeage_age_hi", 55)),
            family_window_primeage=int(
                getattr(config.mnl, "family_search_eligible_months_primeage", 48)
            ),
            illness_window_primeage=int(
                getattr(config.mnl, "illness_search_eligible_months_primeage", 36)
            ),
            age_skill_mean_field_enabled=False,
            attachment_enabled=False,
        )
        self.state_manager._state_table_ref = self.fast_obs.state_table
        self._worker_firm_map: Dict[str, str] = {}
        self._separation_rates_tensor: Optional[torch.Tensor] = None
        self._empirical_separation_rates_tensor: Optional[torch.Tensor] = None
        self._retirement_rates_tensor: Optional[torch.Tensor] = None
        self._wage_tensor: Optional[torch.Tensor] = None
        self._industry_to_idx: Dict[int, int] = {}
        self._year_to_idx: Dict[int, int] = {}
        self.mortality_rates = self._load_mortality_rates()
        self.firm_active: Optional[torch.Tensor] = None
        self.firm_months_empty: Optional[torch.Tensor] = None
        self.firm_months_struggling: Optional[torch.Tensor] = None
        self.firm_age_months: Optional[torch.Tensor] = None
        self.firm_employment: Optional[torch.Tensor] = None
        self.firm_target_size: Optional[torch.Tensor] = None
        self.firm_industries: Optional[torch.Tensor] = None
        self.firm_posted_wages: Optional[torch.Tensor] = None
        self.firm_da_ids: Optional[torch.Tensor] = None
        self.firm_exits_this_year = 0
        self.firm_entries_this_year = 0
        self.n_workers_fixed = config.simulation.n_workers
        self.full_population_df = None
        self.sampled_worker_ids: Set[str] = set()
        self.available_pool_ids: Set[str] = set()
        self._theoretical_population = self.n_workers_fixed
        self.demographic_tracker = None
        from market_mechanisms import ValidatedTradeWarRegime

        _trade_regime_type = ValidatedTradeWarRegime
        self._trade_war_regime = _trade_regime_type(getattr(config, "trade_war", None))
        self._make_specs()

    def _initialize_covid_validation(self) -> None:
        cfg = self.config.covid
        valid_modes = {"as_loaded", "realized", "frozen_2019"}
        if cfg.standard_driver_mode not in valid_modes:
            raise ValueError(
                f"covid.standard_driver_mode must be one of {sorted(valid_modes)}; got {cfg.standard_driver_mode!r}"
            )
        if not cfg.enabled:
            return
        driver_path = os.path.expanduser(str(cfg.driver_file))
        if not os.path.isabs(driver_path):
            driver_path = str(self.config.data.get_parameter_path(driver_path))
        table = CovidDriverTable.from_csv(driver_path)
        table.validate_coverage(
            range(int(cfg.shock_start_year), int(cfg.shock_end_year) + 1), range(1, 19)
        )
        self._covid_regime = CovidTransitionRegime(
            table, seed=int(cfg.rng_seed), release_year=int(cfg.release_year)
        )

    def _apply_covid_standard_driver_mode(self) -> None:
        mode = str(self.config.covid.standard_driver_mode)
        report = {"mode": mode, "anchor_year": None, "target_years": [], "surfaces": {}}
        if mode == "as_loaded":
            report["surfaces"] = {
                "separation_process": "configured fixed industry rates",
                "recession_multiplier": "as_loaded",
                "wage_level": "as_loaded",
                "wage_growth": "as_loaded",
                "labor_force_transition_tensor": "as_loaded_inactive",
            }
            self._covid_standard_driver_report = report
            return
        if self._empirical_separation_rates_tensor is None:
            raise RuntimeError("Empirical separation tensor was not initialized")
        empirical_years = [int(year) for year in sorted(self._year_to_idx) if int(year) >= 2017]
        for year in empirical_years:
            idx = self._year_to_idx[year]
            self._separation_rates_tensor[idx] = self._empirical_separation_rates_tensor[
                idx
            ].clone()
        if mode == "realized":
            report.update(
                {
                    "target_years": empirical_years,
                    "surfaces": {
                        "separation_process": "empirical year-specific level and layoff/quit mix from 2017",
                        "recession_multiplier": "neutral fallback after 2016; no arm contrast",
                        "wage_level": "no forward rows; no arm contrast",
                        "wage_growth": "fallback constants; no arm contrast",
                        "labor_force_transition_tensor": "as_loaded_inactive",
                    },
                }
            )
            self._covid_standard_driver_report = report
            return
        anchor_year = 2019
        if anchor_year not in self._year_to_idx:
            raise RuntimeError(
                "frozen_2019 requested, but the loaded separation/transition extension has no 2019 row"
            )
        anchor_idx = self._year_to_idx[anchor_year]
        target_years = [int(year) for year in sorted(self._year_to_idx) if int(year) >= 2020]
        for year in target_years:
            idx = self._year_to_idx[year]
            self._separation_rates_tensor[idx] = self._separation_rates_tensor[anchor_idx].clone()
        report.update(
            {
                "anchor_year": anchor_year,
                "target_years": target_years,
                "surfaces": {
                    "separation_process": "empirical 2019 level and layoff/quit mix repeated from 2020",
                    "recession_multiplier": "neutral fallback after 2016; no arm contrast",
                    "wage_level": "no forward rows; no arm contrast",
                    "wage_growth": "fallback constants; no arm contrast",
                    "labor_force_transition_tensor": "as_loaded_inactive",
                },
            }
        )
        self._covid_standard_driver_report = report

    def _covid_locked_mask(self) -> torch.Tensor:
        if self._covid_regime is None:
            return torch.zeros(self.n_workers, dtype=torch.bool, device=self.device)
        return torch.tensor(
            [self.state_manager.workers[wid].covid_state != COVID_NONE for wid in self.worker_ids],
            dtype=torch.bool,
            device=self.device,
        )

    def _covid_firm_is_active(self, firm_id: Optional[str]) -> bool:
        if not firm_id or firm_id not in self.state_manager.firms:
            return False
        idx = getattr(self, "_firm_id_to_idx", {}).get(firm_id)
        if idx is None or self.firm_active is None:
            return True
        return bool(self.firm_active[idx].item())

    def _covid_reserved_positions(self, firm_id: str) -> int:
        if self._covid_regime is None:
            return 0
        return sum(
            (
                1
                for ws in self.state_manager.workers.values()
                if ws.covid_state == COVID_TEMP_LAYOFF and ws.covid_attached_employer_id == firm_id
            )
        )

    def _apply_covid_month(self) -> None:
        if self._covid_regime is None:
            self._last_covid_ledger = None
            return
        year = int(self.state_manager.current_year)
        month = int(self.state_manager.current_month)
        has_locked_workers = any(
            (ws.covid_state != COVID_NONE for ws in self.state_manager.workers.values())
        )
        is_release_month = year == int(self.config.covid.release_year) and month == 1
        if not (
            self._covid_regime.drivers.month_is_active(year, month)
            or has_locked_workers
            or is_release_month
        ):
            self._last_covid_ledger = None
            return
        self._last_covid_ledger = self._covid_regime.apply_month(
            self.state_manager, year, month, self._covid_firm_is_active
        )
        self._worker_firm_map.clear()
        for firm_id, firm_state in self.state_manager.firms.items():
            for worker_id in firm_state.employees:
                self._worker_firm_map[worker_id] = firm_id
        self._sync_firm_tensors()
        self.fast_obs.state_table.sync_from_state_manager(self.state_manager, self.worker_ids)
        self.state_manager._update_aggregate_stats()

    def get_covid_validation_metadata(self) -> Dict[str, Any]:
        cfg = self.config.covid
        metadata = {
            "arm_id": cfg.arm_id,
            "covid_enabled": bool(cfg.enabled),
            "standard_driver_mode": cfg.standard_driver_mode,
            "covid_rng_seed": int(cfg.rng_seed),
            "shock_window": [int(cfg.shock_start_year), int(cfg.shock_end_year)],
            "release_year": int(cfg.release_year),
            "benchmark_cmas": [int(value) for value in cfg.benchmark_cmas],
            "benchmark_geography": "Toronto CMA 535 and Hamilton CMA 537; Oshawa CMA 532 remains in the simulated economy but is excluded from LFS fit because the public PUMF pools it with other CMA/non-CMA areas",
            "standard_driver_application": dict(self._covid_standard_driver_report),
            "estimand": "2017-2019 pre-pandemic temporal holdout under observed annual separations; 2020-2021 conditional historical reconstruction/stress test; 2022-2023 recovery conditional on the injected shock",
        }
        if self._covid_regime is not None:
            metadata.update(self._covid_regime.drivers.metadata())
        return metadata

    def get_covid_validation_snapshot(self, year: int, month: int) -> Dict[str, Any]:
        age_bins = (
            ("15-24", 15, 24),
            ("25-34", 25, 34),
            ("35-44", 35, 44),
            ("45-54", 45, 54),
            ("55-64", 55, 64),
            ("65+", 65, 200),
        )
        aggregate = {"E": 0, "U_lfs": 0, "N_lfs": 0, "temp_layoff": 0, "marginal_attached": 0}
        by_industry = {}
        by_age = {name: dict(aggregate) for name, _, _ in age_bins}
        benchmark_cmas = {int(value) for value in self.config.covid.benchmark_cmas}

        def _bucket(worker_state):
            worker = worker_state.worker
            if worker_state.covid_state == COVID_TEMP_LAYOFF:
                return "U_lfs"
            if worker.employment_status == "Employed":
                return "E"
            if worker.employment_status == "Unemployed" and worker.is_searching:
                return "U_lfs"
            return "N_lfs"

        for ws in self.state_manager.workers.values():
            worker = ws.worker
            if int(worker.cma) not in benchmark_cmas:
                continue
            bucket = _bucket(ws)
            aggregate[bucket] += 1
            if ws.covid_state == COVID_TEMP_LAYOFF:
                aggregate["temp_layoff"] += 1
            elif ws.covid_state == COVID_MARGINAL_ATTACHED:
                aggregate["marginal_attached"] += 1
            ind_key = str(int(worker.industry))
            ind = by_industry.setdefault(ind_key, dict(aggregate.fromkeys(aggregate, 0)))
            ind[bucket] += 1
            if ws.covid_state == COVID_TEMP_LAYOFF:
                ind["temp_layoff"] += 1
            elif ws.covid_state == COVID_MARGINAL_ATTACHED:
                ind["marginal_attached"] += 1
            age = int(worker.age)
            for name, lower, upper in age_bins:
                if lower <= age <= upper:
                    age_row = by_age[name]
                    age_row[bucket] += 1
                    if ws.covid_state == COVID_TEMP_LAYOFF:
                        age_row["temp_layoff"] += 1
                    elif ws.covid_state == COVID_MARGINAL_ATTACHED:
                        age_row["marginal_attached"] += 1
                    break

        def _with_rates(counts):
            counts = dict(counts)
            labour_force = counts["E"] + counts["U_lfs"]
            population = labour_force + counts["N_lfs"]
            counts.update(
                {
                    "population": population,
                    "unemployment_rate_lfs": (
                        counts["U_lfs"] / labour_force if labour_force else None
                    ),
                    "employment_rate": counts["E"] / population if population else None,
                    "participation_rate_lfs": labour_force / population if population else None,
                    "temp_layoff_share_population": (
                        counts["temp_layoff"] / population if population else None
                    ),
                    "marginal_share_population": (
                        counts["marginal_attached"] / population if population else None
                    ),
                }
            )
            return counts

        snapshot = {
            "year": int(year),
            "month": int(month),
            "aggregate": _with_rates(aggregate),
            "by_industry": {key: _with_rates(value) for key, value in sorted(by_industry.items())},
            "by_age": {key: _with_rates(value) for key, value in by_age.items()},
        }
        ledger = self._last_covid_ledger
        if ledger is not None and ledger.year == int(year) and (ledger.month == int(month)):
            snapshot["flow_ledger"] = ledger.to_dict()
        if self._covid_regime is not None and self._covid_regime.drivers.has_month(year, month):
            row = self._covid_regime.drivers.require(year, month, 1)
            snapshot["driver"] = {
                "regime_active": bool(row.regime_active),
                "stringency_index": row.stringency_index,
                "shock_intensity": row.shock_intensity,
                "target_excess_temp_layoff_share": row.target_excess_temp_layoff_share,
                "target_excess_marginal_share": row.target_excess_marginal_share,
            }
        return snapshot

    def _make_specs(self):
        env_cfg = self.config.environment
        action_cfg = self.config.action
        worker_obs_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self.n_workers, self._actual_worker_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        firm_obs_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self.n_firms, env_cfg.firm_state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        global_state_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(getattr(env_cfg, "effective_global_state_dim", env_cfg.global_state_dim),),
            dtype=torch.float32,
            device=self.device,
        )
        worker_action_spec = DiscreteTensorSpec(
            n=action_cfg.worker_action_size,
            shape=(self.n_workers, 2),
            dtype=torch.int64,
            device=self.device,
        )
        firm_action_spec = DiscreteTensorSpec(
            n=action_cfg.firm_action_size,
            shape=(self.n_firms,),
            dtype=torch.int64,
            device=self.device,
        )
        self.observation_spec = CompositeSpec(
            worker_obs=worker_obs_spec,
            firm_obs=firm_obs_spec,
            global_state=global_state_spec,
            shape=self._batch_size,
        )
        self.action_spec = CompositeSpec(
            worker_action=worker_action_spec, firm_action=firm_action_spec, shape=self._batch_size
        )
        self.reward_spec = CompositeSpec(
            worker_reward=UnboundedContinuousTensorSpec(
                shape=(self.n_workers,), dtype=torch.float32, device=self.device
            ),
            firm_reward=UnboundedContinuousTensorSpec(
                shape=(self.n_firms,), dtype=torch.float32, device=self.device
            ),
            shape=self._batch_size,
        )
        self.done_spec = CompositeSpec(
            done=DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            terminated=DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            truncated=DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool, device=self.device),
            shape=self._batch_size,
        )

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            import random as _py_random

            _py_random.seed(seed)

    def _build_gpu_caches(self):
        self._worker_firm_map.clear()
        for firm_id, fs in self.state_manager.firms.items():
            for worker_id in fs.employees:
                self._worker_firm_map[worker_id] = firm_id
        sep_rates = self.data_manager.data.separation_rates
        industries = sorted(sep_rates["naics_18"].unique())
        self._industry_to_idx = {ind: idx for idx, ind in enumerate(industries)}
        n_industries = len(industries)
        years = sorted(sep_rates["year"].unique())
        self._year_to_idx = {yr: idx for idx, yr in enumerate(years)}
        n_years = len(years)
        sep_multiplier = getattr(self.config.mnl, "separation_rate_multiplier", 2.5)
        industry_sep_rates = getattr(self.config.mnl, "industry_separation_rates", {})
        sep_tensor = torch.zeros(n_years, n_industries, 2, device=self.device)
        empirical_sep_tensor = torch.zeros_like(sep_tensor)
        for _, row in sep_rates.iterrows():
            yr_idx = self._year_to_idx[int(row["year"])]
            naics_code = int(row["naics_18"])
            ind_idx = self._industry_to_idx.get(naics_code, 0)
            if naics_code in industry_sep_rates:
                sep_rate = industry_sep_rates[naics_code]
            else:
                sep_rate = row["separation_rate"] / 12 * sep_multiplier
            layoff_share = (
                row["layoff_rate"] / row["separation_rate"] if row["separation_rate"] > 0 else 0.5
            )
            sep_tensor[yr_idx, ind_idx, 0] = sep_rate
            sep_tensor[yr_idx, ind_idx, 1] = layoff_share
            empirical_sep_tensor[yr_idx, ind_idx, 0] = float(row["separation_rate"]) / 12.0
            empirical_sep_tensor[yr_idx, ind_idx, 1] = layoff_share
        self._separation_rates_tensor = sep_tensor
        self._empirical_separation_rates_tensor = empirical_sep_tensor
        _industries_using_fallback = [
            int(row["naics_18"])
            for _, row in sep_rates.iterrows()
            if int(row["naics_18"]) not in industry_sep_rates
        ]
        ret_rates = self.data_manager.data.retirement_rates
        max_age = int(ret_rates["age"].max()) + 1
        ret_tensor = torch.zeros(max_age, device=self.device)
        for _, row in ret_rates.iterrows():
            age = int(row["age"])
            if age < max_age:
                ret_tensor[age] = row["retirement_rate"] / 12
        self._retirement_rates_tensor = ret_tensor
        lf_trans = self.data_manager.data.labor_force_transitions
        age_groups = [
            "15_19",
            "20_24",
            "25_29",
            "30_34",
            "35_39",
            "40_44",
            "45_49",
            "50_54",
            "55_59",
            "60_64",
            "65_69",
            "70_plus",
        ]
        self._age_group_to_idx = {ag: i for i, ag in enumerate(age_groups)}
        n_age_groups = len(age_groups)
        lf_trans_tensor = torch.zeros((n_years, n_age_groups, 3), device=self.device)
        for _, row in lf_trans.iterrows():
            yr_idx = self._year_to_idx.get(int(row["year"]), -1)
            ag_idx = self._age_group_to_idx.get(row["age_group"], -1)
            if yr_idx >= 0 and ag_idx >= 0:
                lf_trans_tensor[yr_idx, ag_idx, 0] = row["discouragement_rate"] / 12
                lf_trans_tensor[yr_idx, ag_idx, 1] = row["retirement_share"] / 12
                if ag_idx <= 1:
                    lf_trans_tensor[yr_idx, ag_idx, 2] = 0.004
                elif ag_idx <= 3:
                    lf_trans_tensor[yr_idx, ag_idx, 2] = 0.003
                elif ag_idx <= 7:
                    lf_trans_tensor[yr_idx, ag_idx, 2] = 0.002
                elif ag_idx <= 9:
                    lf_trans_tensor[yr_idx, ag_idx, 2] = 0.001
                else:
                    lf_trans_tensor[yr_idx, ag_idx, 2] = 0.0005
        self._lf_transitions_tensor = lf_trans_tensor
        _earlier_nilf_rate = float(getattr(self.config.mnl, "nilf_to_u_rate", 0.0))
        if _earlier_nilf_rate != 0.0:
            logger.warning(
                f"nilf_to_u_rate is {_earlier_nilf_rate}. The setting is not used and should be 0.0."
            )
        _earlier_sep_mult = float(getattr(self.config.mnl, "separation_rate_multiplier", 1.0))
        _ta_enabled = bool(getattr(self.config.target_adaptation, "enabled", False))
        assert _earlier_nilf_rate == 0.0, f"nilf_to_u_rate must be 0.0; got {_earlier_nilf_rate}"
        assert (
            _earlier_sep_mult == 1.0
        ), f"separation_rate_multiplier must be 1.0 for the fallback path; got {_earlier_sep_mult}"
        assert _ta_enabled is False, f"target_adaptation.enabled must be False; got {_ta_enabled}"
        wage_params = self.data_manager.data.wage_parameters
        wage_tensor = torch.full((n_years, n_industries), 20.0, device=self.device)
        for _, row in wage_params.iterrows():
            yr_idx = self._year_to_idx.get(int(row["year"]), 0)
            ind_idx = self._industry_to_idx.get(int(row["naics_18"]), 0)
            wage_tensor[yr_idx, ind_idx] = row["avg_hourly_wage"]
        self._wage_tensor = wage_tensor
        self._wage_growth_by_year = {}
        for _, row in wage_params.iterrows():
            yr = int(row["year"])
            ind = int(row["naics_18"])
            rate = row.get("wage_growth_rate", float("nan"))
            if yr not in self._wage_growth_by_year:
                self._wage_growth_by_year[yr] = {}
            if not rate != rate:
                self._wage_growth_by_year[yr][ind] = float(rate)
        recession_mult = self.data_manager.data.recession_multipliers
        self._recession_tensor = torch.ones(n_years, device=self.device)
        for _, row in recession_mult.iterrows():
            yr_idx = self._year_to_idx.get(int(row["year"]), -1)
            if yr_idx >= 0:
                self._recession_tensor[yr_idx] = row["separation_multiplier"]
        self._apply_covid_standard_driver_mode()

    def _initialize_firm_tensors(self):
        _correct_employees = {fid: [] for fid in self.state_manager.firms}
        for wid, ws in self.state_manager.workers.items():
            if ws.worker.employment_status == "Employed":
                employer = getattr(ws.worker, "current_employer_id", None)
                if employer and employer in _correct_employees:
                    _correct_employees[employer].append(wid)
        for fid, fs in self.state_manager.firms.items():
            fs.employees = _correct_employees[fid]
        n_firms = len(self.firm_ids)
        device = self.device
        self.firm_active = torch.ones(n_firms, dtype=torch.bool, device=device)
        self.firm_months_empty = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self.firm_months_struggling = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self.firm_age_months = torch.zeros(n_firms, dtype=torch.int32, device=device)
        ta = self.config.target_adaptation
        self._ta_under_streak = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self._ta_lowfill_streak = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self._ta_over_streak = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self._ta_prev_total_hires = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self._ta_fill_window = ta.fill_rate_window
        self._ta_hires_ring = torch.zeros(
            n_firms, self._ta_fill_window, dtype=torch.int32, device=device
        )
        self._ta_ring_idx = 0
        self._q_accum = self._make_quarterly_accum()
        self.firm_employment = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self.firm_target_size = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self.firm_industries = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self.firm_posted_wages = torch.zeros(n_firms, dtype=torch.float32, device=device)
        self.firm_da_ids = torch.zeros(n_firms, dtype=torch.int32, device=device)
        self._firm_id_to_idx = {fid: idx for idx, fid in enumerate(self.firm_ids)}
        import math

        avg_firm_size = max(1.0, self.config.simulation.n_workers / n_firms)
        max_target_size = math.ceil(10.0 * avg_firm_size)
        for i, firm_id in enumerate(self.firm_ids):
            fs = self.state_manager.firms.get(firm_id)
            if fs is None:
                continue
            firm = fs.firm
            n_employees = len(fs.employees)
            self.firm_employment[i] = n_employees
            raw_target = getattr(firm, "target_size", max(1, n_employees))
            p2_cap = max(40, n_employees)
            effective_cap = min(max_target_size, p2_cap)
            if raw_target > effective_cap:
                raw_target = effective_cap
                firm.target_size = effective_cap
            if n_employees > 0:
                _INDUSTRY_SEP_RATES = {
                    1: 0.184,
                    2: 0.245,
                    3: 0.116,
                    4: 0.194,
                    5: 0.165,
                    7: 0.147,
                    8: 0.274,
                    9: 0.156,
                    10: 0.133,
                    11: 0.164,
                    12: 0.291,
                    13: 0.14,
                    14: 0.129,
                    15: 0.234,
                    16: 0.361,
                    17: 0.2,
                    18: 0.115,
                }
                firm_industry = int(firm.industry) if hasattr(firm, "industry") else 0
                sep_rate = _INDUSTRY_SEP_RATES.get(firm_industry, 0.19)
                attrition_factor = max(0.5, 1 - 2 * sep_rate)
                raw_target = max(raw_target, max(3, int(n_employees * attrition_factor)))
            self.firm_target_size[i] = raw_target
            self.firm_industries[i] = int(firm.industry) if hasattr(firm, "industry") else 0
            self.firm_posted_wages[i] = firm.posted_wage if hasattr(firm, "posted_wage") else 20.0
            if hasattr(firm, "da_id") and firm.da_id is not None:
                self.firm_da_ids[i] = int(firm.da_id)
        aggregate_target = float(self.firm_target_size.sum().item())
        aggregate_cap = 0.7 * self.config.simulation.n_workers
        if aggregate_target > aggregate_cap:
            scale = aggregate_cap / aggregate_target
            scaled = torch.clamp((self.firm_target_size.float() * scale).round().int(), min=1)
            for i, firm_id in enumerate(self.firm_ids):
                fs = self.state_manager.firms.get(firm_id)
                if fs is not None:
                    fs.firm.target_size = int(scaled[i].item())
            self.firm_target_size = scaled
        from employment_initialization import capacity as reconcile_initial_capacity

        reconcile_initial_capacity(self)
        self.firm_base_size = self.firm_target_size.clone()
        self.firm_exits_this_year = 0
        self.firm_entries_this_year = 0
        self._ensure_industry_coverage()

    def _ensure_industry_coverage(self):
        if self.firm_industries is None or not hasattr(self, "firm_active"):
            return
        industries_missing = [
            ind
            for ind in sorted(VALID_INDUSTRIES)
            if int(((self.firm_industries == ind) & self.firm_active).sum())
            < MIN_FIRMS_PER_INDUSTRY
        ]
        if industries_missing:
            logger.warning(
                f"Industries below the minimum firm count: {industries_missing}. No firms are added."
            )

    def _sync_firm_tensors(self):
        if self.firm_employment is None:
            return
        correct = {fid: [] for fid in self.state_manager.firms}
        for wid, ws in self.state_manager.workers.items():
            if ws.worker.employment_status == "Employed":
                employer = getattr(ws.worker, "current_employer_id", None)
                if employer and employer in correct:
                    correct[employer].append(wid)
        for fid, fs in self.state_manager.firms.items():
            fs.employees = correct[fid]
        for i, firm_id in enumerate(self.firm_ids):
            if not self.firm_active[i]:
                continue
            fs = self.state_manager.firms.get(firm_id)
            if fs is not None:
                self.firm_employment[i] = self._workplace_staff_count(firm_id, fs)
            else:
                self.firm_employment[i] = 0

    def _initialize_demographics_from_2006_sample(self) -> None:
        self.state_manager = StateManager(self.config)
        if hasattr(self, "_demographic_flow_instrumenter"):
            self.state_manager._demographic_flow_instrumenter = self._demographic_flow_instrumenter
        self.state_manager.initialize_from_data(
            population_df=self.data_manager.sampled_population,
            firms_df=self.data_manager.firms,
            initial_vacancies=None,
            full_firms_df=self.data_manager.full_firms_pool,
        )
        from employment_initialization import initialize as initialize_employment

        initialize_employment(self)
        if self.matching_mechanism.reservation_bundle_enabled:
            self.state_manager.backfill_reservation_bundle_state()
        if getattr(self.config.mnl, "nilf_subtype_enabled", False):
            _nilf_rng = None
            import numpy as _np

            _nilf_rng = _np.random.default_rng(seed=int(_np.random.randint(0, 2**31 - 1)))
            self.state_manager.setup_nilf_subtype_sampler(
                self.data_manager.data.nilf_composition_by_sex, rng=_nilf_rng
            )
            self.state_manager.assign_initial_nilf_subtypes()
        wage_params = self.data_manager.data.wage_parameters
        start_year = self.config.simulation.start_year
        industry_wages = {}
        for _, row in wage_params.iterrows():
            if int(row["year"]) == start_year:
                industry = int(row["naics_18"])
                industry_wages[industry] = row["avg_hourly_wage"]
        self.state_manager.initialize_firm_wages(industry_wages)
        if self.cma_to_das:
            import random

            for firm_id, fs in self.state_manager.firms.items():
                if fs.firm.da_id is None and fs.firm.cma in self.cma_to_das:
                    fs.firm.da_id = random.choice(self.cma_to_das[fs.firm.cma])
        _cent = getattr(self.matching_mechanism, "da_centroids", None)
        if _cent:
            _missing_workers = 0
            _missing_firms = 0
            _sample_missing_das = set()
            for _ws in self.state_manager.workers.values():
                _wd = _ws.worker.da_id
                if _wd is not None and int(_wd) not in _cent:
                    _missing_workers += 1
                    if len(_sample_missing_das) < 10:
                        _sample_missing_das.add(int(_wd))
            for _fs in self.state_manager.firms.values():
                _fd = _fs.firm.da_id
                if _fd is not None and int(_fd) not in _cent:
                    _missing_firms += 1
                    if len(_sample_missing_das) < 10:
                        _sample_missing_das.add(int(_fd))
            if _missing_workers > 0 or _missing_firms > 0:
                raise AssertionError(
                    f"{_missing_workers} workers and {_missing_firms} firms have a da_id that is absent from the {len(_cent)} centroids, so their distances cannot be looked up. Examples: {sorted(_sample_missing_das)}. Add the areas to the centroid file or correct the worker and firm files."
                )
        self.state_manager.assign_worker_types()
        self.state_manager.assign_bounded_rationality_skills(
            self.config.mean_field.skill_distributions
        )
        self.state_manager.compute_worker_type_mean_fields()
        initial_theta = self.state_manager.compute_firm_mean_field()
        if self.config.mean_field.enabled:
            self.state_manager.freeze_theta(initial_theta)
        initial_vacancies = self._generate_initial_vacancies()
        self.state_manager.add_vacancies(initial_vacancies)
        self._validate_worker_industries()
        self.worker_ids = list(self.state_manager.workers.keys())
        self.firm_ids = list(self.state_manager.firms.keys())
        self.n_workers = len(self.worker_ids)
        self.n_firms = len(self.firm_ids)
        self._make_specs()
        self.fast_obs = FastObservationGenerator(
            n_workers=self.n_workers,
            n_firms=self.n_firms,
            worker_obs_dim=self._actual_worker_obs_dim,
            firm_obs_dim=self.config.environment.firm_state_dim,
            n_visible_vacancies=self.config.environment.n_visible_vacancies,
            vacancy_dim=self.config.environment.vacancy_feature_dim,
            history_length=self.config.environment.history_length,
            history_dim=self.config.environment.history_dim,
            device=self.device,
            nilf_subtype_enabled=self._nilf_subtype_enabled,
            school_window=int(getattr(self.config.mnl, "school_search_eligible_months", 36)),
            allow_retirement_reentry=getattr(self.config.mnl, "allow_retirement_reentry", False),
            family_window=int(getattr(self.config.mnl, "family_search_eligible_months", 24)),
            illness_window=int(getattr(self.config.mnl, "illness_search_eligible_months", 6)),
            primeage_windows_active=False,
            primeage_age_lo=int(getattr(self.config.mnl, "nilf_window_primeage_age_lo", 35)),
            primeage_age_hi=int(getattr(self.config.mnl, "nilf_window_primeage_age_hi", 55)),
            family_window_primeage=int(
                getattr(self.config.mnl, "family_search_eligible_months_primeage", 48)
            ),
            illness_window_primeage=int(
                getattr(self.config.mnl, "illness_search_eligible_months_primeage", 36)
            ),
            age_skill_mean_field_enabled=False,
            attachment_enabled=False,
        )
        self.state_manager._state_table_ref = self.fast_obs.state_table
        if self.full_population_df is None:
            self.full_population_df = self.data_manager.data.population
        self.sampled_worker_ids = set(self.worker_ids)
        if self.full_population_df is not None and "worker_id" in self.full_population_df.columns:
            all_worker_ids = set(self.full_population_df["worker_id"].astype(str))
            self.available_pool_ids = all_worker_ids - self.sampled_worker_ids
        else:
            self.available_pool_ids = set()
        if self.demographic_tracker is None:
            self.demographic_tracker = DemographicTracker()
        self._initialize_firm_tensors()
        self._trade_war_regime.reset(self)

    def _reset(self, tensordict: TensorDict = None) -> TensorDict:
        if not hasattr(self, "_first_reset"):
            self._first_reset = True
            self._total_episodes = 0
        self.current_step = 0
        self.current_year = self.start_year
        self.current_month = 1
        self._renewal_br_current_index = 1.0
        if self._first_reset:
            if self.data_manager is None:
                from data.data_loader import DataManager

                self.data_manager = DataManager(self.config).load().prepare()
            if (
                not self.matching_mechanism.da_centroids
                and hasattr(self.data_manager, "data")
                and hasattr(self.data_manager.data, "da_centroids")
                and (self.data_manager.data.da_centroids is not None)
            ):
                self.matching_mechanism.load_da_centroids(self.data_manager.data.da_centroids)
                for _, row in self.data_manager.data.da_centroids.iterrows():
                    cma = int(row["cma"])
                    da_id = int(row["da_id"])
                    if cma not in self.cma_to_das:
                        self.cma_to_das[cma] = []
                    self.cma_to_das[cma].append(da_id)
            self._initialize_demographics_from_2006_sample()
            self._verify_population_integrity("first reset initialization")
            self._first_reset = False
        else:
            self._verify_population_integrity(
                f"PEB reset episode {self._total_episodes} (pre-snap)"
            )
            self._initialize_demographics_from_2006_sample()
            self._verify_population_integrity(
                f"PEB reset episode {self._total_episodes} (post-snap)"
            )
        self._build_gpu_caches()
        self.fast_obs.initialize(self.state_manager, self.worker_ids, self.firm_ids)
        self.current_step = 0
        self.current_year = self.start_year
        self._renewal_carry = 0.0
        if self._net_population is not None:
            self._net_population.reset()
        self._renewal_firm_baseline = None
        self._renewal_pending = None
        self._renewal_br_current_index = 1.0
        self._renewal_audit_created = set()
        self._renewal_audit_market_hired = {}
        self._renewal_audit_base_n = len(self.state_manager.workers)
        self._renewal_audit_earlier_j2j_empty_registry = 0
        self._renewal_audit_earlier_j2j_workers = set()
        self.current_month = 1
        self._total_episodes += 1
        self._trade_observed_stocks = (
            MonthlyModelStocks(self.start_year) if self._trade_war_regime.enabled else None
        )
        self.episode_rewards = {"workers": [], "firms": []}
        if hasattr(self, "matching_mechanism"):
            self.matching_mechanism.reset_stats()
        import capacity_lineage as _capacity_lineage

        _capacity_lineage.initialize_lineages(self)
        return self._get_observation_tensordict()

    def step(self, tensordict: TensorDict) -> TensorDict:
        obs_td = self._step(tensordict)
        done = self.current_step >= self.episode_length
        obs_td["terminated"] = torch.tensor([False], dtype=torch.bool, device=self.device)
        obs_td["truncated"] = torch.tensor([done], dtype=torch.bool, device=self.device)
        obs_td["done"] = torch.tensor([done], dtype=torch.bool, device=self.device)
        return obs_td

    def _endogenous_retirement_enabled(self) -> bool:
        return True

    def _compute_retire_eligible_mask(self) -> torch.Tensor:
        from environment.nilf_subtype import SUBTYPE_RETIREMENT

        st = self.fast_obs.state_table
        age_min = int(self.config.reward.retirement_action_age_min)
        age_ok = st.w_age.to(self.device) >= age_min
        status = st.w_status.to(self.device)
        is_searching = st.w_is_searching.to(self.device)
        status_ok = (status == 0) | (status == 1) & is_searching
        subtype = st.w_nilf_subtype.to(self.device)
        subtype_ok = subtype != SUBTYPE_RETIREMENT
        return age_ok & status_ok & subtype_ok & ~self._covid_locked_mask()

    def _process_endogenous_retirement_actions(
        self,
        worker_actions_tensor: torch.Tensor,
        retire_eligible_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        composite_actions = worker_actions_tensor[..., 0]
        is_retire_action = composite_actions == int(self.retire_action_index)
        n_retire_emitted = int(is_retire_action.sum().item())
        if n_retire_emitted == 0:
            return worker_actions_tensor
        if retire_eligible_mask is None:
            raise RuntimeError(
                f"{n_retire_emitted} workers chose the retirement action (action {int(self.retire_action_index)}) but the action data carry no eligibility mask. The mask must be the one the policy saw when it sampled, stored under worker_retire_eligible_mask."
            )
        eligible = retire_eligible_mask.to(torch.bool)
        valid_retire_mask = is_retire_action & eligible
        invalid_retire_mask = is_retire_action & ~eligible
        n_valid = int(valid_retire_mask.sum().item())
        n_invalid = int(invalid_retire_mask.sum().item())
        if n_valid > 0:
            valid_indices = torch.nonzero(valid_retire_mask, as_tuple=True)[0].cpu().tolist()
            _flow_inst = getattr(self, "_demographic_flow_instrumenter", None)
            _flow_enabled = _flow_inst is not None and _flow_inst.enabled
            st = self.fast_obs.state_table
            for idx in valid_indices:
                wid = self.worker_ids[idx]
                ws = self.state_manager.workers.get(wid)
                if ws is None:
                    continue
                _origin_status = ws.worker.employment_status
                if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
                    self, "_person_trace_active", False
                ):
                    self._person_trace_retirement_origins[wid] = _origin_status
                if _flow_enabled:
                    _retiree_age = int(getattr(ws.worker, "age", -1))
                    _retiree_imm_raw = getattr(ws.worker, "immigrant_status", False)
                    if isinstance(_retiree_imm_raw, str):
                        _retiree_is_imm = _retiree_imm_raw.lower() in (
                            "true",
                            "1",
                            "yes",
                            "immigrant",
                        )
                    else:
                        _retiree_is_imm = bool(_retiree_imm_raw)
                    _flow_inst.record_retirement(
                        age=_retiree_age, pathway="retirement_action", is_immigrant=_retiree_is_imm
                    )
                    _flow_inst.record_worker_lf_event(
                        worker_id=wid,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="retire_action",
                        lf_delta=-1,
                        age=_retiree_age,
                    )
                emp_id = getattr(ws.worker, "current_employer_id", None)
                if emp_id and emp_id in self.state_manager.firms:
                    try:
                        self.state_manager.firms[emp_id].employees.remove(wid)
                    except (ValueError, KeyError):
                        pass
                ws.worker.employment_status = "NILF"
                if _origin_status == "Employed":
                    self._retirements_from_employment_this_step += 1
                elif _origin_status == "Unemployed":
                    self._retirements_from_unemployment_this_step += 1
                ws.worker.current_employer_id = None
                if wid in self._worker_firm_map:
                    del self._worker_firm_map[wid]
                self.state_manager.set_retirement_nilf_subtype(
                    ws, source_state="endogenous_retirement_action"
                )
            st.w_status[valid_retire_mask] = 2
            if hasattr(st, "w_is_searching"):
                st.w_is_searching[valid_retire_mask] = False
            self._endogenous_retire_count_running += n_valid
            self._endogenous_retire_count_rollout += n_valid
        if n_invalid > 0:
            self._invalid_retire_action_count_running += n_invalid
            self._invalid_retire_action_count_rollout += n_invalid
            if self.current_step % 10 == 0 or n_invalid >= 5:
                logger.warning(
                    f"Step {self.current_step}: {n_invalid} workers chose the retirement action without being eligible. Their action was set to no search."
                )
        patched = worker_actions_tensor.clone()
        patched[..., 0] = torch.where(
            is_retire_action, torch.zeros_like(patched[..., 0]), patched[..., 0]
        )
        return patched

    def _step(self, tensordict: TensorDict) -> TensorDict:
        from market_mechanisms import phase as _market_phase

        _market_phase(self, "start_month_after_calendar_and_demographics")
        if self._net_population is not None:
            self._net_population.assert_step_ready()
        n_state = len(self.state_manager.workers)
        if not hasattr(self, "_last_worker_count"):
            self._last_worker_count = n_state
            self._initial_worker_count = n_state
        self._last_worker_count = n_state
        self.state_manager.stats["matches_this_month"] = 0
        self.state_manager.stats["u2e_matches_this_month"] = 0
        self.state_manager.stats["separations_this_month"] = 0
        self.state_manager.start_vacancy_ledger_month()
        import capacity_budget as _capacity_budget

        self._capacity_commit_rejections_this_step = {"u2e": 0, "j2j": 0, "direct_nilf": 0}
        self._capacity_commit_rejection_events_this_step = []
        self._employment_route_ids_this_step = {
            "ordinary_job_loss_separations": set(),
            "ordinary_retirement_separations": set(),
            "firm_exit_displacements": set(),
            "overstaff_purge_displacements": set(),
            "same_month_rehires": set(),
        }
        self._retirements_from_employment_this_step = 0
        self._retirements_from_unemployment_this_step = 0
        self._vacancy_debug = {
            "step": int(self.current_step),
            "sample_firm_ids": self._sample_firm_ids_for_debug(k=5),
            "firms": {},
            "backstop_added": {"industry_floor": 0, "replenish": 0},
        }
        for fid in self._vacancy_debug["sample_firm_ids"]:
            vc_count = self._get_firm_vacancy_count(fid)
            if vc_count < 0:
                continue
            self._vacancy_debug["firms"][fid] = {
                "action": None,
                "vac_before": vc_count,
                "vac_after_rl": None,
                "vac_after_backstop": None,
                "backstop_added": None,
            }
        self._trade_war_regime.apply_month(self)
        _market_phase(self, "environment_value_retirement_conversion")
        if self._endogenous_retirement_enabled():
            _retire_mask = tensordict.get("worker_retire_eligible_mask", None)
            if _retire_mask is not None:
                _retire_mask = _retire_mask.to(self.device)
            patched_worker_action = self._process_endogenous_retirement_actions(
                tensordict["worker_action"], retire_eligible_mask=_retire_mask
            )
        else:
            patched_worker_action = tensordict["worker_action"]
        worker_actions = self._decode_worker_actions_gpu(patched_worker_action)
        firm_actions = self._decode_firm_actions_gpu(tensordict["firm_action"])
        _market_phase(self, "actor_retirement")
        intent_key = "worker_intent_action"
        if intent_key in tensordict.keys():
            intent_search_t, intent_chan_t, intent_int_t = self._decode_worker_actions_tensors(
                tensordict[intent_key]
            )
        else:
            realized_search_t, realized_chan_t, realized_int_t = (
                self._decode_worker_actions_tensors(tensordict["worker_action"])
            )
            intent_search_t, intent_chan_t, intent_int_t = (
                realized_search_t,
                realized_chan_t,
                realized_int_t,
            )
        intent_dec_cpu = intent_search_t.cpu().numpy()
        intent_chan_cpu = intent_chan_t.cpu().numpy()
        intent_int_cpu = intent_int_t.cpu().numpy()
        for i, worker_id in enumerate(self.worker_ids):
            if i >= len(intent_dec_cpu):
                break
            ws = self.state_manager.workers.get(worker_id)
            if ws is not None:
                ws.last_intent_search_decision = int(intent_dec_cpu[i])
                ws.last_intent_search_channel = int(intent_chan_cpu[i])
                ws.last_intent_search_intensity = int(intent_int_cpu[i])
        self._last_firm_vacancy_actions = {
            fid: act["vacancy_management"] for fid, act in firm_actions.items()
        }
        for fid in self._vacancy_debug["sample_firm_ids"]:
            if fid in self._vacancy_debug["firms"]:
                self._vacancy_debug["firms"][fid]["action"] = int(
                    self._last_firm_vacancy_actions.get(fid, -1)
                )
        self._apply_wage_strategies(firm_actions)
        separations = self._process_separations_gpu()
        self._employment_route_ids_this_step["ordinary_job_loss_separations"].update(
            (worker_id for worker_id, _firm_id, sep_type in separations if sep_type != "retirement")
        )
        self._employment_route_ids_this_step["ordinary_retirement_separations"].update(
            (worker_id for worker_id, _firm_id, sep_type in separations if sep_type == "retirement")
        )
        self.state_manager.apply_separations(separations)
        from separation_destination_split import finish_application

        finish_application(self)
        n_trade_purged = 0
        for firm_id, fs in self.state_manager.firms.items():
            target = getattr(fs.firm, "target_size", len(fs.employees))
            excess = len(fs.employees) - target
            if excess > 0:
                n_trade_purged += min(
                    excess, self._trade_war_regime.trade_excess_for_firm(firm_id, len(fs.employees))
                )
                employees_to_shed = list(fs.employees)[-excess:]
                for wid in employees_to_shed:
                    ws = self.state_manager.workers.get(wid)
                    if ws is not None:
                        ws.previous_tenure_months = ws.months_employed
                        ws.previous_wage = ws.worker.current_wage
                        ws.worker.employment_status = "Unemployed"
                        self._employment_route_ids_this_step["overstaff_purge_displacements"].add(
                            wid
                        )
                        ws.worker.unemployment_months = 0
                        ws.worker.is_searching = True
                        ws.worker.current_employer_id = None
                        from environment.state_manager import lwf_draw_retention as _lwf_draw

                        ws.separation_reason = "job_loser"
                        _sex_op = int(getattr(ws.worker, "sex", 1) or 1)
                        ws.lwf_retention_draw = _lwf_draw(
                            previous_tenure_months=ws.previous_tenure_months, sex=_sex_op
                        )
                        from environment.state_manager import grant_ei_if_eligible

                        self.state_manager._ei_period_new_separations += 1
                        if grant_ei_if_eligible(ws, self.config.ei):
                            self.state_manager._ei_period_new_grants += 1
                        ws.months_employed = 0
                        ws.worker.current_wage = 0.0
                        if wid in self._worker_firm_map:
                            del self._worker_firm_map[wid]
                fs.employees = [e for e in fs.employees if e not in employees_to_shed]
        self._trade_war_regime.record_trade_layoffs(n_trade_purged)
        self._employment_route_ids_this_step["ordinary_job_loss_separations"].update(
            (worker_id for worker_id, _firm_id, sep_type in separations if sep_type != "retirement")
        )
        self._employment_route_ids_this_step["ordinary_retirement_separations"].update(
            (worker_id for worker_id, _firm_id, sep_type in separations if sep_type == "retirement")
        )
        from separation_destination_split import tag_destination_routes

        tag_destination_routes(self)
        self.fast_obs.state_table.sync_from_state_manager(self.state_manager, self.worker_ids)
        _stale_status_reads = 0
        if self.current_step % 10 == 0:
            try:
                _truth = torch.zeros(len(self.worker_ids), dtype=torch.long, device=self.device)
                _status_to_int = {"Employed": 0, "Unemployed": 1, "NILF": 2}
                for _i, _wid in enumerate(self.worker_ids):
                    _ws = self.state_manager.workers.get(_wid)
                    if _ws is not None:
                        _truth[_i] = _status_to_int.get(_ws.worker.employment_status, 2)
                _fast = self.fast_obs.state_table.w_status[: len(self.worker_ids)].to(self.device)
                _stale_status_reads = int((_truth != _fast).sum().item())
            except Exception as _sync_check_e:
                logger.warning(f"The consistency check of worker states failed: {_sync_check_e}")
                _stale_status_reads = -1
        self._last_stale_status_reads = _stale_status_reads
        self._process_exogenous_outflows_gpu()
        _market_phase(self, "retirement_separations_and_exogenous_outflows")
        self._process_u_to_nilf_discouragement()
        for worker_id, firm_id, _ in separations:
            if worker_id in self._worker_firm_map:
                del self._worker_firm_map[worker_id]
        separations_by_firm = defaultdict(list)
        for worker_id, firm_id, sep_type in separations:
            ws = self.state_manager.workers.get(worker_id)
            tenure = ws.months_employed if ws else 0
            separations_by_firm[firm_id].append(
                {"tenure_months": tenure, "separation_type": sep_type}
            )
        _opening_vacancy_count = int(self.state_manager._vacancy_ledger_opening)
        self._vac_withdrawn_this_step = 0
        self._vac_exit_purged_this_step = 0
        n_firm_exits = self.process_firm_exits()
        n_firm_entries = self.process_firm_entries()
        from market_mechanisms import refresh_turnover

        refresh_turnover(self, n_firm_exits, n_firm_entries)
        _market_phase(self, "discouragement_and_firm_turnover")
        if self.firm_age_months is not None:
            self.firm_age_months[self.firm_active] += 1
        _firms_with_sep = set(separations_by_firm.keys())
        _firms_with_j2j_prev = self._firms_with_j2j_departure
        sep_counts = {fid: len(seps) for fid, seps in separations_by_firm.items()}
        new_vacancies = self._process_vacancy_creation(firm_actions, separations_by_firm=sep_counts)
        self.state_manager.add_vacancies(new_vacancies, reason="postings_rl")
        self._vac_baseline_accum["sep_total"] += int(sum(sep_counts.values()))
        self._vac_baseline_accum["new_vac"] += int(len(new_vacancies))
        _vac_reason = {"growth": 0, "replacement_sep": 0, "replacement_j2j": 0}
        for v in new_vacancies:
            fid = v.firm_id
            if fid in _firms_with_j2j_prev:
                v.posting_reason = "replacement_j2j"
                _vac_reason["replacement_j2j"] += 1
            elif fid in _firms_with_sep:
                v.posting_reason = "replacement_sep"
                _vac_reason["replacement_sep"] += 1
            else:
                v.posting_reason = "growth"
                _vac_reason["growth"] += 1
        for fid in self._vacancy_debug["sample_firm_ids"]:
            if fid in self._vacancy_debug["firms"]:
                vc_count = self._get_firm_vacancy_count(fid)
                if vc_count >= 0:
                    self._vacancy_debug["firms"][fid]["vac_after_rl"] = vc_count
        _n_injected_ensure = 0
        _n_injected_replenish = 0
        for fid in self._vacancy_debug["sample_firm_ids"]:
            if fid not in self._vacancy_debug["firms"]:
                continue
            after = self._get_firm_vacancy_count(fid)
            if after < 0:
                continue
            self._vacancy_debug["firms"][fid]["vac_after_backstop"] = after
            after_rl = self._vacancy_debug["firms"][fid]["vac_after_rl"]
            if after_rl is None:
                self._vacancy_debug["firms"][fid]["backstop_added"] = None
            else:
                self._vacancy_debug["firms"][fid]["backstop_added"] = int(after - after_rl)
        vacancies_by_firm = defaultdict(list)
        for v in new_vacancies:
            vacancies_by_firm[v.firm_id].append(v)
        self._apply_covid_month()
        search_results = self._process_worker_search_decisions(worker_actions)
        employed_searchers = search_results["employed_searchers"]
        employed_min_wages = search_results["employed_min_wages"]
        unemployed_searchers = search_results["unemployed_searchers"]
        unemployed_discouraged = search_results["unemployed_discouraged"]
        nilf_attempting_reentry = search_results["nilf_attempting_reentry"]
        reentry_result = self._process_nilf_reentry_decisions(nilf_attempting_reentry)
        if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
            self, "_person_trace_active", False
        ):
            self._person_trace_direct_nilf_ids = set(reentry_result["direct_employment_reentries"])
        _n_direct_reentry_fills = int(reentry_result.get("n_direct_employment_reentries", 0))
        _market_phase(self, "search_and_reentry")
        reentry_survivors = reentry_result["successful_reentries"]
        unemployed_searchers.extend(reentry_survivors)
        search_results["unemployed_searchers"] = unemployed_searchers
        n_nilf_reentry = reentry_result["n_successes"]
        self._process_structural_discouragement_gpu()
        n_discouraged_exit = self._process_discouraged_workers(unemployed_discouraged)
        self._nilf_life_event_reentry_this_step = 0
        unemployed_workers = [
            ws
            for ws in self.state_manager.workers.values()
            if ws.worker.employment_status == "Unemployed"
        ]
        unemployed = len(unemployed_workers)
        if self.current_step % 20 == 0:
            self._diagnose_industry_coverage()
        all_workers = dict(self.state_manager.workers)
        for worker_id in employed_searchers:
            if worker_id in all_workers:
                all_workers[worker_id].worker.is_searching = True
        _inst_i = getattr(self, "_demographic_flow_instrumenter", None)
        _i_enabled = _inst_i is not None and _inst_i.enabled
        for worker_id in unemployed_searchers:
            ws = self.state_manager.workers.get(worker_id)
            if ws:
                if (
                    _i_enabled
                    and ws.worker.employment_status == "Unemployed"
                    and (not getattr(ws.worker, "is_searching", False))
                ):
                    _imm_raw_i = getattr(ws.worker, "immigrant_status", False)
                    _is_imm_i = (
                        _imm_raw_i.lower() in ("true", "1", "yes", "immigrant")
                        if isinstance(_imm_raw_i, str)
                        else bool(_imm_raw_i)
                    )
                    _age_i = int(getattr(ws.worker, "age", -1))
                    _inst_i.record_u_no_search_to_u_search(age=_age_i, is_immigrant=_is_imm_i)
                    _inst_i.record_worker_lf_event(
                        worker_id=worker_id,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="I_in",
                        lf_delta=+1,
                        age=_age_i,
                    )
                ws.worker.is_searching = True
        for worker_id in unemployed_discouraged:
            ws = self.state_manager.workers.get(worker_id)
            if ws:
                if (
                    _i_enabled
                    and ws.worker.employment_status == "Unemployed"
                    and getattr(ws.worker, "is_searching", False)
                ):
                    _imm_raw_i2 = getattr(ws.worker, "immigrant_status", False)
                    _is_imm_i2 = (
                        _imm_raw_i2.lower() in ("true", "1", "yes", "immigrant")
                        if isinstance(_imm_raw_i2, str)
                        else bool(_imm_raw_i2)
                    )
                    _age_i2 = int(getattr(ws.worker, "age", -1))
                    _inst_i.record_u_search_to_u_no_search(age=_age_i2, is_immigrant=_is_imm_i2)
                    _inst_i.record_worker_lf_event(
                        worker_id=worker_id,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="I_out",
                        lf_delta=-1,
                        age=_age_i2,
                    )
                ws.worker.is_searching = False
        from employer_mobility import clear_unregistered_employed_search

        clear_unregistered_employed_search(all_workers, employed_searchers)
        self.state_manager.sync_vacancy_dicts()
        _capacity_budget.trim_vacancies(self, reason="capacity_budget_withdrawal")
        _firm_vac_total = sum((fs.n_vacancies for fs in self.state_manager.firms.values()))
        _vac_dict = self.state_manager.get_all_vacancies()
        _vac_dict_count = len(_vac_dict)
        if _firm_vac_total != _vac_dict_count:
            print(
                f"Vacancy counts differ. Firm records hold {_firm_vac_total}, matching received {_vac_dict_count}, difference {_firm_vac_total - _vac_dict_count}"
            )
            if _firm_vac_total > _vac_dict_count * 5:
                for vid, vac in _vac_dict.items():
                    fid = (
                        vac.get("firm_id")
                        if isinstance(vac, dict)
                        else getattr(vac, "firm_id", None)
                    )
        matches, matching_stats = self.matching_mechanism.run_matching(
            workers=all_workers,
            vacancies=_vac_dict,
            worker_actions=worker_actions,
            firm_actions=firm_actions,
            firms=self.state_manager.get_all_firms(),
            employed_min_wages=employed_min_wages,
        )
        for worker_id in employed_searchers:
            ws = self.state_manager.workers.get(worker_id)
            if ws and ws.worker.employment_status == "Employed":
                ws.worker.is_searching = False
        try:
            _executed_unemp = getattr(
                self.matching_mechanism, "_executed_unemployed_searcher_ids", None
            )
            if _executed_unemp:
                _existing = set(search_results["unemployed_searchers"])
                for _wid in _executed_unemp:
                    if _wid not in _existing:
                        search_results["unemployed_searchers"].append(_wid)
                        _existing.add(_wid)
                unemployed_searchers = search_results["unemployed_searchers"]
        except Exception as _exec_e:
            logger.warning(
                f"Could not add the unemployed workers who searched to the search record: {_exec_e}"
            )
        unemployed_matches, j2j_count = self._process_job_to_job_transitions(
            matches, employed_searchers, employed_min_wages
        )
        _capacity_committed_matches = self.state_manager.apply_matches(unemployed_matches)
        if not isinstance(_capacity_committed_matches, list):
            raise ValueError("Binding capacity commitment did not return committed hires")
        unemployed_matches = _capacity_committed_matches
        for match in unemployed_matches:
            self._worker_firm_map[match["worker_id"]] = match["firm_id"]
        _matched_u2e_ids = {match["worker_id"] for match in unemployed_matches}
        if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
            self, "_person_trace_active", False
        ):
            from person_month_trace import committed_j2j_ids

            self._person_trace_committed_u2e_ids = set(_matched_u2e_ids)
            self._person_trace_committed_j2j_ids = committed_j2j_ids(
                self._renewal_last_j2j_matches, j2j_count
            )
        _displaced_this_step = set().union(
            self._employment_route_ids_this_step["ordinary_job_loss_separations"],
            self._employment_route_ids_this_step["firm_exit_displacements"],
            self._employment_route_ids_this_step["overstaff_purge_displacements"],
        )
        self._employment_route_ids_this_step["same_month_rehires"] = (
            _matched_u2e_ids & _displaced_this_step
        )
        matching_stats["j2j_transitions"] = j2j_count
        matching_stats["j2j_offered"] = self.last_matching_stats.get("j2j_offered", 0)
        matching_stats["j2j_accepted"] = self.last_matching_stats.get("j2j_accepted", 0)
        matching_stats["j2j_rejected_wage"] = self.last_matching_stats.get("j2j_rejected_wage", 0)
        matching_stats["employed_searching"] = len(employed_searchers)
        matching_stats["unemployed_active"] = len(unemployed_searchers)
        matching_stats["unemployed_discouraged"] = len(unemployed_discouraged)
        matching_stats["nilf_reentry_attempts"] = len(nilf_attempting_reentry)
        matching_stats["nilf_reentry_success"] = n_nilf_reentry
        matching_stats["discouraged_to_nilf"] = n_discouraged_exit
        matching_stats["separations"] = len(separations)
        matching_stats["ordinary_separations_to_unemployment"] = len(
            self._employment_route_ids_this_step["ordinary_job_loss_separations"]
        )
        matching_stats["ordinary_retirement_separations"] = len(
            self._employment_route_ids_this_step["ordinary_retirement_separations"]
        )
        matching_stats["ordinary_separations_to_temporary_nilf"] = len(
            self._employment_route_ids_this_step["ordinary_temporary_nilf_separations"]
        )
        matching_stats["firm_exit_displacements"] = len(
            self._employment_route_ids_this_step["firm_exit_displacements"]
        )
        matching_stats["overstaff_purge_displacements"] = len(
            self._employment_route_ids_this_step["overstaff_purge_displacements"]
        )
        matching_stats["same_month_rehires"] = len(
            self._employment_route_ids_this_step["same_month_rehires"]
        )
        matching_stats["retirements_from_employment"] = int(
            self._retirements_from_employment_this_step
        )
        matching_stats["retirements_from_unemployment"] = int(
            self._retirements_from_unemployment_this_step
        )
        matching_stats["firm_exits"] = n_firm_exits
        matching_stats["firm_entries"] = n_firm_entries
        matching_stats["active_firms"] = (
            int(self.firm_active.sum()) if self.firm_active is not None else len(self.firm_ids)
        )
        matching_stats["applicants_screened_out"] = matching_stats.get("applicants_screened_out", 0)
        matching_stats["screening_rate"] = matching_stats.get("screening_rate", 0.0)
        self.last_matching_stats = matching_stats
        j2j_worker_ids = []
        for match in matches:
            worker_id = match["worker_id"]
            if worker_id in employed_searchers:
                j2j_worker_ids.append(worker_id)
        search_results["j2j_transitions"] = j2j_worker_ids
        self.fast_obs.state_table.sync_from_state_manager(self.state_manager, self.worker_ids)
        worker_rewards = self._compute_worker_rewards_gpu(
            matches, separations, worker_actions, search_results
        )
        firm_rewards = self._compute_firm_rewards_gpu(
            matches, new_vacancies, separations_by_firm, vacancies_by_firm
        )
        if self._trade_observed_stocks is not None:
            self._trade_observed_stocks.capture(self.state_manager)
        from market_snapshots import capture_monthly as _coverage_capture_monthly

        _coverage_capture_monthly(self)
        _market_phase(self, "matching_and_rewards")
        _n_expired = self.state_manager.advance_time()
        self.current_step += 1
        self.current_month += 1
        self._renewal_monthly_entry()
        _capacity_budget.trim_vacancies(self, reason="capacity_budget_withdrawal")
        _vacancy_event_ledger = self.state_manager.finish_vacancy_ledger_month()
        _unattributed_vacancy_events = [
            row
            for row in _vacancy_event_ledger["events"]
            if row["reason"].startswith("unattributed_")
        ]
        if _vacancy_event_ledger["identity_residual"] != 0 or _unattributed_vacancy_events:
            raise AssertionError(f"vacancy event ledger failed: {_vacancy_event_ledger}")
        _n_filled = len(unemployed_matches) + j2j_count + _n_direct_reentry_fills
        _n_firm_posted = len(new_vacancies)
        self.last_matching_stats["total_vacancies"] = len(self.state_manager.vacancies)
        self.last_matching_stats["vacancies_posted_rl"] = _n_firm_posted
        self.last_matching_stats["vacancies_injected_ensure"] = _n_injected_ensure
        self.last_matching_stats["vacancies_injected_replenish"] = _n_injected_replenish
        self.last_matching_stats["vacancies_filled"] = _n_filled
        self.last_matching_stats["vacancies_expired"] = _n_expired
        self.last_matching_stats["vac_posted_growth"] = _vac_reason["growth"]
        self.last_matching_stats["vac_posted_replacement_sep"] = _vac_reason["replacement_sep"]
        self.last_matching_stats["vac_posted_replacement_j2j"] = _vac_reason["replacement_j2j"]
        self.last_matching_stats["sum_target_size"] = int(self.firm_target_size.sum().item())
        self.last_matching_stats["vacancy_opening_balance"] = _opening_vacancy_count
        self.last_matching_stats["vacancies_withdrawn"] = self._vac_withdrawn_this_step
        _capacity_budget_withdrawals = 0
        for _capacity_event in _vacancy_event_ledger["events"]:
            if _capacity_event["reason"] == "capacity_budget_withdrawal":
                _capacity_delta = _capacity_event["delta"]
                if type(_capacity_delta) is not int or _capacity_delta >= 0:
                    raise ValueError("Invalid capacity vacancy withdrawal source event")
                _capacity_budget_withdrawals -= _capacity_delta
        self.last_matching_stats["capacity_budget_vacancies_withdrawn"] = (
            _capacity_budget_withdrawals
        )
        self.last_matching_stats["vacancies_withdrawn"] += _capacity_budget_withdrawals
        self.last_matching_stats["vacancies_exit_purged"] = self._vac_exit_purged_this_step
        self.last_matching_stats["vacancy_event_ledger"] = _vacancy_event_ledger
        self.last_matching_stats["vacancies_filled_u2e"] = len(unemployed_matches)
        self.last_matching_stats["vacancies_filled_j2j"] = j2j_count
        self.last_matching_stats["vacancies_filled_direct_nilf_reentry"] = _n_direct_reentry_fills
        self.last_matching_stats["direct_nilf_to_employment"] = _n_direct_reentry_fills
        self.last_matching_stats["committed_hires"] = len(unemployed_matches) + j2j_count
        self.last_matching_stats["committed_hires_u2e"] = len(unemployed_matches)
        self.last_matching_stats["committed_hires_j2j"] = j2j_count
        self.last_matching_stats["capacity_commit_rejections"] = sum(
            self._capacity_commit_rejections_this_step.values()
        )
        self.last_matching_stats["capacity_commit_rejections_by_route"] = dict(
            self._capacity_commit_rejections_this_step
        )
        self.last_matching_stats["capacity_commit_rejection_events"] = list(
            self._capacity_commit_rejection_events_this_step
        )
        self._accumulate_quarterly_stats()
        self._log_quarterly_diagnostics()
        if self.current_step % 50 == 0:
            age_dist = {}
            for vac in self.state_manager.vacancies.values():
                age = getattr(vac, "months_posted", 0)
                bucket = min(age, 6)
                age_dist[bucket] = age_dist.get(bucket, 0) + 1
        if self.current_step > 0 and self.current_step % 12 == 0:
            year = self.state_manager.current_year
            total = len(self.state_manager.workers)
            employed = sum(
                (
                    1
                    for ws in self.state_manager.workers.values()
                    if ws.worker.employment_status == "Employed"
                )
            )
            unemployed = sum(
                (
                    1
                    for ws in self.state_manager.workers.values()
                    if ws.worker.employment_status == "Unemployed"
                )
            )
            nilf = sum(
                (
                    1
                    for ws in self.state_manager.workers.values()
                    if ws.worker.employment_status == "NILF"
                )
            )
            logger.warning(
                f"Year {year}, step {self.current_step}: {total} residents, {employed} employed ({employed / total:.1%}), {unemployed} unemployed ({unemployed / total:.1%}), {nilf} outside the labour force ({nilf / total:.1%})"
            )
        if self.current_step > 0 and self.current_step % 3 == 0:
            self._vac_baseline_accum = {
                "sep_total": 0,
                "baseline_sum": 0.0,
                "eff_lambda_sum": 0.0,
                "new_vac": 0,
            }
        if self.current_month > 12:
            self.current_year += 1
            self.current_month = 1
            for ws in self.state_manager.workers.values():
                ws.j2j_this_year = 0
            if self._demographic_flow_instrumenter is not None:
                self._demographic_flow_instrumenter.record_demographics_clamped_call()
            self._process_demographics_clamped()
            self._process_wage_growth()
            if self.firm_active is not None:
                self.firm_exits_this_year = 0
                self.firm_entries_this_year = 0
            self.fast_obs.state_table.sync_from_state_manager(self.state_manager, self.worker_ids)
        done = self.current_step >= self.episode_length
        self.state_manager.compute_worker_type_mean_fields()
        self.fast_obs.update_vacancies(list(self.state_manager.vacancies.values()))
        self.fast_obs.push_history_step()
        self.fast_obs.state_table.sync_from_state_manager(self.state_manager, self.worker_ids)
        obs_td = self._get_observation_tensordict()
        obs_td["worker_reward"] = worker_rewards
        obs_td["firm_reward"] = firm_rewards
        obs_td.set("terminated", torch.tensor([False], dtype=torch.bool, device=self.device))
        obs_td.set("truncated", torch.tensor([done], dtype=torch.bool, device=self.device))
        obs_td.set("done", torch.tensor([done], dtype=torch.bool, device=self.device))
        if hasattr(self, "_vacancy_debug"):
            buckets = {0: [], 1: [], 2: [], 3: [], 4: [], -1: []}
            for fid, fs in self.state_manager.firms.items():
                a = int(getattr(self, "_last_firm_vacancy_actions", {}).get(fid, -1))
                buckets.setdefault(a, []).append(fs.n_vacancies)
            self._vacancy_debug["avg_vacancies_by_action"] = {
                str(a): float(np.mean(v)) if len(v) else None for a, v in buckets.items()
            }
        from age_renewal import enabled as _age_enabled

        if _age_enabled():
            self._net_population.note_calendar(self.current_year, self.current_month)
        return obs_td

    def _get_observation_tensordict(self) -> TensorDict:
        obs = self.fast_obs.get_observations()
        global_state = self.state_manager.get_global_state()
        if global_state.device != self.device:
            global_state = global_state.to(self.device)
        return TensorDict(
            {
                "worker_obs": obs["worker_obs"],
                "firm_obs": obs["firm_obs"],
                "global_state": global_state,
                "worker_context": obs["worker_context"],
                "worker_history": obs["worker_history"],
            },
            batch_size=self._batch_size,
            device=self.device,
        )

    _V0_MAPPING = (5.0, 5.75, 6.5, 7.25, 8.0)

    def _decode_worker_actions_gpu(self, actions: torch.Tensor) -> Dict[str, Dict]:
        action_cfg = self.config.action
        n_channel = action_cfg.worker_search_channel_options
        n_intensity = action_cfg.worker_search_intensity_options
        channel_intensity = n_channel * n_intensity
        if actions.dim() == 1:
            if not getattr(self, "_warned_1d_action", False):
                logger.warning(
                    "The worker action tensor has one dimension. All workers are given reservation-wage level 2 (market rate). This is expected only for a policy without a reservation-wage head."
                )
                self._warned_1d_action = True
            search_actions = actions
            res_wage_actions = torch.full_like(actions, 2)
        else:
            search_actions = actions[..., 0]
            res_wage_actions = actions[..., 1]
        search_decisions = search_actions // channel_intensity
        remainder = search_actions % channel_intensity
        search_channels = remainder // n_intensity
        search_intensities = remainder % n_intensity
        v0_table = torch.tensor(self._V0_MAPPING, device=actions.device, dtype=torch.float32)
        res_wage_clamped = torch.clamp(res_wage_actions, min=0, max=len(self._V0_MAPPING) - 1)
        learned_v0_tensor = v0_table[res_wage_clamped.long()]
        decisions_cpu = search_decisions.cpu().numpy()
        channels_cpu = search_channels.cpu().numpy()
        intensities_cpu = search_intensities.cpu().numpy()
        v0_cpu = learned_v0_tensor.cpu().numpy()
        decoded = {}
        for i, worker_id in enumerate(self.worker_ids):
            if i >= len(channels_cpu):
                break
            decoded[worker_id] = {
                "search_decision": int(decisions_cpu[i]),
                "search_channel": int(channels_cpu[i]),
                "search_intensity": int(intensities_cpu[i]),
                "learned_v0": float(v0_cpu[i]),
                "bounded_rationality_skill": self.state_manager.worker_skills.get(worker_id, 0.5),
            }
            ws = self.state_manager.workers.get(worker_id)
            if ws is not None:
                ws.learned_v0 = float(v0_cpu[i])
                ws.last_res_wage_action = int(res_wage_clamped[i].item())
                ws.last_search_action = (
                    int(decisions_cpu[i]) * channel_intensity
                    + int(channels_cpu[i]) * n_intensity
                    + int(intensities_cpu[i])
                )
                ws.last_search_decision = int(decisions_cpu[i])
                ws.last_search_channel = int(channels_cpu[i])
                ws.last_search_intensity = int(intensities_cpu[i])
        return decoded

    def _decode_worker_actions_tensors(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_cfg = self.config.action
        n_channel = action_cfg.worker_search_channel_options
        n_intensity = action_cfg.worker_search_intensity_options
        channel_intensity = n_channel * n_intensity
        search_actions = actions[..., 0] if actions.dim() > 1 else actions
        search_decisions = search_actions // channel_intensity
        remainder = search_actions % channel_intensity
        search_channels = remainder // n_intensity
        search_intensities = remainder % n_intensity
        return (search_decisions, search_channels, search_intensities)

    def _decode_firm_actions_gpu(self, actions: torch.Tensor) -> Dict[str, Dict]:
        action_cfg = self.config.action
        n_hire = action_cfg.firm_hiring_decision_options
        n_wage = action_cfg.firm_wage_strategy_options
        divisor = n_hire * n_wage
        vacancy_management = actions // divisor
        remainder = actions % divisor
        hiring_decision = remainder // n_wage
        wage_strategy = remainder % n_wage
        vac_cpu = vacancy_management.cpu().numpy()
        hire_cpu = hiring_decision.cpu().numpy()
        wage_cpu = wage_strategy.cpu().numpy()
        decoded = {}
        for i, firm_id in enumerate(self.firm_ids):
            if i >= len(vac_cpu):
                break
            decoded[firm_id] = {
                "vacancy_management": int(vac_cpu[i]),
                "hiring_decision": int(hire_cpu[i]),
                "wage_strategy": int(wage_cpu[i]),
            }
        return decoded

    def _sample_firm_ids_for_debug(self, k: int = 5):
        firm_ids = list(self.state_manager.firms.keys())
        if not firm_ids:
            return []
        k = min(k, len(firm_ids))
        rng = np.random.default_rng(seed=self.current_step + 12345)
        return rng.choice(firm_ids, size=k, replace=False).tolist()

    def _get_firm_vacancy_count(self, firm_id: str) -> int:
        fs = self.state_manager.firms.get(firm_id)
        if fs is None:
            return -1
        return fs.n_vacancies

    def _process_separations_gpu(self) -> List[Tuple[str, str, str]]:
        year = self.state_manager.current_year
        yr_idx = self._year_to_idx.get(year, 0)
        sep_multiplier = float(self._recession_tensor[yr_idx])
        st = self.fast_obs.state_table
        employed_mask = st.w_status == 0
        industry_list = st.w_industry.cpu().long().tolist()
        industry_idx = torch.tensor(
            [self._industry_to_idx.get(ind, 0) for ind in industry_list],
            dtype=torch.long,
            device=self.device,
        )
        sep_rates = self._separation_rates_tensor[yr_idx]
        worker_sep_rates = sep_rates[industry_idx, 0] * sep_multiplier
        worker_layoff_share = sep_rates[industry_idx, 1]
        from separation_destination_split import absolute_rate_vectors

        _absolute_draw_inputs = absolute_rate_vectors(self, worker_sep_rates, st.w_age)
        worker_sep_rates = _absolute_draw_inputs["total"]
        if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
            self, "_person_trace_active", False
        ):
            from person_month_trace import capture_separation_draw

            capture_separation_draw(self, employed_mask, worker_sep_rates, worker_layoff_share)
        rand_sep = torch.rand(self.n_workers, device=self.device)
        rand_type = torch.rand(self.n_workers, device=self.device)
        if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
            self, "_person_trace_active", False
        ):
            from person_month_trace import capture_absolute_random_draws

            capture_absolute_random_draws(self, rand_sep, rand_type, _absolute_draw_inputs)
        sep_mask = employed_mask & (rand_sep < worker_sep_rates)
        layoff_mask = sep_mask & (rand_type < worker_layoff_share)
        quit_mask = sep_mask & ~layoff_mask
        separations = []
        for idx in layoff_mask.nonzero(as_tuple=True)[0].tolist():
            separations.append(
                (
                    self.worker_ids[idx],
                    self._worker_firm_map.get(self.worker_ids[idx], ""),
                    "layoff",
                )
            )
        for idx in quit_mask.nonzero(as_tuple=True)[0].tolist():
            separations.append(
                (self.worker_ids[idx], self._worker_firm_map.get(self.worker_ids[idx], ""), "quit")
            )
        from separation_destination_split import partition_selected_events

        partition_selected_events(self, separations)
        return separations

    def _process_exogenous_outflows_gpu(self) -> int:
        st = self.fast_obs.state_table
        status = st.w_status
        ages = st.w_age.long()
        emp_mask = status == 0
        prime_age_mask = (ages >= 25) & (ages < 55)
        retire_age_mask = ages >= 55
        from learned_participation import validate

        validate(self.config)
        retire_age_mask = torch.zeros_like(retire_age_mask)
        retire_min_age = int(getattr(self.config.reward, "retirement_age_min", 60))
        retire_age_mask = retire_age_mask & (ages < retire_min_age)
        ret_rates = self._retirement_rates_tensor[
            ages.clamp(0, len(self._retirement_rates_tensor) - 1)
        ]
        prime_age_rate = float(getattr(self.config.mnl, "prime_age_e_to_n_rate", 0.008))
        e_to_n_probs = torch.zeros(self.n_workers, device=self.device)
        e_to_n_probs = torch.where(
            prime_age_mask & emp_mask,
            torch.tensor(prime_age_rate, device=self.device),
            e_to_n_probs,
        )
        e_to_n_probs = torch.where(retire_age_mask & emp_mask, ret_rates, e_to_n_probs)
        rand_rolls = torch.rand(self.n_workers, device=self.device)
        e_to_n_mask = emp_mask & (rand_rolls < e_to_n_probs)
        if (
            getattr(self, "_demographic_flow_instrumenter", None) is not None
            and self._demographic_flow_instrumenter.enabled
            and bool(e_to_n_mask.any().item())
        ):
            _inst = self._demographic_flow_instrumenter
            _e2n_idx = torch.nonzero(e_to_n_mask, as_tuple=False).squeeze(-1).tolist()
            for _wi in _e2n_idx:
                _wid = self.worker_ids[_wi] if _wi < len(self.worker_ids) else None
                _ws = self.state_manager.workers.get(_wid) if _wid is not None else None
                _w_age = int(_ws.worker.age) if _ws is not None else -1
                _imm_raw = (
                    getattr(_ws.worker, "immigrant_status", False) if _ws is not None else False
                )
                if isinstance(_imm_raw, str):
                    _is_imm = _imm_raw.lower() in ("true", "1", "yes", "immigrant")
                else:
                    _is_imm = bool(_imm_raw)
                if _w_age >= 55:
                    _inst.record_retirement(age=_w_age, pathway="csv_hazard", is_immigrant=_is_imm)
                    _inst.record_worker_lf_event(
                        worker_id=_wid,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="retire_csv",
                        lf_delta=-1,
                        age=_w_age,
                    )
                elif 25 <= _w_age < 55:
                    _inst.record_prime_age_e_to_n(age=_w_age, is_immigrant=_is_imm)
                    _inst.record_worker_lf_event(
                        worker_id=_wid,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="prime_e2n",
                        lf_delta=-1,
                        age=_w_age,
                    )
        _gate_routed_idx_set: set = set()
        n_e_to_n = 0
        nilf_subtype_enabled = bool(getattr(self.config.mnl, "nilf_subtype_enabled", False))
        retire_e_to_n_idx = (
            set(torch.where(e_to_n_mask & retire_age_mask)[0].tolist()) - _gate_routed_idx_set
        )
        for idx in torch.where(e_to_n_mask)[0].tolist():
            wid = self.worker_ids[idx]
            ws = self.state_manager.workers.get(wid)
            if ws is not None:
                emp_id = getattr(ws.worker, "current_employer_id", None)
                if emp_id and emp_id in self.state_manager.firms:
                    try:
                        self.state_manager.firms[emp_id].employees.remove(wid)
                    except (ValueError, KeyError):
                        pass
                ws.worker.employment_status = "NILF"
                ws.worker.current_employer_id = None
                if wid in self._worker_firm_map:
                    del self._worker_firm_map[wid]
                if nilf_subtype_enabled:
                    if idx in _gate_routed_idx_set:
                        self.state_manager.set_retirement_nilf_subtype(
                            ws, source_state="employed_retirement_gate"
                        )
                    elif idx in retire_e_to_n_idx:
                        self.state_manager.set_retirement_nilf_subtype(
                            ws, source_state="employed_retirement"
                        )
                    else:
                        self.state_manager.assign_exogenous_nilf_subtype(
                            ws, source_state="employed"
                        )
                n_e_to_n += 1
        st.w_status[e_to_n_mask] = 2
        if n_e_to_n > 0:
            self.state_manager._update_aggregate_stats()
        return n_e_to_n

    def _process_structural_discouragement_gpu(self) -> int:
        st = self.fast_obs.state_table
        status = st.w_status
        ages = st.w_age.long()
        unemp_mask = (status == 1) & ~self._covid_locked_mask()
        prime_mask = ages < 45
        older_mask = ages >= 45
        rate_prime = float(getattr(self.config.mnl, "u_to_nilf_struct_rate_prime", 0.0025))
        rate_older = float(getattr(self.config.mnl, "u_to_nilf_struct_rate_older", 0.004))
        u_to_n_probs = torch.zeros(self.n_workers, device=self.device)
        u_to_n_probs = torch.where(
            unemp_mask & prime_mask, torch.tensor(rate_prime, device=self.device), u_to_n_probs
        )
        u_to_n_probs = torch.where(
            unemp_mask & older_mask, torch.tensor(rate_older, device=self.device), u_to_n_probs
        )
        rand_rolls = torch.rand(self.n_workers, device=self.device)
        u_to_n_mask = unemp_mask & (rand_rolls < u_to_n_probs)
        n_u_to_n = 0
        nilf_subtype_enabled = bool(getattr(self.config.mnl, "nilf_subtype_enabled", False))
        _inst_e = getattr(self, "_demographic_flow_instrumenter", None)
        _e_enabled = _inst_e is not None and _inst_e.enabled
        for idx in torch.where(u_to_n_mask)[0].tolist():
            wid = self.worker_ids[idx]
            ws = self.state_manager.workers.get(wid)
            if ws is not None:
                _prior_is_srch_e = bool(getattr(ws.worker, "is_searching", False))
                ws.worker.employment_status = "NILF"
                ws.worker.is_searching = False
                ws.worker.unemployment_months = 0
                if hasattr(ws, "months_unemployed"):
                    ws.months_unemployed = 0
                if nilf_subtype_enabled:
                    self.state_manager.assign_exogenous_nilf_subtype(
                        ws, source_state="unemployed_struct"
                    )
                n_u_to_n += 1
                if _e_enabled:
                    _imm_raw_e = getattr(ws.worker, "immigrant_status", False)
                    _is_imm_e = (
                        _imm_raw_e.lower() in ("true", "1", "yes", "immigrant")
                        if isinstance(_imm_raw_e, str)
                        else bool(_imm_raw_e)
                    )
                    _age_e = int(ws.worker.age)
                    _inst_e.record_u_to_nilf_structural(
                        age=_age_e, is_immigrant=_is_imm_e, prior_is_searching=_prior_is_srch_e
                    )
                    _lf_delta_e = -1 if _prior_is_srch_e else 0
                    _inst_e.record_worker_lf_event(
                        worker_id=wid,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="E",
                        lf_delta=_lf_delta_e,
                        age=_age_e,
                    )
        st.w_status[u_to_n_mask] = 2
        if n_u_to_n > 0:
            self.state_manager._update_aggregate_stats()
        return n_u_to_n

    def _process_u_to_nilf_discouragement(self) -> int:
        disc_cfg = self.config.discouragement
        active_rate = disc_cfg.u_to_nilf_discouragement_monthly_rate
        if active_rate <= 0.0:
            return 0
        age_threshold = int(disc_cfg.u_to_nilf_discouragement_age_threshold)
        duration_threshold = int(disc_cfg.u_to_nilf_discouragement_duration_threshold_months)
        st = self.fast_obs.state_table
        status = st.w_status
        ages = st.w_age.long()
        unemployed_mask = (status == 1) & ~self._covid_locked_mask()
        age_mask = ages >= age_threshold
        eligible_idx = torch.where(unemployed_mask & age_mask)[0].tolist()
        if not eligible_idx:
            return 0
        n_transitions = 0
        n_eligible_after_full_check = 0
        rand_rolls = torch.rand(len(eligible_idx), device="cpu").tolist()
        for j, idx in enumerate(eligible_idx):
            wid = self.worker_ids[idx]
            ws = self.state_manager.workers.get(wid)
            if ws is None:
                continue
            u_months = int(getattr(ws.worker, "unemployment_months", 0))
            if u_months < duration_threshold:
                continue
            ei_eligible = bool(getattr(ws, "ei_eligible", False))
            ei_weeks = int(getattr(ws, "ei_weeks_remaining", 0))
            if ei_eligible and ei_weeks > 0:
                continue
            n_eligible_after_full_check += 1
            if rand_rolls[j] >= active_rate:
                continue
            if not hasattr(self.state_manager, "_u_to_nilf_discouragement_period_durations"):
                self.state_manager._u_to_nilf_discouragement_period_durations = []
            self.state_manager._u_to_nilf_discouragement_period_durations.append(u_months)
            _prior_is_srch_g = bool(getattr(ws.worker, "is_searching", False))
            ws.worker.employment_status = "NILF"
            ws.worker.is_searching = False
            ws.worker.unemployment_months = 0
            if hasattr(ws, "months_unemployed"):
                ws.months_unemployed = 0
            ws.ei_eligible = False
            ws.ei_weeks_remaining = 0
            self.state_manager.set_discouraged_nilf_subtype(
                ws, source_state="unemployed_discouraged"
            )
            st.w_status[idx] = 2
            n_transitions += 1
            _inst_g = getattr(self, "_demographic_flow_instrumenter", None)
            if _inst_g is not None and _inst_g.enabled:
                _imm_raw_g = getattr(ws.worker, "immigrant_status", False)
                _is_imm_g = (
                    _imm_raw_g.lower() in ("true", "1", "yes", "immigrant")
                    if isinstance(_imm_raw_g, str)
                    else bool(_imm_raw_g)
                )
                _age_g = int(ws.worker.age)
                _inst_g.record_u_to_nilf_chronic_65plus(
                    age=_age_g, is_immigrant=_is_imm_g, prior_is_searching=_prior_is_srch_g
                )
                _inst_g.record_worker_lf_event(
                    worker_id=wid,
                    month=self.state_manager.current_month,
                    year=self.state_manager.current_year,
                    channel_tag="G",
                    lf_delta=-1 if _prior_is_srch_g else 0,
                    age=_age_g,
                )
        if not hasattr(self.state_manager, "_u_to_nilf_discouragement_period_count"):
            self.state_manager._u_to_nilf_discouragement_period_count = 0
            self.state_manager._u_to_nilf_discouragement_period_eligible = 0
        self.state_manager._u_to_nilf_discouragement_period_count += n_transitions
        self.state_manager._u_to_nilf_discouragement_period_eligible += n_eligible_after_full_check
        if n_transitions > 0:
            self.state_manager._update_aggregate_stats()
        return n_transitions

    def _load_mortality_rates(self) -> Optional[Dict[int, float]]:
        import pandas as pd
        from pathlib import Path

        try:
            project_root = self.config.data.project_root
        except:
            project_root = Path(__file__).resolve().parent.parent.parent
        mortality_path = project_root / "Data" / "Simulation" / "Parameters" / "mortality_rates.csv"
        try:
            df = pd.read_csv(mortality_path)
            mortality_dict = {}
            for _, row in df.iterrows():
                age = int(row["age"])
                qx_annual = row["qx"]
                qx_monthly = 1 - (1 - qx_annual) ** (1 / 12)
                mortality_dict[age] = qx_monthly
            if not mortality_dict:
                raise ValueError("The mortality table has no rows")
            return mortality_dict
        except FileNotFoundError:
            logger.warning(
                f"Mortality table not found at {mortality_path}. Built-in rates are used."
            )
            return None
        except Exception as e:
            logger.warning(f"Could not read the mortality table ({e}). Built-in rates are used.")
            return None

    def _industry_to_sector(self, industry: int) -> int:
        if industry in [1, 2, 3, 4, 5]:
            return 1
        elif industry >= 7:
            return 2
        else:
            return 0

    def _sample_da_for_cma(self, cma: int) -> Optional[int]:
        das = self.cma_to_das.get(cma, [])
        if not das:
            return None
        return int(np.random.choice(das))

    def _sample_cma_and_da(self, worker_type: str = "general") -> Tuple[int, Optional[int]]:
        if self.cma_to_das:
            available_cmas = list(self.cma_to_das.keys())
            if worker_type == "immigrant":
                cma_weights = []
                for cma in available_cmas:
                    if cma == 535:
                        cma_weights.append(0.7)
                    else:
                        cma_weights.append(len(self.cma_to_das[cma]))
                non_toronto_total = sum(
                    (w for c, w in zip(available_cmas, cma_weights) if c != 535)
                )
                if non_toronto_total > 0:
                    cma_weights = [
                        w if c == 535 else w / non_toronto_total * 0.3
                        for c, w in zip(available_cmas, cma_weights)
                    ]
            else:
                cma_weights = [len(self.cma_to_das[cma]) for cma in available_cmas]
            total_weight = sum(cma_weights)
            if total_weight > 0:
                cma_probs = [w / total_weight for w in cma_weights]
                selected_cma = int(np.random.choice(available_cmas, p=cma_probs))
                selected_da = self._sample_da_for_cma(selected_cma)
                return (selected_cma, selected_da)
        GTHA_CMA_DISTRIBUTION = {535: 0.45, 532: 0.06, 537: 0.12, 550: 0.03, 568: 0.08, 539: 0.07}
        if worker_type == "immigrant":
            GTHA_CMA_DISTRIBUTION = {
                535: 0.7,
                532: 0.04,
                537: 0.08,
                550: 0.02,
                568: 0.06,
                539: 0.04,
            }
        cmas = list(GTHA_CMA_DISTRIBUTION.keys())
        probs = list(GTHA_CMA_DISTRIBUTION.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        selected_cma = int(np.random.choice(cmas, p=probs))
        selected_da = self._sample_da_for_cma(selected_cma)
        return (selected_cma, selected_da)

    def _apply_wage_strategies(self, firm_actions: Dict[str, Dict]):
        WAGE_TIERS = self.config.matching.wage_strategy_multipliers
        self.last_firm_wage_strategy = {}
        for firm_id, action in firm_actions.items():
            fs = self.state_manager.firms.get(firm_id)
            if fs is None:
                continue
            if self.firm_active is not None:
                _fidx = getattr(self, "_firm_id_to_idx", {}).get(firm_id)
                if _fidx is not None and (not bool(self.firm_active[_fidx])):
                    continue
            strategy = action.get("wage_strategy", 1)
            tier_value = WAGE_TIERS.get(strategy, 1.0)
            self.last_firm_wage_strategy[firm_id] = int(strategy)
            fs.previous_wage_premium = fs.wage_premium
            fs.wage_premium = tier_value

    def _get_max_vacancies_for_firm(self, fs) -> int:
        target_size = getattr(fs.firm, "target_size", max(1, len(fs.employees)))
        return max(3, math.ceil(target_size / 3))

    def _workplace_staff_count(self, firm_id, fs):
        return len(fs.employees)

    def _compute_vacancy_baseline(self, firm_id, fs, separations_count: int) -> float:
        target_size = getattr(fs.firm, "target_size", max(1, len(fs.employees)))
        current_employees = self._workplace_staff_count(firm_id, fs)
        reserved_positions = self._covid_reserved_positions(firm_id)
        gap = target_size - current_employees - reserved_positions
        exact_replacement = separations_count
        hr_onboarding_bandwidth = posting_growth_coefficient()
        growth_need = max(0, gap) * hr_onboarding_bandwidth
        return exact_replacement + growth_need

    def _process_vacancy_creation(
        self, firm_actions: Dict[str, Dict], separations_by_firm: Dict[str, int] = None
    ) -> List[Vacancy]:
        new_vacancies = []
        MIN_VACANCY_WAGE = 11.5
        vc = self.config.vacancy
        if separations_by_firm is None:
            separations_by_firm = {}
        for firm_id, action in firm_actions.items():
            fs = self.state_manager.firms.get(firm_id)
            if fs is None:
                continue
            if self.firm_active is not None:
                _fidx = getattr(self, "_firm_id_to_idx", {}).get(firm_id)
                if _fidx is not None and (not bool(self.firm_active[_fidx])):
                    continue
            vacancy_mgmt = action.get("vacancy_management", 2)
            fs.last_vacancy_action = int(vacancy_mgmt)
            fs.last_wage_strategy = int(action.get("wage_strategy", 2))
            MAX_VACANCIES_PER_FIRM = self._get_max_vacancies_for_firm(fs)
            n_new = 0
            if vacancy_mgmt == 0:
                to_remove = list(fs.vacancies.keys())
                if to_remove:
                    self.state_manager.remove_vacancies(to_remove, reason="policy_withdrawal")
                continue
            elif vacancy_mgmt == 1:
                excess = list(fs.vacancies.keys())[1:]
                if excess:
                    self.state_manager.remove_vacancies(excess, reason="policy_withdrawal")
                continue
            else:
                current_employees = self._workplace_staff_count(firm_id, fs)
                reserved_positions = self._covid_reserved_positions(firm_id)
                current_vacancies = fs.n_vacancies
                target_size = getattr(fs.firm, "target_size", None)
                if target_size is None:
                    target_size = max(1, current_employees)
                gap = target_size - current_employees - reserved_positions
                if gap <= 0:
                    continue
                if current_vacancies >= MAX_VACANCIES_PER_FIRM:
                    continue
                sep_count = separations_by_firm.get(firm_id, 0)
                baseline_lambda = self._compute_vacancy_baseline(firm_id, fs, sep_count)
                multiplier = vc.vacancy_action_multipliers.get(vacancy_mgmt, 1.0)
                effective_lambda = baseline_lambda * multiplier
                self._vac_baseline_accum["baseline_sum"] += float(baseline_lambda)
                self._vac_baseline_accum["eff_lambda_sum"] += float(effective_lambda)
                n_new = np.random.poisson(max(0.0, effective_lambda))
                remaining_capacity = max(0, gap - current_vacancies)
                n_new = min(n_new, remaining_capacity)
                room_left = MAX_VACANCIES_PER_FIRM - current_vacancies
                n_new = min(n_new, room_left)
            for _ in range(n_new):
                offered_wage = max(MIN_VACANCY_WAGE, fs.current_wage_offer * fs.wage_premium)
                offered_wage = min(offered_wage, INDUSTRY_WAGE_CEILING.get(fs.firm.industry, 80.0))
                vacancy = Vacancy(
                    id=f"V{self._vacancy_id_counter:08d}",
                    firm_id=firm_id,
                    industry=fs.firm.industry,
                    sector=fs.firm.sector,
                    skill_required=get_industry_skill_requirement(fs.firm.industry),
                    education_required=get_industry_skill_requirement(fs.firm.industry),
                    offered_wage=offered_wage,
                    cma=fs.firm.cma,
                    da_id=fs.firm.da_id,
                )
                new_vacancies.append(vacancy)
                self._vacancy_id_counter += 1
        return new_vacancies

    def _generate_initial_vacancies(self) -> List[Vacancy]:
        vacancies = []
        for firm_id, fs in self.state_manager.firms.items():
            annual_rate = INDUSTRY_VACANCY_RATE_ANNUAL.get(fs.firm.industry, 3.5)
            vacancy_rate = annual_rate / 100.0 / 12.0
            n_vacancies = np.random.poisson(vacancy_rate * fs.firm.n_employees)
            for _ in range(n_vacancies):
                offered_wage = fs.current_wage_offer * fs.wage_premium
                offered_wage = min(offered_wage, INDUSTRY_WAGE_CEILING.get(fs.firm.industry, 80.0))
                vacancy = Vacancy(
                    id=f"V{self._vacancy_id_counter:08d}",
                    firm_id=firm_id,
                    industry=fs.firm.industry,
                    sector=fs.firm.sector,
                    skill_required=get_industry_skill_requirement(fs.firm.industry),
                    education_required=get_industry_skill_requirement(fs.firm.industry),
                    offered_wage=offered_wage,
                    cma=fs.firm.cma,
                    da_id=fs.firm.da_id,
                )
                vacancies.append(vacancy)
                self._vacancy_id_counter += 1
        return vacancies

    _ATTACHMENT_AGE_BANDS = (
        (15, 25, "15-24"),
        (25, 35, "25-34"),
        (35, 45, "35-44"),
        (45, 55, "45-54"),
        (55, 65, "55-64"),
        (65, 200, "65-99"),
    )
    _ATTACHMENT_STATUS_LABELS = ("E", "U_search", "U_no_search", "NILF_eligible", "NILF_ineligible")

    def _compute_worker_rewards_gpu(
        self,
        matches: List[Dict[str, Any]],
        separations: List[Tuple[str, str, str]],
        worker_actions: Dict[str, Dict[str, Any]],
        search_results: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        N = self.n_workers
        device = self.device
        st = self.fast_obs.state_table
        employment_status = st.w_status
        wages = st.w_wage
        ei_eligible = st.w_ei_eligible
        ei_weeks = st.w_ei_weeks
        search_intensity = torch.zeros(N, device=device)
        prev_wages = torch.full((N,), 15.0, device=device)
        wid_to_idx = st.worker_id_to_idx
        for wid, action in worker_actions.items():
            idx = wid_to_idx.get(wid)
            if idx is not None:
                search_intensity[idx] = action.get("search_intensity", 0)
                ws = self.state_manager.workers.get(wid)
                if ws is not None:
                    prev_wages[idx] = getattr(ws, "previous_wage", 15.0)
        BASE_INCOME = getattr(self.config.reward, "base_survival_income", 3.0)
        SEARCH_COST_PER_LEVEL = getattr(self.config.reward, "search_cost_per_level", 0.05)
        LEISURE_VALUE = getattr(self.config.reward, "nilf_leisure_value", 0.1)
        LEISURE_VALUE_PRIMEAGE = getattr(
            self.config.reward, "nilf_leisure_value_primeage", LEISURE_VALUE
        )
        AGE_LO = getattr(self.config.reward, "nilf_leisure_primeage_age_lo", 35)
        AGE_HI = getattr(self.config.reward, "nilf_leisure_primeage_age_hi", 55)
        RETIREMENT_AGE_MIN = int(getattr(self.config.reward, "retirement_age_min", 60))
        RETIREMENT_CPP65 = float(getattr(self.config.reward, "cpp65_avg_monthly", 640.0))
        RETIREMENT_OAS65 = float(getattr(self.config.reward, "oas65_monthly", 540.0))
        RETIREMENT_HOURS = float(getattr(self.config.reward, "retirement_hours_per_month", 160.0))
        employed_mask = employment_status == 0
        unemployed_mask = employment_status == 1
        nilf_mask = employment_status == 2
        income = torch.full((N,), BASE_INCOME, device=device)
        income = torch.where(employed_mask, torch.clamp(wages, min=BASE_INCOME), income)
        ei_receiving = ei_eligible & (ei_weeks > 0)
        ei_income = prev_wages * self.config.ei.ei_replacement_rate
        unemployed_income = torch.where(
            ei_receiving,
            torch.clamp(ei_income, min=BASE_INCOME),
            torch.full_like(income, BASE_INCOME),
        )
        income = torch.where(unemployed_mask, unemployed_income, income)
        from environment.nilf_subtype import SUBTYPE_RETIREMENT

        subtypes_dev = self.fast_obs.state_table.w_nilf_subtype.to(device)
        ages_dev = self.fast_obs.state_table.w_age.to(device).to(torch.float32)
        months_before_65 = torch.clamp((65.0 - ages_dev) * 12.0, min=0.0)
        cpp_early_hourly = RETIREMENT_CPP65 * (1.0 - 0.006 * months_before_65) / RETIREMENT_HOURS
        cpp_full_hourly = (RETIREMENT_CPP65 + RETIREMENT_OAS65) / RETIREMENT_HOURS
        is_65plus = ages_dev >= 65.0
        retirement_hourly_raw = torch.where(
            is_65plus, torch.full_like(cpp_early_hourly, cpp_full_hourly), cpp_early_hourly
        )
        retirement_hourly = torch.clamp(retirement_hourly_raw, min=BASE_INCOME)
        retire_mask = (
            nilf_mask
            & (subtypes_dev == SUBTYPE_RETIREMENT)
            & (ages_dev >= float(RETIREMENT_AGE_MIN))
        )
        income = torch.where(retire_mask, retirement_hourly, income)
        executed_searcher_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if search_results is not None:
            executed_unemp_ids = search_results.get("unemployed_searchers", []) or []
            for wid in executed_unemp_ids:
                idx = wid_to_idx.get(wid)
                if idx is not None:
                    executed_searcher_mask[idx] = True
        charged_searcher_mask = executed_searcher_mask
        search_cost = torch.where(
            charged_searcher_mask,
            search_intensity * SEARCH_COST_PER_LEVEL,
            torch.zeros_like(income),
        )
        ages = self.fast_obs.state_table.w_age.to(device)
        primeage_mask = (ages >= AGE_LO) & (ages < AGE_HI)
        leisure_levels = torch.where(
            primeage_mask,
            torch.full_like(income, LEISURE_VALUE_PRIMEAGE),
            torch.full_like(income, LEISURE_VALUE),
        )
        leisure = torch.where(nilf_mask, leisure_levels, torch.zeros_like(income))
        rewards = torch.log(income) - search_cost + leisure
        return rewards

    def _compute_firm_rewards_gpu(
        self,
        matches: List[Dict[str, Any]],
        new_vacancies: List[Vacancy],
        separations_by_firm: Dict[str, List[Dict]],
        vacancies_by_firm: Dict[str, List[Vacancy]],
    ) -> torch.Tensor:
        rewards = torch.zeros(self.n_firms, device=self.device)
        config = FirmRewardConfig()
        matches_by_firm = defaultdict(list)
        for match in matches:
            firm_id = match["firm_id"]
            matches_by_firm[firm_id].append(
                {
                    "match_quality": match.get("match_quality", 0.5),
                    "previous_unemployment_months": match.get("unemployment_months", 0),
                }
            )
        _decomp_keys = [
            "revenue",
            "counterfactual_revenue",
            "trade_revenue_loss",
            "trade_input_costs",
            "wage_bill",
            "vacancy_costs",
            "profit",
            "scaled_reward",
        ]
        _decomp_sums = {k: 0.0 for k in _decomp_keys}
        _n_active = 0
        _profit_raw_stash: Dict[str, float] = {}
        for i, firm_id in enumerate(self.firm_ids):
            if self.firm_active is not None and (not self.firm_active[i]):
                rewards[i] = 0.0
                continue
            fs = self.state_manager.firms.get(firm_id)
            if fs is None:
                continue
            employees = []
            for emp_id in fs.employees:
                ws = self.state_manager.workers.get(emp_id)
                if ws:
                    employees.append(
                        {
                            "hourly_wage": ws.worker.current_wage,
                            "hours_per_week": ws.worker.hours_per_week,
                            "match_quality": getattr(ws, "match_quality", 0.6),
                            "industry": getattr(ws.worker, "industry", 8),
                            "skill_level": getattr(ws.worker, "skill_level", 2),
                        }
                    )
            vacancies = []
            for vac_id, vac in self.state_manager.vacancies.items():
                if vac.firm_id == firm_id:
                    vacancies.append({"duration_months": getattr(vac, "duration_months", 1)})
            n_employees = len(employees)
            if hasattr(self, "_firm_id_to_idx") and firm_id in self._firm_id_to_idx:
                firm_idx = self._firm_id_to_idx[firm_id]
                target_size = int(self.firm_base_size[firm_idx].item())
            elif self.firm_base_size is not None:
                target_size = int(self.firm_base_size[i].item())
            else:
                target_size = max(1, n_employees)
            firm_state = {
                "employees": employees,
                "vacancies": vacancies,
                "target_size": target_size,
                "wage_premium": fs.wage_premium,
                "previous_wage_premium": fs.previous_wage_premium,
                "industry": fs.firm.industry,
                "trade_demand_multiplier": float(getattr(fs.firm, "trade_demand_multiplier", 1.0)),
                "trade_productivity_multiplier": float(
                    getattr(fs.firm, "trade_productivity_multiplier", 1.0)
                ),
                "trade_input_cost_rate": float(getattr(fs.firm, "trade_input_cost_rate", 0.0)),
            }
            action = {"vacancies_posted": len(vacancies_by_firm.get(firm_id, []))}
            outcome = {
                "new_hires": matches_by_firm.get(firm_id, []),
                "separations": separations_by_firm.get(firm_id, []),
            }
            reward, decomp = compute_firm_reward(
                firm_state,
                action,
                outcome,
                config,
                verbose=False,
                firm_id=i,
                return_decomposition=True,
            )
            rewards[i] = reward
            _profit_raw_stash[firm_id] = float(decomp.get("profit", 0.0))
            _n_active += 1
            for k in _decomp_keys:
                _decomp_sums[k] += decomp.get(k, 0.0)
        self.last_firm_profit_raw = _profit_raw_stash
        _denom = max(_n_active, 1)
        self._firm_reward_decomp = {k: _decomp_sums[k] / _denom for k in _decomp_keys}
        self._firm_reward_decomp["total"] = float(rewards.mean().item())
        return rewards

    def get_validation_metrics(self) -> Dict[str, float]:
        self._trade_war_regime.capture_outcomes(self)
        metrics = self.state_manager.get_metrics()
        metrics.update(self._trade_war_regime.current_metrics())
        if self.last_matching_stats:
            ms = self.last_matching_stats
            metrics["applications"] = ms.get("applications", 0)
            metrics["offers"] = ms.get("offers", 0)
            metrics["matches"] = ms.get("matches", 0)
            metrics["rejections"] = ms.get("rejections", 0)
            metrics["vacancy_fill_rate"] = ms.get("vacancy_fill_rate", 0)
            metrics["acceptance_rate"] = ms.get("acceptance_rate", 0)
            metrics["avg_accept_prob"] = ms.get("avg_accept_prob", 0)
            metrics["n_applicants_total"] = ms.get("n_applicants_total", 0)
            metrics["n_applicants_after_filters"] = ms.get("n_applicants_after_filters", 0)
            metrics["n_offers_created"] = ms.get("n_offers_created", 0)
            metrics["n_offers_sent"] = ms.get("n_offers_sent", 0)
            metrics["screen_reject_commute_gate"] = ms.get("screen_reject_commute_gate", 0)
            metrics["screen_reject_industry_mismatch"] = ms.get(
                "screen_reject_industry_mismatch", 0
            )
            metrics["screen_reject_selectivity"] = ms.get("screen_reject_selectivity", 0)
            n_vac = metrics.get("n_vacancies", 1)
            metrics["apps_per_vacancy"] = metrics["applications"] / max(n_vac, 1)
            commute_values = self.last_matching_stats.get("commute_values", [])
            if commute_values:
                metrics["avg_commute"] = sum(commute_values) / len(commute_values)
                metrics["commute_unit"] = self.last_matching_stats.get("commute_unit", "km")
                metrics["use_travel_time"] = self.last_matching_stats.get("use_travel_time", False)
        return metrics

    def _validate_worker_industries(self):
        import pandas as pd

        VALID_INDUSTRIES = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
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
        probs = list(INDUSTRY_DIST.values())
        probs = [p / sum(probs) for p in probs]
        fixed_from_sample = 0
        invalid_codes_found = {}
        for ws in self.state_manager.workers.values():
            industry = ws.worker.industry
            try:
                ind_int = (
                    int(industry)
                    if industry is not None
                    and (not (isinstance(industry, float) and pd.isna(industry)))
                    else None
                )
            except (ValueError, TypeError):
                ind_int = None
            invalid = ind_int is None or ind_int not in VALID_INDUSTRIES
            if invalid:
                invalid_codes_found[industry] = invalid_codes_found.get(industry, 0) + 1
                ws.worker.industry = int(np.random.choice(industries, p=probs))
                fixed_from_sample += 1
        if fixed_from_sample > 0:
            logger.warning(
                f"Assigned a valid industry to {fixed_from_sample} workers, drawn from the empirical distribution"
            )
            if invalid_codes_found:
                logger.warning(f"Invalid industry codes found: {invalid_codes_found}")
        self._reassign_youth_to_youth_industries()

    def _reassign_youth_to_youth_industries(self):
        for ws in self.state_manager.workers.values():
            if ws.worker.age < 25 and ws.worker.employment_status == "Unemployed":
                if not hasattr(ws.worker, "original_industry"):
                    ws.worker.original_industry = ws.worker.industry
                new_industry = np.random.choice(YOUTH_INDUSTRIES, p=YOUTH_INDUSTRY_PROBS)
                ws.worker.industry = new_industry
                ws.worker.sector = self._industry_to_sector(new_industry)

    def _diagnose_industry_coverage(self):
        from collections import Counter

        VALID_INDUSTRIES = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
        unemployed_searching = [
            ws
            for ws in self.state_manager.workers.values()
            if ws.worker.employment_status == "Unemployed" and ws.worker.is_searching
        ]
        worker_industries = Counter((ws.worker.industry for ws in unemployed_searching))
        vacancy_industries = Counter((v.industry for v in self.state_manager.vacancies.values()))
        firm_industries = Counter((fs.firm.industry for fs in self.state_manager.firms.values()))
        invalid_worker_industries = {
            ind for ind in worker_industries.keys() if ind not in VALID_INDUSTRIES
        }
        if invalid_worker_industries:
            for ind in invalid_worker_industries:
                n_workers = worker_industries[ind]
                logger.error(
                    f"{n_workers} unemployed workers have the invalid industry code {ind} (valid codes are 1 to 5 and 7 to 18) and cannot be matched"
                )
        stranded_workers = 0
        stranded_industries = []
        for industry, n_workers in worker_industries.items():
            n_vacs = vacancy_industries.get(industry, 0)
            if n_vacs == 0 and n_workers > 0:
                stranded_workers += n_workers
                stranded_industries.append((industry, n_workers))
        if stranded_workers > 0:
            logger.warning(
                f"Step {self.current_step}: {stranded_workers} workers are in industries without vacancies"
            )
            for ind, count in sorted(stranded_industries, key=lambda x: -x[1])[:5]:
                n_firms = firm_industries.get(ind, 0)
                is_invalid = " (INVALID CODE!)" if ind not in VALID_INDUSTRIES else ""
                logger.warning(
                    f"  Industry {ind}{is_invalid}: {count} workers, 0 vacancies, {n_firms} firms"
                )

    def _process_worker_search_decisions(self, worker_actions: Dict[str, Dict]) -> Dict[str, Any]:
        _nilf_rate_check = float(getattr(self.config.mnl, "nilf_to_u_rate", 0.0))
        assert (
            _nilf_rate_check == 0.0
        ), f"nilf_to_u_rate is {_nilf_rate_check} but must be 0.0. Re-entry uses eligibility, age and life events."
        employed_searchers = []
        employed_min_wages = {}
        unemployed_searchers = []
        unemployed_discouraged = []
        nilf_attempting_reentry = []
        all_unemployed = []
        for worker_id, ws in self.state_manager.workers.items():
            worker = ws.worker
            action = worker_actions.get(
                worker_id, {"search_decision": 0, "search_channel": 0, "search_intensity": 0}
            )
            if ws.covid_state != COVID_NONE:
                continue
            search_decision = action["search_decision"]
            if worker.employment_status == "Employed":
                import random as _rng

                tenure = getattr(ws, "months_employed", 0)
                industry = getattr(worker, "industry", 8)
                industry_avg = self._get_industry_average_wage(industry)
                wage_ratio = worker.current_wage / max(industry_avg, 1.0)
                search_prob = max(
                    0.02, 0.08 * max(0.3, 1.0 - tenure / 60.0) * max(1.0, 1.5 - wage_ratio)
                )
                if _rng.random() < search_prob:
                    employed_searchers.append(worker_id)
                    j2j_thresholds = self.config.matching.j2j_min_raise_by_intensity
                    min_raise = j2j_thresholds[min(2, len(j2j_thresholds) - 1)]
                    if self.current_step < 3:
                        min_raise = 1.0 + (min_raise - 1.0) * 0.5
                    employed_min_wages[worker_id] = worker.current_wage * min_raise
            elif worker.employment_status == "Unemployed":
                search_intensity = action.get("search_intensity", 0)
                all_unemployed.append((worker_id, search_decision, search_intensity))
            elif worker.employment_status == "NILF":
                from separation_destination_split import block_same_month_reentry

                if block_same_month_reentry(self, ws, worker_id):
                    continue
                from population_participation import record_funnel

                record_funnel(self, ws, search_decision)
                from market_mechanisms import reentry_age_allowed

                if reentry_age_allowed(worker.age):
                    life_event_rate = float(
                        getattr(self.config.mnl, "nilf_life_event_reentry_rate", 0.0)
                    )
                    if worker.age >= 65:
                        life_event_rate = 0.0
                    intent_search_decision = getattr(
                        ws, "last_intent_search_decision", search_decision
                    )
                    subtype_enabled = getattr(self.config.mnl, "nilf_subtype_enabled", False)
                    if subtype_enabled:
                        from environment.nilf_subtype import effective_subtype_windows

                        _school_w, _family_w, _illness_w = effective_subtype_windows(
                            self.config.mnl, int(worker.age)
                        )
                        env_eligible = is_search_eligible(
                            subtype=ws.nilf_subtype,
                            months_in_subtype=ws.nilf_subtype_months,
                            school_window=_school_w,
                            family_window=_family_w,
                            illness_window=_illness_w,
                            allow_retirement_reentry=getattr(
                                self.config.mnl, "allow_retirement_reentry", False
                            ),
                        )
                    else:
                        env_eligible = True
                    is_retired = subtype_enabled and ws.nilf_subtype == SUBTYPE_RETIREMENT
                    if is_retired and (
                        not getattr(self.config.mnl, "allow_retirement_reentry", False)
                    ):
                        will_reenter = False
                        reentry_kind = None
                    elif 20 <= worker.age < 70:
                        will_reenter = False
                        reentry_kind = "slid_pending"
                        nilf_attempting_reentry.append(worker_id)
                    elif intent_search_decision == 1 and env_eligible:
                        will_reenter = False
                        reentry_kind = "rl_pending"
                        nilf_attempting_reentry.append(worker_id)
                    elif (
                        not is_retired
                        and life_event_rate > 0.0
                        and (np.random.random() < life_event_rate)
                    ):
                        will_reenter = True
                        reentry_kind = "life_event"
                    else:
                        will_reenter = False
                        reentry_kind = None
                    if will_reenter:
                        worker.employment_status = "Unemployed"
                        worker.is_searching = True
                        worker.unemployment_months = 0
                        ws.months_nilf = 0
                        ws.months_unemployed = 0
                        ws.separation_reason = "re_entrant"
                        ws.lwf_retention_draw = None
                        if getattr(self.config.mnl, "nilf_subtype_enabled", False):
                            self.state_manager.clear_nilf_subtype(ws)
                        search_intensity_val = action.get("search_intensity", 1)
                        all_unemployed.append((worker_id, 1, max(search_intensity_val, 1)))
                        if reentry_kind == "life_event":
                            if not hasattr(self, "_nilf_life_event_reentry_this_step"):
                                self._nilf_life_event_reentry_this_step = 0
                            self._nilf_life_event_reentry_this_step += 1
                            _inst_c = getattr(self, "_demographic_flow_instrumenter", None)
                            if _inst_c is not None and _inst_c.enabled:
                                _imm_raw_c = getattr(worker, "immigrant_status", False)
                                _is_imm_c = (
                                    _imm_raw_c.lower() in ("true", "1", "yes", "immigrant")
                                    if isinstance(_imm_raw_c, str)
                                    else bool(_imm_raw_c)
                                )
                                _age_c = int(worker.age)
                                _inst_c.record_nilf_reentry_life_event(
                                    age=_age_c, dest="U_search", is_immigrant=_is_imm_c
                                )
                                _inst_c.record_worker_lf_event(
                                    worker_id=worker_id,
                                    month=self.state_manager.current_month,
                                    year=self.state_manager.current_year,
                                    channel_tag="C",
                                    lf_delta=+1,
                                    age=_age_c,
                                )
        for worker_id, search_decision, search_intensity in all_unemployed:
            ws = self.state_manager.workers.get(worker_id)
            ei_receiving = (
                ws
                and getattr(ws, "ei_eligible", False)
                and (getattr(ws, "ei_weeks_remaining", 0) > 0)
            )
            policy_search_decision = (
                getattr(ws, "last_intent_search_decision", search_decision)
                if ws is not None
                else search_decision
            )
            if ei_receiving:
                unemployed_searchers.append(worker_id)
                if ws:
                    _prior_is_searching = bool(getattr(ws.worker, "is_searching", False))
                    if (
                        self._demographic_flow_instrumenter is not None
                        and self._demographic_flow_instrumenter.enabled
                        and (not _prior_is_searching)
                    ):
                        _imm_raw_ei = getattr(ws.worker, "immigrant_status", False)
                        _is_imm_ei = (
                            _imm_raw_ei.lower() in ("true", "1", "yes", "immigrant")
                            if isinstance(_imm_raw_ei, str)
                            else bool(_imm_raw_ei)
                        )
                        _age_ei = int(getattr(ws.worker, "age", -1))
                        self._demographic_flow_instrumenter.record_u_no_search_to_u_search(
                            age=_age_ei, is_immigrant=_is_imm_ei
                        )
                        self._demographic_flow_instrumenter.record_worker_lf_event(
                            worker_id=worker_id,
                            month=self.state_manager.current_month,
                            year=self.state_manager.current_year,
                            channel_tag="I_in_EI",
                            lf_delta=+1,
                            age=_age_ei,
                        )
                    ws.worker.is_searching = True
            elif policy_search_decision == 1 and search_intensity > 0:
                unemployed_searchers.append(worker_id)
            else:
                unemployed_discouraged.append(worker_id)
        n_unemployed = len(all_unemployed)
        if len(unemployed_searchers) < 5 and n_unemployed > 20:
            logger.warning(
                f"Only {len(unemployed_searchers)} of {n_unemployed} unemployed workers are searching"
            )
        return {
            "employed_searchers": employed_searchers,
            "employed_min_wages": employed_min_wages,
            "unemployed_searchers": unemployed_searchers,
            "unemployed_discouraged": unemployed_discouraged,
            "nilf_attempting_reentry": nilf_attempting_reentry,
        }

    _EMPIRICAL_NILF_TO_U_RATE = {
        15: 0.031,
        20: 0.101,
        25: 0.162,
        30: 0.274,
        35: 0.126,
        40: 0.19,
        45: 0.181,
        50: 0.172,
        55: 0.103,
        60: 0.031,
        65: 0.033,
        70: 0.058,
    }

    def _process_nilf_reentry_decisions(self, nilf_attempting_reentry: List[str]) -> Dict[str, Any]:
        empty = {
            "successful_reentries": [],
            "direct_employment_reentries": [],
            "attempts_by_age": {k: 0 for k in self._EMPIRICAL_NILF_TO_U_RATE},
            "successes_by_age": {k: 0 for k in self._EMPIRICAL_NILF_TO_U_RATE},
            "n_attempts": 0,
            "n_successes": 0,
            "n_direct_employment_reentries": 0,
        }
        if not nilf_attempting_reentry:
            return empty
        successful_reentries: List[str] = []
        direct_employment_reentries: List[str] = []
        attempts_by_age = {k: 0 for k in self._EMPIRICAL_NILF_TO_U_RATE}
        successes_by_age = {k: 0 for k in self._EMPIRICAL_NILF_TO_U_RATE}
        subtype_enabled = bool(getattr(self.config.mnl, "nilf_subtype_enabled", False))
        for worker_id in nilf_attempting_reentry:
            ws = self.state_manager.workers.get(worker_id)
            if ws is None:
                continue
            age = ws.worker.age
            if age < 20:
                age_bin = 15
            elif age < 25:
                age_bin = 20
            elif age < 30:
                age_bin = 25
            elif age < 35:
                age_bin = 30
            elif age < 40:
                age_bin = 35
            elif age < 45:
                age_bin = 40
            elif age < 50:
                age_bin = 45
            elif age < 55:
                age_bin = 50
            elif age < 60:
                age_bin = 55
            elif age < 65:
                age_bin = 60
            elif age < 70:
                age_bin = 65
            else:
                age_bin = 70
            attempts_by_age[age_bin] = attempts_by_age.get(age_bin, 0) + 1
            inherited = self._EMPIRICAL_NILF_TO_U_RATE[age_bin]
            from population_participation import reentry_destination_probabilities

            to_e_prob, to_u_prob = reentry_destination_probabilities(ws, self.config, inherited)
            draw = np.random.random()
            destination = "E" if draw < to_e_prob else "U" if draw < to_e_prob + to_u_prob else "N"
            if destination == "E":
                from age_renewal import choose_entrant_firm
                import capacity_budget as _capacity_budget

                candidates = []
                for i, firm_id in enumerate(self.firm_ids):
                    fs = self.state_manager.firms[firm_id]
                    candidates.append(
                        {
                            "firm_id": firm_id,
                            "active": bool(self.firm_active[i]),
                            "open_positions": (
                                _capacity_budget.remaining_slots(self, firm_id)
                                if fs.vacancies
                                else 0
                            ),
                            "industry": int(fs.firm.industry),
                            "cma": int(fs.firm.cma),
                        }
                    )
                firm_id = choose_entrant_firm(
                    candidates, int(ws.worker.industry), int(ws.worker.cma)
                )
                if firm_id is None:
                    destination = "N"
                else:
                    fs = self.state_manager.firms[firm_id]
                    vacancy_id = sorted(fs.vacancies)[0]
                    if not self.state_manager.capacity_commit_allowed(
                        worker_id, firm_id, vacancy_id, "direct_nilf"
                    ):
                        continue
                    self.state_manager.remove_vacancies(
                        [vacancy_id], reason="fill_direct_nilf_reentry"
                    )
                    ws.worker.employment_status = "Employed"
                    ws.worker.is_searching = False
                    ws.worker.current_employer_id = firm_id
                    ws.worker.current_wage = float(fs.current_wage_offer)
                    ws.worker.unemployment_months = 0
                    ws.months_nilf = ws.months_unemployed = ws.months_employed = 0
                    ws.separation_reason = None
                    ws.lwf_retention_draw = None
                    fs.employees.append(worker_id)
                    fs.total_hires += 1
                    fs.total_vacancies_filled += 1
                    idx = self._firm_id_to_idx[firm_id]
                    self.firm_employment[idx] = len(fs.employees)
                    self._worker_firm_map[worker_id] = firm_id
                    if subtype_enabled:
                        self.state_manager.clear_nilf_subtype(ws)
                    successful_reentries.append(worker_id)
                    direct_employment_reentries.append(worker_id)
                    successes_by_age[age_bin] = successes_by_age.get(age_bin, 0) + 1
            if destination == "U":
                ws.worker.employment_status = "Unemployed"
                ws.worker.unemployment_months = 0
                ws.worker.is_searching = True
                ws.months_nilf = 0
                ws.months_unemployed = 0
                ws.separation_reason = "re_entrant"
                ws.lwf_retention_draw = None
                if subtype_enabled:
                    self.state_manager.clear_nilf_subtype(ws)
                successful_reentries.append(worker_id)
                successes_by_age[age_bin] = successes_by_age.get(age_bin, 0) + 1
                _inst_d = getattr(self, "_demographic_flow_instrumenter", None)
                if _inst_d is not None and _inst_d.enabled:
                    _imm_raw_d = getattr(ws.worker, "immigrant_status", False)
                    _is_imm_d = (
                        _imm_raw_d.lower() in ("true", "1", "yes", "immigrant")
                        if isinstance(_imm_raw_d, str)
                        else bool(_imm_raw_d)
                    )
                    _inst_d.record_nilf_reentry_rl_gated(
                        age=int(age), dest="U_search", is_immigrant=_is_imm_d
                    )
                    _inst_d.record_worker_lf_event(
                        worker_id=worker_id,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="D_RL",
                        lf_delta=+1,
                        age=int(age),
                    )
        n_attempts = len(nilf_attempting_reentry)
        n_successes = len(successful_reentries)
        if not hasattr(self, "_reentry_call_counter"):
            self._reentry_call_counter = 0
        self._reentry_call_counter += 1
        self.state_manager._update_aggregate_stats()
        return {
            "successful_reentries": successful_reentries,
            "direct_employment_reentries": direct_employment_reentries,
            "attempts_by_age": attempts_by_age,
            "successes_by_age": successes_by_age,
            "n_attempts": n_attempts,
            "n_successes": n_successes,
            "n_direct_employment_reentries": len(direct_employment_reentries),
        }

    def _process_discouraged_workers(self, unemployed_discouraged: List[str]) -> int:
        if not unemployed_discouraged:
            return 0
        nilf_transitions = []
        for worker_id in unemployed_discouraged:
            ws = self.state_manager.workers.get(worker_id)
            if ws is None:
                continue
            if ws.worker.employment_status != "Unemployed":
                continue
            duration_months = ws.worker.unemployment_months
            if duration_months < 3:
                continue
            age = ws.worker.age
            duration_factor = min(duration_months / 12.0, 1.0)
            if age < 25:
                age_factor = 0.3
            elif age < 55:
                age_factor = 0.5
            elif age < 65:
                age_factor = 0.8
            else:
                age_factor = 1.0
            ei_receiving = (
                getattr(ws, "ei_eligible", False) and getattr(ws, "ei_weeks_remaining", 0) > 0
            )
            if ei_receiving:
                continue
            mnl_cfg = self.config.mnl
            base = max(
                float(getattr(mnl_cfg, "discouragement_rate", 0.004)),
                float(getattr(mnl_cfg, "discouragement_floor", 0.002)),
            )
            slope = float(getattr(mnl_cfg, "discouragement_slope", 0.02))
            cap = float(getattr(mnl_cfg, "discouragement_prob_cap", 0.08))
            nilf_prob = base + slope * duration_factor * age_factor
            nilf_prob = float(np.clip(nilf_prob, 0.0, cap))
            if np.random.random() < nilf_prob:
                _prior_is_srch_f = bool(getattr(ws.worker, "is_searching", False))
                ws.worker.employment_status = "NILF"
                ws.worker.is_searching = False
                ws.worker.unemployment_months = 0
                ws.months_unemployed = 0
                if getattr(self.config.mnl, "nilf_subtype_enabled", False):
                    self.state_manager.set_discouraged_nilf_subtype(
                        ws, source_state="unemployed_discouraged"
                    )
                nilf_transitions.append(worker_id)
                _inst_f = getattr(self, "_demographic_flow_instrumenter", None)
                if _inst_f is not None and _inst_f.enabled:
                    _imm_raw_f = getattr(ws.worker, "immigrant_status", False)
                    _is_imm_f = (
                        _imm_raw_f.lower() in ("true", "1", "yes", "immigrant")
                        if isinstance(_imm_raw_f, str)
                        else bool(_imm_raw_f)
                    )
                    _inst_f.record_u_to_nilf_rl_discouraged(
                        age=int(age), is_immigrant=_is_imm_f, prior_is_searching=_prior_is_srch_f
                    )
                    _inst_f.record_worker_lf_event(
                        worker_id=worker_id,
                        month=self.state_manager.current_month,
                        year=self.state_manager.current_year,
                        channel_tag="F",
                        lf_delta=-1 if _prior_is_srch_f else 0,
                        age=int(age),
                    )
        self.state_manager._update_aggregate_stats()
        return len(nilf_transitions)

    def _process_job_to_job_transitions(
        self, matches: List[Dict], employed_searchers: List[str], min_wages: Dict[str, float]
    ) -> Tuple[List[Dict], int]:
        employed_searcher_set = set(employed_searchers)
        j2j_transitions = 0
        j2j_offered = 0
        j2j_rejected_wage = 0
        j2j_origin_firms = set()
        unemployed_matches = []
        self._renewal_last_j2j_matches = []
        for match in matches:
            worker_id = match["worker_id"]
            new_wage = match.get(
                "negotiated_wage", match.get("offered_wage", match.get("final_wage", 0))
            )
            new_firm_id = match.get("firm_id", "")
            if worker_id in employed_searcher_set:
                j2j_offered += 1
                ws = self.state_manager.workers.get(worker_id)
                if ws is None:
                    continue
                worker = ws.worker
                current_wage = worker.current_wage
                min_acceptable = min_wages.get(worker_id, current_wage * 1.05)
                if new_wage < min_acceptable:
                    j2j_rejected_wage += 1
                    continue
                j2j_annual_cap = getattr(self.config.matching, "j2j_annual_cap", 2)
                if getattr(ws, "j2j_this_year", 0) >= j2j_annual_cap:
                    continue
                if not self.state_manager.capacity_commit_allowed(
                    worker_id, new_firm_id, match.get("vacancy_id"), "j2j", allow_same_employer=True
                ):
                    continue
                old_firm_id = self._renewal_begin_j2j_transaction(
                    worker_id, new_firm_id, match.get("vacancy_id")
                )
                j2j_match = match.copy()
                j2j_match["is_j2j"] = True
                j2j_match["old_firm_id"] = old_firm_id
                j2j_match["old_wage"] = current_wage
                new_industry = match.get("industry", worker.industry)
                ceiling = INDUSTRY_WAGE_CEILING.get(new_industry, 80.0)
                new_wage = min(new_wage, ceiling)
                worker.current_wage = new_wage
                worker.industry = new_industry
                worker.current_employer_id = new_firm_id
                ws.months_employed = 0
                ws.previous_wage = new_wage
                ws.total_job_transitions += 1
                ws.j2j_this_year = getattr(ws, "j2j_this_year", 0) + 1
                vacancy_id = match.get("vacancy_id")
                if vacancy_id and vacancy_id in self.state_manager.vacancies:
                    vacancy = self.state_manager.vacancies.pop(vacancy_id)
                    self.state_manager._record_vacancy_event("fill_j2j", -1)
                    new_firm_state = self.state_manager.firms.get(vacancy.firm_id)
                    if new_firm_state:
                        new_firm_state.vacancies.pop(vacancy.id, None)
                        if worker_id not in new_firm_state.employees:
                            new_firm_state.employees.append(worker_id)
                        new_firm_state.total_hires += 1
                        new_firm_state.total_vacancies_filled += 1
                elif vacancy_id:
                    for fs in self.state_manager.firms.values():
                        if vacancy_id in fs.vacancies:
                            fs.vacancies.pop(vacancy_id)
                            break
                    if new_firm_id:
                        nfs = self.state_manager.firms.get(new_firm_id)
                        if nfs:
                            if worker_id not in nfs.employees:
                                nfs.employees.append(worker_id)
                            nfs.total_hires += 1
                            nfs.total_vacancies_filled += 1
                self._worker_firm_map[worker_id] = new_firm_id
                self._renewal_finish_j2j_transaction(worker_id, new_firm_id)
                self._renewal_last_j2j_matches.append(j2j_match)
                if old_firm_id:
                    j2j_origin_firms.add(old_firm_id)
                j2j_transitions += 1
            else:
                unemployed_matches.append(match)
        self.last_matching_stats["j2j_offered"] = j2j_offered
        self.last_matching_stats["j2j_accepted"] = j2j_transitions
        self.last_matching_stats["j2j_rejected_wage"] = j2j_rejected_wage
        self._firms_with_j2j_departure = j2j_origin_firms
        return (unemployed_matches, j2j_transitions)

    def _process_wage_growth(self) -> int:
        updated_count = 0
        for worker_id, ws in self.state_manager.workers.items():
            worker = ws.worker
            if worker.employment_status != "Employed":
                continue
            current_wage = worker.current_wage
            industry = worker.industry
            tenure_months = ws.months_employed
            year_rates = self._wage_growth_by_year.get(self.current_year, {})
            industry_growth = year_rates.get(industry, WAGE_GROWTH_BY_INDUSTRY.get(industry, 0.025))
            tenure_years = tenure_months / 12.0
            tenure_premium = min(TENURE_PREMIUM_RATE * tenure_years, TENURE_PREMIUM_MAX)
            variation = np.random.uniform(-0.02, 0.02)
            total_growth = industry_growth + tenure_premium + variation
            new_wage = max(current_wage * (1 + total_growth), 10.0)
            ceiling = INDUSTRY_WAGE_CEILING.get(industry, 80.0)
            new_wage = min(new_wage, ceiling)
            worker.current_wage = new_wage
            updated_count += 1
        return updated_count

    def _get_replacement_workers(self, n_needed: int, source: str = "pool") -> List[str]:
        replacements = []
        if n_needed <= 0:
            return replacements
        active_in_dict = set(self.state_manager.workers.keys())
        active_in_list = set(self.worker_ids)
        all_active_ids = active_in_dict | active_in_list
        if self.available_pool_ids and len(self.available_pool_ids) >= n_needed:
            safe_pool = [
                wid for wid in sorted(self.available_pool_ids) if wid not in all_active_ids
            ]
            if len(safe_pool) >= n_needed:
                selected = np.random.choice(safe_pool, size=n_needed, replace=False)
                replacements = list(selected)
            else:
                replacements = safe_pool.copy()
                n_synthetic = n_needed - len(replacements)
                if n_synthetic > 0:
                    synthetic_ids = self._generate_synthetic_worker_ids(n_synthetic, source)
                    replacements.extend(synthetic_ids)
            for wid in replacements:
                self.available_pool_ids.discard(wid)
                self.sampled_worker_ids.add(wid)
        elif self.available_pool_ids:
            safe_pool = [
                wid for wid in sorted(self.available_pool_ids) if wid not in all_active_ids
            ]
            replacements = safe_pool.copy()
            self.sampled_worker_ids.update(replacements)
            self.available_pool_ids.clear()
            n_synthetic = n_needed - len(replacements)
            if n_synthetic > 0:
                synthetic_ids = self._generate_synthetic_worker_ids(n_synthetic, source)
                replacements.extend(synthetic_ids)
        else:
            replacements = self._generate_synthetic_worker_ids(n_needed, source)
        return replacements

    def _generate_synthetic_worker_ids(self, n: int, source: str) -> List[str]:
        base_id = len(self.sampled_worker_ids) + 1000000
        ids = []
        for i in range(n):
            wid = f"syn_{source}_{self.current_year}_{base_id + i}"
            ids.append(wid)
            self.sampled_worker_ids.add(wid)
        return ids

    def _renewal_monthly_entry(self, force=False):
        from age_renewal import enabled as _age_enabled

        if _age_enabled():
            if self._renewal_pending is not None:
                raise RuntimeError(
                    "Monthly entries would be counted twice under age-cohort renewal"
                )
            return

    def _renewal_assert_firm_employment(self):
        if getattr(self, "firm_employment", None) is None:
            raise RuntimeError("firm_employment tensor missing")
        _bad = []
        for _fid, _fs in self.state_manager.firms.items():
            _idx = self._firm_id_to_idx.get(_fid)
            if _idx is None:
                _bad.append((_fid, "missing-index"))
                continue
            _flag = self.firm_active[_idx]
            _active = bool(_flag.item() if hasattr(_flag, "item") else _flag)
            if not _active:
                continue
            _expected = self._workplace_staff_count(_fid, _fs)
            _fs.firm.n_employees = _expected
            _value = self.firm_employment[_idx]
            _observed = int(_value.item() if hasattr(_value, "item") else _value)
            if _observed != _expected:
                _bad.append((_fid, _observed, _expected))
        if _bad:
            raise RuntimeError("firm-employment reconciliation failed: %s" % (_bad[:5],))

    def _renewal_assert_population_registry(self):
        return True

    def _renewal_begin_j2j_transaction(self, worker_id, new_firm_id, vacancy_id):
        _ws = self.state_manager.workers.get(worker_id)
        if _ws is None or _ws.worker.employment_status != "Employed":
            raise RuntimeError(
                "invalid J2J origin worker state: %r"
                % ((worker_id, None if _ws is None else _ws.worker.employment_status),)
            )
        _old_firm_id = getattr(_ws.worker, "current_employer_id", None)
        _memberships = [
            (_fid, _fs.employees.count(worker_id))
            for _fid, _fs in self.state_manager.firms.items()
            if worker_id in _fs.employees
        ]
        _cached = self._worker_firm_map.get(worker_id)
        _is_renewal_created = worker_id in self._renewal_audit_created
        _empty_registry = _old_firm_id is None and _memberships == [] and (_cached is None)
        if _empty_registry and (not _is_renewal_created):
            self._renewal_audit_earlier_j2j_empty_registry += 1
            self._renewal_audit_earlier_j2j_workers.add(worker_id)
            if not new_firm_id or new_firm_id not in self.state_manager.firms:
                raise RuntimeError("invalid J2J destination firm: %r" % ((worker_id, new_firm_id),))
            return None
        if _old_firm_id not in self.state_manager.firms or _memberships != [(_old_firm_id, 1)]:
            raise RuntimeError(
                "invalid J2J origin registry: %r"
                % ((worker_id, _old_firm_id, _memberships, self._worker_firm_map.get(worker_id)),)
            )
        if not new_firm_id or new_firm_id not in self.state_manager.firms:
            raise RuntimeError("invalid J2J destination firm: %r" % ((worker_id, new_firm_id),))
        _vacancy = self.state_manager.vacancies.get(vacancy_id)
        if _vacancy is not None:
            _vacancy_firm = (
                _vacancy.get("firm_id")
                if isinstance(_vacancy, dict)
                else getattr(_vacancy, "firm_id", None)
            )
            if _vacancy_firm != new_firm_id:
                raise RuntimeError(
                    "vacancy/destination mismatch: %r"
                    % ((worker_id, vacancy_id, _vacancy_firm, new_firm_id),)
                )
        if _cached != _old_firm_id:
            self._worker_firm_map[worker_id] = _old_firm_id
        _old_state = self.state_manager.firms[_old_firm_id]
        _old_state.employees.remove(worker_id)
        _old_state.total_separations += 1
        if hasattr(_old_state, "firm"):
            _old_state.firm.n_employees = len(_old_state.employees)
        return _old_firm_id

    def _renewal_finish_j2j_transaction(self, worker_id, new_firm_id):
        _ws = self.state_manager.workers.get(worker_id)
        _memberships = [
            (_fid, _fs.employees.count(worker_id))
            for _fid, _fs in self.state_manager.firms.items()
            if worker_id in _fs.employees
        ]
        _status = None if _ws is None else _ws.worker.employment_status
        _employer = None if _ws is None else getattr(_ws.worker, "current_employer_id", None)
        _mapped = self._worker_firm_map.get(worker_id)
        if (
            _status != "Employed"
            or _employer != new_firm_id
            or _mapped != new_firm_id
            or (_memberships != [(new_firm_id, 1)])
        ):
            raise RuntimeError(
                "invalid J2J destination registry: %r"
                % ((worker_id, _status, _employer, _mapped, _memberships),)
            )
        _new_state = self.state_manager.firms[new_firm_id]
        if hasattr(_new_state, "firm"):
            _new_state.firm.n_employees = len(_new_state.employees)
        return True

    def _renewal_entry_audit_summary(self):
        return {"enabled": False}

    def _renewal_assert_worker_slot_order(self, allow_unmapped_tail=False):
        _ordered = list(self.worker_ids)
        _state_ids = set(self.state_manager.workers.keys())
        _ordered_set = set(_ordered)
        if len(_ordered) != len(_ordered_set):
            raise RuntimeError("duplicate worker_ids in slot-ordered list")
        if _ordered_set != _state_ids:
            raise RuntimeError(
                "Worker IDs differ: only in the list %s, only in the state manager %s"
                % (list(_ordered_set - _state_ids)[:5], list(_state_ids - _ordered_set)[:5])
            )
        _table = getattr(self.fast_obs, "state_table", None)
        _mapping = getattr(_table, "worker_id_to_idx", None)
        if _mapping is None:
            raise RuntimeError("fast-observation worker map is unavailable")
        _permuted = []
        _unmapped = []
        for _i, _wid in enumerate(_ordered):
            _mapped = _mapping.get(_wid)
            if _mapped is None:
                _unmapped.append((_i, _wid))
            elif int(_mapped) != _i:
                _permuted.append((_i, _wid, int(_mapped)))
        if _permuted:
            raise RuntimeError(
                "worker_ids order no longer matches tensor slots; first mismatches (list_idx, id, tensor_idx)=%s"
                % (_permuted[:5],)
            )
        if _unmapped:
            _indices = [_i for _i, _wid in _unmapped]
            _tail = list(range(len(_ordered) - len(_unmapped), len(_ordered)))
            if not allow_unmapped_tail or _indices != _tail:
                raise RuntimeError(
                    "unmapped workers are not the newly appended tail: %s" % (_unmapped[:5],)
                )
        return len(_unmapped)

    def _renewal_resize_obs(self):
        self._renewal_assert_worker_slot_order(allow_unmapped_tail=True)
        self.n_firms = len(self.firm_ids)
        self._make_specs()
        self.fast_obs.resize_all(self.n_workers)
        self.fast_obs.state_table.sync_from_state_manager(self.state_manager, self.worker_ids)
        self._renewal_assert_worker_slot_order(allow_unmapped_tail=False)
        self.state_manager._state_table_ref = self.fast_obs.state_table

    def _process_demographics_clamped(self) -> Dict[str, Any]:
        from age_renewal import enabled as _age_enabled, clamp_disabled

        if _age_enabled():
            return clamp_disabled(self)
        if self.demographic_tracker is None:
            self.demographic_tracker = DemographicTracker()
        _flow_inst = getattr(self, "_demographic_flow_instrumenter", None)
        _flow_enabled = _flow_inst is not None and _flow_inst.enabled
        theoretical_deaths = []
        for worker_id, ws in self.state_manager.workers.items():
            age = ws.worker.age
            if self.mortality_rates and age in self.mortality_rates:
                death_prob = self.mortality_rates[age]
            elif age < 50:
                death_prob = MORTALITY_RATES_BY_AGE[0, 49]
            elif age < 65:
                death_prob = MORTALITY_RATES_BY_AGE[50, 64]
            elif age < 75:
                death_prob = MORTALITY_RATES_BY_AGE[65, 74]
            else:
                death_prob = MORTALITY_RATES_BY_AGE[75, 100]
            if np.random.random() < death_prob:
                theoretical_deaths.append(worker_id)
                if _flow_enabled:
                    _imm_raw = getattr(ws.worker, "immigrant_status", False)
                    _is_imm = (
                        _imm_raw.lower() in ("true", "1", "yes", "immigrant")
                        if isinstance(_imm_raw, str)
                        else bool(_imm_raw)
                    )
                    _flow_inst.record_theoretical_death(age=int(age), is_immigrant=_is_imm)
        n_active = len(self.state_manager.workers)
        n_theoretical_emigrants = int(n_active * ANNUAL_EMIGRATION_RATE)
        remaining_workers = [
            w for w in self.state_manager.workers.keys() if w not in theoretical_deaths
        ]
        theoretical_emigrants = []
        if n_theoretical_emigrants > 0 and remaining_workers:
            theoretical_emigrants = list(
                np.random.choice(
                    remaining_workers,
                    size=min(n_theoretical_emigrants, len(remaining_workers)),
                    replace=False,
                )
            )
            if _flow_enabled:
                for _em_id in theoretical_emigrants:
                    _em_ws = self.state_manager.workers.get(_em_id)
                    if _em_ws is not None:
                        _em_age = int(getattr(_em_ws.worker, "age", -1))
                        _em_imm_raw = getattr(_em_ws.worker, "immigrant_status", False)
                        _em_is_imm = (
                            _em_imm_raw.lower() in ("true", "1", "yes", "immigrant")
                            if isinstance(_em_imm_raw, str)
                            else bool(_em_imm_raw)
                        )
                        _flow_inst.record_theoretical_emigrant(age=_em_age, is_immigrant=_em_is_imm)
        n_theoretical_immigrants = int(n_active * ANNUAL_IMMIGRATION_RATE)
        n_theoretical_entrants = int(n_active * NEW_ENTRANT_RATE)
        if _flow_enabled:
            _flow_inst.record_theoretical_immigrant(n=n_theoretical_immigrants)
            _flow_inst.record_theoretical_youth_entrant(n=n_theoretical_entrants)
        _source_by_worker = {wid: "mortality" for wid in theoretical_deaths}
        for wid in theoretical_emigrants:
            _source_by_worker[wid] = "emigration"
        workers_to_remove = sorted(set(theoretical_deaths + theoretical_emigrants))
        n_to_replace = len(workers_to_remove)
        replacement_ids = self._get_replacement_workers(n_to_replace, "replacement")
        actual_replaced = 0
        for old_id, new_id in zip(workers_to_remove, replacement_ids):
            self._replace_worker(
                old_id, new_id, fix2a_source=_source_by_worker.get(old_id, "other")
            )
            actual_replaced += 1
        n_deaths = len(theoretical_deaths)
        n_emig = len(theoretical_emigrants)
        n_repl_mortality = sum(
            (1 for wid in workers_to_remove if _source_by_worker.get(wid) == "mortality")
        )
        n_repl_emigration = sum(
            (1 for wid in workers_to_remove if _source_by_worker.get(wid) == "emigration")
        )
        self.demographic_tracker.record_step(
            deaths=n_deaths,
            emigrants=0,
            immigrants=n_theoretical_immigrants,
            new_entrants=n_theoretical_entrants,
            actual_replaced=n_repl_mortality,
            source="mortality",
        )
        self.demographic_tracker.record_step(
            deaths=0,
            emigrants=n_emig,
            immigrants=0,
            new_entrants=0,
            actual_replaced=n_repl_emigration,
            source="emigration",
        )
        theoretical_net = (
            n_theoretical_immigrants
            + n_theoretical_entrants
            - len(theoretical_deaths)
            - len(theoretical_emigrants)
        )
        self._theoretical_population += theoretical_net
        return {
            "theoretical": {
                "deaths": len(theoretical_deaths),
                "emigrants": len(theoretical_emigrants),
                "immigrants": n_theoretical_immigrants,
                "new_entrants": n_theoretical_entrants,
                "net_change": theoretical_net,
            },
            "actual": {"replaced": actual_replaced, "population": len(self.state_manager.workers)},
            "theoretical_population": self._theoretical_population,
        }

    def _replace_worker(self, old_id: str, new_id: str, fix2a_source: str = "other"):
        from age_renewal import enabled as age_renewal_enabled

        if age_renewal_enabled():
            raise RuntimeError(
                "Replacement by resampling cannot run together with age-cohort renewal"
            )
        from .state_manager import WorkerState
        from .matching_mechanism_kround import Worker

        if not hasattr(self, "_collision_count"):
            self._collision_count = 0
        if not hasattr(self, "_leak_count"):
            self._leak_count = 0
        MAX_ACCEPTABLE_LEAKS = max(50, int(self.n_workers * 0.05))
        if new_id in self.state_manager.workers or new_id in self.worker_ids:
            self._collision_count += 1
            safe_id = f"SYN_collision_{self.current_step}_{self._collision_count}"
            new_id = safe_id
        count_before = len(self.state_manager.workers)
        old_ws = self.state_manager.workers.get(old_id)
        _flow_inst = getattr(self, "_demographic_flow_instrumenter", None)
        _flow_enabled = _flow_inst is not None and _flow_inst.enabled
        if _flow_enabled and old_ws is not None:
            _old_age = int(getattr(old_ws.worker, "age", -1))
            _old_imm_raw = getattr(old_ws.worker, "immigrant_status", False)
            _old_is_imm = (
                _old_imm_raw.lower() in ("true", "1", "yes", "immigrant")
                if isinstance(_old_imm_raw, str)
                else bool(_old_imm_raw)
            )
            _old_es = getattr(old_ws.worker, "employment_status", None)
            if _old_es == "Employed":
                _old_status = "E"
            elif _old_es == "Unemployed":
                _old_status = (
                    "U_search" if getattr(old_ws.worker, "is_searching", False) else "U_no_search"
                )
            elif _old_es == "NILF":
                _old_status = (
                    "NILF_ineligible"
                    if getattr(old_ws, "nilf_subtype", 0) == 1
                    else "NILF_eligible"
                )
            else:
                _old_status = "unknown"
        else:
            _old_age, _old_is_imm, _old_status = (-1, False, "unknown")
        from population_participation import remove_replaced_worker

        remove_replaced_worker(self, old_id)
        if old_id in self._worker_firm_map:
            firm_id = self._worker_firm_map[old_id]
            if firm_id in self.state_manager.firms:
                fs = self.state_manager.firms[firm_id]
                if old_id in fs.employees:
                    fs.employees.remove(old_id)
            del self._worker_firm_map[old_id]
        if old_id in self.state_manager.workers:
            del self.state_manager.workers[old_id]
        if old_id in self.worker_ids:
            idx = self.worker_ids.index(old_id)
            self.worker_ids[idx] = new_id
        else:
            self.worker_ids.append(new_id)
        _population_donor_row = None
        if (
            self.full_population_df is not None
            and new_id in self.full_population_df["worker_id"].values
        ):
            row = self.full_population_df[self.full_population_df["worker_id"] == new_id].iloc[0]
            _population_donor_row = row
            naics_raw = row.get("naics_18", 8)
            if pd.isna(naics_raw):
                naics = 8
            else:
                naics = int(naics_raw)
            sector_raw = row.get("sector_id", 0)
            sector = int(sector_raw) if not pd.isna(sector_raw) else 0
            exp_raw = row.get("experience_total", 5.0)
            experience = float(exp_raw) if not pd.isna(exp_raw) else 5.0
            da_raw = row.get("da_id")
            da_id = int(da_raw) if pd.notna(da_raw) else None
            emp_status = row.get("employment_status", "Unemployed")
            is_employed = emp_status in ["Employed", "E"]
            emp_status_norm = (
                "Employed" if is_employed else "NILF" if emp_status == "NILF" else "Unemployed"
            )
            wage_raw = row.get("hourly_wage", 20.0)
            wage = float(wage_raw) if is_employed and (not pd.isna(wage_raw)) else 0.0
            wage = min(wage, INDUSTRY_WAGE_CEILING.get(naics, 80.0))
            is_searching = not is_employed and emp_status_norm != "NILF"
            worker = Worker(
                id=new_id,
                age=int(row.get("age", 30)),
                skill_level=(
                    float(row.get("skill_level", 2.0))
                    if not pd.isna(row.get("skill_level"))
                    else 2.0
                ),
                education=int(row.get("education", 2)) if not pd.isna(row.get("education")) else 2,
                industry=naics,
                sector=sector,
                experience_years=experience,
                cma=int(row.get("cma", 535)) if not pd.isna(row.get("cma")) else 535,
                employment_status=emp_status_norm,
                current_wage=wage,
                is_searching=is_searching,
                da_id=da_id,
                sex=normalize_sex_value(row.get("sex")),
            )
        else:
            age = np.random.randint(20, 55)
            education = np.random.choice([1, 2, 3, 4], p=[0.2, 0.35, 0.3, 0.15])
            industry = np.random.choice(YOUTH_INDUSTRIES, p=YOUTH_INDUSTRY_PROBS)
            skill = int(round(min(max(education + np.random.uniform(-0.5, 0.5), 0), 4)))
            is_employed = np.random.random() < 0.7
            wage = 15.0 + education * 3.0 + (age - 22) * 0.15 if is_employed else 0.0
            wage = min(wage, INDUSTRY_WAGE_CEILING.get(industry, 80.0))
            emp_status_norm = "Employed" if is_employed else "Unemployed"
            is_searching = not is_employed
            synthetic_cma, synthetic_da = self._sample_cma_and_da(worker_type="general")
            worker = Worker(
                id=new_id,
                age=age,
                skill_level=skill,
                education=education,
                industry=industry,
                sector=0,
                experience_years=float(max(0, age - 22)),
                cma=synthetic_cma,
                employment_status=emp_status_norm,
                current_wage=wage,
                is_searching=is_searching,
                da_id=synthetic_da,
                sex=sample_sex_for_age(age),
            )
        _months_employed_replacement = 0 if not is_employed else int(np.random.randint(0, 24))
        _months_unemployed_replacement = 0 if is_employed else int(np.random.randint(0, 6))
        ws = WorkerState(
            worker=worker,
            ei_eligible=False,
            ei_weeks_remaining=0,
            months_employed=_months_employed_replacement,
            months_unemployed=_months_unemployed_replacement,
            previous_wage=wage,
            previous_tenure_months=0,
        )
        from population_participation import prepare_replacement

        prepare_replacement(self, ws, old_id, fix2a_source, _population_donor_row)
        if worker.employment_status == "NILF" and getattr(
            self.config.mnl, "nilf_subtype_enabled", False
        ):
            self.state_manager.assign_exogenous_nilf_subtype(ws)
        self.state_manager.workers[new_id] = ws
        self.sampled_worker_ids.discard(old_id)
        self.sampled_worker_ids.add(new_id)
        if _flow_enabled:
            _new_age = int(getattr(worker, "age", -1))
            _new_imm_raw = getattr(worker, "immigrant_status", False)
            _new_is_imm = (
                _new_imm_raw.lower() in ("true", "1", "yes", "immigrant")
                if isinstance(_new_imm_raw, str)
                else bool(_new_imm_raw)
            )
            _new_es = getattr(worker, "employment_status", None)
            if _new_es == "Employed":
                _new_status = "E"
            elif _new_es == "Unemployed":
                _new_status = (
                    "U_search" if getattr(worker, "is_searching", False) else "U_no_search"
                )
            elif _new_es == "NILF":
                _new_status = (
                    "NILF_ineligible" if getattr(ws, "nilf_subtype", 0) == 1 else "NILF_eligible"
                )
            else:
                _new_status = "unknown"
            _flow_inst.record_actual_worker_replacement(
                old_age=_old_age,
                old_is_immigrant=_old_is_imm,
                new_age=_new_age,
                new_is_immigrant=_new_is_imm,
                source=fix2a_source,
                old_status=_old_status,
                new_status=_new_status,
            )
        count_after = len(self.state_manager.workers)
        if count_after != count_before:
            self._leak_count += 1
            if self._leak_count > MAX_ACCEPTABLE_LEAKS:
                raise RuntimeError(
                    f"CRITICAL: {self._leak_count} workers leaked (>{MAX_ACCEPTABLE_LEAKS}), run is compromised. Last leak: {old_id}->{new_id}"
                )

    def _verify_population_integrity(self, context: str = "unknown"):
        sm_ids = set(self.state_manager.workers.keys())
        list_ids = set(self.worker_ids)
        if len(self.worker_ids) != len(list_ids):
            seen = set()
            dupes = []
            for wid in self.worker_ids:
                if wid in seen:
                    dupes.append(wid)
                seen.add(wid)
            raise RuntimeError(f"DUPLICATE worker_ids in list: {len(dupes)} duplicates found")
        in_list_not_sm = list_ids - sm_ids
        in_sm_not_list = sm_ids - list_ids
        if in_list_not_sm or in_sm_not_list:
            raise RuntimeError(
                f"Worker registers differ at {context}: {len(in_list_not_sm)} only in the list, {len(in_sm_not_list)} only in the state manager"
            )

    @staticmethod
    def _make_quarterly_accum() -> dict:
        return {
            "months": 0,
            "growth_posted": 0,
            "replacement_sep": 0,
            "replacement_j2j": 0,
            "replenish": 0,
            "ensure": 0,
            "unemp_seekers": 0,
            "unemp_offered": 0,
            "unemp_matched": 0,
            "offers_to_unemp": 0,
            "emp_seekers": 0,
            "emp_offered": 0,
            "emp_matched": 0,
            "j2j_offered": 0,
            "j2j_accepted": 0,
            "j2j_rejected_wage": 0,
        }

    def _accumulate_quarterly_stats(self):
        ms = self.last_matching_stats
        q = self._q_accum
        q["months"] += 1
        q["growth_posted"] += ms.get("vac_posted_growth", 0)
        q["replacement_sep"] += ms.get("vac_posted_replacement_sep", 0)
        q["replacement_j2j"] += ms.get("vac_posted_replacement_j2j", 0)
        q["replenish"] += ms.get("vacancies_injected_replenish", 0)
        q["ensure"] += ms.get("vacancies_injected_ensure", 0)
        q["unemp_seekers"] += ms.get("n_unemployed_seekers", 0)
        q["unemp_offered"] += ms.get("n_unemployed_offered", 0)
        q["unemp_matched"] += ms.get("n_unemployed_matched", 0)
        q["offers_to_unemp"] += ms.get("offers_to_unemployed", 0)
        q["emp_seekers"] += ms.get("n_employed_seekers", 0)
        q["emp_offered"] += ms.get("n_employed_offered", 0)
        q["emp_matched"] += ms.get("n_employed_matched", 0)
        q["j2j_offered"] += ms.get("j2j_offered", 0)
        q["j2j_accepted"] += ms.get("j2j_accepted", 0)
        q["j2j_rejected_wage"] += ms.get("j2j_rejected_wage", 0)

    def _log_quarterly_diagnostics(self):
        ta = self.config.target_adaptation
        if not getattr(ta, "enabled", False):
            return
        if self.current_month == 0 or self.current_month % ta.frequency_months != 0:
            return
        active = self.firm_active
        if active is None:
            return
        self._q_accum = self._make_quarterly_accum()

    def process_firm_exits(self) -> int:
        return self._process_firm_exits()

    def _process_firm_exits(self) -> int:
        if self.firm_active is None:
            return 0
        self._sync_firm_tensors()
        active_mask = self.firm_active
        active_idx = torch.where(active_mask)[0]
        if len(active_idx) == 0:
            return 0
        employment = self.firm_employment[active_idx]
        target_size = self.firm_target_size[active_idx]
        months_empty = self.firm_months_empty[active_idx]
        months_struggling = self.firm_months_struggling[active_idx]
        industries = self.firm_industries[active_idx]
        exit_prob = torch.zeros(len(active_idx), device=self.device)
        empty_mask = employment == 0
        long_empty_mask = empty_mask & (months_empty >= FIRM_EXIT_PARAMS["empty_threshold_months"])
        exit_prob[long_empty_mask] = FIRM_EXIT_PARAMS["empty_exit_prob"]
        self.firm_months_empty[active_idx[empty_mask]] += 1
        self.firm_months_empty[active_idx[~empty_mask]] = 0
        employment_ratio = employment.float() / (target_size.float() + 1e-06)
        struggling_mask = employment_ratio < FIRM_EXIT_PARAMS["struggling_ratio"]
        long_struggling_mask = struggling_mask & (
            months_struggling >= FIRM_EXIT_PARAMS["struggling_threshold_months"]
        )
        exit_prob[long_struggling_mask & ~long_empty_mask] = FIRM_EXIT_PARAMS[
            "struggling_exit_prob"
        ]
        self.firm_months_struggling[active_idx[struggling_mask]] += 1
        self.firm_months_struggling[active_idx[~struggling_mask]] = 0
        low_risk_mask = exit_prob < 0.01
        small_mask = target_size < FIRM_EXIT_PARAMS["small_threshold"]
        medium_mask = (target_size >= FIRM_EXIT_PARAMS["small_threshold"]) & (
            target_size < FIRM_EXIT_PARAMS["medium_threshold"]
        )
        large_mask = target_size >= FIRM_EXIT_PARAMS["medium_threshold"]
        exit_prob[low_risk_mask & small_mask] = FIRM_EXIT_PARAMS["base_exit_small"]
        exit_prob[low_risk_mask & medium_mask] = FIRM_EXIT_PARAMS["base_exit_medium"]
        exit_prob[low_risk_mask & large_mask] = FIRM_EXIT_PARAMS["base_exit_large"]
        high_turnover_industries = torch.tensor(
            FIRM_EXIT_PARAMS["high_turnover_industries"], device=self.device
        )
        high_turnover_mask = torch.isin(industries, high_turnover_industries)
        exit_prob[high_turnover_mask] *= FIRM_EXIT_PARAMS["high_turnover_multiplier"]
        random_draws = torch.rand(len(active_idx), device=self.device)
        exiting_mask = random_draws < exit_prob
        exiting_idx = active_idx[exiting_mask]
        n_exits = len(exiting_idx)
        if n_exits == 0:
            return 0
        for firm_idx in exiting_idx:
            firm_idx_int = firm_idx.item()
            firm_id = self.firm_ids[firm_idx_int]
            fs = self.state_manager.firms.get(firm_id)
            if fs is None:
                continue
            employees_to_layoff = list(fs.employees)
            for worker_id in employees_to_layoff:
                ws = self.state_manager.workers.get(worker_id)
                if ws is not None:
                    old_wage_at_layoff = float(ws.worker.current_wage)
                    ws.previous_tenure_months = ws.months_employed
                    ws.previous_wage = old_wage_at_layoff
                    ws.months_employed = 0
                    ws.worker.employment_status = "Unemployed"
                    self._employment_route_ids_this_step["firm_exit_displacements"].add(worker_id)
                    ws.worker.is_searching = True
                    ws.worker.unemployment_months = 0
                    ws.worker.current_wage = 0.0
                    if hasattr(ws.worker, "current_employer_id"):
                        ws.worker.current_employer_id = None
                    self.state_manager.apply_layoff_scarring(worker_id)
                    from environment.state_manager import lwf_draw_retention as _lwf_draw

                    ws.separation_reason = "job_loser"
                    _sex = int(getattr(ws.worker, "sex", 1) or 1)
                    ws.lwf_retention_draw = _lwf_draw(
                        previous_tenure_months=ws.previous_tenure_months, sex=_sex
                    )
                    from environment.state_manager import grant_ei_if_eligible

                    self.state_manager._ei_period_new_separations += 1
                    if grant_ei_if_eligible(ws, self.config.ei):
                        self.state_manager._ei_period_new_grants += 1
                    if worker_id in self._worker_firm_map:
                        del self._worker_firm_map[worker_id]
            fs.employees = []
            if fs.vacancies:
                self._vac_exit_purged_this_step = getattr(
                    self, "_vac_exit_purged_this_step", 0
                ) + len(fs.vacancies)
                self.state_manager.remove_vacancies(
                    list(fs.vacancies.keys()), reason="firm_exit_purge"
                )
        self.firm_active[exiting_idx] = False
        self.firm_employment[exiting_idx] = 0
        self.firm_exits_this_year += n_exits
        return n_exits

    def process_firm_entries(self) -> int:
        from .state_manager import FirmState
        from .matching_mechanism_kround import Firm

        if self.firm_active is None:
            return 0
        n_total = self.n_firms
        n_active = int(self.firm_active.sum())
        n_inactive = n_total - n_active
        replacement_entries = int(
            self.firm_exits_this_year * FIRM_ENTRY_PARAMS["replacement_rate"] / 12
        )
        growth_entries = int(n_total * FIRM_ENTRY_PARAMS["growth_rate"])
        min_entries = max(1, n_inactive // 10)
        target_entries = max(replacement_entries + growth_entries, min_entries)
        target_entries = min(target_entries, n_inactive)
        if target_entries == 0:
            return 0
        inactive_idx = torch.where(~self.firm_active)[0]
        n_entries = min(target_entries, len(inactive_idx))
        if n_entries == 0:
            return 0
        entry_idx = inactive_idx[:n_entries]
        for i, firm_idx in enumerate(entry_idx):
            firm_idx_int = firm_idx.item()
            firm_id = self.firm_ids[firm_idx_int]
            scaled_dist = self._get_scaled_entry_size_distribution()
            size_category = self._sample_new_firm_size_category()
            min_size, max_size, _ = scaled_dist[size_category]
            target_size = int(
                torch.randint(min_size, max_size + 1, (1,), device=self.device).item()
            )
            target_size = min(target_size, 20)
            _predecessor_target = int(self.firm_target_size[firm_idx_int].item())
            import capacity_lineage as _capacity_lineage

            _capacity_lineage_authority = None
            _capacity_fresh_target = _predecessor_target == 0
            _capacity_lineage_authority = _capacity_lineage.prepare_entry(
                self, firm_id, _predecessor_target, target_size
            )
            target_size = _capacity_lineage_authority["target"]
            _capacity_fresh_target = _capacity_lineage_authority["fresh_target_assignment"]
            industry = self._sample_new_firm_industry()
            from config.default_config import NAICS_TO_SECTOR

            sector = NAICS_TO_SECTOR.get(industry, 0)
            da_id = self._sample_new_firm_location(industry)
            from market_mechanisms import replace_identity, entry_cma

            da_id = replace_identity(self, firm_id, industry, da_id)
            base_wage = self._get_industry_average_wage(industry)
            wage_lo, wage_hi = self.config.matching.new_firm_wage_range
            posted_wage = base_wage * (
                wage_lo + (wage_hi - wage_lo) * torch.rand(1, device=self.device).item()
            )
            self.firm_active[firm_idx_int] = True
            self.firm_target_size[firm_idx_int] = target_size
            self.firm_base_size[firm_idx_int] = target_size
            self.firm_employment[firm_idx_int] = 0
            self.firm_industries[firm_idx_int] = industry
            self.firm_posted_wages[firm_idx_int] = posted_wage
            self.firm_months_empty[firm_idx_int] = 0
            self.firm_months_struggling[firm_idx_int] = 0
            self.firm_age_months[firm_idx_int] = 0
            self._ta_under_streak[firm_idx_int] = 0
            self._ta_lowfill_streak[firm_idx_int] = 0
            self._ta_over_streak[firm_idx_int] = 0
            self._ta_prev_total_hires[firm_idx_int] = 0
            self._ta_hires_ring[firm_idx_int] = 0
            if self.firm_da_ids is not None:
                self.firm_da_ids[firm_idx_int] = da_id
            fs = self.state_manager.firms.get(firm_id)
            if fs is not None:
                fs.firm.industry = industry
                fs.firm.sector = sector
                fs.firm.target_size = target_size
                fs.firm.n_employees = 0
                if hasattr(fs.firm, "da_id"):
                    fs.firm.da_id = da_id
                fs.employees = []
                fs.current_wage_offer = posted_wage
            else:
                new_firm = Firm(
                    id=firm_id,
                    industry=industry,
                    sector=sector,
                    n_employees=0,
                    cma=535,
                    target_size=target_size,
                    da_id=da_id,
                )
                fs = FirmState(firm=new_firm)
                fs.current_wage_offer = posted_wage
                self.state_manager.firms[firm_id] = fs
            if _capacity_lineage_authority is not None:
                _capacity_lineage.commit_entry(
                    self,
                    firm_id,
                    target_size,
                    br_scaling_applied=bool(_capacity_fresh_target and False),
                )
            initial_vac = max(FIRM_ENTRY_PARAMS.get("initial_vacancies_per_firm", 3), 3)
            n_vacancies = max(initial_vac, min(target_size, 5))
            if _capacity_lineage_authority is not None:
                n_vacancies = _capacity_lineage.initial_vacancy_count(self, firm_id, n_vacancies)
            import capacity_budget as _capacity_budget

            n_vacancies = min(n_vacancies, _capacity_budget.remaining_slots(self, firm_id))
            for v in range(n_vacancies):
                vac_id = f"{firm_id}_vac_{self.current_year}_{self.current_month}_{v}"
                vacancy = Vacancy(
                    id=vac_id,
                    firm_id=firm_id,
                    industry=industry,
                    sector=sector,
                    skill_required=get_industry_skill_requirement(industry),
                    education_required=get_industry_skill_requirement(industry),
                    offered_wage=posted_wage,
                    cma=entry_cma(fs.firm),
                    da_id=da_id,
                )
                self.state_manager.add_vacancies([vacancy], reason="firm_entry_addition")
        self.firm_entries_this_year += n_entries
        return n_entries

    def _get_scaled_entry_size_distribution(self) -> dict:
        import math

        n_workers = self.config.simulation.n_workers
        n_firms = self.config.simulation.n_firms
        avg = max(1.0, n_workers / n_firms)
        multipliers = FIRM_ENTRY_PARAMS.get("entry_size_multipliers")
        static_dist = FIRM_ENTRY_PARAMS["new_firm_size_distribution"]
        if multipliers is None:
            return static_dist
        scaled = {}
        for category, (lo_mult, hi_mult) in multipliers.items():
            prob = static_dist[category][2]
            if lo_mult is None:
                lo = 1
            else:
                lo = max(1, math.ceil(lo_mult * avg))
            hi = max(lo, math.ceil(hi_mult * avg))
            scaled[category] = (lo, hi, prob)
        return scaled

    def _sample_new_firm_size_category(self) -> str:
        categories = list(FIRM_ENTRY_PARAMS["new_firm_size_distribution"].keys())
        probs = [v[2] for v in FIRM_ENTRY_PARAMS["new_firm_size_distribution"].values()]
        idx = torch.multinomial(torch.tensor(probs, device=self.device), 1).item()
        return categories[idx]

    def _sample_new_firm_industry(self) -> int:
        if self.firm_industries is None:
            return 8
        for industry in sorted(VALID_INDUSTRIES):
            in_industry = (self.firm_industries == industry) & self.firm_active
            n_firms = int(in_industry.sum())
            if n_firms < MIN_FIRMS_PER_INDUSTRY:
                return industry
        active_industries = self.firm_industries[self.firm_active]
        if len(active_industries) == 0:
            return 8
        idx = torch.randint(0, len(active_industries), (1,), device=self.device).item()
        return int(active_industries[idx].item())

    def _sample_new_firm_location(self, industry: int) -> int:
        if self.firm_da_ids is None:
            return 0
        industry_mask = (self.firm_industries == industry) & self.firm_active
        if industry_mask.sum() == 0:
            valid_das = self.firm_da_ids[self.firm_active]
            if len(valid_das) > 0:
                idx = torch.randint(0, len(valid_das), (1,), device=self.device).item()
                return int(valid_das[idx].item())
            return 0
        industry_das = self.firm_da_ids[industry_mask]
        idx = torch.randint(0, len(industry_das), (1,), device=self.device).item()
        return int(industry_das[idx].item())

    def _get_industry_average_wage(self, industry: int) -> float:
        if self.firm_posted_wages is None:
            return 18.0
        industry_mask = (self.firm_industries == industry) & self.firm_active
        if industry_mask.sum() == 0:
            return 18.0
        return float(self.firm_posted_wages[industry_mask].mean().item())
