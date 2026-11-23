from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


@dataclass
class MNLConfig:
    beta_wage: float = 2.5
    beta_skill_match: float = 1.2
    beta_commute: float = -0.08
    estimated_beta_wage: float = 2.5
    estimated_beta_commute: float = -0.2
    discouragement_rate: float = 0.004
    discouragement_floor: float = 0.002
    discouragement_slope: float = 0.02
    discouragement_prob_cap: float = 0.08
    nilf_life_event_reentry_rate: float = 0.008
    prime_age_e_to_n_rate: float = 0.005
    u_to_nilf_struct_rate_prime: float = 0.0025
    u_to_nilf_struct_rate_older: float = 0.004
    nilf_subtype_enabled: bool = True
    allow_retirement_reentry: bool = True
    school_search_eligible_months: int = 72
    family_search_eligible_months: int = 48
    illness_search_eligible_months: int = 36
    nilf_window_primeage_age_lo: int = 35
    nilf_window_primeage_age_hi: int = 55
    illness_search_eligible_months_primeage: int = 36
    family_search_eligible_months_primeage: int = 48
    nilf_to_u_rate: float = 0.0
    separation_rate_multiplier: float = 1.0
    bargaining_power_skill_bonus: float = 0.05
    bargaining_skill_divisor: float = 1.5
    negotiate_power_clip_min: float = 0.3
    negotiate_power_clip_max: float = 0.7
    wage_premium_max_pct: float = 0.2
    base_worker_power: float = 0.5
    skill_power_bonus: float = 0.05
    duration_power_penalty: float = 0.02
    min_bargaining_power: float = 0.2
    max_bargaining_power: float = 0.8
    duration_power_penalty_cap: float = 0.15
    education_power_bonus: float = 0.03
    tight_market_power_bonus: float = 0.05
    tight_market_threshold: float = 0.05
    reservation_wage_base_ratio: float = 0.85
    reservation_wage_skill_floor_base: float = 12.0
    reservation_wage_skill_floor_mult: float = 3.0
    commute_reference_min: float = 33.0
    max_acceptable_commute_min: float = 90.0
    wage_gap_penalty_beta: float = -1.5
    age_penalty_min: float = 0.0
    age_penalty_max: float = -0.12
    age_penalty_center: float = 55.0
    age_penalty_steepness: float = 4.0
    age_jfr_15_19: float = 1.1
    age_jfr_20_24: float = 1.05
    age_jfr_25_29: float = 1.0
    age_jfr_30_34: float = 1.0
    age_jfr_35_39: float = 0.95
    age_jfr_40_44: float = 0.95
    age_jfr_45_49: float = 0.9
    age_jfr_50_54: float = 0.9
    age_jfr_55_59: float = 0.85
    age_jfr_60_64: float = 0.8
    age_jfr_65_plus: float = 0.7
    beta_job_security: float = 0.18
    acceptance_temperature: float = 1.8
    channel_effectiveness: Dict[int, float] = field(
        default_factory=lambda: {0: 1.4, 1: 1.0, 2: 0.8, 3: 0.5, 4: 0.7}
    )
    channel_context_modifiers: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "ads_online": {"skill_range": 0.2},
            "direct_contact": {"same_ind_cma": 1.3, "same_cma": 1.1, "distant": 0.7},
            "networking": {"same_ind": 1.25, "diff_ind": 0.95},
            "public_agency": {"quality_mult": 0.9},
            "private_agency": {"perfect": 1.4, "partial": 1.1, "no_match": 0.8},
        }
    )
    commute_desperation_bins: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0, 3, 1.0), (3, 6, 0.8), (6, 12, 0.6), (12, 1000000000.0, 0.4)]
    )
    industry_separation_rates: Dict[int, float] = field(
        default_factory=lambda: {
            1: 0.0168,
            2: 0.0231,
            3: 0.0102,
            4: 0.0178,
            5: 0.0149,
            7: 0.0132,
            8: 0.0263,
            9: 0.014,
            10: 0.0119,
            11: 0.0148,
            12: 0.0282,
            13: 0.0125,
            14: 0.0114,
            15: 0.022,
            16: 0.0366,
            17: 0.0184,
            18: 0.0101,
        }
    )


@dataclass
class DataConfig:

    @property
    def project_root(self) -> Path:
        from market_setup import data_root

        return data_root()

    @property
    def data_dir(self) -> Path:
        return self.project_root / "Data" / "Simulation" / "Parameters"

    @property
    def population_dir(self) -> Path:
        return self.project_root / "Data_Agents" / "Microsim_Output"

    @property
    def firms_dir(self) -> Path:
        return self.project_root / "Data_Agents" / "firm_population"

    population_file: str = "population_2006.csv"
    firms_file: str = "firms_2006.csv"
    year: int = 2006

    def get_population_path(self) -> Path:
        return self.population_dir / self.population_file

    def get_firms_path(self) -> Path:
        return self.firms_dir / self.firms_file

    def get_parameter_path(self, filename: str) -> Path:
        return self.data_dir / filename


@dataclass
class SimulationConfig:
    start_year: int = 2006
    end_year: int = 2016
    months_per_year: int = 12
    max_episode_years: int = 11
    n_workers: int = 2000
    n_firms: int = 100


@dataclass
class EnvironmentConfig:
    n_visible_vacancies: int = 15
    history_length: int = 36
    history_dim: int = 5
    compressed_history_tokens: int = 18
    worker_state_dim: int = 24
    firm_state_dim: int = 22
    vacancy_feature_dim: int = 20
    applicant_feature_dim: int = 25
    global_state_dim: int = 26

    @property
    def effective_global_state_dim(self) -> int:
        return self.global_state_dim + 0


@dataclass
class CovidRegimeConfig:
    enabled: bool = False
    driver_file: str = "covid_transition_drivers_monthly.csv"
    standard_driver_mode: str = "as_loaded"
    arm_id: str = ""
    rng_seed: int = 0
    shock_start_year: int = 2020
    shock_end_year: int = 2021
    release_year: int = 2022
    report_start_year: int = 2017
    benchmark_cmas: Tuple[int, ...] = (535, 537)


@dataclass
class MatchingConfig:
    mnl_config: MNLConfig = field(default_factory=MNLConfig)
    n_matching_rounds: int = 2
    relevance_industry_weight: float = 0.3
    relevance_skill_weight: float = 0.18
    relevance_education_weight: float = 0.12
    commute_gate_center_min: float = 45.0
    commute_gate_steepness_min: float = 12.0
    commute_gate_center_km: float = 20.0
    commute_gate_steepness_km: float = 6.0
    commute_gate_floor: float = 0.01
    commute_gate_very_long_min: float = 120.0
    commute_gate_very_long_km: float = 60.0
    commute_gate_very_long_mult: float = 0.3
    commute_gate_meaningful_threshold: float = 0.1
    min_feasible_vacancies: int = 5
    relevance_min: float = 1e-08
    search_intensity_to_apps: Dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 3, 2: 6, 3: 10, 4: 15}
    )
    match_noise_std: float = 0.05
    skill_weight: float = 0.4
    industry_weight: float = 0.25
    education_weight: float = 0.15
    wage_strategy_multipliers: Dict[int, float] = field(
        default_factory=lambda: {0: 0.85, 1: 1.0, 2: 1.35, 3: 1.05, 4: 1.1}
    )
    worker_res_wage_multipliers: Dict[int, float] = field(
        default_factory=lambda: {0: 0.8, 1: 0.9, 2: 1.0, 3: 1.1, 4: 1.25}
    )
    j2j_min_raise_by_intensity: Tuple[float, ...] = (1.12, 1.08, 1.05, 1.03, 1.02)
    j2j_annual_cap: int = 2
    new_firm_wage_range: Tuple[float, float] = (0.9, 1.1)
    hiring_selectivity_thresholds: Dict[int, float] = field(
        default_factory=lambda: {0: 0.6, 1: 0.4, 2: 0.3}
    )
    min_match_quality_floor: float = 0.5
    mq_underqual_params: Tuple[float, float] = (0.2, 0.3)
    mq_overqual_params: Tuple[float, float] = (0.35, 0.1)
    mq_base_cross_sector_credit: float = 0.6
    mq_edu_penalty_per_level: float = 0.2
    mq_exp_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.05, 2: 0.1, 3: 0.2, 4: 0.25}
    )
    mq_exp_normalization: float = 10.0
    mq_age_adjustments: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0, 25, 0.05),
            (25, 35, 0.05),
            (35, 55, -0.05),
            (55, 1000000000.0, -0.05),
        ]
    )
    rel_same_sector_base: float = 0.5
    rel_diff_sector_base: float = 0.1
    rel_duration_flexibility_bins: List[Tuple[float, float, float, float, float, float]] = field(
        default_factory=lambda: [
            (0, 6, 0.5, 0.0, 0.0, 0.1),
            (6, 12, 0.7, 0.1, 0.7, 0.1),
            (12, 1000000000.0, 0.4, 0.4, 0.4, 0.4),
        ]
    )
    rel_diff_sector_desperate_cap: float = 0.5
    rel_underqual_scores: Dict[int, float] = field(
        default_factory=lambda: {-1: 0.7, -2: 0.35, -3: 0.1}
    )
    rel_overqual_scores: Dict[int, float] = field(
        default_factory=lambda: {1: 0.15, 2: 0.03, 3: 0.01}
    )
    rel_overqual_score_max: float = 0.8
    rel_underqual_duration_boosts: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0, 3, 1.0), (3, 6, 1.1), (6, 12, 1.2), (12, 1000000000.0, 1.4)]
    )
    rel_overqual_duration_boosts: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [(0, 3, 1.0), (3, 6, 1.5), (6, 12, 2.5), (12, 1000000000.0, 4.0)]
    )
    rel_age_entry_level_mults: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0, 25, 1.0),
            (25, 35, 1.0),
            (35, 55, 1.0),
            (55, 1000000000.0, 1.0),
        ]
    )
    rel_edu_diff_1_score: float = 0.5
    rel_edu_diff_2plus_score: float = 0.2
    rel_cross_cma_score: float = 0.05


@dataclass
class VacancyConfig:
    separation_rates_by_industry: Dict[int, float] = field(
        default_factory=lambda: {
            1: 0.015,
            2: 0.02,
            3: 0.01,
            4: 0.016,
            5: 0.014,
            7: 0.012,
            8: 0.023,
            9: 0.013,
            10: 0.011,
            11: 0.014,
            12: 0.024,
            13: 0.012,
            14: 0.011,
            15: 0.02,
            16: 0.03,
            17: 0.017,
            18: 0.01,
        }
    )
    vacancy_action_multipliers: Dict[int, float] = field(
        default_factory=lambda: {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0, 4: 2.0}
    )
    vacancy_base_survival_rate: float = 0.55
    vacancy_duration_hazard_rate: float = 0.08
    vacancy_min_duration_factor: float = 0.5
    vacancy_skill_survival_base: float = 0.7
    vacancy_skill_survival_bonus: float = 0.05
    max_vacancy_duration: int = 6
    vacancy_industry_survival: Dict[int, float] = field(
        default_factory=lambda: {
            1: 0.45,
            2: 0.55,
            3: 0.65,
            4: 0.4,
            5: 0.55,
            7: 0.5,
            8: 0.4,
            9: 0.5,
            10: 0.65,
            11: 0.6,
            12: 0.55,
            13: 0.65,
            14: 0.7,
            15: 0.45,
            16: 0.35,
            17: 0.45,
            18: 0.7,
        }
    )


@dataclass
class ActionConfig:
    worker_search_decision_options: int = 2
    worker_search_channel_options: int = 4
    worker_search_intensity_options: int = 5
    worker_res_wage_options: int = 5
    firm_vacancy_management_options: int = 5
    firm_hiring_decision_options: int = 3
    firm_wage_strategy_options: int = 3

    @property
    def worker_action_size(self) -> int:
        base = (
            self.worker_search_decision_options
            * self.worker_search_channel_options
            * self.worker_search_intensity_options
        )
        return base + 1

    @property
    def firm_action_size(self) -> int:
        return (
            self.firm_vacancy_management_options
            * self.firm_hiring_decision_options
            * self.firm_wage_strategy_options
        )


@dataclass
class DiscouragementConfig:
    u_to_nilf_discouragement_age_threshold: int = 65
    u_to_nilf_discouragement_monthly_rate: float = 0.08
    u_to_nilf_discouragement_duration_threshold_months: int = 24


@dataclass
class EIConfig:
    ei_min_tenure_months: int = 5
    ei_replacement_rate: float = 0.55
    ei_min_weeks: int = 14
    ei_max_weeks: int = 45
    ei_weeks_decrement_per_month: int = 4


@dataclass
class RewardConfig:
    vacancy_posting_cost: float = 2.5
    match_quality_floor: float = 0.6
    base_survival_income: float = 3.0
    search_cost_per_level: float = 0.05
    nilf_leisure_value: float = 0.1
    nilf_leisure_value_primeage: float = 0.1
    nilf_leisure_primeage_age_lo: int = 35
    nilf_leisure_primeage_age_hi: int = 55
    cpp65_avg_monthly: float = 640.0
    oas65_monthly: float = 540.0
    retirement_hours_per_month: float = 160.0
    retirement_age_min: int = 60
    retirement_action_age_min: int = 55
    profit_buffer: float = 100.0
    ei_replacement_rate: float = 0.55
    ei_max_weeks: int = 45
    ei_min_weeks: int = 14


@dataclass
class NetworkConfig:
    transformer_d_model: int = 128
    transformer_n_heads: int = 4
    transformer_n_layers: int = 2
    transformer_dropout: float = 0.1
    transformer_max_seq_len: int = 64
    spatial_input_dim: int = 64
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    encoder_hidden_dim: int = 128
    encoder_output_dim: int = 64
    history_dim: int = 5
    history_length: int = 36
    compressed_history_tokens: int = 18
    worker_state_dim: int = 24
    firm_state_dim: int = 22
    vacancy_feature_dim: int = 20
    applicant_feature_dim: int = 25


@dataclass
class TrainingConfig:
    renewal_net_projection: Optional[str] = None
    total_timesteps: int = 1000000
    n_epochs: int = 4
    batch_size: int = 256
    minibatch_size: int = 64
    actor_lr: float = 0.0001
    critic_lr: float = 0.0005
    worker_actor_lr: Optional[float] = 0.0001
    worker_critic_lr: Optional[float] = 0.0005
    firm_actor_lr: Optional[float] = 0.0001
    firm_critic_lr: Optional[float] = 0.0005
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    worker_entropy_coef: float = 0.08
    worker_res_wage_entropy_coef: float = 0.1
    worker_res_wage_frozen_level: int = 2
    worker_value_clip_epsilon: Optional[float] = None
    worker_value_popart_alpha: float = 0.01
    worker_value_popart_min_std: float = 1.0
    attachment_penalty_alpha: float = 0.3
    enable_reservation_wage_bundle: bool = False
    reservation_decay_per_year: float = 0.05
    reservation_decay_compounded: bool = False
    retire_logit_init_bias: float = -2.0
    retire_sentinel_res_wage_idx: int = 2
    worker_unemployed_policy_weight: float = 3.0
    worker_nilf_policy_weight: float = 2.0
    firm_entropy_coef: float = 0.05
    firm_entropy_coef_min: float = 0.0001
    firm_entropy_decay: float = 0.98
    firm_entropy_warmup_updates: int = 0
    worker_kl_threshold: float = 0.015
    worker_kl_rollback_factor: float = 0.5
    worker_target_kl: Optional[float] = None
    firm_kl_threshold: float = 0.05
    firm_kl_rollback_factor: float = 0.5
    max_grad_norm: float = 0.5
    freeze_firm_policy: bool = True
    policy_seed: Optional[int] = None
    rollout_steps: int = 1024
    device: str = "cuda"


@dataclass
class MeanFieldConfig:
    enabled: bool = True
    max_outer_iterations: int = 10
    inner_training_steps: int = 50000
    skill_distributions: Dict[int, Tuple[float, float]] = field(
        default_factory=lambda: {
            0: (0.4, 0.1),
            1: (0.55, 0.1),
            2: (0.7, 0.1),
            3: (0.25, 0.1),
            4: (0.4, 0.12),
            5: (0.4, 0.1),
            6: (0.55, 0.1),
            7: (0.5, 0.1),
            8: (0.65, 0.1),
        }
    )


@dataclass
class SkillDynamicsConfig:
    enabled: bool = True
    accumulation_young: float = 0.1
    accumulation_prime: float = 0.04
    accumulation_mature: float = 0.01
    accumulation_senior: float = 0.004
    age_young_max: int = 24
    age_prime_max: int = 44
    age_mature_max: int = 54
    edu_modifier_0: float = 0.7
    edu_modifier_1: float = 0.85
    edu_modifier_2: float = 1.0
    edu_modifier_3: float = 1.2
    edu_modifier_4: float = 1.3
    new_job_bonus: float = 1.2
    new_job_threshold_months: int = 12
    skill_ceiling_0: float = 2.0
    skill_ceiling_1: float = 3.0
    skill_ceiling_2: float = 3.5
    skill_ceiling_3: float = 4.0
    skill_ceiling_4: float = 4.5
    depreciation_annual: float = 0.06
    grace_period_months: int = 1
    ltu_threshold_months: int = 6
    ltu_multiplier: float = 2.0
    vltu_threshold_months: int = 12
    vltu_multiplier: float = 2.5
    skill_floor: float = 0.5
    layoff_skill_loss: float = 0.06
    reemployment_boost: float = 0.02
    retirement_skill_decay: float = 0.1
    nilf_depreciation_annual: float = 0.02


@dataclass
class TargetAdaptationConfig:
    enabled: bool = False
    frequency_months: int = 3
    fill_rate_window: int = 6


@dataclass
class TradeWarConfig:
    enabled: bool = False
    scenario: str = "control"
    driver_file: str = ""
    exposure_file: str = ""
    min_target_multiplier: float = 0.25
    min_margin_multiplier: float = 0.1
    max_export_demand_loss: float = 0.8


@dataclass
class Config:
    experiment_name: str = "gtha_labor_marl"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    covid: CovidRegimeConfig = field(default_factory=CovidRegimeConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    vacancy: VacancyConfig = field(default_factory=VacancyConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ei: EIConfig = field(default_factory=EIConfig)
    discouragement: DiscouragementConfig = field(default_factory=DiscouragementConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    skill_dynamics: SkillDynamicsConfig = field(default_factory=SkillDynamicsConfig)
    mean_field: MeanFieldConfig = field(default_factory=MeanFieldConfig)
    target_adaptation: TargetAdaptationConfig = field(default_factory=TargetAdaptationConfig)

    @property
    def mnl(self) -> MNLConfig:
        return self.matching.mnl_config

    def __post_init__(self):
        if not self.vacancy.separation_rates_by_industry:
            self.vacancy.separation_rates_by_industry = dict(self.mnl.industry_separation_rates)
        self.network.history_dim = self.environment.history_dim
        self.network.history_length = self.environment.history_length
        self.network.compressed_history_tokens = self.environment.compressed_history_tokens

    def to_dict(self) -> dict:
        import dataclasses

        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj

        return convert(self)

    trade_war: TradeWarConfig = field(default_factory=TradeWarConfig)


NAICS_TO_SECTOR = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    7: 1,
    8: 1,
    10: 2,
    11: 2,
    12: 3,
    13: 3,
    14: 3,
    15: 4,
    16: 4,
    18: 4,
    9: 5,
    17: 5,
}
