import torch
import numpy as np
import logging
import os
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter

try:
    from .gtha_travel_time import GTHATravelTimeCalculator

    HAS_TRAVEL_TIME = True
except ImportError:
    try:
        from gtha_travel_time import GTHATravelTimeCalculator

        HAS_TRAVEL_TIME = True
    except ImportError:
        HAS_TRAVEL_TIME = False
logger = logging.getLogger(__name__)
INDUSTRY_WAGE_CEILING = {
    1: 49.3,
    2: 119.6,
    3: 77.7,
    4: 58.5,
    5: 55.7,
    7: 73.0,
    8: 50.0,
    9: 50.5,
    10: 108.2,
    11: 86.5,
    12: 56.2,
    13: 70.5,
    14: 63.0,
    15: 68.5,
    16: 41.4,
    17: 54.5,
    18: 58.5,
}
_HOURS_PER_MONTH = 173.2
_FIRM_MARGIN = 0.9
try:
    from ..rewards.firm_rewards import INDUSTRY_OUTPUT_PER_WORKER as _INDUSTRY_OUTPUT_PER_WORKER
    from ..rewards.firm_rewards import SKILL_MULTIPLIER as _SKILL_MULTIPLIER
except ImportError:
    from rewards.firm_rewards import INDUSTRY_OUTPUT_PER_WORKER as _INDUSTRY_OUTPUT_PER_WORKER
    from rewards.firm_rewards import SKILL_MULTIPLIER as _SKILL_MULTIPLIER
CHANNEL_NAMES: Dict[int, str] = {
    0: "ads_online",
    1: "direct_contact",
    2: "networking",
    3: "public_agency",
    4: "private_agency",
}
_CHANNEL_STR_TO_NAME: Dict[str, str] = {
    "Ads/Online": "ads_online",
    "Direct Contact": "direct_contact",
    "Networking": "networking",
    "Public Agency": "public_agency",
    "Private Agency": "private_agency",
}


def _compile_bins_1d(
    bins: List[Tuple[float, float, float]], device
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert bins[0][0] == 0, f"First bin must start at 0, got {bins[0][0]}"
    assert all(
        (bins[i][0] >= bins[i - 1][0] for i in range(1, len(bins)))
    ), "Bins must be sorted by lo"
    for i, (lo, hi, _) in enumerate(bins):
        assert lo < hi, f"Bin {i}: min ({lo}) must be < max ({hi})"
        if i > 0:
            prev_hi = bins[i - 1][1]
            assert lo >= prev_hi, f"Bin {i}: lo ({lo}) < prev max ({prev_hi}), bins overlap"
    assert bins[-1][1] >= 1000000000.0, f"Last bin max ({bins[-1][1]}) should be >= 1e9"
    upper_bounds = torch.tensor([b[1] for b in bins], device=device, dtype=torch.float32)
    values = torch.tensor([b[2] for b in bins], device=device, dtype=torch.float32)
    return (upper_bounds, values)


def _compile_bins_kd(bins: List[Tuple], k: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(bins) > 0, "Bins list must be non-empty"
    assert k > 0, "k (number of value columns) must be > 0"
    assert bins[0][0] == 0, f"First bin must start at 0, got {bins[0][0]}"
    assert all(
        (bins[i][0] >= bins[i - 1][0] for i in range(1, len(bins)))
    ), "Bins must be sorted by lo"
    for i in range(len(bins)):
        lo, hi = (bins[i][0], bins[i][1])
        assert lo < hi, f"Bin {i}: min ({lo}) must be < max ({hi})"
        if i > 0:
            prev_hi = bins[i - 1][1]
            assert lo >= prev_hi, f"Bin {i}: lo ({lo}) < prev max ({prev_hi}), bins overlap"
        assert len(bins[i]) == k + 2, f"Bin {i}: expected {k + 2} elements, got {len(bins[i])}"
    assert bins[-1][1] >= 1000000000.0, f"Last bin max ({bins[-1][1]}) should be >= 1e9"
    upper_bounds = torch.tensor([b[1] for b in bins], device=device, dtype=torch.float32)
    values_matrix = torch.tensor([list(b[2:]) for b in bins], device=device, dtype=torch.float32)
    return (upper_bounds, values_matrix)


def _lookup_bins(x: torch.Tensor, upper_bounds: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(x, upper_bounds[:-1], right=False)
    return values[idx.clamp(max=len(values) - 1)]


def _lookup_bins_kd(
    x: torch.Tensor, upper_bounds: torch.Tensor, values_matrix: torch.Tensor
) -> torch.Tensor:
    idx = torch.bucketize(x, upper_bounds[:-1], right=False)
    idx = idx.clamp(max=values_matrix.shape[0] - 1)
    return values_matrix[idx]


@dataclass
class Worker:
    id: str
    age: int
    skill_level: int
    education: int
    industry: int
    sector: int
    experience_years: float
    cma: int
    employment_status: str
    current_wage: float
    is_searching: bool
    unemployment_months: int = 0
    job_application_attempts: int = 0
    current_employer_id: Optional[str] = None
    hours_per_week: float = 40.0
    da_id: Optional[int] = None
    search_channel: int = 0
    search_intensity: int = 2
    commute_time_min: Optional[float] = None
    archetype_id: int = 0
    archetype_name: str = "goods_core_lowskill"
    sector_id: int = 0
    sector_name: str = "services"
    sex: int = 0
    attachment: float = 1.0


@dataclass
class Vacancy:
    id: str
    firm_id: str
    industry: int
    sector: int
    skill_required: int
    education_required: int
    offered_wage: float
    cma: int
    duration_months: int = 0
    months_posted: int = 0
    job_security: float = 0.5
    da_id: Optional[int] = None
    posting_reason: str = "unknown"


@dataclass
class Firm:
    id: str
    industry: int
    sector: int
    n_employees: int
    cma: int
    target_size: int = None
    da_id: Optional[int] = None
    naics_6digit: Optional[str] = None

    def __post_init__(self):
        if self.target_size is None:
            self.target_size = max(int(self.n_employees * 1.1), self.n_employees)


@dataclass
class MatchResult:
    worker_id: str
    vacancy_id: str
    firm_id: str
    final_wage: float
    match_quality: float = 0.0
    commute_distance_km: Optional[float] = None


try:
    from ..config.default_config import MatchingConfig
except ImportError:
    from config.default_config import MatchingConfig
_ENC_SEP_REASON = {"unknown": 0, "job_loser": 1, "job_leaver": 2, "re_entrant": 3}
_DEC_SEP_REASON = {0: "unknown", 1: "job_loser", 2: "job_leaver", 3: "re_entrant"}
from evaluation import offers as _acceptance_extension


class MatchingMechanism:

    def __init__(self, config: Optional[MatchingConfig] = None, device: str = "cuda"):
        self.config = config or MatchingConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.attachment_enabled = False
        self.attachment_alpha = 0.3
        self.reservation_bundle_enabled = False
        self.reservation_decay_per_year = 0.0
        self.reservation_decay_compounded = False
        self.da_centroids = {}
        self.da_distance_matrix = None
        self.da_travel_time_matrix = None
        self.da_id_to_idx = {}
        self.idx_to_da_id = {}
        self.worker_commute_factors = {}
        self._use_travel_time = False
        self.stats = {
            "applications": 0,
            "offers": 0,
            "matches": 0,
            "rejections": 0,
            "avg_accept_prob": 0.0,
        }
        for _stage in ("app", "screen_pass", "screen_fail", "offer", "match"):
            for _tier in ("same_industry", "same_sector_diff_industry", "diff_sector"):
                self.stats[f"{_stage}_{_tier}"] = 0
        for _stage in ("app", "screen_pass", "screen_fail"):
            for _tier in ("same_industry", "same_sector_diff_industry", "diff_sector"):
                for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                    self.stats[f"{_stage}_{_tier}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"offer_{_bin}"] = 0
            self.stats[f"match_{_bin}"] = 0
            for _reason in (
                "wage_gap",
                "long_commute",
                "low_wage_utility",
                "poor_match_quality",
                "high_reservation",
                "random",
            ):
                self.stats[f"reject_dominant_{_reason}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"n_unemployed_seekers_{_bin}"] = 0
            self.stats[f"n_workers_offered_{_bin}"] = 0
            self.stats[f"n_workers_matched_{_bin}"] = 0
            self.stats[f"n_workers_no_offer_{_bin}"] = 0
            self.stats[f"n_workers_rejected_all_{_bin}"] = 0
        for _fail in ("vacancy_filled", "firm_cap", "bargaining"):
            self.stats[f"match_fail_{_fail}"] = 0
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"match_fail_{_fail}_{_bin}"] = 0
        for _band in ("close_easy", "close_mid", "close_hard", "close_no"):
            self.stats[f"bf_ratio_band_{_band}"] = 0
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"bf_ratio_band_{_band}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"bf_ratio_sum_{_bin}"] = 0.0
            self.stats[f"bf_ratio_n_{_bin}"] = 0
        for _cap in ("posted", "profit", "industry"):
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"bf_cap_{_cap}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"bf_res_sum_{_bin}"] = 0.0
            self.stats[f"bf_fem_sum_{_bin}"] = 0.0
            self.stats[f"bf_posted_sum_{_bin}"] = 0.0
            self.stats[f"bf_prevwage_sum_{_bin}"] = 0.0
        for _tier in ("high_M", "high_F", "low_M", "low_F"):
            self.stats[f"bf_tier_{_tier}"] = 0
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"bf_tier_{_tier}_{_bin}"] = 0
        for _tier in ("high_M", "high_F", "low_M", "low_F"):
            for _band in ("close_easy", "close_mid", "close_hard", "close_no"):
                self.stats[f"bf_tier_{_tier}_{_band}"] = 0
                for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                    self.stats[f"bf_tier_{_tier}_{_band}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"bf_commute_sum_{_bin}"] = 0.0
            self.stats[f"bf_commute_n_{_bin}"] = 0
        self._episode_ever_offered: set = set()
        self._episode_ever_matched: set = set()
        self.rejection_reasons = {
            "low_wage_utility": 0,
            "poor_match_quality": 0,
            "long_commute": 0,
            "high_reservation": 0,
            "age_friction": 0,
            "random_rejection": 0,
        }
        self._matching_step = 0
        self._mq_audit_call_count = 0
        self._realized_premium_buffer = []
        self._wage_premium_audit_call_count = 0
        self._setup_mnl_tensors()
        self._compile_schedules()

    def _compile_schedules(self):
        mnl = self.config.mnl_config
        self._commute_desp_ub, self._commute_desp_vals = _compile_bins_1d(
            mnl.commute_desperation_bins, self.device
        )
        active_mq_age = self.config.mq_age_adjustments
        self._mq_age_ub, self._mq_age_vals = _compile_bins_1d(
            [(float(lo), float(hi), v) for lo, hi, v in active_mq_age], self.device
        )
        exp_w = self.config.mq_exp_weights
        max_skill = max(exp_w.keys()) if exp_w else 4
        self._mq_exp_weight_tensor = torch.zeros(max_skill + 1, device=self.device)
        for skill_lvl, weight in exp_w.items():
            if 0 <= skill_lvl <= max_skill:
                self._mq_exp_weight_tensor[skill_lvl] = weight
        self._rel_dur_flex_ub, self._rel_dur_flex_vals = _compile_bins_kd(
            self.config.rel_duration_flexibility_bins, k=4, device=self.device
        )
        self._rel_uq_boost_ub, self._rel_uq_boost_vals = _compile_bins_1d(
            self.config.rel_underqual_duration_boosts, self.device
        )
        self._rel_oq_boost_ub, self._rel_oq_boost_vals = _compile_bins_1d(
            self.config.rel_overqual_duration_boosts, self.device
        )
        active_rel_age = self.config.rel_age_entry_level_mults
        self._rel_age_entry_ub, self._rel_age_entry_vals = _compile_bins_1d(
            active_rel_age, self.device
        )
        uq_scores = self.config.rel_underqual_scores
        max_uq = max((abs(k) for k in uq_scores.keys())) if uq_scores else 3
        self._rel_uq_scores = torch.ones(max_uq + 1, device=self.device)
        for diff, score in uq_scores.items():
            idx = abs(diff)
            if 0 <= idx <= max_uq:
                self._rel_uq_scores[idx] = score
        oq_scores = self.config.rel_overqual_scores
        max_oq = max((abs(k) for k in oq_scores.keys())) if oq_scores else 3
        self._rel_oq_scores = torch.ones(max_oq + 1, device=self.device)
        for diff, score in oq_scores.items():
            idx = abs(diff)
            if 0 <= idx <= max_oq:
                self._rel_oq_scores[idx] = score
        mnl = self.config.mnl_config
        jfr_bins = [
            (0, 20, getattr(mnl, "age_jfr_15_19", 1.25)),
            (20, 25, getattr(mnl, "age_jfr_20_24", 1.15)),
            (25, 30, getattr(mnl, "age_jfr_25_29", 1.05)),
            (30, 35, getattr(mnl, "age_jfr_30_34", 1.0)),
            (35, 40, getattr(mnl, "age_jfr_35_39", 0.95)),
            (40, 45, getattr(mnl, "age_jfr_40_44", 0.9)),
            (45, 50, getattr(mnl, "age_jfr_45_49", 0.85)),
            (50, 55, getattr(mnl, "age_jfr_50_54", 0.78)),
            (55, 60, getattr(mnl, "age_jfr_55_59", 0.7)),
            (60, 65, getattr(mnl, "age_jfr_60_64", 0.6)),
            (65, 1000000000.0, getattr(mnl, "age_jfr_65_plus", 0.5)),
        ]
        self._jfr_age_ub, self._jfr_age_vals = _compile_bins_1d(jfr_bins, self.device)

    def _setup_mnl_tensors(self):
        _acceptance_extension.configure_matcher(self)
        mnl = self.config.mnl_config
        self._mnl_beta_wage = torch.tensor(mnl.beta_wage, device=self.device)
        self._mnl_beta_commute = torch.tensor(mnl.beta_commute, device=self.device)
        self._mnl_beta_match = torch.tensor(mnl.beta_skill_match, device=self.device)
        self._mnl_beta_security = torch.tensor(mnl.beta_job_security, device=self.device)
        self._mnl_scale = torch.tensor(
            getattr(mnl, "acceptance_temperature", 1.8), device=self.device
        )
        _mult_tuple = (1.0,) * 9
        self._archetype_commute_multipliers = torch.tensor(
            list(_mult_tuple), dtype=self._mnl_beta_commute.dtype, device=self.device
        )

    def reset_episode_acceptance_tracking(self):
        self._episode_ever_offered.clear()
        self._episode_ever_matched.clear()
        if hasattr(self, "_episode_acceptance_invalid_reason"):
            delattr(self, "_episode_acceptance_invalid_reason")
        self._structural_worker_audit = {}

    def _audit_worker_event(self, worker_id: str, event: str, count: int = 1):
        self._person_trace_event(worker_id, event, count)

    _PERSON_TRACE_COUNT_FIELDS = (
        "seeker_months",
        "applications",
        "no_suitable_vacancy_months",
        "screen_pass",
        "screen_fail",
        "screen_fail_firm_threshold",
        "screen_fail_structural_floor",
        "screen_fail_other",
        "offers",
        "preference_accepts",
        "candidate_accepts",
        "offers_rejected",
        "attachment_conversion_failures",
        "bargaining_attempts",
        "bargaining_successes",
        "bargaining_failures",
        "match_fail_vacancy_filled",
        "match_fail_firm_cap",
        "matches",
        "same_employer_nontransition",
        "incoming_staff_selection",
        "outside_market_selection",
    )

    def begin_person_month_trace(self):
        self._person_month_trace = {
            "schema": "complete_matching_person_month_trace_v1",
            "enabled": os.environ.get("WORKER_LEVEL_TRACE", "0") == "1"
            and getattr(self, "_person_trace_active", False),
            "matcher_called": False,
            "no_vacancies": False,
            "no_job_seekers": False,
            "pool_evaluated": False,
            "known_count_fields": list(self._PERSON_TRACE_COUNT_FIELDS),
            "event_coverage": "Observed matching input and actual jobseeker pool; event counts may overlap across rounds. matches is pre-commit.",
            "separation_reason_encoding": "String labels; encoded input values decoded with the existing matching separation-reason map.",
            "workers": {},
        }

    def _person_trace_row(self, worker_id):
        if os.environ.get("WORKER_LEVEL_TRACE", "0") != "1" or not getattr(
            self, "_person_trace_active", False
        ):
            return None
        if not hasattr(self, "_person_month_trace"):
            self.begin_person_month_trace()
        return self._person_month_trace["workers"].setdefault(
            str(worker_id),
            {
                "pre_match": None,
                "events": {key: 0 for key in self._PERSON_TRACE_COUNT_FIELDS},
                "screening_events": [],
                "offer_events": [],
                "bargaining_attempt_details": [],
                "bargaining_failures": [],
            },
        )

    def _person_trace_event(self, worker_id, event, count=1):
        row = self._person_trace_row(worker_id)
        if row is not None:
            row["events"][str(event)] = int(row["events"].get(str(event), 0)) + int(count)

    def _person_trace_register(self, worker_id, worker_state, in_pool=False):
        row = self._person_trace_row(worker_id)
        if row is None:
            return
        if isinstance(worker_state, dict):
            source = worker_state
            get = source.get
            reason = get("separation_reason")
            if isinstance(reason, int):
                reason = _DEC_SEP_REASON.get(reason, "unknown")
            pre = {
                "age": get("age"),
                "status": get("employment_status"),
                "is_searching": get("is_searching"),
                "unemployment_months": get("unemployment_months"),
                "separation_reason": reason,
                "industry": get("naics_18", get("industry")),
                "skill_level": get("skill_level"),
                "previous_wage": get("previous_wage"),
                "current_wage": get("current_wage"),
                "res_wage_action": get("res_wage_action"),
            }
        else:
            worker = worker_state.worker if hasattr(worker_state, "worker") else worker_state
            pre = {
                "age": getattr(worker, "age", None),
                "status": getattr(worker, "employment_status", None),
                "is_searching": getattr(worker, "is_searching", None),
                "unemployment_months": getattr(worker, "unemployment_months", None),
                "separation_reason": getattr(worker_state, "separation_reason", None),
                "industry": getattr(worker, "industry", None),
                "skill_level": getattr(worker, "skill_level", None),
                "previous_wage": getattr(worker_state, "previous_wage", None),
                "current_wage": getattr(worker, "current_wage", None),
                "res_wage_action": getattr(worker_state, "last_res_wage_action", None),
            }
        pre["in_matching_pool"] = bool(in_pool)
        pre["is_matching_seeker"] = bool(in_pool)
        pre = {key: value.item() if hasattr(value, "item") else value for key, value in pre.items()}
        row["pre_match"] = copy.deepcopy(pre)

    def _person_trace_detail(self, worker_id, stage, detail):
        row = self._person_trace_row(worker_id)
        if row is not None:
            row[stage].append(copy.deepcopy(detail))

    def get_person_month_trace(self):
        if not hasattr(self, "_person_month_trace"):
            self.begin_person_month_trace()
        return copy.deepcopy(self._person_month_trace)

    def get_episode_acceptance_stats(self) -> dict:
        invalid_reason = getattr(self, "_episode_acceptance_invalid_reason", None)
        if invalid_reason is not None:
            return {
                "invalid": True,
                "reason": invalid_reason,
                "n_workers_ever_offered": len(self._episode_ever_offered),
                "n_workers_ever_matched": len(self._episode_ever_matched),
            }
        n_off = len(self._episode_ever_offered)
        n_m = len(self._episode_ever_matched)
        return {
            "invalid": False,
            "n_workers_ever_offered": n_off,
            "n_workers_ever_matched": n_m,
            "n_workers_ever_rejected_all": len(
                self._episode_ever_offered - self._episode_ever_matched
            ),
            "pct_ever_matched_given_offered": n_m / n_off if n_off > 0 else 0.0,
        }

    def reset_stats(self):
        self.stats = {
            "applications": 0,
            "offers": 0,
            "matches": 0,
            "rejections": 0,
            "avg_accept_prob": 0.0,
            "applicants_screened_out": 0,
            "screening_rate": 0.0,
            "zero_offers_diagnostic": [],
            "n_applicants_total": 0,
            "n_applicants_after_filters": 0,
            "n_offers_created": 0,
            "n_offers_sent": 0,
            "screen_reject_commute_gate": 0,
            "screen_reject_quality_with_long_commute": 0,
            "screen_reject_industry_mismatch": 0,
            "screen_reject_selectivity": 0,
            "screen_reject_match_quality_floor": 0,
            "mobility_floor_eligible": 0,
            "mobility_floor_additional_passes": 0,
            "screen_quality_below_firm_threshold": 0,
            "screen_quality_floor_binding_band": 0,
            "screen_quality_below_structural_floor": 0,
            "n_job_seekers": 0,
            "n_workers_offered": 0,
            "n_workers_no_offer": 0,
            "n_workers_rejected_all": 0,
            "offers_per_job_seeker": 0.0,
            "share_offered": 0.0,
            "accept_rate_given_offer": 0.0,
            "bf_cap_posted": 0,
            "bf_cap_profit": 0,
            "bf_cap_industry": 0,
        }
        for _stage in ("app", "screen_pass", "screen_fail", "offer", "match"):
            for _tier in ("same_industry", "same_sector_diff_industry", "diff_sector"):
                self.stats[f"{_stage}_{_tier}"] = 0
        for _stage in ("app", "screen_pass", "screen_fail"):
            for _tier in ("same_industry", "same_sector_diff_industry", "diff_sector"):
                for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                    self.stats[f"{_stage}_{_tier}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"offer_{_bin}"] = 0
            self.stats[f"match_{_bin}"] = 0
            for _reason in (
                "wage_gap",
                "long_commute",
                "low_wage_utility",
                "poor_match_quality",
                "high_reservation",
                "random",
            ):
                self.stats[f"reject_dominant_{_reason}_{_bin}"] = 0
            self.stats[f"n_unemployed_seekers_{_bin}"] = 0
            self.stats[f"n_workers_offered_{_bin}"] = 0
            self.stats[f"n_workers_matched_{_bin}"] = 0
            self.stats[f"n_workers_no_offer_{_bin}"] = 0
            self.stats[f"n_workers_rejected_all_{_bin}"] = 0
        for _fail in ("vacancy_filled", "firm_cap", "bargaining"):
            self.stats[f"match_fail_{_fail}"] = 0
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"match_fail_{_fail}_{_bin}"] = 0
        for _band in ("close_easy", "close_mid", "close_hard", "close_no"):
            self.stats[f"bf_ratio_band_{_band}"] = 0
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"bf_ratio_band_{_band}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"bf_ratio_sum_{_bin}"] = 0.0
            self.stats[f"bf_ratio_n_{_bin}"] = 0
        for _cap in ("posted", "profit", "industry"):
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"bf_cap_{_cap}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"bf_res_sum_{_bin}"] = 0.0
            self.stats[f"bf_fem_sum_{_bin}"] = 0.0
            self.stats[f"bf_posted_sum_{_bin}"] = 0.0
            self.stats[f"bf_prevwage_sum_{_bin}"] = 0.0
        for _tier in ("high_M", "high_F", "low_M", "low_F"):
            self.stats[f"bf_tier_{_tier}"] = 0
            for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                self.stats[f"bf_tier_{_tier}_{_bin}"] = 0
        for _tier in ("high_M", "high_F", "low_M", "low_F"):
            for _band in ("close_easy", "close_mid", "close_hard", "close_no"):
                self.stats[f"bf_tier_{_tier}_{_band}"] = 0
                for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
                    self.stats[f"bf_tier_{_tier}_{_band}_{_bin}"] = 0
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            self.stats[f"bf_commute_sum_{_bin}"] = 0.0
            self.stats[f"bf_commute_n_{_bin}"] = 0
        self.rejection_reasons = {
            "wage_gap": 0,
            "low_wage_utility": 0,
            "poor_match_quality": 0,
            "long_commute": 0,
            "high_reservation": 0,
            "age_friction": 0,
            "random_rejection": 0,
        }
        self.rejection_reasons_by_da: Dict[int, Dict[str, int]] = {}
        self._da_by_widx: Dict[int, Optional[int]] = {}
        for k in (
            "reject_component_wage_gap",
            "reject_component_long_commute",
            "reject_component_low_wage_utility",
            "reject_component_poor_match_quality",
            "reject_component_high_reservation",
        ):
            self.stats[k] = 0
        for k in (
            "reject_dominant_wage_gap",
            "reject_dominant_long_commute",
            "reject_dominant_low_wage_utility",
            "reject_dominant_poor_match_quality",
            "reject_dominant_high_reservation",
            "reject_dominant_random",
        ):
            self.stats[k] = 0
        for k in (
            "utility_diff_random_band_0",
            "utility_diff_random_band_1",
            "utility_diff_random_band_2",
            "utility_diff_random_band_3",
            "utility_diff_random_band_4",
        ):
            self.stats[k] = 0
        self.stats["utility_diff_random_sum"] = 0.0
        self.stats["utility_diff_random_n"] = 0
        self.stats["utility_diff_deterministic_sum"] = 0.0
        self.stats["utility_diff_deterministic_n"] = 0
        if hasattr(self, "_screen_diag_all"):
            self._screen_diag_all.clear()
        if hasattr(self, "_screen_diag_reasons"):
            self._screen_diag_reasons.clear()
        self._screen_diag_step = 0

    def compute_bargaining_power(
        self,
        skill_level: int,
        education: int,
        unemployment_months: int,
        unemployment_rate: float = 0.06,
    ) -> float:
        mnl = self.config.mnl_config
        base_power = mnl.base_worker_power
        skill_bonus = (skill_level - 2) * mnl.skill_power_bonus
        duration_penalty = min(
            unemployment_months * mnl.duration_power_penalty, mnl.duration_power_penalty_cap
        )
        edu_bonus = (education - 3) * mnl.education_power_bonus
        market_bonus = (
            mnl.tight_market_power_bonus if unemployment_rate < mnl.tight_market_threshold else 0.0
        )
        power = base_power + skill_bonus - duration_penalty + edu_bonus + market_bonus
        return float(np.clip(power, mnl.min_bargaining_power, mnl.max_bargaining_power))

    def compute_reservation_wage(
        self,
        current_wage: float,
        skill_level: int,
        unemployment_months: int,
        res_wage_action: int = 2,
        previous_wage: Optional[float] = None,
        separation_reason: Optional[str] = None,
        lwf_retention_draw: Optional[float] = None,
    ) -> float:
        mnl = self.config.mnl_config
        cw = float(current_wage or 0.0)
        pw = float(previous_wage) if previous_wage is not None and previous_wage > 0 else cw
        anchor_wage = max(pw, cw)
        if self.reservation_bundle_enabled:
            if (
                separation_reason == "job_loser"
                and lwf_retention_draw is not None
                and (lwf_retention_draw > 0)
            ):
                base_ratio = min(float(mnl.reservation_wage_base_ratio), float(lwf_retention_draw))
            else:
                base_ratio = mnl.reservation_wage_base_ratio
            _yr = unemployment_months / 12.0
            if self.reservation_decay_compounded:
                _km = max(0.0, (1.0 - self.reservation_decay_per_year) ** _yr)
            else:
                _km = max(0.0, 1.0 - self.reservation_decay_per_year * _yr)
            anchor = anchor_wage * base_ratio * _km
        else:
            anchor = anchor_wage * mnl.reservation_wage_base_ratio
        multipliers = self.config.worker_res_wage_multipliers
        mult = float(multipliers.get(int(res_wage_action), 1.0))
        skill_floor = (
            mnl.reservation_wage_skill_floor_base
            + skill_level * mnl.reservation_wage_skill_floor_mult
        )
        return max(anchor * mult, skill_floor)

    def negotiate_wage_nash(
        self,
        posted_wage: float,
        worker_power: float,
        reservation_wage: float,
        worker_skill_level: int = 2,
        v_industry: int = 0,
        v_firm_id: Optional[str] = None,
    ) -> Optional[float]:
        mnl_config = self.config.mnl_config
        skill_bonus = mnl_config.bargaining_power_skill_bonus
        adjusted_power = (
            worker_power
            + skill_bonus * (worker_skill_level - 2.5) / mnl_config.bargaining_skill_divisor
        )
        adjusted_power = float(
            np.clip(
                adjusted_power,
                mnl_config.negotiate_power_clip_min,
                mnl_config.negotiate_power_clip_max,
            )
        )
        posted_wage_cap = posted_wage * (1.0 + mnl_config.wage_premium_max_pct)
        trade_margin_multiplier = float(
            getattr(self, "_trade_margin_multiplier_by_firm", {}).get(str(v_firm_id), 1.0)
        )
        profitability_cap = (
            _INDUSTRY_OUTPUT_PER_WORKER.get(v_industry, 4000.0)
            * _SKILL_MULTIPLIER.get(worker_skill_level, 0.9)
            / _HOURS_PER_MONTH
            * _FIRM_MARGIN
            * trade_margin_multiplier
        )
        industry_ceiling = INDUSTRY_WAGE_CEILING.get(v_industry, 80.0)
        firm_effective_max = min(posted_wage_cap, profitability_cap, industry_ceiling)
        if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
            self, "_person_trace_active", False
        ):
            self._person_trace_last_bargain_caps = {
                "posted_wage_cap": float(posted_wage_cap),
                "profitability_cap": float(profitability_cap),
                "industry_ceiling": float(industry_ceiling),
                "firm_effective_max": float(firm_effective_max),
                "posted_cap": float(posted_wage_cap),
                "profit_cap": float(profitability_cap),
                "industry_cap": float(industry_ceiling),
                "binding_cap": (
                    "posted"
                    if posted_wage_cap <= profitability_cap and posted_wage_cap <= industry_ceiling
                    else "profit" if profitability_cap <= industry_ceiling else "industry"
                ),
            }
        if reservation_wage > firm_effective_max:
            if not (not hasattr(self, "stats") or self.stats is None):
                self.stats["bargaining_failures"] = self.stats.get("bargaining_failures", 0) + 1
                if "bargaining_failure_by_industry" not in self.stats:
                    self.stats["bargaining_failure_by_industry"] = {}
                if "bargaining_failure_by_skill" not in self.stats:
                    self.stats["bargaining_failure_by_skill"] = {}
                _ind_d = self.stats["bargaining_failure_by_industry"]
                _ind_d[int(v_industry)] = _ind_d.get(int(v_industry), 0) + 1
                _sk_d = self.stats["bargaining_failure_by_skill"]
                _sk_d[int(worker_skill_level)] = _sk_d.get(int(worker_skill_level), 0) + 1
                if posted_wage_cap <= profitability_cap and posted_wage_cap <= industry_ceiling:
                    self.stats["bf_cap_posted"] = self.stats.get("bf_cap_posted", 0) + 1
                    self._last_bf_binding_cap = "posted"
                elif profitability_cap <= industry_ceiling:
                    self.stats["bf_cap_profit"] = self.stats.get("bf_cap_profit", 0) + 1
                    self._last_bf_binding_cap = "profit"
                else:
                    self.stats["bf_cap_industry"] = self.stats.get("bf_cap_industry", 0) + 1
                    self._last_bf_binding_cap = "industry"
                self._last_bf_firm_effective_max = float(firm_effective_max)
            return None
        negotiated = reservation_wage + adjusted_power * (firm_effective_max - reservation_wage)
        return float(negotiated)

    def load_da_centroids(self, centroids_df):
        for _, row in centroids_df.iterrows():
            da_id = int(row["da_id"])
            self.da_centroids[da_id] = (row["lat"], row["lon"])
        self.precompute_da_distance_matrix()

    def precompute_da_distance_matrix(self):
        if not self.da_centroids:
            logger.warning(
                "No dissemination-area centroids are loaded, so the distance matrix cannot be precomputed"
            )
            return
        da_ids = sorted(self.da_centroids.keys())
        n_das = len(da_ids)
        self.da_id_to_idx = {da_id: idx for idx, da_id in enumerate(da_ids)}
        self.idx_to_da_id = {idx: da_id for idx, da_id in enumerate(da_ids)}
        if HAS_TRAVEL_TIME:
            calculator = GTHATravelTimeCalculator(
                da_centroids=self.da_centroids, device=str(self.device)
            )
            self.da_travel_time_matrix = calculator.compute_travel_time_matrix()
            self._use_travel_time = True
            self._compute_haversine_da_matrix(da_ids, n_das)
        else:
            if not HAS_TRAVEL_TIME:
                logger.warning(
                    "The travel-time module could not be imported. Great-circle distances are used."
                )
            self._compute_haversine_da_matrix(da_ids, n_das)
            self._use_travel_time = False

    def _compute_haversine_da_matrix(self, da_ids, n_das):
        coords = []
        for da_id in da_ids:
            lat, lon = self.da_centroids[da_id]
            coords.append([lat, lon])
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
        lat = coords_tensor[:, 0]
        lon = coords_tensor[:, 1]
        lat1 = torch.deg2rad(lat.unsqueeze(1))
        lat2 = torch.deg2rad(lat.unsqueeze(0))
        lon1 = torch.deg2rad(lon.unsqueeze(1))
        lon2 = torch.deg2rad(lon.unsqueeze(0))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        R = 6371.0
        self.da_distance_matrix = R * c

    def compute_distance_matrix_gpu(
        self,
        worker_da_ids: List[int],
        vacancy_da_ids: List[int],
        worker_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        n_workers = len(worker_da_ids)
        n_vacancies = len(vacancy_da_ids)
        if n_workers == 0 or n_vacancies == 0:
            return torch.empty(n_workers, n_vacancies, device=self.device)
        matrix = None
        if self._use_travel_time and self.da_travel_time_matrix is not None:
            matrix = self.da_travel_time_matrix
        elif self.da_distance_matrix is not None:
            matrix = self.da_distance_matrix
        if matrix is not None:
            w_idx_list = [self.da_id_to_idx.get(da, -1) for da in worker_da_ids]
            v_idx_list = [self.da_id_to_idx.get(da, -1) for da in vacancy_da_ids]
            _missing_w = [(i, worker_da_ids[i]) for i, idx in enumerate(w_idx_list) if idx < 0]
            _missing_v = [(i, vacancy_da_ids[i]) for i, idx in enumerate(v_idx_list) if idx < 0]
            if _missing_w or _missing_v:
                raise AssertionError(
                    f"{len(_missing_w)} worker areas and {len(_missing_v)} vacancy areas are absent from the distance matrix (size {len(self.da_id_to_idx)}). Examples for workers (index, area): {_missing_w[:5]}. Examples for vacancies: {_missing_v[:5]}."
                )
            w_indices = torch.tensor(w_idx_list, dtype=torch.long, device=self.device)
            v_indices = torch.tensor(v_idx_list, dtype=torch.long, device=self.device)
            result = matrix[w_indices][:, v_indices]
            if worker_ids and self.worker_commute_factors:
                factors = torch.tensor(
                    [self.worker_commute_factors.get(wid, 1.0) for wid in worker_ids],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(1)
                result = result * factors
            return result
        logger.warning(
            "No distance or travel-time matrix is available. Great-circle distances are computed pair by pair."
        )
        return self._compute_haversine_matrix(worker_da_ids, vacancy_da_ids)

    def _compute_haversine_matrix(
        self, worker_da_ids: List[int], vacancy_da_ids: List[int]
    ) -> torch.Tensor:
        if not self.da_centroids:
            return None
        worker_coords = []
        vacancy_coords = []
        for da_id in worker_da_ids:
            if da_id in self.da_centroids:
                lat, lon = self.da_centroids[da_id]
                worker_coords.append([lat, lon])
            else:
                worker_coords.append([0.0, 0.0])
        for da_id in vacancy_da_ids:
            if da_id in self.da_centroids:
                lat, lon = self.da_centroids[da_id]
                vacancy_coords.append([lat, lon])
            else:
                vacancy_coords.append([0.0, 0.0])
        w_coords = torch.tensor(worker_coords, dtype=torch.float32, device=self.device)
        v_coords = torch.tensor(vacancy_coords, dtype=torch.float32, device=self.device)
        w_lat_rad = torch.deg2rad(w_coords[:, 0].unsqueeze(1))
        w_lon_rad = torch.deg2rad(w_coords[:, 1].unsqueeze(1))
        v_lat_rad = torch.deg2rad(v_coords[:, 0].unsqueeze(0))
        v_lon_rad = torch.deg2rad(v_coords[:, 1].unsqueeze(0))
        dlat = v_lat_rad - w_lat_rad
        dlon = v_lon_rad - w_lon_rad
        a = (
            torch.sin(dlat / 2) ** 2
            + torch.cos(w_lat_rad) * torch.cos(v_lat_rad) * torch.sin(dlon / 2) ** 2
        )
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        return 6371.0 * c

    def _build_worker_tensors(self, workers: List[Dict]) -> Dict[str, torch.Tensor]:
        n = len(workers)
        if n == 0:
            empty = lambda: torch.empty(0, device=self.device)
            return {
                k: empty()
                for k in [
                    "industry",
                    "sector",
                    "skill",
                    "education",
                    "cma",
                    "experience",
                    "wage",
                    "previous_wage",
                    "unemployment_months",
                    "learned_v0",
                    "res_wage_action",
                ]
            }
        return {
            "industry": torch.tensor(
                [w.get("naics_18", 0) for w in workers], dtype=torch.int32, device=self.device
            ),
            "sector": torch.tensor(
                [w.get("sector", 0) for w in workers], dtype=torch.int32, device=self.device
            ),
            "skill": torch.tensor(
                [w.get("skill_level", 2) for w in workers], dtype=torch.int32, device=self.device
            ),
            "education": torch.tensor(
                [w.get("education", 3) for w in workers], dtype=torch.int32, device=self.device
            ),
            "cma": torch.tensor(
                [w.get("cma", 535) for w in workers], dtype=torch.int32, device=self.device
            ),
            "experience": torch.tensor(
                [w.get("experience_total", 0) for w in workers],
                dtype=torch.float32,
                device=self.device,
            ),
            "wage": torch.tensor(
                [w.get("current_wage", 15.0) for w in workers],
                dtype=torch.float32,
                device=self.device,
            ),
            "previous_wage": torch.tensor(
                [w.get("previous_wage", w.get("current_wage", 15.0)) for w in workers],
                dtype=torch.float32,
                device=self.device,
            ),
            "unemployment_months": torch.tensor(
                [w.get("unemployment_months", 0) for w in workers],
                dtype=torch.float32,
                device=self.device,
            ),
            "age": torch.tensor(
                [w.get("age", 30) for w in workers], dtype=torch.int32, device=self.device
            ),
            "archetype_id": torch.tensor(
                [w.get("archetype_id", 0) for w in workers], dtype=torch.int32, device=self.device
            ),
            "learned_v0": torch.tensor(
                [w.get("learned_v0", 6.5) for w in workers], dtype=torch.float32, device=self.device
            ),
            "attachment": torch.tensor(
                [w.get("attachment", 1.0) for w in workers], dtype=torch.float32, device=self.device
            ),
            "sex": torch.tensor(
                [w.get("sex", 0) for w in workers], dtype=torch.int32, device=self.device
            ),
            "previous_tenure_months": torch.tensor(
                [w.get("previous_tenure_months", 0) for w in workers],
                dtype=torch.int32,
                device=self.device,
            ),
            "separation_reason": torch.tensor(
                [w.get("separation_reason", 0) for w in workers],
                dtype=torch.int32,
                device=self.device,
            ),
            "lwf_retention_draw": torch.tensor(
                [w.get("lwf_retention_draw", 0.0) for w in workers],
                dtype=torch.float32,
                device=self.device,
            ),
            "res_wage_action": torch.tensor(
                [w.get("res_wage_action", 2) for w in workers],
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def _build_vacancy_tensors(self, vacancies: List[Dict]) -> Dict[str, torch.Tensor]:
        n = len(vacancies)
        if n == 0:
            empty = lambda: torch.empty(0, device=self.device)
            return {
                k: empty()
                for k in ["industry", "sector", "skill", "education", "cma", "wage", "security"]
            }
        return {
            "industry": torch.tensor(
                [v.get("naics_18", 0) for v in vacancies], dtype=torch.int32, device=self.device
            ),
            "sector": torch.tensor(
                [v.get("sector", 0) for v in vacancies], dtype=torch.int32, device=self.device
            ),
            "skill": torch.tensor(
                [v.get("skill_required", 2) for v in vacancies],
                dtype=torch.int32,
                device=self.device,
            ),
            "education": torch.tensor(
                [v.get("education_required", 3) for v in vacancies],
                dtype=torch.int32,
                device=self.device,
            ),
            "cma": torch.tensor(
                [v.get("cma", 535) for v in vacancies], dtype=torch.int32, device=self.device
            ),
            "wage": torch.tensor(
                [v.get("offered_wage", 20.0) for v in vacancies],
                dtype=torch.float32,
                device=self.device,
            ),
            "security": torch.tensor(
                [v.get("job_security", 0.5) for v in vacancies],
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def compute_match_quality_matrix_gpu(
        self, worker_tensors: Dict[str, torch.Tensor], vacancy_tensors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        n_workers = worker_tensors["skill"].shape[0]
        n_vacancies = vacancy_tensors["skill"].shape[0]
        if n_workers == 0 or n_vacancies == 0:
            return torch.empty(n_workers, n_vacancies, device=self.device)
        w_skill = worker_tensors["skill"].unsqueeze(1).float()
        w_industry = worker_tensors["industry"].unsqueeze(1)
        w_sector = worker_tensors["sector"].unsqueeze(1)
        w_edu = worker_tensors["education"].unsqueeze(1).float()
        w_exp = worker_tensors["experience"].unsqueeze(1)
        w_age = worker_tensors["age"].unsqueeze(1).float()
        v_skill = vacancy_tensors["skill"].unsqueeze(0).float()
        v_industry = vacancy_tensors["industry"].unsqueeze(0)
        v_sector = vacancy_tensors["sector"].unsqueeze(0)
        v_edu = vacancy_tensors["education"].unsqueeze(0).float()
        skill_diff = w_skill - v_skill
        uq_penalty, uq_floor = self.config.mq_underqual_params
        underqual_match = torch.where(
            skill_diff >= 0,
            torch.tensor(1.0, device=self.device),
            torch.clamp(1.0 + skill_diff * uq_penalty, min=uq_floor),
        )
        oq_penalty, oq_floor = self.config.mq_overqual_params
        overqual_match = torch.where(
            skill_diff <= 0,
            torch.tensor(1.0, device=self.device),
            torch.clamp(1.0 - skill_diff * oq_penalty, min=oq_floor),
        )
        skill_match = underqual_match * overqual_match
        same_industry = (w_industry == v_industry).float()
        same_sector = (w_sector == v_sector).float()
        industry_match = same_industry + self.config.mq_base_cross_sector_credit * same_sector * (
            1 - same_industry
        )
        edu_match = torch.clamp(
            1 - torch.abs(w_edu - v_edu) * self.config.mq_edu_penalty_per_level, min=0
        )
        exp_score = torch.clamp(w_exp / self.config.mq_exp_normalization, max=1.0)
        v_skill_idx = v_skill.long().clamp(0, self._mq_exp_weight_tensor.shape[0] - 1)
        exp_weight = self._mq_exp_weight_tensor[v_skill_idx]
        exp_contribution = exp_weight * exp_score
        entry_level_mask = (v_skill <= 2).float()
        age_adj_raw = _lookup_bins(w_age.squeeze(1), self._mq_age_ub, self._mq_age_vals).unsqueeze(
            1
        )
        age_adjustment = entry_level_mask * age_adj_raw
        quality_pre_mult = (
            self.config.skill_weight * skill_match
            + self.config.industry_weight * industry_match
            + self.config.education_weight * edu_match
            + exp_contribution
            + age_adjustment
        )
        jfr_multiplier = _lookup_bins(
            w_age.squeeze(1).float(), self._jfr_age_ub, self._jfr_age_vals
        ).unsqueeze(1)
        quality = quality_pre_mult * jfr_multiplier
        self._mq_audit_call_count += 1
        noise = (
            torch.randn(n_workers, n_vacancies, device=self.device) * self.config.match_noise_std
        )
        quality = torch.clamp(quality + noise, 0, 1)
        return quality

    def compute_acceptance_probs_sparse(
        self,
        worker_tensors: Dict[str, torch.Tensor],
        vacancy_tensors: Dict[str, torch.Tensor],
        match_quality_matrix: torch.Tensor,
        wage_multipliers: torch.Tensor,
        worker_indices: torch.Tensor,
        vacancy_indices: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None,
        _acceptance_actual_offer: bool = False,
    ) -> torch.Tensor:
        n_pairs = worker_indices.shape[0]
        if n_pairs == 0:
            return torch.empty(0, device=self.device)
        v_wage = vacancy_tensors["wage"][vacancy_indices] * wage_multipliers[vacancy_indices]
        v_security = vacancy_tensors["security"][vacancy_indices]
        w_unemp_months = worker_tensors["unemployment_months"][worker_indices]
        pair_qualities = match_quality_matrix[worker_indices, vacancy_indices]
        commute_utility = torch.zeros(n_pairs, device=self.device)
        if distance_matrix is not None:
            travel_time_min = distance_matrix[worker_indices, vacancy_indices]
            center = self.config.mnl_config.commute_reference_min
            desperation_commute = _lookup_bins(
                w_unemp_months.float(), self._commute_desp_ub, self._commute_desp_vals
            )
            _arche_raw = worker_tensors["archetype_id"][worker_indices].long()
            _n_table = self._archetype_commute_multipliers.numel()
            _valid_mask = (_arche_raw >= 0) & (_arche_raw < _n_table)
            _arche_safe = torch.where(_valid_mask, _arche_raw, torch.zeros_like(_arche_raw))
            _gathered = self._archetype_commute_multipliers[_arche_safe]
            archetype_mult = torch.where(_valid_mask, _gathered, torch.ones_like(_gathered))
            effective_beta_commute = self._mnl_beta_commute * desperation_commute * archetype_mult
            commute_utility = effective_beta_commute * (travel_time_min - center) / center
            max_commute = self.config.mnl_config.max_acceptable_commute_min
            excess = torch.clamp(travel_time_min - max_commute, min=0.0)
            commute_utility = (
                commute_utility + self._mnl_beta_commute * archetype_mult * excess / center
            )
        job_utility = (
            self._mnl_beta_wage * torch.log(v_wage + 1)
            + self._mnl_beta_match * pair_qualities
            + self._mnl_beta_security * v_security
            + commute_utility
        )
        _acceptance_a_i = job_utility
        if _acceptance_extension.is_recentred():
            job_utility = job_utility + self._acceptance_recentre_c
        mnl = self.config.mnl_config
        wage_gap_beta = float(getattr(mnl, "wage_gap_penalty_beta", -1.5))
        w_current_wage = worker_tensors["wage"][worker_indices]
        w_skill_for_res = worker_tensors["skill"][worker_indices]
        if "previous_wage" in worker_tensors:
            w_prior_wage = worker_tensors["previous_wage"][worker_indices]
            anchor_wage = torch.maximum(w_prior_wage, w_current_wage)
        else:
            anchor_wage = w_current_wage
        if self.reservation_bundle_enabled and "separation_reason" in worker_tensors:
            _sep_v = worker_tensors["separation_reason"][worker_indices].long()
            if "lwf_retention_draw" in worker_tensors:
                _ret_v = worker_tensors["lwf_retention_draw"][worker_indices].float()
            else:
                _ret_v = torch.zeros_like(anchor_wage)
            _baseline = float(mnl.reservation_wage_base_ratio)
            _retention_clipped = torch.clamp(_ret_v, max=_baseline)
            _is_displaced = (_sep_v == 1) & (_ret_v > 0)
            base_ratio_vec = torch.where(
                _is_displaced, _retention_clipped, torch.full_like(_ret_v, _baseline)
            )
            if "unemployment_months" in worker_tensors:
                _u_mo = worker_tensors["unemployment_months"][worker_indices].float()
            else:
                _u_mo = torch.zeros_like(anchor_wage)
            _yr_vec = _u_mo / 12.0
            _K = float(self.reservation_decay_per_year)
            if self.reservation_decay_compounded:
                km_factor_vec = torch.clamp(
                    torch.pow(torch.tensor(1.0 - _K, device=_yr_vec.device), _yr_vec), min=0.0
                )
            else:
                km_factor_vec = torch.clamp(1.0 - _K * _yr_vec, min=0.0)
            anchor = anchor_wage * base_ratio_vec * km_factor_vec
        else:
            anchor = anchor_wage * mnl.reservation_wage_base_ratio
        _mult_dict = self.config.worker_res_wage_multipliers
        _mult_table = torch.tensor(
            [float(_mult_dict.get(i, 1.0)) for i in range(5)],
            dtype=torch.float32,
            device=self.device,
        )
        if "res_wage_action" in worker_tensors:
            _w_rwa = worker_tensors["res_wage_action"][worker_indices].long().clamp_(0, 4)
            mult_vec = _mult_table[_w_rwa]
        else:
            mult_vec = torch.ones_like(anchor)
        skill_floor = (
            mnl.reservation_wage_skill_floor_base
            + w_skill_for_res.float() * mnl.reservation_wage_skill_floor_mult
        )
        res_wage_vec = torch.maximum(anchor * mult_vec, skill_floor)
        gap_ratio = torch.clamp(1.0 - v_wage / (res_wage_vec + 1e-06), min=0.0)
        wage_gap_utility = wage_gap_beta * gap_ratio
        job_utility = job_utility + wage_gap_utility
        reservation_utility = worker_tensors["learned_v0"][worker_indices]
        w_age = worker_tensors["age"][worker_indices].float()
        utility_diff = job_utility - reservation_utility
        center = torch.tensor(mnl.age_penalty_center, device=self.device)
        steep = torch.tensor(mnl.age_penalty_steepness, device=self.device)
        raw = torch.sigmoid((w_age - center) / steep)
        pen_min = torch.tensor(mnl.age_penalty_min, device=self.device)
        pen_max = torch.tensor(mnl.age_penalty_max, device=self.device)
        age_penalty = pen_min + (pen_max - pen_min) * raw
        utility_diff = utility_diff + age_penalty
        accept_prob = torch.sigmoid(utility_diff / self._mnl_scale)
        accept_prob = torch.clamp(accept_prob, min=0.0, max=1.0)
        if _acceptance_actual_offer:
            _acceptance_extension.record_context(
                self,
                worker_tensors,
                worker_indices,
                vacancy_indices,
                _acceptance_a_i,
                utility_diff,
                accept_prob,
                job_security=v_security,
            )
        wage_utility = self._mnl_beta_wage * torch.log(v_wage + 1)
        match_utility = self._mnl_beta_match * pair_qualities
        self._last_rejection_diagnostics = {
            "wage_utility": wage_utility.cpu(),
            "match_utility": match_utility.cpu(),
            "commute_utility": commute_utility.cpu(),
            "wage_gap_utility": wage_gap_utility.cpu(),
            "job_utility": job_utility.cpu(),
            "reservation_utility": reservation_utility.cpu(),
            "utility_diff": utility_diff.cpu(),
            "pre_friction_prob": accept_prob.cpu().clone(),
        }
        friction = torch.ones_like(w_age)
        self._last_rejection_diagnostics["friction_multiplier"] = friction.cpu()
        self._last_rejection_diagnostics["final_accept_prob"] = accept_prob.cpu()
        return accept_prob

    def compute_relevance_matrix_gpu(
        self,
        worker_tensors: Dict[str, torch.Tensor],
        vacancy_tensors: Dict[str, torch.Tensor],
        worker_da_ids: Optional[List[int]] = None,
        vacancy_da_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        n_workers = worker_tensors["skill"].shape[0]
        n_vacancies = vacancy_tensors["skill"].shape[0]
        if n_workers == 0 or n_vacancies == 0:
            return torch.empty(n_workers, n_vacancies, device=self.device)
        w_industry = worker_tensors["industry"].unsqueeze(1)
        w_sector = worker_tensors["sector"].unsqueeze(1)
        w_skill = worker_tensors["skill"].unsqueeze(1)
        w_edu = worker_tensors["education"].unsqueeze(1)
        w_cma = worker_tensors["cma"].unsqueeze(1)
        w_unemp_duration = worker_tensors["unemployment_months"].unsqueeze(1)
        v_industry = vacancy_tensors["industry"].unsqueeze(0)
        v_sector = vacancy_tensors["sector"].unsqueeze(0)
        v_skill = vacancy_tensors["skill"].unsqueeze(0)
        v_edu = vacancy_tensors["education"].unsqueeze(0)
        v_cma = vacancy_tensors["cma"].unsqueeze(0)
        same_industry = (w_industry == v_industry).float()
        same_sector = (w_sector == v_sector).float()
        rel_ss_base = self.config.rel_same_sector_base
        rel_ds_base = self.config.rel_diff_sector_base
        dur_coeffs = _lookup_bins_kd(
            w_unemp_duration.squeeze(1).float(), self._rel_dur_flex_ub, self._rel_dur_flex_vals
        )
        ss_mult = dur_coeffs[:, 0].unsqueeze(1)
        ss_offset = dur_coeffs[:, 1].unsqueeze(1)
        ds_mult = dur_coeffs[:, 2].unsqueeze(1)
        ds_offset = dur_coeffs[:, 3].unsqueeze(1)
        same_sector_score = torch.clamp(rel_ss_base * ss_mult + ss_offset, max=1.0)
        diff_sector_score = torch.clamp(
            rel_ds_base * ds_mult + ds_offset, max=self.config.rel_diff_sector_desperate_cap
        )
        industry_score = (
            same_industry
            + (1 - same_industry) * same_sector * same_sector_score
            + (1 - same_industry) * (1 - same_sector) * diff_sector_score
        )
        if (
            worker_da_ids is not None
            and vacancy_da_ids is not None
            and (self.da_distance_matrix is not None or self.da_travel_time_matrix is not None)
            and (len(worker_da_ids) == n_workers)
            and (len(vacancy_da_ids) == n_vacancies)
        ):
            commute_matrix = self.compute_distance_matrix_gpu(worker_da_ids, vacancy_da_ids)
            use_minutes = bool(self._use_travel_time and self.da_travel_time_matrix is not None)
            if use_minutes:
                center = torch.tensor(self.config.commute_gate_center_min, device=self.device)
                steep = torch.tensor(self.config.commute_gate_steepness_min, device=self.device)
                very_long_thr = torch.tensor(
                    self.config.commute_gate_very_long_min, device=self.device
                )
            else:
                center = torch.tensor(self.config.commute_gate_center_km, device=self.device)
                steep = torch.tensor(self.config.commute_gate_steepness_km, device=self.device)
                very_long_thr = torch.tensor(
                    self.config.commute_gate_very_long_km, device=self.device
                )
            commute_score = torch.sigmoid((center - commute_matrix) / steep)
            very_long_mask = commute_matrix > very_long_thr
            commute_score = torch.where(
                very_long_mask,
                commute_score * float(self.config.commute_gate_very_long_mult),
                commute_score,
            )
            commute_score = torch.clamp(commute_score, min=float(self.config.commute_gate_floor))
            unit = "min" if use_minutes else "km"
            thr = float(self.config.commute_gate_meaningful_threshold)
            feasible_count = (commute_score > thr).sum(dim=1)
            min_k = int(self.config.min_feasible_vacancies)
            n_isolated = int((feasible_count < min_k).sum().item())
            self._last_commute_diagnostics = {
                "n_isolated_workers": n_isolated,
                "n_total_workers": n_workers,
                "pct_isolated": n_isolated / max(n_workers, 1),
                "mean_commute_score": float(commute_score.mean().item()),
                f"feasible_30{unit}": int((commute_matrix <= 30).any(dim=1).sum().item()),
                f"feasible_45{unit}": int((commute_matrix <= 45).any(dim=1).sum().item()),
                f"feasible_60{unit}": int((commute_matrix <= 60).any(dim=1).sum().item()),
            }
        else:
            same_cma = (w_cma == v_cma).float()
            commute_score = same_cma + self.config.rel_cross_cma_score * (1 - same_cma)
            commute_score = torch.clamp(commute_score, min=self.config.commute_gate_floor)
            self._last_commute_diagnostics = None
        w_age = worker_tensors["age"].unsqueeze(1).float()
        skill_diff = w_skill.float() - v_skill.float()
        neg_diff = (-skill_diff).long().clamp(0, self._rel_uq_scores.shape[0] - 1)
        underqual_base = torch.where(
            skill_diff >= 0, torch.tensor(1.0, device=self.device), self._rel_uq_scores[neg_diff]
        )
        underqual_duration_boost = _lookup_bins(
            w_unemp_duration.squeeze(1).float(), self._rel_uq_boost_ub, self._rel_uq_boost_vals
        ).unsqueeze(1)
        underqual_score = torch.clamp(underqual_base * underqual_duration_boost, max=1.0)
        pos_diff = skill_diff.long().clamp(0, self._rel_oq_scores.shape[0] - 1)
        overqual_base = torch.where(
            skill_diff <= 0, torch.tensor(1.0, device=self.device), self._rel_oq_scores[pos_diff]
        )
        overqual_duration_boost = _lookup_bins(
            w_unemp_duration.squeeze(1).float(), self._rel_oq_boost_ub, self._rel_oq_boost_vals
        ).unsqueeze(1)
        overqual_score = torch.clamp(
            overqual_base * overqual_duration_boost, max=self.config.rel_overqual_score_max
        )
        skill_score = underqual_score * overqual_score
        entry_level_mask = (v_skill <= 2).float()
        age_mult_raw = _lookup_bins(
            w_age.squeeze(1), self._rel_age_entry_ub, self._rel_age_entry_vals
        ).unsqueeze(1)
        age_multiplier = torch.where(
            entry_level_mask > 0, age_mult_raw, torch.tensor(1.0, device=self.device)
        )
        skill_score = skill_score * age_multiplier
        edu_diff = v_edu - w_edu
        education_score = torch.where(
            edu_diff <= 0,
            1.0,
            torch.where(
                edu_diff == 1,
                torch.tensor(self.config.rel_edu_diff_1_score, device=self.device),
                torch.tensor(self.config.rel_edu_diff_2plus_score, device=self.device),
            ),
        )
        base_match = (
            self.config.relevance_industry_weight * industry_score
            + self.config.relevance_skill_weight * skill_score
            + self.config.relevance_education_weight * education_score
        )
        relevance = base_match * commute_score
        relevance = torch.clamp(relevance, min=self.config.relevance_min)
        return relevance

    def apply_channel_modifiers_gpu(
        self,
        base_relevance: torch.Tensor,
        worker_tensors: Dict[str, torch.Tensor],
        vacancy_tensors: Dict[str, torch.Tensor],
        channel: str,
    ) -> torch.Tensor:
        w_industry = worker_tensors["industry"].unsqueeze(1)
        w_skill = worker_tensors["skill"].unsqueeze(1)
        w_cma = worker_tensors["cma"].unsqueeze(1)
        v_industry = vacancy_tensors["industry"].unsqueeze(0)
        v_skill = vacancy_tensors["skill"].unsqueeze(0)
        v_cma = vacancy_tensors["cma"].unsqueeze(0)
        same_industry = (w_industry == v_industry).float()
        same_cma = (w_cma == v_cma).float()
        skill_diff = torch.abs(w_skill - v_skill)
        skill_match = (skill_diff == 0).float()
        channel_effectiveness = getattr(self.config.mnl_config, "channel_effectiveness", {})
        channel_to_int = {
            "Ads/Online": 0,
            "Direct Contact": 1,
            "Networking": 2,
            "Public Agency": 3,
            "Private Agency": 4,
        }
        channel_int = channel_to_int.get(channel, 0)
        base_effectiveness = channel_effectiveness.get(channel_int, 1.0)
        channel_name = _CHANNEL_STR_TO_NAME.get(channel, channel)
        ch_mods = self.config.mnl_config.channel_context_modifiers.get(channel_name, {})
        if channel_name == "ads_online":
            skill_score_approx = torch.where(
                skill_diff == 0,
                1.0,
                torch.where(skill_diff == 1, 0.7, torch.where(skill_diff == 2, 0.3, 0.1)),
            )
            skill_range = ch_mods.get("skill_range", 0.2)
            modifier = base_effectiveness * (1.0 + skill_range * (skill_score_approx - 0.5))
        elif channel_name == "direct_contact":
            modifier = torch.where(
                same_industry * same_cma > 0,
                torch.tensor(
                    base_effectiveness * ch_mods.get("same_ind_cma", 1.3), device=self.device
                ),
                torch.where(
                    same_cma > 0,
                    torch.tensor(
                        base_effectiveness * ch_mods.get("same_cma", 1.1), device=self.device
                    ),
                    torch.tensor(
                        base_effectiveness * ch_mods.get("distant", 0.7), device=self.device
                    ),
                ),
            )
        elif channel_name == "networking":
            modifier = torch.where(
                same_industry > 0,
                torch.tensor(
                    base_effectiveness * ch_mods.get("same_ind", 1.25), device=self.device
                ),
                torch.tensor(
                    base_effectiveness * ch_mods.get("diff_ind", 0.95), device=self.device
                ),
            )
        elif channel_name == "public_agency":
            modifier = torch.tensor(
                base_effectiveness * ch_mods.get("quality_mult", 0.9), device=self.device
            )
        elif channel_name == "private_agency":
            perfect_match = same_industry * skill_match
            partial_match = (same_industry + skill_match) * (1 - perfect_match)
            modifier = torch.where(
                perfect_match > 0,
                torch.tensor(base_effectiveness * ch_mods.get("perfect", 1.4), device=self.device),
                torch.where(
                    partial_match > 0,
                    torch.tensor(
                        base_effectiveness * ch_mods.get("partial", 1.1), device=self.device
                    ),
                    torch.tensor(
                        base_effectiveness * ch_mods.get("no_match", 0.8), device=self.device
                    ),
                ),
            )
        else:
            modifier = torch.tensor(1.0, device=self.device)
        return base_relevance * modifier

    def run_k_round_matching_gpu(
        self,
        worker_tensors: Dict[str, torch.Tensor],
        vacancy_tensors: Dict[str, torch.Tensor],
        worker_ids: List[str],
        vacancy_ids: List[str],
        vacancy_to_firm: Dict[str, str],
        applications: Dict[str, List[str]],
        wage_strategies: Dict[str, int],
        workers_list: Optional[List[Dict]] = None,
        vacancies_list: Optional[List[Dict]] = None,
        hiring_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        n_workers = len(worker_ids)
        n_vacancies = len(vacancy_ids)
        K = self.config.n_matching_rounds
        if n_workers == 0 or n_vacancies == 0:
            return []
        worker_id_to_idx = {wid: i for i, wid in enumerate(worker_ids)}
        vacancy_id_to_idx = {vid: i for i, vid in enumerate(vacancy_ids)}
        _app_counts = Counter((wid for applicants in applications.values() for wid in applicants))
        for _wid in worker_ids:
            self._audit_worker_event(_wid, "seeker_months")
            _n_apps = int(_app_counts.get(_wid, 0))
            if _n_apps:
                self._audit_worker_event(_wid, "applications", _n_apps)
            else:
                self._audit_worker_event(_wid, "no_suitable_vacancy_months")
        distance_matrix = None
        if workers_list and vacancies_list and self.da_centroids:
            worker_da_ids = [w.get("da_id") for w in workers_list]
            vacancy_da_ids = [v.get("da_id") for v in vacancies_list]
            if any((da is not None for da in worker_da_ids)) and any(
                (da is not None for da in vacancy_da_ids)
            ):
                distance_matrix = self.compute_distance_matrix_gpu(
                    [da if da is not None else 0 for da in worker_da_ids],
                    [da if da is not None else 0 for da in vacancy_da_ids],
                    worker_ids=worker_ids,
                )
        match_quality_matrix = self.compute_match_quality_matrix_gpu(
            worker_tensors, vacancy_tensors
        )
        wage_multipliers = torch.ones(len(vacancy_ids), dtype=torch.float32, device=self.device)
        self._sparse_worker_tensors = worker_tensors
        self._sparse_vacancy_tensors = vacancy_tensors
        self._sparse_match_quality = match_quality_matrix
        self._sparse_wage_mult = wage_multipliers
        self._sparse_distance_matrix = distance_matrix
        if hiring_thresholds:
            threshold_list = [hiring_thresholds.get(vid, 0.3) for vid in vacancy_ids]
            hiring_threshold_tensor = torch.tensor(
                threshold_list, dtype=torch.float32, device=self.device
            )
        else:
            hiring_threshold_tensor = torch.zeros(
                n_vacancies, dtype=torch.float32, device=self.device
            )
        ranked_applicants = {}
        screening_stats = {"total_applicants": 0, "screened_out": 0}
        _use_min = bool(self._use_travel_time and self.da_travel_time_matrix is not None)
        _commute_gate_thr = (
            self.config.commute_gate_center_min if _use_min else self.config.commute_gate_center_km
        )
        for vid, applicant_wids in applications.items():
            if vid not in vacancy_id_to_idx:
                continue
            v_idx = vacancy_id_to_idx[vid]
            w_indices = [worker_id_to_idx[wid] for wid in applicant_wids if wid in worker_id_to_idx]
            if not w_indices:
                continue
            w_idx_tensor = torch.tensor(w_indices, dtype=torch.long, device=self.device)
            qualities = match_quality_matrix[w_idx_tensor, v_idx]
            threshold = hiring_threshold_tensor[v_idx]
            structural_floor = getattr(self.config, "min_match_quality_floor", 0.25)
            effective_threshold = max(float(threshold), structural_floor)
            pass_screen = qualities >= effective_threshold
            n_this = len(w_indices)
            n_failed = int((~pass_screen).sum().item())
            screening_stats["total_applicants"] += n_this
            screening_stats["screened_out"] += n_failed
            if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
                self, "_person_trace_active", False
            ):
                _trace_passes = pass_screen.cpu().tolist()
                _trace_qualities = qualities.cpu().tolist()
                for _trace_i, _trace_widx in enumerate(w_indices):
                    _trace_wid = worker_ids[_trace_widx]
                    _trace_passed = bool(_trace_passes[_trace_i])
                    self._person_trace_event(
                        _trace_wid, "screen_pass" if _trace_passed else "screen_fail"
                    )
                    _trace_floor = float(structural_floor)
                    self._person_trace_detail(
                        _trace_wid,
                        "screening_events",
                        {
                            "vacancy_id": str(vid),
                            "firm_id": str(vacancy_to_firm.get(vid, "")),
                            "match_quality": float(_trace_qualities[_trace_i]),
                            "firm_threshold": float(threshold),
                            "structural_floor": _trace_floor,
                            "effective_threshold": max(float(threshold), _trace_floor),
                            "passed": _trace_passed,
                            "vacancy_industry": int(vacancy_tensors["industry"][v_idx].item()),
                        },
                    )
            _v_ind_int = int(vacancy_tensors["industry"][v_idx].item())
            _v_sec_int = int(vacancy_tensors["sector"][v_idx].item())
            _w_inds_screen = worker_tensors["industry"][w_idx_tensor]
            _w_secs_screen = worker_tensors["sector"][w_idx_tensor]
            _same_ind_s = _w_inds_screen == _v_ind_int
            _same_sec_s = _w_secs_screen == _v_sec_int
            _tier_same_ind = _same_ind_s
            _tier_same_sec = ~_same_ind_s & _same_sec_s
            _tier_diff_sec = ~_same_ind_s & ~_same_sec_s
            _w_durs_screen = worker_tensors["unemployment_months"][w_idx_tensor]
            _dur_0_6 = _w_durs_screen < 6
            _dur_6_12 = (_w_durs_screen >= 6) & (_w_durs_screen < 12)
            _dur_12_inf = _w_durs_screen >= 12
            for _tier_name, _tier_mask in (
                ("same_industry", _tier_same_ind),
                ("same_sector_diff_industry", _tier_same_sec),
                ("diff_sector", _tier_diff_sec),
            ):
                _n_pass = int((pass_screen & _tier_mask).sum().item())
                _n_fail = int((~pass_screen & _tier_mask).sum().item())
                self.stats[f"screen_pass_{_tier_name}"] += _n_pass
                self.stats[f"screen_fail_{_tier_name}"] += _n_fail
                for _bin_name, _bin_mask in (
                    ("dur_0_6", _dur_0_6),
                    ("dur_6_12", _dur_6_12),
                    ("dur_12_inf", _dur_12_inf),
                ):
                    _n_p = int((pass_screen & _tier_mask & _bin_mask).sum().item())
                    _n_f = int((~pass_screen & _tier_mask & _bin_mask).sum().item())
                    self.stats[f"screen_pass_{_tier_name}_{_bin_name}"] += _n_p
                    self.stats[f"screen_fail_{_tier_name}_{_bin_name}"] += _n_f
            if n_failed > 0:
                failed_mask = ~pass_screen
                failed_w_idx = w_idx_tensor[failed_mask]
                failed_qualities = qualities[failed_mask]
                if distance_matrix is not None:
                    failed_commutes = distance_matrix[failed_w_idx, v_idx]
                else:
                    failed_commutes = None
                v_ind = int(vacancy_tensors["industry"][v_idx].item())
                v_sec = int(vacancy_tensors["sector"][v_idx].item())
                failed_industries = worker_tensors["industry"][failed_w_idx]
                failed_sectors = worker_tensors["sector"][failed_w_idx]
                for k in range(len(failed_w_idx)):
                    q = float(failed_qualities[k].item())
                    firm_thr = float(threshold)
                    floor = float(structural_floor)
                    floor_only_binding = floor > firm_thr and firm_thr <= q < floor
                    _failed_wid = worker_ids[int(failed_w_idx[k].item())]
                    if q < firm_thr:
                        self._audit_worker_event(_failed_wid, "screen_fail_firm_threshold")
                    elif q < floor:
                        self._audit_worker_event(_failed_wid, "screen_fail_structural_floor")
                    else:
                        self._audit_worker_event(_failed_wid, "screen_fail_other")
                    if q < firm_thr:
                        self.stats["screen_quality_below_firm_threshold"] += 1
                    if q < floor:
                        self.stats["screen_quality_below_structural_floor"] += 1
                    if floor_only_binding:
                        self.stats["screen_quality_floor_binding_band"] += 1
                    _scr_reason = None
                    if (
                        failed_commutes is not None
                        and float(failed_commutes[k].item()) > _commute_gate_thr * 1.5
                    ):
                        self.stats["screen_reject_commute_gate"] += 1
                        self.stats["screen_reject_quality_with_long_commute"] += 1
                        _scr_reason = "commute_gate"
                    elif (
                        int(failed_industries[k].item()) != v_ind
                        and int(failed_sectors[k].item()) != v_sec
                    ):
                        self.stats["screen_reject_industry_mismatch"] += 1
                        _scr_reason = "industry_mismatch"
                    elif floor_only_binding:
                        self.stats["screen_reject_match_quality_floor"] += 1
                        _scr_reason = "match_quality_floor"
                    else:
                        self.stats["screen_reject_selectivity"] += 1
                        _scr_reason = "selectivity"
                    _scr_home_da = self._da_by_widx.get(int(failed_w_idx[k].item()))
                    if _scr_home_da is not None and _scr_reason is not None:
                        _scr_by_da_entry = self.rejection_reasons_by_da.get(_scr_home_da)
                        if _scr_by_da_entry is None:
                            _scr_by_da_entry = {}
                            self.rejection_reasons_by_da[_scr_home_da] = _scr_by_da_entry
                        _scr_by_da_entry[_scr_reason] = _scr_by_da_entry.get(_scr_reason, 0) + 1
                if not hasattr(self, "_screen_diag_all"):
                    self._screen_diag_all = []
                    self._screen_diag_reasons = []
                    self._screen_diag_step = 0
                for k in range(len(failed_w_idx)):
                    fq_val = float(failed_qualities[k].item())
                    if (
                        failed_commutes is not None
                        and float(failed_commutes[k].item()) > _commute_gate_thr * 1.5
                    ):
                        reason = "commute"
                    elif (
                        int(failed_industries[k].item()) != v_ind
                        and int(failed_sectors[k].item()) != v_sec
                    ):
                        reason = "industry"
                    else:
                        reason = "selectivity"
                    self._screen_diag_all.append(fq_val)
                    self._screen_diag_reasons.append(reason)
                self._screen_diag_step += 1
                if self._screen_diag_step % 10 == 0 and len(self._screen_diag_all) >= 5:
                    self._screen_diag_all.clear()
                    self._screen_diag_reasons.clear()
            surviving_indices = w_idx_tensor[pass_screen]
            surviving_qualities = qualities[pass_screen]
            if len(surviving_indices) == 0:
                continue
            n_survivors = len(surviving_indices)
            v_idx_repeated = torch.full((n_survivors,), v_idx, dtype=torch.long, device=self.device)
            accept_probs = self.compute_acceptance_probs_sparse(
                worker_tensors,
                vacancy_tensors,
                match_quality_matrix,
                wage_multipliers,
                surviving_indices,
                v_idx_repeated,
                distance_matrix=distance_matrix,
            )
            expected_fill = surviving_qualities * accept_probs
            sorted_order = torch.argsort(expected_fill, descending=True)
            ranked_applicants[v_idx] = surviving_indices[sorted_order].tolist()
        self.stats["applicants_screened_out"] = screening_stats["screened_out"]
        self.stats["screening_rate"] = screening_stats["screened_out"] / max(
            screening_stats["total_applicants"], 1
        )
        self.stats["n_applicants_total"] = screening_stats["total_applicants"]
        self.stats["n_applicants_after_filters"] = (
            screening_stats["total_applicants"] - screening_stats["screened_out"]
        )
        matched_workers = set()
        rf_outside_workers = set()
        filled_vacancies = set()
        matches = []
        all_accept_probs = []
        all_attachment_adjusted_probs = []
        attachment_conversion_failures_total = 0
        workers_offered = set()
        offers_to_unemployed = 0
        offers_to_employed = 0
        firm_hires_this_step = {}
        firm_vacancy_count = {}
        for vid in vacancy_ids:
            fid = vacancy_to_firm.get(vid, "")
            if fid:
                firm_vacancy_count[fid] = firm_vacancy_count.get(fid, 0) + 1
        vacancy_offer_position = {v_idx: 0 for v_idx in ranked_applicants}
        for round_k in range(K):
            round_offers = []
            for v_idx, ranked_list in ranked_applicants.items():
                if v_idx in filled_vacancies:
                    continue
                pos = vacancy_offer_position[v_idx]
                while pos < len(ranked_list):
                    w_idx = ranked_list[pos]
                    if w_idx not in matched_workers and w_idx not in rf_outside_workers:
                        round_offers.append((w_idx, v_idx))
                        vacancy_offer_position[v_idx] = pos + 1
                        break
                    pos += 1
                    vacancy_offer_position[v_idx] = pos
            if not round_offers:
                break
            offer_w_indices = torch.tensor(
                [o[0] for o in round_offers], dtype=torch.long, device=self.device
            )
            offer_v_indices = torch.tensor(
                [o[1] for o in round_offers], dtype=torch.long, device=self.device
            )
            offer_accept_probs = self.compute_acceptance_probs_sparse(
                self._sparse_worker_tensors,
                self._sparse_vacancy_tensors,
                self._sparse_match_quality,
                self._sparse_wage_mult,
                offer_w_indices,
                offer_v_indices,
                distance_matrix=self._sparse_distance_matrix,
                _acceptance_actual_offer=True,
            )
            all_accept_probs.extend(offer_accept_probs.tolist())
            rand_worker = torch.rand(len(round_offers), device=self.device)
            worker_accepts = rand_worker < offer_accept_probs
            _acceptance_extension.record_decisions(self, offer_accept_probs, worker_accepts)
            if self.attachment_enabled:
                worker_attachment = self._sparse_worker_tensors["attachment"][offer_w_indices]
                alpha = float(self.attachment_alpha)
                attachment_mult = alpha + (1.0 - alpha) * worker_attachment
                rand_attachment = torch.rand(len(round_offers), device=self.device)
                attachment_passes = rand_attachment < attachment_mult
                accepts = worker_accepts & attachment_passes
                _adj = offer_accept_probs * attachment_mult
                all_attachment_adjusted_probs.extend(_adj.tolist())
                _conv_fail = (worker_accepts & ~attachment_passes).sum().item()
                attachment_conversion_failures_total += int(_conv_fail)
            else:
                accepts = worker_accepts
            offer_qualities = match_quality_matrix[offer_w_indices, offer_v_indices]
            offer_wages = (
                vacancy_tensors["wage"][offer_v_indices] * wage_multipliers[offer_v_indices]
            )
            accepts_cpu = accepts.cpu().tolist()
            worker_accepts_cpu = worker_accepts.cpu().tolist()
            qualities_cpu = offer_qualities.cpu().tolist()
            wages_cpu = offer_wages.cpu().tolist()
            worker_offers_this_round = {}
            if len(round_offers) > 0:
                _ow_inds = worker_tensors["industry"][offer_w_indices]
                _ow_secs = worker_tensors["sector"][offer_w_indices]
                _ov_inds = vacancy_tensors["industry"][offer_v_indices]
                _ov_secs = vacancy_tensors["sector"][offer_v_indices]
                _same_ind_o = _ow_inds == _ov_inds
                _same_sec_o = _ow_secs == _ov_secs
                self.stats["offer_same_industry"] += int(_same_ind_o.sum().item())
                self.stats["offer_same_sector_diff_industry"] += int(
                    (~_same_ind_o & _same_sec_o).sum().item()
                )
                self.stats["offer_diff_sector"] += int((~_same_ind_o & ~_same_sec_o).sum().item())
                for _wi in offer_w_indices.tolist():
                    _b = self._dur_bin_by_widx.get(int(_wi))
                    if _b is not None:
                        self.stats[f"offer_{_b}"] += 1
            diag = getattr(self, "_last_rejection_diagnostics", {})
            for i, (w_idx, v_idx) in enumerate(round_offers):
                workers_offered.add(w_idx)
                self._audit_worker_event(worker_ids[w_idx], "offers")
                if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
                    self, "_person_trace_active", False
                ):
                    if worker_accepts_cpu[i]:
                        self._person_trace_event(worker_ids[w_idx], "preference_accepts")
                    if accepts_cpu[i]:
                        self._person_trace_event(worker_ids[w_idx], "candidate_accepts")
                    if self.attachment_enabled and worker_accepts_cpu[i] and (not accepts_cpu[i]):
                        self._person_trace_event(
                            worker_ids[w_idx], "attachment_conversion_failures"
                        )
                    self._person_trace_detail(
                        worker_ids[w_idx],
                        "offer_events",
                        {
                            "round_index": int(round_k),
                            "vacancy_id": str(vacancy_ids[v_idx]),
                            "firm_id": str(vacancy_to_firm.get(vacancy_ids[v_idx], "")),
                            "posted_wage": float(wages_cpu[i]),
                            "match_quality": float(qualities_cpu[i]),
                            "preference_accepted": bool(worker_accepts_cpu[i]),
                            "operational_accepted": bool(accepts_cpu[i]),
                            "preference_acceptance_probability": float(
                                offer_accept_probs[i].item()
                            ),
                            "commute_value": (
                                float(distance_matrix[w_idx, v_idx].item())
                                if distance_matrix is not None
                                else None
                            ),
                            "commute_units": "minutes" if self._use_travel_time else "km",
                        },
                    )
                if accepts_cpu[i]:
                    if w_idx not in worker_offers_this_round:
                        worker_offers_this_round[w_idx] = []
                    worker_offers_this_round[w_idx].append((v_idx, qualities_cpu[i], wages_cpu[i]))
                elif self.attachment_enabled and worker_accepts_cpu[i]:
                    continue
                else:
                    self.stats["rejections"] += 1
                    self._audit_worker_event(worker_ids[w_idx], "offers_rejected")
                    if diag and i < len(diag.get("utility_diff", [])):
                        wage_u = float(diag["wage_utility"][i])
                        match_u = float(diag["match_utility"][i])
                        commute_u = float(diag["commute_utility"][i])
                        wage_gap_u = (
                            float(diag.get("wage_gap_utility", torch.zeros(1))[i])
                            if "wage_gap_utility" in diag and i < len(diag["wage_gap_utility"])
                            else 0.0
                        )
                        utility_diff = float(diag["utility_diff"][i])
                        _rej_reason = None
                        if wage_gap_u < -0.3:
                            self.rejection_reasons["wage_gap"] += 1
                            _rej_reason = "wage_gap"
                        elif commute_u < -0.5:
                            self.rejection_reasons["long_commute"] += 1
                            _rej_reason = "long_commute"
                        elif wage_u < 2.5:
                            self.rejection_reasons["low_wage_utility"] += 1
                            _rej_reason = "low_wage_utility"
                        elif match_u < 0.3:
                            self.rejection_reasons["poor_match_quality"] += 1
                            _rej_reason = "poor_match_quality"
                        elif utility_diff < 0:
                            self.rejection_reasons["high_reservation"] += 1
                            _rej_reason = "high_reservation"
                        else:
                            self.rejection_reasons["random_rejection"] += 1
                            _rej_reason = "random_rejection"
                        _rej_home_da = self._da_by_widx.get(int(w_idx))
                        if _rej_home_da is not None and _rej_reason is not None:
                            _by_da_entry = self.rejection_reasons_by_da.get(_rej_home_da)
                            if _by_da_entry is None:
                                _by_da_entry = {}
                                self.rejection_reasons_by_da[_rej_home_da] = _by_da_entry
                            _by_da_entry[_rej_reason] = _by_da_entry.get(_rej_reason, 0) + 1
                        if wage_gap_u < -0.3:
                            self.stats["reject_component_wage_gap"] += 1
                        if commute_u < -0.5:
                            self.stats["reject_component_long_commute"] += 1
                        if wage_u < 2.5:
                            self.stats["reject_component_low_wage_utility"] += 1
                        if match_u < 0.3:
                            self.stats["reject_component_poor_match_quality"] += 1
                        if utility_diff < 0:
                            self.stats["reject_component_high_reservation"] += 1
                        if utility_diff < 0:
                            _contribs = {
                                "wage_gap": max(0.0, -wage_gap_u - 0.3),
                                "long_commute": max(0.0, -commute_u - 0.5),
                                "low_wage_utility": max(0.0, 2.5 - wage_u),
                                "poor_match_quality": max(0.0, 0.3 - match_u),
                            }
                            _dom = max(_contribs, key=_contribs.get)
                            _dur_b = self._dur_bin_by_widx.get(int(w_idx))
                            if _contribs[_dom] > 0:
                                self.stats[f"reject_dominant_{_dom}"] += 1
                                if _dur_b is not None:
                                    self.stats[f"reject_dominant_{_dom}_{_dur_b}"] += 1
                            else:
                                self.stats["reject_dominant_high_reservation"] += 1
                                if _dur_b is not None:
                                    self.stats[f"reject_dominant_high_reservation_{_dur_b}"] += 1
                            self.stats["utility_diff_deterministic_sum"] += utility_diff
                            self.stats["utility_diff_deterministic_n"] += 1
                        else:
                            self.stats["reject_dominant_random"] += 1
                            _dur_b = self._dur_bin_by_widx.get(int(w_idx))
                            if _dur_b is not None:
                                self.stats[f"reject_dominant_random_{_dur_b}"] += 1
                            if utility_diff < 0.2:
                                self.stats["utility_diff_random_band_0"] += 1
                            elif utility_diff < 0.5:
                                self.stats["utility_diff_random_band_1"] += 1
                            elif utility_diff < 1.0:
                                self.stats["utility_diff_random_band_2"] += 1
                            elif utility_diff < 2.0:
                                self.stats["utility_diff_random_band_3"] += 1
                            else:
                                self.stats["utility_diff_random_band_4"] += 1
                            self.stats["utility_diff_random_sum"] += utility_diff
                            self.stats["utility_diff_random_n"] += 1
                self.stats["offers"] += 1
                if w_idx in self._unemployed_idx_set:
                    offers_to_unemployed += 1
                else:
                    offers_to_employed += 1
            for w_idx, offers in worker_offers_this_round.items():
                if w_idx in matched_workers or w_idx in rf_outside_workers:
                    continue
                best_offer = max(offers, key=lambda x: (x[1], x[2]))
                v_idx, quality, posted_wage = best_offer
                if v_idx in filled_vacancies:
                    self.stats["match_fail_vacancy_filled"] += 1
                    self._person_trace_event(worker_ids[w_idx], "match_fail_vacancy_filled")
                    _dur_vf = self._dur_bin_by_widx.get(int(w_idx))
                    if _dur_vf is not None:
                        self.stats[f"match_fail_vacancy_filled_{_dur_vf}"] += 1
                    continue
                firm_id = vacancy_to_firm.get(vacancy_ids[v_idx], "")
                if firm_id:
                    max_hires = firm_vacancy_count.get(firm_id, 1)
                    current_hires = firm_hires_this_step.get(firm_id, 0)
                    if current_hires >= max_hires:
                        self.stats["match_fail_firm_cap"] += 1
                        self._person_trace_event(worker_ids[w_idx], "match_fail_firm_cap")
                        _dur_fc = self._dur_bin_by_widx.get(int(w_idx))
                        if _dur_fc is not None:
                            self.stats[f"match_fail_firm_cap_{_dur_fc}"] += 1
                        continue
                commute_value = None
                if self._sparse_distance_matrix is not None:
                    commute_value = float(self._sparse_distance_matrix[w_idx, v_idx].item())
                w_skill = int(worker_tensors["skill"][w_idx].item())
                w_edu = int(worker_tensors["education"][w_idx].item())
                w_unemp_months = int(worker_tensors["unemployment_months"][w_idx].item())
                w_current_wage = float(worker_tensors["wage"][w_idx].item())
                w_previous_wage = (
                    float(worker_tensors["previous_wage"][w_idx].item())
                    if "previous_wage" in worker_tensors
                    else w_current_wage
                )
                w_res_wage_action = int(worker_tensors["res_wage_action"][w_idx].item())
                v_industry = int(vacancy_tensors["industry"][v_idx].item())
                worker_power = self.compute_bargaining_power(
                    skill_level=w_skill, education=w_edu, unemployment_months=w_unemp_months
                )
                _sep_int = (
                    int(worker_tensors["separation_reason"][w_idx].item())
                    if "separation_reason" in worker_tensors
                    else 0
                )
                _ret_draw = (
                    float(worker_tensors["lwf_retention_draw"][w_idx].item())
                    if "lwf_retention_draw" in worker_tensors
                    else 0.0
                )
                reservation_wage = self.compute_reservation_wage(
                    current_wage=w_current_wage,
                    skill_level=w_skill,
                    unemployment_months=w_unemp_months,
                    res_wage_action=w_res_wage_action,
                    previous_wage=w_previous_wage,
                    separation_reason=_DEC_SEP_REASON.get(_sep_int, "unknown"),
                    lwf_retention_draw=_ret_draw if _ret_draw > 0 else None,
                )
                negotiated_wage = self.negotiate_wage_nash(
                    posted_wage=posted_wage,
                    worker_power=worker_power,
                    reservation_wage=reservation_wage,
                    worker_skill_level=w_skill,
                    v_industry=v_industry,
                    v_firm_id=firm_id,
                )
                if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
                    self, "_person_trace_active", False
                ):
                    self._person_trace_event(worker_ids[w_idx], "bargaining_attempts")
                    _trace_bargain = {
                        "round_index": int(round_k),
                        "vacancy_id": str(vacancy_ids[v_idx]),
                        "firm_id": str(firm_id),
                        "reservation_wage": float(reservation_wage),
                        "posted_wage": float(posted_wage),
                        "previous_wage": float(w_previous_wage),
                        "current_wage": float(w_current_wage),
                        "worker_skill_level": int(w_skill),
                        "unemployment_months": int(w_unemp_months),
                        "reservation_action": int(w_res_wage_action),
                        "separation_reason": _DEC_SEP_REASON.get(_sep_int, "unknown"),
                        "commute_value": commute_value,
                        "commute_units": "minutes" if self._use_travel_time else "km",
                        "commute_minutes": commute_value if self._use_travel_time else None,
                        "negotiated_wage": (
                            None if negotiated_wage is None else float(negotiated_wage)
                        ),
                        "failed": negotiated_wage is None,
                        **self._person_trace_last_bargain_caps,
                    }
                    self._person_trace_detail(
                        worker_ids[w_idx], "bargaining_attempt_details", _trace_bargain
                    )
                    if negotiated_wage is None:
                        self._person_trace_detail(
                            worker_ids[w_idx], "bargaining_failures", _trace_bargain
                        )
                    else:
                        self._person_trace_event(worker_ids[w_idx], "bargaining_successes")
                if negotiated_wage is None:
                    self.stats["match_fail_bargaining"] += 1
                    self._audit_worker_event(worker_ids[w_idx], "bargaining_failures")
                    _dur_bg = self._dur_bin_by_widx.get(int(w_idx))
                    if _dur_bg is not None:
                        self.stats[f"match_fail_bargaining_{_dur_bg}"] += 1
                    _fem = getattr(self, "_last_bf_firm_effective_max", None)
                    if _fem is not None and _fem > 1e-06:
                        _ratio = float(reservation_wage) / float(_fem)
                        if _ratio <= 1.1:
                            _band = "close_easy"
                        elif _ratio <= 1.25:
                            _band = "close_mid"
                        elif _ratio <= 1.5:
                            _band = "close_hard"
                        else:
                            _band = "close_no"
                        self.stats[f"bf_ratio_band_{_band}"] += 1
                        if _dur_bg is not None:
                            self.stats[f"bf_ratio_band_{_band}_{_dur_bg}"] += 1
                            self.stats[f"bf_ratio_sum_{_dur_bg}"] += _ratio
                            self.stats[f"bf_ratio_n_{_dur_bg}"] += 1
                    _cap_b = getattr(self, "_last_bf_binding_cap", None)
                    if _cap_b is not None and _dur_bg is not None:
                        self.stats[f"bf_cap_{_cap_b}_{_dur_bg}"] += 1
                    if _fem is not None and _dur_bg is not None:
                        self.stats[f"bf_res_sum_{_dur_bg}"] += float(reservation_wage)
                        self.stats[f"bf_fem_sum_{_dur_bg}"] += float(_fem)
                        self.stats[f"bf_posted_sum_{_dur_bg}"] += float(posted_wage)
                        self.stats[f"bf_prevwage_sum_{_dur_bg}"] += float(w_previous_wage)
                    _sex_v = (
                        int(worker_tensors["sex"][w_idx].item()) if "sex" in worker_tensors else 0
                    )
                    _ten_v = (
                        int(worker_tensors["previous_tenure_months"][w_idx].item())
                        if "previous_tenure_months" in worker_tensors
                        else 0
                    )
                    _high = _ten_v >= 60
                    _sex_lbl = "F" if _sex_v == 2 else "M"
                    _ten_lbl = "high" if _high else "low"
                    _tier_lbl = f"{_ten_lbl}_{_sex_lbl}"
                    self.stats[f"bf_tier_{_tier_lbl}"] += 1
                    if _dur_bg is not None:
                        self.stats[f"bf_tier_{_tier_lbl}_{_dur_bg}"] += 1
                    if _fem is not None and _fem > 1e-06:
                        self.stats[f"bf_tier_{_tier_lbl}_{_band}"] += 1
                        if _dur_bg is not None:
                            self.stats[f"bf_tier_{_tier_lbl}_{_band}_{_dur_bg}"] += 1
                    if commute_value is not None and _dur_bg is not None:
                        self.stats[f"bf_commute_sum_{_dur_bg}"] += float(commute_value)
                        self.stats[f"bf_commute_n_{_dur_bg}"] += 1
                    logger.isEnabledFor(logging.DEBUG)
                    continue
                n_offers = len(offers)
                is_employed = w_unemp_months == 0 and w_current_wage > 5.0
                wage_change = negotiated_wage - w_current_wage
                if not hasattr(self, "_match_diag_accum"):
                    self._match_diag_accum = {
                        "n": 0,
                        "posted_gt": 0,
                        "sum_ratio": 0.0,
                        "by_offers": {},
                        "j2j": {"n": 0, "sum_ratio": 0.0, "gt": 0},
                        "u2e": {"n": 0, "sum_ratio": 0.0, "gt": 0},
                        "wage_changes": [],
                        "u_prev_wages": [],
                    }
                self._match_diag_accum["n"] += 1
                self._match_diag_accum["wage_changes"].append(wage_change)
                if not is_employed:
                    self._match_diag_accum["u_prev_wages"].append(w_previous_wage)
                if w_current_wage > 0:
                    ratio = posted_wage / w_current_wage
                    self._match_diag_accum["sum_ratio"] += ratio
                    if posted_wage > w_current_wage:
                        self._match_diag_accum["posted_gt"] += 1
                    bucket = self._match_diag_accum["by_offers"]
                    if n_offers not in bucket:
                        bucket[n_offers] = {"n": 0, "sum_ratio": 0.0}
                    bucket[n_offers]["n"] += 1
                    bucket[n_offers]["sum_ratio"] += ratio
                    key = "j2j" if is_employed else "u2e"
                    self._match_diag_accum[key]["n"] += 1
                    self._match_diag_accum[key]["sum_ratio"] += ratio
                    if posted_wage > w_current_wage:
                        self._match_diag_accum[key]["gt"] += 1
                if self._match_diag_accum["n"] % 500 == 0:
                    d = self._match_diag_accum
                    if len(d["wage_changes"]) > 2000:
                        d["wage_changes"] = d["wage_changes"][-1000:]
                    if len(d["u_prev_wages"]) > 1000:
                        d["u_prev_wages"] = d["u_prev_wages"][-500:]
                match_record = {
                    "worker_id": worker_ids[w_idx],
                    "vacancy_id": vacancy_ids[v_idx],
                    "firm_id": vacancy_to_firm.get(vacancy_ids[v_idx], ""),
                    "offered_wage": posted_wage,
                    "negotiated_wage": negotiated_wage,
                    "worker_power": worker_power,
                    "match_quality": quality,
                    "reservation_wage": reservation_wage,
                }
                if self._use_travel_time:
                    match_record["commute_time_min"] = commute_value
                else:
                    match_record["commute_distance_km"] = commute_value
                matches.append(match_record)
                self._audit_worker_event(worker_ids[w_idx], "matches")
                if negotiated_wage < reservation_wage - 1e-06:
                    self.stats["bargaining_invariant_violations"] = (
                        self.stats.get("bargaining_invariant_violations", 0) + 1
                    )
                matched_workers.add(w_idx)
                filled_vacancies.add(v_idx)
                self.stats["matches"] += 1
                _mw_ind = int(worker_tensors["industry"][w_idx].item())
                _mv_ind = int(vacancy_tensors["industry"][v_idx].item())
                if _mw_ind == _mv_ind:
                    self.stats["match_same_industry"] += 1
                else:
                    _mw_sec = int(worker_tensors["sector"][w_idx].item())
                    _mv_sec = int(vacancy_tensors["sector"][v_idx].item())
                    if _mw_sec == _mv_sec:
                        self.stats["match_same_sector_diff_industry"] += 1
                    else:
                        self.stats["match_diff_sector"] += 1
                _dur_m = self._dur_bin_by_widx.get(int(w_idx))
                if _dur_m is not None:
                    self.stats[f"match_{_dur_m}"] += 1
                match_firm_id = match_record.get("firm_id", "")
                if match_firm_id:
                    firm_hires_this_step[match_firm_id] = (
                        firm_hires_this_step.get(match_firm_id, 0) + 1
                    )
        if all_accept_probs:
            self.stats["avg_accept_prob"] = sum(all_accept_probs) / len(all_accept_probs)
            self.stats["avg_mnl_accept_prob"] = self.stats["avg_accept_prob"]
        else:
            self.stats["avg_accept_prob"] = 0.0
            self.stats["avg_mnl_accept_prob"] = 0.0
        if all_attachment_adjusted_probs:
            self.stats["avg_attachment_adjusted_match_prob"] = sum(
                all_attachment_adjusted_probs
            ) / len(all_attachment_adjusted_probs)
        else:
            self.stats["avg_attachment_adjusted_match_prob"] = self.stats["avg_mnl_accept_prob"]
        self.stats["attachment_conversion_failures"] = int(attachment_conversion_failures_total)
        self.stats["n_offers_created"] = self.stats["offers"]
        self.stats["n_offers_sent"] = self.stats["offers"]
        try:
            self._episode_ever_offered.update((worker_ids[i] for i in workers_offered))
            self._episode_ever_matched.update((worker_ids[i] for i in matched_workers))
        except (IndexError, TypeError, KeyError) as _e:
            self._episode_acceptance_invalid_reason = f"{type(_e).__name__}: {_e} (n_workers_offered={len(workers_offered)}, n_matched={len(matched_workers)}, worker_ids_len={(len(worker_ids) if hasattr(worker_ids, '__len__') else '?')})"
            logger.error(
                f"Could not update the matching counts by tier: {self._episode_acceptance_invalid_reason}"
            )
        n_seekers = len(worker_ids)
        n_offered = len(workers_offered)
        n_matched_set = len(matched_workers)
        self.stats["n_job_seekers"] = n_seekers
        self.stats["n_workers_offered"] = n_offered
        self.stats["n_workers_no_offer"] = n_seekers - n_offered
        self.stats["n_workers_rejected_all"] = n_offered - n_matched_set
        self.stats["offers_per_job_seeker"] = (
            self.stats["offers"] / n_seekers if n_seekers > 0 else 0.0
        )
        self.stats["share_offered"] = n_offered / n_seekers if n_seekers > 0 else 0.0
        self.stats["accept_rate_given_offer"] = n_matched_set / n_offered if n_offered > 0 else 0.0
        for _wi in workers_offered:
            _b = self._dur_bin_by_widx.get(int(_wi))
            if _b is not None:
                self.stats[f"n_workers_offered_{_b}"] += 1
                if int(_wi) in matched_workers:
                    self.stats[f"n_workers_matched_{_b}"] += 1
        for _bin in ("dur_0_6", "dur_6_12", "dur_12_inf"):
            _n_seek = self.stats[f"n_unemployed_seekers_{_bin}"]
            _n_off = self.stats[f"n_workers_offered_{_bin}"]
            _n_match = self.stats[f"n_workers_matched_{_bin}"]
            self.stats[f"n_workers_no_offer_{_bin}"] = _n_seek - _n_off
            self.stats[f"n_workers_rejected_all_{_bin}"] = _n_off - _n_match
        unemp_offered = {w for w in workers_offered if w in self._unemployed_idx_set}
        unemp_matched = {w for w in matched_workers if w in self._unemployed_idx_set}
        n_unemp = self._n_unemployed_seekers
        n_unemp_offered = len(unemp_offered)
        n_unemp_matched = len(unemp_matched)
        self.stats["n_unemployed_seekers"] = n_unemp
        self.stats["n_unemployed_offered"] = n_unemp_offered
        self.stats["n_unemployed_matched"] = n_unemp_matched
        self.stats["offers_to_unemployed"] = offers_to_unemployed
        self.stats["offers_to_employed"] = offers_to_employed
        self.stats["offers_per_unemployed_seeker"] = (
            offers_to_unemployed / n_unemp if n_unemp > 0 else 0.0
        )
        self.stats["share_unemployed_offered"] = n_unemp_offered / n_unemp if n_unemp > 0 else 0.0
        self.stats["matches_per_unemployed_seeker"] = (
            n_unemp_matched / n_unemp if n_unemp > 0 else 0.0
        )
        self.stats["accept_rate_given_offer_unemployed"] = (
            n_unemp_matched / n_unemp_offered if n_unemp_offered > 0 else 0.0
        )
        n_emp = self._n_employed_seekers
        emp_offered = {w for w in workers_offered if w not in self._unemployed_idx_set}
        emp_matched = {w for w in matched_workers if w not in self._unemployed_idx_set}
        self.stats["n_employed_seekers"] = n_emp
        self.stats["n_employed_offered"] = len(emp_offered)
        self.stats["n_employed_matched"] = len(emp_matched)
        self.stats["offers_per_employed_seeker"] = offers_to_employed / n_emp if n_emp > 0 else 0.0
        total_offers = self.stats.get("offers", 0)
        if total_offers == 0 and applications:
            diag_list = []
            for vid, applicant_wids in applications.items():
                if not applicant_wids:
                    continue
                if vid not in vacancy_id_to_idx:
                    continue
                v_idx = vacancy_id_to_idx[vid]
                threshold = float(hiring_threshold_tensor[v_idx].item())
                posted_wage = float(
                    vacancy_tensors["wage"][v_idx].item() * wage_multipliers[v_idx].item()
                )
                for wid in applicant_wids:
                    if wid not in worker_id_to_idx:
                        continue
                    w_idx = worker_id_to_idx[wid]
                    quality = float(match_quality_matrix[w_idx, v_idx].item())
                    w_skill = int(worker_tensors["skill"][w_idx].item())
                    w_wage = float(worker_tensors["wage"][w_idx].item())
                    w_prev_wage = (
                        float(worker_tensors["previous_wage"][w_idx].item())
                        if "previous_wage" in worker_tensors
                        else w_wage
                    )
                    w_unemp = int(worker_tensors["unemployment_months"][w_idx].item())
                    w_res_wage_action = (
                        int(worker_tensors["res_wage_action"][w_idx].item())
                        if "res_wage_action" in worker_tensors
                        else 2
                    )
                    _sep_int = (
                        int(worker_tensors["separation_reason"][w_idx].item())
                        if "separation_reason" in worker_tensors
                        else 0
                    )
                    _ret_draw = (
                        float(worker_tensors["lwf_retention_draw"][w_idx].item())
                        if "lwf_retention_draw" in worker_tensors
                        else 0.0
                    )
                    res_wage = self.compute_reservation_wage(
                        current_wage=w_wage,
                        skill_level=w_skill,
                        unemployment_months=w_unemp,
                        res_wage_action=w_res_wage_action,
                        previous_wage=w_prev_wage,
                        separation_reason=_DEC_SEP_REASON.get(_sep_int, "unknown"),
                        lwf_retention_draw=_ret_draw if _ret_draw > 0 else None,
                    )
                    commute = None
                    if distance_matrix is not None:
                        commute = float(distance_matrix[w_idx, v_idx].item())
                    if quality < threshold:
                        reason = f"failed_threshold (quality {quality:.3f} < {threshold:.3f})"
                    else:
                        reason = f"passed_screen (quality {quality:.3f}, wage_ratio={posted_wage / (res_wage + 1e-06):.2f})"
                    diag_list.append(
                        {
                            "vacancy_id": vid,
                            "worker_id": wid,
                            "match_quality": quality,
                            "threshold": threshold,
                            "posted_wage": posted_wage,
                            "reservation_wage": res_wage,
                            "commute": commute,
                            "reason": reason,
                        }
                    )
            self.stats["zero_offers_diagnostic"] = diag_list
        return matches

    def process_applications_gpu(
        self,
        worker_tensors: Dict[str, torch.Tensor],
        vacancy_tensors: Dict[str, torch.Tensor],
        worker_ids: List[str],
        vacancy_ids: List[str],
        worker_actions: Dict[str, Dict[str, Any]],
        job_seekers: Optional[List[Dict[str, Any]]] = None,
        vacancies_list: Optional[List[Dict[str, Any]]] = None,
        employed_min_wages: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[str]]:
        n_workers = len(worker_ids)
        n_vacancies = len(vacancy_ids)
        if n_workers == 0 or n_vacancies == 0:
            return {vid: [] for vid in vacancy_ids}
        channel_map = {
            0: "Ads/Online",
            1: "Direct Contact",
            2: "Networking",
            3: "Public Agency",
            4: "Private Agency",
        }
        workers_by_channel = {ch: [] for ch in channel_map.values()}
        worker_channel = {}
        for i, wid in enumerate(worker_ids):
            action = worker_actions.get(wid, {})
            channel_int = action.get("search_channel", 0)
            channel_name = channel_map.get(channel_int, "Ads/Online")
            workers_by_channel[channel_name].append(i)
            worker_channel[wid] = channel_name
        worker_da_ids = None
        vacancy_da_ids = None
        if job_seekers is not None and vacancies_list is not None:
            worker_da_ids = [w.get("da_id") for w in job_seekers]
            vacancy_da_ids = [v.get("da_id") for v in vacancies_list]
        base_relevance = self.compute_relevance_matrix_gpu(
            worker_tensors,
            vacancy_tensors,
            worker_da_ids=worker_da_ids,
            vacancy_da_ids=vacancy_da_ids,
        )
        channel_relevance = {}
        for channel_name in channel_map.values():
            if len(workers_by_channel[channel_name]) > 0:
                channel_relevance[channel_name] = self.apply_channel_modifiers_gpu(
                    base_relevance, worker_tensors, vacancy_tensors, channel_name
                )
        if employed_min_wages is None:
            employed_min_wages = {}
        vacancy_wages = None
        if employed_min_wages and vacancies_list is not None:
            vacancy_wages = torch.tensor(
                [v.get("offered_wage", 0.0) for v in vacancies_list],
                dtype=torch.float32,
                device=self.device,
            )
        applications = {vid: [] for vid in vacancy_ids}
        for i, wid in enumerate(worker_ids):
            action = worker_actions.get(wid, {})
            search_intensity = action.get("search_intensity", 2)
            n_apps = self.config.search_intensity_to_apps.get(search_intensity, 0)
            if n_apps == 0:
                continue
            br_skill = action.get("bounded_rationality_skill", 0.5)
            n_apps = max(1, int(n_apps * br_skill))
            channel_name = worker_channel[wid]
            worker_relevance = channel_relevance[channel_name][i]
            sharpening_power = 1.0 + 0.1 * search_intensity
            worker_relevance = torch.pow(worker_relevance, sharpening_power)
            probs = torch.clamp(worker_relevance, min=0.01)
            probs = probs / probs.sum()
            n_sample = min(n_apps, n_vacancies)
            selected_v_indices = torch.multinomial(probs, n_sample, replacement=False)
            if wid in employed_min_wages and vacancy_wages is not None:
                min_wage = employed_min_wages[wid]
                v_wages = vacancy_wages[selected_v_indices]
                wage_ok = v_wages >= min_wage
                selected_v_indices = selected_v_indices[wage_ok]
                if len(selected_v_indices) == 0:
                    continue
            for v_idx in selected_v_indices.tolist():
                applications[vacancy_ids[v_idx]].append(wid)
                self.stats["applications"] += 1
            if len(selected_v_indices) > 0:
                _w_ind = int(worker_tensors["industry"][i].item())
                _w_sec = int(worker_tensors["sector"][i].item())
                _v_inds = vacancy_tensors["industry"][selected_v_indices]
                _v_secs = vacancy_tensors["sector"][selected_v_indices]
                _same_ind = _v_inds == _w_ind
                _same_sec = _v_secs == _w_sec
                _n_same_ind = int(_same_ind.sum().item())
                _n_same_sec_diff_ind = int((~_same_ind & _same_sec).sum().item())
                _n_diff_sec = int((~_same_ind & ~_same_sec).sum().item())
                self.stats["app_same_industry"] += _n_same_ind
                self.stats["app_same_sector_diff_industry"] += _n_same_sec_diff_ind
                self.stats["app_diff_sector"] += _n_diff_sec
                _dur = float(worker_tensors["unemployment_months"][i].item())
                if _dur < 6:
                    _bin = "dur_0_6"
                elif _dur < 12:
                    _bin = "dur_6_12"
                else:
                    _bin = "dur_12_inf"
                self.stats[f"app_same_industry_{_bin}"] += _n_same_ind
                self.stats[f"app_same_sector_diff_industry_{_bin}"] += _n_same_sec_diff_ind
                self.stats[f"app_diff_sector_{_bin}"] += _n_diff_sec
        return applications

    def run_matching(
        self,
        workers: Dict[str, Any],
        vacancies: Dict[str, Any],
        worker_actions: Dict[str, Dict[str, Any]],
        firm_actions: Optional[Dict[str, Dict[str, Any]]] = None,
        firms: Optional[Dict[str, Any]] = None,
        employed_min_wages: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self._employed_min_wages = employed_min_wages or {}
        self.reset_stats()
        self._matching_step += 1
        if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
            self, "_person_trace_active", False
        ):
            if not hasattr(self, "_person_month_trace"):
                self.begin_person_month_trace()
            self._person_month_trace["matcher_called"] = True
            self._person_month_trace["no_vacancies"] = not bool(vacancies)
            for _trace_wid, _trace_ws in workers.items():
                self._person_trace_register(_trace_wid, _trace_ws, in_pool=False)
        if not vacancies:
            rf = None
            if rf is not None:
                rf.begin([], [], {}, self.config.n_matching_rounds)
            return ([], self.stats)

        def worker_to_dict(ws) -> Dict[str, Any]:
            w = ws.worker if hasattr(ws, "worker") else ws
            v0 = getattr(ws, "learned_v0", 6.5)
            res_wage_action = getattr(ws, "last_res_wage_action", 2)
            current_wage = float(getattr(w, "current_wage", 0.0) or 0.0)
            previous_wage = float(getattr(ws, "previous_wage", current_wage) or current_wage)
            if hasattr(w, "__dict__"):
                return {
                    "worker_id": w.id,
                    "age": w.age,
                    "skill_level": w.skill_level,
                    "education": w.education,
                    "naics_18": w.industry,
                    "sector": w.sector,
                    "experience_total": w.experience_years,
                    "cma": w.cma,
                    "employment_status": w.employment_status,
                    "current_wage": current_wage,
                    "is_searching": w.is_searching,
                    "unemployment_months": getattr(w, "unemployment_months", 0),
                    "da_id": getattr(w, "da_id", None),
                    "learned_v0": float(v0),
                    "res_wage_action": int(res_wage_action),
                    "previous_wage": previous_wage,
                    "attachment": float(getattr(w, "attachment", 1.0)),
                    "sex": int(getattr(w, "sex", 0)),
                    "previous_tenure_months": int(getattr(ws, "previous_tenure_months", 0)),
                    "separation_reason": _ENC_SEP_REASON.get(
                        getattr(ws, "separation_reason", "unknown"), 0
                    ),
                    "lwf_retention_draw": float(getattr(ws, "lwf_retention_draw", None) or 0.0),
                    "archetype_id": int(getattr(w, "archetype_id", 0) or 0),
                }
            return w

        def vacancy_to_dict(v) -> Dict[str, Any]:
            if hasattr(v, "__dict__"):
                return {
                    "vacancy_id": v.id,
                    "firm_id": v.firm_id,
                    "naics_18": v.industry,
                    "sector": v.sector,
                    "skill_required": v.skill_required,
                    "education_required": v.education_required,
                    "offered_wage": v.offered_wage,
                    "cma": v.cma,
                    "job_security": getattr(v, "job_security", 0.5),
                    "da_id": getattr(v, "da_id", None),
                }
            return v

        workers_list = [worker_to_dict(w) for w in workers.values()]
        vacancies_list = [vacancy_to_dict(v) for v in vacancies.values()]

        def _age_bucket(a):
            if 15 <= a < 25:
                return "15-24"
            if 25 <= a < 35:
                return "25-34"
            if 35 <= a < 45:
                return "35-44"
            if 45 <= a < 55:
                return "45-54"
            if 55 <= a < 65:
                return "55-64"
            return "65+"

        new_sep_pool_by_age = {
            ab: [] for ab in ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        }
        existing_unemp_pool_by_age = {
            ab: [] for ab in ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        }
        n_newly_unemployed = 0
        n_forced_search_new_sep = 0
        for w in workers_list:
            if w.get("employment_status") != "Unemployed":
                continue
            wid = w.get("worker_id")
            age = int(w.get("age", 0))
            ab = _age_bucket(age)
            unemp_m = w.get("unemployment_months", 0)
            if unemp_m == 0:
                n_newly_unemployed += 1
                if not w.get("is_searching", False):
                    n_forced_search_new_sep += 1
                w["is_searching"] = True
                new_sep_pool_by_age[ab].append(wid)
            elif w.get("is_searching", False):
                existing_unemp_pool_by_age[ab].append(wid)
        self._last_new_sep_pool_by_age = new_sep_pool_by_age
        self._last_existing_unemp_pool_by_age = existing_unemp_pool_by_age
        self._last_newly_unemployed_count = n_newly_unemployed
        self._last_forced_search_count = n_forced_search_new_sep
        job_seekers = [
            w
            for w in workers_list
            if w.get("is_searching", False)
            and w.get("employment_status") in ("Unemployed", "Employed")
        ]
        if os.environ.get("WORKER_LEVEL_TRACE", "0") == "1" and getattr(
            self, "_person_trace_active", False
        ):
            self._person_month_trace["pool_evaluated"] = True
            self._person_month_trace["no_job_seekers"] = not bool(job_seekers)
            for _trace_worker in job_seekers:
                self._person_trace_register(_trace_worker["worker_id"], _trace_worker, in_pool=True)
        n_employed_seekers = sum(
            (1 for w in job_seekers if w.get("employment_status") == "Employed")
        )
        n_unemployed_seekers = len(job_seekers) - n_employed_seekers
        self._n_unemployed_seekers = n_unemployed_seekers
        self._n_employed_seekers = n_employed_seekers
        self._unemployed_idx_set = {
            i for i, w in enumerate(job_seekers) if w.get("employment_status") == "Unemployed"
        }
        self._dur_bin_by_widx: Dict[int, str] = {}
        self._da_by_widx: Dict[int, Optional[int]] = {}
        for _i, _w in enumerate(job_seekers):
            _da = _w.get("da_id")
            try:
                self._da_by_widx[_i] = int(_da) if _da is not None else None
            except (TypeError, ValueError):
                self._da_by_widx[_i] = None
            if _w.get("employment_status") != "Unemployed":
                continue
            _dur = int(_w.get("unemployment_months", 0) or 0)
            if _dur < 6:
                _b = "dur_0_6"
            elif _dur < 12:
                _b = "dur_6_12"
            else:
                _b = "dur_12_inf"
            self._dur_bin_by_widx[_i] = _b
            self.stats[f"n_unemployed_seekers_{_b}"] += 1
        self._executed_unemployed_searcher_ids = [
            w["worker_id"] for w in job_seekers if w.get("employment_status") == "Unemployed"
        ]
        if not job_seekers:
            rf = None
            if rf is not None:
                rf.begin(
                    [],
                    [v["vacancy_id"] for v in vacancies_list],
                    {v["vacancy_id"]: v["firm_id"] for v in vacancies_list},
                    self.config.n_matching_rounds,
                )
            return ([], self.stats)
        worker_ids = [w["worker_id"] for w in job_seekers]
        vacancy_ids = [v["vacancy_id"] for v in vacancies_list]
        worker_tensors = self._build_worker_tensors(job_seekers)
        vacancy_tensors = self._build_vacancy_tensors(vacancies_list)
        vacancy_to_firm = {v["vacancy_id"]: v.get("firm_id", "") for v in vacancies_list}
        wage_strategies = {}
        if firm_actions:
            for firm_id, action in firm_actions.items():
                wage_strategies[firm_id] = action.get("wage_strategy", 2)
        hiring_thresholds = {}
        if firm_actions:
            firm_hiring_decisions = {}
            for firm_id, action in firm_actions.items():
                firm_hiring_decisions[firm_id] = action.get("hiring_decision", 2)
            for v in vacancies_list:
                vid = v["vacancy_id"]
                firm_id = v.get("firm_id", "")
                decision = firm_hiring_decisions.get(firm_id, 2)
                hiring_thresholds[vid] = self.config.hiring_selectivity_thresholds.get(
                    decision, 0.3
                )
        applications = self.process_applications_gpu(
            worker_tensors,
            vacancy_tensors,
            worker_ids,
            vacancy_ids,
            worker_actions,
            job_seekers=job_seekers,
            vacancies_list=vacancies_list,
            employed_min_wages=self._employed_min_wages,
        )
        matches = self.run_k_round_matching_gpu(
            worker_tensors,
            vacancy_tensors,
            worker_ids,
            vacancy_ids,
            vacancy_to_firm,
            applications,
            wage_strategies,
            job_seekers,
            vacancies_list,
            hiring_thresholds=hiring_thresholds,
        )
        self.stats["job_finding_rate"] = len(matches) / len(job_seekers) if job_seekers else 0
        self.stats["vacancy_fill_rate"] = (
            len(matches) / len(vacancies_list) if vacancies_list else 0
        )
        self.stats["acceptance_rate"] = (
            self.stats["matches"] / self.stats["offers"] if self.stats["offers"] > 0 else 0
        )
        matched_ids = {m["worker_id"] for m in matches}
        unmatched_workers = [w for w in job_seekers if w["worker_id"] not in matched_ids]
        if unmatched_workers:
            apps_per_worker = {}
            for vid, applicants in applications.items():
                for wid in applicants:
                    apps_per_worker[wid] = apps_per_worker.get(wid, 0) + 1
        if matches:
            worker_powers = [m.get("worker_power", 0.5) for m in matches]
            offered_wages = [m.get("offered_wage", 0) for m in matches]
            negotiated_wages = [m.get("negotiated_wage", 0) for m in matches]
            if self._use_travel_time:
                commute_values = [
                    m.get("commute_time_min")
                    for m in matches
                    if m.get("commute_time_min") is not None
                ]
                commute_unit = "min"
            else:
                commute_values = [
                    m.get("commute_distance_km")
                    for m in matches
                    if m.get("commute_distance_km") is not None
                ]
                commute_unit = "km"
            self.stats["nash_bargaining"] = {
                "n_negotiations": len(matches),
                "avg_worker_power": float(np.mean(worker_powers)),
                "min_worker_power": float(np.min(worker_powers)),
                "max_worker_power": float(np.max(worker_powers)),
                "std_worker_power": float(np.std(worker_powers)) if len(worker_powers) > 1 else 0.0,
                "avg_offered_wage": float(np.mean(offered_wages)),
                "avg_negotiated_wage": float(np.mean(negotiated_wages)),
                "avg_wage_premium": float(np.mean(negotiated_wages) - np.mean(offered_wages)),
                "avg_wage_premium_pct": (
                    float((np.mean(negotiated_wages) / np.mean(offered_wages) - 1) * 100)
                    if np.mean(offered_wages) > 0
                    else 0.0
                ),
                "worker_power_sample": worker_powers[:100],
                "wage_premium_sample": [
                    n - o for o, n in zip(offered_wages[:100], negotiated_wages[:100])
                ],
            }
            self.stats["commute_values"] = commute_values
            self.stats["commute_unit"] = commute_unit
            self.stats["use_travel_time"] = self._use_travel_time
        else:
            self.stats["nash_bargaining"] = {
                "n_negotiations": 0,
                "avg_worker_power": 0.0,
                "min_worker_power": 0.0,
                "max_worker_power": 0.0,
                "std_worker_power": 0.0,
                "avg_offered_wage": 0.0,
                "avg_negotiated_wage": 0.0,
                "avg_wage_premium": 0.0,
                "avg_wage_premium_pct": 0.0,
                "worker_power_sample": [],
                "wage_premium_sample": [],
            }
            self.stats["commute_distances"] = []
        for reason, count in self.rejection_reasons.items():
            self.stats[f"accept_reject_{reason}"] = int(count)
        self.stats["rejection_reasons_by_da"] = {
            int(_da): dict(_reasons) for _da, _reasons in self.rejection_reasons_by_da.items()
        }
        return (matches, self.stats)
