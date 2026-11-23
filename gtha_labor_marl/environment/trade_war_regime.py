from __future__ import annotations
import csv
import hashlib
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)
DRIVER_COLUMNS = {
    "scenario",
    "year",
    "month",
    "product_group",
    "us_tariff_rate",
    "canada_countertariff_rate",
    "uncertainty_index",
}
EXPOSURE_COLUMNS = {
    "priority",
    "scope",
    "scenario",
    "code",
    "label",
    "product_group",
    "export_share_us",
    "imported_input_share",
    "export_demand_elasticity",
    "tariff_pass_through",
    "productivity_drag_full",
    "employment_elasticity",
    "imported_input_share_basis",
    "current_outlays_over_revenue",
}
INPUT_SHARE_BASIS = "US_inputs_over_total_cost_before_tariff_coverage"


def input_basket_key(scope: str, code: str) -> str:
    return f"input::{scope}:{code}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(row: Mapping[str, str], name: str, *, lo: float, hi: float) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}={row.get(name)!r}") from exc
    if not math.isfinite(value) or value < lo or value > hi:
        raise ValueError(f"{name} must be finite and in [{lo}, {hi}], got {value}")
    return value


def normalize_naics6(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join((ch for ch in text if ch.isdigit()))
    return digits or None


@dataclass(frozen=True)
class DriverCell:
    scenario: str
    year: int
    month: int
    product_group: str
    us_tariff_rate: float
    canada_countertariff_rate: float
    uncertainty_index: float


@dataclass(frozen=True)
class FirmExposure:
    priority: int
    scope: str
    code: str
    label: str
    product_group: str
    export_share_us: float
    imported_input_share: float
    export_demand_elasticity: float
    tariff_pass_through: float
    productivity_drag_full: float
    employment_elasticity: float
    current_outlays_over_revenue: float


@dataclass(frozen=True)
class FirmShock:
    firm_id: str
    industry: int
    da_id: Optional[int]
    exposure_label: str
    product_group: str
    baseline_target: int
    shocked_target: int
    export_demand_loss: float
    imported_input_cost_rate: float
    productivity_loss: float
    demand_multiplier: float
    productivity_multiplier: float
    margin_multiplier: float
    employment_multiplier: float
    input_basket: str
    input_tariff_rate: float
    imported_input_cost_outlay_rate: float


def load_driver(path: Path) -> Dict[Tuple[str, int, int, str], DriverCell]:
    if not path.is_file():
        raise FileNotFoundError(f"trade-war driver not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(DRIVER_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"trade-war driver is missing columns: {missing}")
        cells: Dict[Tuple[str, int, int, str], DriverCell] = {}
        for line_no, row in enumerate(reader, start=2):
            try:
                scenario = row["scenario"].strip()
                product_group = row["product_group"].strip()
                year = int(row["year"])
                month = int(row["month"])
                if not scenario or not product_group:
                    raise ValueError("scenario and product_group must be non-empty")
                if month < 1 or month > 12:
                    raise ValueError(f"month must be in 1..12, got {month}")
                cell = DriverCell(
                    scenario=scenario,
                    year=year,
                    month=month,
                    product_group=product_group,
                    us_tariff_rate=_as_float(row, "us_tariff_rate", lo=0.0, hi=1.0),
                    canada_countertariff_rate=_as_float(
                        row, "canada_countertariff_rate", lo=0.0, hi=1.0
                    ),
                    uncertainty_index=_as_float(row, "uncertainty_index", lo=0.0, hi=1.0),
                )
                if product_group.startswith("input::"):
                    if cell.us_tariff_rate or cell.uncertainty_index:
                        raise ValueError(
                            "input-basket rows cannot carry export or uncertainty shocks"
                        )
                elif cell.canada_countertariff_rate:
                    raise ValueError(
                        "Canadian input tariffs must be in input-basket rows, not output groups"
                    )
                if scenario == "control" and any(
                    (cell.us_tariff_rate, cell.canada_countertariff_rate, cell.uncertainty_index)
                ):
                    raise ValueError("control shocks must all be zero")
            except Exception as exc:
                raise ValueError(f"invalid trade-war driver row {line_no}: {exc}") from exc
            key = (scenario, year, month, product_group)
            if key in cells:
                raise ValueError(f"duplicate trade-war driver key at row {line_no}: {key}")
            cells[key] = cell
    if not cells:
        raise ValueError("trade-war driver is empty")
    return cells


def load_exposures(path: Path) -> Dict[str, List[FirmExposure]]:
    if not path.is_file():
        raise FileNotFoundError(f"trade-war exposure file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(EXPOSURE_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"trade-war exposure file is missing columns: {missing}")
        by_scenario: Dict[str, List[FirmExposure]] = {}
        seen: set[Tuple[str, str, str]] = set()
        for line_no, row in enumerate(reader, start=2):
            try:
                scenario = row["scenario"].strip()
                if not scenario:
                    raise ValueError("scenario must be non-empty")
                scope = row["scope"].strip().lower()
                code = row["code"].strip()
                if scope not in {"naics6_prefix", "naics18", "default"}:
                    raise ValueError(f"unsupported scope {scope!r}")
                if scope == "default":
                    code = "*"
                else:
                    code = normalize_naics6(code)
                    if not code:
                        raise ValueError(f"scope {scope!r} requires a numeric code")
                    if scope == "naics18" and int(code) not in range(1, 19):
                        raise ValueError(f"NAICS-18 code must be in 1..18, got {code}")
                if row["imported_input_share_basis"].strip() != INPUT_SHARE_BASIS:
                    raise ValueError(
                        "imported-input share must explicitly use the Route A outlay basis"
                    )
                outlays_over_revenue = _as_float(
                    row, "current_outlays_over_revenue", lo=0.0, hi=float("inf")
                )
                if outlays_over_revenue <= 0:
                    raise ValueError("current_outlays_over_revenue must be positive")
                exposure = FirmExposure(
                    priority=int(row["priority"]),
                    scope=scope,
                    code=code,
                    label=row["label"].strip(),
                    product_group=row["product_group"].strip(),
                    export_share_us=_as_float(row, "export_share_us", lo=0.0, hi=1.0),
                    imported_input_share=_as_float(row, "imported_input_share", lo=0.0, hi=1.0),
                    export_demand_elasticity=_as_float(
                        row, "export_demand_elasticity", lo=0.0, hi=5.0
                    ),
                    tariff_pass_through=_as_float(row, "tariff_pass_through", lo=0.0, hi=1.0),
                    productivity_drag_full=_as_float(row, "productivity_drag_full", lo=0.0, hi=1.0),
                    employment_elasticity=_as_float(row, "employment_elasticity", lo=0.0, hi=2.0),
                    current_outlays_over_revenue=outlays_over_revenue,
                )
                if not exposure.label or not exposure.product_group:
                    raise ValueError("label and product_group must be non-empty")
            except Exception as exc:
                raise ValueError(f"invalid trade-war exposure row {line_no}: {exc}") from exc
            key = (scenario, scope, code)
            if key in seen:
                raise ValueError(f"duplicate trade-war exposure key at row {line_no}: {key}")
            seen.add(key)
            by_scenario.setdefault(scenario, []).append(exposure)
    if not by_scenario:
        raise ValueError("trade-war exposure file defines no scenario")
    ladders = {
        scenario: frozenset(((item.scope, item.code) for item in rules))
        for scenario, rules in by_scenario.items()
    }
    reference_scenario = sorted(ladders)[0]
    reference = ladders[reference_scenario]
    for scenario in sorted(by_scenario):
        if ladders[scenario] != reference:
            raise ValueError(
                f"trade-war exposure ladder differs between scenarios: {scenario!r} missing={sorted(reference - ladders[scenario])} extra={sorted(ladders[scenario] - reference)} (against {reference_scenario!r}); a rule is never taken from another scenario"
            )
        defaults = [item for item in by_scenario[scenario] if item.scope == "default"]
        if len(defaults) != 1:
            raise ValueError(
                f"trade-war exposure scenario {scenario!r} requires exactly one default row, found {len(defaults)}"
            )
    return {
        scenario: sorted(
            rules,
            key=lambda item: (
                item.priority,
                len(normalize_naics6(item.code) or "") if item.scope == "naics6_prefix" else 0,
            ),
            reverse=True,
        )
        for scenario, rules in by_scenario.items()
    }


def select_exposures(
    by_scenario: Dict[str, List[FirmExposure]], scenario: str
) -> List[FirmExposure]:
    if scenario not in by_scenario:
        raise ValueError(
            f"trade exposure scenario {scenario!r} is absent; available={sorted(by_scenario)}"
        )
    return by_scenario[scenario]


class TradeWarRegime:

    def __init__(self, config: Any):
        self.config = config
        self.enabled = bool(getattr(config, "enabled", False))
        self.scenario = str(getattr(config, "scenario", "control"))
        self.driver_path = Path(str(getattr(config, "driver_file", ""))).expanduser()
        self.exposure_path = Path(str(getattr(config, "exposure_file", ""))).expanduser()
        self.min_target_multiplier = float(getattr(config, "min_target_multiplier", 0.25))
        self.min_margin_multiplier = float(getattr(config, "min_margin_multiplier", 0.1))
        self.max_export_demand_loss = float(getattr(config, "max_export_demand_loss", 0.8))
        self._cells: Dict[Tuple[str, int, int, str], DriverCell] = {}
        self._exposures: List[FirmExposure] = []
        self._exposures_by_scenario: Dict[str, List[FirmExposure]] = {}
        self._baseline_targets: Dict[str, int] = {}
        self._baseline_firm_objects: Dict[str, int] = {}
        self._was_active: Dict[str, bool] = {}
        self._current_shocks: Dict[str, FirmShock] = {}
        self._current_key: Optional[Tuple[int, int]] = None
        self._history: List[Dict[str, Any]] = []
        self._current_record: Optional[Dict[str, Any]] = None
        if self.enabled:
            if not 0.0 < self.min_target_multiplier <= 1.0:
                raise ValueError("trade min_target_multiplier must be in (0, 1]")
            if not 0.0 < self.min_margin_multiplier <= 1.0:
                raise ValueError("trade min_margin_multiplier must be in (0, 1]")
            if not 0.0 <= self.max_export_demand_loss < 1.0:
                raise ValueError("trade max_export_demand_loss must be in [0, 1)")
            self._cells = load_driver(self.driver_path)
            self._exposures_by_scenario = load_exposures(self.exposure_path)
            self._validate_scenario_coverage()

    @property
    def provenance(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "scenario": self.scenario,
            "driver_file": self.driver_path.name,
            "driver_sha256": sha256_file(self.driver_path),
            "exposure_file": self.exposure_path.name,
            "exposure_sha256": sha256_file(self.exposure_path),
            "min_target_multiplier": self.min_target_multiplier,
            "min_margin_multiplier": self.min_margin_multiplier,
            "max_export_demand_loss": self.max_export_demand_loss,
            "policy_observation_changed": False,
            "input_share_basis": INPUT_SHARE_BASIS,
            "input_cost_rate_basis": "additional_input_cost_over_counterfactual_revenue",
            "input_tariff_lookup": "exposure_rule_basket_not_output_product_group",
        }

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def _validate_scenario_coverage(self) -> None:
        driver_scenarios = {cell.scenario for cell in self._cells.values()}
        exposure_scenarios = set(self._exposures_by_scenario)
        if driver_scenarios != exposure_scenarios:
            raise ValueError(
                f"trade driver and exposure scenario sets differ: driver_only={sorted(driver_scenarios - exposure_scenarios)} exposure_only={sorted(exposure_scenarios - driver_scenarios)}"
            )
        scenario_cells = [cell for cell in self._cells.values() if cell.scenario == self.scenario]
        if not scenario_cells:
            available = sorted(driver_scenarios)
            raise ValueError(f"trade scenario {self.scenario!r} is absent; available={available}")
        self._exposures = select_exposures(self._exposures_by_scenario, self.scenario)
        required_groups = {item.product_group for item in self._exposures}
        required_groups.update(
            (input_basket_key(item.scope, item.code) for item in self._exposures)
        )
        if {cell.product_group for cell in scenario_cells} != required_groups:
            raise ValueError(
                "trade driver group set differs from output groups plus exposure-rule input baskets"
            )
        years = sorted({cell.year for cell in scenario_cells})
        for year in range(years[0], years[-1] + 1):
            for month in range(1, 13):
                for group in required_groups:
                    if (self.scenario, year, month, group) not in self._cells:
                        raise ValueError(
                            f"trade driver coverage gap: scenario={self.scenario} year={year} month={month} product_group={group}"
                        )
        self.first_driver_year = years[0]
        self.last_driver_year = years[-1]

    def reset(self, env: Any) -> None:
        self._baseline_targets = {}
        self._baseline_firm_objects = {}
        self._was_active = {}
        self._current_shocks = {}
        self._current_key = None
        self._history = []
        self._current_record = None
        if not self.enabled:
            return
        for firm_id, firm_state in env.state_manager.firms.items():
            key = str(firm_id)
            self._baseline_targets[key] = max(1, int(getattr(firm_state.firm, "target_size", 1)))
            self._baseline_firm_objects[key] = id(firm_state.firm)
            self._was_active[key] = self._is_active_firm(env, firm_id)
        self.apply_month(env)

    def before_capacity_update(self, env):
        if not self.enabled:
            return
        for firm_id, state in env.state_manager.firms.items():
            key = str(firm_id)
            shock = self._current_shocks.get(key)
            if (
                shock is not None
                and shock.shocked_target != shock.baseline_target
                and (self._baseline_firm_objects.get(key) == id(state.firm))
                and self._is_active_firm(env, firm_id)
            ):
                target = self._baseline_targets[key]
                state.firm.target_size = target
                self._set_target_tensors(env, firm_id, key, target)

    def after_capacity_update(self, env):
        if not self.enabled:
            return
        for firm_id, state in env.state_manager.firms.items():
            key = str(firm_id)
            same_object = self._baseline_firm_objects.get(key) == id(state.firm)
            baseline = self._capture_live_baseline(key, state.firm)
            shock = self._current_shocks.get(key)
            if (
                same_object
                and shock is not None
                and self._is_active_firm(env, firm_id)
                and (shock.employment_multiplier != 1.0)
            ):
                target = max(1, int(round(baseline * shock.employment_multiplier)))
                state.firm.target_size = target
                self._set_target_tensors(env, firm_id, key, target)
                self._current_shocks[key] = replace(
                    shock, baseline_target=baseline, shocked_target=target
                )

    def resolve_exposure(self, firm: Any) -> FirmExposure:
        naics6 = normalize_naics6(getattr(firm, "naics_6digit", None))
        industry = str(int(getattr(firm, "industry", 0)))
        default: Optional[FirmExposure] = None
        for exposure in self._exposures:
            if exposure.scope == "default":
                default = exposure
            elif exposure.scope == "naics6_prefix":
                prefix = normalize_naics6(exposure.code)
                if naics6 and prefix and naics6.startswith(prefix):
                    return exposure
            elif exposure.scope == "naics18" and industry == str(int(exposure.code)):
                return exposure
        if default is None:
            raise AssertionError("validated exposure table lost its default row")
        return default

    def _driver_cell(self, year: int, month: int, product_group: str) -> DriverCell:
        if year < self.first_driver_year:
            return DriverCell(self.scenario, year, month, product_group, 0.0, 0.0, 0.0)
        if year > self.last_driver_year:
            raise RuntimeError(
                f"trade-war simulation year {year} exceeds driver end {self.last_driver_year}; refusing silent extrapolation"
            )
        return self._cells[self.scenario, year, month, product_group]

    def _capture_live_baseline(self, firm_id: str, firm: Any) -> int:
        baseline = max(1, int(getattr(firm, "target_size", 1)))
        self._baseline_targets[firm_id] = baseline
        self._baseline_firm_objects[firm_id] = id(firm)
        return baseline

    def _shock_for(
        self, firm_id: str, firm: Any, year: int, month: int, baseline: int
    ) -> Tuple[FirmShock, bool]:
        exposure = self.resolve_exposure(firm)
        cell = self._driver_cell(year, month, exposure.product_group)
        basket = input_basket_key(exposure.scope, exposure.code)
        input_cell = self._driver_cell(year, month, basket)
        export_loss = min(
            self.max_export_demand_loss,
            exposure.export_share_us * exposure.export_demand_elasticity * cell.us_tariff_rate,
        )
        input_cost_outlay = (
            exposure.imported_input_share
            * exposure.tariff_pass_through
            * input_cell.canada_countertariff_rate
        )
        input_cost = input_cost_outlay * exposure.current_outlays_over_revenue
        if not math.isfinite(input_cost):
            raise RuntimeError("non-finite revenue-normalized input cost")
        productivity_loss = min(0.95, exposure.productivity_drag_full * cell.uncertainty_index)
        active = any((value > 0.0 for value in (export_loss, input_cost, productivity_loss)))
        demand_multiplier = max(0.0, 1.0 - export_loss)
        productivity_multiplier = max(0.0, 1.0 - productivity_loss)
        margin_multiplier = max(
            self.min_margin_multiplier, demand_multiplier * productivity_multiplier - input_cost
        )
        employment_base = max(
            self.min_target_multiplier, demand_multiplier * max(0.0, 1.0 - input_cost)
        )
        employment_multiplier = max(
            self.min_target_multiplier, employment_base**exposure.employment_elasticity
        )
        shocked_target = max(1, int(round(baseline * employment_multiplier)))
        return (
            FirmShock(
                firm_id=firm_id,
                industry=int(getattr(firm, "industry", 0)),
                da_id=getattr(firm, "da_id", None),
                exposure_label=exposure.label,
                product_group=exposure.product_group,
                baseline_target=baseline,
                shocked_target=shocked_target,
                export_demand_loss=export_loss,
                imported_input_cost_rate=input_cost,
                productivity_loss=productivity_loss,
                demand_multiplier=demand_multiplier,
                productivity_multiplier=productivity_multiplier,
                margin_multiplier=margin_multiplier,
                employment_multiplier=employment_multiplier,
                input_basket=basket,
                input_tariff_rate=input_cell.canada_countertariff_rate,
                imported_input_cost_outlay_rate=input_cost_outlay,
            ),
            active,
        )

    @staticmethod
    def _set_target_tensors(env: Any, firm_id_raw: Any, firm_id: str, target: int) -> None:
        firm_idx = getattr(env, "_firm_id_to_idx", {}).get(firm_id_raw)
        if firm_idx is None:
            firm_idx = getattr(env, "_firm_id_to_idx", {}).get(firm_id)
        if firm_idx is None:
            return
        if getattr(env, "firm_target_size", None) is not None:
            env.firm_target_size[firm_idx] = target
        if getattr(env, "firm_base_size", None) is not None:
            env.firm_base_size[firm_idx] = target

    @staticmethod
    def _is_active_firm(env: Any, firm_id_raw: Any) -> bool:
        active_tensor = getattr(env, "firm_active", None)
        if active_tensor is None:
            return True
        index_map = getattr(env, "_firm_id_to_idx", {})
        firm_idx = index_map.get(firm_id_raw, index_map.get(str(firm_id_raw)))
        if firm_idx is None:
            return False
        value = active_tensor[firm_idx]
        return bool(value.item() if hasattr(value, "item") else value)

    def apply_month(self, env: Any) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        year = int(env.current_year)
        month = int(env.current_month)
        key = (year, month)
        if self._current_key == key:
            return self._current_record
        shocks: Dict[str, FirmShock] = {}
        vacancy_ids_to_remove: List[str] = []
        industry_rollup: Dict[int, Dict[str, float]] = {}
        da_rollup: Dict[int, Dict[str, float]] = {}
        product_rollup: Dict[str, Dict[str, float]] = {}
        total_weight = 0.0
        weighted_margin = 0.0
        weighted_productivity = 0.0
        baseline_jobs = 0
        shocked_jobs = 0
        exposed_firms = 0
        affected_firms = 0
        for firm_id_raw, firm_state in env.state_manager.firms.items():
            firm_id = str(firm_id_raw)
            firm = firm_state.firm
            is_active = self._is_active_firm(env, firm_id_raw)
            if not is_active:
                self._was_active[firm_id] = False
                firm.trade_margin_multiplier = 1.0
                firm.trade_demand_multiplier = 1.0
                firm.trade_productivity_multiplier = 1.0
                firm.trade_input_cost_rate = 0.0
                continue
            previous_shock = self._current_shocks.get(firm_id)
            object_changed = self._baseline_firm_objects.get(firm_id) != id(firm)
            reentered = self._was_active.get(firm_id) is False
            self._was_active[firm_id] = True
            if object_changed or reentered:
                baseline = self._capture_live_baseline(firm_id, firm)
            else:
                baseline = self._baseline_targets.get(firm_id)
                if baseline is None:
                    baseline = self._capture_live_baseline(firm_id, firm)
            shock, active = self._shock_for(firm_id, firm, year, month, baseline)
            if not active:
                if previous_shock is not None and any(
                    (
                        value != 1.0
                        for value in (
                            previous_shock.demand_multiplier,
                            previous_shock.productivity_multiplier,
                            previous_shock.employment_multiplier,
                        )
                    )
                ):
                    firm.target_size = baseline
                    self._set_target_tensors(env, firm_id_raw, firm_id, baseline)
                else:
                    baseline = self._capture_live_baseline(firm_id, firm)
                    shock, active = self._shock_for(firm_id, firm, year, month, baseline)
            shocks[firm_id] = shock
            if active:
                firm.target_size = shock.shocked_target
                self._set_target_tensors(env, firm_id_raw, firm_id, shock.shocked_target)
            firm.trade_margin_multiplier = shock.margin_multiplier
            firm.trade_demand_multiplier = shock.demand_multiplier
            firm.trade_productivity_multiplier = shock.productivity_multiplier
            firm.trade_input_cost_rate = shock.imported_input_cost_rate
            firm.trade_exposure_label = shock.exposure_label
            firm.trade_product_group = shock.product_group
            allowed_open = max(0, shock.shocked_target - len(firm_state.employees))
            firm_vacancies = sorted(
                firm_state.vacancies.values(),
                key=lambda vacancy: (int(getattr(vacancy, "months_posted", 0)), str(vacancy.id)),
                reverse=True,
            )
            if active and len(firm_vacancies) > allowed_open:
                vacancy_ids_to_remove.extend(
                    (str(vacancy.id) for vacancy in firm_vacancies[allowed_open:])
                )
            weight = float(max(1, shock.baseline_target))
            total_weight += weight
            weighted_margin += weight * shock.margin_multiplier
            weighted_productivity += weight * shock.productivity_multiplier
            baseline_jobs += shock.baseline_target
            shocked_jobs += shock.shocked_target
            if active:
                exposed_firms += 1
            if shock.shocked_target < shock.baseline_target:
                affected_firms += 1
            for rollup, rollup_key in (
                (industry_rollup, shock.industry),
                (da_rollup, int(shock.da_id) if shock.da_id is not None else -1),
                (product_rollup, shock.product_group),
            ):
                bucket = rollup.setdefault(
                    rollup_key,
                    {
                        "firms": 0.0,
                        "baseline_target_jobs": 0.0,
                        "shocked_target_jobs": 0.0,
                        "weighted_margin_sum": 0.0,
                        "weight": 0.0,
                    },
                )
                bucket["firms"] += 1.0
                bucket["baseline_target_jobs"] += shock.baseline_target
                bucket["shocked_target_jobs"] += shock.shocked_target
                bucket["weighted_margin_sum"] += weight * shock.margin_multiplier
                bucket["weight"] += weight
        if vacancy_ids_to_remove:
            env.state_manager.remove_vacancies(vacancy_ids_to_remove)
        margin_map = {firm_id: shock.margin_multiplier for firm_id, shock in shocks.items()}
        env.matching_mechanism._trade_margin_multiplier_by_firm = margin_map

        def finalize_rollup(source: Mapping[Any, Mapping[str, float]]) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for rollup_key, bucket_raw in source.items():
                bucket = dict(bucket_raw)
                weight = max(bucket.pop("weight"), 1.0)
                weighted = bucket.pop("weighted_margin_sum")
                bucket["mean_margin_multiplier"] = weighted / weight
                bucket["target_jobs_lost"] = (
                    bucket["baseline_target_jobs"] - bucket["shocked_target_jobs"]
                )
                bucket["firms"] = int(bucket["firms"])
                result[str(rollup_key)] = bucket
            return result

        record: Dict[str, Any] = {
            "scenario": self.scenario,
            "year": year,
            "month": month,
            "exposed_firms": exposed_firms,
            "affected_firms": affected_firms,
            "baseline_target_jobs": baseline_jobs,
            "shocked_target_jobs": shocked_jobs,
            "target_jobs_lost": baseline_jobs - shocked_jobs,
            "vacancies_withdrawn": len(vacancy_ids_to_remove),
            "trade_layoffs": 0,
            "realized_employment": 0,
            "realized_vacancies": 0,
            "realized_hourly_payroll": 0.0,
            "realized_profit": 0.0,
            "effective_output_worker_equivalents": 0.0,
            "realized_employment_weighted_productivity_multiplier": 0.0,
            "mean_margin_multiplier": weighted_margin / max(total_weight, 1.0),
            "mean_productivity_multiplier": weighted_productivity / max(total_weight, 1.0),
            "by_industry": finalize_rollup(industry_rollup),
            "by_da": finalize_rollup(da_rollup),
            "by_product_group": finalize_rollup(product_rollup),
        }
        self._current_key = key
        self._current_shocks = shocks
        self._current_record = record
        self._history.append(record)
        return record

    def capture_outcomes(self, env: Any) -> None:
        if not self.enabled or self._current_record is None:
            return
        aggregate = {
            "active_firms": 0.0,
            "employment": 0.0,
            "vacancies": 0.0,
            "hourly_payroll": 0.0,
            "profit": 0.0,
            "effective_output_worker_equivalents": 0.0,
        }
        productivity_weighted_employment = 0.0
        by_industry: Dict[int, Dict[str, float]] = {}
        by_da: Dict[int, Dict[str, float]] = {}
        by_product_group: Dict[str, Dict[str, float]] = {}
        profits = getattr(env, "last_firm_profit_raw", {}) or {}
        for firm_id_raw, firm_state in env.state_manager.firms.items():
            firm_id = str(firm_id_raw)
            shock = self._current_shocks.get(firm_id)
            if shock is None:
                continue
            is_active = self._is_active_firm(env, firm_id_raw)
            employment = float(len(firm_state.employees)) if is_active else 0.0
            vacancies = float(len(firm_state.vacancies)) if is_active else 0.0
            payroll = 0.0
            if is_active:
                for worker_id in firm_state.employees:
                    worker_state = env.state_manager.workers.get(worker_id)
                    if worker_state is not None:
                        payroll += float(worker_state.worker.current_wage)
            profit = float(profits.get(firm_id_raw, profits.get(firm_id, 0.0)))
            output_equiv = employment * shock.demand_multiplier * shock.productivity_multiplier
            productivity_weighted_employment += employment * shock.productivity_multiplier
            values = {
                "active_firms": float(is_active),
                "employment": employment,
                "vacancies": vacancies,
                "hourly_payroll": payroll,
                "profit": profit,
                "effective_output_worker_equivalents": output_equiv,
            }
            for name, value in values.items():
                aggregate[name] += value
            for rollup, key in (
                (by_industry, shock.industry),
                (by_da, int(shock.da_id) if shock.da_id is not None else -1),
                (by_product_group, shock.product_group),
            ):
                bucket = rollup.setdefault(key, {name: 0.0 for name in aggregate})
                for name, value in values.items():
                    bucket[name] += value
        record = self._current_record
        record["realized_employment"] = int(aggregate["employment"])
        record["realized_vacancies"] = int(aggregate["vacancies"])
        record["realized_hourly_payroll"] = aggregate["hourly_payroll"]
        record["realized_profit"] = aggregate["profit"]
        record["effective_output_worker_equivalents"] = aggregate[
            "effective_output_worker_equivalents"
        ]
        record["realized_employment_weighted_productivity_multiplier"] = (
            productivity_weighted_employment / aggregate["employment"]
            if aggregate["employment"] > 0.0
            else 0.0
        )
        record["outcomes_by_industry"] = {str(key): values for key, values in by_industry.items()}
        record["outcomes_by_da"] = {str(key): values for key, values in by_da.items()}
        record["outcomes_by_product_group"] = {
            str(key): values for key, values in by_product_group.items()
        }

    def trade_excess_for_firm(self, firm_id: str, employees: int) -> int:
        shock = self._current_shocks.get(str(firm_id))
        if shock is None:
            return 0
        baseline_supported = min(int(employees), shock.baseline_target)
        return max(0, baseline_supported - shock.shocked_target)

    def record_trade_layoffs(self, count: int) -> None:
        if self.enabled and self._current_record is not None:
            self._current_record["trade_layoffs"] += int(count)

    def current_metrics(self) -> Dict[str, float]:
        if not self.enabled or self._current_record is None:
            return {}
        record = self._current_record
        return {
            "trade_target_jobs_lost": float(record["target_jobs_lost"]),
            "trade_layoffs": float(record["trade_layoffs"]),
            "trade_vacancies_withdrawn": float(record["vacancies_withdrawn"]),
            "trade_realized_employment": float(record["realized_employment"]),
            "trade_realized_vacancies": float(record["realized_vacancies"]),
            "trade_realized_hourly_payroll": float(record["realized_hourly_payroll"]),
            "trade_realized_profit": float(record["realized_profit"]),
            "trade_effective_output_worker_equivalents": float(
                record["effective_output_worker_equivalents"]
            ),
            "trade_realized_employment_weighted_productivity_multiplier": float(
                record["realized_employment_weighted_productivity_multiplier"]
            ),
            "trade_mean_margin_multiplier": float(record["mean_margin_multiplier"]),
            "trade_mean_productivity_multiplier": float(record["mean_productivity_multiplier"]),
        }


__all__ = [
    "DRIVER_COLUMNS",
    "EXPOSURE_COLUMNS",
    "DriverCell",
    "FirmExposure",
    "FirmShock",
    "TradeWarRegime",
    "load_driver",
    "load_exposures",
    "select_exposures",
    "normalize_naics6",
    "sha256_file",
]
