from __future__ import annotations
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple
import numpy as np
import pandas as pd

COVID_NONE = "none"
COVID_TEMP_LAYOFF = "temporary_layoff"
COVID_MARGINAL_ATTACHED = "marginally_attached"
COVID_STATES = {COVID_NONE, COVID_TEMP_LAYOFF, COVID_MARGINAL_ATTACHED}
PROBABILITY_COLUMNS = (
    "temporary_layoff_entry_hazard",
    "temporary_layoff_recall_hazard",
    "marginal_attachment_entry_hazard",
    "marginal_attachment_reentry_hazard",
)
REQUIRED_COLUMNS = (
    "year",
    "month",
    "naics_18",
    "regime_active",
    *PROBABILITY_COLUMNS,
    "telework_capacity",
    "layoff_exposure_multiplier",
    "stringency_index",
    "shock_intensity",
    "target_excess_temp_layoff_share",
    "target_excess_marginal_share",
)


@dataclass(frozen=True)
class CovidMonthDrivers:
    year: int
    month: int
    naics_18: int
    regime_active: bool
    temporary_layoff_entry_hazard: float
    temporary_layoff_recall_hazard: float
    marginal_attachment_entry_hazard: float
    marginal_attachment_reentry_hazard: float
    telework_capacity: float
    layoff_exposure_multiplier: float
    stringency_index: float
    shock_intensity: float
    target_excess_temp_layoff_share: float
    target_excess_marginal_share: float


class CovidDriverTable:

    def __init__(
        self, rows: Dict[Tuple[int, int, int], CovidMonthDrivers], source: Path, source_sha256: str
    ):
        self._rows = rows
        self.source = source
        self.source_sha256 = source_sha256
        self.months = tuple(sorted({(y, m) for y, m, _ in rows}))

    @classmethod
    def from_csv(cls, path: Path | str) -> "CovidDriverTable":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"COVID regime enabled but driver file does not exist: {source}"
            )
        payload = source.read_bytes()
        frame = pd.read_csv(source)
        missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
        if missing:
            raise ValueError(f"COVID driver file {source} is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"COVID driver file {source} contains no rows")
        frame = frame.loc[:, REQUIRED_COLUMNS].copy()
        for name in ("year", "month", "naics_18"):
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(int)
        frame["regime_active"] = frame["regime_active"].map(_parse_bool)
        numeric = [
            name
            for name in REQUIRED_COLUMNS
            if name not in {"year", "month", "naics_18", "regime_active"}
        ]
        for name in numeric:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
            if not np.isfinite(frame[name]).all():
                raise ValueError(f"COVID driver column {name} must contain finite values")
        if not frame["month"].between(1, 12).all():
            bad = frame.loc[~frame["month"].between(1, 12), ["year", "month"]]
            raise ValueError(
                f"COVID driver months must be in 1..12; invalid rows: {bad.to_dict('records')}"
            )
        if not frame["naics_18"].between(1, 18).all():
            bad = frame.loc[~frame["naics_18"].between(1, 18), ["year", "month", "naics_18"]]
            raise ValueError(
                f"COVID drivers require explicit NAICS-18 rows in 1..18; invalid rows: {bad.to_dict('records')}"
            )
        duplicated = frame.duplicated(["year", "month", "naics_18"], keep=False)
        if duplicated.any():
            bad = frame.loc[duplicated, ["year", "month", "naics_18"]]
            raise ValueError(f"Duplicate COVID driver keys: {bad.to_dict('records')}")
        bounded = (
            *PROBABILITY_COLUMNS,
            "telework_capacity",
            "shock_intensity",
            "target_excess_temp_layoff_share",
            "target_excess_marginal_share",
        )
        for name in bounded:
            if not frame[name].between(0.0, 1.0).all():
                raise ValueError(f"COVID driver column {name} must lie in [0, 1]")
        nonnegative = (
            "layoff_exposure_multiplier",
            "stringency_index",
            "target_excess_temp_layoff_share",
            "target_excess_marginal_share",
        )
        for name in nonnegative:
            if (frame[name] < 0.0).any():
                raise ValueError(f"COVID driver column {name} must be nonnegative")
        if (frame["stringency_index"] > 100.0).any():
            raise ValueError("COVID driver stringency_index must lie in [0, 100]")
        entry_sum = frame[
            ["temporary_layoff_entry_hazard", "marginal_attachment_entry_hazard"]
        ].sum(axis=1)
        if (entry_sum > 1.0 + 1e-12).any():
            bad = frame.loc[entry_sum > 1.0 + 1e-12, ["year", "month", "naics_18"]]
            raise ValueError(
                f"COVID employed-state entry hazards must sum to <= 1; invalid keys: {bad.to_dict('records')}"
            )
        invariant_columns = (
            "regime_active",
            "temporary_layoff_recall_hazard",
            "marginal_attachment_entry_hazard",
            "marginal_attachment_reentry_hazard",
            "stringency_index",
            "shock_intensity",
            "target_excess_temp_layoff_share",
            "target_excess_marginal_share",
        )
        for (year, month), group in frame.groupby(["year", "month"], sort=True):
            for name in invariant_columns:
                if group[name].nunique(dropna=False) != 1:
                    raise ValueError(f"COVID driver {name} varies within {year}-{month:02d}")
        rows: Dict[Tuple[int, int, int], CovidMonthDrivers] = {}
        for record in frame.to_dict("records"):
            row = CovidMonthDrivers(**record)
            rows[row.year, row.month, row.naics_18] = row
        return cls(rows, source, sha256(payload).hexdigest())

    def validate_coverage(
        self, years: Iterable[int], industries: Iterable[int] = range(1, 19)
    ) -> None:
        missing = []
        for year in years:
            for month in range(1, 13):
                for industry in industries:
                    if (int(year), month, int(industry)) not in self._rows:
                        missing.append((int(year), month, int(industry)))
        if missing:
            preview = missing[:12]
            suffix = "..." if len(missing) > len(preview) else ""
            raise ValueError(
                f"COVID driver file lacks {len(missing)} required year/month/industry rows: {preview}{suffix}"
            )

    def get(self, year: int, month: int, industry: int) -> Optional[CovidMonthDrivers]:
        return self._rows.get((int(year), int(month), int(industry)))

    def require(self, year: int, month: int, industry: int) -> CovidMonthDrivers:
        row = self.get(year, month, industry)
        if row is None:
            raise KeyError(
                f"Missing COVID driver row for {int(year)}-{int(month):02d}, NAICS {int(industry)}"
            )
        return row

    def has_month(self, year: int, month: int) -> bool:
        return (int(year), int(month)) in self.months

    def month_is_active(self, year: int, month: int) -> bool:
        rows = [row for (y, m, _), row in self._rows.items() if y == int(year) and m == int(month)]
        return bool(rows and rows[0].regime_active)

    def metadata(self) -> Dict[str, object]:
        return {
            "driver_file": str(self.source),
            "driver_sha256": self.source_sha256,
            "first_month": list(self.months[0]) if self.months else None,
            "last_month": list(self.months[-1]) if self.months else None,
            "n_rows": len(self._rows),
        }


@dataclass
class CovidMonthLedger:
    year: int
    month: int
    regime_active: bool
    temp_stock_start: int = 0
    marginal_stock_start: int = 0
    temp_entries: int = 0
    marginal_entries: int = 0
    same_employer_recalls: int = 0
    temp_releases_to_search: int = 0
    marginal_reentries_to_search: int = 0
    marginal_releases_to_nilf: int = 0
    temp_stock_end: int = 0
    marginal_stock_end: int = 0
    temp_entries_by_industry: Dict[str, int] = field(default_factory=dict)
    marginal_entries_by_industry: Dict[str, int] = field(default_factory=dict)

    @property
    def temp_identity_residual(self) -> int:
        return (
            self.temp_stock_start
            + self.temp_entries
            - self.same_employer_recalls
            - self.temp_releases_to_search
            - self.temp_stock_end
        )

    @property
    def marginal_identity_residual(self) -> int:
        return (
            self.marginal_stock_start
            + self.marginal_entries
            - self.marginal_reentries_to_search
            - self.marginal_releases_to_nilf
            - self.marginal_stock_end
        )

    def assert_balanced(self) -> None:
        if self.temp_identity_residual != 0 or self.marginal_identity_residual != 0:
            raise RuntimeError(
                f"COVID stock-flow identity failed for {self.year}-{self.month:02d}: temp={self.temp_identity_residual}, marginal={self.marginal_identity_residual}"
            )

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["temp_identity_residual"] = self.temp_identity_residual
        result["marginal_identity_residual"] = self.marginal_identity_residual
        return result


class CovidTransitionRegime:

    def __init__(self, drivers: CovidDriverTable, seed: int, release_year: int = 2022):
        self.drivers = drivers
        self.seed = int(seed)
        self.release_year = int(release_year)
        self.rng = np.random.default_rng(self.seed)
        self.history = []

    def apply_month(
        self, state_manager, year: int, month: int, is_firm_active: Callable[[Optional[str]], bool]
    ) -> CovidMonthLedger:
        year = int(year)
        month = int(month)
        workers = state_manager.workers
        ledger = CovidMonthLedger(
            year=year,
            month=month,
            regime_active=self.drivers.month_is_active(year, month),
            temp_stock_start=self._count_state(workers, COVID_TEMP_LAYOFF),
            marginal_stock_start=self._count_state(workers, COVID_MARGINAL_ATTACHED),
        )
        if year == self.release_year and month == 1:
            self._release_carryover(state_manager, ledger)
        elif self.drivers.has_month(year, month):
            self._apply_exits(state_manager, year, month, is_firm_active, ledger)
            if ledger.regime_active:
                self._apply_entries(state_manager, year, month, ledger)
        ledger.temp_stock_end = self._count_state(workers, COVID_TEMP_LAYOFF)
        ledger.marginal_stock_end = self._count_state(workers, COVID_MARGINAL_ATTACHED)
        ledger.assert_balanced()
        self.history.append(ledger.to_dict())
        return ledger

    @staticmethod
    def _count_state(workers: Mapping[str, object], covid_state: str) -> int:
        return sum((1 for ws in workers.values() if ws.covid_state == covid_state))

    def _apply_exits(
        self,
        state_manager,
        year: int,
        month: int,
        is_firm_active: Callable[[Optional[str]], bool],
        ledger: CovidMonthLedger,
    ) -> None:
        for worker_id in sorted(state_manager.workers):
            ws = state_manager.workers[worker_id]
            state = ws.covid_state
            if state == COVID_NONE:
                continue
            industry = int(getattr(ws.worker, "industry", 0))
            row = self.drivers.require(year, month, industry)
            if state == COVID_TEMP_LAYOFF:
                employer_id = ws.covid_attached_employer_id
                if not employer_id or not is_firm_active(employer_id):
                    self._release_temp_to_search(ws)
                    ledger.temp_releases_to_search += 1
                elif self.rng.random() < row.temporary_layoff_recall_hazard:
                    if employer_id:
                        self._recall_same_employer(state_manager, worker_id, employer_id)
                        ledger.same_employer_recalls += 1
            elif state == COVID_MARGINAL_ATTACHED:
                if self.rng.random() < row.marginal_attachment_reentry_hazard:
                    self._release_marginal_to_search(state_manager, ws)
                    ledger.marginal_reentries_to_search += 1

    def _apply_entries(
        self, state_manager, year: int, month: int, ledger: CovidMonthLedger
    ) -> None:
        for worker_id in sorted(state_manager.workers):
            ws = state_manager.workers[worker_id]
            worker = ws.worker
            if ws.covid_state != COVID_NONE or worker.employment_status != "Employed":
                continue
            industry = int(getattr(worker, "industry", 0))
            row = self.drivers.require(year, month, industry)
            draw = self.rng.random()
            if draw < row.temporary_layoff_entry_hazard:
                self._enter_temp_layoff(state_manager, worker_id)
                ledger.temp_entries += 1
                _increment(ledger.temp_entries_by_industry, industry)
            elif draw < row.temporary_layoff_entry_hazard + row.marginal_attachment_entry_hazard:
                self._enter_marginal(state_manager, worker_id)
                ledger.marginal_entries += 1
                _increment(ledger.marginal_entries_by_industry, industry)

    def _enter_temp_layoff(self, state_manager, worker_id: str) -> None:
        from environment.state_manager import grant_ei_if_eligible

        ws = state_manager.workers[worker_id]
        worker = ws.worker
        employer_id = getattr(worker, "current_employer_id", None)
        self._save_pre_shock(ws)
        self._detach(state_manager, worker_id, employer_id)
        ws.covid_state = COVID_TEMP_LAYOFF
        ws.covid_attached_employer_id = employer_id
        ws.covid_months_in_state = 0
        worker.employment_status = "Unemployed"
        worker.is_searching = False
        worker.unemployment_months = 0
        worker.commute_time_min = None
        ws.months_employed = 0
        ws.months_unemployed = 0
        ws.separation_reason = "unknown"
        ws.lwf_retention_draw = None
        state_manager._ei_period_new_separations += 1
        if grant_ei_if_eligible(ws, state_manager.config.ei):
            state_manager._ei_period_new_grants += 1

    def _enter_marginal(self, state_manager, worker_id: str) -> None:
        ws = state_manager.workers[worker_id]
        worker = ws.worker
        employer_id = getattr(worker, "current_employer_id", None)
        self._save_pre_shock(ws)
        self._detach(state_manager, worker_id, employer_id)
        ws.covid_state = COVID_MARGINAL_ATTACHED
        ws.covid_attached_employer_id = None
        ws.covid_months_in_state = 0
        worker.employment_status = "NILF"
        worker.is_searching = False
        worker.unemployment_months = 0
        worker.commute_time_min = None
        ws.months_employed = 0
        ws.months_nilf = 0
        ws.ei_eligible = False
        ws.ei_weeks_remaining = 0
        ws.separation_reason = "unknown"
        ws.lwf_retention_draw = None
        if hasattr(state_manager, "clear_nilf_subtype"):
            state_manager.clear_nilf_subtype(ws)

    @staticmethod
    def _save_pre_shock(ws) -> None:
        worker = ws.worker
        ws.previous_tenure_months = int(ws.months_employed)
        ws.previous_wage = float(worker.current_wage)
        ws.covid_pre_shock_wage = float(worker.current_wage)
        ws.covid_pre_shock_hours_per_week = float(worker.hours_per_week)
        ws.covid_pre_shock_tenure_months = int(ws.months_employed)
        ws.covid_pre_shock_commute_time_min = worker.commute_time_min

    @staticmethod
    def _detach(state_manager, worker_id: str, employer_id: Optional[str]) -> None:
        if employer_id and employer_id in state_manager.firms:
            firm_state = state_manager.firms[employer_id]
            if worker_id in firm_state.employees:
                firm_state.employees.remove(worker_id)
            firm_state.firm.n_employees = len(firm_state.employees)
        state_manager.workers[worker_id].worker.current_employer_id = None

    @staticmethod
    def _recall_same_employer(state_manager, worker_id: str, employer_id: str) -> None:
        ws = state_manager.workers[worker_id]
        worker = ws.worker
        firm_state = state_manager.firms[employer_id]
        if worker_id not in firm_state.employees:
            firm_state.employees.append(worker_id)
        firm_state.firm.n_employees = len(firm_state.employees)
        worker.current_employer_id = employer_id
        worker.employment_status = "Employed"
        worker.is_searching = False
        worker.unemployment_months = 0
        worker.current_wage = float(ws.covid_pre_shock_wage)
        worker.hours_per_week = float(ws.covid_pre_shock_hours_per_week)
        worker.commute_time_min = ws.covid_pre_shock_commute_time_min
        ws.months_employed = int(ws.covid_pre_shock_tenure_months + ws.covid_months_in_state)
        ws.months_unemployed = 0
        ws.ei_eligible = False
        ws.ei_weeks_remaining = 0
        ws.separation_reason = "unknown"
        ws.lwf_retention_draw = None
        _clear_covid_state(ws)

    def _release_temp_to_search(self, ws) -> None:
        from environment.state_manager import HIGH_SENIORITY_THRESHOLD_MONTHS, LWF_RETENTION

        worker = ws.worker
        if getattr(worker, "employment_status", None) == "Employed":
            _r5_4h_forensic = {
                "worker_id_hint": id(worker),
                "employment_status": worker.employment_status,
                "current_employer_id": getattr(worker, "current_employer_id", None),
                "current_wage": getattr(worker, "current_wage", None),
                "covid_state": ws.covid_state,
                "covid_attached_employer_id": getattr(ws, "covid_attached_employer_id", None),
                "covid_months_in_state": getattr(ws, "covid_months_in_state", None),
            }
            raise RuntimeError(
                "A temporarily laid-off worker due for release to search is already employed: %r"
                % (_r5_4h_forensic,)
            )
        worker.employment_status = "Unemployed"
        worker.is_searching = True
        worker.unemployment_months = max(0, int(ws.covid_months_in_state))
        ws.months_unemployed = worker.unemployment_months
        ws.separation_reason = "job_loser"
        tier = (
            "high_seniority"
            if ws.previous_tenure_months >= HIGH_SENIORITY_THRESHOLD_MONTHS
            else "average"
        )
        sex = "F" if int(getattr(worker, "sex", 1) or 1) == 2 else "M"
        lo, hi = LWF_RETENTION[tier, sex]
        ws.lwf_retention_draw = float(self.rng.uniform(lo, hi))
        _clear_covid_state(ws)

    @staticmethod
    def _release_marginal_to_search(state_manager, ws) -> None:
        worker = ws.worker
        if getattr(worker, "employment_status", None) == "Employed":
            _r5_4h_forensic = {
                "worker_id_hint": id(worker),
                "employment_status": worker.employment_status,
                "current_employer_id": getattr(worker, "current_employer_id", None),
                "current_wage": getattr(worker, "current_wage", None),
                "covid_state": ws.covid_state,
                "covid_attached_employer_id": getattr(ws, "covid_attached_employer_id", None),
                "covid_months_in_state": getattr(ws, "covid_months_in_state", None),
            }
            raise RuntimeError(
                "A marginally attached worker due for release to search is already employed: %r"
                % (_r5_4h_forensic,)
            )
        worker.employment_status = "Unemployed"
        worker.is_searching = True
        worker.unemployment_months = 0
        ws.months_unemployed = 0
        ws.months_nilf = 0
        ws.separation_reason = "re_entrant"
        ws.lwf_retention_draw = None
        if hasattr(state_manager, "clear_nilf_subtype"):
            state_manager.clear_nilf_subtype(ws)
        _clear_covid_state(ws)

    def _release_carryover(self, state_manager, ledger: CovidMonthLedger) -> None:
        for worker_id in sorted(state_manager.workers):
            ws = state_manager.workers[worker_id]
            if ws.covid_state == COVID_TEMP_LAYOFF:
                self._release_temp_to_search(ws)
                ledger.temp_releases_to_search += 1
            elif ws.covid_state == COVID_MARGINAL_ATTACHED:
                _clear_covid_state(ws)
                ledger.marginal_releases_to_nilf += 1


def _parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse regime_active value as boolean: {value!r}")


def _increment(counter: Dict[str, int], industry: int) -> None:
    key = str(int(industry))
    counter[key] = counter.get(key, 0) + 1


def _clear_covid_state(ws) -> None:
    ws.covid_state = COVID_NONE
    ws.covid_attached_employer_id = None
    ws.covid_months_in_state = 0
    ws.covid_pre_shock_wage = 0.0
    ws.covid_pre_shock_hours_per_week = 0.0
    ws.covid_pre_shock_tenure_months = 0
    ws.covid_pre_shock_commute_time_min = None
