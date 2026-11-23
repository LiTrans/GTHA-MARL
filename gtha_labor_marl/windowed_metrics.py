from __future__ import annotations
import math
from typing import Any, Dict, List, Sequence
from eval_aggregation import FULL_11_INDICES, STANDARD_BANDS


def _windowed_mean(series: Sequence[float], indices: Sequence[int]) -> float:
    vals = [
        float(series[i])
        for i in indices
        if 0 <= i < len(series)
        and series[i] is not None
        and (not (isinstance(series[i], float) and math.isnan(series[i])))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


EXPECTED_YEARS = len(FULL_11_INDICES)


def _native_annual(log: Dict[str, Any]) -> Dict[str, Any]:
    return (log.get("cohort_macros_by_age_native") or {}).get("annual") or {}


def _u_lf_native_pct(log: Dict[str, Any]) -> List[float]:
    annual = _native_annual(log)
    if not annual:
        return []
    n_years = 0
    for c, d in annual.items():
        n_years = max(n_years, len(d.get("pop") or []))
    out: List[float] = []
    for y in range(n_years):
        lf_total = 0.0
        u_active_total = 0.0
        for c, d in annual.items():
            pop = d.get("pop") or []
            lfpr = d.get("lfpr_aligned") or []
            ulf = d.get("ulf_aligned") or []
            if y >= len(pop) or y >= len(lfpr) or y >= len(ulf):
                continue
            cohort_lf = float(pop[y]) * float(lfpr[y])
            lf_total += cohort_lf
            u_active_total += cohort_lf * float(ulf[y])
        out.append(100.0 * u_active_total / lf_total if lf_total > 0 else float("nan"))
    return out


def _cohort_nilf_native_pct(label: str):

    def _f(log: Dict[str, Any]) -> List[float]:
        annual = _native_annual(log)
        cohort = annual.get(label, {}) or {}
        ts = cohort.get("nilf_rate_aligned") or []
        return [float(v) * 100.0 if v is not None else None for v in ts]

    return _f


def _u_lf_pct_aggregate(log: Dict[str, Any]) -> List[float]:
    ts = log.get("unemployment_rate") or []
    return [v * 100.0 if v is not None else None for v in ts]


def _cohort_nilf_aggregate_pct(label: str):

    def _f(log: Dict[str, Any]) -> List[float]:
        cmba = log.get("cohort_macros_by_age", {}) or {}
        annual = (cmba.get("annual") if isinstance(cmba.get("annual"), dict) else cmba) or {}
        cohort = annual.get(label, {}) or {}
        ts = (
            cohort.get("nilf_rate_aligned")
            or cohort.get("nilf_rate")
            or cohort.get("nilf_pct")
            or []
        )
        ts_clean = [v for v in ts if v is not None]
        if ts_clean and max(ts_clean) <= 1.0:
            return [v * 100.0 if v is not None else None for v in ts]
        return list(ts)

    return _f


METRIC_EXTRACTORS = [
    ("U_LF_native_pct", _u_lf_native_pct, STANDARD_BANDS["U_LF_pct"]),
    ("NILF_35_44_native_pct", _cohort_nilf_native_pct("35-44"), STANDARD_BANDS["NILF_35_44_pct"]),
    ("NILF_55_64_native_pct", _cohort_nilf_native_pct("55-64"), STANDARD_BANDS["NILF_55_64_pct"]),
    ("NILF_65_99_native_pct", _cohort_nilf_native_pct("65-99"), STANDARD_BANDS["NILF_65_99_pct"]),
]
DESCRIPTIVE_METRIC_EXTRACTORS = [
    ("U_LF_aggregate_pct", _u_lf_pct_aggregate, STANDARD_BANDS["U_LF_pct"]),
    (
        "NILF_35_44_aggregate_pct",
        _cohort_nilf_aggregate_pct("35-44"),
        STANDARD_BANDS["NILF_35_44_pct"],
    ),
    (
        "NILF_55_64_aggregate_pct",
        _cohort_nilf_aggregate_pct("55-64"),
        STANDARD_BANDS["NILF_55_64_pct"],
    ),
    (
        "NILF_65_99_aggregate_pct",
        _cohort_nilf_aggregate_pct("65-99"),
        STANDARD_BANDS["NILF_65_99_pct"],
    ),
]
