from __future__ import annotations
import math
from typing import Any, Dict, Sequence, Tuple

try:
    from scipy.stats import t as _student_t

    def _t_975(df: int) -> float:
        return float(_student_t.ppf(0.975, df=max(df, 1)))

except Exception:
    _T_TABLE = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
        40: 2.021,
        60: 2.0,
        120: 1.98,
    }

    def _t_975(df: int) -> float:
        df = max(int(df), 1)
        keys = sorted(_T_TABLE.keys())
        for k in keys:
            if df <= k:
                return _T_TABLE[k]
        return 1.96


FULL_11_INDICES: Tuple[int, ...] = tuple(range(11))
LAST_3_INDICES: Tuple[int, ...] = (8, 9, 10)


def _mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and (not (isinstance(x, float) and math.isnan(x)))]
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: Sequence[float], ddof: int = 1) -> float:
    xs = [float(x) for x in xs if x is not None and (not (isinstance(x, float) and math.isnan(x)))]
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum(((x - m) ** 2 for x in xs)) / (n - ddof))


def _ci_half(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and (not (isinstance(x, float) and math.isnan(x)))]
    n = len(xs)
    if n < 2:
        return float("nan")
    return _t_975(n - 1) * _std(xs, ddof=1) / math.sqrt(n)


def _windowed_mean(time_series: Sequence[float], indices: Sequence[int]) -> float:
    vals = []
    for i in indices:
        if 0 <= i < len(time_series):
            v = time_series[i]
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(float(v))
    return _mean(vals) if vals else float("nan")


def aggregate_per_rollout_results(per_rollout_values: Sequence[float]) -> Dict[str, float]:
    vals = [
        float(v)
        for v in per_rollout_values
        if v is not None and (not (isinstance(v, float) and math.isnan(v)))
    ]
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "std": _std(vals, ddof=1),
        "ci_half": _ci_half(vals),
        "ci_lo": _mean(vals) - _ci_half(vals) if len(vals) >= 2 else float("nan"),
        "ci_hi": _mean(vals) + _ci_half(vals) if len(vals) >= 2 else float("nan"),
        "per_rollout": vals,
    }


def windowed_aggregate(
    per_rollout_series: Sequence[Sequence[float]], window: str
) -> Dict[str, float]:
    indices = (
        FULL_11_INDICES if window == "full_11" else LAST_3_INDICES if window == "last_3" else None
    )
    if indices is None:
        raise ValueError(f"unknown window {window!r}; expected 'full_11' or 'last_3'")
    per_rollout_window_means = [_windowed_mean(s, indices) for s in per_rollout_series]
    return aggregate_per_rollout_results(per_rollout_window_means)


def classify_band_membership(mean: float, ci_half: float, band_lo: float, band_hi: float) -> str:
    if any((isinstance(v, float) and math.isnan(v) for v in (mean, ci_half, band_lo, band_hi))):
        return "NA"
    ci_lo = mean - ci_half
    ci_hi = mean + ci_half
    in_band = mean >= band_lo and mean <= band_hi
    ci_in = ci_lo >= band_lo and ci_hi <= band_hi
    if in_band and ci_in:
        return "PASS"
    if mean < band_lo and ci_hi >= band_lo:
        return "EDGE"
    if mean > band_hi and ci_lo <= band_hi:
        return "EDGE"
    if in_band and (not ci_in):
        return "EDGE"
    return "FAIL"


def emit_dual_window_verdict(
    per_rollout_series: Sequence[Sequence[float]], band_lo: float, band_hi: float
) -> Dict[str, Any]:
    full = windowed_aggregate(per_rollout_series, "full_11")
    last = windowed_aggregate(per_rollout_series, "last_3")
    verdict_full = classify_band_membership(full["mean"], full["ci_half"], band_lo, band_hi)
    verdict_last = classify_band_membership(last["mean"], last["ci_half"], band_lo, band_hi)

    def _ci_in(w):
        return w["mean"] - w["ci_half"] >= band_lo and w["mean"] + w["ci_half"] <= band_hi

    def _ci_out(w):
        ci_lo = w["mean"] - w["ci_half"]
        ci_hi = w["mean"] + w["ci_half"]
        return ci_hi < band_lo or ci_lo > band_hi

    if _ci_in(full) and _ci_out(last) or (_ci_in(last) and _ci_out(full)):
        final = "WINDOW-SENSITIVE"
    else:
        final = verdict_full
    return {
        "band": [band_lo, band_hi],
        "full_11": full,
        "last_3": last,
        "verdict_full_11": verdict_full,
        "verdict_last_3": verdict_last,
        "final_classification": final,
    }


STANDARD_BANDS: Dict[str, Tuple[float, float]] = {
    "U_LF_pct": (4.0, 8.0),
    "NILF_55_64_pct": (28.0, 42.0),
    "NILF_65_99_pct": (76.0, 92.0),
    "NILF_35_44_pct": (float("-inf"), 20.0),
}
