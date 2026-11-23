from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

SUBTYPE_NONE = 0
SUBTYPE_RETIREMENT = 1
SUBTYPE_SCHOOL = 2
SUBTYPE_ILLNESS = 3
SUBTYPE_FAMILY = 4
SUBTYPE_DISCOURAGED = 5
_EXOGENOUS_SUBTYPES = (SUBTYPE_RETIREMENT, SUBTYPE_SCHOOL, SUBTYPE_ILLNESS, SUBTYPE_FAMILY)
_EXOGENOUS_PCT_COLS = ("pct_retirement", "pct_school", "pct_illness", "pct_family")
SUBTYPE_NAME = {
    SUBTYPE_NONE: "none",
    SUBTYPE_RETIREMENT: "retirement",
    SUBTYPE_SCHOOL: "school",
    SUBTYPE_ILLNESS: "illness",
    SUBTYPE_FAMILY: "family",
    SUBTYPE_DISCOURAGED: "discouraged",
}
SUBTYPE_OUTSIDE_OPTION = {
    SUBTYPE_NONE: 0.1,
    SUBTYPE_RETIREMENT: 0.12,
    SUBTYPE_SCHOOL: 0.2,
    SUBTYPE_ILLNESS: 0.1,
    SUBTYPE_FAMILY: 0.1,
    SUBTYPE_DISCOURAGED: 0.0,
}


def age_to_age_group(age: int) -> str:
    if age < 15:
        return "15_19"
    if age >= 70:
        return "70_plus"
    if age >= 65:
        return "65_69"
    lo = int(age) // 5 * 5
    return f"{lo}_{lo + 4}"


def _sex_int_to_label(sex: int) -> str:
    if sex == 1:
        return "male"
    if sex == 2:
        return "female"
    return "unknown"


def sample_sex_for_age(age: int, rng=None) -> int:
    if rng is None:
        import numpy as _np

        rng = _np.random
    if age < 25:
        m_share = 0.5075
    elif age < 55:
        m_share = 0.4825
    elif age < 65:
        m_share = 0.4889
    elif age < 75:
        m_share = 0.4655
    else:
        m_share = 0.4127
    return 1 if rng.random() < m_share else 2


def normalize_sex_value(value) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and value != value:
        return 0
    if isinstance(value, bool):
        return 0
    try:
        import numpy as _np

        if isinstance(value, _np.integer):
            iv = int(value)
            return iv if iv in (1, 2) else 0
    except ImportError:
        pass
    if isinstance(value, int):
        return value if value in (1, 2) else 0
    s = str(value).strip().lower()
    if s in ("m", "male", "1"):
        return 1
    if s in ("f", "female", "2"):
        return 2
    return 0


class NILFSubtypeSampler:

    def __init__(
        self, nilf_composition_by_sex: pd.DataFrame, rng: Optional[np.random.Generator] = None
    ):
        self._rng = rng if rng is not None else np.random.default_rng()
        self._probs = {}
        self._fallback_by_age = {}
        self._load(nilf_composition_by_sex)

    def _load(self, df: pd.DataFrame):
        missing = set(["year", "age_group", "sex"] + list(_EXOGENOUS_PCT_COLS)) - set(df.columns)
        if missing:
            raise ValueError(f"NILFSubtypeSampler: by-sex CSV is missing columns {missing}")
        for _, row in df.iterrows():
            key = (int(row["year"]), str(row["age_group"]), str(row["sex"]).lower())
            p = np.array([float(row[c]) for c in _EXOGENOUS_PCT_COLS], dtype=np.float64)
            s = p.sum()
            if s > 0:
                p = p / s
            else:
                p = np.array([1.0, 0.0, 0.0, 0.0])
            self._probs[key] = p
        for (age_group, sex_label), grp in df.groupby(["age_group", "sex"]):
            counts = (
                grp[list(_EXOGENOUS_PCT_COLS)].values * grp["nilf_total"].values.reshape(-1, 1)
            ).sum(axis=0)
            s = counts.sum()
            if s > 0:
                self._fallback_by_age[str(age_group), str(sex_label).lower()] = counts / s
        self._uniform = np.array([0.25, 0.25, 0.25, 0.25])

    def _lookup_probs(self, year: int, age_group: str, sex_label: str) -> np.ndarray:
        p = self._probs.get((year, age_group, sex_label))
        if p is not None:
            return p
        candidates = [y for y, ag, sx in self._probs.keys() if ag == age_group and sx == sex_label]
        if candidates:
            nearest = min(candidates, key=lambda y: abs(y - year))
            return self._probs[nearest, age_group, sex_label]
        p = self._fallback_by_age.get((age_group, sex_label))
        if p is not None:
            return p
        return self._uniform

    def sample(self, year: int, age: int, sex: int) -> int:
        age_group = age_to_age_group(age)
        sex_label = _sex_int_to_label(sex)
        if sex_label == "unknown":
            p_male = self._lookup_probs(year, age_group, "male")
            p_female = self._lookup_probs(year, age_group, "female")
            p = 0.5 * (p_male + p_female)
        else:
            p = self._lookup_probs(year, age_group, sex_label)
        idx = int(self._rng.choice(len(_EXOGENOUS_SUBTYPES), p=p))
        return _EXOGENOUS_SUBTYPES[idx]


def effective_subtype_windows(config_mnl, age: int) -> tuple:
    school_w = int(getattr(config_mnl, "school_search_eligible_months", 36))
    base_family = int(getattr(config_mnl, "family_search_eligible_months", 24))
    base_illness = int(getattr(config_mnl, "illness_search_eligible_months", 6))
    return (school_w, base_family, base_illness)


def is_search_eligible(
    subtype: int,
    months_in_subtype: int,
    school_window: int,
    family_window: int,
    illness_window: int,
    allow_retirement_reentry: bool = False,
) -> bool:
    if subtype == SUBTYPE_NONE:
        return True
    if subtype == SUBTYPE_RETIREMENT:
        return allow_retirement_reentry is True
    if subtype == SUBTYPE_DISCOURAGED:
        return True
    if subtype == SUBTYPE_SCHOOL:
        return months_in_subtype >= school_window
    if subtype == SUBTYPE_FAMILY:
        return months_in_subtype >= family_window
    if subtype == SUBTYPE_ILLNESS:
        return months_in_subtype >= illness_window
    return False
