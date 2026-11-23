import pandas as pd

WORKER_ARCHETYPES = {
    0: "native_low_skill",
    1: "native_medium_skill",
    2: "native_high_skill",
    3: "recent_immigrant_low_skill",
    4: "recent_immigrant_high_skill",
    5: "mid_immigrant_low_skill",
    6: "mid_immigrant_high_skill",
    7: "established_immigrant_low_skill",
    8: "established_immigrant_high_skill",
}


def assign_worker_archetype(row) -> int:
    is_immigrant = row.get("immigrant_status", False)
    if isinstance(is_immigrant, str):
        is_immigrant = is_immigrant.lower() in ("true", "1", "yes")
    elif hasattr(pd, "isna") and pd.isna(is_immigrant):
        is_immigrant = False
    else:
        is_immigrant = bool(is_immigrant)
    skill_level = row.get("skill_level", 1)
    if hasattr(pd, "isna") and pd.isna(skill_level):
        skill_level = 1
    skill_level = int(skill_level)
    is_high_skill = skill_level >= 3
    if not is_immigrant:
        if skill_level <= 1:
            return 0
        elif skill_level == 2:
            return 1
        else:
            return 2
    else:
        years = row.get("years_since_arrival", 10)
        if hasattr(pd, "isna") and pd.isna(years) or years == "":
            years = 7.5
        years = float(years)
        if years < 5:
            return 4 if is_high_skill else 3
        elif years <= 10:
            return 6 if is_high_skill else 5
        else:
            return 8 if is_high_skill else 7
