from __future__ import annotations
from decimal import Decimal
import math

EMPLOYMENT_GROWTH = float(Decimal("3746.2") / Decimal("3308.7") - 1)
POPULATION_15PLUS_GROWTH = float(Decimal("6036.4") / Decimal("5178.5") - 1)
OFFICIAL_ELASTICITY = float(
    (Decimal("3746.2") / Decimal("3308.7") - 1) / (Decimal("6036.4") / Decimal("5178.5") - 1)
)


def indexed_capacity_targets(
    targets,
    active,
    before_population,
    after_population,
    filled=None,
    reference_population=None,
    reference_capacity=None,
    allocation_weights=None,
):
    if (
        type(before_population) is not int
        or type(after_population) is not int
        or before_population < 1
        or (after_population < 1)
    ):
        raise ValueError("Invalid population for the regional capacity index")
    if len(targets) != len(active) or not targets:
        raise ValueError("capacity target/activity shape mismatch")
    clean = []
    for target, is_active in zip(targets, active):
        if type(target) is not int or target < 0 or type(is_active) is not bool:
            raise ValueError("invalid capacity target/activity value")
        clean.append(target)
    if filled is None:
        filled = [0] * len(targets)
    if len(filled) != len(targets) or any(
        (type(value) is not int or value < 0 for value in filled)
    ):
        raise ValueError("invalid filled-employment floors")
    active_indexes = [i for i, value in enumerate(active) if value]
    current = sum((clean[i] for i in active_indexes))
    if not active_indexes or current < 1:
        raise ValueError("no positive active capacity")
    if reference_population is None:
        reference_population = before_population
    if reference_capacity is None:
        reference_capacity = current
    if (
        type(reference_population) is not int
        or reference_population < 1
        or type(reference_capacity) is not int
        or (reference_capacity < 1)
    ):
        raise ValueError("Invalid reference for the regional capacity index")
    population_growth = after_population / reference_population - 1.0
    desired = round(reference_capacity * (1.0 + OFFICIAL_ELASTICITY * population_growth))
    floor_total = sum((filled[i] for i in active_indexes))
    if floor_total > desired:
        raise ValueError("Filled employment exceeds the regional capacity")
    if allocation_weights is None:
        allocation_weights = clean
    if len(allocation_weights) != len(targets) or any(
        (type(value) not in (int, float) or value < 0 for value in allocation_weights)
    ):
        raise ValueError("invalid capacity-allocation weights")
    weight_total = sum((allocation_weights[i] for i in active_indexes))
    if weight_total <= 0:
        raise ValueError("no positive active allocation weight")
    distributable = desired - floor_total
    exact = {
        i: filled[i] + allocation_weights[i] * distributable / weight_total for i in active_indexes
    }
    result = list(clean)
    for i in active_indexes:
        result[i] = math.floor(exact[i])
    remainder = desired - sum((result[i] for i in active_indexes))
    order = sorted(active_indexes, key=lambda i: (-(exact[i] - result[i]), i))
    for i in order[:remainder]:
        result[i] += 1
    if sum((result[i] for i in active_indexes)) != desired:
        raise ValueError("The regional capacity allocation failed")
    return result


def install(age_renewal_module) -> None:
    if getattr(age_renewal_module, "_regional_capacity_installed", False):
        return
    original_index = age_renewal_module.index_firm_capacity
    age_renewal_module.indexed_capacity_targets = indexed_capacity_targets

    def index_firm_capacity(*args, **kwargs):
        receipt = original_index(*args, **kwargs)
        if receipt is not None:
            receipt = dict(receipt)
            receipt["mode"] = "official_population_elasticity_capacity_v1"
            receipt["official_elasticity"] = OFFICIAL_ELASTICITY
            receipt["official_employment_growth_2006_2016"] = EMPLOYMENT_GROWTH
            receipt["official_population_15plus_growth_2006_2016"] = POPULATION_15PLUS_GROWTH
        return receipt

    age_renewal_module.index_firm_capacity = index_firm_capacity
    age_renewal_module._regional_capacity_installed = True
