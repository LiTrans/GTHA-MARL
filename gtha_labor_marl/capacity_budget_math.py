from __future__ import annotations
from fractions import Fraction
import math


class BudgetInfeasible(ValueError):

    def __init__(self, *, budget: int, filled_total: int):
        self.budget = budget
        self.filled_total = filled_total
        self.deficit = filled_total - budget
        super().__init__(
            f"occupied employment exceeds funded capacity budget: filled={filled_total}, budget={budget}, deficit={self.deficit}"
        )


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a nonnegative integer")
    return value


def _values(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(label + " must be a nonempty list or tuple")
    return list(values)


def _integer_vector(values, label):
    result = _values(values, label)
    for value in result:
        _integer(value, label + " value")
    return result


def _active_vector(active, size, *, allow_none):
    if active is None and allow_none:
        return [True] * size
    values = _values(active, "activity")
    if len(values) != size or any((type(value) is not bool for value in values)):
        raise ValueError("activity must contain one boolean per slot")
    return values


def _weights(values, size):
    result = _values(values, "allocation weights")
    if len(result) != size:
        raise ValueError("allocation weight shape mismatch")
    converted = []
    for value in result:
        if (
            type(value) not in (int, float)
            or value < 0
            or (type(value) is float and (not math.isfinite(value)))
        ):
            raise ValueError("allocation weights must be finite nonnegative numbers")
        converted.append(Fraction(value))
    return converted


def _largest_remainders(total, weights):
    if total == 0:
        return [0] * len(weights)
    denominator = sum(weights)
    if denominator <= 0:
        raise ValueError("positive unallocated budget has no positive allocation weight")
    exact = [weight * total / denominator for weight in weights]
    result = [quota.numerator // quota.denominator for quota in exact]
    remainder = total - sum(result)
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - result[i]), i))
    for index in order[:remainder]:
        result[index] += 1
    if sum(result) != total:
        raise ArithmeticError("largest-remainder allocation did not conserve its total")
    return result


def allocate_funded(
    budget: int, filled: list[int], weights: list[int | float], active: list[bool] | None = None
) -> list[int]:
    budget = _integer(budget, "budget")
    floors = _integer_vector(filled, "filled floors")
    activity = _active_vector(active, len(floors), allow_none=True)
    proportions = _weights(weights, len(floors))
    if any((floor and (not on) for floor, on in zip(floors, activity))):
        raise ValueError("inactive slot has filled employment")
    occupied = sum(floors)
    if occupied > budget:
        raise BudgetInfeasible(budget=budget, filled_total=occupied)
    slack = _largest_remainders(budget - occupied, proportions)
    result = [floor + extra for floor, extra in zip(floors, slack)]
    if sum(result) != budget or any((value < floor for value, floor in zip(result, floors))):
        raise ArithmeticError("funded allocation violated its budget or occupied floors")
    return result


def fund_birth(
    targets: list[int], filled: list[int], active: list[bool], slot: int, request: int, budget: int
) -> dict:
    budget = _integer(budget, "budget")
    wanted = _integer(request, "birth request")
    target = _integer_vector(targets, "targets")
    floors = _integer_vector(filled, "filled floors")
    if len(target) != len(floors):
        raise ValueError("target/filled shape mismatch")
    activity = _active_vector(active, len(target), allow_none=False)
    slot = _integer(slot, "birth slot")
    if slot >= len(target):
        raise ValueError("birth slot is outside the funded roster")
    if any((floor and (not on) for floor, on in zip(floors, activity))):
        raise ValueError("inactive slot has filled employment")
    if sum(floors) > budget:
        raise BudgetInfeasible(budget=budget, filled_total=sum(floors))
    if any((floor > quota for floor, quota in zip(floors, target))):
        raise ValueError("filled employment exceeds a donor's current quota")
    if sum(target) != budget:
        raise ValueError("current reserved target stock does not equal funded budget")
    if target[slot] != 0 or floors[slot] != 0:
        raise ValueError("new birth requires a target-zero unoccupied slot")
    donor_slack = [
        quota - floor if on and index != slot else 0
        for index, (quota, floor, on) in enumerate(zip(target, floors, activity))
    ]
    funded = min(wanted, sum(donor_slack))
    debits = _largest_remainders(funded, [Fraction(value) for value in donor_slack])
    result = [quota - debit for quota, debit in zip(target, debits)]
    result[slot] = funded
    donor_debits = {index: debit for index, debit in enumerate(debits) if debit}
    if (
        sum(result) != budget
        or any((value < floor for value, floor in zip(result, floors)))
        or any(
            (
                result[index] != target[index]
                for index, on in enumerate(activity)
                if not on and index != slot
            )
        )
    ):
        raise ArithmeticError("birth funding violated a reservation or occupied floor")
    return {
        "targets": result,
        "funded_target": funded,
        "requested_target": wanted,
        "shortfall": wanted - funded,
        "donor_debits": donor_debits,
    }
