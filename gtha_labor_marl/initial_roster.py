from collections import Counter
import hashlib
import math


def require(value, message):
    if not value:
        raise ValueError(message)


def rank(identifier):
    return hashlib.sha256(("scale_contract_20261102|" + str(identifier)).encode()).hexdigest()


def allocate(total, weights, lower=None, upper=None):
    keys = sorted(weights, key=str)
    if type(total) is not int or total < 0 or (not keys):
        raise ValueError("Invalid total or empty weights")
    if any((not math.isfinite(weights[k]) or weights[k] < 0 for k in keys)):
        raise ValueError("Invalid weight")
    result = {k: (lower or {}).get(k, 0) for k in keys}
    limits = {k: (upper or {}).get(k, total) for k in keys}
    if any((type(result[k]) is not int or result[k] < 0 or result[k] > limits[k] for k in keys)):
        raise ValueError("Invalid support bound")
    remaining = total - sum(result.values())
    if remaining < 0 or sum(limits.values()) < total:
        raise ValueError("Infeasible bounded allocation")
    while remaining:
        available = [k for k in keys if result[k] < limits[k]]
        denominator = sum((weights[k] for k in available))
        if not available or denominator <= 0:
            raise ValueError("No positive weight for remaining allocation")
        exact = {k: remaining * weights[k] / denominator for k in available}
        additions = {k: min(math.floor(exact[k]), limits[k] - result[k]) for k in available}
        for k, value in additions.items():
            result[k] += value
        remaining -= sum(additions.values())
        if remaining:
            order = sorted(
                (k for k in available if result[k] < limits[k]),
                key=lambda k: (-(exact[k] - math.floor(exact[k])), str(k)),
            )
            for k in order:
                if not remaining:
                    break
                result[k] += 1
                remaining -= 1
    if sum(result.values()) != total:
        raise AssertionError("Allocation did not conserve total")
    return result


def selected_roster(profiles, rule_for, count, apportion, rank):
    require(count > 0, "Firm count must be positive")
    all_firms = [firm for industry in sorted(profiles) for firm in profiles[industry]]
    selected, support = ({}, set())
    for firm in sorted(all_firms, key=lambda item: item["source_firm_id"]):
        cell = (firm["industry"], firm["cma"], rule_for(firm["naics6"], firm["industry"]))
        if cell not in support:
            selected[firm["source_firm_id"]] = firm
            support.add(cell)
    lower = Counter((firm["industry"] for firm in selected.values()))
    for industry, values in sorted(profiles.items()):
        for firm in values:
            if lower[industry] >= 2:
                break
            if firm["source_firm_id"] not in selected:
                selected[firm["source_firm_id"]] = firm
                lower[industry] += 1
    support_count = len(selected)
    require(support_count <= count, "Positive source-cell support exceeds selected scale")
    quotas = apportion(
        count,
        {i: len(v) for i, v in profiles.items()},
        dict(lower),
        {i: len(v) for i, v in profiles.items()},
    )
    for industry, values in sorted(profiles.items()):
        for firm in sorted(
            values, key=lambda item: (rank(item["source_firm_id"]), item["source_firm_id"])
        ):
            if lower[industry] == quotas[industry]:
                break
            if firm["source_firm_id"] not in selected:
                selected[firm["source_firm_id"]] = firm
                lower[industry] += 1
    require(len(selected) == count, "Distinct source firms cannot supply requested scale")
    return (sorted(selected.values(), key=lambda item: item["source_firm_id"]), support_count)
