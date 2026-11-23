from collections import defaultdict
from statistics import fmean

DATES = {(year, month) for year in range(2006, 2012) for month in range(2, 13)}


def prime_age(months):
    dates = set()
    opening = hires = 0
    for record in months:
        date = tuple(record["calendar"])
        if date not in DATES:
            continue
        if date in dates:
            raise ValueError("Duplicate prime-age job-finding month")
        dates.add(date)
        for band in ("25-34", "35-44", "45-54"):
            cell = record["endpoint_flows_by_opening_age"][band]
            origins, transitions = (cell["opening_U_raw"], cell["endpoint_U_to_E"])
            if not 0 <= transitions <= origins:
                raise ValueError("Invalid prime-age transition counts")
            opening += origins
            hires += transitions
    if dates != DATES or not opening:
        raise ValueError("Incomplete prime-age job-finding series")
    return 100 * hires / opening


def linked_counts(flows):
    counts = defaultdict(lambda: [0, 0, 0])
    for (
        _,
        _,
        year,
        month,
        end_year,
        end_month,
        clock,
        ages,
        definition,
        origins,
        linked,
        hired,
    ) in flows:
        if clock != "after_market_to_next_after_market" or definition != "raw":
            continue
        if not 0 <= hired <= linked <= origins:
            raise ValueError("Invalid linked job-finding exposures")
        if ages == "16to69" and 2006 <= year <= 2011 and (month <= 11):
            labels = [("annual_16_69", str(year)), ("annual_16_69", "2006-2011")]
        elif ages == "15plus" and end_year in (2007, 2009) and (7 <= end_month <= 12):
            labels = [("monthly_15plus", f"{end_year}-{end_month:02d}")]
        else:
            continue
        for label in labels:
            v = counts[label]
            v[0] += origins
            v[1] += linked
            v[2] += hired
    return counts


def linked_rates(per_history, policy_seeds):
    rows = []
    for seed in policy_seeds:
        blocks = [per_history[seed, h] for h in range(50)]
        periods = set(blocks[0])
        if any((set(block) != periods for block in blocks)):
            raise ValueError("Unmatched periods in consecutive-month job finding")
        for period in sorted(periods):
            counts = [block[period] for block in blocks]
            origins, linked, hired = map(sum, zip(*counts))
            if not linked:
                raise ValueError("No linked unemployed workers for a policy-period")
            rows.append(
                dict(
                    seed=seed,
                    series=period[0],
                    period=period[1],
                    origins=origins,
                    linked=linked,
                    moved=hired,
                    rate_pct=100 * hired / linked,
                    equal_history_pct=fmean((100 * c[2] / c[1] for c in counts if c[1])),
                    histories_with_linked=sum((c[1] > 0 for c in counts)),
                )
            )
        for year in (2007, 2009):
            months = [
                r["rate_pct"]
                for r in rows
                if r["seed"] == seed
                and r["series"] == "monthly_15plus"
                and r["period"].startswith(str(year))
            ]
            if len(months) != 6:
                raise ValueError("Six monthly rates required for December recession comparison")
            rows.append(
                dict(
                    seed=seed,
                    series="trailing_six_month_15plus",
                    period=f"{year}-12",
                    origins=None,
                    linked=None,
                    moved=None,
                    rate_pct=fmean(months),
                    equal_history_pct=None,
                    histories_with_linked=None,
                )
            )
    return rows
