from collections import Counter
import gzip
import json
import re

BIRTH_ID = re.compile("r6x_cohort_(\\d{4})_(532|535|537)_(\\d+)_(\\d+)")
STATUS = {"Unemployed": "U", "Employed": "E", "NILF": "N"}
ENTRANT_FIELDS = (
    "seed",
    "history_index",
    "entry_year",
    "age",
    "cma",
    "initial_status",
    "initial_searching",
    "first_observed_employment_month",
    "registry_exit_month",
    "administrative_followup_months",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def states(mapping):
    return {
        wid: (STATUS[state["status"]], state["searching"], state["age"], state["cma"])
        for wid, state in mapping.items()
    }


def flow_rows(src, dst, keys):
    counts = [[[0, 0, 0] for _ in range(2)] for _ in range(2)]
    for wid, (status, searching, age, _) in src.items():
        if status != "U" or age < 15:
            continue
        dest = dst.get(wid)
        linked = dest is not None
        employed = linked and dest[0] == "E"
        for scope in range(2 if 16 <= age <= 69 else 1):
            for definition in range(2 if searching else 1):
                cell = counts[scope][definition]
                cell[0] += 1
                cell[1] += linked
                cell[2] += employed
    for i, age_scope in enumerate(("15plus", "16to69")):
        for j, definition in enumerate(("raw", "active")):
            yield (*keys, age_scope, definition, *counts[i][j])


def stock_rows(mapping, entrants, keys):
    groups = [[0] * 5 for _ in range(3)]
    for wid, (status, searching, _, _) in mapping.items():
        ix = (
            1
            if wid in entrants and entrants[wid]["initial_status"] == "U"
            else 2 if wid in entrants else 0
        )
        row = groups[ix]
        row[0] += 1
        row[1] += status == "E"
        row[2] += status == "U"
        row[3] += status == "U" and searching
        row[4] += status == "N"
    for group, row in zip(
        ("initial_residents", "renewal_entered_U", "renewal_entered_other"), groups
    ):
        require(row[1] + row[2] + row[4] == row[0], str(keys))
        yield (*keys, group, *row)


def history_tables(source, seed, history, horizon=216):
    flows, stocks, entrants = ([], [], {})
    previous_close, previous_open, previous_period = (None, None, None)
    initial_ids = set()
    disappeared = set()
    checks = Counter()
    n = 0
    with gzip.open(source, "rt", encoding="utf8") as months:
        for t, line in enumerate(months):
            row = json.loads(line)
            y, m, index = row["period"]
            require(
                row["schema"] == "current_native_forward_source_market_month_v1"
                and index == t
                and ((y, m) == (2006 + t // 12, 1 + t % 12))
                and (index < horizon),
                str(source),
            )
            opening, closing = (states(row["opening"]), states(row["after_market"]))
            require(
                opening.keys() == closing.keys(),
                f"Intra-market registry change at {str(source)} {y}-{m}",
            )
            if t == 0:
                initial_ids = set(opening)
            else:
                entrants_now = opening.keys() - previous_close.keys()
                exits = previous_close.keys() - opening.keys()
                require(not entrants_now or m == 7, str(source))
                require(not entrants_now & (disappeared | initial_ids), str(source))
                for wid in exits:
                    disappeared.add(wid)
                    if wid in entrants:
                        record = entrants[wid]
                        record["registry_exit_month"] = t - record["t"] + 1
                for wid in entrants_now:
                    st, searching, age, cma = opening[wid]
                    match = BIRTH_ID.fullmatch(wid)
                    require(
                        match and (int(match[1]), int(match[2]), int(match[3])) == (y, cma, age),
                        wid,
                    )
                    require(wid not in entrants and st in {"E", "U", "N"}, wid)
                    entrants[wid] = {
                        "entry_year": y,
                        "t": t,
                        "age": age,
                        "cma": cma,
                        "initial_status": st,
                        "initial_searching": bool(searching),
                        "first_observed_employment_month": 0 if st == "E" else None,
                        "registry_exit_month": None,
                        "administrative_followup_months": horizon - t,
                    }
                keys = (seed, history, *previous_period, y, m)
                flows.extend(flow_rows(previous_open, opening, (*keys, "opening_to_next_opening")))
                flows.extend(
                    flow_rows(previous_close, closing, (*keys, "after_market_to_next_after_market"))
                )
                checks["reconciled_boundaries"] += 1
                checks["boundary_entrants"] += len(entrants_now)
                if m != 1:
                    checks["age_disagreements_within_year"] += sum(
                        (
                            opening[w][2] != previous_close[w][2]
                            for w in opening.keys() & previous_close.keys()
                        )
                    )
            keys = (seed, history, y, m)
            flows.extend(flow_rows(opening, closing, (*keys, y, m, "opening_to_after_market")))
            stocks.extend(stock_rows(opening, entrants, (*keys, "opening")))
            stocks.extend(stock_rows(closing, entrants, (*keys, "after_market")))
            for wid, (status, _, _, _) in closing.items():
                if wid not in entrants:
                    continue
                record = entrants[wid]
                if record["first_observed_employment_month"] is None:
                    if opening[wid][0] == "E":
                        record["first_observed_employment_month"] = t - record["t"]
                    elif status == "E":
                        record["first_observed_employment_month"] = t - record["t"] + 1
            previous_period, previous_open, previous_close = ((y, m), opening, closing)
            n += 1
    require(n == horizon and len(entrants) == checks["boundary_entrants"], str(source))
    return (
        flows,
        stocks,
        [(seed, history, *(row[col] for col in ENTRANT_FIELDS[2:])) for row in entrants.values()],
    )


def holdout_trajectories(source, outcomes, seed, history):
    outcome = json.loads(outcomes.read_text(encoding="utf8"))
    if sorted(outcome["benchmark_cmas"]) != [535, 537]:
        raise ValueError("Expected Toronto and Hamilton census metropolitan areas")
    initial = set()
    births = {}
    records = []
    detail = []
    ever_employed = set()
    ever_left_active_u = set()
    with gzip.open(source, "rb") as f:
        for t, line in enumerate(f):
            if t >= 168:
                break
            head, sep, _ = line.partition(b',"routes":')
            row = json.loads(head + b"}" if sep and b'"after_market":' in head else line)
            y, m, _ = row["period"]
            if t == 0:
                initial = set(row["opening"])
            elif m == 7:
                for wid, state in row["opening"].items():
                    if wid not in initial and wid not in births:
                        assert BIRTH_ID.fullmatch(wid)
                        births[wid] = (
                            "entered_U" if state["status"] == "Unemployed" else "entered_other"
                        )
            comparator = outcome["months"][t]["after_market"]["aggregate"]
            for observation in ("opening", "after_market"):
                for wid, state in row[observation].items():
                    if births.get(wid) != "entered_U":
                        continue
                    if state["status"] == "Employed":
                        ever_employed.add(wid)
                    if not (state["status"] == "Unemployed" and state["searching"]):
                        ever_left_active_u.add(wid)
            counts = {k: [0, 0, 0] for k in ["initial", "entered_U", "entered_other"]}
            split = {
                k: 0
                for k in [
                    "continuous_observed_active_U",
                    "left_active_U_without_employment",
                    "previously_employed",
                ]
            }
            for wid, state in row["after_market"].items():
                if state["age"] < 15 or state["cma"] not in (535, 537):
                    continue
                cat = "initial" if wid in initial else births[wid]
                status = state["status"]
                is_u = state["covid_state"] in ("temporary_layoff", "temp_layoff") or (
                    status == "Unemployed" and state["searching"]
                )
                c = counts[cat]
                c[0] += 1
                c[1] += not is_u and status == "Employed"
                c[2] += is_u
                if cat == "entered_U" and is_u:
                    split[
                        (
                            "previously_employed"
                            if wid in ever_employed
                            else (
                                "left_active_U_without_employment"
                                if wid in ever_left_active_u
                                else "continuous_observed_active_U"
                            )
                        )
                    ] += 1
            totals = [sum((k[i] for k in counts.values())) for i in range(3)]
            assert totals == [comparator["population"], comparator["E"], comparator["U_lfs"]], (
                source,
                t,
                totals,
                comparator,
            )
            assert sum(split.values()) == counts["entered_U"][2]
            if y >= 2017:
                for name, count in split.items():
                    detail.append(
                        (
                            seed,
                            history,
                            y,
                            m,
                            name,
                            count,
                            100 * count / sum(totals[1:]) if sum(totals[1:]) else None,
                            100 * count / totals[2] if totals[2] else None,
                            (
                                100 * count / counts["entered_U"][2]
                                if counts["entered_U"][2]
                                else None
                            ),
                        )
                    )
            for cat, (p, e, u) in counts.items():
                all_lf = sum(totals[1:])
                records.append(
                    (
                        seed,
                        history,
                        y,
                        m,
                        cat,
                        p,
                        e,
                        u,
                        100 * u / all_lf if all_lf else None,
                        100 * u / totals[2] if totals[2] else None,
                        100 * totals[2] / all_lf if all_lf else None,
                        100 * u / (e + u) if e + u else None,
                    )
                )
    return (records, detail)


def first_job_counts(entrants):
    counts = {"all_entries": 0, "entered_unemployed": 0, "entered_searching_unemployed": 0}
    counts.update(
        {
            f"{name}_{month}": 0
            for month in (1, 3, 6, 12)
            for name in ("ever_employed", "exited_before_job")
        }
    )
    for row in entrants:
        person = dict(zip(ENTRANT_FIELDS, row))
        if not 2007 <= person["entry_year"] <= 2018:
            continue
        counts["all_entries"] += 1
        if person["initial_status"] != "U":
            continue
        counts["entered_unemployed"] += 1
        counts["entered_searching_unemployed"] += person["initial_searching"]
        if person["administrative_followup_months"] < 12:
            raise ValueError("Incomplete July entrant cohort follow-up")
        job, exit_month = (person["first_observed_employment_month"], person["registry_exit_month"])
        if job is not None and exit_month is not None and (job >= exit_month):
            raise ValueError("Employment follows registry exit")
        for month in (1, 3, 6, 12):
            employed = job is not None and job <= month
            counts[f"ever_employed_{month}"] += employed
            counts[f"exited_before_job_{month}"] += (
                not employed and exit_month is not None and (exit_month <= month)
            )
    return counts
