import math
import statistics

SCHEMA = "checkpoint_panel_monthly_aggregates_v1"
AGE_BANDS = ("15-24", "25-34", "35-44", "45-54", "55-64", "65+")
STATUS = ("Employed", "Unemployed", "NILF")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, label, nonnegative=False):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    value = float(value)
    require(math.isfinite(value), label + " nonfinite")
    require(not nonnegative or value >= 0, label + " negative")
    return value


def band(age):
    return (
        "under15"
        if age < 15
        else (
            "15-24"
            if age < 25
            else (
                "25-34"
                if age < 35
                else (
                    "35-44" if age < 45 else "45-54" if age < 55 else "55-64" if age < 65 else "65+"
                )
            )
        )
    )


def native(worker):
    value = getattr(worker, "archetype_id", None)
    try:
        return value is None or int(value) in (0, 1, 2)
    except (ValueError, TypeError):
        return True


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    i = (len(values) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def moments(values):
    return {
        "n": len(values),
        "sum": math.fsum(values),
        "sum_squares": math.fsum((v * v for v in values)),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "std_population": statistics.pstdev(values) if values else None,
        "p95": quantile(values, 0.95),
    }


def age_cell():
    return {
        "P": 0,
        "E": 0,
        "U_raw": 0,
        "N_raw": 0,
        "U_active": 0,
        "N_aligned": 0,
        "LF_raw": 0,
        "LF_aligned": 0,
        "unemployment_duration_months_sum": 0.0,
        "unemployment_duration_raw_U_n": 0,
        "unemployment_duration_active_months_sum": 0.0,
        "unemployment_duration_active_U_n": 0,
    }


def flow_cell():
    return {
        "opening_E": 0,
        "opening_U_raw": 0,
        "opening_U_active": 0,
        "opening_N_raw": 0,
        "endpoint_E_to_U": 0,
        "endpoint_E_to_N": 0,
        "endpoint_U_to_E": 0,
        "endpoint_U_to_N": 0,
        "endpoint_active_U_to_E": 0,
        "endpoint_active_U_to_N": 0,
        "endpoint_N_to_E": 0,
        "endpoint_N_to_U": 0,
        "opening_person_exits": 0,
        "closing_person_entrants": 0,
    }


def vector(env, name):
    value = getattr(env, name, None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def snapshot(workers):
    result = {}
    for key, ws in workers.items():
        w = ws.worker
        age = number(getattr(w, "age", None), "age", True)
        require(age is not None, "Worker age absent")
        status = getattr(w, "employment_status", None)
        require(status in STATUS, "Unknown worker status")
        result[key] = {
            "age": age,
            "status": status,
            "searching": bool(getattr(w, "is_searching", False)),
            "native": native(w),
            "da": getattr(w, "da_id", None),
        }
    return result


class MonthlyAggregates:
    schema = SCHEMA

    def __init__(self):
        self.months = []
        self._opening = None
        self._spatial = {}
        self._year = None

    def before(self, env, action, month_index, calendar):
        require(self._opening is None, "Previous month observer did not close")
        require(month_index == len(self.months), "Observer month sequence differs")
        self._opening = snapshot(env.state_manager.workers)
        self._identity = (int(month_index), tuple((int(v) for v in calendar)))

    def after(self, env, month_index, calendar):
        identity = (int(month_index), tuple((int(v) for v in calendar)))
        require(
            self._opening is not None and identity == self._identity,
            "Observer before/after identity mismatch",
        )
        year, month = identity[1]
        workers = env.state_manager.workers
        closing = snapshot(workers)
        all_age = {label: age_cell() for label in ("all", *AGE_BANDS, "under15")}
        native_age = {label: age_cell() for label in ("all15_99", *AGE_BANDS[:5], "65-99")}
        flows = {label: flow_cell() for label in ("all", *AGE_BANDS, "under15")}
        unknown_duration = 0
        stock_commutes = []
        industry_wages = {}
        da = {}
        industry_firms = {}

        def da_cell(da_id):
            key = "UNKNOWN" if da_id is None else str(da_id)
            return da.setdefault(
                key,
                {
                    "da_id": None if da_id is None else str(da_id),
                    "P": 0,
                    "E": 0,
                    "U_raw": 0,
                    "N_raw": 0,
                    "U_active": 0,
                    "resident_commute_minutes_sum": 0.0,
                    "resident_commute_n": 0,
                    "workplace_resident_filled": 0,
                    "opening_U_raw": 0,
                    "endpoint_U_to_E": 0,
                    "endpoint_U_to_N": 0,
                },
            )

        for key, ws in workers.items():
            w = ws.worker
            state = closing[key]
            label = band(state["age"])
            status = state["status"]
            search = state["searching"]
            duration = None
            if status == "Unemployed":
                duration = number(
                    getattr(w, "unemployment_months", None), "unemployment months", True
                )
                if duration is None:
                    unknown_duration += 1
            groups = [all_age["all"], all_age[label]]
            if state["native"] and 15 <= state["age"] < 100:
                groups.extend(
                    [native_age["all15_99"], native_age["65-99" if state["age"] >= 65 else label]]
                )
            for cell in groups:
                cell["P"] += 1
                cell[{"Employed": "E", "Unemployed": "U_raw", "NILF": "N_raw"}[status]] += 1
                cell["U_active"] += int(status == "Unemployed" and search)
                cell["N_aligned"] += int(
                    status == "NILF" or (status == "Unemployed" and (not search))
                )
                cell["LF_raw"] += int(status != "NILF")
                cell["LF_aligned"] += int(
                    status == "Employed" or (status == "Unemployed" and search)
                )
                if duration is not None:
                    cell["unemployment_duration_months_sum"] += duration
                    cell["unemployment_duration_raw_U_n"] += 1
                    if search:
                        cell["unemployment_duration_active_months_sum"] += duration
                        cell["unemployment_duration_active_U_n"] += 1
            spatial = da_cell(state["da"])
            spatial["P"] += 1
            spatial[{"Employed": "E", "Unemployed": "U_raw", "NILF": "N_raw"}[status]] += 1
            spatial["U_active"] += int(status == "Unemployed" and search)
            if status == "Employed":
                employer = getattr(w, "current_employer_id", None)
                firm_state = env.state_manager.firms.get(employer)
                code = (
                    getattr(firm_state.firm, "industry", None) if firm_state is not None else None
                )
                code = "UNKNOWN" if code is None else str(code)
                item = industry_wages.setdefault(
                    code,
                    {
                        "industry": code,
                        "employed_n": 0,
                        "positive_wage_n": 0,
                        "missing_or_nonpositive_wage_n": 0,
                        "_values": [],
                    },
                )
                item["employed_n"] += 1
                wage = number(getattr(w, "current_wage", None), "hourly wage")
                if wage is not None and wage > 0:
                    item["_values"].append(wage)
                    item["positive_wage_n"] += 1
                else:
                    item["missing_or_nonpositive_wage_n"] += 1
                commute = number(getattr(w, "commute_time_min", None), "commute minutes", True)
                if commute is not None:
                    stock_commutes.append(commute)
                    spatial["resident_commute_minutes_sum"] += commute
                    spatial["resident_commute_n"] += 1
                workplace = (
                    getattr(firm_state.firm, "da_id", None) if firm_state is not None else None
                )
                da_cell(workplace)["workplace_resident_filled"] += 1
        for key, opening in self._opening.items():
            label = band(opening["age"])
            target = closing.get(key)
            old = opening["status"]
            new = None if target is None else target["status"]
            for cell in (flows["all"], flows[label]):
                cell[
                    {
                        "Employed": "opening_E",
                        "Unemployed": "opening_U_raw",
                        "NILF": "opening_N_raw",
                    }[old]
                ] += 1
                cell["opening_U_active"] += int(old == "Unemployed" and opening["searching"])
                if target is None:
                    cell["opening_person_exits"] += 1
                    continue
                route = {
                    ("Employed", "Unemployed"): "endpoint_E_to_U",
                    ("Employed", "NILF"): "endpoint_E_to_N",
                    ("Unemployed", "Employed"): "endpoint_U_to_E",
                    ("Unemployed", "NILF"): "endpoint_U_to_N",
                    ("NILF", "Employed"): "endpoint_N_to_E",
                    ("NILF", "Unemployed"): "endpoint_N_to_U",
                }.get((old, new))
                if route:
                    cell[route] += 1
                if old == "Unemployed" and opening["searching"] and (new in ("Employed", "NILF")):
                    cell["endpoint_active_U_to_" + ("E" if new == "Employed" else "N")] += 1
            if old == "Unemployed":
                cell = da_cell(opening["da"])
                cell["opening_U_raw"] += 1
                if new == "Employed":
                    cell["endpoint_U_to_E"] += 1
                elif new == "NILF":
                    cell["endpoint_U_to_N"] += 1
        for key, state in closing.items():
            if key not in self._opening:
                for cell in (flows["all"], flows[band(state["age"])]):
                    cell["closing_person_entrants"] += 1
        for item in industry_wages.values():
            item["positive_hourly_wage"] = moments(item.pop("_values"))
        active = vector(env, "firm_active")
        industries = vector(env, "firm_industries")
        employment = vector(env, "firm_employment")
        targets = vector(env, "firm_target_size")
        posted = vector(env, "firm_posted_wages")
        firm_ids = getattr(env, "firm_ids", None)
        profits = getattr(env, "last_firm_profit_raw", None)
        if active is not None:
            require(
                all(
                    (
                        v is not None and len(v) == len(active)
                        for v in (industries, employment, targets, posted)
                    )
                ),
                "Firm tensor coverage incomplete",
            )
            for i, live in enumerate(active):
                code = str(industries[i])
                item = industry_firms.setdefault(
                    code,
                    {
                        "industry": code,
                        "slots": 0,
                        "active_firms": 0,
                        "employee_claims": 0,
                        "raw_target_positions": 0,
                        "_posted": [],
                        "_profits": [],
                        "missing_profit_n": 0,
                    },
                )
                item["slots"] += 1
                if not live:
                    continue
                item["active_firms"] += 1
                item["employee_claims"] += int(employment[i])
                item["raw_target_positions"] += int(targets[i])
                item["_posted"].append(number(posted[i], "posted wage"))
                fid = firm_ids[i] if firm_ids is not None and i < len(firm_ids) else None
                profit = (
                    None
                    if profits is None or fid not in profits
                    else number(profits[fid], "raw firm profit")
                )
                if profit is None:
                    item["missing_profit_n"] += 1
                else:
                    item["_profits"].append(profit)
            for item in industry_firms.values():
                item["posted_hourly_wage"] = moments(item.pop("_posted"))
                item["raw_profit_prelog"] = moments(item.pop("_profits"))
                item["raw_profit_coverage"] = (
                    "COMPLETE" if item["missing_profit_n"] == 0 else "PARTIAL_OR_MISSING_NOT_ZERO"
                )
                if item["active_firms"] and item["raw_profit_prelog"]["n"] == 0:
                    item["raw_profit_prelog"]["sum"] = None
                    item["raw_profit_prelog"]["sum_squares"] = None
        matcher = getattr(env, "matching_mechanism", None)
        actual_mode = getattr(matcher, "_use_travel_time", None)
        matrix = getattr(matcher, "da_travel_time_matrix", None)
        require(
            actual_mode is None or isinstance(actual_mode, bool), "Actual matching mode not boolean"
        )
        stats = getattr(env, "last_matching_stats", None)
        flow_values = None if not isinstance(stats, dict) else stats.get("commute_values")
        flow_moments = (
            None
            if flow_values is None
            else moments([number(v, "match commute", True) for v in flow_values])
        )
        if self._year is None:
            self._year = year
        require(year == self._year, "Spatial year changed before December close")
        for key, cell in da.items():
            dest = self._spatial.setdefault(
                key,
                {
                    "da_id": cell["da_id"],
                    "months_with_resident_or_workplace_exposure": 0,
                    **{k: 0 for k in cell if k != "da_id"},
                },
            )
            dest["months_with_resident_or_workplace_exposure"] += 1
            for name, value in cell.items():
                if name != "da_id":
                    dest[name] += value
        spatial_annual = None
        if month == 12:
            spatial_annual = [
                {"year": year, **self._spatial[key], "December": da.get(key)}
                for key in sorted(self._spatial)
            ]
            self._spatial = {}
            self._year = None
        result = {
            "month_index": month_index,
            "calendar": [year, month],
            "timing": "Observer immediately around native env._step; after state is postmarket/pre-external-renewal. Native clock placement remains declared by runner.",
            "age_resident_raw_and_aligned": all_age,
            "age_native_15_99_raw_and_aligned": native_age,
            "endpoint_flows_by_opening_age": flows,
            "duration_missing_raw_U_n": unknown_duration,
            "duration_unit": "unemployment_months; multiply by4.33 exactly once when reporting weeks",
            "industry_worker_wages": list(industry_wages.values()),
            "industry_firms": list(industry_firms.values()) if active is not None else None,
            "commute": {
                "stock_physical_unit": "min",
                "stock_full_positive_and_zero_minutes": moments(stock_commutes),
                "actual_matcher_use_travel_time": actual_mode,
                "actual_travel_time_matrix_present": matrix is not None,
                "flow_unit": (
                    "min" if actual_mode is True else "km" if actual_mode is False else None
                ),
                "flow": flow_moments,
                "native_last_matching_use_travel_time": (
                    stats.get("use_travel_time") if isinstance(stats, dict) else None
                ),
                "native_last_matching_commute_unit": (
                    stats.get("commute_unit") if isinstance(stats, dict) else None
                ),
            },
            "spatial_annual": spatial_annual,
            "coverage": {
                "resident_worker_ids_opening": len(self._opening),
                "resident_worker_ids_closing": len(closing),
                "home_DA_unknown_P": da.get("UNKNOWN", {}).get("P", 0),
                "positive_wage_missing_or_nonpositive_E": sum(
                    (r["missing_or_nonpositive_wage_n"] for r in industry_wages.values())
                ),
                "employed_unknown_employer_industry": industry_wages.get("UNKNOWN", {}).get(
                    "employed_n", 0
                ),
            },
            "observer_actor_calls": 0,
            "observer_RNG_calls": 0,
        }
        require(
            all_age["all"]["P"] == sum((all_age[k]["P"] for k in (*AGE_BANDS, "under15"))),
            "Age count partition fails",
        )
        require(
            all_age["all"]["P"]
            == all_age["all"]["E"] + all_age["all"]["U_raw"] + all_age["all"]["N_raw"],
            "Closing status partition fails",
        )
        require(
            sum((r["employed_n"] for r in industry_wages.values())) == all_age["all"]["E"],
            "Industry employment partition fails",
        )
        self.months.append(result)
        self._opening = None
        return result

    def validate(self, expected_months=None):
        require(self._opening is None, "The observer has an unclosed step")
        if expected_months is not None:
            require(len(self.months) == expected_months, "Observer monthly coverage differs")
        require(
            [r["month_index"] for r in self.months] == list(range(len(self.months))),
            "Observer months are not contiguous",
        )
        return {
            "schema": SCHEMA,
            "months": len(self.months),
            "complete": True,
            "no_actor_or_RNG_calls": True,
        }

    def payload(self):
        return {
            "schema": SCHEMA,
            "months": self.months,
            "validation": self.validate(),
            "limits": [
                "Endpoint flows use exact opening statuses/ages and same-ID poststep endpoints; they are not independent mechanism counters and do not replace native reconciliation.",
                "Current unemployed-stock durations retain months and count denominators; reported weeks require one4.33 conversion.",
                "Industry wages are positive-employed person-weighted moments by current employer industry; raw profit is observed prelog reward telemetry, missing values remain absent.",
                "Spatial annual rows sum monthly exposures and separately preserve December stocks; DAs without observed exposure do not acquire fabricated rates.",
                "The observer has no actor calls, random draws or economic-state writes; primary and auxiliary observer scope is controlled by the runner.",
            ],
        }
