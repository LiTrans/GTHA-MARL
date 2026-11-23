import argparse
from copy import copy
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from . import offers
from .acceptance import require
from .monthly import MonthlyAggregates
from .tariff import SCENARIOS


def dump(path, value):

    def scalar(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(type(item).__name__)

    def finite(item):
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(x) for x in item]
        if isinstance(item, float) and (not math.isfinite(item)):
            return None
        if hasattr(item, "tolist"):
            return finite(item.tolist())
        return item

    with path.open("w", encoding="utf8") as stream:
        json.dump(finite(value), stream, default=scalar, allow_nan=False)


def write_monthly(output, aggregate, seed, index):
    with (output / "monthly.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "policy_seed",
                "replication_index",
                "evaluation_seed",
                "year",
                "month",
                "employment_rate",
                "participation_rate",
                "unemployment_rate",
                "search_participation_rate",
                "search_unemployment_rate",
                "population",
                "employed",
                "unemployed",
                "nilf",
            )
        )
        for month in aggregate.months:
            cells = month["age_resident_raw_and_aligned"]["all"]

            def rate(n, d):
                return n / d if d else None

            writer.writerow(
                (
                    seed,
                    index,
                    seed * 100 + index,
                    *month["calendar"],
                    rate(cells["E"], cells["P"]),
                    rate(cells["LF_raw"], cells["P"]),
                    rate(cells["U_raw"], cells["LF_raw"]),
                    rate(cells["LF_aligned"], cells["P"]),
                    rate(cells["U_active"], cells["LF_aligned"]),
                    cells["P"],
                    cells["E"],
                    cells["U_raw"],
                    cells["N_raw"],
                )
            )
    dump(output / "monthly.json", aggregate.payload())


def validate_intervention(evaluator, checkpoint, family, workers, firms):
    verify = evaluator.assert_eval_config_matches_manifest
    snapshot = json.loads((checkpoint.parent / "manifest.json").read_text(encoding="utf-8-sig"))[
        "config_snapshot"
    ]
    expected = {}
    if family == "untrained_random":
        expected["training.freeze_firm_policy"] = (True, False)
    elif family == "firm_fixed":
        expected["training.freeze_firm_policy"] = (False, True)
    elif family == "bundle_off":
        expected.update(
            {
                "training.enable_reservation_wage_bundle": (True, False),
                "training.reservation_decay_per_year": (0.075, 0.05),
                "training.reservation_decay_compounded": (True, False),
            }
        )
    size = (snapshot["simulation"]["n_workers"], snapshot["simulation"]["n_firms"])
    if (workers, firms) != size:
        require(
            family in ("historical", "spatial", "worker_groups")
            and (2 * workers, 2 * firms) == (5 * size[0], 5 * size[1]),
            f"This checkpoint was trained with {size[0]} workers and {size[1]} firm slots. Pass --workers and --firms with those values. The historical and spatial designs also accept 2.5 times both.",
        )
        expected.update(
            {"simulation.n_workers": (size[0], workers), "simulation.n_firms": (size[1], firms)}
        )

    def check(config, path):
        require(Path(path).resolve() == checkpoint, "Unexpected evaluation checkpoint")
        reference = copy(config)
        for block in {k.split(".")[0] for k in expected}:
            setattr(reference, block, copy(getattr(config, block)))
        for name, (source, destination) in expected.items():
            block, field = name.split(".")
            require(
                evaluator._values_equal(snapshot[block][field], source)
                and evaluator._values_equal(getattr(getattr(config, block), field), destination),
                f"Unexpected checkpoint or evaluation value for {name}",
            )
            setattr(getattr(reference, block), field, source)
        return verify(reference, path)

    evaluator.assert_eval_config_matches_manifest = check
    return expected


def parse_args():
    parser = argparse.ArgumentParser(description="Run one paired evaluation history.")
    parser.add_argument(
        "family",
        choices=(
            "historical",
            "untrained_random",
            "untrained_heuristic",
            "firm_fixed",
            "bundle_off",
            "acceptance",
            "forward",
            "tariff",
            "spatial",
            "worker_groups",
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("METRO_DATA_ROOT", ".")))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history", type=int, choices=range(50), default=0, metavar="0..49")
    parser.add_argument("--workers", type=int, default=1000)
    parser.add_argument("--firms", type=int, default=80)
    parser.add_argument(
        "--months",
        type=int,
        help="Number of months to simulate, in whole years (historical families)",
    )
    parser.add_argument("--arm", help="Forward arm, acceptance arm or tariff scenario")
    parser.add_argument("--fit", type=Path, help="Fit file for the recentred acceptance arm")
    parser.add_argument("--covid-driver", type=Path)
    parser.add_argument("--tariff-driver", type=Path)
    parser.add_argument("--exposures", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    for key in (
        "data",
        "output",
        "checkpoint",
        "fit",
        "covid_driver",
        "tariff_driver",
        "exposures",
    ):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, path.resolve())
    forward = args.family in ("forward", "tariff")
    acceptance = args.family == "acceptance"
    arm = args.arm or {"forward": "RZ_ON", "acceptance": "baseline", "tariff": "control"}.get(
        args.family
    )
    modes = {
        "FN_ON": ("frozen_2019", True),
        "FN_OFF": ("frozen_2019", False),
        "RZ_ON": ("realized", True),
        "RZ_OFF": ("realized", False),
    }
    if args.family == "forward":
        require(arm in modes, "Select FN_ON, FN_OFF, RZ_ON or RZ_OFF")
    elif args.family == "tariff":
        require(arm in SCENARIOS, "Unknown tariff scenario")
        require(
            all((p and p.is_file() for p in (args.tariff_driver, args.exposures))),
            "Supply tariff and exposure CSVs",
        )
    elif acceptance:
        require(arm in ("baseline", "recentred"), "Unknown acceptance arm")
        require(
            (arm == "recentred") == bool(args.fit), "Supply a fit only for recentred acceptance"
        )
    else:
        require(arm is None, "This evaluation family takes no --arm")
    if forward:
        require(
            args.covid_driver and args.covid_driver.is_file(),
            "Supply a pandemic schedule for this arm",
        )
    months = {"acceptance": 168, "forward": 216, "tariff": 300, "worker_groups": 12}.get(
        args.family, 132
    )
    if args.months is not None:
        require(
            args.family != "worker_groups" or args.months == 12,
            "The worker-group evaluation runs for 12 months",
        )
        require(
            not forward
            and (not acceptance)
            and (args.months % 12 == 0)
            and (12 <= args.months <= 132),
            "--months applies to whole-year historical evaluations",
        )
        months = args.months
    seed, index = (args.seed, args.history)
    evaluation_seed = 100 * seed + index
    calendar = [(2006 + t // 12, t % 12 + 1) for t in range(months)]
    os.environ["METRO_DATA_ROOT"] = str(args.data.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gtha_labor_marl"))
    import numpy as np
    import torch
    import run_evaluation as evaluator
    import capacity_budget
    from config.default_config import Config
    from curriculum import apply_profile, load_profile
    from market_setup import bind_environment, configure

    args.checkpoint = args.checkpoint.resolve(strict=True)
    with args.checkpoint.open("rb") as ckpt:
        checkpoint_sha256 = hashlib.file_digest(ckpt, "sha256").hexdigest()
    if args.family.startswith("untrained"):
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        require(
            raw.get("update_count") == 0 and raw.get("total_steps") == 0,
            "Supply the initial, untrained checkpoint for this policy seed",
        )
        del raw
    intervention = validate_intervention(
        evaluator, args.checkpoint, args.family, args.workers, args.firms
    )
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    os.chdir(output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(evaluation_seed)
    np.random.seed(evaluation_seed)
    random.seed(evaluation_seed)
    base = evaluator.LaborMarketEnv
    bind_environment(base)
    probe = Config()
    probe.simulation.end_year = calendar[-1][0]
    configure(probe, simulation_seed=evaluation_seed)
    fixed = args.family in ("untrained_heuristic", "firm_fixed")
    km = args.family != "bundle_off"
    kwargs = dict(
        checkpoint_path=str(args.checkpoint),
        n_workers=args.workers,
        n_firms=args.firms,
        end_year=calendar[-1][0],
        max_episode_years=months // 12,
        rollout_reseed=evaluation_seed,
        freeze_firm_policy=fixed,
        enable_reservation_wage_bundle=km,
        net_population_projection=probe.training.renewal_net_projection,
    )
    aggregate = MonthlyAggregates()
    holder = {}
    job = dict(
        policy_seed=seed,
        evaluation_seed=evaluation_seed,
        family=args.family,
        calendar=calendar,
        scenario="RZ_ON" if args.family == "tariff" else arm,
        output=output,
    )
    if forward:
        from . import forward as observer

        base = observer.wrap_environment(base, job)
        native_reconcile = evaluator._reconcile_employment_mechanisms

        def reconcile(flow, counts):
            return holder["env"]._forward_source_observer.reconcile(native_reconcile, flow, counts)

        evaluator._reconcile_employment_mechanisms = reconcile
        mode, enabled = modes[job["scenario"]]
        kwargs.update(
            enable_covid_regime=enabled,
            covid_driver_file=str(args.covid_driver.resolve()),
            standard_driver_mode=mode,
            covid_arm_id=job["scenario"],
            covid_seed=evaluation_seed,
        )
    if args.family == "tariff":
        import age_renewal
        import capacity_lineage
        from environment.trade_war_regime import TradeWarRegime
        from market_mechanisms import ValidatedTradeWarRegime
        from . import tariff_budget, tariff_capacity

        tariff_capacity.install(
            TradeWarRegime,
            evaluator.LaborMarketEnv,
            age_renewal,
            capacity_lineage,
            (ValidatedTradeWarRegime,),
        )
        tariff_budget.install(capacity_budget, capacity_lineage, TradeWarRegime)
        native_aggregate = evaluator.aggregate_year_metrics
        macro_months = []

        def aggregate_year(rows):
            require(
                len(rows) == 12 and len(macro_months) + 12 <= months,
                "Unexpected tariff year in the monthly metrics",
            )
            for row in rows:
                names = (
                    "unemployment_rate",
                    "employment_rate",
                    "participation_rate",
                    "vacancy_rate",
                    "job_finding_rate",
                    "separation_rate",
                )
                rates = {k: float(row[k]) for k in names}
                require(
                    all((math.isfinite(v) for v in rates.values())), "Undefined tariff monthly rate"
                )
                macro_months.append(
                    dict(
                        calendar=calendar[len(macro_months)],
                        native_reported_year=int(row["year"]),
                        native_reported_month=int(row["month"]),
                        **rates,
                    )
                )
            return native_aggregate(rows)

        evaluator.aggregate_year_metrics = aggregate_year
        kwargs.update(
            enable_trade_war=True,
            trade_war_scenario=arm,
            trade_war_driver=str(args.tariff_driver.resolve()),
            trade_war_exposures=str(args.exposures.resolve()),
            covid_arm_id="TARIFF1000",
        )
    if acceptance:
        from .offer_matches import finalized_matches, trace_scope

        pin, constant = (None, 0.0)
        if args.fit:
            source = args.fit.read_bytes()
            pin = dict(path=str(args.fit.resolve()), sha256=hashlib.sha256(source).hexdigest())
            constant = json.loads(source)["c_s"]
        state = offers.AcceptanceObserver(seed, evaluation_seed, output, arm, constant, pin)
        offers.install(state)
        stream = gzip.open(output / "FINALIZED_MATCHES.jsonl.gz", "xt", encoding="utf8")
        snapshots = []
    if args.family == "spatial":
        from .spatial import SpatialObserver

        spatial = SpatialObserver(
            output, firm_csv_path=args.data / "Data_Agents/firm_population/firms_2006.csv"
        )

    class ObservedEnvironment(base):

        def __init__(self, config, *extra, **kw):
            profile = load_profile(3)
            steps = profile["rollout_steps"]
            apply_profile(config, 3, seed, steps, steps * profile["updates"])
            config.simulation.n_workers, config.simulation.n_firms = (args.workers, args.firms)
            config.simulation.end_year, config.simulation.max_episode_years = (
                calendar[-1][0],
                months // 12,
            )
            config.training.freeze_firm_policy = config.training.freeze_learned_firm_network = fixed
            config.training.enable_reservation_wage_bundle = (
                config.training.reservation_decay_compounded
            ) = km
            config.training.reservation_decay_per_year = 0.075 if km else 0.05
            configure(config, simulation_seed=evaluation_seed)
            if acceptance:
                offers.configure_evaluation_config(config)
            super().__init__(config, *extra, **kw)
            holder["env"] = self
            if forward:
                self._forward_source_observer = observer.ForwardObserver(self, job)

        def _step(self, action):
            i = len(aggregate.months)
            period = calendar[i]
            observed_clock = (
                int(self.state_manager.current_year),
                int(self.state_manager.current_month),
            )
            require(
                observed_clock == period,
                "Market calendar differs from the requested evaluation window",
            )
            aggregate.before(self, action, i, period)
            if i == 0:
                capacity_finalize = self._net_population.finalize

                def record_boundary(env, obs):
                    result = capacity_finalize(env, obs)
                    aggregate.months[-1]["renewal_capacity"] = capacity_budget.audit(env)
                    return result

                self._net_population.finalize = record_boundary
            if args.family == "spatial":
                if i == 0:
                    old_finalize = self._net_population.finalize

                    def finalize(env, obs):
                        result = old_finalize(env, obs)
                        j = spatial.renewal_months
                        spatial.after_renewal(env, j, calendar[j])
                        return result

                    self._net_population.finalize = finalize
                spatial.before(self, action, i, period)
            if acceptance:
                state.begin(self, i, *period)
                opening = {
                    w: dict(cma=int(s.worker.cma), status=str(s.worker.employment_status))
                    for w, s in self.state_manager.workers.items()
                }
                with trace_scope(self):
                    result = super()._step(action)
                    for entry in finalized_matches(self, opening, i, job):
                        stream.write(json.dumps(entry, allow_nan=False) + "\n")
                    snapshots.append(self.get_covid_validation_snapshot(*period))
                    state.end(self)
            else:
                result = super()._step(action)
            aggregate.after(self, i, period)
            aggregate.months[-1]["market_capacity"] = capacity_budget.audit(self)
            if args.family == "spatial":
                spatial.after(self, i, period)
            return result

    evaluator.LaborMarketEnv = ObservedEnvironment
    log = evaluator.run_evaluation(**kwargs)
    aggregate.validate(months)
    write_monthly(output, aggregate, seed, index)
    if args.family == "worker_groups":
        dump(
            output / "worker_groups.json",
            evaluator.analyze_archetype_performance(holder["env"].state_manager.workers),
        )
    if acceptance:
        state.finish(months)
        stream.close()
        dump(
            output / "ACCEPTANCE_TH_MONTHLY.json",
            dict(
                schema="acceptance_native_TH_market_close_snapshots_v1",
                months=snapshots,
                benchmark_cmas=[535, 537],
                native_get_covid_validation_snapshot=True,
                COVID_regime_enabled=False,
                separation_standard_driver_mode="as_loaded",
            ),
        )
    if forward:
        detailed = holder["env"]._forward_source_observer
        detailed.finish()
        dump(
            output / "FORWARD_OUTCOMES_MONTHLY.json",
            dict(months=detailed.outcomes, benchmark_cmas=[535, 537]),
        )
    if args.family == "spatial":
        spatial.finish(months)
    if args.family == "tariff":
        dump(
            output / "TARIFF_NATIVE_MONTHLY_METRICS.json",
            dict(schema="tariff_native_monthly_metrics_v1", months=macro_months),
        )
    dump(output / "simulation_log.json", log)
    dump(
        output / "history.json",
        dict(
            checkpoint_sha256=checkpoint_sha256,
            evaluation_overrides=intervention,
            policy_seed=seed,
            replication_index=index,
            family=args.family,
            arm=arm,
            evaluation_seed=evaluation_seed,
            workers=args.workers,
            firms=args.firms,
            months=months,
        ),
    )
    print(f"Saved the {args.family} history ({months} months) to {output}")


if __name__ == "__main__":
    main()
