import os
from functools import wraps
from pathlib import Path


def data_root():
    return Path(os.environ.get("METRO_DATA_ROOT", Path(__file__).resolve().parents[1])).resolve()


def settings():
    return {
        "end_year": int(os.environ.get("METRO_END_YEAR", "2016")),
        "employment_initialization": "workplace_distribution_initial_only_v1",
        "workplace_data_dir": "Data/Simulation/Parameters/workplace",
        "demographic_mode": "july_net_cohort_renewal_v1",
        "learned_participation_rule": "learned_55_plus_reversible_v1",
        "macro_correction": "slid_competing_risks_absorbing_retirement_v1",
        "participation_decisions": {
            "reentry": "slid_2006_2011_age_competing_risks_v1",
            "retirement": "learned_55_plus_absorbing_v2",
        },
        "population_design_seed": 42,
    }


def select_firms(pool, config):
    from initial_roster import selected_roster, allocate, rank
    from market_mechanisms import source_profiles, rule_for

    size = config.simulation.n_firms
    profiles, _, _ = source_profiles()
    rows, _ = selected_roster(profiles, rule_for, size, allocate, rank)
    identifiers = [row["source_firm_id"] for row in rows]
    indexed = pool.set_index(pool.firm_id.astype(str), drop=False)
    if len(identifiers) != size or len(set(identifiers)) != size or (not indexed.index.is_unique):
        raise ValueError("Initial firm roster does not cover the requested scale uniquely")
    return indexed.loc[identifiers].reset_index(drop=True).copy()


def configure(config, simulation_seed=None):
    seed = simulation_seed
    if seed is None:
        seed = getattr(config, "_simulation_seed", config.seed)
    config._simulation_seed = int(seed)
    os.environ.update(
        SEPARATION_DESTINATION_SPLIT_SEED=str(seed),
        ABSOLUTE_SEPARATION_PARAMETERS=str(Path(__file__).with_name("separation_parameters.json")),
        METRO_END_YEAR=str(config.simulation.end_year),
    )
    import age_renewal
    import capacity_lineage
    import regional_capacity
    from market_snapshots import plan
    from population_participation import configure_participation

    plan.cache_clear()
    age_renewal.source_stocks.cache_clear()
    regional_capacity.install(age_renewal)
    capacity_lineage.install(age_renewal)
    config.training.renewal_net_projection = str(
        data_root() / "Data/Simulation/Parameters/net_projection.csv"
    )
    config.reward.retirement_age_min = 60
    config.mnl.nilf_subtype_enabled = True
    configure_participation(config)
    return config


def bind_environment(env_class):
    if getattr(env_class, "_market_reset_installed", False):
        return
    original = env_class._reset

    @wraps(original)
    def reset(env, *args, **kwargs):
        result = original(env, *args, **kwargs)
        controller = getattr(env, "_net_population", None)
        if controller is not None:
            controller.reset()
            controller.seed(env.config._simulation_seed)
        from market_snapshots import capture_initial

        capture_initial(env)
        firms = env.state_manager.firms
        workers = env.state_manager.workers
        if (
            len(workers) != env.config.simulation.n_workers
            or len(firms) != env.config.simulation.n_firms
        ):
            raise ValueError("Reset counts differ from the requested model scale")
        audit = env._initial_employment_audit
        if any(
            (
                len(firms[row["firm_id"]].employees) != row["initial_jobs"]
                or firms[row["firm_id"]].firm.target_size < row["initial_jobs"]
                for row in audit["firms"]
            )
        ):
            raise ValueError(
                "Initial firm staffing differs from workplace allocations or exceeds capacity"
            )
        env._fixed_count_reset_audits = [
            {"checks": "PASS", "workers": len(workers), "firms": len(firms)}
        ]
        return result

    env_class._reset = reset
    env_class._market_reset_installed = True
