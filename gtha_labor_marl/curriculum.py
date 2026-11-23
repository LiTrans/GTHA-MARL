import json
from pathlib import Path


def load_profile(stage):
    path = Path(__file__).with_name("training_profiles.json")
    return json.loads(path.read_text(encoding="utf-8"))[str(stage)]


def apply_profile(config, stage, seed, rollout_steps, total_timesteps):
    profile = load_profile(stage)
    for section in ("training", "network"):
        target = getattr(config, section)
        for key, value in profile[section].items():
            if value is not None:
                if not hasattr(target, key):
                    raise AttributeError(f"Unknown {section} setting {key}")
                setattr(target, key, value)
    config.seed = seed
    config.simulation.start_year = 2006
    config.simulation.end_year = 2016
    config.simulation.max_episode_years = 11
    training = config.training
    training.rollout_steps = rollout_steps
    training.total_timesteps = total_timesteps
    training.n_minibatches = training.batch_size // training.minibatch_size
    training.enable_reservation_wage_bundle = True
    training.reservation_decay_per_year = 0.075
    training.reservation_decay_compounded = True
    training.worker_res_wage_frozen_level = 2
    if stage in (1, 2):
        training.worker_res_wage_entropy_coef = 0.02
    matching = config.matching.mnl_config
    matching.acceptance_temperature = profile["acceptance_temperature"]
    matching.beta_wage = matching.estimated_beta_wage
    matching.beta_commute = matching.estimated_beta_commute
    config.mean_field.enabled = True
    config.mean_field.inner_training_steps = min(50000, total_timesteps)
    config.mean_field.max_outer_iterations = 10
    return config


TRAINING_METRICS = ("worker/mean_reward_raw", "firm/mean_reward_raw", "worker_kl", "firm_kl")


def training_rows(metrics, metadata, seed, stage, offset):
    import math
    import pandas as pd

    profile = load_profile(stage)
    updates = profile["updates"]
    args = metadata["cli_args"]
    expected = (seed, stage, profile["rollout_steps"], updates * profile["rollout_steps"])
    identity = tuple((args[name] for name in ("seed", "stage", "rollout_steps", "total_timesteps")))
    if identity != expected:
        raise ValueError(f"Incorrect seed, stage or training budget for seed {seed}, stage {stage}")
    columns = ["update", *TRAINING_METRICS]
    table = pd.read_csv(metrics, usecols=columns, float_precision="round_trip")[columns]
    if table["update"].tolist() != list(range(1, updates + 1)):
        raise ValueError(f"Incomplete update sequence for seed {seed}, stage {stage}")
    for update, *values in table.itertuples(index=False, name=None):
        if not all((math.isfinite(value) for value in values)):
            raise ValueError(
                f"Nonfinite training measurement for seed {seed}, stage {stage}, update {update}"
            )
        yield (seed, stage, update, offset + update, *values)


def write_training_table(training, output):
    import csv
    import os
    from tempfile import NamedTemporaryFile

    training, output = (Path(training), Path(output))
    rows = []
    for seed in range(42, 47):
        offset = 0
        for stage in (1, 2, 3):
            folder = training / f"policy{seed}" / f"stage{stage}"
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig"))
            rows.extend(
                training_rows(folder / "metrics_history.csv", manifest, seed, stage, offset)
            )
            offset += load_profile(stage)["updates"]
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        dir=output.parent, mode="w", encoding="utf-8", newline="", delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("policy_seed", "stage", "stage_update", "cumulative_update", *TRAINING_METRICS)
            )
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare the five-policy training diagnostics for Extended Data Figure 9."
    )
    parser.add_argument(
        "--training",
        type=Path,
        required=True,
        help="Directory containing policy42 through policy46",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV file")
    options = parser.parse_args()
    print(f"Saved {write_training_table(options.training, options.output)} training updates.")
