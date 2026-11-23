import hashlib
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_frozen_eval_config(config, checkpoint, *, skip_manifest_assert=False):
    if not config.trade_war.enabled or not config.trade_war.scenario:
        raise ValueError("Select a tariff scenario before running its evaluation")
    return {
        "policy_seed": config.training.policy_seed,
        "scenario": config.trade_war.scenario,
        "checkpoint_sha256": sha256(checkpoint),
        "driver_sha256": sha256(config.trade_war.driver_file),
        "exposures_sha256": sha256(config.trade_war.exposure_file),
    }
