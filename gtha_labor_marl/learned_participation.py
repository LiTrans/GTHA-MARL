MODE = "learned_55_plus_reversible_v1"


def configure(config, manifest):
    if manifest.get("learned_participation_rule") != MODE:
        raise ValueError("The learned-participation mode is missing")
    if not config.mnl.nilf_subtype_enabled:
        raise ValueError("Reversible retirement requires the observed subtype state")
    if config.reward.retirement_age_min != 60:
        raise ValueError("Inherited pension-income age must remain 60")
    config.reward.retirement_action_age_min = 55
    absorbing = manifest.get("macro_correction") == "slid_competing_risks_absorbing_retirement_v1"
    config.mnl.allow_retirement_reentry = not absorbing
    config.training.macro_absorbing_retirement = absorbing


def validate(config, retirement_absorbing=None):
    if retirement_absorbing is None:
        retirement_absorbing = bool(getattr(config.training, "macro_absorbing_retirement", False))
    expected = (
        config.reward.retirement_action_age_min == 55,
        config.reward.retirement_age_min == 60,
        getattr(config.mnl, "allow_retirement_reentry", False) is (not retirement_absorbing),
        True,
    )
    if not all(expected):
        raise ValueError("Learned participation gates or pension separation disagree")
