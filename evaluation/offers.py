from __future__ import annotations
import gzip
import hashlib
import json
import math
from pathlib import Path

_STATE = None
BETAS = ("beta_wage", "beta_skill_match", "beta_commute", "beta_job_security")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def _tolist(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return list(value)


def is_recentred():
    return _STATE is not None and _STATE.arm == "recentred"


def configure_evaluation_config(config):
    if _STATE is None:
        return
    mnl = config.matching.mnl_config
    if _STATE.arm == "recentred":
        for key in BETAS:
            setattr(mnl, key, 0.0)
    _STATE.readbacks.append(
        {
            "role": "post_evaluation_config_overrides",
            "arm": _STATE.arm,
            "active_betas": {key: float(getattr(mnl, key)) for key in BETAS},
        }
    )


def configure_matcher(matcher):
    if _STATE is None:
        return
    import torch

    config = matcher.config.mnl_config
    tau = float(getattr(config, "acceptance_temperature"))
    if tau != 1.2:
        raise ValueError("Acceptance temperature differs from 1.2")
    before = {key: float(getattr(config, key)) for key in BETAS}
    if _STATE.arm == "recentred":
        for key in BETAS:
            setattr(config, key, 0.0)
    constant = _STATE.constant if _STATE.arm == "recentred" else 0.0
    matcher._acceptance_recentre_c = torch.tensor(constant, device=matcher.device)
    quantized = float(matcher._acceptance_recentre_c.item())
    if abs(quantized - constant) > 2e-06 * max(1.0, abs(constant)):
        raise ValueError("Float32 recentring constant readback differs")
    _STATE.readbacks.append(
        {
            "role": "matcher_initialization",
            "original_betas": before,
            "active_betas": {key: float(getattr(config, key)) for key in BETAS},
            "requested_c_s": constant,
            "active_c_s": quantized,
            "tau": tau,
        }
    )


def record_context(
    matcher,
    workers,
    worker_indices,
    vacancy_indices,
    attributes,
    utility_diff,
    probabilities,
    job_security=None,
):
    if _STATE is None or _STATE.matcher is not matcher:
        return
    _STATE.context(
        matcher,
        workers,
        worker_indices,
        vacancy_indices,
        attributes,
        utility_diff,
        probabilities,
        job_security,
    )


def record_decisions(matcher, probabilities, accepted):
    if _STATE is None or _STATE.matcher is not matcher:
        return
    _STATE.decisions(probabilities, accepted)


class AcceptanceObserver:

    def __init__(
        self,
        policy_seed,
        evaluation_seed,
        path,
        arm="baseline",
        constant=0.0,
        derived_parameter_pin=None,
    ):
        if evaluation_seed not in range(100 * policy_seed, 100 * policy_seed + 50):
            raise ValueError("Evaluation identity differs from the policy's seed block")
        if arm not in ("baseline", "recentred") or not math.isfinite(constant):
            raise ValueError("Invalid acceptance arm/constant")
        if arm == "baseline" and constant != 0.0:
            raise ValueError("Baseline constant must be zero")
        if arm == "recentred" and (not derived_parameter_pin):
            raise ValueError("The recentred arm requires its fit file")
        if derived_parameter_pin:
            if _sha(derived_parameter_pin["path"]) != derived_parameter_pin["sha256"]:
                raise ValueError("Derived parameter manifest drift")
            derived = json.loads(Path(derived_parameter_pin["path"]).read_text(encoding="utf-8"))
            entry = derived
            if (
                entry["schema"] != "current_policy_acceptance_recentre_fit_v1"
                or entry["status"] != "FROZEN_AFTER_50_BASELINE_HISTORIES"
                or entry["policy_seed"] != policy_seed
                or (entry["tau"] != 1.2)
                or (entry["baseline_histories"] != 50)
                or (
                    entry["baseline_seed_block"]
                    != list(range(policy_seed * 100, policy_seed * 100 + 50))
                )
                or (entry["c_s"] != constant)
                or (entry.get("source_arm") != "baseline")
            ):
                raise ValueError("The requested constant differs from the policy's baseline fit")
        self.policy_seed, self.evaluation_seed = (policy_seed, evaluation_seed)
        self.path, self.arm, self.constant = (Path(path), arm, float(constant))
        self.derived_parameter_pin = derived_parameter_pin
        self.matcher = None
        self.months = []
        self.current = None
        self.pending = None
        self.offers = self.decided = 0
        self.readbacks = []
        self.streams = None
        self.closed = False
        self.security_count = 0
        self.security_sum = 0.0
        self.security_sum_squares = 0.0
        self.security_min, self.security_max = (None, None)
        self.security_not_inherited_half = 0

    def _open(self):
        if self.streams is not None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        names = ("offer_context_records.jsonl.gz", "realized_accept_log.jsonl.gz")
        self.streams = {name: gzip.open(self.path / name, "xt", encoding="utf-8") for name in names}

    def begin(self, env, month_index, year, calendar_month):
        if self.current is not None or month_index != len(self.months):
            raise ValueError("Acceptance month continuity differs")
        if (year, calendar_month) != (2006 + month_index // 12, month_index % 12 + 1):
            raise ValueError("Acceptance calendar differs")
        self._open()
        self.matcher = env.matching_mechanism
        self.current = {
            "month_index": month_index,
            "year": year,
            "calendar_month": calendar_month,
            "opening_offer_id": self.offers,
            "recorded_offers": 0,
        }

    def context(
        self,
        matcher,
        workers,
        worker_indices,
        vacancy_indices,
        attributes,
        utility_diff,
        probabilities,
        job_security=None,
    ):
        if self.current is None or self.pending is not None:
            raise ValueError("Actual offer contexts are unpaired or outside a market month")
        if "cma" not in workers:
            raise ValueError("Worker-side CMA absent from actual offer tensors")
        wi, vi = (_tolist(worker_indices), _tolist(vacancy_indices))
        a, u, p = (_tolist(attributes), _tolist(utility_diff), _tolist(probabilities))
        cma = _tolist(workers["cma"][worker_indices])
        if job_security is None:
            raise ValueError("The job-security values of the offers were not observed")
        security = _tolist(job_security)
        if len({len(x) for x in (wi, vi, a, u, p, cma, security)}) != 1:
            raise ValueError("Offer context tensor alignment differs")
        constant = float(getattr(matcher, "_acceptance_recentre_c").item())
        r = _tolist(utility_diff - attributes - matcher._acceptance_recentre_c)
        rows = []
        for index in range(len(p)):
            row = {
                "schema": "actual_worker_offer_context_v1",
                "arm": self.arm,
                "policy_seed": self.policy_seed,
                "evaluation_seed": self.evaluation_seed,
                "rollout_reseed": self.evaluation_seed,
                "offer_id": self.offers + index,
                "month_index": self.current["month_index"],
                "year": self.current["year"],
                "calendar_month": self.current["calendar_month"],
                "worker_tensor_index": wi[index],
                "vacancy_tensor_index": vi[index],
                "cma": int(cma[index]),
                "a_i": float(a[index]),
                "r_i": float(r[index]),
                "tau": 1.2,
                "c_s": constant,
                "p_i_B": float(p[index]),
            }
            row["job_security"] = float(security[index])
            if not math.isfinite(row["job_security"]):
                raise ValueError("Non-finite job-security value of an offer")
            self.security_count += 1
            self.security_sum += row["job_security"]
            self.security_sum_squares += row["job_security"] ** 2
            self.security_min = (
                row["job_security"]
                if self.security_min is None
                else min(self.security_min, row["job_security"])
            )
            self.security_max = (
                row["job_security"]
                if self.security_max is None
                else max(self.security_max, row["job_security"])
            )
            self.security_not_inherited_half += row["job_security"] != 0.5
            if row["cma"] not in (532, 535, 537):
                raise ValueError("Unknown worker-side CMA in actual offer")
            if not all((math.isfinite(row[key]) for key in ("a_i", "r_i", "p_i_B"))):
                raise ValueError("Non-finite offer context")
            if not 0.0 <= row["p_i_B"] <= 1.0:
                raise ValueError("Offer probability outside unit interval")
            z = max(-60.0, min(60.0, (row["a_i"] + row["r_i"] + constant) / 1.2))
            if abs(1.0 / (1.0 + math.exp(-z)) - row["p_i_B"]) > 1e-06:
                raise ValueError("Live offer probability differs from logit identity")
            self.streams["offer_context_records.jsonl.gz"].write(
                json.dumps(row, allow_nan=False) + "\n"
            )
            rows.append(row)
        self.pending = rows
        self.offers += len(rows)
        self.current["recorded_offers"] += len(rows)

    def decisions(self, probabilities, accepted):
        p, a = (_tolist(probabilities), _tolist(accepted))
        if self.pending is None or len(self.pending) != len(p) or len(p) != len(a):
            raise ValueError("Context/worker-Bernoulli offer identity differs")
        for row, probability, acceptance in zip(self.pending, p, a):
            if float(probability) != row["p_i_B"]:
                raise ValueError("Worker Bernoulli probability differs from logged context")
            decision = {
                key: row[key]
                for key in (
                    "policy_seed",
                    "evaluation_seed",
                    "rollout_reseed",
                    "offer_id",
                    "year",
                    "calendar_month",
                    "month_index",
                    "cma",
                    "p_i_B",
                    "arm",
                )
            }
            decision["accepted"] = bool(acceptance)
            self.streams["realized_accept_log.jsonl.gz"].write(json.dumps(decision) + "\n")
        self.decided += len(p)
        self.pending = None

    def end(self, env):
        if self.current is None or self.pending is not None or self.offers != self.decided:
            raise ValueError("Incomplete context/Bernoulli reconciliation")
        native = env.last_matching_stats.get("offers")
        if type(native) is not int or native != self.current["recorded_offers"]:
            raise ValueError("The observed offer count differs from the matching counters")
        self.current["native_offers"] = native
        self.current["reconciliation_passed"] = True
        self.months.append(self.current)
        self.current = None
        self.matcher = None

    def finish(self, expected_months=168):
        global _STATE
        if self.closed or self.current is not None or self.pending is not None:
            raise ValueError("Acceptance observer finish is not clean")
        if len(self.months) != expected_months:
            raise ValueError("Acceptance horizon incomplete")
        self._open()
        for stream in self.streams.values():
            stream.close()
        self.closed = True
        receipt = {
            "schema": "rate_matched_acceptance_observer_v2_receipt_v1",
            "policy_seed": self.policy_seed,
            "evaluation_seed": self.evaluation_seed,
            "arm": self.arm,
            "months": len(self.months),
            "offer_count": self.offers,
            "bernoulli_offer_count": self.decided,
            "all_monthly_funnel_counts_match": True,
            "derived_parameter_manifest": self.derived_parameter_pin,
            "matcher_readbacks": self.readbacks,
            "monthly_reconciliations": self.months,
            "actual_offer_job_security": {
                "count": self.security_count,
                "mean": self.security_sum / self.security_count if self.security_count else None,
                "variance_population": (
                    max(
                        0.0,
                        self.security_sum_squares / self.security_count
                        - (self.security_sum / self.security_count) ** 2,
                    )
                    if self.security_count
                    else None
                ),
                "min": self.security_min,
                "max": self.security_max,
                "values_different_from_inherited_half": self.security_not_inherited_half,
                "all_equal_inherited_half": (
                    self.security_not_inherited_half == 0 if self.security_count else None
                ),
                "actual_native_v_security_tensor_observed": True,
                "interpretation": "Structural observation only, not an empirical effect or confirmatory endpoint.",
            },
            "streams": {
                name: {"path": str((self.path / name).resolve()), "sha256": _sha(self.path / name)}
                for name in self.streams
            },
        }
        with (self.path / "ACCEPTANCE_RECEIPT.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        if _STATE is self:
            _STATE = None
        return receipt


def install(observer):
    global _STATE
    if _STATE is not None:
        raise ValueError("Acceptance extension already installed in this process")
    _STATE = observer
