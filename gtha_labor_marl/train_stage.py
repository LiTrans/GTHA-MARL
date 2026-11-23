import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import torch
import numpy as np


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def _hash_file_sha256(path: Path) -> str:
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _compute_codebase_checksum(repo_root: Path) -> tuple:
    import hashlib
    import fnmatch

    repo_root = repo_root.resolve()
    file_hashes = []
    SKIP_EXACT = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "wandb",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "outputs",
        "runs",
        "checkpoints",
        "logs",
        "experiments",
        "tensorboard",
        "mlruns",
    }
    SKIP_PATTERNS = (
        "*_local",
        "*_diag*",
        "*_smoke*",
        "*_eval*",
        "metrics_*",
        "sep_diag*",
        "comp_*_smoke*",
    )

    def _is_skipped_dir(name: str) -> bool:
        if name in SKIP_EXACT:
            return True
        for pat in SKIP_PATTERNS:
            if fnmatch.fnmatch(name, pat):
                return True
        return False

    SOURCE_EXTENSIONS = (".py", ".sh")
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not _is_skipped_dir(d)]
        for fname in files:
            fpath = Path(root) / fname
            try:
                rel = str(fpath.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                rel = str(fpath)
            lower = fname.lower()
            include = False
            if lower.endswith(SOURCE_EXTENSIONS):
                include = True
            elif lower.endswith(".json") and (rel.startswith("config/") or "/config/" in rel):
                include = True
            if not include:
                continue
            file_hashes.append((rel, _hash_file_sha256(fpath)))
    file_hashes.sort(key=lambda x: x[0])
    rollup = hashlib.sha256()
    for rel, h in file_hashes:
        rollup.update(rel.encode("utf-8"))
        rollup.update(b"\x00")
        rollup.update(h.encode("utf-8"))
        rollup.update(b"\n")
    return (rollup.hexdigest(), len(file_hashes))


def write_run_manifest(
    output_path: Path, config, args, repo_root: Path, seeds: dict, data_files: list
) -> Path:
    import platform
    from dataclasses import asdict, is_dataclass

    def _config_to_dict(obj):
        if is_dataclass(obj):
            return {k: _config_to_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: _config_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_config_to_dict(x) for x in obj]
        return convert_to_serializable(obj)

    try:
        config_snapshot = _config_to_dict(config)
    except Exception:
        config_snapshot = {"_serialization_failed": True}
    data_hashes = {}
    for d in data_files or []:
        d_path = Path(d) if not isinstance(d, Path) else d
        if d_path.exists() and d_path.is_file():
            data_hashes[str(d_path.name)] = _hash_file_sha256(d_path)
    codebase_hash, codebase_n_files = _compute_codebase_checksum(repo_root)
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "cli_args": {k: convert_to_serializable(v) for k, v in vars(args).items()},
        "seeds": seeds,
        "data_hashes": data_hashes,
        "codebase_hash": codebase_hash,
        "codebase_n_files": codebase_n_files,
        "config_snapshot": config_snapshot,
    }
    manifest_path = Path(output_path) / "manifest.json"
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not write the run record {manifest_path}: {e}")
    return manifest_path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_theta_distance(theta_old: dict, theta_new: dict) -> float:
    diffs = []
    for ind in theta_old.get("market_tightness_by_industry", {}):
        old_val = theta_old["market_tightness_by_industry"].get(ind, 1.0)
        new_val = theta_new["market_tightness_by_industry"].get(ind, 1.0)
        diffs.append((old_val - new_val) ** 2)
    for key in [
        "aggregate_unemployment_rate",
        "aggregate_employment_rate",
        "aggregate_vacancy_rate",
    ]:
        diffs.append((theta_old.get(key, 0) - theta_new.get(key, 0)) ** 2)
    return np.sqrt(np.mean(diffs)) if diffs else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one curriculum stage of the worker and firm policies."
    )
    parser.add_argument("--n_workers", type=int, default=2000, help="Number of initial residents")
    parser.add_argument("--n_firms", type=int, default=100, help="Number of firm slots")
    parser.add_argument(
        "--total_timesteps", type=int, default=500000, help="Total rollout steps of the stage"
    )
    parser.add_argument("--rollout_steps", type=int, default=1024, help="Rollout steps per update")
    parser.add_argument("--seed", type=int, default=42, help="Policy seed")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory of the stage"
    )
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3], required=True, help="Curriculum stage"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu or cuda")
    return parser.parse_args()


def setup_output_dir(args):
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"training_run_{timestamp}"
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cfg_dump = vars(args).copy()
    cfg_dump["start_time"] = datetime.now().isoformat()
    with open(output_path / "config.json", "w") as f:
        json.dump(cfg_dump, f, indent=2)
    file_handler = logging.FileHandler(
        output_path / "training.log", encoding="utf-8", errors="replace"
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    return output_path


def select_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                return torch.device("cuda")
            except Exception as e:
                logger.warning(f"CUDA available but failed: {e}")
                return torch.device("cpu")
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    output_path = setup_output_dir(args)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Population: {args.n_workers} initial residents, {args.n_firms} firm slots")
    device = select_device(args.device)
    logger.info(f"Device: {device}")
    try:
        from config.default_config import Config
        from environment.labor_market_env_gpu import LaborMarketEnv
        from models.dt_mappo import DTMAPPO
    except ImportError as e:
        logger.error(f"Failed to import training components: {e}")
        logger.error("Run the command from the repository directory.")
        logger.error(f"Current working directory: {os.getcwd()}")
        sys.exit(1)
    config = Config()
    config.simulation.n_workers = args.n_workers
    config.simulation.n_firms = args.n_firms
    config.training.device = str(device)
    expected_updates = args.total_timesteps // args.rollout_steps
    logger.info(f"Updates in this stage: {expected_updates}")
    import random as _py_random

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _py_random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from curriculum import apply_profile

    apply_profile(config, args.stage, args.seed, args.rollout_steps, args.total_timesteps)
    env = LaborMarketEnv(config=config)
    actual_worker_dim = env.fast_obs.state_table.worker_obs_dim
    actual_firm_dim = env.fast_obs.state_table.firm_obs_dim
    if config.network.worker_state_dim != actual_worker_dim:
        logger.warning(
            f"Worker observation size set to {actual_worker_dim} (the settings gave {config.network.worker_state_dim})"
        )
        config.network.worker_state_dim = actual_worker_dim
    if config.network.firm_state_dim != actual_firm_dim:
        logger.warning(
            f"Firm observation size set to {actual_firm_dim} (the settings gave {config.network.firm_state_dim})"
        )
        config.network.firm_state_dim = actual_firm_dim
    trainer = DTMAPPO(config, env)
    if args.stage == 1 and (not args.resume_from):
        trainer.save(str(output_path / "initial_checkpoint.pt"))
    try:
        _data_files = []
        _data_roots = []
        try:
            _data_roots.append(config.data.data_dir)
        except Exception:
            pass
        try:
            _data_roots.append(config.data.population_dir)
        except Exception:
            pass
        try:
            _data_roots.append(config.data.firms_dir)
        except Exception:
            pass
        _earlier_lower = PROJECT_ROOT / "data"
        if _earlier_lower.exists() and _earlier_lower.is_dir():
            _data_roots.append(_earlier_lower)
        for _data_dir in _data_roots:
            if _data_dir is None or not _data_dir.exists():
                continue
            for _candidate in (
                "separation_rates_2006_2016.csv",
                "labor_force_transitions_2006_2016.csv",
                "wage_parameters_2006_2016.csv",
                "retirement_rates.csv",
                "da_centroids_gtha.csv",
                "firms_2006.csv",
                "firm_location_provenance.json",
            ):
                _p = _data_dir / _candidate
                if _p.exists() and _p not in _data_files:
                    _data_files.append(_p)
            for _p in sorted(_data_dir.rglob("*.csv")):
                if _p not in _data_files:
                    _data_files.append(_p)
        _seeds_dict = {
            "master": args.seed,
            "torch": args.seed,
            "numpy": args.seed,
            "python_random": args.seed,
        }
        _manifest_path = write_run_manifest(
            output_path=output_path,
            config=config,
            args=args,
            repo_root=PROJECT_ROOT,
            seeds=_seeds_dict,
            data_files=_data_files,
        )
        with open(_manifest_path) as _f:
            _manifest = json.load(_f)
    except Exception as _e:
        logger.warning(f"Could not write the run record: {_e}")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            logger.error(f"Checkpoint not found: {resume_path}")
            sys.exit(1)
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.load(str(resume_path))
        try:
            same_stage = resume_path.resolve().is_relative_to(output_path.resolve())
        except AttributeError:
            same_stage = str(resume_path.resolve()).startswith(str(output_path.resolve()))
        if not same_stage:
            trainer.total_steps = 0
        if args.stage == 2:
            trainer.firm_entropy_anneal_reset_update = trainer.update_count
        _lr_map = [
            (
                trainer.worker_actor_optimizer,
                config.training.worker_actor_lr or config.training.actor_lr,
                "worker_actor",
            ),
            (
                trainer.worker_critic_optimizer,
                config.training.worker_critic_lr or config.training.critic_lr,
                "worker_critic",
            ),
            (
                trainer.firm_actor_optimizer,
                config.training.firm_actor_lr or config.training.actor_lr,
                "firm_actor",
            ),
            (
                trainer.firm_critic_optimizer,
                config.training.firm_critic_lr or config.training.critic_lr,
                "firm_critic",
            ),
        ]
        for optimizer, target_lr, name in _lr_map:
            for pg in optimizer.param_groups:
                group_target_lr = pg.get("target_lr", target_lr)
                pg["lr"] = group_target_lr
                pg["initial_lr"] = group_target_lr
        if (
            hasattr(trainer, "worker_actor_scheduler")
            and trainer.worker_actor_scheduler is not None
        ):
            from torch.optim.lr_scheduler import LinearLR, LambdaLR

            total_steps = config.training.total_timesteps // config.training.rollout_steps
            trainer.worker_actor_scheduler = LinearLR(
                trainer.worker_actor_optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps,
            )
            trainer.worker_critic_scheduler = LinearLR(
                trainer.worker_critic_optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps,
            )
            trainer.firm_actor_scheduler = LinearLR(
                trainer.firm_actor_optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps,
            )
            trainer.firm_critic_scheduler = LambdaLR(
                trainer.firm_critic_optimizer, lr_lambda=lambda epoch: 1.0
            )
            if args.stage == 2:
                trainer.worker_actor_scheduler = LambdaLR(
                    trainer.worker_actor_optimizer, lr_lambda=lambda epoch: 1.0
                )
                trainer.worker_critic_scheduler = LambdaLR(
                    trainer.worker_critic_optimizer, lr_lambda=lambda epoch: 1.0
                )
                trainer.firm_actor_scheduler = LambdaLR(
                    trainer.firm_actor_optimizer, lr_lambda=lambda epoch: 1.0
                )
    metrics_history = []
    update_times = []
    start_time = time.time()
    try:
        mf_config = config.mean_field
        use_mean_field = mf_config.enabled
        if use_mean_field:
            inner_updates_per_iter = max(1, mf_config.inner_training_steps // args.rollout_steps)
            max_outer = mf_config.max_outer_iterations
        else:
            inner_updates_per_iter = expected_updates
            max_outer = 1
        global_update_idx = 0
        theta_distances = []
        for outer_iter in range(max_outer):
            if use_mean_field:
                current_theta = env.state_manager.compute_firm_mean_field()
                env.state_manager.freeze_theta(current_theta)
            for inner_idx in range(inner_updates_per_iter):
                if global_update_idx >= expected_updates:
                    break
                update_start = time.time()
                rollout_stats = trainer.collect_rollout()
                metrics = trainer.update()
                metrics.update(
                    {
                        "worker/mean_reward_raw": rollout_stats["mean_worker_reward"],
                        "firm/mean_reward_raw": rollout_stats["mean_firm_reward"],
                    }
                )
                update_time = time.time() - update_start
                update_times.append(update_time)
                progress = (global_update_idx + 1) / expected_updates * 100
                avg_update_time = np.mean(update_times[-10:])
                eta_seconds = avg_update_time * (expected_updates - global_update_idx - 1)
                eta = timedelta(seconds=int(eta_seconds))
                if (global_update_idx + 1) % 1 == 0:
                    state_stats = env.state_manager.stats
                    n_employed = state_stats.get("employed", 0)
                    n_unemployed = state_stats.get("unemployed", 0)
                    n_nilf = state_stats.get("nilf", 0)
                    emp_rate = n_employed / args.n_workers * 100
                    labor_force = n_employed + n_unemployed
                    unemp_rate = n_unemployed / labor_force * 100 if labor_force > 0 else 0
                    nilf_rate = n_nilf / args.n_workers * 100
                    worker_reward = metrics.get("worker/mean_reward_raw", 0)
                    firm_reward = metrics.get("firm/mean_reward_raw", 0)
                    worker_ev = metrics.get("worker_explained_var", float("nan"))
                    firm_ev = metrics.get("firm_explained_var", float("nan"))
                    outer_tag = f" | outer iteration {outer_iter + 1}" if use_mean_field else ""
                    logger.info(
                        f"Update {global_update_idx + 1:4d}/{expected_updates}{outer_tag} | {progress:5.1f}% | {update_time:5.1f}s | ETA {eta} | employed {emp_rate:4.1f}% | unemployed {unemp_rate:4.1f}% | outside labour force {nilf_rate:4.1f}% | worker reward {worker_reward:+.6f} | firm reward {firm_reward:+.6f} | worker explained variance {worker_ev:+.3f} | firm explained variance {firm_ev:+.3f}"
                    )
                    rw_freq = rollout_stats.get("worker_res_wage_freq", []) or [0.0] * 5
                    search_ent = metrics.get("worker_search_entropy", float("nan"))
                    res_wage_ent = metrics.get("worker_res_wage_entropy", float("nan"))
                    matches_mean = rollout_stats.get("matches_per_step_mean", 0.0)
                    matches_total = rollout_stats.get("matches_per_step_total", 0)
                    _act_freq = np.asarray(
                        rollout_stats.get("worker_action_freq", []) or [], dtype=float
                    )
                    _search_max_prob = float(_act_freq.max()) if _act_freq.size else 0.0
                    if _act_freq.size:
                        _unif = 1.0 / _act_freq.size
                        _search_kl_uniform = float(
                            np.sum(_act_freq * (np.log(_act_freq + 1e-12) - np.log(_unif)))
                        )
                    else:
                        _search_kl_uniform = 0.0
                    _snap_rows = rollout_stats.get("worker_snapshot", []) or []
                    _per_skill_search_max_prob = {}
                    _per_skill_search_kl_uniform = {}
                    if _snap_rows:
                        _by_skill = {}
                        for _r in _snap_rows:
                            _sk = _r.get("skill_level")
                            _a = _r.get("search_action")
                            if _sk is None or _a is None:
                                continue
                            _by_skill.setdefault(int(_sk), []).append(int(_a))
                        _n_bins = int(_act_freq.size) if _act_freq.size else 40
                        for _sk in sorted(_by_skill):
                            _arr = np.bincount(_by_skill[_sk], minlength=_n_bins).astype(float)
                            if _arr.sum() == 0:
                                continue
                            _p = _arr / _arr.sum()
                            _per_skill_search_max_prob[_sk] = float(_p.max())
                            _per_skill_search_kl_uniform[_sk] = float(
                                np.sum(_p * (np.log(_p + 1e-12) - np.log(1.0 / _n_bins)))
                            )
                    _cohort_counts = rollout_stats.get("cohort_action_counts") or {}
                    _per_cohort_metrics = {}
                    if _cohort_counts:
                        _kl_all = _search_kl_uniform
                        _n_all = int(
                            sum(
                                (
                                    int(np.asarray(_c, dtype=float).sum())
                                    for _c in _cohort_counts.values()
                                )
                            )
                        )
                        _short_name = {
                            "Unemployed": "unemp",
                            "Employed": "emp",
                            "NILF_eligible": "nilf_elig",
                            "NILF_ineligible": "nilf_inelig",
                        }
                        for _coh in ("Unemployed", "Employed", "NILF_eligible", "NILF_ineligible"):
                            _arr = np.asarray(_cohort_counts.get(_coh, []) or [], dtype=float)
                            _n = int(_arr.sum())
                            if _arr.size and _n > 0:
                                _p = _arr / _n
                                _unif = 1.0 / _arr.size
                                _kl = float(np.sum(_p * (np.log(_p + 1e-12) - np.log(_unif))))
                                _mp = float(_p.max())
                            else:
                                _kl = 0.0
                                _mp = 0.0
                            _per_cohort_metrics[f"kl_{_short_name[_coh]}"] = _kl
                            _per_cohort_metrics[f"n_{_short_name[_coh]}"] = _n
                            _per_cohort_metrics[f"max_prob_{_short_name[_coh]}"] = _mp
                        _arr_u = np.asarray(_cohort_counts.get("Unemployed", []) or [], dtype=float)
                        _arr_ne = np.asarray(
                            _cohort_counts.get("NILF_eligible", []) or [], dtype=float
                        )
                        if _arr_u.size and _arr_ne.size and (_arr_u.size == _arr_ne.size):
                            _arr_act = _arr_u + _arr_ne
                            _n_act = int(_arr_act.sum())
                            if _n_act > 0:
                                _p_act = _arr_act / _n_act
                                _unif = 1.0 / _arr_act.size
                                _kl_act = float(
                                    np.sum(_p_act * (np.log(_p_act + 1e-12) - np.log(_unif)))
                                )
                                _mp_act = float(_p_act.max())
                            else:
                                _kl_act = 0.0
                                _mp_act = 0.0
                        else:
                            _kl_act = 0.0
                            _mp_act = 0.0
                            _n_act = 0
                        _per_cohort_metrics["kl_active_search"] = _kl_act
                        _per_cohort_metrics["n_active_search"] = _n_act
                        _per_cohort_metrics["max_prob_active_search"] = _mp_act
                        _per_cohort_metrics["kl_all"] = _kl_all
                        _per_cohort_metrics["n_all"] = _n_all
                        _per_skill_cohort = (
                            rollout_stats.get("cohort_per_skill_action_counts") or {}
                        )
                        if _per_skill_cohort:

                            def _kl_n_from_arr(_arr):
                                _n = int(_arr.sum())
                                if _arr.size and _n > 0:
                                    _p = _arr / _n
                                    _unif = 1.0 / _arr.size
                                    return (
                                        float(np.sum(_p * (np.log(_p + 1e-12) - np.log(_unif)))),
                                        _n,
                                    )
                                return (0.0, _n)

                            _short = {
                                "Unemployed": "unemp",
                                "Employed": "emp",
                                "NILF_eligible": "nilf_elig",
                                "NILF_ineligible": "nilf_inelig",
                            }
                            for _coh in (
                                "Unemployed",
                                "Employed",
                                "NILF_eligible",
                                "NILF_ineligible",
                            ):
                                _tag = _short[_coh]
                                for _s in range(5):
                                    _arr = np.asarray(
                                        _per_skill_cohort.get(_coh, {}).get(_s, []) or [],
                                        dtype=float,
                                    )
                                    _kl_s, _n_s = _kl_n_from_arr(_arr)
                                    _per_cohort_metrics[f"kl_{_tag}_s{_s}"] = _kl_s
                                    _per_cohort_metrics[f"n_{_tag}_s{_s}"] = _n_s
                            for _s in range(5):
                                _arr_us = np.asarray(
                                    _per_skill_cohort.get("Unemployed", {}).get(_s, []) or [],
                                    dtype=float,
                                )
                                _arr_nes = np.asarray(
                                    _per_skill_cohort.get("NILF_eligible", {}).get(_s, []) or [],
                                    dtype=float,
                                )
                                _arr_em = np.asarray(
                                    _per_skill_cohort.get("Employed", {}).get(_s, []) or [],
                                    dtype=float,
                                )
                                _arr_nin = np.asarray(
                                    _per_skill_cohort.get("NILF_ineligible", {}).get(_s, []) or [],
                                    dtype=float,
                                )
                                if (
                                    _arr_us.size
                                    and _arr_nes.size
                                    and (_arr_us.size == _arr_nes.size)
                                ):
                                    _arr_act_s = _arr_us + _arr_nes
                                else:
                                    _arr_act_s = np.zeros(0, dtype=float)
                                _kl_act_s, _n_act_s = _kl_n_from_arr(_arr_act_s)
                                _per_cohort_metrics[f"kl_active_search_s{_s}"] = _kl_act_s
                                _per_cohort_metrics[f"n_active_search_s{_s}"] = _n_act_s
                                _arrs = [
                                    a for a in (_arr_us, _arr_em, _arr_nes, _arr_nin) if a.size
                                ]
                                if _arrs and len({a.size for a in _arrs}) == 1:
                                    _arr_all_s = sum(_arrs)
                                else:
                                    _arr_all_s = np.zeros(0, dtype=float)
                                _kl_all_s, _n_all_s = _kl_n_from_arr(_arr_all_s)
                                _per_cohort_metrics[f"kl_all_s{_s}"] = _kl_all_s
                                _per_cohort_metrics[f"n_all_s{_s}"] = _n_all_s
                    if _snap_rows and _act_freq.size:
                        _n_bins = int(_act_freq.size)
                        _unif_p = 1.0 / _n_bins
                        _subsets = {
                            "E": lambda s: s == "Employed",
                            "U": lambda s: s == "Unemployed",
                            "NILF": lambda s: s == "NILF",
                            "U+NILF": lambda s: s in ("Unemployed", "NILF"),
                        }
                        for _tag, _pred in _subsets.items():
                            _acts = [
                                int(_r["search_action"])
                                for _r in _snap_rows
                                if _r.get("search_action") is not None
                                and _pred(str(_r.get("employment_status", "")))
                            ]
                            _n = len(_acts)
                            if _n == 0:
                                continue
                            _bc = np.bincount(_acts, minlength=_n_bins).astype(float)
                            _p = _bc / _bc.sum()
                            _mp = float(_p.max())
                            _kl = float(np.sum(_p * (np.log(_p + 1e-12) - np.log(_unif_p))))
                    if not globals().get("_per_skill_sanity_checked", False):
                        globals()["_per_skill_sanity_checked"] = True
                    metrics_entry_extras = {
                        "worker_res_wage_freq": rw_freq,
                        "worker_search_entropy": (
                            float(search_ent) if not np.isnan(search_ent) else 0.0
                        ),
                        "worker_res_wage_entropy": (
                            float(res_wage_ent) if not np.isnan(res_wage_ent) else 0.0
                        ),
                        "worker_search_max_prob": _search_max_prob,
                        "worker_search_kl_uniform": _search_kl_uniform,
                        "per_skill_search_max_prob": _per_skill_search_max_prob,
                        "per_skill_search_kl_uniform": _per_skill_search_kl_uniform,
                        "matches_per_step_mean": float(matches_mean),
                        "matches_per_step_total": int(matches_total),
                        "worker_decision_p_search_mean": rollout_stats.get(
                            "worker_decision_p_search_mean"
                        ),
                        "worker_decision_p_search_std_within_batch": rollout_stats.get(
                            "worker_decision_p_search_std_within_batch"
                        ),
                        "worker_decision_head_entropy": rollout_stats.get(
                            "worker_decision_head_entropy"
                        ),
                        "worker_decision_logit_spread": rollout_stats.get(
                            "worker_decision_logit_spread"
                        ),
                        "worker_decision_p_search_mean_stuck_cohort": rollout_stats.get(
                            "worker_decision_p_search_mean_stuck_cohort"
                        ),
                        "worker_decision_p_search_std_within_batch_stuck_cohort": rollout_stats.get(
                            "worker_decision_p_search_std_within_batch_stuck_cohort"
                        ),
                        "worker_decision_head_entropy_stuck_cohort": rollout_stats.get(
                            "worker_decision_head_entropy_stuck_cohort"
                        ),
                        "worker_decision_logit_spread_stuck_cohort": rollout_stats.get(
                            "worker_decision_logit_spread_stuck_cohort"
                        ),
                        "worker_decision_stuck_cohort_size_mean": rollout_stats.get(
                            "worker_decision_stuck_cohort_size_mean"
                        ),
                        "env_retire_audit_observed_55_59": rollout_stats.get(
                            "env_retire_audit_observed_55_59"
                        ),
                        "env_retire_audit_observed_60_64": rollout_stats.get(
                            "env_retire_audit_observed_60_64"
                        ),
                        "env_retire_audit_observed_65plus": rollout_stats.get(
                            "env_retire_audit_observed_65plus"
                        ),
                        "env_retire_audit_expected_55_59": rollout_stats.get(
                            "env_retire_audit_expected_55_59"
                        ),
                        "env_retire_audit_expected_60_64": rollout_stats.get(
                            "env_retire_audit_expected_60_64"
                        ),
                        "env_retire_audit_expected_65plus": rollout_stats.get(
                            "env_retire_audit_expected_65plus"
                        ),
                        "env_retire_audit_ratio_55_59": rollout_stats.get(
                            "env_retire_audit_ratio_55_59"
                        ),
                        "env_retire_audit_ratio_60_64": rollout_stats.get(
                            "env_retire_audit_ratio_60_64"
                        ),
                        "env_retire_audit_ratio_65plus": rollout_stats.get(
                            "env_retire_audit_ratio_65plus"
                        ),
                        "env_retire_audit_employed_55_59_pm": rollout_stats.get(
                            "env_retire_audit_employed_55_59_pm"
                        ),
                        "env_retire_audit_employed_60_64_pm": rollout_stats.get(
                            "env_retire_audit_employed_60_64_pm"
                        ),
                        "env_retire_audit_employed_65plus_pm": rollout_stats.get(
                            "env_retire_audit_employed_65plus_pm"
                        ),
                        "env_retire_audit_gate_routed_55_64": rollout_stats.get(
                            "env_retire_audit_gate_routed_55_64"
                        ),
                        "env_retire_audit_stock_55_64_frac_mean": rollout_stats.get(
                            "env_retire_audit_stock_55_64_frac_mean"
                        ),
                    }
                    try:
                        if isinstance(_per_cohort_metrics, dict) and _per_cohort_metrics:
                            metrics_entry_extras.update(_per_cohort_metrics)
                    except NameError:
                        pass
                    try:
                        snap = rollout_stats.get("worker_snapshot", []) or []
                        if snap:
                            snap_path = Path(args.output_dir) / "res_wage_snapshots.csv"
                            write_header = not snap_path.exists()
                            with open(snap_path, "a", newline="") as f:
                                import csv

                                writer = csv.writer(f)
                                if write_header:
                                    writer.writerow(
                                        [
                                            "update",
                                            "worker_id",
                                            "skill_level",
                                            "education",
                                            "employment_status",
                                            "unemployment_months",
                                            "learned_v0",
                                            "search_action",
                                            "search_decision",
                                            "search_channel",
                                            "search_intensity",
                                        ]
                                    )
                                u = global_update_idx + 1
                                for row in snap:
                                    writer.writerow(
                                        [
                                            u,
                                            row.get("worker_id"),
                                            row.get("skill_level"),
                                            row.get("education"),
                                            row.get("employment_status"),
                                            row.get("unemployment_months"),
                                            row.get("learned_v0"),
                                            row.get("search_action"),
                                            row.get("search_decision"),
                                            row.get("search_channel"),
                                            row.get("search_intensity"),
                                        ]
                                    )
                    except Exception as _e:
                        logger.warning(f"Could not write the worker snapshot: {_e}")
                    metrics_entry = {
                        "update": global_update_idx + 1,
                        "outer_iter": outer_iter + 1 if use_mean_field else 0,
                        "timestep": trainer.total_steps,
                        "update_time": update_time,
                        "n_employed": n_employed,
                        "n_unemployed": n_unemployed,
                        "n_nilf": n_nilf,
                        "employment_rate": emp_rate,
                        "unemployment_rate": unemp_rate,
                        "nilf_rate": nilf_rate,
                        **metrics_entry_extras,
                        "worker_action_freq": rollout_stats.get("worker_action_freq", []),
                        "firm_action_freq": rollout_stats.get("firm_action_freq", []),
                        "wage_sample": rollout_stats.get("wage_sample", []),
                        "tenure_sample": rollout_stats.get("tenure_sample", []),
                        "wages_by_industry": rollout_stats.get("wages_by_industry", {}),
                        "firm_wage_strategy_argmax_by_industry": rollout_stats.get(
                            "firm_wage_strategy_argmax_by_industry", {}
                        ),
                        "firm_wage_strategy_entropy_by_industry": rollout_stats.get(
                            "firm_wage_strategy_entropy_by_industry", {}
                        ),
                        "firm_vacancy_management_argmax_by_industry": rollout_stats.get(
                            "firm_vacancy_management_argmax_by_industry", {}
                        ),
                        "firm_posted_wage_mean_by_industry": rollout_stats.get(
                            "firm_posted_wage_mean_by_industry", {}
                        ),
                        "firm_posted_wage_std_by_industry": rollout_stats.get(
                            "firm_posted_wage_std_by_industry", {}
                        ),
                        "firm_posted_wage_p90_p10_ratio": rollout_stats.get(
                            "firm_posted_wage_p90_p10_ratio", float("nan")
                        ),
                        "firm_posted_wage_deciles": rollout_stats.get(
                            "firm_posted_wage_deciles", [float("nan")] * 9
                        ),
                        "realized_wage_gini": rollout_stats.get("realized_wage_gini", float("nan")),
                        "realized_wage_p90_p10_ratio": rollout_stats.get(
                            "realized_wage_p90_p10_ratio", float("nan")
                        ),
                        "realized_wage_sample_is_population": rollout_stats.get(
                            "realized_wage_sample_is_population", False
                        ),
                        **{
                            k: float(v) if isinstance(v, (int, float, np.number)) else v
                            for k, v in metrics.items()
                        },
                    }
                    m_stats = getattr(env, "last_matching_stats", {}) or {}
                    metrics_entry["applications"] = m_stats.get("applications", 0)
                    metrics_entry["offers"] = m_stats.get("offers", m_stats.get("n_offers_sent", 0))
                    metrics_entry["matches"] = m_stats.get("matches", 0)
                    metrics_entry["separations"] = m_stats.get("separations", 0)
                    metrics_entry["total_vacancies"] = m_stats.get(
                        "total_vacancies", len(env.state_manager.vacancies)
                    )
                    metrics_entry["j2j_offered"] = m_stats.get("j2j_offered", 0)
                    metrics_entry["j2j_accepted"] = m_stats.get("j2j_accepted", 0)
                    metrics_entry["j2j_rejected_wage"] = m_stats.get("j2j_rejected_wage", 0)
                    metrics_entry["employed_searching"] = m_stats.get("employed_searching", 0)
                    metrics_entry["unemployed_active"] = m_stats.get("unemployed_active", 0)
                    _v_cnt = metrics_entry.get("total_vacancies", 0) or 0
                    _e_cnt = metrics_entry.get("n_employed", 0) or 0
                    metrics_entry["vacancy_ratio_V_over_EV"] = (
                        float(_v_cnt) / float(_e_cnt + _v_cnt)
                        if _e_cnt + _v_cnt > 0
                        else float("nan")
                    )
                    f_decomp = getattr(env, "_firm_reward_decomp", None)
                    if f_decomp:
                        _decomp_map = {
                            "employee_value": "firm_emp_val",
                            "vacancy_cost": "firm_vac_cost",
                            "hiring": "firm_hiring",
                            "separation": "firm_separation",
                            "staffing_penalty": "firm_staffing_penalty",
                            "dense_understaff": "firm_dense_under",
                            "dense_overstaff": "firm_dense_over",
                            "dense_hire": "firm_dense_hire",
                            "excess_vacancy": "firm_excess_vac",
                            "growth": "firm_growth",
                            "adjustment_cost": "firm_adjustment_cost",
                        }
                        for src_key, csv_col in _decomp_map.items():
                            metrics_entry[csv_col] = f_decomp.get(src_key, 0.0)
                    metrics_history.append(metrics_entry)
                if (global_update_idx + 1) % 10 == 0:
                    checkpoint_path = output_path / f"checkpoint_{global_update_idx + 1:04d}.pt"
                    trainer.save(str(checkpoint_path))
                    logger.info(f"  Saved checkpoint: {checkpoint_path.name}")
                    import pandas as pd

                    pd.DataFrame(metrics_history).to_csv(
                        output_path / "metrics_history.csv", index=False
                    )
                if (global_update_idx + 1) % 50 == 0:
                    try:
                        _snap_training_log = {
                            "metrics_history": metrics_history,
                            "config": vars(args),
                            "total_updates": len(metrics_history),
                            "total_timesteps": args.total_timesteps,
                            "update_times": update_times,
                            "mean_field_enabled": use_mean_field,
                            "theta_distances": theta_distances,
                            "mean_field_outer_iterations": (
                                len(theta_distances) if use_mean_field else 0
                            ),
                            "worker_value_loss": [
                                m.get("worker_value_loss", 0) for m in metrics_history
                            ],
                            "firm_value_loss": [
                                m.get("firm_value_loss", 0) for m in metrics_history
                            ],
                            "worker_policy_loss": [
                                m.get("worker_policy_loss", 0) for m in metrics_history
                            ],
                            "firm_policy_loss": [
                                m.get("firm_policy_loss", 0) for m in metrics_history
                            ],
                            "worker_entropy": [m.get("worker_entropy", 0) for m in metrics_history],
                            "firm_entropy": [m.get("firm_entropy", 0) for m in metrics_history],
                            "worker_kl": [m.get("worker_kl", 0) for m in metrics_history],
                            "firm_kl": [m.get("firm_kl", 0) for m in metrics_history],
                            "worker_explained_var": [
                                m.get("worker_explained_var", 0) for m in metrics_history
                            ],
                            "firm_explained_var": [
                                m.get("firm_explained_var", 0) for m in metrics_history
                            ],
                            "worker_reward": [
                                m.get("worker/mean_reward_raw", 0) for m in metrics_history
                            ],
                            "firm_reward": [
                                m.get("firm/mean_reward_raw", 0) for m in metrics_history
                            ],
                            "worker_action_freq": [
                                m.get("worker_action_freq", []) for m in metrics_history
                            ],
                            "firm_action_freq": [
                                m.get("firm_action_freq", []) for m in metrics_history
                            ],
                            "firm_wage_strategy_argmax_by_industry": [
                                m.get("firm_wage_strategy_argmax_by_industry", {})
                                for m in metrics_history
                            ],
                            "firm_wage_strategy_entropy_by_industry": [
                                m.get("firm_wage_strategy_entropy_by_industry", {})
                                for m in metrics_history
                            ],
                            "firm_vacancy_management_argmax_by_industry": [
                                m.get("firm_vacancy_management_argmax_by_industry", {})
                                for m in metrics_history
                            ],
                            "firm_posted_wage_mean_by_industry": [
                                m.get("firm_posted_wage_mean_by_industry", {})
                                for m in metrics_history
                            ],
                            "firm_posted_wage_std_by_industry": [
                                m.get("firm_posted_wage_std_by_industry", {})
                                for m in metrics_history
                            ],
                            "firm_posted_wage_p90_p10_ratio": [
                                m.get("firm_posted_wage_p90_p10_ratio", 0) for m in metrics_history
                            ],
                            "worker_value_pred_errors": [
                                m.get("worker_value_pred_errors", []) for m in metrics_history
                            ],
                            "firm_value_pred_errors": [
                                m.get("firm_value_pred_errors", []) for m in metrics_history
                            ],
                            "wage_samples": [m.get("wage_sample", []) for m in metrics_history],
                            "tenure_samples": [m.get("tenure_sample", []) for m in metrics_history],
                            "wages_by_industry": [
                                m.get("wages_by_industry", {}) for m in metrics_history
                            ],
                        }
                        _snap_serializable = convert_to_serializable(_snap_training_log)
                        with open(output_path / "training_log.json", "w") as f:
                            json.dump(_snap_serializable, f, indent=2)
                        _snap_sim_log = {
                            "final_state": {
                                "n_workers": args.n_workers,
                                "n_firms": args.n_firms,
                                "employment_rate": metrics_history[-1].get("employment_rate", 0),
                                "unemployment_rate": metrics_history[-1].get(
                                    "unemployment_rate", 0
                                ),
                                "nilf_rate": metrics_history[-1].get("nilf_rate", 0),
                            },
                            "snapshot_update": global_update_idx + 1,
                            "note": "Periodic snapshot (every 100 updates)",
                        }
                        with open(output_path / "simulation_log.json", "w") as f:
                            json.dump(convert_to_serializable(_snap_sim_log), f, indent=2)
                    except Exception as e:
                        logger.warning(f"  Could not save the snapshot: {e}")
                global_update_idx += 1
            if use_mean_field:
                env.state_manager.unfreeze_theta()
                for _ in range(3):
                    with torch.no_grad():
                        trainer.collect_rollout()
                eval_theta = env.state_manager.compute_firm_mean_field()
                theta_diff = compute_theta_distance(current_theta, eval_theta)
                theta_distances.append(theta_diff)
            if global_update_idx >= expected_updates:
                break
        total_time = time.time() - start_time
        logger.info(f"Training completed in {timedelta(seconds=int(total_time))}")
        trainer.save(str(output_path / "final_checkpoint.pt"))
        import pandas as pd

        pd.DataFrame(metrics_history).to_csv(output_path / "metrics_history.csv", index=False)
        metrics_serializable = convert_to_serializable(metrics_history)
        with open(output_path / "metrics_history.json", "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        training_log = {
            "metrics_history": metrics_history,
            "config": vars(args),
            "total_updates": len(metrics_history),
            "total_timesteps": args.total_timesteps,
            "update_times": update_times,
            "mean_field_enabled": use_mean_field,
            "theta_distances": theta_distances,
            "mean_field_outer_iterations": len(theta_distances) if use_mean_field else 0,
            "worker_value_loss": [m.get("worker_value_loss", 0) for m in metrics_history],
            "firm_value_loss": [m.get("firm_value_loss", 0) for m in metrics_history],
            "worker_policy_loss": [m.get("worker_policy_loss", 0) for m in metrics_history],
            "firm_policy_loss": [m.get("firm_policy_loss", 0) for m in metrics_history],
            "worker_entropy": [m.get("worker_entropy", 0) for m in metrics_history],
            "firm_entropy": [m.get("firm_entropy", 0) for m in metrics_history],
            "worker_kl": [m.get("worker_kl", 0) for m in metrics_history],
            "firm_kl": [m.get("firm_kl", 0) for m in metrics_history],
            "worker_explained_var": [m.get("worker_explained_var", 0) for m in metrics_history],
            "firm_explained_var": [m.get("firm_explained_var", 0) for m in metrics_history],
            "worker_reward": [m.get("worker/mean_reward_raw", 0) for m in metrics_history],
            "firm_reward": [m.get("firm/mean_reward_raw", 0) for m in metrics_history],
            "worker_action_freq": [m.get("worker_action_freq", []) for m in metrics_history],
            "firm_action_freq": [m.get("firm_action_freq", []) for m in metrics_history],
            "firm_wage_strategy_argmax_by_industry": [
                m.get("firm_wage_strategy_argmax_by_industry", {}) for m in metrics_history
            ],
            "firm_wage_strategy_entropy_by_industry": [
                m.get("firm_wage_strategy_entropy_by_industry", {}) for m in metrics_history
            ],
            "firm_vacancy_management_argmax_by_industry": [
                m.get("firm_vacancy_management_argmax_by_industry", {}) for m in metrics_history
            ],
            "firm_posted_wage_mean_by_industry": [
                m.get("firm_posted_wage_mean_by_industry", {}) for m in metrics_history
            ],
            "firm_posted_wage_std_by_industry": [
                m.get("firm_posted_wage_std_by_industry", {}) for m in metrics_history
            ],
            "firm_posted_wage_p90_p10_ratio": [
                m.get("firm_posted_wage_p90_p10_ratio", 0) for m in metrics_history
            ],
            "worker_value_pred_errors": [
                m.get("worker_value_pred_errors", []) for m in metrics_history
            ],
            "firm_value_pred_errors": [
                m.get("firm_value_pred_errors", []) for m in metrics_history
            ],
            "wage_samples": [m.get("wage_sample", []) for m in metrics_history],
            "tenure_samples": [m.get("tenure_sample", []) for m in metrics_history],
            "wages_by_industry": [m.get("wages_by_industry", {}) for m in metrics_history],
        }
        training_log_serializable = convert_to_serializable(training_log)
        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_log_serializable, f, indent=2)
        simulation_log = {
            "final_state": {
                "n_workers": args.n_workers,
                "n_firms": args.n_firms,
                "employment_rate": (
                    metrics_history[-1].get("employment_rate", 0) if metrics_history else 0
                ),
                "unemployment_rate": (
                    metrics_history[-1].get("unemployment_rate", 0) if metrics_history else 0
                ),
                "nilf_rate": metrics_history[-1].get("nilf_rate", 0) if metrics_history else 0,
            },
            "note": "Full simulation_log requires running validation test with micro/meso/macro metrics",
        }
        simulation_log_serializable = convert_to_serializable(simulation_log)
        with open(output_path / "simulation_log.json", "w") as f:
            json.dump(simulation_log_serializable, f, indent=2)
        env.state_manager.export_state(str(output_path / "final_state.pkl"))
        logger.info(f"All outputs saved to: {output_path}")
        logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
        logger.info(f"Average time per update: {np.mean(update_times):.1f}s")
        if metrics_history:
            final = metrics_history[-1]
            logger.info(f"Final employment rate: {final['employment_rate']:.1f}%")
            logger.info(f"Final unemployment rate: {final['unemployment_rate']:.1f}%")
            logger.info(f"Final share outside the labour force: {final['nilf_rate']:.1f}%")
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        if metrics_history:
            import pandas as pd

            pd.DataFrame(metrics_history).to_csv(
                output_path / "metrics_history_partial.csv", index=False
            )
        trainer.save(str(output_path / "interrupted_checkpoint.pt"))
        if metrics_history:
            try:
                _int_training_log = {
                    "metrics_history": metrics_history,
                    "config": vars(args),
                    "total_updates": len(metrics_history),
                    "total_timesteps": args.total_timesteps,
                    "update_times": update_times,
                    "mean_field_enabled": use_mean_field,
                    "theta_distances": theta_distances,
                    "mean_field_outer_iterations": len(theta_distances) if use_mean_field else 0,
                    "worker_value_loss": [m.get("worker_value_loss", 0) for m in metrics_history],
                    "firm_value_loss": [m.get("firm_value_loss", 0) for m in metrics_history],
                    "worker_policy_loss": [m.get("worker_policy_loss", 0) for m in metrics_history],
                    "firm_policy_loss": [m.get("firm_policy_loss", 0) for m in metrics_history],
                    "worker_entropy": [m.get("worker_entropy", 0) for m in metrics_history],
                    "firm_entropy": [m.get("firm_entropy", 0) for m in metrics_history],
                    "worker_kl": [m.get("worker_kl", 0) for m in metrics_history],
                    "firm_kl": [m.get("firm_kl", 0) for m in metrics_history],
                    "worker_explained_var": [
                        m.get("worker_explained_var", 0) for m in metrics_history
                    ],
                    "firm_explained_var": [m.get("firm_explained_var", 0) for m in metrics_history],
                    "worker_reward": [m.get("worker/mean_reward_raw", 0) for m in metrics_history],
                    "firm_reward": [m.get("firm/mean_reward_raw", 0) for m in metrics_history],
                    "worker_action_freq": [
                        m.get("worker_action_freq", []) for m in metrics_history
                    ],
                    "firm_action_freq": [m.get("firm_action_freq", []) for m in metrics_history],
                    "firm_wage_strategy_argmax_by_industry": [
                        m.get("firm_wage_strategy_argmax_by_industry", {}) for m in metrics_history
                    ],
                    "firm_wage_strategy_entropy_by_industry": [
                        m.get("firm_wage_strategy_entropy_by_industry", {}) for m in metrics_history
                    ],
                    "firm_vacancy_management_argmax_by_industry": [
                        m.get("firm_vacancy_management_argmax_by_industry", {})
                        for m in metrics_history
                    ],
                    "firm_posted_wage_mean_by_industry": [
                        m.get("firm_posted_wage_mean_by_industry", {}) for m in metrics_history
                    ],
                    "firm_posted_wage_std_by_industry": [
                        m.get("firm_posted_wage_std_by_industry", {}) for m in metrics_history
                    ],
                    "firm_posted_wage_p90_p10_ratio": [
                        m.get("firm_posted_wage_p90_p10_ratio", 0) for m in metrics_history
                    ],
                    "worker_value_pred_errors": [
                        m.get("worker_value_pred_errors", []) for m in metrics_history
                    ],
                    "firm_value_pred_errors": [
                        m.get("firm_value_pred_errors", []) for m in metrics_history
                    ],
                    "wage_samples": [m.get("wage_sample", []) for m in metrics_history],
                    "tenure_samples": [m.get("tenure_sample", []) for m in metrics_history],
                    "wages_by_industry": [m.get("wages_by_industry", {}) for m in metrics_history],
                    "note": "Interrupted by user (KeyboardInterrupt)",
                }
                with open(output_path / "training_log_interrupted.json", "w") as f:
                    json.dump(convert_to_serializable(_int_training_log), f, indent=2)
                _int_sim_log = {
                    "final_state": {
                        "n_workers": args.n_workers,
                        "n_firms": args.n_firms,
                        "employment_rate": metrics_history[-1].get("employment_rate", 0),
                        "unemployment_rate": metrics_history[-1].get("unemployment_rate", 0),
                        "nilf_rate": metrics_history[-1].get("nilf_rate", 0),
                    },
                    "snapshot_update": len(metrics_history),
                    "note": "Interrupted by user (KeyboardInterrupt)",
                }
                with open(output_path / "simulation_log_interrupted.json", "w") as f:
                    json.dump(convert_to_serializable(_int_sim_log), f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save the logs of the interrupted run: {e}")
        logger.info(f"Partial results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
