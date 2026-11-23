import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, List, Optional
import numpy as np
import random as _py_random
from collections import defaultdict
import logging
import gc
from .actor_critic import FirmActorCritic, FactorizedWorkerActorCritic

logger = logging.getLogger(__name__)
BR_SKILL_COL = 30


def _retirement_audit_rollout_dict(env) -> dict:
    NAN_KEYS = {
        "env_retire_audit_observed_55_59": float("nan"),
        "env_retire_audit_observed_60_64": float("nan"),
        "env_retire_audit_observed_65plus": float("nan"),
        "env_retire_audit_expected_55_59": float("nan"),
        "env_retire_audit_expected_60_64": float("nan"),
        "env_retire_audit_expected_65plus": float("nan"),
        "env_retire_audit_ratio_55_59": float("nan"),
        "env_retire_audit_ratio_60_64": float("nan"),
        "env_retire_audit_ratio_65plus": float("nan"),
        "env_retire_audit_employed_55_59_pm": float("nan"),
        "env_retire_audit_employed_60_64_pm": float("nan"),
        "env_retire_audit_employed_65plus_pm": float("nan"),
        "env_retire_audit_gate_routed_55_64": float("nan"),
        "env_retire_audit_stock_55_64_frac_mean": float("nan"),
    }
    return NAN_KEYS


class RolloutBuffer:

    def __init__(
        self,
        buffer_size: int,
        n_workers: int,
        n_firms: int,
        worker_obs_dim: int,
        firm_obs_dim: int,
        global_state_dim: int,
        device: str = "cuda",
        n_visible_vacancies: int = 15,
        vacancy_dim: int = 20,
        history_length: int = 36,
        history_dim: int = 5,
        endogenous_retirement: bool = False,
    ):
        self.buffer_size = buffer_size
        self.n_workers = n_workers
        self.n_firms = n_firms
        self.device = device
        self.worker_obs = torch.zeros(buffer_size, n_workers, worker_obs_dim, device=device)
        self.worker_actions = torch.zeros(
            buffer_size, n_workers, 2, dtype=torch.long, device=device
        )
        self.worker_log_probs = torch.zeros(buffer_size, n_workers, device=device)
        self.worker_rewards = torch.zeros(buffer_size, n_workers, device=device)
        self.worker_values = torch.zeros(buffer_size, n_workers, device=device)
        self.worker_dones = torch.zeros(buffer_size, n_workers, device=device)
        self.endogenous_retirement = bool(endogenous_retirement)
        if self.endogenous_retirement:
            self.worker_retire_eligible_mask = torch.zeros(
                buffer_size, n_workers, dtype=torch.bool, device=device
            )
        else:
            self.worker_retire_eligible_mask = None
        self.worker_context = torch.zeros(
            buffer_size, n_workers, n_visible_vacancies, vacancy_dim, device=device
        )
        self.worker_history = torch.zeros(
            buffer_size, n_workers, history_length, history_dim, device=device
        )
        self.firm_obs = torch.zeros(buffer_size, n_firms, firm_obs_dim, device=device)
        self.firm_actions = torch.zeros(buffer_size, n_firms, dtype=torch.long, device=device)
        self.firm_log_probs = torch.zeros(buffer_size, n_firms, device=device)
        self.firm_rewards = torch.zeros(buffer_size, n_firms, device=device)
        self.firm_values = torch.zeros(buffer_size, n_firms, device=device)
        self.firm_dones = torch.zeros(buffer_size, n_firms, device=device)
        self.global_states = torch.zeros(buffer_size, global_state_dim, device=device)
        self.worker_advantages = torch.zeros(buffer_size, n_workers, device=device)
        self.worker_returns = torch.zeros(buffer_size, n_workers, device=device)
        self.firm_advantages = torch.zeros(buffer_size, n_firms, device=device)
        self.firm_returns = torch.zeros(buffer_size, n_firms, device=device)
        self.truncated = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.boundary_worker_values: Dict[int, torch.Tensor] = {}
        self.boundary_firm_values: Dict[int, torch.Tensor] = {}
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    def add(
        self,
        worker_obs: torch.Tensor,
        worker_action: torch.Tensor,
        worker_log_prob: torch.Tensor,
        worker_reward: torch.Tensor,
        worker_value: torch.Tensor,
        worker_done: torch.Tensor,
        worker_context: torch.Tensor,
        worker_history: torch.Tensor,
        firm_obs: torch.Tensor,
        firm_action: torch.Tensor,
        firm_log_prob: torch.Tensor,
        firm_reward: torch.Tensor,
        firm_value: torch.Tensor,
        firm_done: torch.Tensor,
        global_state: torch.Tensor,
        worker_retire_eligible_mask: Optional[torch.Tensor] = None,
    ):
        self.worker_obs[self.ptr] = worker_obs
        self.worker_actions[self.ptr] = worker_action
        self.worker_log_probs[self.ptr] = worker_log_prob
        self.worker_rewards[self.ptr] = worker_reward
        self.worker_values[self.ptr] = worker_value
        self.worker_dones[self.ptr] = worker_done
        self.worker_context[self.ptr] = worker_context
        self.worker_history[self.ptr] = worker_history
        self.firm_obs[self.ptr] = firm_obs
        self.firm_actions[self.ptr] = firm_action
        self.firm_log_probs[self.ptr] = firm_log_prob
        self.firm_rewards[self.ptr] = firm_reward
        self.firm_values[self.ptr] = firm_value
        self.firm_dones[self.ptr] = firm_done
        self.global_states[self.ptr] = global_state
        if self.endogenous_retirement:
            if worker_retire_eligible_mask is None:
                raise RuntimeError(
                    "The rollout buffer needs worker_retire_eligible_mask. The rollout collector computes it at the step where the policy sampled, so the update uses the same mask."
                )
            self.worker_retire_eligible_mask[self.ptr] = worker_retire_eligible_mask
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def mark_truncation(
        self, t: int, boundary_worker_value: torch.Tensor, boundary_firm_value: torch.Tensor
    ):
        self.truncated[t] = True
        self.boundary_worker_values[t] = boundary_worker_value.detach().to(self.device)
        self.boundary_firm_values[t] = boundary_firm_value.detach().to(self.device)

    def compute_returns_and_advantages(
        self,
        last_worker_value: torch.Tensor,
        last_firm_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        last_gae_worker = 0
        for t in reversed(range(self.ptr)):
            if t in self.boundary_worker_values:
                delta = (
                    self.worker_rewards[t]
                    + gamma * self.boundary_worker_values[t]
                    - self.worker_values[t]
                )
                last_gae_worker = delta
                self.worker_advantages[t] = last_gae_worker
                continue
            if t == self.ptr - 1:
                next_value = last_worker_value
            else:
                next_value = self.worker_values[t + 1]
            next_non_terminal = 1.0 - self.worker_dones[t]
            delta = (
                self.worker_rewards[t]
                + gamma * next_value * next_non_terminal
                - self.worker_values[t]
            )
            last_gae_worker = delta + gamma * gae_lambda * next_non_terminal * last_gae_worker
            self.worker_advantages[t] = last_gae_worker
        self.worker_returns[: self.ptr] = (
            self.worker_advantages[: self.ptr] + self.worker_values[: self.ptr]
        )
        last_gae_firm = 0
        for t in reversed(range(self.ptr)):
            if t in self.boundary_firm_values:
                delta = (
                    self.firm_rewards[t]
                    + gamma * self.boundary_firm_values[t]
                    - self.firm_values[t]
                )
                last_gae_firm = delta
                self.firm_advantages[t] = last_gae_firm
                continue
            if t == self.ptr - 1:
                next_value = last_firm_value
            else:
                next_value = self.firm_values[t + 1]
            next_non_terminal = 1.0 - self.firm_dones[t]
            delta = (
                self.firm_rewards[t] + gamma * next_value * next_non_terminal - self.firm_values[t]
            )
            last_gae_firm = delta + gamma * gae_lambda * next_non_terminal * last_gae_firm
            self.firm_advantages[t] = last_gae_firm
        self.firm_returns[: self.ptr] = (
            self.firm_advantages[: self.ptr] + self.firm_values[: self.ptr]
        )

    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        batches = []
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_indices = indices[start:end]
            batch = {
                "worker_obs": self.worker_obs[batch_indices],
                "worker_actions": self.worker_actions[batch_indices],
                "worker_log_probs": self.worker_log_probs[batch_indices],
                "worker_values": self.worker_values[batch_indices],
                "worker_advantages": self.worker_advantages[batch_indices],
                "worker_returns": self.worker_returns[batch_indices],
                "worker_context": self.worker_context[batch_indices],
                "worker_history": self.worker_history[batch_indices],
                "firm_obs": self.firm_obs[batch_indices],
                "firm_actions": self.firm_actions[batch_indices],
                "firm_log_probs": self.firm_log_probs[batch_indices],
                "firm_values": self.firm_values[batch_indices],
                "firm_advantages": self.firm_advantages[batch_indices],
                "firm_returns": self.firm_returns[batch_indices],
                "global_states": self.global_states[batch_indices],
            }
            if self.endogenous_retirement and self.worker_retire_eligible_mask is not None:
                batch["worker_retire_eligible_mask"] = self.worker_retire_eligible_mask[
                    batch_indices
                ]
            batches.append(batch)
        return batches

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False
        self.truncated.zero_()
        self.boundary_worker_values.clear()
        self.boundary_firm_values.clear()


def _non_encoder_params(module):
    return [p for n, p in module.named_parameters() if not n.startswith("encoder.")]


class DTMAPPO:

    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.device = config.training.device
        self.worker_ac = FactorizedWorkerActorCritic(config).to(self.device)
        self.firm_ac = FirmActorCritic(config).to(self.device)
        actor_lr = getattr(config.training, "actor_lr", 0.0003)
        critic_lr = getattr(config.training, "critic_lr", 0.001)
        w_actor_lr = getattr(config.training, "worker_actor_lr", None) or actor_lr
        w_critic_lr = getattr(config.training, "worker_critic_lr", None) or critic_lr
        f_actor_lr = getattr(config.training, "firm_actor_lr", None) or actor_lr
        f_critic_lr = getattr(config.training, "firm_critic_lr", None) or critic_lr
        w_encoder_lr = (w_actor_lr + w_critic_lr) / 2
        self.worker_encoder_optimizer = Adam(self.worker_ac.encoder.parameters(), lr=w_encoder_lr)
        self.worker_actor_optimizer = Adam(_non_encoder_params(self.worker_ac.actor), lr=w_actor_lr)
        self._warmup_active = False
        self._warmup_updates = 0
        self._warmup_start_frac = 1.0
        self._warmup_judgment_start = 0
        self.worker_critic_optimizer = Adam(
            _non_encoder_params(self.worker_ac.critic), lr=w_critic_lr
        )
        f_encoder_lr = (f_actor_lr + f_critic_lr) / 2
        self.firm_encoder_optimizer = Adam(self.firm_ac.encoder.parameters(), lr=f_encoder_lr)
        self.firm_actor_optimizer = Adam(_non_encoder_params(self.firm_ac.actor), lr=f_actor_lr)
        self.firm_critic_optimizer = Adam(_non_encoder_params(self.firm_ac.critic), lr=f_critic_lr)
        total_steps = config.training.total_timesteps // config.training.rollout_steps
        self.worker_actor_scheduler = LinearLR(
            self.worker_actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
        self.worker_critic_scheduler = LinearLR(
            self.worker_critic_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
        self.firm_actor_scheduler = LinearLR(
            self.firm_actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
        from torch.optim.lr_scheduler import LambdaLR as _LambdaLR

        self.firm_critic_scheduler = _LambdaLR(
            self.firm_critic_optimizer, lr_lambda=lambda epoch: 1.0
        )
        env_cfg = config.environment
        n_workers = env.n_workers
        n_firms = env.n_firms
        actual_worker_obs_dim = env.fast_obs.state_table.worker_obs_dim
        actual_firm_obs_dim = env.fast_obs.state_table.firm_obs_dim
        self.identity_aware_ppo = getattr(env, "_net_population", None) is not None
        buffer_class = RolloutBuffer
        if self.identity_aware_ppo:
            from .identity_joint_buffer import JointIdentityRolloutBuffer

            buffer_class = JointIdentityRolloutBuffer
        self.buffer = buffer_class(
            buffer_size=config.training.rollout_steps,
            n_workers=n_workers,
            n_firms=n_firms,
            worker_obs_dim=actual_worker_obs_dim,
            firm_obs_dim=actual_firm_obs_dim,
            global_state_dim=getattr(
                env_cfg, "effective_global_state_dim", env_cfg.global_state_dim
            ),
            device=self.device,
            n_visible_vacancies=env_cfg.n_visible_vacancies,
            vacancy_dim=env_cfg.vacancy_feature_dim,
            history_length=env_cfg.history_length,
            history_dim=env_cfg.history_dim,
            endogenous_retirement=True,
        )
        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0
        self._last_obs = None
        self._worker_value_popart_initialized = False
        self._worker_ret_mean_ema = torch.zeros((), device=self.device)
        self._worker_ret_std_ema = torch.ones((), device=self.device)
        self._worker_popart_just_initialized = False
        self.stats = defaultdict(list)

    def _worker_popart_enabled(self) -> bool:
        return True

    def _worker_value_denorm(self, v_norm: torch.Tensor) -> torch.Tensor:
        if not self._worker_popart_enabled():
            return v_norm
        return v_norm * self._worker_ret_std_ema + self._worker_ret_mean_ema

    def _worker_value_norm(self, v_raw: torch.Tensor) -> torch.Tensor:
        if not self._worker_popart_enabled():
            return v_raw
        return (v_raw - self._worker_ret_mean_ema) / self._worker_ret_std_ema

    def _rescale_worker_critic_output(
        self,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
    ) -> None:
        final_linear = self.worker_ac.critic.mlp[-1]
        dev = final_linear.weight.device
        old_mean_d = old_mean.to(dev)
        old_std_d = old_std.to(dev)
        new_mean_d = new_mean.to(dev)
        new_std_d = new_std.to(dev)
        ratio = old_std_d / new_std_d
        with torch.no_grad():
            final_linear.weight.mul_(ratio)
            final_linear.bias.copy_(
                (final_linear.bias * old_std_d + old_mean_d - new_mean_d) / new_std_d
            )

    def _update_worker_popart_stats(self, raw_returns: torch.Tensor) -> None:
        if not self._worker_popart_enabled():
            return
        alpha = float(getattr(self.config.training, "worker_value_popart_alpha", 0.01))
        min_std = float(getattr(self.config.training, "worker_value_popart_min_std", 1.0))
        batch_mean = raw_returns.mean().detach()
        batch_std = torch.clamp(raw_returns.std().detach(), min=min_std)
        if not self._worker_value_popart_initialized:
            self._worker_ret_mean_ema = batch_mean.to(self.device)
            self._worker_ret_std_ema = batch_std.to(self.device)
            self._worker_value_popart_initialized = True
            self._worker_popart_just_initialized = True
        else:
            old_mean = self._worker_ret_mean_ema
            old_std = self._worker_ret_std_ema
            new_mean = (1.0 - alpha) * old_mean + alpha * batch_mean
            new_std = (1.0 - alpha) * old_std + alpha * batch_std
            new_std = torch.clamp(new_std, min=min_std)
            self._rescale_worker_critic_output(old_mean, old_std, new_mean, new_std)
            self._worker_ret_mean_ema = new_mean
            self._worker_ret_std_ema = new_std

    def collect_rollout(self, progress_callback=None) -> Dict[str, float]:
        self.buffer.reset()
        _joint_identity = bool(getattr(self.buffer, "ragged_workers", False))
        if _joint_identity:
            from .identity_joint_buffer import worker_keys
        if self.total_steps == 0 or self._last_obs is None:
            obs_td = self.env.reset()
        else:
            obs_td = self._last_obs
        episode_rewards_workers = []
        episode_rewards_firms = []
        n_worker_actions = self.config.action.worker_action_size
        n_firm_actions = self.config.action.firm_action_size
        n_res_wage = getattr(self.config.action, "worker_res_wage_options", 5)
        worker_action_counts = torch.zeros(n_worker_actions, device=self.device)
        firm_action_counts = torch.zeros(n_firm_actions, device=self.device)
        worker_res_wage_counts = torch.zeros(n_res_wage, device=self.device)
        matches_per_step: list = []
        p_search_steps: list = []
        p_search_std_steps: list = []
        decision_entropy_steps: list = []
        decision_logit_spread_steps: list = []
        p_search_steps_cohort: list = []
        p_search_std_steps_cohort: list = []
        decision_entropy_steps_cohort: list = []
        decision_logit_spread_steps_cohort: list = []
        cohort_size_steps: list = []
        _COHORT_KEYS = ("Unemployed", "Employed", "NILF_eligible", "NILF_ineligible")
        _N_SKILL_BINS = 5
        cohort_action_counts = {
            k: torch.zeros(n_worker_actions, device=self.device) for k in _COHORT_KEYS
        }
        cohort_per_skill_action_counts = {
            k: torch.zeros(_N_SKILL_BINS, n_worker_actions, device=self.device)
            for k in _COHORT_KEYS
        }
        _subtype_enabled = bool(getattr(self.config.mnl, "nilf_subtype_enabled", False))
        _school_w = int(getattr(self.config.mnl, "school_search_eligible_months", 36))
        _family_w = int(getattr(self.config.mnl, "family_search_eligible_months", 24))
        _illness_w = int(getattr(self.config.mnl, "illness_search_eligible_months", 6))
        _SUBTYPE_NONE, _SUBTYPE_RETIREMENT = (0, 1)
        _SUBTYPE_SCHOOL, _SUBTYPE_ILLNESS = (2, 3)
        _SUBTYPE_FAMILY, _SUBTYPE_DISCOURAGED = (4, 5)
        for step in range(self.config.training.rollout_steps):
            with torch.no_grad():
                worker_obs = obs_td["worker_obs"].to(self.device)
                firm_obs = obs_td["firm_obs"].to(self.device)
                global_state = obs_td["global_state"].to(self.device)
                worker_context = obs_td.get("worker_context")
                if worker_context is not None:
                    worker_context = worker_context.to(self.device)
                worker_history = obs_td.get("worker_history")
                if worker_history is not None:
                    worker_history = worker_history.to(self.device)
                if _joint_identity:
                    worker_obs = worker_obs.detach().clone()
                    firm_obs = firm_obs.detach().clone()
                    global_state = global_state.detach().clone()
                    if worker_context is not None:
                        worker_context = worker_context.detach().clone()
                    if worker_history is not None:
                        worker_history = worker_history.detach().clone()
                worker_skill = worker_obs[:, BR_SKILL_COL]
                _retire_mask = None
                if getattr(self.env, "_endogenous_retirement_enabled", lambda: False)():
                    _retire_mask = self.env._compute_retire_eligible_mask()
                _worker_ac_kwargs = {
                    "context": worker_context,
                    "history": worker_history,
                    "skill_param": worker_skill,
                }
                if _retire_mask is not None:
                    _worker_ac_kwargs["retire_eligible_mask"] = _retire_mask
                worker_out = self.worker_ac.get_action_and_value(
                    worker_obs,
                    global_state.unsqueeze(0).expand(worker_obs.shape[0], -1),
                    **_worker_ac_kwargs,
                )
                if "p_search" in worker_out:
                    _p_search = worker_out["p_search"]
                    _dec_ent = worker_out["decision_entropy"]
                    _dec_spread = worker_out["decision_logit_spread"]
                    p_search_steps.append(float(_p_search.mean().item()))
                    p_search_std_steps.append(
                        float(_p_search.std().item()) if _p_search.numel() > 1 else 0.0
                    )
                    decision_entropy_steps.append(float(_dec_ent.mean().item()))
                    decision_logit_spread_steps.append(float(_dec_spread.abs().mean().item()))
                    try:
                        st = self.env.fast_obs.state_table
                        fo = self.env.fast_obs
                        if hasattr(fo, "compute_search_eligibility_mask"):
                            _eligible = fo.compute_search_eligibility_mask()
                        else:
                            _eligible = st.compute_search_eligibility_mask()
                        _age_band = (st.w_age >= 35) & (st.w_age <= 54)
                        _u_no_search = (st.w_status == 1) & ~st.w_is_searching
                        _nilf_eligible = (st.w_status == 2) & _eligible
                        _cohort_mask = (_age_band & (_u_no_search | _nilf_eligible)).to(self.device)
                        _n_cohort = int(_cohort_mask.sum().item())
                        cohort_size_steps.append(_n_cohort)
                        if _n_cohort > 0:
                            _p_c = _p_search[_cohort_mask]
                            _e_c = _dec_ent[_cohort_mask]
                            _s_c = _dec_spread[_cohort_mask]
                            p_search_steps_cohort.append(float(_p_c.mean().item()))
                            p_search_std_steps_cohort.append(
                                float(_p_c.std().item()) if _p_c.numel() > 1 else 0.0
                            )
                            decision_entropy_steps_cohort.append(float(_e_c.mean().item()))
                            decision_logit_spread_steps_cohort.append(
                                float(_s_c.abs().mean().item())
                            )
                    except Exception:
                        cohort_size_steps.append(0)
                if getattr(self.config.training, "freeze_firm_policy", False):
                    n_f = firm_obs.shape[0]
                    firm_actions_heuristic = torch.full(
                        (n_f,), 22, dtype=torch.long, device=self.device
                    )
                    firm_log_probs_heuristic = torch.zeros(n_f, device=self.device)
                    gs_expanded = global_state.unsqueeze(0).expand(n_f, -1)
                    firm_value = self.firm_ac.critic(gs_expanded, firm_obs).squeeze(-1)
                    firm_out = {
                        "action": firm_actions_heuristic,
                        "log_prob": firm_log_probs_heuristic,
                        "value": firm_value,
                    }
                else:
                    firm_out = self.firm_ac.get_action_and_value(
                        firm_obs, global_state.unsqueeze(0).expand(firm_obs.shape[0], -1)
                    )
            try:
                with torch.no_grad():
                    _st = self.env.fast_obs.state_table
                    _n_w = self.env.n_workers
                    _status = _st.w_status[:_n_w].to(self.device)
                    _subtype = _st.w_nilf_subtype[:_n_w].to(self.device)
                    _months = _st.w_nilf_subtype_months[:_n_w].to(self.device)
                    _skills = _st.w_skill[:_n_w].to(self.device).long().clamp(0, _N_SKILL_BINS - 1)
                    if _subtype_enabled:
                        _elig = (
                            (_subtype == _SUBTYPE_NONE)
                            | (_subtype == _SUBTYPE_DISCOURAGED)
                            | (_subtype == _SUBTYPE_SCHOOL) & (_months >= _school_w)
                            | (_subtype == _SUBTYPE_FAMILY) & (_months >= _family_w)
                            | (_subtype == _SUBTYPE_ILLNESS) & (_months >= _illness_w)
                        )
                    else:
                        _elig = torch.ones_like(_subtype, dtype=torch.bool)
                    _mask_unemp = _status == 1
                    _mask_emp = _status == 0
                    _mask_nilf = _status == 2
                    _cohort_masks = {
                        "Unemployed": _mask_unemp,
                        "Employed": _mask_emp,
                        "NILF_eligible": _mask_nilf & _elig,
                        "NILF_ineligible": _mask_nilf & ~_elig,
                    }
                    _w_act = worker_out["action"]
                    _search_actions = _w_act[..., 0] if _w_act.dim() > 1 else _w_act
                    _search_actions = _search_actions[:_n_w].to(self.device).long()
                    for _coh, _m in _cohort_masks.items():
                        if not bool(_m.any()):
                            continue
                        _coh_actions = _search_actions[_m]
                        cohort_action_counts[_coh] += torch.bincount(
                            _coh_actions.clamp(0, n_worker_actions - 1), minlength=n_worker_actions
                        ).to(cohort_action_counts[_coh].dtype)
                        _coh_skills = _skills[_m]
                        for _s in range(_N_SKILL_BINS):
                            _sk_mask = _coh_skills == _s
                            if not bool(_sk_mask.any()):
                                continue
                            cohort_per_skill_action_counts[_coh][_s] += torch.bincount(
                                _coh_actions[_sk_mask].clamp(0, n_worker_actions - 1),
                                minlength=n_worker_actions,
                            ).to(cohort_per_skill_action_counts[_coh][_s].dtype)
            except Exception as _cohort_e:
                if step == 0:
                    logger.warning(f"Could not count actions by labour-force status: {_cohort_e}")
            action_td = obs_td.clone()
            action_td["worker_action"] = worker_out["action"].cpu()
            if "intent_action" in worker_out:
                action_td["worker_intent_action"] = worker_out["intent_action"].cpu()
            action_td["firm_action"] = firm_out["action"].cpu()
            if _retire_mask is not None:
                action_td["worker_retire_eligible_mask"] = _retire_mask.cpu()
            _joint_action_keys = worker_keys(self) if _joint_identity else None
            next_obs_td = self.env.step(action_td)
            next_data = next_obs_td.get("next", next_obs_td)
            if _joint_identity:
                if worker_keys(self) != _joint_action_keys:
                    raise ValueError(
                        "Population identity changed before action-time reward capture"
                    )
                _joint_worker_reward = next_data["worker_reward"].detach().clone()
                _joint_firm_reward = next_data["firm_reward"].detach().clone()
                _controller = getattr(self.env, "_net_population", None)
                if _controller is not None:
                    next_data = _controller.finalize(self.env, next_data)
                _joint_next_keys = worker_keys(self)
                self.buffer.set_transition(
                    _joint_action_keys, _joint_next_keys, int(self.episode_count)
                )
            terminated_tensor = (
                next_data.get("terminated", torch.zeros_like(next_data["done"]))
                .float()
                .to(self.device)
            )
            ctx = worker_context if _joint_identity else obs_td.get("worker_context")
            if ctx is None:
                ctx = torch.zeros(
                    worker_obs.shape[0],
                    self.config.environment.n_visible_vacancies,
                    self.config.environment.vacancy_feature_dim,
                    device=self.device,
                )
            else:
                ctx = ctx.to(self.device)
            hist = worker_history if _joint_identity else obs_td.get("worker_history")
            if hist is None:
                hist = torch.zeros(
                    worker_obs.shape[0],
                    self.config.environment.history_length,
                    self.config.environment.history_dim,
                    device=self.device,
                )
            else:
                hist = hist.to(self.device)
            self.buffer.add(
                worker_obs=worker_obs,
                worker_action=worker_out["action"],
                worker_log_prob=worker_out["log_prob"],
                worker_reward=(
                    _joint_worker_reward if _joint_identity else next_data["worker_reward"]
                ).to(self.device),
                worker_value=self._worker_value_denorm(worker_out["value"]),
                worker_done=terminated_tensor.expand(worker_obs.shape[0]),
                worker_context=ctx,
                worker_history=hist,
                firm_obs=firm_obs,
                firm_action=firm_out["action"],
                firm_log_prob=firm_out["log_prob"],
                firm_reward=(
                    _joint_firm_reward if _joint_identity else next_data["firm_reward"]
                ).to(self.device),
                firm_value=firm_out["value"],
                firm_done=terminated_tensor.expand(self.env.n_firms),
                global_state=global_state,
                worker_retire_eligible_mask=_retire_mask,
            )
            worker_rewards_step = (
                _joint_worker_reward if _joint_identity else next_data["worker_reward"]
            )
            firm_rewards_step = _joint_firm_reward if _joint_identity else next_data["firm_reward"]
            episode_rewards_workers.append(worker_rewards_step.mean().item())
            episode_rewards_firms.append(firm_rewards_step.mean().item())
            _worker_search_actions = (
                worker_out["action"][..., 0]
                if worker_out["action"].dim() > 1
                else worker_out["action"]
            )
            for a in _worker_search_actions:
                if 0 <= a < len(worker_action_counts):
                    worker_action_counts[a] += 1
            if worker_out["action"].dim() > 1:
                _worker_res_wage_actions = worker_out["action"][..., 1]
                for a in _worker_res_wage_actions:
                    if 0 <= a < len(worker_res_wage_counts):
                        worker_res_wage_counts[a] += 1
            try:
                _ms = getattr(self.env, "last_matching_stats", None) or {}
                matches_per_step.append(int(_ms.get("matches", 0)))
            except Exception:
                matches_per_step.append(0)
            for a in firm_out["action"]:
                if 0 <= a < len(firm_action_counts):
                    firm_action_counts[a] += 1
            if not hasattr(self, "_reward_stats"):
                self._reward_stats = {
                    "worker_min": float("inf"),
                    "worker_max": float("-inf"),
                    "firm_min": float("inf"),
                    "firm_max": float("-inf"),
                }
            self._reward_stats["worker_min"] = min(
                self._reward_stats["worker_min"], worker_rewards_step.min().item()
            )
            self._reward_stats["worker_max"] = max(
                self._reward_stats["worker_max"], worker_rewards_step.max().item()
            )
            self._reward_stats["firm_min"] = min(
                self._reward_stats["firm_min"], firm_rewards_step.min().item()
            )
            self._reward_stats["firm_max"] = max(
                self._reward_stats["firm_max"], firm_rewards_step.max().item()
            )
            if progress_callback is not None:
                progress_callback(step + 1, self.config.training.rollout_steps)
            if next_data["done"].item():
                with torch.no_grad():
                    _b_global = next_data["global_state"].to(self.device)
                    _b_worker_obs = next_data["worker_obs"].to(self.device)
                    _b_firm_obs = next_data["firm_obs"].to(self.device)
                    _b_ctx = next_data.get("worker_context")
                    if _b_ctx is not None:
                        _b_ctx = _b_ctx.to(self.device)
                    _b_hist = next_data.get("worker_history")
                    if _b_hist is not None:
                        _b_hist = _b_hist.to(self.device)
                    _b_worker_value = self.worker_ac.critic(
                        _b_global.unsqueeze(0).expand(self.env.n_workers, -1),
                        _b_worker_obs,
                        context=_b_ctx,
                        history=_b_hist,
                    ).squeeze(-1)
                    _b_worker_value = self._worker_value_denorm(_b_worker_value)
                    _b_firm_value = self.firm_ac.critic(
                        _b_global.unsqueeze(0).expand(self.env.n_firms, -1), _b_firm_obs
                    ).squeeze(-1)
                self.buffer.mark_truncation(self.buffer.ptr - 1, _b_worker_value, _b_firm_value)
                self.episode_count += 1
                obs_td = self.env.reset()
            else:
                obs_td = next_data
            self.total_steps += 1
        self._last_obs = obs_td
        with torch.no_grad():
            last_global = obs_td["global_state"].to(self.device)
            last_worker_obs = obs_td["worker_obs"].to(self.device)
            last_firm_obs = obs_td["firm_obs"].to(self.device)
            last_ctx = obs_td.get("worker_context")
            if last_ctx is not None:
                last_ctx = last_ctx.to(self.device)
            last_hist = obs_td.get("worker_history")
            if last_hist is not None:
                last_hist = last_hist.to(self.device)
            last_worker_value = self.worker_ac.critic(
                last_global.unsqueeze(0).expand(self.env.n_workers, -1),
                last_worker_obs,
                context=last_ctx,
                history=last_hist,
            ).squeeze(-1)
            last_worker_value = self._worker_value_denorm(last_worker_value)
            last_firm_value = self.firm_ac.critic(
                last_global.unsqueeze(0).expand(self.env.n_firms, -1), last_firm_obs
            ).squeeze(-1)
        if _joint_identity:
            self.buffer.set_last_keys(worker_keys(self))
        self.buffer.compute_returns_and_advantages(
            last_worker_value,
            last_firm_value,
            gamma=self.config.training.gamma,
            gae_lambda=self.config.training.gae_lambda,
        )
        reward_stats = getattr(self, "_reward_stats", {})
        self._reward_stats = {
            "worker_min": float("inf"),
            "worker_max": float("-inf"),
            "firm_min": float("inf"),
            "firm_max": float("-inf"),
        }
        total_worker_actions = worker_action_counts.sum().item()
        total_firm_actions = firm_action_counts.sum().item()
        worker_action_freq = (
            (worker_action_counts / max(total_worker_actions, 1)).cpu().numpy().tolist()
        )
        firm_action_freq = (firm_action_counts / max(total_firm_actions, 1)).cpu().numpy().tolist()
        total_res_wage = worker_res_wage_counts.sum().item()
        worker_res_wage_freq = (
            (worker_res_wage_counts / max(total_res_wage, 1)).cpu().numpy().tolist()
        )
        worker_snapshot: List[Dict[str, float]] = []
        try:
            for wid, ws in self.env.state_manager.workers.items():
                worker_snapshot.append(
                    {
                        "worker_id": str(wid),
                        "skill_level": int(getattr(ws.worker, "skill_level", 2)),
                        "education": int(getattr(ws.worker, "education", 3)),
                        "unemployment_months": int(getattr(ws.worker, "unemployment_months", 0)),
                        "employment_status": str(getattr(ws.worker, "employment_status", "NILF")),
                        "learned_v0": float(getattr(ws, "learned_v0", 6.5)),
                        "search_action": int(getattr(ws, "last_search_action", 0)),
                        "search_decision": int(getattr(ws, "last_search_decision", 0)),
                        "search_channel": int(getattr(ws, "last_search_channel", 0)),
                        "search_intensity": int(getattr(ws, "last_search_intensity", 0)),
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to build worker snapshot: {e}")
        wage_sample = []
        wages_by_industry = {}
        _wage_population: list = []
        try:
            for worker_id, ws in self.env.state_manager.workers.items():
                if ws.worker.employment_status == "Employed" and ws.worker.current_wage > 0:
                    wage = float(ws.worker.current_wage)
                    wage_sample.append(wage)
                    industry = getattr(ws.worker, "industry", None)
                    if industry is not None:
                        if industry not in wages_by_industry:
                            wages_by_industry[industry] = []
                        wages_by_industry[industry].append(wage)
            _wage_population = list(wage_sample)
            if len(wage_sample) > 500:
                wage_sample = list(np.random.choice(wage_sample, 500, replace=False).astype(float))
        except Exception as e:
            logger.warning(f"Failed to collect wage samples: {e}")
            wage_sample = []
            wages_by_industry = {}
            _wage_population = []
        realized_wage_gini = float("nan")
        realized_wage_p90_p10_ratio = float("nan")
        _n_employed_before_cap = sum(
            (
                1
                for _ws in self.env.state_manager.workers.values()
                if _ws.worker.employment_status == "Employed" and _ws.worker.current_wage > 0
            )
        )
        _dispersion_source = _wage_population
        realized_wage_sample_is_population = bool(len(_dispersion_source) == _n_employed_before_cap)
        if not realized_wage_sample_is_population and _n_employed_before_cap > 500:
            logger.warning(
                f"The wage sample was cut to 500 workers ({_n_employed_before_cap} employed, {len(wage_sample)} sampled), so the wage dispersion measures of this update carry sampling noise."
            )
        try:
            if len(_dispersion_source) >= 10:
                _wa = np.sort(np.asarray(_dispersion_source, dtype=np.float64))
                _n = len(_wa)
                _cum_weighted = float(np.sum(np.arange(1, _n + 1) * _wa))
                _sum = float(np.sum(_wa))
                if _sum > 0:
                    realized_wage_gini = 2.0 * _cum_weighted / (_n * _sum) - (_n + 1) / _n
                _p10 = float(np.percentile(_wa, 10))
                _p90 = float(np.percentile(_wa, 90))
                if _p10 > 0:
                    realized_wage_p90_p10_ratio = _p90 / _p10
        except Exception as _rw_e:
            logger.warning(f"Could not compute the wage dispersion measures: {_rw_e}")
        mean_wages_by_industry = {}
        for ind, wages in wages_by_industry.items():
            if len(wages) > 0:
                mean_wages_by_industry[int(ind)] = {
                    "mean": float(np.mean(wages)),
                    "median": float(np.median(wages)),
                    "std": float(np.std(wages)) if len(wages) > 1 else 0.0,
                    "n": len(wages),
                }
        tenure_sample = []
        try:
            for worker_id, ws in self.env.state_manager.workers.items():
                if ws.worker.employment_status == "Employed" and ws.months_employed > 0:
                    tenure_sample.append(int(ws.months_employed))
            if len(tenure_sample) > 500:
                tenure_sample = list(
                    np.random.choice(tenure_sample, 500, replace=False).astype(int)
                )
        except Exception as e:
            logger.warning(f"Failed to collect tenure samples: {e}")
            tenure_sample = []
        nash_bargaining_stats = {}
        try:
            if hasattr(self.env, "last_matching_stats") and self.env.last_matching_stats:
                nash_data = self.env.last_matching_stats.get("nash_bargaining", {})
                if nash_data:
                    nash_bargaining_stats = {
                        "n_negotiations": nash_data.get("n_negotiations", 0),
                        "avg_worker_power": nash_data.get("avg_worker_power", 0.0),
                        "min_worker_power": nash_data.get("min_worker_power", 0.0),
                        "max_worker_power": nash_data.get("max_worker_power", 0.0),
                        "std_worker_power": nash_data.get("std_worker_power", 0.0),
                        "avg_offered_wage": nash_data.get("avg_offered_wage", 0.0),
                        "avg_negotiated_wage": nash_data.get("avg_negotiated_wage", 0.0),
                        "avg_wage_premium": nash_data.get("avg_wage_premium", 0.0),
                        "avg_wage_premium_pct": nash_data.get("avg_wage_premium_pct", 0.0),
                        "worker_power_sample": nash_data.get("worker_power_sample", []),
                        "wage_premium_sample": nash_data.get("wage_premium_sample", []),
                    }
        except Exception as e:
            logger.warning(f"Failed to collect Nash bargaining stats: {e}")
        commute_stats = {}
        try:
            if hasattr(self.env, "last_matching_stats") and self.env.last_matching_stats:
                commute_values = self.env.last_matching_stats.get("commute_values", [])
                commute_unit = self.env.last_matching_stats.get("commute_unit", "km")
                use_travel_time = self.env.last_matching_stats.get("use_travel_time", False)
                if commute_values:
                    commute_stats = {
                        "values": commute_values[:100],
                        "unit": commute_unit,
                        "use_travel_time": use_travel_time,
                        "mean": float(np.mean(commute_values)),
                        "median": float(np.median(commute_values)),
                    }
        except Exception as e:
            logger.warning(f"Failed to collect commute stats: {e}")
        cohort_action_counts_serial: Dict[str, list] = {}
        cohort_per_skill_action_counts_serial: Dict[str, Dict[int, list]] = {}
        try:
            for _coh, _t in cohort_action_counts.items():
                cohort_action_counts_serial[_coh] = _t.detach().cpu().numpy().tolist()
            for _coh, _t2 in cohort_per_skill_action_counts.items():
                cohort_per_skill_action_counts_serial[_coh] = {
                    int(_s): _t2[_s].detach().cpu().numpy().tolist() for _s in range(_N_SKILL_BINS)
                }
        except Exception as _ser_e:
            logger.warning(f"Could not store the action counts by labour-force status: {_ser_e}")
        gc.collect()
        firm_wage_strategy_argmax_by_industry = {}
        firm_wage_strategy_entropy_by_industry = {}
        firm_vacancy_management_argmax_by_industry = {}
        firm_posted_wage_mean_by_industry = {}
        firm_posted_wage_std_by_industry = {}
        try:
            _n_wage_opts = int(self.config.action.firm_wage_strategy_options)
            _n_vac_opts = int(self.config.action.firm_vacancy_management_options)
            _by_ind_wage_strat = {}
            _by_ind_vac_mgmt = {}
            _by_ind_posted = {}
            _all_posted = []
            for _fs in self.env.state_manager.firms.values():
                _ind = int(getattr(_fs.firm, "industry", -1))
                _by_ind_wage_strat.setdefault(_ind, []).append(
                    int(getattr(_fs, "last_wage_strategy", 2))
                )
                _by_ind_vac_mgmt.setdefault(_ind, []).append(
                    int(getattr(_fs, "last_vacancy_action", 2))
                )
                _w = float(getattr(_fs, "current_wage_offer", 0.0))
                if _w > 0:
                    _by_ind_posted.setdefault(_ind, []).append(_w)
                    _all_posted.append(_w)
            for _ind, _strats in _by_ind_wage_strat.items():
                _counts = np.bincount(np.array(_strats, dtype=int), minlength=_n_wage_opts).astype(
                    float
                )
                _p = _counts / max(_counts.sum(), 1.0)
                _ent = float(-np.sum(_p[_p > 0] * np.log(_p[_p > 0]))) if _p.sum() > 0 else 0.0
                firm_wage_strategy_argmax_by_industry[_ind] = int(np.argmax(_counts))
                firm_wage_strategy_entropy_by_industry[_ind] = _ent
            for _ind, _vacs in _by_ind_vac_mgmt.items():
                _counts = np.bincount(np.array(_vacs, dtype=int), minlength=_n_vac_opts).astype(
                    float
                )
                firm_vacancy_management_argmax_by_industry[_ind] = int(np.argmax(_counts))
            for _ind, _wages in _by_ind_posted.items():
                firm_posted_wage_mean_by_industry[_ind] = float(np.mean(_wages))
                firm_posted_wage_std_by_industry[_ind] = (
                    float(np.std(_wages)) if len(_wages) > 1 else 0.0
                )
            if len(_all_posted) >= 10:
                _p10 = float(np.percentile(_all_posted, 10))
                _p90 = float(np.percentile(_all_posted, 90))
                firm_posted_wage_p90_p10_ratio = _p90 / _p10 if _p10 > 0 else float("nan")
            else:
                firm_posted_wage_p90_p10_ratio = float("nan")
            if len(_all_posted) >= 10:
                _dec = np.percentile(_all_posted, [10, 20, 30, 40, 50, 60, 70, 80, 90])
                firm_posted_wage_deciles = [float(x) for x in _dec]
            else:
                firm_posted_wage_deciles = [float("nan")] * 9
        except Exception as _fd_e:
            logger.warning(f"Could not aggregate the firm action measures: {_fd_e}")
            firm_posted_wage_p90_p10_ratio = float("nan")
            firm_posted_wage_deciles = [float("nan")] * 9
        return {
            "mean_worker_reward": np.mean(episode_rewards_workers),
            "mean_firm_reward": np.mean(episode_rewards_firms),
            "worker_reward_min": reward_stats.get("worker_min", 0),
            "worker_reward_max": reward_stats.get("worker_max", 0),
            "firm_reward_min": reward_stats.get("firm_min", 0),
            "firm_reward_max": reward_stats.get("firm_max", 0),
            "total_steps": self.total_steps,
            "episodes": self.episode_count,
            "worker_action_freq": worker_action_freq,
            "firm_action_freq": firm_action_freq,
            "worker_res_wage_freq": worker_res_wage_freq,
            "worker_snapshot": worker_snapshot,
            "matches_per_step_mean": float(np.mean(matches_per_step)) if matches_per_step else 0.0,
            "matches_per_step_total": int(sum(matches_per_step)),
            "wage_sample": wage_sample,
            "tenure_sample": tenure_sample,
            "nash_bargaining": nash_bargaining_stats,
            "wages_by_industry": mean_wages_by_industry,
            "commute_stats": commute_stats,
            "cohort_action_counts": cohort_action_counts_serial,
            "cohort_per_skill_action_counts": cohort_per_skill_action_counts_serial,
            "firm_wage_strategy_argmax_by_industry": firm_wage_strategy_argmax_by_industry,
            "firm_wage_strategy_entropy_by_industry": firm_wage_strategy_entropy_by_industry,
            "firm_vacancy_management_argmax_by_industry": firm_vacancy_management_argmax_by_industry,
            "firm_posted_wage_mean_by_industry": firm_posted_wage_mean_by_industry,
            "firm_posted_wage_std_by_industry": firm_posted_wage_std_by_industry,
            "firm_posted_wage_p90_p10_ratio": firm_posted_wage_p90_p10_ratio,
            "firm_posted_wage_deciles": firm_posted_wage_deciles,
            "realized_wage_gini": realized_wage_gini,
            "realized_wage_p90_p10_ratio": realized_wage_p90_p10_ratio,
            "realized_wage_sample_is_population": realized_wage_sample_is_population,
            "worker_decision_p_search_mean": (
                float(np.mean(p_search_steps)) if p_search_steps else float("nan")
            ),
            "worker_decision_p_search_std_within_batch": (
                float(np.mean(p_search_std_steps)) if p_search_std_steps else float("nan")
            ),
            "worker_decision_head_entropy": (
                float(np.mean(decision_entropy_steps)) if decision_entropy_steps else float("nan")
            ),
            "worker_decision_logit_spread": (
                float(np.mean(decision_logit_spread_steps))
                if decision_logit_spread_steps
                else float("nan")
            ),
            "worker_decision_p_search_mean_stuck_cohort": (
                float(np.mean(p_search_steps_cohort)) if p_search_steps_cohort else float("nan")
            ),
            "worker_decision_p_search_std_within_batch_stuck_cohort": (
                float(np.mean(p_search_std_steps_cohort))
                if p_search_std_steps_cohort
                else float("nan")
            ),
            "worker_decision_head_entropy_stuck_cohort": (
                float(np.mean(decision_entropy_steps_cohort))
                if decision_entropy_steps_cohort
                else float("nan")
            ),
            "worker_decision_logit_spread_stuck_cohort": (
                float(np.mean(decision_logit_spread_steps_cohort))
                if decision_logit_spread_steps_cohort
                else float("nan")
            ),
            "worker_decision_stuck_cohort_size_mean": (
                float(np.mean(cohort_size_steps)) if cohort_size_steps else float("nan")
            ),
            **_retirement_audit_rollout_dict(self.env),
        }

    def _apply_warmup_lr(self, absolute_update_count: int) -> None:
        if not getattr(self, "_warmup_active", False):
            return
        warmup_n = int(getattr(self, "_warmup_updates", 0))
        start_frac = float(getattr(self, "_warmup_start_frac", 1.0))
        if warmup_n <= 0:
            return
        if getattr(self, "_warmup_local_offset", None) is None:
            self._warmup_local_offset = int(absolute_update_count)
        local_idx = int(absolute_update_count) - int(self._warmup_local_offset)
        if local_idx < 0:
            local_idx = 0
        if local_idx >= warmup_n:
            rw_factor = 1.0
        else:
            denom = max(1, warmup_n - 1)
            rw_factor = start_frac + (1.0 - start_frac) * (local_idx / denom)
        for pg in self.worker_actor_optimizer.param_groups:
            name = pg.get("name", "")
            target = pg.get("target_lr", pg.get("lr", None))
            if target is None:
                continue
            if name == "worker_actor_res_wage_head":
                pg["lr"] = rw_factor * target
            else:
                pg["lr"] = target

    def update(self) -> Dict[str, float]:
        n_epochs = self.config.training.n_epochs
        minibatch_size = self.config.training.minibatch_size
        self._apply_warmup_lr(self.update_count)
        self._encoder_grad_norms_this_update = []
        worker_adv = self.buffer.worker_advantages[: self.buffer.ptr]
        if worker_adv.numel() > 1:
            w_std = torch.clamp(worker_adv.std(), min=0.1)
            worker_adv = (worker_adv - worker_adv.mean()) / (w_std + 1e-08)
            self.buffer.worker_advantages[: self.buffer.ptr] = worker_adv
        firm_adv = self.buffer.firm_advantages[: self.buffer.ptr]
        if firm_adv.numel() > 1:
            f_std = torch.clamp(firm_adv.std(), min=0.1)
            firm_adv = (firm_adv - firm_adv.mean()) / (f_std + 1e-08)
            self.buffer.firm_advantages[: self.buffer.ptr] = firm_adv
        worker_ret = self.buffer.worker_returns[: self.buffer.ptr]
        if self._worker_popart_enabled():
            if worker_ret.numel() > 1:
                self._update_worker_popart_stats(worker_ret)
                self._worker_ret_mean = self._worker_ret_mean_ema
                self._worker_ret_std = self._worker_ret_std_ema
        elif worker_ret.numel() > 1:
            self._worker_ret_mean = worker_ret.mean()
            self._worker_ret_std = torch.clamp(worker_ret.std(), min=1.0)
            self.buffer.worker_returns[: self.buffer.ptr] = (
                worker_ret - self._worker_ret_mean
            ) / self._worker_ret_std
            self.buffer.worker_values[: self.buffer.ptr] = (
                self.buffer.worker_values[: self.buffer.ptr] - self._worker_ret_mean
            ) / self._worker_ret_std
        elif worker_ret.numel() > 1:
            self._worker_ret_mean = torch.zeros((), device=worker_ret.device)
            self._worker_ret_std = torch.ones((), device=worker_ret.device)
        firm_ret = self.buffer.firm_returns[: self.buffer.ptr]
        if firm_ret.numel() > 1:
            self._firm_ret_mean = firm_ret.mean()
            self._firm_ret_std = torch.ones(1, device=firm_ret.device)
        worker_policy_losses = []
        worker_value_losses = []
        worker_entropies = []
        worker_search_entropies = []
        worker_res_wage_entropies = []
        worker_kls = []
        worker_kls_to_prior = []
        worker_res_wage_kls_uniform = []
        worker_kl_prior_by_skill_sum = torch.zeros(5, device=self.device)
        worker_kl_prior_by_skill_count = torch.zeros(5, device=self.device)
        worker_kl_prior_by_duration_sum = torch.zeros(12, device=self.device)
        worker_kl_prior_by_duration_count = torch.zeros(12, device=self.device)
        worker_explained_vars = []
        worker_value_pred_errors = []
        worker_raw_critic_value_std = float("nan")
        worker_denorm_value_std = float("nan")
        firm_policy_losses = []
        firm_value_losses = []
        firm_entropies = []
        firm_kls = []
        firm_explained_vars = []
        firm_value_pred_errors = []
        clip_epsilon = self.config.training.clip_epsilon
        tc = self.config.training
        worker_entropy_coef = getattr(tc, "worker_entropy_coef", tc.entropy_coef)
        worker_res_wage_entropy_coef = getattr(
            tc, "worker_res_wage_entropy_coef", worker_entropy_coef
        )
        firm_ent_start = getattr(tc, "firm_entropy_coef", tc.entropy_coef)
        firm_ent_min = getattr(tc, "firm_entropy_coef_min", 0.001)
        firm_ent_decay = getattr(tc, "firm_entropy_decay", 0.995)
        anneal_reset = getattr(self, "firm_entropy_anneal_reset_update", 0)
        warmup = getattr(tc, "firm_entropy_warmup_updates", 0)
        effective_counter = max(0, self.update_count - anneal_reset - warmup)
        firm_entropy_coef = max(firm_ent_min, firm_ent_start * firm_ent_decay**effective_counter)
        worker_kl_threshold = getattr(tc, "worker_kl_threshold", 0.015)
        worker_kl_rollback = getattr(tc, "worker_kl_rollback_factor", 0.5)
        worker_lr_reduced = False
        firm_kl_threshold = getattr(tc, "firm_kl_threshold", 0.02)
        firm_kl_rollback = getattr(tc, "firm_kl_rollback_factor", 0.5)
        firm_lr_reduced = False
        worker_target_kl = getattr(tc, "worker_target_kl", None)
        worker_target_kl_exceeded = False
        worker_target_kl_fired_at = None
        for epoch in range(n_epochs):
            batches = self.buffer.get_batches(minibatch_size)
            for batch_idx, batch in enumerate(batches):
                batch_len = batch["worker_obs"].shape[0]
                n_workers = (
                    batch["worker_obs"].shape[1]
                    if getattr(self.buffer, "ragged_workers", False)
                    else self.env.n_workers
                )
                b_worker_obs = batch["worker_obs"].view(-1, batch["worker_obs"].shape[-1])
                b_worker_actions = batch["worker_actions"].view(
                    -1, batch["worker_actions"].shape[-1]
                )
                b_worker_old_log_probs = batch["worker_log_probs"].view(-1)
                b_worker_old_values = batch["worker_values"].view(-1)
                b_worker_advantages = batch["worker_advantages"].view(-1)
                b_worker_returns = batch["worker_returns"].view(-1)
                b_worker_retire_mask = None
                if "worker_retire_eligible_mask" in batch:
                    b_worker_retire_mask = batch["worker_retire_eligible_mask"].reshape(-1)
                if "worker_global_states" in batch:
                    b_global_states = batch["worker_global_states"]
                else:
                    b_global_states = (
                        batch["global_states"]
                        .unsqueeze(1)
                        .expand(-1, n_workers, -1)
                        .reshape(-1, batch["global_states"].shape[-1])
                    )
                b_worker_context = batch["worker_context"].view(
                    batch_len * n_workers,
                    batch["worker_context"].shape[2],
                    batch["worker_context"].shape[3],
                )
                b_worker_history = batch["worker_history"].view(
                    batch_len * n_workers,
                    batch["worker_history"].shape[2],
                    batch["worker_history"].shape[3],
                )
                CHUNK_SIZE = 512
                n_total = b_worker_obs.shape[0]
                self.worker_encoder_optimizer.zero_grad()
                self.worker_actor_optimizer.zero_grad()
                self.worker_critic_optimizer.zero_grad()
                total_policy_loss = 0.0
                total_value_loss = 0.0
                total_entropy = 0.0
                total_search_entropy = 0.0
                total_res_wage_entropy = 0.0
                total_kl = 0.0
                total_kl_to_prior = 0.0
                total_res_wage_kl_uniform = 0.0
                all_values = []
                all_returns = []
                for start in range(0, n_total, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_total)
                    chunk_size = end - start
                    chunk_skill = b_worker_obs[start:end, BR_SKILL_COL]
                    chunk_retire_mask = (
                        b_worker_retire_mask[start:end]
                        if b_worker_retire_mask is not None
                        else None
                    )
                    _eval_kwargs = {
                        "context": b_worker_context[start:end],
                        "history": b_worker_history[start:end],
                        "skill_param": chunk_skill,
                    }
                    if chunk_retire_mask is not None:
                        _eval_kwargs["retire_eligible_mask"] = chunk_retire_mask
                    chunk_eval = self.worker_ac.evaluate(
                        b_worker_obs[start:end],
                        b_worker_actions[start:end],
                        b_global_states[start:end],
                        **_eval_kwargs,
                    )
                    chunk_old_log_probs = b_worker_old_log_probs[start:end]
                    chunk_old_values = b_worker_old_values[start:end]
                    chunk_advantages = b_worker_advantages[start:end]
                    chunk_returns = b_worker_returns[start:end]
                    log_ratio_raw = chunk_eval["log_prob"] - chunk_old_log_probs
                    finite_mask = torch.isfinite(log_ratio_raw)
                    n_nonfinite = int((~finite_mask).sum().item())
                    if n_nonfinite > 0:
                        if not hasattr(self, "_n_nonfinite_logratio_total"):
                            self._n_nonfinite_logratio_total = 0
                            self._n_nonfinite_logratio_chunks = 0
                        self._n_nonfinite_logratio_total += n_nonfinite
                        self._n_nonfinite_logratio_chunks += 1
                        if (
                            self._n_nonfinite_logratio_chunks <= 5
                            or self._n_nonfinite_logratio_chunks % 50 == 0
                        ):
                            logger.warning(
                                f"{n_nonfinite} of {chunk_size} samples in this batch have a non-finite log ratio ({self._n_nonfinite_logratio_total} so far in {self._n_nonfinite_logratio_chunks} batches). They are left out of the loss and the divergence."
                            )
                    log_ratio = torch.where(
                        finite_mask, log_ratio_raw, torch.zeros_like(log_ratio_raw)
                    )
                    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    chunk_kl = (ratio - 1 - log_ratio).mean()
                    surr1 = ratio * chunk_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * chunk_advantages
                    )
                    chunk_obs_for_weights = b_worker_obs[start:end]
                    _is_unemp = chunk_obs_for_weights[:, 5] > 0.5
                    _is_nilf = chunk_obs_for_weights[:, 6] > 0.5
                    _w_e = 1.0
                    _w_u = float(
                        getattr(self.config.training, "worker_unemployed_policy_weight", 1.0)
                    )
                    _w_n = float(getattr(self.config.training, "worker_nilf_policy_weight", 1.0))
                    sample_weights = torch.full_like(chunk_advantages, _w_e)
                    sample_weights = torch.where(
                        _is_unemp, torch.full_like(sample_weights, _w_u), sample_weights
                    )
                    sample_weights = torch.where(
                        _is_nilf, torch.full_like(sample_weights, _w_n), sample_weights
                    )
                    sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-08)
                    per_sample_policy_loss = -torch.min(surr1, surr2)
                    chunk_policy_loss = (sample_weights * per_sample_policy_loss).mean()
                    new_values = chunk_eval["value"]
                    _vc_override = getattr(self.config.training, "worker_value_clip_epsilon", None)
                    _vc_eps = float(_vc_override) if _vc_override is not None else clip_epsilon
                    if self._worker_popart_enabled():
                        chunk_returns_used = self._worker_value_norm(chunk_returns)
                        chunk_old_values_used = self._worker_value_norm(chunk_old_values)
                    else:
                        chunk_returns_used = chunk_returns
                        chunk_old_values_used = chunk_old_values
                    if self._worker_popart_just_initialized:
                        chunk_old_values_used = new_values.detach()
                    v_clipped = chunk_old_values_used + torch.clamp(
                        new_values - chunk_old_values_used, -_vc_eps, _vc_eps
                    )
                    vl_unclipped = (new_values - chunk_returns_used) ** 2
                    vl_clipped = (v_clipped - chunk_returns_used) ** 2
                    chunk_value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
                    chunk_entropy = chunk_eval["entropy"].mean()
                    chunk_search_entropy = chunk_eval.get("search_entropy")
                    chunk_res_wage_entropy = chunk_eval.get("res_wage_entropy")
                    if chunk_search_entropy is not None and chunk_res_wage_entropy is not None:
                        entropy_bonus = (
                            worker_entropy_coef * chunk_search_entropy.mean()
                            + worker_res_wage_entropy_coef * chunk_res_wage_entropy.mean()
                        )
                    else:
                        entropy_bonus = worker_entropy_coef * chunk_entropy
                    actor_loss = chunk_policy_loss - entropy_bonus
                    critic_loss = self.config.training.value_loss_coef * chunk_value_loss
                    scale = chunk_size / n_total
                    (actor_loss * scale).backward(retain_graph=True)
                    (critic_loss * scale).backward()
                    total_policy_loss += chunk_policy_loss.item() * scale
                    total_value_loss += chunk_value_loss.item() * scale
                    total_entropy += chunk_entropy.item() * scale
                    if chunk_search_entropy is not None:
                        total_search_entropy += chunk_search_entropy.mean().item() * scale
                    if chunk_res_wage_entropy is not None:
                        total_res_wage_entropy += chunk_res_wage_entropy.mean().item() * scale
                    total_kl += chunk_kl.item() * scale
                    all_values.append(chunk_eval["value"].detach())
                    all_returns.append(chunk_returns.detach())
                    del chunk_eval
                all_worker_params = (
                    list(self.worker_ac.encoder.parameters())
                    + _non_encoder_params(self.worker_ac.actor)
                    + _non_encoder_params(self.worker_ac.critic)
                )
                nn.utils.clip_grad_norm_(all_worker_params, self.config.training.max_grad_norm)
                if self.update_count == 0 and epoch == 0:
                    nn.utils.clip_grad_norm_(self.worker_ac.encoder.parameters(), float("inf"))
                self.worker_encoder_optimizer.step()
                self.worker_actor_optimizer.step()
                self.worker_critic_optimizer.step()
                worker_policy_losses.append(total_policy_loss)
                worker_value_losses.append(total_value_loss)
                worker_entropies.append(total_entropy)
                worker_search_entropies.append(total_search_entropy)
                worker_res_wage_entropies.append(total_res_wage_entropy)
                worker_kls.append(total_kl)
                worker_kls_to_prior.append(total_kl_to_prior)
                worker_res_wage_kls_uniform.append(total_res_wage_kl_uniform)
                if total_kl > worker_kl_threshold and (not worker_lr_reduced):
                    for pg in self.worker_actor_optimizer.param_groups:
                        pg["lr"] *= worker_kl_rollback
                    worker_lr_reduced = True
                if worker_target_kl is not None:
                    if not worker_target_kl_exceeded and total_kl > 1.5 * worker_target_kl:
                        worker_target_kl_exceeded = True
                        worker_target_kl_fired_at = (epoch, batch_idx, float(total_kl))
                        break
                all_values_cat = torch.cat(all_values)
                all_returns_cat = torch.cat(all_returns)
                worker_raw_critic_value_std = float(all_values_cat.std().item())
                if self._worker_popart_enabled():
                    all_values_cat = self._worker_value_denorm(all_values_cat)
                worker_denorm_value_std = float(all_values_cat.std().item())
                var_returns = all_returns_cat.var()
                if var_returns > 1e-08:
                    explained_var = 1 - (all_returns_cat - all_values_cat).var() / var_returns
                    worker_explained_vars.append(explained_var.item())
                else:
                    worker_explained_vars.append(0.0)
                pred_errors = (all_values_cat - all_returns_cat).cpu().numpy()
                if len(pred_errors) > 500:
                    sample_idx = np.random.choice(len(pred_errors), 500, replace=False)
                    pred_errors = pred_errors[sample_idx]
                worker_value_pred_errors.extend(pred_errors.tolist())
                del all_values, all_returns, all_values_cat, all_returns_cat
                n_firms = self.env.n_firms
                b_firm_obs = batch["firm_obs"].view(-1, batch["firm_obs"].shape[-1])
                b_firm_actions = batch["firm_actions"].view(-1)
                b_firm_old_log_probs = batch["firm_log_probs"].view(-1)
                b_firm_old_values = batch["firm_values"].view(-1)
                b_firm_advantages = batch["firm_advantages"].view(-1)
                b_firm_returns = batch["firm_returns"].view(-1)
                b_global_states_firm = (
                    batch["global_states"]
                    .unsqueeze(1)
                    .expand(-1, n_firms, -1)
                    .reshape(-1, batch["global_states"].shape[-1])
                )
                FIRM_CHUNK_SIZE = 512
                n_total_firms = b_firm_obs.shape[0]
                _freeze_firm = getattr(self.config.training, "freeze_firm_policy", False)
                self.firm_encoder_optimizer.zero_grad()
                if not _freeze_firm:
                    self.firm_actor_optimizer.zero_grad()
                self.firm_critic_optimizer.zero_grad()
                total_firm_policy_loss = 0.0
                total_firm_value_loss = 0.0
                total_firm_entropy = 0.0
                total_firm_kl = 0.0
                all_firm_values = []
                all_firm_returns = []
                for start in range(0, n_total_firms, FIRM_CHUNK_SIZE):
                    end = min(start + FIRM_CHUNK_SIZE, n_total_firms)
                    chunk_size = end - start
                    chunk_eval = self.firm_ac.evaluate(
                        b_firm_obs[start:end],
                        b_firm_actions[start:end],
                        b_global_states_firm[start:end],
                    )
                    chunk_old_log_probs = b_firm_old_log_probs[start:end]
                    chunk_old_values = b_firm_old_values[start:end]
                    chunk_advantages = b_firm_advantages[start:end]
                    chunk_returns = b_firm_returns[start:end]
                    log_ratio_raw = chunk_eval["log_prob"] - chunk_old_log_probs
                    finite_mask = torch.isfinite(log_ratio_raw)
                    n_nonfinite = int((~finite_mask).sum().item())
                    if n_nonfinite > 0:
                        if not hasattr(self, "_n_nonfinite_logratio_total"):
                            self._n_nonfinite_logratio_total = 0
                            self._n_nonfinite_logratio_chunks = 0
                        self._n_nonfinite_logratio_total += n_nonfinite
                        self._n_nonfinite_logratio_chunks += 1
                        if (
                            self._n_nonfinite_logratio_chunks <= 5
                            or self._n_nonfinite_logratio_chunks % 50 == 0
                        ):
                            logger.warning(
                                f"{n_nonfinite} of {chunk_size} samples in this batch have a non-finite log ratio ({self._n_nonfinite_logratio_total} so far in {self._n_nonfinite_logratio_chunks} batches). They are left out of the loss and the divergence."
                            )
                    log_ratio = torch.where(
                        finite_mask, log_ratio_raw, torch.zeros_like(log_ratio_raw)
                    )
                    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    chunk_kl = (ratio - 1 - log_ratio).mean()
                    surr1 = ratio * chunk_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * chunk_advantages
                    )
                    chunk_policy_loss = -torch.min(surr1, surr2).mean()
                    new_values = chunk_eval["value"]
                    v_clipped = chunk_old_values + torch.clamp(
                        new_values - chunk_old_values, -1.0, 1.0
                    )
                    vl_unclipped = (new_values - chunk_returns) ** 2
                    vl_clipped = (v_clipped - chunk_returns) ** 2
                    chunk_value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()
                    chunk_entropy = chunk_eval["entropy"].mean()
                    scale = chunk_size / n_total_firms
                    if not _freeze_firm:
                        actor_loss = chunk_policy_loss - firm_entropy_coef * chunk_entropy
                        (actor_loss * scale).backward(retain_graph=True)
                    critic_loss = self.config.training.value_loss_coef * chunk_value_loss
                    (critic_loss * scale).backward()
                    total_firm_policy_loss += chunk_policy_loss.item() * scale
                    total_firm_value_loss += chunk_value_loss.item() * scale
                    total_firm_entropy += chunk_entropy.item() * scale
                    total_firm_kl += chunk_kl.item() * scale
                    all_firm_values.append(chunk_eval["value"].detach())
                    all_firm_returns.append(chunk_returns.detach())
                    del chunk_eval
                all_firm_params = (
                    list(self.firm_ac.encoder.parameters())
                    + _non_encoder_params(self.firm_ac.actor)
                    + _non_encoder_params(self.firm_ac.critic)
                )
                nn.utils.clip_grad_norm_(all_firm_params, self.config.training.max_grad_norm)
                if not _freeze_firm:
                    self.firm_actor_optimizer.step()
                self.firm_encoder_optimizer.step()
                self.firm_critic_optimizer.step()
                firm_policy_losses.append(total_firm_policy_loss)
                firm_value_losses.append(total_firm_value_loss)
                firm_entropies.append(total_firm_entropy)
                firm_kls.append(total_firm_kl)
                if total_firm_kl > firm_kl_threshold and (not firm_lr_reduced):
                    for pg in self.firm_actor_optimizer.param_groups:
                        pg["lr"] *= firm_kl_rollback
                    firm_lr_reduced = True
                all_firm_values_cat = torch.cat(all_firm_values)
                all_firm_returns_cat = torch.cat(all_firm_returns)
                var_firm_returns = all_firm_returns_cat.var()
                var_residual = (all_firm_returns_cat - all_firm_values_cat).var()
                if var_firm_returns > 1e-08:
                    firm_explained_var = 1 - var_residual / var_firm_returns
                    firm_explained_vars.append(firm_explained_var.item())
                else:
                    firm_explained_vars.append(0.0)
                firm_pred_errors = (all_firm_values_cat - all_firm_returns_cat).cpu().numpy()
                if len(firm_pred_errors) > 500:
                    sample_idx = np.random.choice(len(firm_pred_errors), 500, replace=False)
                    firm_pred_errors = firm_pred_errors[sample_idx]
                firm_value_pred_errors.extend(firm_pred_errors.tolist())
                del all_firm_values, all_firm_returns, all_firm_values_cat, all_firm_returns_cat
            if worker_target_kl_exceeded:
                break
        if worker_target_kl is not None:
            if worker_target_kl_exceeded:
                _e, _b, _kl = worker_target_kl_fired_at
        if worker_lr_reduced:
            for pg in self.worker_actor_optimizer.param_groups:
                pg["lr"] /= worker_kl_rollback
        if firm_lr_reduced:
            for pg in self.firm_actor_optimizer.param_groups:
                pg["lr"] /= firm_kl_rollback
        if self.worker_actor_scheduler:
            self.worker_actor_scheduler.step()
        if self.worker_critic_scheduler:
            self.worker_critic_scheduler.step()
        if self.firm_actor_scheduler:
            self.firm_actor_scheduler.step()
        if self.firm_critic_scheduler:
            self.firm_critic_scheduler.step()
        self.update_count += 1
        gc.collect()
        self._maybe_emit_encoder_rep_audit()
        with torch.no_grad():
            _skill_mean = torch.where(
                worker_kl_prior_by_skill_count > 0,
                worker_kl_prior_by_skill_sum / worker_kl_prior_by_skill_count.clamp(min=1),
                torch.zeros_like(worker_kl_prior_by_skill_sum),
            )
            _duration_mean = torch.where(
                worker_kl_prior_by_duration_count > 0,
                worker_kl_prior_by_duration_sum / worker_kl_prior_by_duration_count.clamp(min=1),
                torch.zeros_like(worker_kl_prior_by_duration_sum),
            )
        worker_kl_to_prior_by_skill = {int(i): float(_skill_mean[i].item()) for i in range(5)}
        worker_kl_to_prior_by_duration = {
            int(i): float(_duration_mean[i].item()) for i in range(12)
        }
        mean_worker_reward = self.buffer.worker_rewards[: self.buffer.ptr].mean().item()
        mean_firm_reward = self.buffer.firm_rewards[: self.buffer.ptr].mean().item()
        if self._worker_popart_just_initialized:
            self._worker_popart_just_initialized = False
        return {
            "worker_forward_preflight": None,
            "worker_forward_post_update": None,
            "worker/mean_reward": mean_worker_reward,
            "firm/mean_reward": mean_firm_reward,
            "worker_policy_loss": np.mean(worker_policy_losses),
            "worker_value_loss": np.mean(worker_value_losses),
            "worker_entropy": np.mean(worker_entropies),
            "worker_search_entropy": (
                np.mean(worker_search_entropies) if worker_search_entropies else 0.0
            ),
            "worker_res_wage_entropy": (
                np.mean(worker_res_wage_entropies) if worker_res_wage_entropies else 0.0
            ),
            "worker_kl": np.mean(worker_kls),
            "worker_kl_to_prior": np.mean(worker_kls_to_prior) if worker_kls_to_prior else 0.0,
            "worker_kl_to_prior_by_skill": worker_kl_to_prior_by_skill,
            "worker_kl_to_prior_by_duration": worker_kl_to_prior_by_duration,
            "worker_res_wage_kl_uniform": (
                float(np.mean(worker_res_wage_kls_uniform)) if worker_res_wage_kls_uniform else 0.0
            ),
            "worker_res_wage_lr": float(
                next(
                    (
                        pg["lr"]
                        for pg in self.worker_actor_optimizer.param_groups
                        if pg.get("name", "") == "worker_actor_res_wage_head"
                    ),
                    self.worker_actor_optimizer.param_groups[0]["lr"],
                )
            ),
            "worker_explained_var": np.mean(worker_explained_vars),
            "worker_value_pred_errors": worker_value_pred_errors,
            "worker_raw_critic_value_std": worker_raw_critic_value_std,
            "worker_denorm_value_std": worker_denorm_value_std,
            "worker_popart_ret_mean_ema": (
                float(self._worker_ret_mean_ema.item())
                if hasattr(self._worker_ret_mean_ema, "item")
                else 0.0
            ),
            "worker_popart_ret_std_ema": (
                float(self._worker_ret_std_ema.item())
                if hasattr(self._worker_ret_std_ema, "item")
                else 1.0
            ),
            "worker_popart_enabled": self._worker_popart_enabled(),
            "firm_update_performed": True,
            "firm_policy_loss": np.mean(firm_policy_losses),
            "firm_value_loss": np.mean(firm_value_losses),
            "firm_entropy": np.mean(firm_entropies),
            "firm_kl": np.mean(firm_kls),
            "firm_explained_var": np.mean(firm_explained_vars),
            "firm_value_pred_errors": firm_value_pred_errors,
            "update_count": self.update_count,
            "firm_entropy_coef": firm_entropy_coef,
            "firm_kl_lr_reduced": firm_lr_reduced,
        }

    @torch.no_grad()
    def _maybe_emit_encoder_rep_audit(self) -> None:
        return

    def save(self, path: str, history: dict = None):
        checkpoint = {
            "worker_ac": self.worker_ac.state_dict(),
            "firm_ac": self.firm_ac.state_dict(),
            "worker_encoder_optimizer": self.worker_encoder_optimizer.state_dict(),
            "worker_actor_optimizer": self.worker_actor_optimizer.state_dict(),
            "worker_critic_optimizer": self.worker_critic_optimizer.state_dict(),
            "firm_encoder_optimizer": self.firm_encoder_optimizer.state_dict(),
            "firm_actor_optimizer": self.firm_actor_optimizer.state_dict(),
            "firm_critic_optimizer": self.firm_critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "worker_value_popart": {
                "enabled": self._worker_popart_enabled(),
                "initialized": bool(self._worker_value_popart_initialized),
                "mean_ema": (
                    float(self._worker_ret_mean_ema.item())
                    if hasattr(self._worker_ret_mean_ema, "item")
                    else 0.0
                ),
                "std_ema": (
                    float(self._worker_ret_std_ema.item())
                    if hasattr(self._worker_ret_std_ema, "item")
                    else 1.0
                ),
            },
            "rng_state": {
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python": _py_random.getstate(),
            },
        }
        if history is not None:
            checkpoint["history"] = history
        torch.save(checkpoint, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        def fix_state_dict(state_dict):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            return new_state_dict

        worker_state = checkpoint["worker_ac"]
        firm_state = checkpoint["firm_ac"]
        if any((k.startswith("_orig_mod.") for k in worker_state.keys())):
            worker_state = fix_state_dict(worker_state)
            firm_state = fix_state_dict(firm_state)
        current_worker_shapes = {k: v.shape for k, v in self.worker_ac.state_dict().items()}
        checkpoint_worker_keys = set(worker_state.keys())
        current_worker_keys = set(current_worker_shapes.keys())
        missing_from_checkpoint = current_worker_keys - checkpoint_worker_keys
        expanded_keys = []
        distilled_keys = []
        bad_shape_keys = []
        filtered_worker_state = {}
        for k in checkpoint_worker_keys & current_worker_keys:
            v = worker_state[k]
            cur_shape = current_worker_shapes[k]
            if v.shape == cur_shape:
                filtered_worker_state[k] = v
                continue
            if (
                k.endswith("actor.decision_head.weight")
                and len(v.shape) == 2
                and (len(cur_shape) == 2)
                and (v.shape == (2, cur_shape[1]))
                and (cur_shape[0] == 3)
            ):
                current_weight = self.worker_ac.actor.decision_head.weight.data
                expanded = current_weight.clone().to(dtype=v.dtype, device=v.device)
                expanded[:2] = v
                filtered_worker_state[k] = expanded
                distilled_keys.append(k)
                continue
            if k.endswith("actor.decision_head.bias") and v.shape == (2,) and (cur_shape == (3,)):
                current_bias = self.worker_ac.actor.decision_head.bias.data
                expanded = current_bias.clone().to(dtype=v.dtype, device=v.device)
                expanded[:2] = v
                filtered_worker_state[k] = expanded
                distilled_keys.append(k)
                continue
            if (
                len(v.shape) == len(cur_shape)
                and v.shape[:-1] == cur_shape[:-1]
                and (cur_shape[-1] == v.shape[-1] + 1)
            ):
                expanded = torch.zeros(*cur_shape, dtype=v.dtype, device=v.device)
                expanded[..., : v.shape[-1]] = v
                filtered_worker_state[k] = expanded
                expanded_keys.append(k)
            else:
                bad_shape_keys.append((k, tuple(v.shape), tuple(cur_shape)))
        if bad_shape_keys or missing_from_checkpoint:
            err_lines = [
                "[WORKER AC LOAD] Refusing to partial-load worker policy. For v5.3, the only supported worker AC mismatch is +1 on the input dim (attachment column appended)."
            ]
            if bad_shape_keys:
                err_lines.append(f"\n  {len(bad_shape_keys)} key(s) with non-v5.3 shape mismatch:")
                for k, ckpt_sh, cur_sh in bad_shape_keys[:10]:
                    err_lines.append(f"    - {k}: checkpoint {ckpt_sh} vs current {cur_sh}")
                if len(bad_shape_keys) > 10:
                    err_lines.append(f"    ... +{len(bad_shape_keys) - 10} more")
            if missing_from_checkpoint:
                err_lines.append(
                    f"\n  {len(missing_from_checkpoint)} parameter(s) in current model NOT present in checkpoint: {sorted(list(missing_from_checkpoint))[:10]}{(' ...' if len(missing_from_checkpoint) > 10 else '')}"
                )
            err_lines.append(
                "\nInvestigate architectural drift between this build and the parent checkpoint before forcing load (or build a checkpoint that matches the current model exactly)."
            )
            raise RuntimeError("\n".join(err_lines))
        self.worker_ac.load_state_dict(filtered_worker_state, strict=False)
        _warmup_active = bool(getattr(self, "_warmup_active", False))
        _worker_reset_optimizers = bool(expanded_keys) or bool(distilled_keys) or _warmup_active
        current_firm_shapes = {k: v.shape for k, v in self.firm_ac.state_dict().items()}
        filtered_firm_state = {
            k: v
            for k, v in firm_state.items()
            if k in current_firm_shapes and v.shape == current_firm_shapes[k]
        }
        self.firm_ac.load_state_dict(filtered_firm_state, strict=False)
        if "worker_actor_optimizer" in checkpoint:
            if not _worker_reset_optimizers:
                if "worker_encoder_optimizer" in checkpoint:
                    self.worker_encoder_optimizer.load_state_dict(
                        checkpoint["worker_encoder_optimizer"]
                    )
                self.worker_actor_optimizer.load_state_dict(checkpoint["worker_actor_optimizer"])
                self.worker_critic_optimizer.load_state_dict(checkpoint["worker_critic_optimizer"])
            try:
                if "firm_encoder_optimizer" in checkpoint:
                    self.firm_encoder_optimizer.load_state_dict(
                        checkpoint["firm_encoder_optimizer"]
                    )
                self.firm_actor_optimizer.load_state_dict(checkpoint["firm_actor_optimizer"])
                self.firm_critic_optimizer.load_state_dict(checkpoint["firm_critic_optimizer"])
            except (ValueError, RuntimeError):
                pass
        self.total_steps = checkpoint["total_steps"]
        self.update_count = checkpoint["update_count"]
        self.episode_count = checkpoint["episode_count"]
        popart_state = checkpoint.get("worker_value_popart", None)
        if popart_state is not None and popart_state.get("initialized", False):
            self._worker_ret_mean_ema = torch.tensor(
                float(popart_state.get("mean_ema", 0.0)), device=self.device
            )
            self._worker_ret_std_ema = torch.tensor(
                float(popart_state.get("std_ema", 1.0)), device=self.device
            )
            self._worker_value_popart_initialized = True
            if popart_state.get("enabled", False):
                if not self._worker_popart_enabled():
                    logger.warning(
                        f"The checkpoint was trained with return normalisation. It is switched on for this trainer. Mean {float(self._worker_ret_mean_ema):.4f}, standard deviation {float(self._worker_ret_std_ema):.4f}"
                    )
        elif self._worker_popart_enabled():
            logger.warning(
                "The checkpoint holds no return-normalisation statistics. Mean 0 and standard deviation 1 are used, so the critic outputs are read on the raw scale."
            )
        _rng = checkpoint.get("rng_state")
        if _rng is not None:
            try:
                np.random.set_state(_rng["numpy"])
                _torch_state = _rng["torch"]
                if isinstance(_torch_state, torch.Tensor):
                    torch.set_rng_state(_torch_state.cpu())
                _cuda_states = _rng.get("torch_cuda")
                if _cuda_states is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(
                        [s.cpu() if isinstance(s, torch.Tensor) else s for s in _cuda_states]
                    )
                _py_random.setstate(_rng["python"])
            except Exception as e:
                logger.warning(
                    f"Could not restore the random-generator state ({e}). The generators keep the state set by the seed of this process."
                )
        else:
            logger.warning(
                "The checkpoint holds no random-generator state, so the generators keep the state set by the seed of this process."
            )
