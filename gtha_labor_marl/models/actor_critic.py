import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict
from .dual_transformer import WorkerEncoder, FirmEncoder


class ActorNetwork(nn.Module):

    def __init__(self, encoder: nn.Module, action_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        layers = []
        in_dim = encoder.output_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(obs, context, history)
        logits = self.mlp(encoded)
        return logits

    def get_action(
        self,
        obs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        skill_param: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, context, history)
        if skill_param is not None:
            temperature = 1.0 / (skill_param.unsqueeze(-1) + 0.5)
            logits = logits / temperature
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return (action, log_prob, entropy)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        skill_param: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, context, history)
        if skill_param is not None:
            temperature = 1.0 / (skill_param.unsqueeze(-1) + 0.5)
            logits = logits / temperature
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return (log_prob, entropy)


class CriticNetwork(nn.Module):

    def __init__(
        self,
        global_state_dim: int,
        encoder: Optional[nn.Module] = None,
        hidden_dims: list = [256, 128],
        use_agent_obs: bool = True,
        detach_encoder: bool = False,
        final_bias_init: Optional[float] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_agent_obs = use_agent_obs and encoder is not None
        self.detach_encoder = bool(detach_encoder)
        in_dim = global_state_dim
        if self.use_agent_obs and encoder is not None:
            in_dim += encoder.output_dim
        layers = []
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        final_linear = nn.Linear(in_dim, 1)
        if final_bias_init is not None:
            nn.init.constant_(final_linear.bias, float(final_bias_init))
        layers.append(final_linear)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        global_state: torch.Tensor,
        agent_obs: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_agent_obs and agent_obs is not None:
            agent_encoded = self.encoder(agent_obs, context, history)
            if self.detach_encoder:
                agent_encoded = agent_encoded.detach()
            x = torch.cat([global_state, agent_encoded], dim=-1)
        else:
            x = global_state
        value = self.mlp(x)
        return value


class FactorizedWorkerActor(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dims: list = [256, 128],
        n_channel: int = 4,
        n_intensity: int = 5,
        res_wage_dim: int = 5,
        freeze_res_wage: bool = False,
        res_wage_frozen_level: int = 2,
        nilf_subtype_enabled: bool = False,
        eligibility_obs_col: int = 29,
        endogenous_retirement: bool = False,
        retire_logit_init_bias: float = -2.0,
        retire_sentinel_res_wage_idx: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_channel = int(n_channel)
        self.n_intensity = int(n_intensity)
        self.search_no_search_threshold = self.n_channel * self.n_intensity
        self.res_wage_dim = int(res_wage_dim)
        self.endogenous_retirement = bool(endogenous_retirement)
        self.retire_action_id = (
            2 * self.n_channel * self.n_intensity if self.endogenous_retirement else None
        )
        self.retire_sentinel_res_wage_idx = int(retire_sentinel_res_wage_idx)
        n_decision = 3 if self.endogenous_retirement else 2
        self.n_decision = n_decision
        self.search_dim = 2 * self.n_channel * self.n_intensity + (
            1 if self.endogenous_retirement else 0
        )
        self.nilf_subtype_enabled = bool(nilf_subtype_enabled)
        self.eligibility_obs_col = int(eligibility_obs_col)
        trunk_layers = []
        in_dim = encoder.output_dim
        for h in hidden_dims:
            trunk_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.decision_head = nn.Linear(in_dim, n_decision)
        self.channel_head = nn.Linear(in_dim, self.n_channel)
        self.intensity_head = nn.Linear(in_dim, self.n_intensity)
        self.res_wage_head = nn.Linear(in_dim, self.res_wage_dim)
        for head in (
            self.decision_head,
            self.channel_head,
            self.intensity_head,
            self.res_wage_head,
        ):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0.0)
        self.res_wage_head.bias.data[2] = 2.0
        if self.endogenous_retirement:
            with torch.no_grad():
                self.decision_head.weight.data[2].zero_()
                self.decision_head.bias.data[2] = float(retire_logit_init_bias)
        self.freeze_res_wage = bool(freeze_res_wage)
        self.res_wage_frozen_level = int(res_wage_frozen_level)
        if self.freeze_res_wage:
            for p in self.res_wage_head.parameters():
                p.requires_grad = False

    def forward(self, obs, context=None, history=None):
        hidden = self.trunk(self.encoder(obs, context, history))
        return (
            self.decision_head(hidden),
            self.channel_head(hidden),
            self.intensity_head(hidden),
            self.res_wage_head(hidden),
            hidden,
        )

    def _apply_temperature(self, logits, skill_param):
        if skill_param is None:
            return logits
        temperature = 1.0 / (skill_param.unsqueeze(-1) + 0.5)
        return logits / temperature

    def _decision_eligibility_mask(self, obs, decision_logits):
        apply_mask = (
            self.nilf_subtype_enabled
            and obs.dim() >= 2
            and (obs.shape[-1] > self.eligibility_obs_col)
        )
        if not apply_mask:
            return (decision_logits, None, False)
        eligible = obs[..., self.eligibility_obs_col] > 0.5
        decision_allowed = torch.ones_like(decision_logits, dtype=torch.bool)
        decision_allowed[~eligible, 1] = False
        masked = decision_logits.masked_fill(~decision_allowed, float("-inf"))
        return (masked, eligible, True)

    def _apply_retire_mask(self, decision_logits, retire_eligible_mask):
        if retire_eligible_mask is None:
            raise RuntimeError(
                "A retirement eligibility mask is required. It is computed by the environment from the worker state before the action (age, labour-force status and retirement status) and passed in by the rollout collector and the policy update."
            )
        if decision_logits.shape[-1] != 3:
            raise RuntimeError(
                f"A retirement mask was passed to a decision head with {decision_logits.shape[-1]} outputs (expected 3)."
            )
        masked = decision_logits.clone()
        ineligible = ~retire_eligible_mask.to(torch.bool)
        masked[..., 2] = torch.where(
            ineligible, torch.full_like(masked[..., 2], float("-inf")), masked[..., 2]
        )
        return masked

    def get_action(
        self,
        obs,
        context=None,
        history=None,
        deterministic: bool = False,
        skill_param=None,
        retire_eligible_mask=None,
    ):
        decision_logits, channel_logits, intensity_logits, res_wage_logits, _ = self.forward(
            obs, context, history
        )
        decision_logits = self._apply_temperature(decision_logits, skill_param)
        channel_logits = self._apply_temperature(channel_logits, skill_param)
        intensity_logits = self._apply_temperature(intensity_logits, skill_param)
        masked_decision_logits, eligible, apply_mask = self._decision_eligibility_mask(
            obs, decision_logits
        )
        if self.endogenous_retirement:
            masked_decision_logits = self._apply_retire_mask(
                masked_decision_logits, retire_eligible_mask
            )
        decision_dist = Categorical(logits=masked_decision_logits)
        if deterministic:
            decision_action = masked_decision_logits.argmax(dim=-1)
        else:
            decision_action = decision_dist.sample()
        log_p_decision = decision_dist.log_prob(decision_action)
        decision_entropy = decision_dist.entropy()
        is_search = decision_action == 1
        is_retire = (
            decision_action == 2 if self.endogenous_retirement else torch.zeros_like(is_search)
        )
        channel_dist = Categorical(logits=channel_logits)
        intensity_dist = Categorical(logits=intensity_logits)
        B = decision_action.shape[0]
        if deterministic:
            channel_action_full = channel_logits.argmax(dim=-1)
            intensity_action_full = intensity_logits.argmax(dim=-1)
            channel_action = torch.where(
                is_search, channel_action_full, torch.zeros_like(channel_action_full)
            )
            intensity_action = torch.where(
                is_search, intensity_action_full, torch.zeros_like(intensity_action_full)
            )
        else:
            channel_action = torch.zeros(B, dtype=torch.long, device=decision_action.device)
            intensity_action = torch.zeros(B, dtype=torch.long, device=decision_action.device)
            if int(is_search.sum().item()) > 0:
                searcher_chan = Categorical(logits=channel_logits[is_search]).sample()
                searcher_int = Categorical(logits=intensity_logits[is_search]).sample()
                channel_action[is_search] = searcher_chan
                intensity_action[is_search] = searcher_int
        is_search_f = is_search.float()
        log_p_chan = channel_dist.log_prob(channel_action) * is_search_f
        log_p_int = intensity_dist.log_prob(intensity_action) * is_search_f
        log_prob_search = log_p_decision + log_p_chan + log_p_int
        decision_probs = torch.softmax(masked_decision_logits, dim=-1)
        p_search = decision_probs[..., 1]
        chan_entropy = channel_dist.entropy()
        int_entropy = intensity_dist.entropy()
        search_entropy_total = decision_entropy + p_search * (chan_entropy + int_entropy)
        composite_action = torch.where(
            is_search,
            self.search_no_search_threshold + channel_action * self.n_intensity + intensity_action,
            torch.zeros_like(channel_action),
        )
        if self.endogenous_retirement:
            retire_sentinel = torch.full_like(composite_action, int(self.retire_action_id))
            composite_action = torch.where(is_retire, retire_sentinel, composite_action)
        if self.freeze_res_wage:
            res_wage_action = torch.full_like(composite_action, self.res_wage_frozen_level)
            log_prob = log_prob_search
            res_wage_entropy = torch.zeros_like(search_entropy_total)
            entropy = search_entropy_total
        else:
            res_wage_dist = Categorical(logits=res_wage_logits)
            if deterministic:
                res_wage_action = res_wage_logits.argmax(dim=-1)
            else:
                res_wage_action = res_wage_dist.sample()
            if self.endogenous_retirement:
                sentinel_idx = int(self.retire_sentinel_res_wage_idx)
                res_wage_action = torch.where(
                    is_search, res_wage_action, torch.full_like(res_wage_action, sentinel_idx)
                )
                log_p_res = res_wage_dist.log_prob(res_wage_action) * is_search_f
                res_wage_entropy = res_wage_dist.entropy() * p_search
                log_prob = log_prob_search + log_p_res
                entropy = search_entropy_total + res_wage_entropy
            else:
                log_prob = log_prob_search + res_wage_dist.log_prob(res_wage_action)
                res_wage_entropy = res_wage_dist.entropy()
                entropy = search_entropy_total + res_wage_entropy
        if apply_mask and (~eligible).any():
            unmasked_decision_dist = Categorical(logits=decision_logits)
            if deterministic:
                unmasked_intent_decision = decision_logits.argmax(dim=-1)
            else:
                unmasked_intent_decision = unmasked_decision_dist.sample()
            intent_is_search = unmasked_intent_decision == 1
            intent_composite = torch.where(
                intent_is_search,
                self.search_no_search_threshold
                + channel_action * self.n_intensity
                + intensity_action,
                torch.zeros_like(channel_action),
            )
            intent_search_action = torch.where(eligible, composite_action, intent_composite)
        else:
            intent_search_action = composite_action
        realized_combined = torch.stack([composite_action, res_wage_action], dim=1)
        intent_combined = torch.stack([intent_search_action, res_wage_action], dim=1)
        diagnostics = {
            "p_search": p_search.detach(),
            "decision_entropy": decision_entropy.detach(),
            "decision_logit_spread": (decision_logits[..., 1] - decision_logits[..., 0]).detach(),
        }
        if self.endogenous_retirement:
            p_retire = decision_probs[..., 2]
            diagnostics["p_retire"] = p_retire.detach()
            if retire_eligible_mask is not None:
                diagnostics["retire_eligible_count"] = int(
                    retire_eligible_mask.to(torch.bool).sum().item()
                )
            else:
                diagnostics["retire_eligible_count"] = -1
        return (realized_combined, log_prob, entropy, intent_combined, diagnostics)

    def evaluate_actions(
        self, obs, actions, context=None, history=None, skill_param=None, retire_eligible_mask=None
    ):
        composite_action = actions[..., 0]
        res_wage_action = actions[..., 1]
        if self.endogenous_retirement:
            is_retire = composite_action == int(self.retire_action_id)
            is_search = (composite_action >= self.search_no_search_threshold) & ~is_retire
            decision_action = torch.zeros_like(composite_action)
            decision_action = torch.where(
                is_search, torch.ones_like(decision_action), decision_action
            )
            decision_action = torch.where(
                is_retire, torch.full_like(decision_action, 2), decision_action
            )
        else:
            is_retire = torch.zeros_like(composite_action, dtype=torch.bool)
            is_search = composite_action >= self.search_no_search_threshold
            decision_action = is_search.long()
        offset = composite_action - self.search_no_search_threshold
        channel_action = torch.where(
            is_search,
            torch.div(offset, self.n_intensity, rounding_mode="floor"),
            torch.zeros_like(composite_action),
        )
        intensity_action = torch.where(
            is_search, offset % self.n_intensity, torch.zeros_like(composite_action)
        )
        decision_logits, channel_logits, intensity_logits, res_wage_logits, _ = self.forward(
            obs, context, history
        )
        decision_logits = self._apply_temperature(decision_logits, skill_param)
        channel_logits = self._apply_temperature(channel_logits, skill_param)
        intensity_logits = self._apply_temperature(intensity_logits, skill_param)
        masked_decision_logits, _eligible, _apply_mask = self._decision_eligibility_mask(
            obs, decision_logits
        )
        if self.endogenous_retirement:
            masked_decision_logits = self._apply_retire_mask(
                masked_decision_logits, retire_eligible_mask
            )
        decision_dist = Categorical(logits=masked_decision_logits)
        channel_dist = Categorical(logits=channel_logits)
        intensity_dist = Categorical(logits=intensity_logits)
        log_p_decision = decision_dist.log_prob(decision_action)
        is_search_f = is_search.float()
        log_p_chan = channel_dist.log_prob(channel_action) * is_search_f
        log_p_int = intensity_dist.log_prob(intensity_action) * is_search_f
        log_prob_search = log_p_decision + log_p_chan + log_p_int
        decision_entropy = decision_dist.entropy()
        decision_probs = torch.softmax(masked_decision_logits, dim=-1)
        p_search = decision_probs[..., 1]
        search_entropy_total = decision_entropy + p_search * (
            channel_dist.entropy() + intensity_dist.entropy()
        )
        if self.freeze_res_wage:
            log_prob = log_prob_search
            res_wage_entropy = torch.zeros_like(search_entropy_total)
            entropy = search_entropy_total
        else:
            res_wage_dist = Categorical(logits=res_wage_logits)
            if self.endogenous_retirement:
                log_p_res = res_wage_dist.log_prob(res_wage_action) * is_search_f
                res_wage_entropy = res_wage_dist.entropy() * p_search
                log_prob = log_prob_search + log_p_res
                entropy = search_entropy_total + res_wage_entropy
            else:
                log_prob = log_prob_search + res_wage_dist.log_prob(res_wage_action)
                res_wage_entropy = res_wage_dist.entropy()
                entropy = search_entropy_total + res_wage_entropy
        return (log_prob, entropy, search_entropy_total, res_wage_entropy, res_wage_logits)


class FactorizedWorkerActorCritic(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = WorkerEncoder(config.network)
        _subtype_enabled = bool(getattr(config.mnl, "nilf_subtype_enabled", False))
        _n_channel = int(getattr(config.action, "worker_search_channel_options", 4))
        _n_intensity = int(getattr(config.action, "worker_search_intensity_options", 5))
        _expected_action_size = 2 * _n_channel * _n_intensity + 1
        _actual_action_size = int(
            getattr(config.action, "worker_action_size", _expected_action_size)
        )
        if _actual_action_size != _expected_action_size:
            _retire_tag = " (Learnt endogenous retirement ON, +1 sentinel)"
            raise RuntimeError(
                f"FactorizedWorkerActorCritic requires worker_action_size == 2 * n_channel * n_intensity{' + 1'} = {_expected_action_size}{_retire_tag}, got {_actual_action_size}. action decomposition assumes the standard 2x{_n_channel}x{_n_intensity} composite layout{' plus retire sentinel'}."
            )
        self.actor = FactorizedWorkerActor(
            encoder=self.encoder,
            hidden_dims=config.network.actor_hidden_dims,
            n_channel=_n_channel,
            n_intensity=_n_intensity,
            res_wage_dim=getattr(config.action, "worker_res_wage_options", 5),
            freeze_res_wage=True,
            res_wage_frozen_level=getattr(config.training, "worker_res_wage_frozen_level", 2),
            nilf_subtype_enabled=_subtype_enabled,
            eligibility_obs_col=29,
            endogenous_retirement=True,
            retire_logit_init_bias=float(getattr(config.training, "retire_logit_init_bias", -2.0)),
            retire_sentinel_res_wage_idx=int(
                getattr(config.training, "retire_sentinel_res_wage_idx", 2)
            ),
        )
        self.critic = CriticNetwork(
            global_state_dim=getattr(
                config.environment,
                "effective_global_state_dim",
                config.environment.global_state_dim,
            ),
            encoder=self.encoder,
            hidden_dims=config.network.critic_hidden_dims,
            use_agent_obs=True,
            detach_encoder=False,
            final_bias_init=None,
        )

    def forward(self, obs, global_state, context=None, history=None):
        decision_logits, channel_logits, intensity_logits, res_wage_logits, _ = self.actor(
            obs, context, history
        )
        value = self.critic(global_state, obs, context, history)
        return {
            "decision_logits": decision_logits,
            "channel_logits": channel_logits,
            "intensity_logits": intensity_logits,
            "res_wage_logits": res_wage_logits,
            "value": value,
        }

    def get_action_and_value(
        self,
        obs,
        global_state,
        context=None,
        history=None,
        deterministic: bool = False,
        skill_param=None,
        retire_eligible_mask=None,
    ):
        action, log_prob, entropy, intent_action, diagnostics = self.actor.get_action(
            obs,
            context,
            history,
            deterministic=deterministic,
            skill_param=skill_param,
            retire_eligible_mask=retire_eligible_mask,
        )
        value = self.critic(global_state, obs, context, history)
        out = {
            "action": action,
            "intent_action": intent_action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value.squeeze(-1),
            "p_search": diagnostics["p_search"],
            "decision_entropy": diagnostics["decision_entropy"],
            "decision_logit_spread": diagnostics["decision_logit_spread"],
        }
        if "p_retire" in diagnostics:
            out["p_retire"] = diagnostics["p_retire"]
            out["retire_eligible_count"] = diagnostics["retire_eligible_count"]
        return out

    def evaluate(
        self,
        obs,
        actions,
        global_state,
        context=None,
        history=None,
        skill_param=None,
        retire_eligible_mask=None,
    ):
        log_prob, entropy, search_entropy, res_wage_entropy, res_wage_logits = (
            self.actor.evaluate_actions(
                obs,
                actions,
                context,
                history,
                skill_param=skill_param,
                retire_eligible_mask=retire_eligible_mask,
            )
        )
        value = self.critic(global_state, obs, context, history)
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "res_wage_logits": res_wage_logits,
            "search_entropy": search_entropy,
            "res_wage_entropy": res_wage_entropy,
            "value": value.squeeze(-1),
        }


class FirmActorCritic(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = FirmEncoder(config.network)
        self.actor = ActorNetwork(
            encoder=self.encoder,
            action_dim=config.action.firm_action_size,
            hidden_dims=config.network.actor_hidden_dims,
        )
        self.critic = CriticNetwork(
            global_state_dim=getattr(
                config.environment,
                "effective_global_state_dim",
                config.environment.global_state_dim,
            ),
            encoder=self.encoder,
            hidden_dims=config.network.critic_hidden_dims,
            use_agent_obs=True,
        )
        self._apply_posting_bias(config.action.firm_action_size)

    def _apply_posting_bias(self, action_size: int):
        output_layer = self.actor.mlp[-1]
        with torch.no_grad():
            bias = output_layer.bias
            actions_per_vac = action_size // 5
            for vac_level in range(5):
                start = vac_level * actions_per_vac
                end = start + actions_per_vac
                if vac_level <= 1:
                    bias[start:end] -= 0.5
                elif vac_level == 2:
                    bias[start:end] += 0.15
                elif vac_level == 3:
                    bias[start:end] += 0.25
                elif vac_level == 4:
                    bias[start:end] += 0.1

    def forward(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.actor(obs, context, history)
        value = self.critic(global_state, obs, context, history)
        return {"logits": logits, "value": value}

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        global_state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        action, log_prob, entropy = self.actor.get_action(obs, context, history, deterministic)
        value = self.critic(global_state, obs, context, history)
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value.squeeze(-1),
        }

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        global_state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        log_prob, entropy = self.actor.evaluate_actions(obs, actions, context, history)
        value = self.critic(global_state, obs, context, history)
        return {"log_prob": log_prob, "entropy": entropy, "value": value.squeeze(-1)}
