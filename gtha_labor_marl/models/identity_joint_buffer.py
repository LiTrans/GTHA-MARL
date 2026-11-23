import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


class WorkerGenerations:

    def __init__(self):
        self.episode = None
        self.live = {}
        self.generations = {}

    def snapshot(self, ids, workers, episode):
        ids = tuple(ids)
        require(type(episode) is int and episode >= 0, "Explicit episode identity required")
        require(
            len(ids) == len(set(ids)) and set(ids) == set(workers),
            "Worker registry/identity mismatch",
        )
        if episode != self.episode:
            require(
                self.episode is None or episode > self.episode, "Episode identity moved backwards"
            )
            self.live = {}
            self.generations = {}
            self.episode = episode
        current = {}
        keys = []
        for wid in ids:
            obj = workers[wid]
            person = getattr(obj, "worker", None)
            previous = self.live.get(wid)
            if previous is None or previous[0] is not obj or previous[1] is not person:
                self.generations[wid] = self.generations.get(wid, -1) + 1
            current[wid] = (obj, person)
            keys.append((episode, str(wid), self.generations[wid]))
        self.live = current
        require(len(keys) == len(set(keys)), "Worker string identities collide")
        return tuple(keys)


def worker_keys(trainer):
    tracker = getattr(trainer, "_joint_worker_generations", None)
    if tracker is None:
        tracker = trainer._joint_worker_generations = WorkerGenerations()
    env = trainer.env
    ids = tuple(env.worker_ids)
    index = env.fast_obs.state_table.worker_id_to_idx
    require(
        len(ids) == env.n_workers and all((index.get(w) == i for i, w in enumerate(ids))),
        "Worker observation rows do not match stable registry IDs",
    )
    return tracker.snapshot(ids, env.state_manager.workers, int(trainer.episode_count))


class JointIdentityRolloutBuffer:
    ragged_workers = True
    WORKER_FIELDS = (
        "worker_obs",
        "worker_actions",
        "worker_log_probs",
        "worker_rewards",
        "worker_values",
        "worker_dones",
        "worker_context",
        "worker_history",
        "worker_retire_eligible_mask",
    )
    FIRM_FIELDS = (
        "firm_obs",
        "firm_actions",
        "firm_log_probs",
        "firm_rewards",
        "firm_values",
        "firm_dones",
    )

    def __init__(
        self,
        buffer_size,
        n_workers,
        n_firms,
        worker_obs_dim,
        firm_obs_dim,
        global_state_dim,
        device="cpu",
        endogenous_retirement=False,
        **dimensions,
    ):
        self.buffer_size, self.n_workers, self.n_firms = (buffer_size, n_workers, n_firms)
        self.device = device
        self.endogenous_retirement = bool(endogenous_retirement)
        self.reset()

    def reset(self):
        for name in (
            *self.WORKER_FIELDS,
            *self.FIRM_FIELDS,
            "worker_advantages",
            "worker_returns",
            "firm_advantages",
            "firm_returns",
            "global_states",
            "worker_global_states",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self.steps = []
        self.ptr = 0
        self.full = False
        self.path_start_idx = 0
        self._pending = None
        self._last_keys = None
        self.boundary_worker_values = {}
        self.boundary_firm_values = {}
        self.calendar_steps = 0

    def set_transition(self, keys, next_keys, episode):
        require(self._pending is None, "Previous identity transition was not stored")
        keys, next_keys = (tuple(keys), tuple(next_keys))
        require(
            keys and len(keys) == len(set(keys)) and (len(next_keys) == len(set(next_keys))),
            "Duplicate/empty worker keys",
        )
        require(
            all((k[0] == episode for k in keys + next_keys)), "Transition crossed an unmarked reset"
        )
        if self.steps and (
            not (self.steps[-1].get("truncated", False) or self.steps[-1].get("terminated", False))
        ):
            require(
                self.steps[-1]["next_keys"] == keys,
                "The population changed between a step and the next action",
            )
        self._pending = (keys, next_keys, episode)

    def add(
        self,
        worker_obs,
        worker_action,
        worker_log_prob,
        worker_reward,
        worker_value,
        worker_done,
        worker_context,
        worker_history,
        firm_obs,
        firm_action,
        firm_log_prob,
        firm_reward,
        firm_value,
        firm_done,
        global_state,
        worker_retire_eligible_mask=None,
    ):
        require(
            self._pending is not None and len(self.steps) < self.buffer_size,
            "Missing identity metadata or rollout overflow",
        )
        keys, next_keys, episode = self._pending
        fields = dict(
            worker_obs=worker_obs,
            worker_actions=worker_action,
            worker_log_probs=worker_log_prob,
            worker_rewards=worker_reward,
            worker_values=worker_value,
            worker_dones=worker_done,
            worker_context=worker_context,
            worker_history=worker_history,
        )
        if self.endogenous_retirement:
            require(
                worker_retire_eligible_mask is not None
                and worker_retire_eligible_mask.dtype == torch.bool,
                "Action-time retirement mask required",
            )
            fields["worker_retire_eligible_mask"] = worker_retire_eligible_mask
        saved = {}
        for name, value in fields.items():
            require(
                value.shape[0] == len(keys) and bool(torch.isfinite(value).all()),
                "Worker decision row/value mismatch: " + name,
            )
            saved[name] = value.detach().clone().to(self.device)
        firms = dict(
            firm_obs=firm_obs,
            firm_actions=firm_action,
            firm_log_probs=firm_log_prob,
            firm_rewards=firm_reward,
            firm_values=firm_value,
            firm_dones=firm_done,
        )
        for name, value in firms.items():
            require(
                value.shape[0] == self.n_firms and bool(torch.isfinite(value).all()),
                "Fixed firm-axis mismatch: " + name,
            )
            firms[name] = value.detach().clone().to(self.device)
        require(bool(torch.isfinite(global_state).all()), "Nonfinite global state")
        self.steps.append(
            dict(
                keys=keys,
                next_keys=next_keys,
                episode=episode,
                fields=saved,
                firms=firms,
                global_state=global_state.detach().clone().to(self.device),
                truncated=False,
                terminated=bool(worker_done.bool().all()) and bool(firm_done.bool().all()),
            )
        )
        self._pending = None
        self.ptr = len(self.steps)
        self.full = self.ptr == self.buffer_size

    def mark_truncation(self, t, boundary_worker_value, boundary_firm_value):
        require(
            0 <= t < len(self.steps) and (not self.steps[t]["truncated"]),
            "Duplicate/invalid truncation",
        )
        require(
            not self.steps[t]["terminated"], "A true termination cannot be a time-limit truncation"
        )
        require(
            boundary_worker_value.shape == (len(self.steps[t]["next_keys"]),),
            "Pre-reset worker bootstrap identity rows differ",
        )
        require(
            boundary_firm_value.shape == (self.n_firms,), "Pre-reset firm bootstrap rows differ"
        )
        require(
            bool(torch.isfinite(boundary_worker_value).all())
            and bool(torch.isfinite(boundary_firm_value).all()),
            "Nonfinite pre-reset bootstrap",
        )
        self.steps[t]["truncated"] = True
        self.boundary_worker_values[t] = boundary_worker_value.detach().clone().to(self.device)
        self.boundary_firm_values[t] = boundary_firm_value.detach().clone().to(self.device)

    def set_last_keys(self, keys):
        keys = tuple(keys)
        require(len(keys) == len(set(keys)), "Duplicate final worker identities")
        self._last_keys = keys

    def compute_returns_and_advantages(
        self, last_worker_value, last_firm_value, gamma=0.99, gae_lambda=0.95
    ):
        require(
            self.steps and self._pending is None and (self._last_keys is not None),
            "Incomplete rollout/bootstrap identities",
        )
        require(0 <= gamma <= 1 and 0 <= gae_lambda <= 1, "Invalid GAE coefficients")
        require(
            last_worker_value.shape == (len(self._last_keys),),
            "Final bootstrap identity rows differ",
        )
        require(last_firm_value.shape == (self.n_firms,), "Final firm bootstrap rows differ")
        require(
            bool(torch.isfinite(last_worker_value).all())
            and bool(torch.isfinite(last_firm_value).all()),
            "Nonfinite final bootstrap",
        )
        future = {}
        for t in reversed(range(len(self.steps))):
            step = self.steps[t]
            f = step["fields"]
            if step["terminated"]:
                source_keys, next_values = ((), f["worker_values"].new_empty(0))
            elif step["truncated"]:
                source_keys, next_values = (step["next_keys"], self.boundary_worker_values[t])
            elif t == len(self.steps) - 1:
                source_keys, next_values = (self._last_keys, last_worker_value)
                require(
                    source_keys == step["next_keys"],
                    "The final observation is not the recorded next state",
                )
            else:
                following = self.steps[t + 1]
                require(
                    following["keys"] == step["next_keys"],
                    "A next decision is missing or an episode boundary is unmarked",
                )
                source_keys, next_values = (following["keys"], following["fields"]["worker_values"])
            lookup = dict(zip(source_keys, next_values.detach().reshape(-1).tolist()))
            active = set(step["next_keys"])
            terminated = f["worker_dones"].reshape(-1).bool()
            alive = (
                torch.tensor(
                    [key in active for key in step["keys"]], dtype=torch.bool, device=self.device
                )
                & ~terminated
            )
            aligned = []
            trace = []
            for i, key in enumerate(step["keys"]):
                if bool(alive[i]):
                    require(key in lookup, "The value of the next state is missing")
                    aligned.append(lookup[key])
                else:
                    aligned.append(0.0)
                if bool(alive[i]) and (not step["truncated"]) and (t < len(self.steps) - 1):
                    require(key in future, "The advantage of the next state is missing")
                    trace.append(future[key])
                else:
                    trace.append(0.0)
            next_tensor = torch.tensor(aligned, dtype=f["worker_values"].dtype, device=self.device)
            continuation = torch.tensor(trace, dtype=f["worker_values"].dtype, device=self.device)
            delta = (
                f["worker_rewards"].reshape(-1)
                + gamma * next_tensor
                - f["worker_values"].reshape(-1)
            )
            advantage = delta + gamma * gae_lambda * continuation
            f["worker_advantages"], f["worker_returns"] = (
                advantage,
                advantage + f["worker_values"].reshape(-1),
            )
            f["worker_dones"] = (~alive).float()
            future = dict(zip(step["keys"], advantage.detach().tolist()))
        gae = torch.zeros(self.n_firms, device=self.device)
        for t in reversed(range(len(self.steps))):
            step = self.steps[t]
            f = step["firms"]
            if step["truncated"]:
                delta = f["firm_rewards"] + gamma * self.boundary_firm_values[t] - f["firm_values"]
                gae = delta
            else:
                following = (
                    last_firm_value
                    if t == len(self.steps) - 1
                    else self.steps[t + 1]["firms"]["firm_values"]
                )
                alive = 1.0 - f["firm_dones"]
                delta = f["firm_rewards"] + gamma * following * alive - f["firm_values"]
                gae = delta + gamma * gae_lambda * alive * gae
            f["firm_advantages"], f["firm_returns"] = (gae.clone(), gae + f["firm_values"])
        self.calendar_steps = len(self.steps)
        self.offsets = [0]
        self.sample_keys = []
        for step in self.steps:
            self.offsets.append(self.offsets[-1] + len(step["keys"]))
            self.sample_keys.extend(step["keys"])
        names = tuple(self.steps[0]["fields"])
        for name in names:
            packed = torch.cat([step["fields"][name] for step in self.steps], dim=0)
            setattr(self, name, packed.unsqueeze(1))
        for name in (*self.FIRM_FIELDS, "firm_advantages", "firm_returns"):
            setattr(self, name, torch.stack([step["firms"][name] for step in self.steps]))
        self.global_states = torch.stack([step["global_state"] for step in self.steps])
        self.worker_global_states = torch.cat(
            [step["global_state"].unsqueeze(0).expand(len(step["keys"]), -1) for step in self.steps]
        )
        self.worker_field_names = names
        self.ptr = len(self.sample_keys)
        for step in self.steps:
            step.pop("fields")
            step.pop("firms")
            step.pop("global_state")

    def get_batches(self, batch_size):
        require(
            type(batch_size) is int and batch_size > 0 and (self.calendar_steps > 0),
            "Invalid calendar minibatch",
        )
        order = np.random.permutation(self.calendar_steps)
        batches = []
        for start in range(0, self.calendar_steps, batch_size):
            times = order[start : start + batch_size]
            rows = torch.tensor(
                [i for t in times for i in range(self.offsets[t], self.offsets[t + 1])],
                dtype=torch.long,
                device=self.device,
            )
            batch = {name: getattr(self, name)[rows] for name in self.worker_field_names}
            for name in (*self.FIRM_FIELDS, "firm_advantages", "firm_returns"):
                batch[name] = getattr(self, name)[times]
            batch["global_states"] = self.global_states[times]
            batch["worker_global_states"] = self.worker_global_states[rows]
            batches.append(batch)
        return batches
