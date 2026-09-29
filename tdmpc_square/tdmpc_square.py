
import numpy as np
import torch
import torch.nn.functional as F

from collections import deque

from tdmpc_square.common import math
from tdmpc_square.common.scale import RunningScale
from tdmpc_square.common.world_model import WorldModel

from copy import deepcopy


class TDMPC2:
	"""Planning–Learning MPC agent."""

	def __init__(self, cfg):
		self.cfg = cfg
		if torch.cuda.is_available():
			self.device = torch.device(cfg.device)
		else:
			self.device = torch.device("cpu")
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam(
			[
				{
					"params": self.model._encoder.parameters(),
					"lr": self.cfg.lr * self.cfg.enc_lr_scale,
				},
				{"params": self.model._dynamics.parameters()},
				{"params": self.model._reward.parameters()},
				{"params": self.model._Qs.parameters()},
				{
					"params": self.model._task_emb.parameters()
					if self.cfg.multitask
					else []
				},
			],
			lr=self.cfg.lr,
		)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
		)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.log_pi_scale = RunningScale(cfg)
		self.cfg.iterations += 2 * int(
			cfg.action_dim >= 20
		)
		self.discount = (
			torch.tensor(
				[self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
				device=self.device,
			)
			if self.cfg.multitask
			else self._get_discount(cfg.episode_length)
		)

		self.u_mean = 0.0
		self.u_std = 1.0

		self._ep_return_stats = deque(maxlen=int(getattr(self.cfg, "si_stats_window", 256)))
		self._last_si_stats = {
			"si/active": 0.0,
			"si/loss": 0.0,
			"si/weight_mean": 0.0,
			"si/weight_max": 0.0,
			"si/G_med": 0.0,
			"si/G_std": 0.0,
		}

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
				episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
				float: Discount factor for the task.
		"""
		frac = episode_length / self.cfg.discount_denom
		return min(
			max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
		)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
				fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
				fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=self.device)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, use_pi=False, return_info=False):
		"""
		Select an action by planning in latent space or by using the policy prior.

		Args:
				obs (torch.Tensor): Observation from the environment.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (int): Task index (only used for multi-task experiments).
				use_pi (bool): Whether to use the policy prior instead of planning.
				return_info (bool): Whether to return additional info such as planner diagnostics.
		Returns:
			If return_info=False:
				a, mu, std
			If return_info=True:
				a, mu, std, plan_info
		"""
		if not isinstance(obs, torch.Tensor):
			obs = torch.as_tensor(obs, dtype=torch.float32)
		obs = obs.to(self.device, non_blocking=True)
		if obs.ndim == 1:
			obs = obs.unsqueeze(0)
		if task is not None and not isinstance(task, torch.Tensor):
			task = torch.tensor([task], device=self.device)


		z = self.model.encode(obs, task)

		plan_info = None
		if self.cfg.mpc and not use_pi:
			if return_info:
				a, mu, std, plan_info = self.plan(
					z, t0=t0, eval_mode=eval_mode, task=task, return_info=True
				)
			else:
				a, mu, std = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		else:
			mu, pi, log_pi, log_std = self.model.pi(z, task)
			if eval_mode:
				a = mu[0]
			else:
				a = pi[0]
			mu, std = mu[0], log_std.exp()[0]
		if return_info:
			return a.cpu(), mu.cpu(), std.cpu(), plan_info
		return a.cpu(), mu.cpu(), std.cpu()

	@torch.no_grad()
	def _ensemble_stats(self, z, a, task, target=False):
		"""
		Returns ensemble statistics for Q(z, a):
			q_mean: [batch]
			q_std:  [batch]
			q_min:  [batch]
			q_bottomk: [batch]  (mean of k smallest heads)
			q_all: [num_q, batch]  (raw per-head Q values, used by random_min/random_avg)
		"""

		qs = self.model.Q(z, a, task, return_type="all", target=target)


		qs = torch.stack(
			[math.two_hot_inv(q, self.cfg) for q in qs], dim=0
		)

		q_mean = qs.mean(dim=0)
		q_std = qs.std(dim=0).clamp_(min=self.cfg.q_std_min, max=self.cfg.q_std_max)
		q_min = qs.min(dim=0).values
		k = min(self.cfg.actor_bottomk, qs.shape[0])
		qs_sorted, _ = torch.sort(qs, dim=0)
		q_bottomk = qs_sorted[:k].mean(dim=0)

		return q_mean, q_std, q_min, q_bottomk, qs


	def _adaptive_coef(self, q_std, coef_max):
		"""
		Adaptive coefficient:
			coef = coef_max * sigmoid((q_std - u_mean) / u_std)
		q_std: [batch]
		returns: [batch]
		"""
		u_mean = torch.as_tensor(self.u_mean, device=q_std.device, dtype=q_std.dtype)
		u_std = torch.as_tensor(self.u_std, device=q_std.device, dtype=q_std.dtype)
		u = (q_std - u_mean) / (u_std + 1e-6)
		return coef_max * torch.sigmoid(u)

	def _adaptive_coef_floored(self, q_std, coef_max, coef_floor):
		"""
		Floor + adaptive cap (σ-gated pessimism family):
			coef = coef_floor + (coef_max - coef_floor) * sigmoid((q_std - u_mean) / u_std)
		Recovers _adaptive_coef when coef_floor == 0.
		Gives constant coef_max when coef_floor == coef_max.
		"""
		u_mean = torch.as_tensor(self.u_mean, device=q_std.device, dtype=q_std.dtype)
		u_std = torch.as_tensor(self.u_std, device=q_std.device, dtype=q_std.dtype)
		u = (q_std - u_mean) / (u_std + 1e-6)
		return coef_floor + (coef_max - coef_floor) * torch.sigmoid(u)

	@torch.no_grad()
	def _update_si_stats(self, episode_return_batch):
		"""Push unique episode-returns from this batch into the running pool.

		Episode-return values within a horizon slice are duplicated (one per
		transition in the slice). We dedupe per-batch to avoid biasing the
		median toward whichever episodes happen to be sampled densely.
		"""
		if episode_return_batch is None or not bool(getattr(self.cfg, "self_imitation", False)):
			return
		x = episode_return_batch.detach().reshape(-1).cpu()

		x = x[torch.isfinite(x)]
		if x.numel() == 0:
			return
		for v in torch.unique(x).tolist():
			self._ep_return_stats.append(float(v))

	def _si_running_stats(self):
		"""Return (G_med, G_std) over the running episode-return pool."""
		if len(self._ep_return_stats) == 0:
			return 0.0, 0.0
		t = torch.tensor(list(self._ep_return_stats), dtype=torch.float32)


		t = t[torch.isfinite(t)]
		if t.numel() == 0:
			return 0.0, 0.0
		return float(t.median().item()), float(t.std(unbiased=False).item())

	@torch.no_grad()
	def _update_uncertainty_stats(self, q_std_values):
		"""
		Update running mean/std of ensemble disagreement using EMA.
		q_std_values: 1D tensor or flattened tensor of observed q_std values
		"""
		x = q_std_values.detach().float().reshape(-1)
		if x.numel() == 0:
			return

		batch_mean = x.mean().item()
		batch_std = x.std(unbiased=False).item()

		tau = self.cfg.uncertainty_ema_tau
		self.u_mean = (1.0 - tau) * self.u_mean + tau * batch_mean
		self.u_std = (1.0 - tau) * self.u_std + tau * max(batch_std, 1e-6)

	def _aggregate_q(self, q_mean, q_std, q_min, q_bottomk, mode, coef_max=None, coef=None, start_frac=0.35, sharpness=12.0, ucb_coef=None, q_all=None):
		"""Aggregate decoded heads: average all, or random average/minimum of two."""

		if mode == "avg":
			return q_mean
		elif mode == "random_avg":

			assert q_all is not None, "random_avg requires q_all=[num_q, batch]"
			qidx = torch.randperm(q_all.shape[0], device=q_all.device)[:2]
			return q_all[qidx].mean(0)
		elif mode == "random_min":

			assert q_all is not None, "random_min requires q_all=[num_q, batch]"
			qidx = torch.randperm(q_all.shape[0], device=q_all.device)[:2]
			return q_all[qidx].min(0).values
		else:
			raise NotImplementedError(f"Unknown Q aggregation mode: {mode}")

	def _actor_q(self, z, a, task):
		"""
		Q used inside the actor objective.
		"""
		q_mean, q_std, q_min, q_bottomk, q_all = self._ensemble_stats(z, a, task, target=False)

		return self._aggregate_q(
			q_mean=q_mean,
			q_std=q_std,
			q_min=q_min,
			q_bottomk=q_bottomk,
			q_all=q_all,
			mode=self.cfg.actor_q_mode,
			coef_max=self.cfg.actor_uncertainty_max,
			coef=self.cfg.actor_uncertainty_coef,
			start_frac=self.cfg.actor_td_start_frac,
			sharpness=self.cfg.actor_td_sharpness,
			ucb_coef=getattr(self.cfg, 'actor_ucb_coef', None),
		)

	@torch.no_grad()
	def _target_q(self, z, a, task):
		"""
		Q used inside the TD target.
		"""
		q_mean, q_std, q_min, q_bottomk, q_all = self._ensemble_stats(z, a, task, target=True)


		self._update_uncertainty_stats(q_std)

		return self._aggregate_q(
			q_mean=q_mean,
			q_std=q_std,
			q_min=q_min,
			q_bottomk=q_bottomk,
			q_all=q_all,
			mode=self.cfg.td_target_q_mode,
			coef_max=self.cfg.lambda_uncertainty_td_max,
			coef=None,
			start_frac=self.cfg.td_start_frac,
			sharpness=self.cfg.td_sharpness,
		)


	# Source uses online all-head mean/disagreement; see implementation notes.
	@torch.no_grad()
	def _estimate_value_adaptive(self, z, actions, task, horizon, eval_mode=False):
		"""
		Adaptive uncertainty-aware MPPI trajectory scoring.

		Score:
			sum_t gamma^t [ r_t - beta_t * q_std_t ]
			+ gamma^H [ q_mean_H - lambda_H * q_std_H ]

		where:
			beta_t   = beta_max   * sigmoid((q_std_t - u_mean) / u_std)
			lambda_H = lambda_max * sigmoid((q_std_H - u_mean) / u_std)
		"""
		G, discount = 0, 1
		observed_q_stds = []

		for t in range(horizon):
			a = actions[t]


			reward = math.two_hot_inv(self.model.reward(z, a, task), self.cfg)


			q_mean, q_std, q_min, q_bottomk, _ = self._ensemble_stats(z, a, task, target=False)
			observed_q_stds.append(q_std)


			beta_t = self._adaptive_coef(q_std, self.cfg.beta_uncertainty_max)


			reward = reward - beta_t * q_std


			z = self.model.next(z, a, task)


			G += discount * reward

			discount *= (
				self.discount[torch.tensor(task)]
				if self.cfg.multitask
				else self.discount
			)


		a_pi = self.model.pi(z, task)[1]


		q_mean, q_std, q_min, q_bottomk, _ = self._ensemble_stats(z, a_pi, task, target=False)
		observed_q_stds.append(q_std)


		lam_H = self._adaptive_coef_floored(
			q_std,
			self.cfg.lambda_uncertainty_max,
			getattr(self.cfg, "lambda_uncertainty_floor", 0.0),
		)
		terminal_Q = q_mean - lam_H * q_std


		self._update_uncertainty_stats(torch.cat([x.reshape(-1) for x in observed_q_stds], dim=0))

		return G + discount * terminal_Q

	def _correlated_noise(self, shape, beta=None):
		"""
		Generate AR(1) temporally correlated noise on GPU.
		Handles 3D (H, N, A) for plan().
		Returns i.i.d. Gaussian when beta=0.
		"""
		if beta is None:
			beta = getattr(self.cfg, 'planner_noise_beta', 0.0)
		white = torch.randn(shape, device=self.device)
		if beta <= 0:
			return white
		alpha = (1.0 - beta ** 2) ** 0.5
		if len(shape) == 3:
			corr = torch.empty_like(white)
			corr[0] = white[0]
			for t in range(1, shape[0]):
				corr[t] = beta * corr[t - 1] + alpha * white[t]
		elif len(shape) == 4:
			corr = torch.empty_like(white)
			corr[:, 0] = white[:, 0]
			for t in range(1, shape[1]):
				corr[:, t] = beta * corr[:, t - 1] + alpha * white[:, t]
		else:
			return white
		return corr

	def _estimate_value_dispatch(self, z, actions, task, horizon):
		"""Route to the proposed adaptive single-state value estimator."""
		return self._estimate_value_adaptive(z, actions, task, horizon).nan_to_num_(0)

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None, return_info=False):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
				z (torch.Tensor): Latent state from which to plan.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (Torch.Tensor): Task index (only used for multi-task experiments).
				return_info (bool): Whether to return additional planning information.

		Returns:
			If return_info=False:
				a, mu, std
			If return_info=True:
				a, mu, std, plan_info

			plan_info contains:
				- seq: chosen H-step action sequence
				- pred_value: planner-predicted value of that chosen sequence
				- elite_probs: final MPPI elite probabilities
				- planner_type: adaptive
		"""

		H = self.cfg.horizon

		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				H,
				self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(H - 1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]


		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(H, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(
			H, self.cfg.action_dim, device=self.device
		)
		if not t0:

			prev = self._prev_mean[1:]
			copy_len = min(prev.shape[0], H - 1)
			mean[:copy_len] = prev[:copy_len]
		actions = torch.empty(
			H,
			self.cfg.num_samples,
			self.cfg.action_dim,
			device=self.device,
		)
		if self.cfg.num_pi_trajs > 0:
			actions[:, : self.cfg.num_pi_trajs] = pi_actions


		for _iter in range(self.cfg.iterations):

			N_sample = self.cfg.num_samples - self.cfg.num_pi_trajs
			noise = self._correlated_noise((H, N_sample, self.cfg.action_dim))
			actions[:, self.cfg.num_pi_trajs:] = (
				mean.unsqueeze(1) + std.unsqueeze(1) * noise
			).clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]


			value = self._estimate_value_dispatch(z, actions, task, H)


			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value = value[elite_idxs]
			elite_actions = actions[:, elite_idxs]
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			std = torch.sqrt(
				torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1)
				/ (score.sum(0) + 1e-9)
			).clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]


		probs = score.squeeze(-1)

		if eval_mode:
			chosen_idx = int(torch.argmax(probs).item())
		else:
			chosen_idx = int(
				np.random.choice(
					np.arange(probs.shape[0]),
					p=probs.detach().cpu().numpy(),
				)
			)

		chosen_seq = elite_actions[:, chosen_idx]
		chosen_value = elite_value[chosen_idx].squeeze()

		self._prev_mean = mean
		mu = chosen_seq[0]
		first_std = std[0]

		if not eval_mode:
			a = mu + first_std * torch.randn(self.cfg.action_dim, device=first_std.device)
		else:
			a = mu

		if return_info:
			plan_info = {
				"seq": chosen_seq.detach().clone(),
				"pred_value": chosen_value.detach().clone(),
				"elite_probs": probs.detach().clone(),
				"planner_type": self.cfg.planner_type,
			}
			return a.clamp_(-1, 1), mu, first_std, plan_info

		return a.clamp_(-1, 1), mu, first_std

	def update_pi(self, zs, action, mu, std, task, episode_return=None):
		"""
		Update policy using a sequence of latent states.

		Args:
				zs (torch.Tensor): Sequence of latent states.
				action (torch.Tensor): Sequence of actions.
				task (torch.Tensor): Task index (only used for multi-task experiments).
				episode_return (torch.Tensor or None): Per-slice realized episode return
					G(τ) used by the self-imitation loss. Shape [B]. Ignored unless
					self.cfg.self_imitation is True.

		Returns:
				float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)

		if self.cfg.actor_conservatism:

			qs = self._actor_q(zs, pis, task)
		else:

			qs = self.model.Q(zs, pis, task, return_type="avg")

		self.scale.update(qs[0])
		qs = self.scale(qs)

		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))


		self._last_gate_stats = {
			"gate_mean": 0.0,
			"gate_adv_mean": 0.0,
			"gate_unc_mean": 0.0,
			"gate_stab_mean": 0.0,
			"planner_adv_mean": 0.0,
			"planner_qstd_mean": 0.0,
			"planner_std_mean": 0.0,
		}

		if self.cfg.actor_mode=="residual":

			action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
			std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
			eps = (pis - mu) / std
			log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)


			log_pis_prior = self.scale(log_pis_prior) if self.scale.value > self.cfg.scale_threshold else torch.zeros_like(log_pis_prior)

			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
			pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss

		else:
			raise NotImplementedError

		self._last_si_stats = {
			"si/active": 0.0,
			"si/loss": 0.0,
			"si/weight_mean": 0.0,
			"si/weight_max": 0.0,
			"si/G_med": 0.0,
			"si/G_std": 0.0,
		}
		if bool(getattr(self.cfg, "self_imitation", False)) and episode_return is not None:
			current_step = int(getattr(self, "_step", 0))
			warmup = int(getattr(self.cfg, "si_warmup_steps", 200000))
			G_med, G_std = self._si_running_stats()
			min_gstd = float(getattr(self.cfg, "si_min_gstd", 0.0))
			self._last_si_stats["si/G_med"] = float(G_med)
			self._last_si_stats["si/G_std"] = float(G_std)
			if current_step >= warmup and G_std >= min_gstd:
				temp_cfg = getattr(self.cfg, "si_temperature", "auto")
				beta = G_std if (isinstance(temp_cfg, str) and temp_cfg == "auto") else float(temp_cfg)
				beta = max(beta, 1e-3)
				w_max = float(getattr(self.cfg, "si_max_weight", 10.0))
				si_coef = float(getattr(self.cfg, "si_coef", 0.5))
				G_med_t = torch.as_tensor(G_med, device=self.device, dtype=action.dtype)
				G = episode_return.to(self.device, dtype=action.dtype).reshape(-1)
				w = torch.exp((G - G_med_t) / beta).clamp(0.0, w_max).detach()

				T = zs.size(0)
				w_TB = w.unsqueeze(0).expand(T, -1)
				log_pi_act = self.model.log_pi_action(zs, action, task)
				if log_pi_act.dim() == 3 and log_pi_act.size(-1) == 1:
					log_pi_act = log_pi_act.squeeze(-1)
				si_loss = si_coef * (-(w_TB * log_pi_act).mean())
				pi_loss = pi_loss + si_loss
				self._last_si_stats = {
					"si/active":      1.0,
					"si/loss":        float(si_loss.detach().item()),
					"si/weight_mean": float(w.mean().item()),
					"si/weight_max":  float(w.max().item()),
					"si/G_med":       float(G_med),
					"si/G_std":       float(G_std),
				}

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model._pi.parameters(), self.cfg.grad_clip_norm
		)

		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item(), q_loss.item(), prior_loss.item(), self._last_gate_stats

	# Hybrid observed rewards and actor/model rollout.
	@torch.no_grad()
	def _td_target(self, next_z, reward, task, z=None):
		"""
		Compute the TD-target for (obs[t], action[t]) transitions.

		Buffer semantics:
		- next_z[t] = encode(obs[t+1]) = state after action[t]
		- reward[t] = r_t = reward for taking action[t] from obs[t]
		- Standard 1-step target for (obs[t], action[t]):
			Q_target[t] = reward[t] + γ · Q(next_z[t], π(next_z[t]))
		- Multi-step n target for (obs[t], action[t]):
			Q_target[t] = Σ_{k=0..n-1} γ^k · r_{t+k} + γ^n · Q(s_{t+n}, π)
			= reward[t] + γ·(reward[t+1] + γ·(... + γ·Q(s_{t+n}, π)))
			where s_{t+n} = next_z[t+n-1] (the state after action[t+n-1]).

		When t+n exceeds the buffer horizon T, we roll the learned model
		forward from the last available in-buffer latent to compute extra
		rewards and the terminal Q.

		mtd_mode='imagination_only' uses pure model rollout from z[t] for the
		full n steps, ignoring all buffer rewards. Requires the caller to pass
		z = encode(obs[:-1]).
		"""
		discount = (
			self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		)

		def _terminal_q(z):
			pi = self.model.pi(z, task)[1]
			if self.cfg.actor_conservatism:
				return self._target_q(z, pi, task)
			else:
				return self.model.Q(z, pi, task, return_type="min", target=True)

		if not getattr(self.cfg, 'multistep_td', False):

			return reward + discount * _terminal_q(next_z)

		T = next_z.shape[0]
		n = int(getattr(self.cfg, 'multistep_td_n', 3))

		mtd_mode = str(getattr(self.cfg, 'mtd_mode', 'full'))
		targets = torch.empty_like(reward)

		if mtd_mode == 'imagination_only':
			assert z is not None, \
				"mtd_mode='imagination_only' requires z=encode(obs[:-1]) to be passed to _td_target"
			for t in range(T):
				z_curr = z[t]
				G = torch.zeros_like(reward[t])
				disc = 1.0
				for _ in range(n):
					pi_k = self.model.pi(z_curr, task)[1]
					r_model = math.two_hot_inv(self.model.reward(z_curr, pi_k, task), self.cfg)
					G = G + disc * r_model
					z_curr = self.model.next(z_curr, pi_k, task)
					disc = disc * discount
				targets[t] = G + disc * _terminal_q(z_curr)
			return targets

		for t in range(T):

			G = torch.zeros_like(reward[t])
			disc = 1.0


			z_curr = None


			effective_steps = 0

			for k in range(n):
				src_idx = t + k
				if src_idx < T and z_curr is None:

					G = G + disc * reward[src_idx]
					effective_steps += 1

					if src_idx == T - 1 and k < n - 1:

						z_curr = next_z[src_idx]
				else:

					if mtd_mode == 'buffer_only':

						break

					pi_k = self.model.pi(z_curr, task)[1]
					r_model = math.two_hot_inv(self.model.reward(z_curr, pi_k, task), self.cfg)
					G = G + disc * r_model
					z_curr = self.model.next(z_curr, pi_k, task)
					effective_steps += 1
				disc = disc * discount

			if z_curr is not None:
				z_terminal = z_curr
			else:

				z_terminal = next_z[t + effective_steps - 1]

			G = G + disc * _terminal_q(z_terminal)
			targets[t] = G

		return targets

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
				buffer (common.buffer.Buffer): Replay buffer.

		Returns:
				dict: Dictionary of training statistics.
		"""
		if self.cfg.multitask and self.cfg.task in {"mt30","mt80"}:

			obs, action, reward, task = buffer.sample()
			mu = action.detach().clone()
			std = torch.full_like(action, self.cfg.max_std)
			episode_return = None
		else:

			obs, action, mu, std, reward, task, episode_return = buffer.sample()

		# Source-episode undiscounted returns enter the finite statistics queue.
		self._update_si_stats(episode_return)


		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			if str(getattr(self.cfg, 'mtd_mode', 'full')) == 'imagination_only':
				z_curr_all = self.model.encode(obs[:-1], task)
				td_targets = self._td_target(next_z, reward, task, z=z_curr_all)
			else:
				td_targets = self._td_target(next_z, reward, task)


		self.optim.zero_grad(set_to_none=True)
		self.model.train()


		zs = torch.empty(
			self.cfg.horizon + 1,
			self.cfg.batch_size,
			self.cfg.latent_dim,
			device=self.device,
		)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t + 1] = z


		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type="all")
		reward_preds = self.model.reward(_zs, action, task)


		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += (
				math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
				* self.cfg.rho**t
			)
			for q in range(self.cfg.num_q):
				value_loss += (
					math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
					* self.cfg.rho**t
				)
		consistency_loss *= 1 / self.cfg.horizon
		reward_loss *= 1 / self.cfg.horizon
		value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)

		total_loss = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.value_coef * value_loss
		)


		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(), self.cfg.grad_clip_norm
		)
		self.optim.step()


		with torch.no_grad():
			_zs_det = _zs.detach()
			_, pis_diag, _, _ = self.model.pi(_zs_det, task)
			q_mean_diag, q_std_diag, q_min_diag, _, _ = self._ensemble_stats(
				_zs_det.reshape(-1, _zs_det.shape[-1]),
				pis_diag.reshape(-1, pis_diag.shape[-1]),
				task, target=False
			)

			td_target_scalar = td_targets.reshape(-1)


		_ep_return_arg = episode_return.detach() if (episode_return is not None) else None
		pi_loss, pi_loss_q, pi_loss_prior, gate_stats = self.update_pi(_zs_det, action.detach(), mu.detach(), std.detach(), task, _ep_return_arg)


		self.model.soft_update_target_Q()


		self.model.eval()

		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"pi_loss_q": pi_loss_q,
			"pi_loss_prior": pi_loss_prior,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),

			"q_mean": float(q_mean_diag.mean().item()),
			"q_std_mean": float(q_std_diag.mean().item()),
			"q_std_max": float(q_std_diag.max().item()),
			"q_min_mean": float(q_min_diag.mean().item()),

			"td_target_mean": float(td_target_scalar.mean().item()),
			"td_target_std": float(td_target_scalar.std().item()),

			"q_td_error": float((q_mean_diag.mean() - td_target_scalar.mean()).abs().item()),

			"uncertainty_ema_mean": float(self.u_mean),
			"uncertainty_ema_std": float(self.u_std),

			"consistency_loss_raw": float(consistency_loss.item()),

			"gate_mean": gate_stats.get("gate_mean", 0.0),
			"gate_adv_mean": gate_stats.get("gate_adv_mean", 0.0),
			"gate_unc_mean": gate_stats.get("gate_unc_mean", 0.0),
			"gate_stab_mean": gate_stats.get("gate_stab_mean", 0.0),
			"planner_adv_mean": gate_stats.get("planner_adv_mean", 0.0),
			"planner_qstd_mean": gate_stats.get("planner_qstd_mean", 0.0),
			"planner_std_mean": gate_stats.get("planner_std_mean", 0.0),

			"si/active":       float(self._last_si_stats.get("si/active", 0.0)),
			"si/loss":         float(self._last_si_stats.get("si/loss", 0.0)),
			"si/weight_mean":  float(self._last_si_stats.get("si/weight_mean", 0.0)),
			"si/weight_max":   float(self._last_si_stats.get("si/weight_max", 0.0)),
			"si/G_med":        float(self._last_si_stats.get("si/G_med", 0.0)),
			"si/G_std":        float(self._last_si_stats.get("si/G_std", 0.0)),
			"si/pool_size":    float(len(self._ep_return_stats)),
		}


