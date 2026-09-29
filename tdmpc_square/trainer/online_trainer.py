from time import time

import math
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from tdmpc_square.trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def _reset_with_seed(self, seed=None):
        """Reset env with optional seed, robust to gym/gymnasium signatures."""
        if seed is None:
            out = self.env.reset()
        else:
            try:
                out = self.env.reset(seed=seed)
            except TypeError:
                out = self.env.reset()

        if isinstance(out, tuple):
            obs = out[0]
        else:
            obs = out
        return obs

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        """Additionally covers planner and optional actor policy."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=True)
            while not done:
                action, _, _ = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                ep_succ = float(info.get("success", 0.0))
                tag = "succ" if ep_succ >= 1.0 else "fail"
                self.logger.video.save(
                    self._step, key=f"results/video/ep{i:02d}_{tag}"
                )

        if self.cfg.eval_pi:

            ep_rewards_pi, ep_successes_pi = [], []
            for i in range(self.cfg.eval_episodes):

                obs, done, ep_reward, t = self._reset_with_seed(seed=i), False, 0, 0
                while not done:
                    action, _, _ = self.agent.act(
                        obs, t0=t == 0, eval_mode=True, use_pi=True
                    )
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards_pi.append(ep_reward)
                ep_successes_pi.append(info["success"])

        out = dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_reward_max=float(np.nanmax(ep_rewards)),
            episode_reward_min=float(np.nanmin(ep_rewards)),
            episode_reward_std=float(np.nanstd(ep_rewards)),
            episode_success=np.nanmean(ep_successes),
            episode_success_max=float(np.nanmax(ep_successes)),
        )

        if self.cfg.eval_pi:
            out["episode_reward_pi"] = np.nanmean(ep_rewards_pi)
            out["episode_success_pi"] = np.nanmean(ep_successes_pi)
        return out

    def to_td(self, obs, action=None, mu=None, std=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if mu is None:
            mu = torch.full_like(action, float("nan"))
        if std is None:
            std = torch.full_like(action, float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                mu=mu.unsqueeze(0),
                std=std.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True

        while self._step <= self.cfg.steps:

            if self._step % self.cfg.eval_freq == 0:
                eval_next = True


            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                    if self.cfg.save_agent:
                        try:
                            self.logger.save_agent(
                                self.agent,
                                identifier=f"step_{self._step}",
                                to_wandb=False,
                            )
                        except Exception as _e:
                            print(f"[checkpoint] step save failed: {_e}", flush=True)
                        er_mean = float(eval_metrics.get('episode_reward', float('-inf')))
                        if not hasattr(self, '_best_eval_R') or er_mean > self._best_eval_R:
                            self._best_eval_R = er_mean
                            try:
                                self.logger.save_agent(
                                    self.agent, identifier="best", to_wandb=True
                                )
                            except Exception as _e:
                                print(f"[checkpoint] best save failed: {_e}", flush=True)

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())

                    results_metrics = {'return': train_metrics['episode_reward'],
                                       'episode_length': len(self._tds[1:]),
                                       'success': train_metrics['episode_success'],
                                       'success_subtasks': info['success_subtasks'],
                                       'step': self._step,}

                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    full_ep_td = torch.cat(self._tds)
                    self._ep_idx = self.buffer.add(full_ep_td)

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]


            if self._step > self.cfg.seed_steps:
                t0 = len(self._tds) == 1
                action, mu, std = self.agent.act(obs, t0=t0)
            else:
                action = self.env.rand_act()
                mu, std = action.detach().clone(), torch.full_like(action, math.exp(self.cfg.log_std_max))
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            self._tds.append(self.to_td(obs, action, mu, std, reward))


            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    self.agent._step = self._step
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1

        self.logger.finish(self.agent)
