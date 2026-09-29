from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


def _resolve_success_bar(env):
    """Walk down the gym wrapper chain and find a `task.success_bar` attribute.

    humanoid-bench envs declare `success_bar` as a class attribute on the task
    object (e.g. Hurdle inherits Walk.success_bar=700) but never populate
    info['success'] in step(). We resolve the bar post-hoc and evaluate success
    from cumulative reward at episode end.
    """
    node = env
    for _ in range(8):
        task = getattr(getattr(node, "unwrapped", node), "task", None)
        if task is not None:
            bar = getattr(task, "success_bar", None)
            if isinstance(bar, (int, float)) and not isinstance(bar, bool):
                return float(bar)
        node = getattr(node, "env", None)
        if node is None:
            return None
    return None


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.

    Also populates `info['success']` for envs that declare `task.success_bar`
    but don't emit a success signal per step (e.g. humanoid-bench locomotion
    tasks: hurdle, stair, balance_hard, etc.). Success is computed post-hoc as
    `cumulative_episode_reward >= success_bar` on episode termination/truncation.
    """

    def __init__(self, env):
        super().__init__(env)
        self._success_bar = _resolve_success_bar(env)
        self._ep_cum_reward = 0.0
        self._ep_success_raised = False

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        x = torch.from_numpy(x)
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, task_idx=None):
        obs, info = self.env.reset()

        self._ep_cum_reward = 0.0
        self._ep_success_raised = False
        return self._obs_to_tensor(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.numpy())
        info = defaultdict(float, info)


        self._ep_cum_reward += float(reward)

        env_success = bool(info.get("success", 0))
        if (
            (done or truncated)
            and not env_success
            and not self._ep_success_raised
            and self._success_bar is not None
            and self._ep_cum_reward >= self._success_bar
        ):
            info["success"] = 1.0
            self._ep_success_raised = True
        else:
            info["success"] = float(info["success"])

        return (
            self._obs_to_tensor(obs),
            torch.tensor(reward, dtype=torch.float32),
            done,
            truncated,
            info,
        )
