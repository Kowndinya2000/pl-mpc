from copy import deepcopy
import warnings

import gymnasium as gym

from tdmpc_square.envs.wrappers.multitask import MultitaskWrapper
from tdmpc_square.envs.wrappers.pixels import PixelWrapper
from tdmpc_square.envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(
        f"Missing dependencies for task {task}; install dependencies to use this environment."
    )


warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print("Creating multi-task environment with tasks:", cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)
    if cfg.multitask:
        env = make_multitask_env(cfg)

    else:
        if cfg.task == "ri_insert":
            from rhythmic_insertion.env import make_env as factory
        elif cfg.task.startswith("humanoid_"):
            from tdmpc_square.envs.humanoid import make_env as factory
        else:
            from tdmpc_square.envs.dmcontrol import make_env as factory
        env = factory(cfg)
        env = TensorWrapper(env)
    if cfg.get("obs", "state") == "rgb":
        env = PixelWrapper(cfg, env)
    try:
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:
        cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
