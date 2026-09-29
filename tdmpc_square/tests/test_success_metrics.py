"""Check cumulative-return success fallback and explicit environment success."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from tdmpc_square.envs.wrappers.tensor import TensorWrapper, _resolve_success_bar


def _make_mock_env(success_bar=700):
    """Build a mock gym env with a task.success_bar attribute and a configurable
    step()/reset() that returns numeric reward + done flag we control externally."""
    env = MagicMock()
    env.unwrapped = env
    env.task = SimpleNamespace(success_bar=success_bar)


    obs_space = MagicMock()
    obs_space.shape = (4,)
    action_space = MagicMock()
    action_space.sample = lambda: np.zeros(3, dtype=np.float32)
    action_space.shape = (3,)
    env.observation_space = obs_space
    env.action_space = action_space


    env.reset.return_value = (np.zeros(4, dtype=np.float32), {})

    return env


def test_resolve_success_bar_basic():
    env = _make_mock_env(success_bar=800)
    assert _resolve_success_bar(env) == 800.0


def test_resolve_success_bar_missing():
    env = MagicMock()
    env.unwrapped = env

    del env.task

    env.env = None

    env.mock_add_spec(["unwrapped", "env"])
    env.unwrapped = env
    env.env = None
    assert _resolve_success_bar(env) is None


def test_success_not_raised_when_below_bar():
    env = _make_mock_env(success_bar=700)

    step_count = [0]
    def step(a):
        step_count[0] += 1
        done = step_count[0] >= 1000
        return (np.zeros(4, dtype=np.float32), 0.5, done, False, {})
    env.step = step

    w = TensorWrapper(env)
    w.reset()
    for _ in range(1000):
        _, _, done, _, info = w.step(torch.zeros(3))
        if done:
            break
    assert info["success"] == 0.0, f"expected 0, got {info['success']}"


def test_success_raised_when_above_bar():
    env = _make_mock_env(success_bar=700)

    step_count = [0]
    def step(a):
        step_count[0] += 1
        done = step_count[0] >= 1000
        return (np.zeros(4, dtype=np.float32), 0.8, done, False, {})
    env.step = step

    w = TensorWrapper(env)
    w.reset()
    for _ in range(1000):
        _, _, done, _, info = w.step(torch.zeros(3))
        if done:
            break
    assert info["success"] == 1.0, f"expected 1, got {info['success']}"


def test_success_raised_on_truncation_not_just_termination():
    """Episode reaching max_steps (truncation) should still get success=1 if
    cumulative reward crosses the bar."""
    env = _make_mock_env(success_bar=700)
    step_count = [0]
    def step(a):
        step_count[0] += 1
        truncated = step_count[0] >= 1000
        return (np.zeros(4, dtype=np.float32), 0.8, False, truncated, {})
    env.step = step

    w = TensorWrapper(env)
    w.reset()
    for _ in range(1000):
        _, _, done, truncated, info = w.step(torch.zeros(3))
        if done or truncated:
            break
    assert info["success"] == 1.0


def test_success_zero_for_early_termination_below_bar():
    """Fall terminating the episode before reaching bar → success=0."""
    env = _make_mock_env(success_bar=700)
    step_count = [0]
    def step(a):
        step_count[0] += 1

        done = step_count[0] >= 100
        return (np.zeros(4, dtype=np.float32), 0.8, done, False, {})
    env.step = step

    w = TensorWrapper(env)
    w.reset()
    for _ in range(200):
        _, _, done, _, info = w.step(torch.zeros(3))
        if done:
            break
    assert info["success"] == 0.0, f"expected 0, got {info['success']}"


def test_success_resets_between_episodes():
    """First ep crosses bar (success=1); second ep stays below — must be 0."""
    env = _make_mock_env(success_bar=700)


    step_count = [0]
    ep_idx = [0]
    rewards_per_ep = [0.8, 0.1]

    def step(a):
        step_count[0] += 1
        done = step_count[0] >= 1000
        return (np.zeros(4, dtype=np.float32), rewards_per_ep[ep_idx[0]], done, False, {})

    env.step = step
    w = TensorWrapper(env)


    w.reset()
    step_count[0] = 0
    ep_idx[0] = 0
    for _ in range(1000):
        _, _, done, _, info = w.step(torch.zeros(3))
        if done:
            break
    assert info["success"] == 1.0


    w.reset()
    step_count[0] = 0
    ep_idx[0] = 1
    for _ in range(1000):
        _, _, done, _, info = w.step(torch.zeros(3))
        if done:
            break
    assert info["success"] == 0.0


def test_env_provided_success_is_respected():
    """If env explicitly sets info['success']=1 (e.g. reach, push), the wrapper
    must NOT override with its bar-threshold logic."""
    env = _make_mock_env(success_bar=700)
    step_count = [0]
    def step(a):
        step_count[0] += 1
        done = step_count[0] >= 100
        info = {"success": 1} if done else {}
        return (np.zeros(4, dtype=np.float32), 0.1, done, False, info)
    env.step = step

    w = TensorWrapper(env)
    w.reset()
    for _ in range(200):
        _, _, done, _, info = w.step(torch.zeros(3))
        if done:
            break
    assert info["success"] == 1.0, f"expected 1 (env-set), got {info['success']}"


def test_no_success_bar_leaves_info_zero():
    """Task with no success_bar attr → wrapper leaves info['success']=0."""
    env = _make_mock_env(success_bar=None)

    env.task = SimpleNamespace()


    env.env = None

    step_count = [0]
    def step(a):
        step_count[0] += 1
        done = step_count[0] >= 100
        return (np.zeros(4, dtype=np.float32), 999.0, done, False, {})
    env.step = step

    w = TensorWrapper(env)
    w.reset()
    for _ in range(200):
        _, _, done, _, info = w.step(torch.zeros(3))
        if done:
            break

    assert info["success"] == 0.0
