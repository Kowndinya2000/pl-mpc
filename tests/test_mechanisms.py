"""Behavioral checks of hybrid targets, terminal scoring and return distillation."""
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tdmpc_square.common.buffer import Buffer
from tdmpc_square.tdmpc_square import TDMPC2


torch.set_num_threads(1)


class ScalarModel:
    def __init__(self):
        self.reward_calls = 0
        self.q_queries = []

    def pi(self, z, task):
        return None, torch.zeros_like(z), None, None

    def reward(self, z, action, task):
        self.reward_calls += 1
        return torch.full_like(z[..., :1], 7.)

    def next(self, z, action, task):
        return z + 1

    def Q(self, z, action, task, return_type, target=False):
        self.q_queries.append((target, return_type))
        if return_type == 'all':
            return torch.stack([torch.full_like(z[..., :1], x) for x in [1., 3., 9.]])
        return torch.full_like(z[..., :1], 10.)


def scalar_agent():
    a = object.__new__(TDMPC2)
    a.cfg = SimpleNamespace(multitask=False, actor_conservatism=False, multistep_td=True,
        multistep_td_n=3, mtd_mode='full', num_bins=0, q_std_min=1e-6, q_std_max=1e6,
        actor_bottomk=2, beta_uncertainty_max=0., lambda_uncertainty_max=.5,
        lambda_uncertainty_floor=.5, uncertainty_ema_tau=.01)
    a.discount = .9
    a.model = ScalarModel()
    a.u_mean, a.u_std = 0., 1.
    return a


def test_hybrid_mtd_uses_three_two_one_observed_rewards_and_detaches():
    a = scalar_agent()
    rewards = torch.ones(3, 2, 1, requires_grad=True)
    latents = torch.zeros_like(rewards, requires_grad=True)
    result = a._td_target(latents, rewards, None)
    # Reward 7 comes from the model only after the observed replay slice ends.
    expected = torch.tensor([1+.9+.81+.729*10, 1+.9+.81*7+.729*10,
                             1+.9*7+.81*7+.729*10])
    torch.testing.assert_close(result[:, 0, 0], expected)
    assert a.model.reward_calls == 3
    assert not result.requires_grad
    assert all(target for target, _ in a.model.q_queries)


@pytest.mark.parametrize('mode,expected,model_calls', [
    ('buffer_only', [10., 10., 10.], 0),
    ('imagination_only', [26.26, 26.26, 26.26], 9),
])
def test_mtd_ablation_paths(mode, expected, model_calls):
    a = scalar_agent(); a.cfg.mtd_mode = mode
    result = a._td_target(torch.zeros(3, 1, 1), torch.ones(3, 1, 1), None, z=torch.zeros(3, 1, 1))
    torch.testing.assert_close(result.flatten(), torch.tensor(expected))
    assert a.model.reward_calls == model_calls


def test_ate_terminal_scoring_and_td_target_independence():
    a = scalar_agent()
    score = a._estimate_value_adaptive(torch.zeros(2, 1), torch.zeros(3, 2, 1), None, 3)
    heads = torch.tensor([1., 3., 9.])
    expected = 7*(1+.9+.81) + .729*(heads.mean() - .5*heads.std())
    torch.testing.assert_close(score, expected.expand(2, 1))
    assert a.model.q_queries and all(not target for target, _ in a.model.q_queries)
    before = a._td_target(torch.zeros(3, 1, 1), torch.ones(3, 1, 1), None)
    a.cfg.lambda_uncertainty_max = 0.
    a.cfg.lambda_uncertainty_floor = 0.
    after = a._td_target(torch.zeros(3, 1, 1), torch.ones(3, 1, 1), None)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_ate_sigmoid_and_ema():
    a = scalar_agent()
    assert a._adaptive_coef(torch.tensor(0.), .5).item() == pytest.approx(.25)
    assert a._adaptive_coef(torch.tensor(3.), .5) > a._adaptive_coef(torch.tensor(-3.), .5)
    a._update_uncertainty_stats(torch.tensor([2., 4.]))
    assert a.u_mean == pytest.approx(.03)
    assert a.u_std == pytest.approx(1.)


def tiny_agent():
    import tdmpc_square
    cfg = OmegaConf.load(Path(tdmpc_square.__file__).with_name('config.yaml'))
    cfg.merge_with(dict(device='cpu', obs_shape={'state': [8]}, action_dim=4,
        episode_length=1000, multitask=False, task_dim=0, batch_size=4,
        latent_dim=32, mlp_dim=32, enc_dim=32, num_q=3, bin_size=.2))
    return TDMPC2(cfg)


def test_rad_queue_is_finite_sorted_per_batch_and_uses_population_sd():
    a = tiny_agent(); a._ep_return_stats = deque(maxlen=4)
    a._update_si_stats(torch.tensor([9., 1., 1., 5., float('nan')]))
    assert list(a._ep_return_stats) == [1., 5., 9.]
    a._update_si_stats(torch.tensor([9., 3.]))
    assert list(a._ep_return_stats) == [5., 9., 3., 9.]
    median, sd = a._si_running_stats()
    assert median == 5.
    assert sd == pytest.approx(torch.tensor([5., 9., 3., 9.]).std(unbiased=False).item())


def test_rad_warmup_adds_weighted_loss_without_replacing_backbone():
    torch.manual_seed(5)
    a = tiny_agent()
    a._update_si_stats(torch.tensor([0., 50., 100.]))
    zs = torch.randn(3, 4, 32); actions = torch.tanh(torch.randn(3, 4, 4))
    mu, std = torch.zeros_like(actions), torch.ones_like(actions)
    returns = torch.tensor([0., 50., 100., 1000.], requires_grad=True)
    off, before, active = deepcopy(a), deepcopy(a), deepcopy(a)
    off.cfg.self_imitation = False
    before._step, active._step = 199999, 200000
    def update(agent):
        torch.manual_seed(17)
        return agent.update_pi(zs, actions, mu, std, None, returns)
    off_losses, before_losses = update(off), update(before)
    assert before._last_si_stats['si/active'] == 0
    assert off_losses[:3] == before_losses[:3]
    for x, y in zip(off.model._pi.parameters(), before.model._pi.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    median, sd = active._si_running_stats()
    weights = torch.exp((returns.detach()-median)/sd).clamp(0, 10)
    expected_rad = .5*(-weights[None, :]*active.model.log_pi_action(zs, actions, None).squeeze(-1)).mean()
    active_losses = update(active)
    assert active._last_si_stats['si/active'] == 1
    assert active._last_si_stats['si/weight_max'] == 10
    assert active._last_si_stats['si/loss'] == pytest.approx(expected_rad.item(), rel=1e-6)
    assert active_losses[0] - before_losses[0] == pytest.approx(expected_rad.item(), abs=1e-4)
    assert active_losses[1:3] == before_losses[1:3]
    assert returns.grad is None


def test_replay_return_annotation_and_episode_boundaries():
    cfg = SimpleNamespace(device='cpu', buffer_size=128, steps=128, batch_size=8, horizon=3)
    buffer = Buffer(cfg)
    for value in [1., 10.]:
        reward = torch.full((12,), value); reward[0] = float('nan')
        td = TensorDict(dict(obs=torch.full((12, 4), value), action=torch.zeros(12, 2),
                             mu=torch.zeros(12, 2), std=torch.ones(12, 2), reward=reward), [12])
        buffer.add(td)
        assert torch.all(td['episode_return'] == 11*value)
    obs, actions, mu, std, reward, task, returns = buffer.sample()
    assert obs.shape == (4, 8, 4) and reward.shape == (3, 8, 1)
    assert torch.all(obs == obs[0:1])
    assert torch.allclose(returns, obs[0, :, 0]*11)
    assert torch.isfinite(reward).all()


def test_actor_q_returns_detached_values():
    a = tiny_agent()
    z = torch.randn(4, 32)
    action = torch.randn(4, 4, requires_grad=True)
    assert not a._actor_q(z, action, None).requires_grad
