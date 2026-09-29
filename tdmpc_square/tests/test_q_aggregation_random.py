"""Tests for random-subsample Q-aggregation modes matching vanilla TDMPC².

`random_min` and `random_avg` use `torch.randperm` to pick 2 of N heads on each call
and then min/mean. This matches upstream tdmpc2/common/world_model.py Q() behavior.

Run with: pytest tdmpc_square/tdmpc_square/tests/test_q_aggregation_random.py -v
"""
from __future__ import annotations

import pytest
import torch

from tdmpc_square.tdmpc_square import TDMPC2


def _agg(mode, q_mean=None, q_std=None, q_min=None, q_bottomk=None, q_all=None,
         coef=None, ucb_coef=None, coef_max=None):
    """Helper: call _aggregate_q as an unbound static-style method."""

    if q_all is not None and q_mean is None:
        q_mean = q_all.mean(0)
    if q_all is not None and q_std is None:
        q_std = q_all.std(0)
    if q_all is not None and q_min is None:
        q_min = q_all.min(0).values
    if q_all is not None and q_bottomk is None:
        qs_sorted, _ = torch.sort(q_all, dim=0)
        q_bottomk = qs_sorted[:2].mean(0)

    return TDMPC2._aggregate_q(
        None,
        q_mean=q_mean, q_std=q_std, q_min=q_min, q_bottomk=q_bottomk,
        mode=mode, coef=coef, ucb_coef=ucb_coef, coef_max=coef_max,
        q_all=q_all,
    )

def test_random_min_picks_2_and_takes_min():
    """Feed known ensemble; verify result is min of some 2-element subset."""
    torch.manual_seed(0)
    q_all = torch.tensor([
        [10.0, 10.0],
        [20.0, 20.0],
        [30.0, 30.0],
        [40.0, 40.0],
        [50.0, 50.0],
    ])
    result = _agg("random_min", q_all=q_all)

    assert result.shape == (2,)
    for v in result.tolist():
        assert v in {10.0, 20.0, 30.0, 40.0}, f"got {v}, expected min of 2 of 5 heads"


def test_random_min_is_stochastic():
    """Same input, multiple calls → different selections → variance > 0."""
    torch.manual_seed(0)
    q_all = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])

    outputs = set()
    for _ in range(100):
        r = _agg("random_min", q_all=q_all)
        outputs.add(r.item())


    assert len(outputs) >= 3, f"expected stochastic output, got {outputs}"


def test_random_min_handles_3_heads():
    """Edge case: num_q=3. randperm[:2] picks 2 of 3."""
    torch.manual_seed(42)
    q_all = torch.tensor([[100.0], [200.0], [300.0]])
    r = _agg("random_min", q_all=q_all)
    assert r.item() in {100.0, 200.0}, f"min of 2 of 3 (values 100, 200, 300); got {r.item()}"


def test_random_min_requires_q_all():
    """Asserts if q_all is None."""
    with pytest.raises(AssertionError, match="random_min requires q_all"):
        _agg("random_min", q_mean=torch.tensor([1.0]), q_std=torch.tensor([0.1]),
             q_min=torch.tensor([0.5]), q_bottomk=torch.tensor([0.6]))

def test_random_avg_picks_2_and_takes_mean():
    """Feed known ensemble; verify result is mean of some 2-element subset."""
    torch.manual_seed(0)
    q_all = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])
    r = _agg("random_avg", q_all=q_all)

    valid = {15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0}
    assert r.item() in valid, f"got {r.item()}, expected mean of 2 of 5"


def test_random_avg_is_stochastic():
    torch.manual_seed(0)
    q_all = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    outputs = set()
    for _ in range(100):
        r = _agg("random_avg", q_all=q_all)
        outputs.add(r.item())

    assert len(outputs) >= 3, f"expected stochastic output, got {outputs}"


def test_random_avg_requires_q_all():
    with pytest.raises(AssertionError, match="random_avg requires q_all"):
        _agg("random_avg", q_mean=torch.tensor([1.0]), q_std=torch.tensor([0.1]),
             q_min=torch.tensor([0.5]), q_bottomk=torch.tensor([0.6]))

def test_matches_upstream_implementation():
    """Verify our random_min/random_avg produce the same pattern as upstream:
    tdmpc2/common/world_model.py:
        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2
    """
    torch.manual_seed(7)
    q_all = torch.randn(5, 64)


    rng_state = torch.random.get_rng_state()
    our_min = _agg("random_min", q_all=q_all)
    torch.random.set_rng_state(rng_state)
    qidx = torch.randperm(q_all.shape[0])[:2]
    expected_min = q_all[qidx].min(0).values
    assert torch.allclose(our_min, expected_min), "random_min must match upstream formula"

    torch.random.set_rng_state(rng_state)
    our_avg = _agg("random_avg", q_all=q_all)
    torch.random.set_rng_state(rng_state)
    qidx = torch.randperm(q_all.shape[0])[:2]
    expected_avg = q_all[qidx].mean(0)
    assert torch.allclose(our_avg, expected_avg), "random_avg must match upstream formula"


def test_avg_still_works_without_q_all():
    r = _agg("avg", q_mean=torch.tensor([100.0]), q_std=torch.tensor([10.0]),
             q_min=torch.tensor([80.0]), q_bottomk=torch.tensor([85.0]))
    assert r.item() == 100.0


