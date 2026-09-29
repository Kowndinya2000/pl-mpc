"""Short CPU checks of the simulator, replay, model update and checkpoint round-trip."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_module
from tdmpc_square.common.parser import parse_cfg
from tdmpc_square.common.seed import set_seed
from tdmpc_square.common.buffer import Buffer
from tdmpc_square.envs import make_env
from tdmpc_square.tdmpc_square import TDMPC2
from tdmpc_square.trainer.online_trainer import OnlineTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='humanoid_h1hand-balance_hard-v0')
    parser.add_argument('--checkpoint-out', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with initialize_config_module(version_base='1.3', config_module='tdmpc_square'):
        cfg = compose(config_name='config', overrides=[f'task={args.task}', 'device=cpu',
            'num_samples=16', 'num_elites=4', 'num_pi_trajs=2', 'iterations=1',
            'batch_size=2', 'steps=128', 'buffer_size=128', 'save_video=false'])
    # parse_cfg expects Hydra's launch context; smoke uses the same resolved fields.
    from unittest.mock import patch
    with patch('hydra.utils.get_original_cwd', return_value=str(Path.cwd())):
        cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    env = make_env(cfg)
    try:
        obs, _ = env.reset()
        assert torch.isfinite(obs).all()
        agent = TDMPC2(cfg)
        trainer = object.__new__(OnlineTrainer)
        trainer.env = env
        episode = [trainer.to_td(obs)]
        for step in range(8):
            action, mu, std = agent.act(obs, t0=step == 0, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            assert torch.isfinite(obs).all() and torch.isfinite(reward)
            episode.append(trainer.to_td(obs, action, mu, std, reward))
            if terminated or truncated:
                break
        assert len(episode) > cfg.horizon + 1, 'Smoke episode ended before a replay slice was available'
        buffer = Buffer(cfg)
        buffer.add(torch.cat(episode))
        agent._step = cfg.si_warmup_steps
        metrics = agent.update(buffer)
        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics['si/active'] == 1.0
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = args.checkpoint_out or Path(tmp) / 'smoke.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            agent.save(checkpoint)
            restored = TDMPC2(cfg)
            restored.load(checkpoint)
            for a, b in zip(agent.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            action, _, _ = restored.act(obs, t0=True, eval_mode=True)
            assert torch.isfinite(action).all()
        print(f'PASS {args.task}: reset, {len(episode)-1} steps, replay, model/RAD update, checkpoint round-trip')
    finally:
        env.close()


if __name__ == '__main__':
    main()
