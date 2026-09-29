"""Check local logging and optional W&B metadata without network access."""
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['tdmpc_square', 'baselines/tdmpc_square_original/tdmpc_square'])
def logging_stack(request, tmp_path):
    package = ROOT / request.param
    spec = importlib.util.spec_from_file_location('tested_logger', package / 'common/logger.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = OmegaConf.load(package / 'config.yaml')
    cfg.work_dir = tmp_path
    cfg.task_title = 'Dog Stand'
    cfg.task = 'dog-stand'
    cfg.obs_shape = {'state': [8]}
    cfg.action_dim = 4
    cfg.exp_name = 'trial_one'
    return module, cfg


def test_local_logging_needs_no_wandb_import(logging_stack, monkeypatch):
    module, cfg = logging_stack
    original_import = builtins.__import__

    def no_wandb(name, *args, **kwargs):
        assert name != 'wandb', 'Local logging must not import or initialize W&B'
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_wandb)
    logger = module.Logger(cfg)
    logger.log({'step': 0, 'episode_reward': np.float32(1.25)}, 'eval')
    logger.log({'step': 50, 'episode_reward': np.float32(2.5)}, 'eval')
    rows = [json.loads(line) for line in (cfg.work_dir / 'eval.jsonl').read_text().splitlines()]
    assert [row['episode_reward'] for row in rows] == [1.25, 2.5]
    assert pd.read_csv(cfg.work_dir / 'eval.csv')['step'].tolist() == [0, 50]
    assert (cfg.work_dir / 'config.yaml').exists()
    logger.finish()


def test_enabled_default_entity_and_method_metadata(logging_stack, monkeypatch):
    module, cfg = logging_stack
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    cfg.disable_wandb = False
    logger = module.Logger(cfg)
    metadata = fake.init.call_args.kwargs
    assert metadata['entity'] is None
    assert cfg.wandb_method in metadata['name']
    assert metadata['name'].endswith(f'.trial_one.seed-{cfg.seed}')
    assert metadata['group'] == f'{cfg.wandb_method}-dog-stand-trial-one'
    assert cfg.wandb_method in metadata['tags']
    assert f'seed:{cfg.seed}' in metadata['tags']
    group = module.cfg_to_group(cfg)
    cfg.seed += 1
    assert module.cfg_to_group(cfg) == group
    logger.log({'step': 50, 'episode_reward': 2.5, 'episode_success': 1.0}, 'eval')
    fake.log.assert_called_once_with(
        {'eval/step': 50, 'eval/episode_reward': 2.5, 'eval/episode_success': 1.0}, step=50)
    logger.log({'step': 50, 'return': 2.5}, 'results')
    fake.log.assert_called_with({'results/return': 2.5}, step=50)
    logger.finish()
    fake.finish.assert_called_once()


def test_custom_offline_metadata(logging_stack, monkeypatch):
    module, cfg = logging_stack
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    cfg.merge_with(dict(disable_wandb=False, wandb_entity='test-team',
        wandb_project='test-project', wandb_run_name='custom-run',
        wandb_group='custom-group', wandb_tags=['ablation'], wandb_mode='offline'))
    module.Logger(cfg).finish()
    metadata = fake.init.call_args.kwargs
    assert metadata['mode'] == 'offline'
    assert metadata['entity'] == 'test-team'
    assert metadata['project'] == 'test-project'
    assert metadata['name'] == 'custom-run'
    assert metadata['group'] == 'custom-group'
    assert metadata['tags'][-1] == 'ablation'


def test_model_upload_can_be_disabled_without_losing_local_checkpoint(logging_stack, monkeypatch):
    module, cfg = logging_stack
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    cfg.disable_wandb = False
    cfg.wandb_save_model = False
    agent = MagicMock()
    agent.save.side_effect = lambda path: path.write_bytes(b'test-checkpoint')
    logger = module.Logger(cfg)
    logger.save_agent(agent)
    assert (logger.model_dir / 'final.pt').read_bytes() == b'test-checkpoint'
    fake.log_artifact.assert_not_called()
    cfg.wandb_save_model = True
    logger = module.Logger(cfg)
    logger.save_agent(agent)
    fake.log_artifact.assert_called_once()


@pytest.mark.parametrize('key,value,error', [
    ('wandb_project', None, 'wandb_project'),
    ('wandb_mode', 'invalid', 'wandb_mode'),
])
def test_invalid_enabled_logging_configuration_fails_clearly(logging_stack, key, value, error):
    module, cfg = logging_stack
    cfg.disable_wandb = False
    cfg[key] = value
    with pytest.raises(ValueError, match=error):
        module.Logger(cfg)


def test_missing_optional_sdk_explains_installation(logging_stack, monkeypatch):
    module, cfg = logging_stack
    cfg.disable_wandb = False
    monkeypatch.setitem(sys.modules, 'wandb', None)
    with pytest.raises(ModuleNotFoundError, match='requirements/wandb.txt'):
        module.Logger(cfg)
