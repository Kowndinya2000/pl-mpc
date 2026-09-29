"""Check supported recipes and isolation of the baseline entrypoint."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_module
from tdmpc_square.common.parser import parse_cfg


def configuration(overrides):
    with initialize_config_module(version_base='1.3', config_module='tdmpc_square'):
        cfg = compose(config_name='config', overrides=overrides)
    with patch('hydra.utils.get_original_cwd', return_value='/tmp'):
        return parse_cfg(cfg)


@pytest.mark.parametrize('overrides', [[], ['multistep_td=false'], ['self_imitation=false'],
    ['lambda_uncertainty_max=0', 'lambda_uncertainty_floor=0', 'beta_uncertainty_max=0'],
    ['si_warmup_steps=750000'], ['mtd_mode=buffer_only'], ['mtd_mode=imagination_only']])
def test_supported_recipes(overrides):
    cfg = configuration(overrides)
    assert cfg.horizon == 3 and cfg.disable_wandb


@pytest.mark.parametrize('override', ['planner_type=fixedA', 'actor_mode=unknown',
    'mtd_mode=unknown', 'actor_q_mode=bottomk', 'td_target_q_mode=ucb'])
def test_unsupported_modes_fail_early(override):
    with pytest.raises(ValueError, match='unsupported'):
        configuration([override])


def test_baseline_launcher_selects_separate_configuration():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/train_baseline.py'), '--cfg', 'job'],
                            cwd=root, text=True, capture_output=True, check=True)
    assert 'self_imitation:' not in result.stdout
    assert 'actor_conservatism:' not in result.stdout
    assert 'prior_coef:' in result.stdout
