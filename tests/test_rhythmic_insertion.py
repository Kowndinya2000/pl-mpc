"""CPU checks for insertion configuration, assets and reward."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rhythmic_insertion.config import ROOT, compose_sim_config, reset_state_path
from rhythmic_insertion.reward import insertion_reward


@pytest.mark.parametrize('size', [1, 3, 5])
def test_reset_assets_and_sim_defaults(size):
    cfg = {'ri_asset_size': size}
    with np.load(reset_state_path(cfg), allow_pickle=False) as bank:
        assert bank['dof_pos'].shape == (512, 18)
        assert bank['plug_root'].shape == (512, 13)
        assert np.isfinite(bank['plug_root']).all()
    with compose_sim_config(cfg) as sim:
        assert sim.task.env.desired_subassemblies == [f'nut{size}_bolt{size}_wrench{size}']
        assert sim.task.env.numEnvs == 1
        assert sim.task.env.numObservations == 17
        assert sim.task.rl.pos_action_scale == [0.01] * 3
        assert sim.task.rl.max_episode_length == 256


def test_hydra_context_survives_success_and_failure():
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'sim/cfg')):
        original = GlobalHydra.instance().hydra
        with compose_sim_config({}):
            assert GlobalHydra.instance().hydra is not original
        assert GlobalHydra.instance().hydra is original
        with pytest.raises(ValueError):
            with compose_sim_config({'ri_episode_length': 0}):
                pass
        assert GlobalHydra.instance().hydra is original


def test_missing_or_mismatched_reset_states_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        reset_state_path({'ri_reset_states': str(tmp_path / 'absent.npz')})
    with pytest.raises(ValueError, match='match'):
        reset_state_path({'ri_asset_size': 1, 'ri_reset_states': str(reset_state_path({}))})
    with pytest.raises(ValueError, match='must be'):
        reset_state_path({'ri_asset_size': 2})


def test_reward_thresholds_and_bounds():
    nut = np.array([0., 0., 0.5])
    reward, info = insertion_reward(nut, nut)
    assert reward == pytest.approx(0.5 * np.exp(-0.005**2 / (2 * 0.03**2)) + 0.5)
    assert info['rew_engaged'] == info['rew_inserted'] == 1
    reward, info = insertion_reward(nut + [0.06, 0., 0.], nut)
    assert info['rew_engaged'] == info['rew_inserted'] == 0
    assert 0 <= reward <= 0.5
    _, info = insertion_reward(nut + [0., 0., 0.008], nut)
    assert info['rew_engaged'] == 1 and info['rew_inserted'] == 0
    assert insertion_reward(nut + [0., 0., 1.], nut)[0] < 1e-6


def test_all_urdf_meshes_are_local_and_present():
    root = (ROOT / 'assets').resolve()
    for urdf in root.rglob('*.urdf'):
        for mesh in ET.parse(urdf).iter('mesh'):
            path = (urdf.parent / mesh.attrib['filename']).resolve()
            assert root in path.parents and path.is_file()
