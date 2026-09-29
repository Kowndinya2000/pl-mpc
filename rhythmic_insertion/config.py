"""Relocatable simulator configuration and reset assets."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from rhythmic_insertion.sim import register_resolvers

ROOT = Path(__file__).resolve().parent


def reset_state_path(cfg):
    size = int(cfg.get("ri_asset_size", 5))
    if size not in (1, 3, 5):
        raise ValueError("ri_asset_size must be 1, 3 or 5.")
    custom = cfg.get("ri_reset_states")
    path = Path(custom).expanduser().resolve() if custom else ROOT / "assets/reset_states" / f"size{size}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Reset states not found: {path}")
    with np.load(path, allow_pickle=False) as bank:
        expected = [f"nut{size}_bolt{size}_wrench{size}"]
        if bank["subassemblies"].tolist() != expected:
            raise ValueError("Reset states do not match ri_asset_size.")
    return path


@contextmanager
def compose_sim_config(cfg, state_path=None, camera=False):
    """Keep the caller's Hydra state intact while constructing the simulator."""
    register_resolvers()
    global_hydra = GlobalHydra.instance()
    previous = global_hydra.hydra
    global_hydra.clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "sim/cfg")):
            size = int(cfg.get("ri_asset_size", 5))
            device = str(cfg.get("device", "cuda:0"))
            sim_cfg = compose(config_name="config", overrides=[
                "task.env.numEnvs=1",
                f"++task.env.camera_resolution={int(cfg.get('ri_camera_resolution', 256))}", f"++task.env.enableCameraSensors={str(camera).lower()}",
                f"++task.env.desired_subassemblies=[nut{size}_bolt{size}_wrench{size}]",
                f"sim_device={device}", f"rl_device={device}",
                f"seed={int(cfg.get('seed', 1))}",
            ])
            sim_cfg.task.env.cache_states_path = str(state_path or reset_state_path(cfg))
            sim_cfg.task.rl.observation_mode = "full"
            sim_cfg.task.rl.reward_type = "sdf"
            sim_cfg.task.rl.max_episode_length = int(cfg.get("ri_episode_length", 256))
            if sim_cfg.task.rl.max_episode_length < 2:
                raise ValueError("ri_episode_length must be at least 2.")
            yield sim_cfg
    finally:
        GlobalHydra.instance().hydra = previous
