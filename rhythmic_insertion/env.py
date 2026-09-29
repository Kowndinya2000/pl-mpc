"""Gymnasium interface for the rhythmic wrench–nut insertion task."""
import isaacgym  # Isaac Gym must be imported before PyTorch.
from isaacgym import gymapi
import gymnasium as gym
import numpy as np
import torch

from rhythmic_insertion.config import compose_sim_config, reset_state_path


class RhythmicInsertionEnv(gym.Env):
    """One KUKA/Robotiq scene with a 17-value observation and six actions."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, vec_env, camera=False, camera_resolution=256):
        self._vec = vec_env
        self._device = vec_env.device
        self.max_episode_steps = int(vec_env.max_episode_length)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self._camera_handle = None
        self._camera_resolution = int(camera_resolution)
        self._closed = False
        self._camera_handle = vec_env.camera_handle if camera else None

    def _observation(self):
        return self._vec.obs_buf[0].detach().cpu().numpy().astype(np.float32, copy=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self._vec.reset_idx(torch.arange(1, device=self._device))
        self._vec.compute_observations()
        return self._observation(), {}

    def _reward(self):
        """Bounded approach, engagement and insertion reward."""
        from rhythmic_insertion.reward import insertion_reward
        vec = self._vec
        return insertion_reward(
            vec.wrench_head_pos[0].detach().cpu().numpy(),
            vec.socket_pos[0].detach().cpu().numpy(),
            float(vec.socket_heights[0]), float(vec.cfg_task.rl.success_height_thresh),
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Expected six finite action values.")
        a = torch.as_tensor(action, device=self._device).clamp(-1, 1).unsqueeze(0)
        _, _, done, extras = self._vec.step(a)
        reward, info = self._reward()
        for key, value in extras.items():
            if key == "time_outs":
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                info[key] = float(value.item())
            elif isinstance(value, (int, float, bool)):
                info[key] = value
        # This metric uses the task's height/keypoint criterion, independent of reward.
        from rhythmic_insertion.sim.tasks.industreal_wrench import industreal_algo_utils as algo
        success = algo.check_plug_inserted_in_socket(
            plug_pos=self._vec.wrench_head_pos, socket_pos=self._vec.socket_pos,
            keypoints_plug=self._vec.keypoints_plug, keypoints_socket=self._vec.keypoints_socket,
            socket_heights=self._vec.socket_heights,
            cfg_task=self._vec.cfg_task, progress_buf=self._vec.progress_buf,
        )
        info["success"] = float(success[0])
        # The task ends only at its time limit.
        return self._observation(), reward, False, bool(done[0]), info

    def render(self):
        if self._camera_handle is None:
            raise RuntimeError("Enable ri_enable_camera or save_video before creating the environment.")
        vec = self._vec
        vec.gym.fetch_results(vec.sim, True)
        vec.gym.step_graphics(vec.sim)
        vec.gym.render_all_camera_sensors(vec.sim)
        rgba = vec.gym.get_camera_image(vec.sim, vec.env_ptrs[0], self._camera_handle, gymapi.IMAGE_COLOR)
        side = self._camera_resolution
        return np.asarray(rgba).reshape(side, side, 4)[..., :3].copy()

    def close(self):
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self._device)
        self._vec.gym.fetch_results(self._vec.sim, True)
        if self._camera_handle is not None:
            self._vec.gym.destroy_camera_sensor(self._vec.sim, self._vec.env_ptrs[0], self._camera_handle)
            self._camera_handle = None
        if self._vec.viewer is not None:
            self._vec.gym.destroy_viewer(self._vec.viewer)
        self._vec.gym.destroy_sim(self._vec.sim)
        from rhythmic_insertion.sim.tasks.base import vec_task
        vec_task.EXISTING_SIM = None


def make_env(cfg):
    """Construct the configured insertion task; all assets resolve within this package."""
    if cfg.get("task", "ri_insert") != "ri_insert":
        raise ValueError("The rhythmic insertion task is named ri_insert.")
    device = str(cfg.get("device", "cuda:0"))
    if not device.startswith("cuda:"):
        raise ValueError("Rhythmic insertion requires an NVIDIA GPU; use device=cuda:0.")
    state_path = reset_state_path(cfg)
    camera = bool(cfg.get("ri_enable_camera", False) or cfg.get("save_video", False))
    from rhythmic_insertion.sim.tasks.industreal_wrench.industreal_task_wrench_insert import IndustRealTaskWrenchInsert
    from omegaconf import OmegaConf
    # The constructor composes subordinate YAML files in this context.
    with compose_sim_config(cfg, state_path, camera) as sim_cfg:
        task_cfg = OmegaConf.to_container(sim_cfg.task, resolve=True)
        vec = IndustRealTaskWrenchInsert(
            cfg=task_cfg, rl_device=device, sim_device=device,
            graphics_device_id=int(cfg.get("ri_graphics_device_id", 0)) if camera else -1,
            headless=True, virtual_screen_capture=False, force_render=False,
        )
    try:
        return RhythmicInsertionEnv(vec, camera, int(cfg.get("ri_camera_resolution", 256)))
    except Exception:
        vec.gym.destroy_sim(vec.sim)
        from rhythmic_insertion.sim.tasks.base import vec_task
        vec_task.EXISTING_SIM = None
        raise
