# Rhythmic wrench–nut insertion simulation

The `ri_insert` task simulates a KUKA iiwa14 arm and Robotiq three-finger gripper placing a grasped wrench onto a nut on an upright bolt. The release includes the simulator task, controller, Gymnasium interface, robot/object meshes, and initial physics states for object sizes 5, 3 and 1. Size 5 is the training default. The initial states contain robot and object poses and controller targets; they contain no policy weights, rewards, or evaluation trajectories.

## Installation

Use a separate **Python 3.8** environment for **NVIDIA Isaac Gym Preview 4**. The HumanoidBench/DMControl Python 3.11 environments are separate. Download and extract the SDK from [NVIDIA's Isaac Gym download page](https://developer.nvidia.com/isaac-gym/download) and follow its system prerequisites. The SDK is installed separately under NVIDIA's license. A compatible NVIDIA GPU and driver, Linux x86-64, and a C++ build toolchain are required. The SDK compiles its PyTorch extension on first use.

From the release root, with Python 3.8 available:

```bash
python3.8 -m venv .venv-ri
source .venv-ri/bin/activate
python -m pip install --upgrade 'pip<25.1' 'setuptools<76' wheel
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/rhythmic-insertion.txt
# Set this to your extracted NVIDIA SDK directory.
export ISAAC_GYM_ROOT=/path/to/isaacgym
python -m pip install --no-deps -e "$ISAAC_GYM_ROOT/python"
python -m pip install --no-deps -e .
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m rhythmic_insertion.smoke --steps 16
```

Keep NumPy below 1.24 because the SDK uses the older NumPy scalar aliases. If a Conda-provided Python reports that `libpython3.8.so` is missing, activate that environment and prepend its library directory with `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`.

To check another object size:

```bash
python -m rhythmic_insertion.smoke --size 3 --steps 16
python -m rhythmic_insertion.smoke --size 1 --steps 16
```

The first launch may spend several minutes preparing collision meshes. These caches are created by the SDK outside the release source. The validated training workflow uses state-only simulation (`save_video=false`). Optional camera rendering requires a compatible Vulkan driver; check it separately with `python -m rhythmic_insertion.smoke --render --steps 16` before enabling video recording.

## Environment interface and defaults

| Setting | Default / meaning |
| --- | --- |
| Task | `task=ri_insert` |
| Object size | `ri_asset_size=5`; supported values: `1`, `3`, `5` |
| Observation | 17 floats: current wrench-head pose, target pose and relative position in the noisy nut frame |
| Action | Six values in `[-1,1]`: wrench-head translation/rotation displacement; scales are 0.01 m and 0.01 rad |
| Episode counter | `ri_episode_length=256`; the inherited simulator time limit is reached after 255 control transitions |
| Reset state | `ri_reset_states=null` selects the bundled states matching the object size |
| Camera | `ri_enable_camera=false`; `save_video=true` also enables it |
| Camera resolution | `ri_camera_resolution=256` |
| Graphics device | `ri_graphics_device_id=0`; select the Vulkan device used for rendering |

Reset states start above the nut at the hardest initialization stage. Custom state files must use the bundled NumPy archive schema and match the selected size; missing or mismatched files raise an error. `reset(seed=...)` restores a sampled physics state and refreshes the observation. `step(action)` returns `(observation, reward, terminated, truncated, info)`; episodes end at the time limit, and `info['success']` reports the current insertion criterion.

The reward is `0.5 * exp(-d² / (2 * 0.03²)) + 0.3 * engaged + 0.2 * inserted`, where `d` is the distance from the wrench head to 5 mm above the nut origin. Engagement requires absolute height error below half the nut height and radial error below 50 mm; insertion uses a 5 mm height threshold and the same radial bound. The separate success criterion requires absolute height error below 5 mm and summed distance between four wrench/nut keypoints below 50 mm.

For direct use, import the simulator before PyTorch:

```python
from rhythmic_insertion.env import make_env
import numpy as np

env = make_env({"task": "ri_insert", "device": "cuda:0", "ri_asset_size": 5})
try:
    observation, info = env.reset(seed=1)
    observation, reward, terminated, truncated, info = env.step(np.zeros(6))
finally:
    env.close()
```

## PL-MPC and baseline training

Use the same release training interfaces and logging configuration:

```bash
python -m tdmpc_square.train task=ri_insert ri_asset_size=5 \
  steps=1000000 seed=1 device=cuda:0 iterations=8 \
  eval_freq=10000 eval_episodes=10 exp_name=insertion

python scripts/train_baseline.py task=ri_insert ri_asset_size=5 \
  steps=1000000 seed=1 device=cuda:0 iterations=8 \
  eval_freq=10000 eval_episodes=10 exp_name=insertion-baseline
```

These commands use the PL-MPC and TD-M(PC)² configurations documented in [experiment recipes](experiments.md). Checkpoint evaluation uses `python -m tdmpc_square.evaluate task=ri_insert checkpoint=/path/to/checkpoint.pt` with the same training configuration overrides and object size. For baseline checkpoints, use `python scripts/evaluate_baseline.py task=ri_insert checkpoint=/path/to/checkpoint.pt` with matching overrides.

W&B is disabled by default. To enable it in this Python 3.8 environment, install `requirements/rhythmic-insertion-wandb.txt`, create an account, and run `wandb login` as described in the [main README](../README.md#weights--biases-logging). Add `disable_wandb=false wandb_project=pl-mpc` to either training command. `wandb_entity=null` uses the logged-in account's default entity. Automatic names include task, method, experiment and seed; optional offline logging uses `wandb_mode=offline`. Local CSVs, videos and checkpoints follow the same output layout as the other tasks.
