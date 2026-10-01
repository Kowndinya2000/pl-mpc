# Beyond Policy Alignment: Closing the Planning–Learning Loop for Robot Control with Learned World Models

Kowndinya Boyalakuntla, Yuhan Liu, Abdeslam Boularias  
Department of Computer Science, Rutgers University

[Website](https://pl-mpc-humanoid.github.io/) · [Paper](https://arxiv.org/abs/2609.39751)

```bash
git clone https://github.com/Kowndinya2000/pl-mpc.git
cd pl-mpc
```

**PL-MPC (Planning–Learning MPC)** builds on TD-M(PC)². This package provides training and checkpoint-evaluation code, 13 HumanoidBench locomotion environments, DMControl integration, and the rhythmic wrench–nut insertion simulator.

The configurable components are hybrid multi-step TD targets (**MTD**), adaptive terminal estimates (**ATE**), and return-weighted actor distillation (**RAD**). See [experiment recipes](docs/experiments.md) for tasks, configuration switches, and the TD-M(PC)² baseline launcher.

## Install

Tested on Linux x86-64 with Python 3.11 and PyTorch 2.3.1. CPU is sufficient for the checks below. For full training, use an NVIDIA CUDA environment with a compatible driver; the installation command below uses CUDA 12.1 wheels.

Use separate environments for the simulator stacks. [Rhythmic insertion setup](docs/rhythmic-insertion.md) uses Python 3.8 and NVIDIA Isaac Gym Preview 4; its task code, meshes and reset states are bundled.

| Suite | MuJoCo | dm_control | Requirements |
| --- | --- | --- | --- |
| HumanoidBench | 3.1.6 | 1.0.20 | `requirements/humanoidbench.txt` |
| DMControl | 3.3.4 | 1.0.31 | `requirements/dmcontrol.txt` |

From the extracted release root, create a CPU HumanoidBench environment:

```bash
python3.11 -m venv .venv-hb
source .venv-hb/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/humanoidbench.txt -e .
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Install DMControl in its own environment, also from the release root:

```bash
python3.11 -m venv .venv-dmc
source .venv-dmc/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/dmcontrol.txt -e .
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Run the installation commands in order so the chosen PyTorch wheel is installed before the suite requirements. `python -m pip check` checks dependency compatibility after installation.

For GPU training, create a separate environment and install the CUDA wheel before the suite requirements:

```bash
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

These wheel selections follow the [PyTorch 2.3.1 installation instructions](https://docs.pytorch.org/get-started/previous-versions/).

MuJoCo needs system OpenGL/EGL libraries even for the HumanoidBench state-observation smoke. All HumanoidBench task meshes are bundled. HumanoidBench and DMControl need no private account or logging account. Rhythmic insertion requires a separate NVIDIA Isaac Gym SDK installation.

## Quick start

```bash
# Verify the distributed files on Linux
sha256sum -c MANIFEST.sha256
```

In the HumanoidBench environment:

```bash
source .venv-hb/bin/activate
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/smoke.py --task humanoid_h1hand-balance_hard-v0
python -m pytest -q -p no:cacheprovider
```

In the DMControl environment:

```bash
source .venv-dmc/bin/activate
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/smoke.py --task dog-stand
```

The smoke uses eight interaction steps and a single update with a small planning sample count. It does not measure learned performance. It creates a temporary checkpoint unless `--checkpoint-out PATH` is supplied.

## Train and evaluate

Training runs use local logging by default (`disable_wandb=true`). These are full-run recipes, not commands needed for the quick start:

```bash
# HumanoidBench; set seed for each independent training run
python -m tdmpc_square.train task=humanoid_h1hand-balance_hard-v0 seed=0 steps=3000000

# DMControl environment
python -m tdmpc_square.train task=dog-stand seed=0 steps=3000000

# Evaluate a checkpoint produced by the matching task/configuration
python -m tdmpc_square.evaluate task=dog-stand seed=0 \
  checkpoint=logs/dog-stand/0/default/models/final.pt eval_episodes=10 save_video=false
```

Add `device=cpu` for CPU operation. Training at the default budget on CPU is slow. `output_dir=/your/output/directory` changes the output root. Runs write `config.yaml`, `eval.csv`, `train.jsonl`, `eval.jsonl`, and checkpoints under `logs/<task>/<seed>/<exp_name>/`. Checkpoints store model weights, not a complete resumable optimizer/replay state. Use the same model configuration when evaluating. No trained research checkpoints are bundled.

[Experiment recipes](docs/experiments.md) list tasks, ablations, and the separate baseline launcher. `python -m tdmpc_square.train --cfg job` prints the configuration without running training. See [Weights & Biases logging](#weights--biases-logging) below for account setup and optional online/offline tracking.

## Configuration defaults

The main config is [`tdmpc_square/config.yaml`](tdmpc_square/config.yaml). CLI `key=value` arguments override it for that run. For example, the commands above explicitly choose `seed=0`; the config default is `seed=1`.

| Setting | PL-MPC default | TD-M(PC)² baseline default |
| --- | --- | --- |
| Task | `dog-stand` | `dog-run` |
| Interaction budget | `3000000` | `2000000` |
| Seed / device | `1` / `cuda:0` | `1` / `cuda:0` |
| Batch size / learning rate | `256` / `0.0003` | `256` / `0.0003` |
| Planning horizon / samples / elites | `3` / `512` / `64` | `3` / `512` / `64` |
| Evaluation | 10 episodes every 50K steps, plus initial evaluation | Same |
| Local outputs | `logs/<task>/<seed>/<exp_name>/` | Same |
| Experiment label | `default` | `default` |
| W&B project / method label | `pl-mpc` / `pl-mpc` | `pl-mpc` / `tdmpc-square` |
| W&B enabled | No (`disable_wandb=true`) | No |
| Local metrics / checkpoints / video | On / on / off | Same |

PL-MPC enables MTD, ATE, and RAD by default: `multistep_td_n=3`, `lambda_uncertainty_max=0.5`, `si_coef=0.5`, and `si_warmup_steps=200000`. The baseline has its own [config](baselines/tdmpc_square_original/tdmpc_square/config.yaml); the 3M-step baseline recipe overrides its default budget explicitly.

Environment construction fills observation/action dimensions, episode length, and `seed_steps=max(1000, 5*episode_length)`. The planner starts from `iterations=6` and adds two iterations when the action dimension is at least 20. Leaving `model_size` unset uses the explicit network dimensions in the config. Supply `checkpoint` only when evaluating a saved model. The `???` entries are runtime-derived or optional fields, not account credentials.

Use `python -m tdmpc_square.train --cfg job` or `bash scripts/train_baseline.sh --cfg job` to inspect CLI overrides before starting a run. These commands print the configuration before environment-dependent fields are filled. Choose a distinct `exp_name` for each method, variant, or repeated run with the same task and seed; reusing the same local path can overwrite checkpoints and mix logs.

## Weights & Biases logging

W&B is optional. With the default `disable_wandb=true`, no W&B account or package is needed. Local metric files are controlled by `save_csv`, local checkpoints by `save_agent`, and videos by `save_video`.

### Account and authentication

1. [Create a W&B account](https://wandb.ai/site/) or sign in. Choose an existing account/team namespace where you can create projects; this is the `wandb_entity` value, not an email address.
2. In **User Settings**, create a personal API key, give it a name, and copy the full key. See the [official W&B setup guide](https://docs.wandb.ai/models/quickstart).
3. In the activated simulator environment, install the optional SDK and authenticate on the machine that will run training:

```bash
python -m pip install -r requirements/wandb.txt
wandb login
```

Paste the API key at the login prompt. For noninteractive jobs, supply `WANDB_API_KEY` through your job's secret environment. Keep the key out of YAML configs and source files. `wandb_entity=null` uses the logged-in account's default entity; set an existing entity explicitly when selecting a team. [W&B entity and project reference](https://docs.wandb.ai/models/ref/python/functions/init).

### Start a tracked run

Replace `YOUR_ENTITY` with that namespace:

```bash
python -m tdmpc_square.train task=dog-stand seed=0 exp_name=main \
  disable_wandb=false wandb_mode=online \
  wandb_entity=YOUR_ENTITY wandb_project=pl-mpc

# Baseline: use a separate local experiment label and the same dashboard project.
bash scripts/train_baseline.sh task=dog-stand seed=0 steps=3000000 exp_name=baseline \
  disable_wandb=false wandb_mode=online \
  wandb_entity=YOUR_ENTITY wandb_project=pl-mpc
```

The project can be created in the W&B UI or on the first run when your entity permits project creation ([project guide](https://docs.wandb.ai/models/track/project-page)). Open that entity's `pl-mpc` project to see runs and charts.

| W&B field | Default construction | Example for the first command |
| --- | --- | --- |
| Display name | `<task>.<method>.<exp_name>.seed-<seed>` | `dog-stand.pl-mpc.main.seed-0` |
| Group | `<method>-<task>-<exp_name>` (experiment punctuation becomes hyphens) | `pl-mpc-dog-stand-main` |
| Tags | Method, task, normalized experiment label, seed | `pl-mpc`, `dog-stand`, `main`, `seed:0` |

Seeds share a group, while PL-MPC and the baseline have different method labels. Set `wandb_run_name=my-run` or `wandb_group=my-group` to override display metadata; append tags with `'wandb_tags=[ablation,without-rad]'`. Set `exp_name` separately to choose the local output directory. A repeated display name creates another W&B run; it does not resume training.

Training logs the resolved configuration and `train/*`, `eval/*`, and `results/*` metrics against environment interaction steps. Useful charts include `eval/episode_reward`, `eval/episode_success`, and training losses. With `eval_pi=true`, `eval/*_pi` metrics evaluate the actor without planning. Metrics are written when episodes finish; evaluation can occur after the requested step interval. Local files include `config.yaml`, `train.jsonl`, `eval.jsonl`, `results.jsonl`, `eval.csv`, and `models/`.

With `save_agent=true wandb_save_model=true`, PL-MPC sends best and final checkpoints as model artifacts; the baseline sends the final checkpoint. Periodic PL-MPC step checkpoints stay local. Use `wandb_save_model=false` to keep checkpoints local while tracking metrics. `save_video=true` saves evaluation videos and logs them to W&B. The standalone `python -m tdmpc_square.evaluate` command prints checkpoint-evaluation results and optionally saves videos; it does not create a W&B run.

### Offline tracking and later sync

To create W&B run files without uploading during training:

```bash
python -m tdmpc_square.train task=dog-stand seed=0 exp_name=offline-main \
  disable_wandb=false wandb_mode=offline wandb_entity=YOUR_ENTITY
```

The run directory is printed by W&B and is under `logs/dog-stand/0/offline-main/wandb/offline-run-*`. When ready to upload, authenticate and sync the specific directory:

```bash
wandb login
wandb sync --entity YOUR_ENTITY --project pl-mpc /path/to/offline-run-DATE-RUNID
```

`wandb_mode` explicitly selects online or offline behavior when W&B is enabled. To use only the release's local logging, set `disable_wandb=true`. See [W&B sync](https://docs.wandb.ai/models/ref/cli/wandb-sync).

## Troubleshooting

- Keep NumPy at 1.26.4 with this PyTorch version. Installing the suite requirements supplies that pin.
- Keep the two MuJoCo stacks separate. A single mixed environment is not supported.
- If EGL fails, install a working system EGL driver. `MUJOCO_GL=osmesa` additionally requires the system OSMesa library; it is not supplied by pip.
- Run commands from the extracted release root after installation. Use `save_video=false` for inexpensive checks.

Third-party copyright and license notices are preserved in [THIRD_PARTY.md](THIRD_PARTY.md) and `licenses/`.

## Citation

```bibtex
@misc{boyalakuntla2026plmpc,
  title         = {Beyond Policy Alignment: Closing the Planning-Learning Loop for Robot Control with Learned World Models},
  author        = {Kowndinya Boyalakuntla and Yuhan Liu and Abdeslam Boularias},
  year          = {2026},
  eprint        = {2609.39751},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.39751}
}
```
