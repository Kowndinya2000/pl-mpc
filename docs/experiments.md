# Experiment recipes

## Tasks

The HumanoidBench task name is `humanoid_h1hand-<suffix>-v0`. Supported suffixes are:

`walk`, `stand`, `run`, `crawl`, `maze`, `stair`, `slide`, `sit_simple`, `sit_hard`, `pole`, `balance_simple`, `hurdle`, `balance_hard`.

Assets for these 13 H1Hand position-control environments are bundled. Use the separate DMControl environment for `dog-stand`, `dog-trot`, `humanoid-stand`, and `humanoid-walk`.

The default HumanoidBench recipe uses 3M interaction steps, ten evaluation episodes every 50K steps, and `multistep_td_n=3 horizon=3`. Evaluation occurs at episode boundaries. DMControl uses action repeat two and a 500-decision-step episode; the logged `step` counts agent-environment interactions.

## Configuration

Append these flags to `python -m tdmpc_square.train task=... seed=0 steps=3000000`:

| Variant | Flags |
| --- | --- |
| PL-MPC | Defaults |
| Without MTD | `multistep_td=false` |
| Without ATE | `lambda_uncertainty_max=0 lambda_uncertainty_floor=0 beta_uncertainty_max=0` |
| Without RAD | `self_imitation=false` |
| RAD warmup sensitivity | `si_warmup_steps=750000` |
| Observed-only TD targets | `mtd_mode=buffer_only` |
| Model-only TD targets | `mtd_mode=imagination_only` |

RAD uses the `self_imitation` and `si_*` configuration keys. Its default warmup is `si_warmup_steps=200000`, with coefficient `si_coef=0.5`. MTD defaults to `mtd_mode=full`, combining observed replay rewards with model-predicted rewards when the target extends beyond the sampled slice.

## TD-M(PC)² baseline

Run the separately bundled baseline through its launcher:

```bash
bash scripts/train_baseline.sh task=humanoid_h1hand-hurdle-v0 seed=0 steps=3000000 exp_name=baseline
```

The baseline uses the same Python package name as PL-MPC. The launcher selects its import path in a separate process; install only the main release package in the environment.

## Short checks

To exercise the training entrypoint with a short CPU run:

```bash
python -m tdmpc_square.train task=humanoid_h1hand-balance_hard-v0 device=cpu \
  steps=8 eval_episodes=1 eval_pi=false save_video=false \
  num_samples=16 num_elites=4 num_pi_trajs=2 iterations=1 exp_name=entrypoint-smoke
```

This performs initial evaluation and eight-counter-budget interactions, without reaching the replay-seeding threshold. `scripts/smoke.py` exercises simulator steps, replay sampling, a model/actor update including RAD, and checkpoint save/load. These are software checks, not trained-performance evaluations.

## Rhythmic insertion

`ri_insert` uses the separately installed Isaac Gym stack. See [simulation setup and training](rhythmic-insertion.md) for object sizes, reset assets, task defaults, and logging.
