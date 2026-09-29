import os
import sys

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")

import warnings

warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from tdmpc_square.common.parser import parse_cfg
from tdmpc_square.common.seed import set_seed
from tdmpc_square.envs import make_env
from tdmpc_square.tdmpc_square import TDMPC2

torch.backends.cudnn.benchmark = True


@hydra.main(version_base="1.3", config_name="config", config_path=".")
def evaluate(cfg: dict):
    """Evaluate a single-task checkpoint with MPPI using the training configuration."""
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(
        colored(
            f'Model size: {cfg.get("model_size", "default")}', "blue", attrs=["bold"]
        )
    )
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    if not cfg.multitask and ("mt80" in cfg.checkpoint or "mt30" in cfg.checkpoint):
        print(
            colored(
                "Warning: single-task evaluation of multi-task models is not currently supported.",
                "red",
                attrs=["bold"],
            )
        )
        print(
            colored(
                "To evaluate a multi-task model, use task=mt80 or task=mt30.",
                "red",
                attrs=["bold"],
            )
        )


    env = make_env(cfg)


    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.checkpoint)


    if cfg.multitask:
        print(
            colored(
                f"Evaluating agent on {len(cfg.tasks)} tasks:", "yellow", attrs=["bold"]
            )
        )
    else:
        print(colored(f"Evaluating agent on {cfg.task}:", "yellow", attrs=["bold"]))
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    if cfg.multitask:
        raise ValueError("This release supports single-task evaluation.")
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None
        ep_rewards, ep_successes = [], []
        for i in range(cfg.eval_episodes):
            obs, done, ep_reward, t = env.reset(task_idx=task_idx)[0], False, 0, 0
            if cfg.save_video:
                frames = [env.render()]
            while not done:
                action, _, _ = agent.act(obs, t0=t == 0, task=task_idx, eval_mode=True)
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                ep_reward += float(reward)
                t += 1
                if cfg.save_video:
                    frames.append(env.render())
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if cfg.save_video:
                imageio.mimsave(
                    os.path.join(video_dir, f"{task}-{i}.mp4"), frames, fps=15
                )
        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)
        if cfg.multitask:
            scores.append(
                ep_successes * 100 if task.startswith("mw-") else ep_rewards / 10
            )
        print(
            colored(
                f"  {task:<22}" f"\tR: {ep_rewards:.01f}  " f"\tS: {ep_successes:.02f}",
                "yellow",
            )
        )
    if cfg.multitask:
        print(
            colored(
                f"Normalized score: {np.mean(scores):.02f}", "yellow", attrs=["bold"]
            )
        )

    env.close()

if __name__ == "__main__":
    evaluate()
