import os
import datetime
import re
import json
import imageio
import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf

from tdmpc_square.common import TASK_SET


CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("episode", "E", "int"),
    ("step", "I", "int"),
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
    "results": "magenta",
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """
    Pretty-printing of current run information.
    Logger calls this method at initialization.
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs),
            _limstr(v),
        )

    observations = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("task", cfg.task_title),
        ("steps", f"{int(cfg.steps):,}"),
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """
    Return a wandb-safe group name for logging.
    Optionally returns group name as list.
    """
    lst = [cfg.wandb_method, cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, cfg, wandb, fps=15):
        self.cfg = cfg
        self._save_dir = make_dir(cfg.work_dir / "eval_video")
        self._wandb = wandb
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = bool(self._save_dir and enabled)
        self.record(env)

    def record(self, env):
        if self.enabled:
            self.frames.append(env.render())

    def save(self, step, key="videos/eval_video"):
        if self.enabled and len(self.frames) > 0:
            frames = np.stack(self.frames)
            filename = self._save_dir / f"{step}_{key.replace('/', '_')}.mp4"
            imageio.mimsave(filename, frames, fps=self.fps)
            if self._wandb:
                self._wandb.log({key: self._wandb.Video(str(filename))}, step=step)


class Logger:
    """Primary logging object. Logs either locally or using wandb."""

    def __init__(self, cfg):
        self._log_dir = make_dir(cfg.work_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_csv = cfg.save_csv
        self._save_agent = cfg.save_agent
        self._wandb_save_model = cfg.wandb_save_model
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = []
        OmegaConf.save(cfg, self._log_dir / "config.yaml")
        print_run(cfg)
        self.project = cfg.wandb_project
        self.entity = cfg.wandb_entity
        if cfg.disable_wandb:
            print(colored("W&B disabled; local logging is active.", "blue", attrs=["bold"]))
            self._wandb = None
            self._video = VideoRecorder(cfg, None) if cfg.save_video else None
            return
        if not self.project or str(self.project).lower() == "none":
            raise ValueError("Set wandb_project to a project name when enabling W&B.")
        if cfg.wandb_mode not in {"online", "offline"}:
            raise ValueError("wandb_mode must be online or offline; use disable_wandb=true to disable W&B.")
        if self.entity is not None and str(self.entity).lower() in {"", "none"}:
            self.entity = None
        os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "W&B is enabled but unavailable. Install it with: python -m pip install -r requirements/wandb.txt"
            ) from exc

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=cfg.wandb_run_name or f"{cfg.task}.{cfg.wandb_method}.{cfg.exp_name}.seed-{cfg.seed}",
            group=cfg.wandb_group or self._group,
            tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"] + list(cfg.wandb_tags),
            mode=cfg.wandb_mode,
            dir=str(self._log_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        message = "W&B logging is offline; sync the run later." if cfg.wandb_mode == "offline" else "Logs will be synced with W&B."
        print(colored(message, "blue", attrs=["bold"]))
        self._wandb = wandb
        self._video = VideoRecorder(cfg, self._wandb) if cfg.save_video else None

    @property
    def video(self):
        return self._video

    @property
    def model_dir(self):
        return self._model_dir

    def save_agent(self, agent=None, identifier="final", to_wandb=True):
        if self._save_agent and agent:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp)
            if self._wandb and to_wandb and self._wandb_save_model:
                artifact = self._wandb.Artifact(
                    self._group + "-" + str(self._seed) + "-" + str(identifier),
                    type="model",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def finish(self, agent=None):
        try:
            self.save_agent(agent)
        except Exception as e:
            print(colored(f"Failed to save model: {e}", "red"))
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise ValueError(f"invalid log format type: {ty}")

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def pprint_multitask(self, d, cfg):
        """Pretty-print evaluation metrics for multi-task training."""
        print(
            colored(
                f"Evaluated agent on {len(cfg.tasks)} tasks:", "yellow", attrs=["bold"]
            )
        )
        dmcontrol_reward = []
        metaworld_reward = []
        metaworld_success = []
        for k, v in d.items():
            if "+" not in k:
                continue
            task = k.split("+")[1]
            if task in TASK_SET["mt30"] and k.startswith("episode_reward"):
                dmcontrol_reward.append(v)
                print(colored(f"  {task:<22}\tR: {v:.01f}", "yellow"))
            elif (
                task in TASK_SET["mt80"] and task not in TASK_SET["mt30"]
            ):
                if k.startswith("episode_reward"):
                    metaworld_reward.append(v)
                elif k.startswith("episode_success"):
                    metaworld_success.append(v)
                    print(colored(f"  {task:<22}\tS: {v:.02f}", "yellow"))
        dmcontrol_reward = np.nanmean(dmcontrol_reward)
        d["episode_reward+avg_dmcontrol"] = dmcontrol_reward
        print(
            colored(
                f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}',
                "yellow",
                attrs=["bold"],
            )
        )
        if cfg.task == "mt80":
            metaworld_reward = np.nanmean(metaworld_reward)
            metaworld_success = np.nanmean(metaworld_success)
            d["episode_reward+avg_metaworld"] = metaworld_reward
            d["episode_success+avg_metaworld"] = metaworld_success
            print(
                colored(
                    f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}',
                    "yellow",
                    attrs=["bold"],
                )
            )
            print(
                colored(
                    f'  {"metaworld":<22}\tS: {metaworld_success:.02f}',
                    "yellow",
                    attrs=["bold"],
                )
            )

    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb:
            if category in {"train", "eval", "results"}:
                xkey = "step"
            elif category == "pretrain":
                xkey = "iteration"
            metrics = {category + "/" + k: v for k, v in d.items()
                       if not (category == "results" and k == "step")}
            self._wandb.log(metrics, step=d[xkey])
        if self._save_csv:
            scalars = {key: value.item() if hasattr(value, "item") else value
                       for key, value in d.items()}
            with open(self._log_dir / f"{category}.jsonl", "a") as stream:
                stream.write(json.dumps(scalars) + "\n")
            if category == "eval":
                self._eval.append(scalars)
                pd.DataFrame(self._eval).to_csv(self._log_dir / "eval.csv", index=False)
        if category != "results":
            self._print(d, category)
