import os
import sys

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")

os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
import torch

import hydra
from termcolor import colored

from tdmpc_square.common.parser import parse_cfg
from tdmpc_square.common.seed import set_seed
from tdmpc_square.common.buffer import Buffer
from tdmpc_square.envs import make_env
from tdmpc_square.tdmpc_square import TDMPC2

from tdmpc_square.trainer.online_trainer import OnlineTrainer
from tdmpc_square.common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(version_base="1.3", config_name="config", config_path=".")
def train(cfg: dict):
    """Train the vendored TD-M(PC)² baseline with the selected suite."""

    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    trainer_cls = OnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
