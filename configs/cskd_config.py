import os
import torch

from cskd.config import ConfigBase


class Config(ConfigBase):
    # base
    exp_name = os.path.splitext(os.path.basename(__file__))[0]

    # deit
    deit_loss_type = "hard"  # none, soft, hard
    deit_alpha = 0.5
    deit_tau = 1.0

    # cskd
    cskd_decay_func = "linear"
    cskd_loss_type = "hard"
    cksd_loss_weight = 1.0


if __name__ == "__main__":
    cfg = Config.instance()
    cfg.print()
