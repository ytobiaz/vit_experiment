import math
import torch
from torch.nn import functional as F

from .config import ConfigBase

__all__ = ["CSKDLoss"]


class CSKDLoss(torch.nn.Module):
    """
    Cumulative Spatial Knowledge Distillation Loss
    """
    def __init__(
        self, 
        cfg: ConfigBase,
        criterion: torch.nn.Module,
        teacher: torch.nn.Module,
    ):
        super().__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.teacher = teacher

    def forward(
        self,
        inputs,
        outputs,
        labels,
        epoch,
        max_epoch
    ):
        if not isinstance(outputs, torch.Tensor):
            outputs, stu_deit_logits, stu_dense_logits = outputs
        loss_base = self.criterion(outputs, labels)
        if self.cfg.deit_loss_type == 'none':
            # no distill loss
            return loss_base

        with torch.no_grad():
            tea_dense_logits = self.teacher(inputs)
            tea_global_logits = tea_dense_logits.mean(dim=(2,3))
        loss_deit = self.get_loss_deit(stu_deit_logits, tea_global_logits)
        loss_cskd = self.get_loss_cskd(stu_dense_logits, tea_dense_logits, 
                    tea_global_logits, epoch, max_epoch)
        alpha = self.cfg.deit_alpha
        loss = loss_base * (1 - alpha) + loss_deit * alpha + \
            loss_cskd * self.cfg.cksd_loss_weight
        return loss

    def align_stu_logits(self, stu_dense_logits):
        N, M, C = stu_dense_logits.shape
        stu_dense_logits = stu_dense_logits.permute(0, 2, 1).reshape(N, C, 14, 14)
        stu_dense_logits = F.avg_pool2d(stu_dense_logits, kernel_size=2, stride=2)
        return stu_dense_logits

    def get_decay_ratio(self, epoch, max_epoch):
        x = epoch / max_epoch
        if self.cfg.cskd_decay_func == 'linear':
            return 1 - x
        elif self.cfg.cskd_decay_func == 'x2':
            return (1 - x) ** 2
        elif self.cfg.cskd_decay_func == 'cos':
            return math.cos(math.pi * 0.5 * x)
        else:
            raise NotImplementedError(self.cfg.cskd_decay_func)

    def get_loss_deit(
        self,
        stu_deit_logits,
        tea_global_logits,
    ):
        # deit loss
        if self.cfg.deit_loss_type == 'soft':
            T = self.cfg.deit_tau
            loss_deit = F.kl_div(
                F.log_softmax(stu_deit_logits / T, dim=-1),
                F.log_softmax(tea_global_logits / T, dim=-1),
                reduction='sum',
                log_target=True,
            ) * (T * T) / stu_deit_logits.numel()
        elif self.cfg.deit_loss_type == 'hard':
            loss_deit = F.cross_entropy(stu_deit_logits, tea_global_logits.argmax(dim=-1))
        else:
            raise NotImplementedError(self.cfg.deit_loss_type)
        return loss_deit

    def get_loss_cskd(
        self,
        stu_dense_logits,
        teacher_dense_logits,
        teacher_global_logits,
        epoch,
        max_epoch,
    ):
        stu_dense_logits = self.align_stu_logits(stu_dense_logits)
        decay_ratio = self.get_decay_ratio(epoch, max_epoch)
        N, C = teacher_global_logits.shape
        teacher_logits = decay_ratio * teacher_dense_logits + \
            (1 - decay_ratio) * teacher_global_logits.reshape(N, C, 1, 1)
        # cskd loss
        if self.cfg.cskd_loss_type == "hard":
            loss_cskd = F.cross_entropy(
                stu_dense_logits, 
                teacher_logits.argmax(dim=1)
                )
        elif self.cfg.cskd_loss_type == "soft":
            T = self.cfg.deit_tau
            loss_cskd = F.kl_div(
                F.log_softmax(stu_dense_logits / T, dim=1),
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / stu_dense_logits.size(0)
        else:
            raise NotImplementedError(self.cfg.cskd_loss_type)
        return loss_cskd
        
# use this for fine-tuning      
# import math
# import torch
# from torch.nn import functional as F

# from .config import ConfigBase

# __all__ = ["CSKDLoss"]


# class CSKDLoss(torch.nn.Module):
#     """
#     Cumulative Spatial Knowledge Distillation Loss
#     """
#     def __init__(
#         self,
#         cfg: ConfigBase,
#         criterion: torch.nn.Module,
#         teacher: torch.nn.Module,
#     ):
#         super().__init__()
#         self.cfg = cfg
#         self.criterion = criterion
#         self.teacher = teacher

#     def forward(
#         self,
#         inputs,
#         outputs,
#         labels,
#         epoch,
#         max_epoch
#     ):
#         # If student returned (out, deit_logits, dense_logits), unpack them;
#         # otherwise just compute base CE loss and return.
#         if isinstance(outputs, tuple) and len(outputs) == 3:
#             out, stu_deit_logits, stu_dense_logits = outputs
#         else:
#             return self.criterion(outputs, labels)

#         # base classification loss
#         loss_base = self.criterion(out, labels)
#         if self.cfg.deit_loss_type == 'none':
#             return loss_base

#         # teacher forward
#         with torch.no_grad():
#             tea_dense_logits = self.teacher(inputs)  # [N, M, C] or [N, C, H, W]
#             # assume [N, M, C] or collapse spatial dims if needed
#             if tea_dense_logits.dim() == 4:
#                 # [N, C, H, W] → treat H*W as M
#                 N, C, H, W = tea_dense_logits.shape
#                 tea_dense_logits = tea_dense_logits.view(N, C, H * W).permute(0, 2, 1)
#             tea_global_logits = tea_dense_logits.mean(dim=1)  # [N, C]

#         # distillation losses
#         loss_deit = self.get_loss_deit(stu_deit_logits, tea_global_logits)
#         loss_cskd = self.get_loss_cskd(
#             stu_dense_logits, tea_dense_logits, tea_global_logits, epoch, max_epoch
#         )

#         alpha = self.cfg.deit_alpha
#         return (
#             loss_base * (1 - alpha) +
#             loss_deit * alpha +
#             loss_cskd * self.cfg.cksd_loss_weight
#         )

#     def align_stu_logits(self, stu_dense_logits):
#         """
#         Convert student dense logits [N, M, C] → [N, C, H, W] for spatial loss.
#         Assumes M = H*W for some H=W.
#         """
#         N, M, C = stu_dense_logits.shape
#         # infer H, W (for DeiT-style patch grid)
#         side = int(math.sqrt(M))
#         x = stu_dense_logits.permute(0, 2, 1).reshape(N, C, side, side)
#         # optionally pool to match teacher resolution
#         return F.avg_pool2d(x, kernel_size=2, stride=2)

#     def get_decay_ratio(self, epoch, max_epoch):
#         x = epoch / max_epoch
#         if self.cfg.cskd_decay_func == 'linear':
#             return 1 - x
#         elif self.cfg.cskd_decay_func == 'x2':
#             return (1 - x) ** 2
#         elif self.cfg.cskd_decay_func == 'cos':
#             return math.cos(math.pi * 0.5 * x)
#         else:
#             raise NotImplementedError(self.cfg.cskd_decay_func)

#     def get_loss_deit(
#         self,
#         stu_deit_logits: torch.Tensor,
#         tea_global_logits: torch.Tensor,
#     ):
#         # deit distillation loss
#         if self.cfg.deit_loss_type == 'soft':
#             T = self.cfg.deit_tau
#             return F.kl_div(
#                 F.log_softmax(stu_deit_logits / T, dim=-1),
#                 F.log_softmax(tea_global_logits / T, dim=-1),
#                 reduction='sum',
#                 log_target=True,
#             ) * (T * T) / stu_deit_logits.numel()
#         elif self.cfg.deit_loss_type == 'hard':
#             return F.cross_entropy(
#                 stu_deit_logits,
#                 tea_global_logits.argmax(dim=-1)
#             )
#         else:
#             raise NotImplementedError(self.cfg.deit_loss_type)

#     def get_loss_cskd(
#         self,
#         stu_dense_logits: torch.Tensor,
#         teacher_dense_logits: torch.Tensor,
#         teacher_global_logits: torch.Tensor,
#         epoch: int,
#         max_epoch: int,
#     ):
#         # align student spatial logits to [N, C, H, W]
#         stu_dense = self.align_stu_logits(stu_dense_logits)
#         decay = self.get_decay_ratio(epoch, max_epoch)
#         N, C = teacher_global_logits.shape

#         # blend teacher dense and global for soft targets
#         teacher_dense = teacher_dense_logits  # [N, M, C]
#         teacher_logits = (
#             decay * teacher_dense +
#             (1 - decay) * teacher_global_logits.reshape(N, 1, C)
#         )  # [N, M, C]

#         # reshape teacher_logits → [N, C, M]
#         teacher_logits = teacher_logits.permute(0, 2, 1)

#         if self.cfg.cskd_loss_type == "hard":
#             return F.cross_entropy(
#                 stu_dense,
#                 teacher_logits.argmax(dim=1)
#             )
#         elif self.cfg.cskd_loss_type == "soft":
#             T = self.cfg.deit_tau
#             return F.kl_div(
#                 F.log_softmax(stu_dense / T, dim=1),
#                 F.log_softmax(teacher_logits / T, dim=1),
#                 reduction='sum',
#                 log_target=True
#             ) * (T * T) / stu_dense.size(0)
#         else:
#             raise NotImplementedError(self.cfg.cskd_loss_type)
