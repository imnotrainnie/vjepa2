# Implements EK_PLAN_PART3B §8
import logging
import math

import torch

__all__ = ["init_opt", "WarmupCosineLRSchedule", "CosineWDSchedule"]

logger = logging.getLogger(__name__)


def init_opt(classifiers, opt_kwargs, iterations_per_epoch, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for classifier, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for _, p in classifier.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("weight_decay"),
                "mc_final_wd": kwargs.get("final_weight_decay"),
                "lr": kwargs.get("start_lr"),
                "weight_decay": kwargs.get("weight_decay"),
            }
        ]
        logger.info("Using AdamW for EK classifier")
        optimizers.append(torch.optim.AdamW(param_groups))
        schedulers.append(WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
        wd_schedulers.append(CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
        scalers.append(torch.cuda.amp.GradScaler() if use_bfloat16 else None)
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule:
    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(final_lr, final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            group["lr"] = new_lr


class CosineWDSchedule:
    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
