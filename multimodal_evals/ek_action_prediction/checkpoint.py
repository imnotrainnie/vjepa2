# Implements EK_PLAN_PART3B §9
import os

import torch


def _strip_ddp(state_dict):
    out = {}
    for key, value in state_dict.items():
        out[key[len("module.") :] if key.startswith("module.") else key] = value
    return out


def save_classifier_checkpoint(path, classifiers, optimizers, epoch, rank, world_size):
    if rank != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "world_size": int(world_size),
        "classifiers": [_strip_ddp(classifier.state_dict()) for classifier in classifiers],
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
    }
    torch.save(payload, path)


def load_classifier_checkpoint(path, classifiers, optimizers, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    for classifier, state_dict in zip(classifiers, ckpt["classifiers"]):
        target = classifier.module if hasattr(classifier, "module") else classifier
        target.load_state_dict(state_dict, strict=True)
    if optimizers is not None:
        for optimizer, state_dict in zip(optimizers, ckpt["optimizers"]):
            optimizer.load_state_dict(state_dict)
    return ckpt.get("epoch", 0)
