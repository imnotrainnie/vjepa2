# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
from typing import Any

import torch
from torch.serialization import MAP_LOCATION

from src.utils.logging import get_logger

logger = get_logger(os.path.basename(__file__))


def robust_checkpoint_loader(r_path: str, map_location: MAP_LOCATION = "cpu", max_retries: int = 3) -> Any:
    """
    Loads a checkpoint from a path, retrying up to max_retries times if the checkpoint is not found.
    """
    retries = 0

    while retries < max_retries:
        try:
            return torch.load(r_path, map_location=map_location)
        except Exception as e:
            logger.warning(f"Encountered exception when loading checkpoint {e}")
            retries += 1
            if retries < max_retries:
                sleep_time_s = (2**retries) * random.uniform(1.0, 1.1)
                logger.warning(f"Sleeping {sleep_time_s}s and trying again, count {retries}/{max_retries}")
                time.sleep(sleep_time_s)
                continue
            else:
                raise e


def load_multimodal_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | None = None,
    vjepa_checkpoint_path: str | None = None,
    strict: bool = False,
    map_location: MAP_LOCATION = "cpu",
) -> dict[str, Any]:
    """Load a multimodal JEPA checkpoint and/or pretrained V-JEPA encoder weights.

    Existing checkpoint formats in this repository use different encoder keys, so
    this function tries the common V-JEPA names and loads with strict=False.
    """
    loaded: dict[str, Any] = {}

    if vjepa_checkpoint_path is not None:
        vjepa_checkpoint = robust_checkpoint_loader(vjepa_checkpoint_path, map_location=map_location)
        encoder_state = None
        for key in ("encoder", "ema_encoder", "target_encoder"):
            if isinstance(vjepa_checkpoint, dict) and key in vjepa_checkpoint:
                encoder_state = vjepa_checkpoint[key]
                loaded["vjepa_encoder_key"] = key
                break
        if encoder_state is None:
            encoder_state = vjepa_checkpoint
            loaded["vjepa_encoder_key"] = "<root>"

        cleaned_state = {}
        for key, value in encoder_state.items():
            cleaned_state[key.replace("module.", "").replace("backbone.", "")] = value
        missing, unexpected = model.v_encoder.load_state_dict(cleaned_state, strict=False)
        loaded["vjepa_missing_keys"] = missing
        loaded["vjepa_unexpected_keys"] = unexpected

    if checkpoint_path is not None:
        checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        loaded["checkpoint"] = checkpoint
        loaded["model_missing_keys"] = missing
        loaded["model_unexpected_keys"] = unexpected

    return loaded
