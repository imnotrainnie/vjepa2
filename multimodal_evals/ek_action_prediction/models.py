# Implements EK_PLAN_PART2 §3
import logging
from typing import Dict

import torch

from multimodal_evals.action_anticipation_frozen.models import AttentiveClassifier
from src.models.multimodal_jepa import MultimodalJEPA, build_vjepa2_1_vitb_encoder
from src.models.text_encoder import SigLIPTextEncoder
from src.utils.checkpoint_loader import load_multimodal_checkpoint

logger = logging.getLogger(__name__)


def init_module(model_cfg: Dict[str, object], device: torch.device, rank: int = 0) -> MultimodalJEPA:
    v_cfg = model_cfg["v_encoder"]
    l_cfg = model_cfg["l_encoder"]
    proj_cfg = model_cfg["projectors"]
    pred_cfg = model_cfg["predictor"]

    v_encoder = build_vjepa2_1_vitb_encoder(
        checkpoint_path=v_cfg.get("checkpoint_path"),
        img_size=v_cfg.get("img_size", 384),
        patch_size=v_cfg.get("patch_size", 16),
        tubelet_size=v_cfg.get("tubelet_size", 2),
        num_frames=v_cfg.get("num_frames", 32),
    )
    l_encoder = SigLIPTextEncoder(
        model_name=l_cfg.get("model_name", "google/siglip-large-patch16-384"),
        max_length=l_cfg.get("max_length", 64),
        freeze=True,
        local_files_only=l_cfg.get("local_files_only", False),
    )

    model = MultimodalJEPA(
        v_encoder=v_encoder,
        l_encoder=l_encoder,
        video_dim=v_cfg.get("embed_dim", 768),
        text_dim=l_cfg.get("embed_dim", 1024),
        shared_dim=proj_cfg.get("shared_dim", 512),
        projector_hidden_dim=proj_cfg.get("hidden_dim", 1024),
        predictor_dim=pred_cfg.get("predictor_dim", 384),
        predictor_depth=pred_cfg.get("depth", 12),
        predictor_num_heads=pred_cfg.get("num_heads", 8),
        img_size=v_cfg.get("img_size", 384),
        patch_size=v_cfg.get("patch_size", 16),
        tubelet_size=v_cfg.get("tubelet_size", 2),
        freeze_encoders=True,
    ).to(device)

    load_info = load_multimodal_checkpoint(
        model,
        checkpoint_path=model_cfg.get("checkpoint"),
        strict=False,
        map_location="cpu",
    )
    if rank == 0:
        missing = load_info.get("model_missing_keys", [])
        unexpected = load_info.get("model_unexpected_keys", [])
        logger.info("[load_multimodal_checkpoint] missing_keys=%d, unexpected_keys=%d", len(missing), len(unexpected))
        logger.info("[load_multimodal_checkpoint] missing[:20] = %s", list(missing)[:20])
        logger.info("[load_multimodal_checkpoint] unexpected[:20] = %s", list(unexpected)[:20])

    _freeze_modules([model.v_encoder, model.l_encoder, model.projectors, model.predictor])
    model.eval()
    return model


def _freeze_modules(modules):
    for module in modules:
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad = False


def _log_trainable_params(model, classifiers, rank: int) -> None:
    if rank != 0:
        return
    n_frozen = sum(param.numel() for param in model.parameters())
    n_train = sum(param.numel() for classifier in classifiers for param in classifier.parameters() if param.requires_grad)
    logger.info("Frozen backbone params=%s; trainable head params=%s", f"{n_frozen:,}", f"{n_train:,}")
    if any(param.requires_grad for param in model.parameters()):
        raise RuntimeError("backbone must be frozen")
    if not any(param.requires_grad for classifier in classifiers for param in classifier.parameters()):
        raise RuntimeError("classifier must have trainable parameters")


def init_classifier(
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    num_classifiers: int,
    verb_classes: Dict[int, int],
    noun_classes: Dict[int, int],
    action_classes: Dict[tuple, int],
    device: torch.device,
):
    if embed_dim % num_heads != 0:
        raise AssertionError(f"classifier.num_heads={num_heads} must divide embed_dim={embed_dim}")
    return [
        AttentiveClassifier(
            verb_classes=verb_classes,
            noun_classes=noun_classes,
            action_classes=action_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=num_blocks,
            use_activation_checkpointing=True,
        ).to(device)
        for _ in range(num_classifiers)
    ]
