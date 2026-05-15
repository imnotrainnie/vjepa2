from typing import Dict

import torch

from src.models.multimodal_jepa import MultimodalJEPA, build_vjepa2_1_vitb_encoder
from src.models.text_encoder import SigLIPTextEncoder
from src.utils.checkpoint_loader import load_multimodal_checkpoint
from multimodal_evals.action_anticipation_frozen.models import AttentiveClassifier


def init_module(model_cfg: Dict[str, object], device: torch.device) -> MultimodalJEPA:
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
        freeze=l_cfg.get("freeze", True),
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
        freeze_encoders=v_cfg.get("freeze", True) and l_cfg.get("freeze", True),
    ).to(device)

    load_multimodal_checkpoint(model, checkpoint_path=model_cfg.get("checkpoint"), map_location="cpu")

    _freeze_modules(
        [
            model.v_encoder,
            model.l_encoder,
            model.v_proj_ctx,
            model.l_proj_ctx,
            model.v_proj_tgt,
            model.l_proj_tgt,
            model.predictor,
        ]
    )
    model.eval()
    return model


def _freeze_modules(modules):
    for module in modules:
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad = False


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
