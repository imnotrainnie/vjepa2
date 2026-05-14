import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.multimodal_predictor import Modality, MultimodalPredictor
from src.models.projectors import MultimodalProjectors
from src.models.text_encoder import SigLIPTextEncoder


class MultimodalJEPA(nn.Module):
    """Multimodal JEPA model with frozen video/text encoders and trainable projectors/predictor."""

    def __init__(
        self,
        v_encoder: nn.Module,
        l_encoder: Optional[SigLIPTextEncoder] = None,
        video_dim: int = 768,
        text_dim: int = 1024,
        shared_dim: int = 512,
        projector_hidden_dim: int = 1024,
        predictor_dim: int = 384,
        predictor_depth: int = 12,
        predictor_num_heads: int = 8,
        img_size: int = 384,
        patch_size: int = 16,
        tubelet_size: int = 2,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.v_encoder = v_encoder
        self.l_encoder = l_encoder or SigLIPTextEncoder(freeze=freeze_encoders)

        self.projectors = MultimodalProjectors(
            video_dim=video_dim,
            text_dim=text_dim,
            hidden_dim=projector_hidden_dim,
            shared_dim=shared_dim,
        )
        self.predictor = MultimodalPredictor(
            shared_dim=shared_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )

        if freeze_encoders:
            self.freeze_encoders()

    @property
    def v_proj_ctx(self) -> nn.Module:
        return self.projectors.v_proj_ctx

    @property
    def l_proj_ctx(self) -> nn.Module:
        return self.projectors.l_proj_ctx

    @property
    def v_proj_tgt(self) -> nn.Module:
        return self.projectors.v_proj_tgt

    @property
    def l_proj_tgt(self) -> nn.Module:
        return self.projectors.l_proj_tgt

    def freeze_encoders(self) -> None:
        self.v_encoder.eval()
        for param in self.v_encoder.parameters():
            param.requires_grad = False
        if hasattr(self.l_encoder, "freeze"):
            self.l_encoder.freeze()
        else:
            self.l_encoder.eval()
            for param in self.l_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        batch: Dict[str, object],
        device: Optional[torch.device] = None,
        modality_pair: Optional[Tuple[Modality, Modality]] = None,
    ) -> Dict[str, object]:
        if device is None:
            device = next(self.parameters()).device

        ctx_mod, tgt_mod = modality_pair or (random.choice(("V", "L")), random.choice(("V", "L")))

        z_ctx = self.encode_context(batch, ctx_mod, device)
        z_tgt = self.encode_target(batch, tgt_mod, device)
        z_pred = self.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=z_tgt.size(1))

        return {
            "z_pred": z_pred,
            "z_tgt": z_tgt,
            "z_ctx": z_ctx,
            "ctx_mod": ctx_mod,
            "tgt_mod": tgt_mod,
        }

    def encode_context(self, batch: Dict[str, object], modality: Modality, device: torch.device) -> torch.Tensor:
        if modality == "V":
            video_ctx = batch["video_ctx"].to(device, non_blocking=True)
            with torch.no_grad():
                h_ctx = self.v_encoder(video_ctx)
            return self.v_proj_ctx(h_ctx)

        text_ctx = batch["text_ctx"]
        with torch.no_grad():
            h_ctx = self.l_encoder(text_ctx)
        return self.l_proj_ctx(h_ctx)

    def encode_target(self, batch: Dict[str, object], modality: Modality, device: torch.device) -> torch.Tensor:
        if modality == "V":
            video_tgt = batch["video_tgt"].to(device, non_blocking=True)
            with torch.no_grad():
                h_tgt = self.v_encoder(video_tgt)
            return self.v_proj_tgt(h_tgt)

        text_tgt = batch["text_tgt"]
        with torch.no_grad():
            h_tgt = self.l_encoder(text_tgt)
        return self.l_proj_tgt(h_tgt)


def build_vjepa2_1_vitb_encoder(
    checkpoint_path: Optional[str] = None,
    img_size: int = 384,
    num_frames: int = 32,
    patch_size: int = 16,
    tubelet_size: int = 2,
) -> nn.Module:
    from src.hub.backbones import _clean_backbone_key
    from src.utils.checkpoint_loader import robust_checkpoint_loader

    try:
        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        encoder_kwargs = {
            "img_size": (img_size, img_size),
            "patch_size": patch_size,
            "num_frames": num_frames,
            "tubelet_size": tubelet_size,
            "use_sdpa": True,
            "use_rope": True,
            "img_temporal_dim_size": 1,
            "interpolate_rope": True,
        }
    except ModuleNotFoundError:
        from src.models import vision_transformer as vit_encoder

        encoder_kwargs = {
            "img_size": (img_size, img_size),
            "patch_size": patch_size,
            "num_frames": num_frames,
            "tubelet_size": tubelet_size,
            "use_sdpa": True,
            "use_rope": True,
        }

    encoder = vit_encoder.vit_base(**encoder_kwargs)

    if checkpoint_path:
        checkpoint = robust_checkpoint_loader(checkpoint_path, map_location="cpu")
        for key in ("encoder", "ema_encoder", "target_encoder"):
            if key in checkpoint:
                encoder.load_state_dict(_clean_backbone_key(checkpoint[key]), strict=False)
                break
        else:
            encoder.load_state_dict(_clean_backbone_key(checkpoint), strict=False)

    return encoder
