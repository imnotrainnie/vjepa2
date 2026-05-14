import math
from functools import partial
from typing import Literal

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_

Modality = Literal["V", "L"]


class MultimodalPredictor(nn.Module):
    """Single predictor for V->V, V->L, L->V, and L->L prediction."""

    def __init__(
        self,
        shared_dim: int = 512,
        predictor_dim: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        img_size: int = 384,
        patch_size: int = 16,
        tubelet_size: int = 2,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.predictor_dim = predictor_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.init_std = init_std

        self.modality_embed = nn.Embedding(2, shared_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, shared_dim))
        self.input_proj = nn.Linear(shared_dim, predictor_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    use_rope=False,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.output_proj = nn.Linear(predictor_dim, shared_dim)

        self.apply(self._init_weights)
        self._rescale_blocks()

    @staticmethod
    def _modality_id(modality: Modality) -> int:
        if modality == "V":
            return 0
        if modality == "L":
            return 1
        raise ValueError(f"Unsupported modality {modality!r}; expected 'V' or 'L'")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            trunc_normal_(module.weight, std=self.init_std)

    def _rescale_blocks(self) -> None:
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, z_ctx: torch.Tensor, ctx_mod: Modality, tgt_mod: Modality, n_tgt: int) -> torch.Tensor:
        if z_ctx.ndim != 3:
            raise ValueError(f"z_ctx must have shape [B, N_ctx, D], got {tuple(z_ctx.shape)}")

        batch_size, n_ctx, dim = z_ctx.shape
        if dim != self.shared_dim:
            raise ValueError(f"Expected shared_dim={self.shared_dim}, got {dim}")

        ctx_mod_id = torch.tensor(self._modality_id(ctx_mod), device=z_ctx.device)
        tgt_mod_id = torch.tensor(self._modality_id(tgt_mod), device=z_ctx.device)
        z_ctx = z_ctx + self.modality_embed(ctx_mod_id).view(1, 1, -1)

        mask_tokens = self.mask_token.expand(batch_size, n_tgt, -1)
        mask_tokens = mask_tokens + self.modality_embed(tgt_mod_id).view(1, 1, -1)

        tokens = torch.cat([z_ctx, mask_tokens], dim=1)
        tokens = self.input_proj(tokens)
        tokens = tokens + self._get_joint_pos_embed(n_ctx, n_tgt, ctx_mod, tgt_mod, tokens.device, tokens.dtype)

        for block in self.blocks:
            tokens = block(tokens, mask=None, attn_mask=None)

        tokens = self.norm(tokens)
        return self.output_proj(tokens[:, n_ctx:, :])

    def _get_joint_pos_embed(
        self,
        n_ctx: int,
        n_tgt: int,
        ctx_mod: Modality,
        tgt_mod: Modality,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx_pos = self._get_pos_embed(n_ctx, ctx_mod, device, dtype)
        tgt_pos = self._get_pos_embed(n_tgt, tgt_mod, device, dtype)
        return torch.cat([ctx_pos, tgt_pos], dim=1)

    def _get_pos_embed(self, seq_len: int, modality: Modality, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if modality == "V":
            grid_size = self.img_size // self.patch_size
            tokens_per_tubelet = grid_size * grid_size
            if seq_len % tokens_per_tubelet == 0:
                grid_depth = seq_len // tokens_per_tubelet
                pos_embed = get_3d_sincos_pos_embed(
                    self.predictor_dim,
                    grid_size,
                    grid_depth,
                    cls_token=False,
                )
                return torch.from_numpy(pos_embed).to(device=device, dtype=dtype).unsqueeze(0)

        position = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.arange(self.predictor_dim, dtype=dtype, device=device)
        div_term = 10000 ** (2 * (div_term // 2) / self.predictor_dim)

        pos_embed = position / div_term
        pos_embed[:, 0::2] = torch.sin(pos_embed[:, 0::2])
        pos_embed[:, 1::2] = torch.cos(pos_embed[:, 1::2])
        return pos_embed.unsqueeze(0)
