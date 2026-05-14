from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.sigreg import SIGReg


def prepare_sigreg_input(z_ctx: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Convert context and target projections from [B, N, D] to [T, B, D]."""
    if z_ctx.ndim != 3 or z_tgt.ndim != 3:
        raise ValueError("z_ctx and z_tgt must both have shape [B, N, D]")
    if z_ctx.size(0) != z_tgt.size(0) or z_ctx.size(-1) != z_tgt.size(-1):
        raise ValueError("z_ctx and z_tgt must share batch size and feature dimension")

    return torch.cat([z_ctx.transpose(0, 1), z_tgt.transpose(0, 1)], dim=0)


class MultimodalLoss(nn.Module):
    """MSE prediction loss plus SIGReg distribution regularization."""

    def __init__(self, lambda_sigreg: float = 0.1, sigreg_knots: int = 17, sigreg_num_proj: int = 1024):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def forward(self, z_pred: torch.Tensor, z_tgt: torch.Tensor, z_ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_mse = F.mse_loss(z_pred, z_tgt.detach())
        loss_sigreg = self.sigreg(prepare_sigreg_input(z_ctx, z_tgt))
        loss = loss_mse + self.lambda_sigreg * loss_sigreg

        return {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_sigreg": loss_sigreg,
        }
