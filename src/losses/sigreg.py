import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Args:
        knots: Number of quadrature knots for the Epps-Pulley statistic.
        num_proj: Number of random one-dimensional projections.

    Input shape:
        proj: [T, B, D]
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects [T, B, D], got shape {tuple(proj.shape)}")

        random_proj = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        random_proj = random_proj.div_(random_proj.norm(p=2, dim=0).clamp_min(1e-12))

        t = self.t.to(dtype=proj.dtype)
        phi = self.phi.to(dtype=proj.dtype)
        weights = self.weights.to(dtype=proj.dtype)

        x_t = (proj @ random_proj).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()
