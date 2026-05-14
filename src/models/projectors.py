import torch.nn as nn


def make_projector(input_dim: int, hidden_dim: int = 1024, output_dim: int = 512) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class MultimodalProjectors(nn.Module):
    """Independent context/target projectors for video and language tokens."""

    def __init__(
        self,
        video_dim: int = 768,
        text_dim: int = 1024,
        hidden_dim: int = 1024,
        shared_dim: int = 512,
    ):
        super().__init__()
        self.v_proj_ctx = make_projector(video_dim, hidden_dim, shared_dim)
        self.l_proj_ctx = make_projector(text_dim, hidden_dim, shared_dim)
        # self.v_proj_tgt = make_projector(video_dim, hidden_dim, shared_dim)
        # self.l_proj_tgt = make_projector(text_dim, hidden_dim, shared_dim)
        # 2. 【核心修改】直接共享权重：让 Tgt 指针指向 Ctx
        self.v_proj_tgt = self.v_proj_ctx
        self.l_proj_tgt = self.l_proj_ctx
