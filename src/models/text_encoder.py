from typing import Iterable, Optional

import torch
import torch.nn as nn


class SigLIPTextEncoder(nn.Module):
    """Frozen SigLIP text encoder wrapper returning sequence features."""

    def __init__(
        self,
        model_name: str = "google/siglip-large-patch16-384",
        max_length: int = 64,
        freeze: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoTokenizer, SiglipTextModel
        except ImportError as exc:
            raise ImportError("SigLIPTextEncoder requires the `transformers` package.") from exc

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.encoder = SiglipTextModel.from_pretrained(model_name, local_files_only=local_files_only)

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def tokenize(self, text: Iterable[str], device: Optional[torch.device] = None):
        tokens = self.tokenizer(
            list(text),
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            tokens = {key: value.to(device) for key, value in tokens.items()}
        return tokens

    def forward(self, text: Iterable[str]) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        tokens = self.tokenize(text, device=device)
        return self.encoder(**tokens).last_hidden_state
