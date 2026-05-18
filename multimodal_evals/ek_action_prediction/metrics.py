# Implements EK_PLAN_PART3B §7
import torch
import torch.distributed as dist


class LocalClassMeanRecall:
    """Per-class top-k recall with explicit distributed reduction."""

    def __init__(self, num_classes: int, k: int, device: torch.device):
        self.num_classes = max(int(num_classes), 1)
        self.k = int(k)
        self.TP = torch.zeros(self.num_classes, device=device)
        self.FN = torch.zeros(self.num_classes, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.numel() == 0:
            return
        labels = labels.to(torch.long)
        probs = torch.sigmoid(logits.float())
        topk = min(self.k, probs.size(1))
        preds = probs.topk(topk, dim=1).indices
        match = (preds == labels.view(-1, 1)).any(dim=1)
        if match.any():
            self.TP.index_add_(0, labels[match], torch.ones(int(match.sum()), device=self.TP.device))
        if (~match).any():
            self.FN.index_add_(0, labels[~match], torch.ones(int((~match).sum()), device=self.FN.device))

    def compute(self, reduce: bool = True, eps: float = 1e-8):
        TP = self.TP.clone()
        FN = self.FN.clone()
        if reduce and dist.is_available() and dist.is_initialized():
            dist.all_reduce(TP)
            dist.all_reduce(FN)
        denom = TP + FN
        if denom.sum() <= 0:
            return float("nan"), float("nan")
        nch = (denom > 0).sum().clamp_min(1)
        recall = 100.0 * (TP / (denom + eps)).sum() / nch
        acc = 100.0 * TP.sum() / denom.sum().clamp_min(1)
        return float(recall.item()), float(acc.item())

    def reset(self) -> None:
        self.TP.zero_()
        self.FN.zero_()

    def gather_state(self):
        TP = self.TP.clone()
        FN = self.FN.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(TP)
            dist.all_reduce(FN)
        return TP, FN
