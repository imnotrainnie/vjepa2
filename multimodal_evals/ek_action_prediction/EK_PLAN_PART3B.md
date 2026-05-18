# 多模态 V-JEPA EK 动作预测下游任务 实施蓝图 (Part 3B / 4)
> 本部分：**新建辅助模块的精确实现 (`metrics.py` / `optim.py` / `checkpoint.py`)。**

---

## 7. 新建 `metrics.py`：`LocalClassMeanRecall`

### 7.1 完整伪代码
```python
import torch
import torch.distributed as dist
import torch.nn.functional as F


class LocalClassMeanRecall:
    """Per-class top-k Recall metric with explicit reduce step.

    与 multimodal_evals.action_anticipation_frozen.metrics.ClassMeanRecall 的区别：
      * 不在 update 内 all_reduce；
      * 提供 compute(reduce=True/False)，用于在 evaluate 末尾一次性汇总四象限指标。
    """

    def __init__(self, num_classes: int, k: int, device: torch.device):
        self.num_classes = max(int(num_classes), 1)
        self.k = int(k)
        self.TP = torch.zeros(self.num_classes, device=device)
        self.FN = torch.zeros(self.num_classes, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # logits: [b, C]; labels: [b]
        if logits.numel() == 0:
            return
        probs = torch.sigmoid(logits.float())                    # 保持与上游 ClassMeanRecall 一致
        topk = min(self.k, probs.size(1))
        preds = probs.topk(topk, dim=1).indices                  # [b, k]
        match = (preds == labels.view(-1, 1)).any(dim=1)         # [b]
        ones_tp = torch.ones(int(match.sum()),  device=self.TP.device)
        ones_fn = torch.ones(int((~match).sum()), device=self.FN.device)
        if ones_tp.numel() > 0:
            self.TP.index_add_(0, labels[ match].to(torch.long), ones_tp)
        if ones_fn.numel() > 0:
            self.FN.index_add_(0, labels[~match].to(torch.long), ones_fn)

    def compute(self, reduce: bool = True, eps: float = 1e-8):
        TP, FN = self.TP.clone(), self.FN.clone()
        if reduce and dist.is_available() and dist.is_initialized():
            dist.all_reduce(TP)
            dist.all_reduce(FN)
        denom = TP + FN
        nch = (denom > 0).sum().clamp_min(1)
        if denom.sum() <= 0:
            return float("nan"), float("nan")
        recall = 100.0 * (TP / (denom + eps)).sum() / nch
        acc = 100.0 * TP.sum() / denom.sum().clamp_min(1)
        return float(recall.item()), float(acc.item())

    def reset(self) -> None:
        self.TP.zero_(); self.FN.zero_()
```

### 7.2 行为约束
- `update` 入参 `logits` 已经是分类头输出，**严禁**在内部 softmax；保留与上游一致的 sigmoid + top-k。
- `match` 中 `labels` 必须为 `torch.long`（label 来自 verb_map / noun_map / action_map）。
- 空象限调用 `compute()` 时返回 `(nan, nan)`，让 evaluate 顺利打印。

---

## 8. 新建 `optim.py`：复制最小化 LR/WD 调度

### 8.1 复制要点（从 `multimodal_evals/action_anticipation_frozen/utils.py`）
- 完整复制三个类 / 函数：`init_opt`、`WarmupCosineLRSchedule`、`CosineWDSchedule`。
- 顶部增加：
  ```python
  __all__ = ["init_opt", "WarmupCosineLRSchedule", "CosineWDSchedule"]
  ```
- **不要**通过 `from multimodal_evals.action_anticipation_frozen.utils import ...` 引用——
  Codex 必须保留本目录自包含，避免上游目录将来下线。
- `init_opt` 内 `torch.cuda.amp.GradScaler()` 仅在 `use_bfloat16=True` 时被构造，但对于 bf16
  实际上 scaler 不会被调用（参 Part 3 § 5.4）；可保留以兼容签名，不要简化掉。

### 8.2 入参字典对齐 yaml
- `opt_kwargs[i]` 必含字段：
  - `weight_decay` (= `ref_wd`)
  - `final_weight_decay` (= `final_wd`)
  - `start_lr`
  - `lr` (= `ref_lr`)
  - `final_lr`
  - `warmup` (epochs 单位)
- yaml 中以 `optimization.multihead_kwargs: [{...}]` 提供（即便只有 1 个分类头，也用 list 包一层）。

---

## 9. 新建 `checkpoint.py`：仅保存任务头

### 9.1 完整伪代码
```python
import os, torch

def _strip_ddp(state_dict):
    out = {}
    for k, v in state_dict.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out

def save_classifier_checkpoint(path, classifiers, optimizers, epoch, rank, world_size):
    if rank != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "world_size": int(world_size),
        "classifiers": [_strip_ddp(c.state_dict()) for c in classifiers],
        "optimizers": [o.state_dict() for o in optimizers],
    }
    torch.save(payload, path)

def load_classifier_checkpoint(path, classifiers, optimizers, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    for c, sd in zip(classifiers, ckpt["classifiers"]):
        # DDP-wrapped classifier 接受裸 key（已在保存时 strip 过）
        target = c.module if hasattr(c, "module") else c
        target.load_state_dict(sd, strict=True)
    if optimizers is not None:
        for o, sd in zip(optimizers, ckpt["optimizers"]):
            o.load_state_dict(sd)
    return ckpt.get("epoch", 0)
```

### 9.2 调用约束
- 保存时机：`if (epoch+1) % save_interval == 0`；同时维护 `latest.pt`（覆盖写）。
- **任务头自身**用 `strict=True` 加载；与上游 `MultimodalJEPA` 的 `strict=False` 区分开。
- `_strip_ddp` 是必需的：训练时 classifier 被 `DistributedDataParallel` 包装，state_dict key 前缀为 `module.`，
  否则单卡 / 多卡 ckpt 不兼容。

---

## 10. 模块导入路径汇总（防止循环 / 拼写错误）

```python
# 新文件之间
from multimodal_evals.ek_action_prediction.metrics import LocalClassMeanRecall
from multimodal_evals.ek_action_prediction.optim import (
    init_opt, WarmupCosineLRSchedule, CosineWDSchedule,
)
from multimodal_evals.ek_action_prediction.checkpoint import (
    save_classifier_checkpoint, load_classifier_checkpoint,
)

# 复用上游
from multimodal_evals.action_anticipation_frozen.models import AttentiveClassifier  # 只读引用
from src.models.multimodal_jepa import MultimodalJEPA, build_vjepa2_1_vitb_encoder
from src.models.text_encoder import SigLIPTextEncoder
from src.utils.checkpoint_loader import load_multimodal_checkpoint
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, get_logger
```

> 注：现存 `eval.py` 已用 `from multimodal_evals.action_anticipation_frozen.metrics import ClassMeanRecall`，
> **请在重写时移除该 import**，改用 `LocalClassMeanRecall`。
