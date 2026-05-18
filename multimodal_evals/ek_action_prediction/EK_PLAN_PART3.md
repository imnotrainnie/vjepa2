# 多模态 V-JEPA EK 动作预测下游任务 实施蓝图 (Part 3 / 4)
> 本部分：**训练循环、四象限评估循环、DDP 接线。**
> 配套模块（`metrics.py` / `optim.py` / `checkpoint.py`）放在 Part 3B 详述。

---

## 5. 训练循环 `train_one_epoch(...)`（重写）

### 5.1 函数签名
```python
def train_one_epoch(
    epoch: int,
    model: MultimodalJEPA,
    classifiers: List[nn.Module],          # DDP-wrapped 或原始；都接受
    optimizers: List[torch.optim.Optimizer],
    schedulers: List["WarmupCosineLRSchedule"],
    wd_schedulers: List["CosineWDSchedule"],
    scalers: List[Optional[torch.cuda.amp.GradScaler]],
    data_loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
    verb_map: Dict[int, int],
    noun_map: Dict[int, int],
    action_map: Dict[Tuple[int, int], int],
    criterion: nn.Module,
    log_interval: int,
    rank: int,
) -> Dict[str, float]:
```

### 5.2 单次迭代伪代码（**逐行严格按此实现**）
```text
for itr, batch in enumerate(data_loader):
    for s in schedulers:    s.step()
    for w in wd_schedulers: w.step()

    # === 标签预计算 ============================================================
    verb_labels   = torch.tensor([verb_map[int(v)]   for v in batch["verb_class"]], device=device)
    noun_labels   = torch.tensor([noun_map[int(n)]   for n in batch["noun_class"]], device=device)
    action_labels = torch.tensor(
        [action_map[(int(v), int(n))] for v, n in zip(batch["verb_class"], batch["noun_class"])],
        device=device,
    )

    pair_indices = split_by_pair(batch["ctx_mod"], batch["tgt_mod"])
    B_total = sum(len(idx) for idx in pair_indices.values())
    if B_total == 0: continue

    for opt in optimizers: opt.zero_grad(set_to_none=True)

    pair_loss_log = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
        for pair, indices in pair_indices.items():
            if not indices: continue
            ctx_mod, tgt_mod = pair
            idx_tensor = torch.tensor(indices, device=device)

            # === ① 骨干前向：no_grad 强制（参 Part 2 § 4.2） ===================
            concat_feat = compute_concat_feat(model, batch, indices, ctx_mod, tgt_mod, device)
            v_lbl = verb_labels.index_select(0, idx_tensor)
            n_lbl = noun_labels.index_select(0, idx_tensor)
            a_lbl = action_labels.index_select(0, idx_tensor)

            # === ② 任务头前向 + 三损失 ==========================================
            for clf in classifiers:
                logits = clf(concat_feat)
                loss = (criterion(logits["verb"],   v_lbl)
                      + criterion(logits["noun"],   n_lbl)
                      + criterion(logits["action"], a_lbl))
                w = len(indices) / B_total          # § 4.4 子批权重
                (loss * w).backward()               # 累积梯度
                pair_loss_log[pair] = loss.detach().item()

    for opt in optimizers: opt.step()

    if itr % log_interval == 0 and rank == 0:
        msg = " | ".join(f"{p[0]}->{p[1]}: {v:.3f}" for p, v in pair_loss_log.items())
        logger.info(f"[Train E{epoch} it{itr}] {msg}")
```

### 5.3 辅助函数：`compute_concat_feat(model, batch, indices, ctx_mod, tgt_mod, device)`
- **必须**用 `torch.no_grad()` 包裹（参 Part 2 § 4.2）。
- 返回形状 `[b, N_ctx + n_tgt, 512]` 的张量，最后 `.detach()` 一次（双保险）。

```text
def compute_concat_feat(model, batch, indices, ctx_mod, tgt_mod, device):
    sub = select_batch(batch, indices)
    with torch.no_grad():
        z_ctx  = model.encode_context(sub, ctx_mod, device)
        z_tgt  = model.encode_target (sub, tgt_mod, device)
        z_pred = model.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=z_tgt.size(1))
        feat = torch.cat([z_ctx, z_pred], dim=1).detach()
    return feat
```

### 5.4 与 bfloat16 / GradScaler 的关系
- bfloat16 训练**不需要** `GradScaler`，传 `scalers=[None]` 即可。
- `use_bfloat16=False` 时 fp32 训练，依旧 `scalers=[None]`（任务头很小，没必要混合精度）。
- **不要**在 `compute_concat_feat` 内再嵌套 autocast；外层 `torch.autocast` 已涵盖。

---

## 6. 评估循环 `evaluate(...)`（重写）

### 6.1 函数签名
```python
def evaluate(
    model: MultimodalJEPA,
    classifiers: List[nn.Module],
    data_loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
    verb_map: Dict[int, int],
    noun_map: Dict[int, int],
    action_map: Dict[Tuple[int, int], int],
    rank: int,
) -> Dict[str, Dict[str, float]]:
```

### 6.2 四象限独立指标
- 维护 `metrics_by_pair[pair][head]`，`pair ∈ {(V,V),(V,L),(L,V),(L,L)}`，`head ∈ {verb,noun,action}`。
- 每个值是一个 `LocalClassMeanRecall(num_classes=K[head], k=5, device=device)`（见 Part 3B § 7）。
- **不要**用 `action_anticipation_frozen/metrics.py::ClassMeanRecall`：它 `__call__` 内就 `all_reduce`，
  与"按象限本地累计、末尾一次性 reduce" 的目标冲突。

### 6.3 伪代码
```text
for clf in classifiers: clf.train(False)
K = {"verb": len(verb_map), "noun": len(noun_map), "action": len(action_map)}
metrics = {pair: {h: LocalClassMeanRecall(K[h], 5, device) for h in K} for pair in PAIR_NAMES}

with torch.no_grad():
    for batch in data_loader:
        pair_indices = split_by_pair(batch["ctx_mod"], batch["tgt_mod"])
        verb_lbl   = torch.tensor([verb_map[int(v)] for v in batch["verb_class"]], device=device)
        noun_lbl   = torch.tensor([noun_map[int(n)] for n in batch["noun_class"]], device=device)
        action_lbl = torch.tensor(
            [action_map[(int(v), int(n))] for v, n in zip(batch["verb_class"], batch["noun_class"])],
            device=device,
        )
        with torch.autocast(device.type, torch.bfloat16, enabled=use_bfloat16):
            for pair, indices in pair_indices.items():
                if not indices: continue
                ctx_mod, tgt_mod = pair
                idx = torch.tensor(indices, device=device)
                concat_feat = compute_concat_feat(model, batch, indices, ctx_mod, tgt_mod, device)
                logits = classifiers[0](concat_feat)
                metrics[pair]["verb"  ].update(logits["verb"],   verb_lbl.index_select(0, idx))
                metrics[pair]["noun"  ].update(logits["noun"],   noun_lbl.index_select(0, idx))
                metrics[pair]["action"].update(logits["action"], action_lbl.index_select(0, idx))

report = {}
for pair, head_map in metrics.items():
    name = PAIR_NAMES[pair]
    report[name] = {}
    for head, m in head_map.items():
        recall, acc = m.compute(reduce=True)            # 内部一次性 all_reduce
        report[name][head] = {"recall@5": recall, "acc@5": acc}
        if rank == 0:
            logger.info(f"[VAL] {name:5s} {head:6s}: recall@5={recall:6.2f}  acc@5={acc:6.2f}")
return report
```

### 6.4 打印格式（强约束，仅 rank==0）
- 共 12 行：四象限 × {verb, noun, action}，例：
  `[VAL] V->V verb  : recall@5= 12.34  acc@5= 56.78`
- 缺失象限（验证集太小、某象限未覆盖）：打印 `recall@5=  nan  acc@5=  nan`，**不要** `raise`。

---

## 10. DDP 接线（`main` 中的关键步骤）

### 10.1 进程组初始化
- 沿用 `src.utils.distributed.init_distributed`；
- 若 `world_size, rank == 1, 0` 且 `dist` 未 init，则在 `ensure_distributed()` 内构造一个
  `gloo`/`nccl` 单进程组（已存在的实现保留），保证 `dist.all_reduce` 在 metrics 中可用。
- **关键修正**：原 `eval.py::ensure_distributed` 中 `init_process_group(backend, rank=0, world_size=1)`
  必须配套设置 `os.environ["MASTER_ADDR"/"MASTER_PORT"]`，且仅在 `not dist.is_initialized()` 时调用。

### 10.2 设备与 DDP 包装
```text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): torch.cuda.set_device(device)
model = init_module(cfg["model"], device)             # frozen, 不包 DDP
classifiers = init_classifier(...)
if world_size > 1:
    classifiers = [DistributedDataParallel(c, device_ids=[device.index], static_graph=True)
                   for c in classifiers]
```
- **强制**：`model` 不要 DDP；任何 `requires_grad=False` 的模块进 DDP 都会触发
  "DDP find_unused_parameters=False but has no grads" 报错。
- `static_graph=True` 与 V-JEPA 原版保持一致；分类头无条件分支。

### 10.3 取参 / 优化器
- 用本目录 `optim.init_opt(classifiers, opt_kwargs, iterations_per_epoch, num_epochs, use_bfloat16)`；
- `param_groups` 中的 `mc_*` 字段全部由 yaml `optimization.multihead_kwargs[0]` 注入（仅 1 个 head 时也用 list 包一层）。
