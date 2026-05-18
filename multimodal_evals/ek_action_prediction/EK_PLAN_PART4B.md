# 多模态 V-JEPA EK 动作预测下游任务 实施蓝图 (Part 4B / 4)
> 本部分：**给 Codex 的执行指令 (Discipline Checklist)。**

---

## 13. 给 Codex 的执行指令 (Discipline Checklist)

> 实现中遇到本节没明确的细节时，**遵从 IMPLEMENTATION_PLAN.md 的整体风格**，不要自行发明新约定。

### 13.1 加载逻辑硬要求
- **骨干 ckpt**：
  ```python
  load_multimodal_checkpoint(
      model,
      checkpoint_path=cfg["model"]["checkpoint"],
      strict=False,
      map_location="cpu",
  )
  ```
  `strict=False` **不可更改**，理由：
  - `text_encoder` 已在 `SigLIPTextEncoder.__init__` 内从 HuggingFace 加载完毕；ckpt 内 `l_encoder.encoder.*`
    为冗余覆盖，匹配即可。
  - 实测 ckpt 顶层 prefix：`v_encoder.* / l_encoder.encoder.* / projectors.* / predictor.*`。
    任何升级版 multimodal_jepa.py 的字段变化都依赖 `strict=False` 才不会崩溃。
- 加载完成后 rank==0 必须打印：
  ```
  [load_multimodal_checkpoint] missing_keys=K1, unexpected_keys=K2
  [load_multimodal_checkpoint] missing[:20] = [...]
  [load_multimodal_checkpoint] unexpected[:20] = [...]
  ```
- **任务头 ckpt**：`load_classifier_checkpoint(...)` 内部 `strict=True`。任务头自身权重不允许丢字段。

### 13.2 `torch.no_grad()` 范围（再次强调，最高优先级）
- **必须**包裹的调用：
  - `model.encode_context(sub, ctx_mod, device)`
  - `model.encode_target(sub, tgt_mod, device)`
  - `model.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=...)`
  - 紧跟其后的 `torch.cat([z_ctx, z_pred], dim=1)` 与 `.detach()`
- **必须**排除的调用：
  - `classifier(concat_feat)` 前向；
  - `criterion(logits[...], labels)` 损失计算；
  - `loss.backward()` 与 `optimizer.step()`。
- 写作模式（再贴一次，**Codex 严格照搬**）：
  ```python
  with torch.no_grad():
      z_ctx  = model.encode_context(sub, ctx_mod, device)
      z_tgt  = model.encode_target (sub, tgt_mod, device)
      z_pred = model.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=z_tgt.size(1))
      concat_feat = torch.cat([z_ctx, z_pred], dim=1).detach()
  # ↓↓ 退出 no_grad，此处恢复梯度图
  logits = classifier(concat_feat)
  ```
- 不要"整段 forward 一锅 `torch.no_grad()`"——分类头会被同时阻断梯度，loss 不会下降；
- 不要在 `no_grad` 内 `enable_grad()` 局部反转：`requires_grad=False` 的叶子参数已经无法产生梯度，
  这种写法只制造混淆。

### 13.3 冻结自检（启动时一次）
```python
assert all(not p.requires_grad for p in model.v_encoder.parameters())
assert all(not p.requires_grad for p in model.l_encoder.parameters())
assert all(not p.requires_grad for p in model.projectors.parameters())
assert all(not p.requires_grad for p in model.predictor.parameters())
assert any(p.requires_grad for c in classifiers for p in c.parameters())
```
- 自检失败必须 `raise RuntimeError(...)`。

### 13.4 DDP 准则
- `MultimodalJEPA` **永远不**包 DDP（无可训练参数）。
- `AttentiveClassifier` 仅在 `world_size > 1` 时 DDP，`static_graph=True`，不设 `find_unused_parameters`。
- 跨进程指标聚合统一在 `LocalClassMeanRecall.compute(reduce=True)` 内一次完成；
  **禁止**在外层手动 `dist.all_reduce` 同一对 `(TP, FN)`。

### 13.5 AMP / dtype 准则
- 训练 / 评估前向：`with torch.autocast(device.type, torch.bfloat16, enabled=use_bfloat16):` 包裹。
- 不要嵌套 autocast；尤其不要在 `compute_concat_feat` 内再起一层。
- `LocalClassMeanRecall.update` 内 `logits.float()` 后再 sigmoid，避免 bf16 top-k 不稳。

### 13.6 多对子批权重（防止象限失衡）
- 每个 (ctx, tgt) pair 子批反向时按 `w = len(indices)/B_total` 缩放损失，再 `(loss * w).backward()`。
- **禁止**改写为"每对 pair 各自 `optimizer.step()`"——会让 V→V 与 L→L 的更新步骤不一致。

### 13.7 日志 / 落盘准则
- 任何 `[Train ...] / [VAL ...]` 行仅 rank==0 打印；其它 rank `logger.setLevel(logging.ERROR)`。
- `folder/tag/log_r{rank}.csv`（CSVLogger）可选实现：本任务**可暂缓**，但要在 main 内 `if rank == 0: os.makedirs(folder, exist_ok=True)`。
- `latest.pt` 仅 rank==0 写盘；其它 rank `return` 立即返回。
- `save_interval` 不再额外保留历史 ckpt（`epoch_{e}.pt`），覆盖写 `latest.pt` 即可。

### 13.8 数据健壮性
- `EKMultimodalDataset.__getitem__` 中：若 `text_state_context` 或 `text_state_target` 经 `clean_text`
  后为空字符串，**保留**继续传入 tokenizer——`SigLIPTextEncoder` 接受空串（pad 出空白序列）。
- `build_label_maps` 必须用 `train_dataset.samples`（整套数据），不要用 `split_samples`，避免 val 出现训练未见类。

### 13.9 风格 / 禁忌
- **禁止**改动 `/data/vjepa2/src/**` 与 `/data/vjepa2/multimodal_evals/action_anticipation_frozen/**`。
- **禁止**在 `init_module` 内对 `MultimodalJEPA` 做 DDP 包装或 `.cuda()` 兜底（device 由 yaml/main 决定）。
- **禁止**在 train 循环里 `model.train()`——骨干永远 `eval()`。
- **禁止**新增对 `evals.*` 的依赖；本任务全程使用 `multimodal_evals.*`。
- **禁止**用 `try / except KeyError` 静默吞掉 `action_map[...]` 缺失；显式 `raise` 并提示扩大训练集。
- **禁止**新建 `__init__.py` 之外的额外公共导出。
- **禁止**在 train 路径中遗漏 `.detach()`——这是分类头梯度污染骨干的最大风险点。

### 13.10 自验脚本（Codex 实施完毕后必须跑通）
1. 单 GPU 干跑：
   ```bash
   python -m multimodal_evals.ek_action_prediction.main \
          --fname multimodal_evals/ek_action_prediction/ek_action_prediction.yaml \
          --devices cuda:0 --debugmode True
   ```
   - 期待：第一个 `[Train E0 it0]` 出现；每个 epoch 结束打印 12 行 `[VAL ...]`；不报 NaN / OOM。
2. 多 GPU 干跑（如可用）：去掉 `--debugmode True`，`--devices cuda:0 cuda:1`。
3. 仅推断：`--val_only` 或 yaml `val_only: true`，必须仅打印 12 行 `[VAL ...]` 后退出。
4. 续训：先训练若干 epoch 留下 `latest.pt`，将 yaml `resume: true` 重启，确认从中间 epoch 接续。

### 13.11 交付物清单
- 修改：`dataset.py`、`models.py`、`eval.py`、`ek_action_prediction.yaml`。
- 新建：`metrics.py`、`optim.py`、`checkpoint.py`、`main.py`。
- 文档：本套 `EK_PLAN_PART{1,2,3,3B,4A,4B}.md` 已在仓库内，Codex 不需要再生成。
- 每个修改 / 新建 `.py` 文件的首行必须写：
  ```python
  # Implements EK_PLAN_PART{X} §{Y}
  ```
  以便后续 review 定位。

---

## 14. 完结声明
- 上述蓝图已覆盖：文件系统规划、模型加载、张量形状、训练 / 评估循环、四象限指标、DDP 接线、
  YAML 配置、入口装配以及 Codex 的纪律准则。
- Codex 在任何"不确定要不要这么做"的位置，回到本文档对应 §，照搬即可；
  若仍模糊，**优先牺牲灵活性，确保前向无梯度泄漏与四象限指标正确**。
