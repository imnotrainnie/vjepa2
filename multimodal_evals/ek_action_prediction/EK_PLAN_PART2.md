# 多模态 V-JEPA EK 动作预测下游任务 实施蓝图 (Part 2 / 4)
> 本部分：**模型加载、冻结策略、AttentiveClassifier 头的精确接线、Forward 张量流转。**
> 仍只允许修改 `/data/vjepa2/multimodal_evals/ek_action_prediction/` 下文件。

---

## 3. 预训练模型加载 (`models.py`)

### 3.1 `init_module(model_cfg, device) -> MultimodalJEPA`
完善要点（在已有实现上叠加，保持函数签名兼容 `eval.py`）：

```text
# 步骤 1：按 yaml 构造 v_encoder / l_encoder
#   - build_vjepa2_1_vitb_encoder(checkpoint_path=v_cfg["checkpoint_path"], ...)
#   - SigLIPTextEncoder(model_name=l_cfg["model_name"], max_length=l_cfg["max_length"], freeze=True)
# 步骤 2：构造 MultimodalJEPA(...).to(device)
#   - shared_dim / hidden_dim / predictor_dim / predictor_depth / num_heads 全部从 yaml 注入
#   - freeze_encoders=True
# 步骤 3：load_multimodal_checkpoint(model, checkpoint_path=model_cfg["checkpoint"],
#                                   strict=False, map_location="cpu")
#   - strict=False 是【硬性要求】：
#       * 上游 v_encoder ckpt 已在 build_vjepa2_1_vitb_encoder 内部 strict=False 加载，
#         本次再次以 strict=False 加载全图（含 projector + predictor）；
#       * l_encoder.encoder.* 来自 huggingface 预下载，二次写入只是冗余覆盖，无害。
#   - 必须在 rank==0 打印 missing / unexpected keys（截断到前 20 条），用于调试。
# 步骤 4：_freeze_modules([v_encoder, l_encoder, projectors.{v|l}_proj_{ctx|tgt}, predictor])
#   - 每个 module .eval(); for p in module.parameters(): p.requires_grad=False
# 步骤 5：model.eval()  且 return model（保持 nn.Module，不要 DDP 包装）。
#   - 关键：分类头之外的所有前向必须在 with torch.no_grad(): 内调用（参 § 4.2）。
```

> **绝对禁止**：把 `MultimodalJEPA` 包进 `DistributedDataParallel`。它没有可训练参数，DDP 反而会触发
> "no grads received" 报错。

### 3.2 `init_classifier(...) -> List[AttentiveClassifier]`
- 复用 `multimodal_evals.action_anticipation_frozen.models.AttentiveClassifier`。
- 入参约束（与 yaml 对齐）：
  - `embed_dim = projectors.shared_dim = 512`
  - `num_heads = classifier.num_heads`（建议 8 或 16，必须能整除 512）
  - `depth = classifier.num_blocks`（默认 2）
  - `verb_classes / noun_classes / action_classes` 三个 dict，长度即类别数。
- 返回 `[AttentiveClassifier(...).to(device)]`（**当前只用 1 个分类头**，预留 list 接口给未来 ensemble）。

### 3.3 `_freeze_modules` 不变；新增 `_log_trainable_params(model, classifiers, rank)`：
```text
if rank == 0:
    n_frozen = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for c in classifiers for p in c.parameters() if p.requires_grad)
    logger.info(f"Frozen backbone params={n_frozen:,}; trainable head params={n_train:,}")
    assert all(not p.requires_grad for p in model.parameters()), "backbone must be frozen"
```

---

## 4. Forward 张量流转 (训练 / 推断共用)

### 4.1 单步前向（已 split 同象限 sub-batch 后）
设 `pair=(ctx_mod, tgt_mod)`，子批样本数 `b=len(indices)`，统一记 `D=512`：

```text
# ① 装载子批（CPU/GPU 切换） ----------------------------------------------------
sub = select_batch(batch, indices)
#   sub["video_ctx"]: [b, 3, 32, H, W]   (仅 ctx_mod=='V' 时使用)
#   sub["video_tgt"]: [b, 3,  2, H, W]   (仅 tgt_mod=='V' 时使用)
#   sub["text_ctx"]:  List[str] (b)
#   sub["text_tgt"]:  List[str] (b)

# ② 冻结骨干前向（梯度严格阻断） ----------------------------------------------
with torch.no_grad():
    z_ctx = model.encode_context(sub, ctx_mod, device)
    #   if ctx_mod=='V':  h_ctx=[b, 9216, 768] -> v_proj_ctx -> z_ctx=[b, 9216, 512]
    #   if ctx_mod=='L':  h_ctx=[b,   64,1024] -> l_proj_ctx -> z_ctx=[b,   64, 512]

    z_tgt = model.encode_target(sub, tgt_mod, device)
    #   if tgt_mod=='V':  z_tgt=[b, 576, 512]    (注：n_tgt 取决于 num_frames_tgt=2)
    #   if tgt_mod=='L':  z_tgt=[b,  64, 512]

    n_tgt = z_tgt.size(1)
    z_pred = model.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=n_tgt)
    #   z_pred: [b, n_tgt, 512]

# ③ 任务头输入构造 ------------------------------------------------------------
concat_feat = torch.cat([z_ctx, z_pred], dim=1)
#   shape: [b, N_ctx + n_tgt, 512]
#   N_ctx + n_tgt 可能取值（按 ctx,tgt）：
#       V,V → 9216 + 576 = 9792
#       V,L → 9216 +  64 = 9280
#       L,V →   64 + 576 = 640
#       L,L →   64 +  64 = 128
#   注意：该张量序列长度差异极大，AttentivePooler 用 cross-attn 不会受影响，
#         但显存峰值出现在 V→V，应据此设置 batch_size。

# ④ 分类头前向（仅本步要计算梯度） --------------------------------------------
logits = classifier(concat_feat)
#   返回 dict: {"verb": [b, V], "noun": [b, N], "action": [b, A]}
#   其中 V=len(verb_map), N=len(noun_map), A=len(action_map)

return logits
```

### 4.2 `torch.no_grad()` 包裹范围（强约束）
- **必须包裹**：`encode_context`, `encode_target`, `predictor`。
- **必须排除**：`classifier(concat_feat)`（要训练）以及任何 loss / backward 调用。
- **不要**把整段前向放进 `with torch.no_grad():`。Codex 常见错误：把 classifier 也一起包了，导致 loss 不更新。
- 也**不要**只对 classifier 单独 `enable_grad()`：`no_grad` 上下文管理器即便在内部 `enable_grad()` 仍会被骨干层的 `requires_grad=False` 阻断，但建议显式分两段写：

```text
# 推荐写法（防错）
with torch.no_grad():
    z_ctx  = model.encode_context(sub, ctx_mod, device)
    z_tgt  = model.encode_target(sub, tgt_mod, device)
    z_pred = model.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=z_tgt.size(1))
    concat_feat = torch.cat([z_ctx, z_pred], dim=1).detach()
# 退出 no_grad 后再过分类头：
logits = classifier(concat_feat)
```

> 注：`.detach()` 是冗余但安全的双保险；不会影响数值，避免误启分支 backward。

### 4.3 AttentiveClassifier 数值健壮性
- 已有 `AttentiveClassifier` 在 forward 入口检查 NaN 并 `exit(1)`：保留行为；
  但在 bfloat16 下偶发 NaN，**建议**在 Codex 实现里改为 `if torch.isnan(x).any(): raise RuntimeError(...)`。
  这是 Part 4 中的 "纪律准则" 之一，本处先记一个 TODO。

### 4.4 多对子批的合并策略（关键）
- 一个 `DataLoader` 取出的 `batch` 同时含 4 种象限的样本（顺序由 `__getitem__` 决定）；
  实际可能某个象限的 indices 为空（极小批时）。
- **必须**按 `pair_indices = split_by_pair(ctx_mods, tgt_mods)` 拆分后逐象限 forward + loss：
  - 不同象限的 `concat_feat` 序列长度不同，**不能**强行 stack。
  - 每个象限 loss 单独累加到 `total_loss`，最后 `optimizer.step()` 一次。
- 反向传播采用"累积梯度"模式：

```text
optimizer.zero_grad(set_to_none=True)
total_loss = 0
for pair, indices in pair_indices.items():
    if not indices: continue
    logits = forward_one_pair(...)         # 上述 ①-④
    loss   = ce(verb) + ce(noun) + ce(action)        # § 5
    (loss * (b / B)).backward()            # 按子批权重缩放，避免大象限被淹没
    total_loss += loss.detach() * b
optimizer.step()
```
其中 `B = sum(len(idx) for idx in pair_indices.values())`，保证四象限合并后是一次"等价批"更新。
