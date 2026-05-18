# 多模态 V-JEPA EK 动作预测下游任务 实施蓝图 (Part 1 / 4)

> 适用范围：仅完善 `/data/vjepa2/multimodal_evals/ek_action_prediction/` 下的代码。
> 上游约束：所有 V-JEPA / multimodal_jepa / 数据集 / 损失等基础组件已固化于 `IMPLEMENTATION_PLAN.md`，本文档不再重述。
> 接收者：Codex（代码生成模型）。**请严格按章节执行，不得擅自调用 `/data/vjepa2/multimodal_evals/` 以外的可写路径。**

---

## 0. 任务背景与最终目标

### 0.1 任务定位
- 下游任务：Epic-Kitchens (EK) 动作预测（verb / noun / action 三个分类头）。
- 上游骨干：已训练好的 `MultimodalJEPA`（`src/models/multimodal_jepa.py`），权重位于
  `/data/vjepa2/checkpoints/checkpoint_epoch_100.pt`（`epoch=100, model_state_dict=...`）。
- 该 ckpt 的 `model_state_dict` 顶层 key 前缀（已实测）：
  - `v_encoder.*`, `l_encoder.encoder.*`, `projectors.{v|l}_proj_{ctx|tgt}.*`,
    `predictor.{input_proj|mask_token|modality_embed|blocks|norm|output_proj}.*`。
- **唯一可训练部分**：`AttentiveClassifier`（任务头）。其它模块全部 `eval()` + `requires_grad=False`。

### 0.2 数据源
- `/data/eku/vjepa_state_transitions.jsonl`：共 **200** 行；每行字段（已实测）：
  - `clip_id`, `action_narration`, `verb_class`, `noun_class`,
    `text_state_context`, `text_state_target`,
    `visual_frame_context_paths` (32 张), `visual_frame_target_paths` (2 张)。
- 数据 200 条 × 4 象限 ⇒ 数据集 `__len__ = 800`。

### 0.3 任务头前向输入（强约束）
```
concat_feat = torch.cat([z_ctx, z_pred], dim=1)
# z_ctx: [B, N_ctx, D]    z_pred: [B, N_tgt, D]
# D = shared_dim = 512
```
- `z_ctx` 来自 `MultimodalJEPA.encode_context(...)`（视频或文本，过 v/l encoder + 对应 proj_ctx）。
- `z_pred` 来自 `MultimodalJEPA.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=z_tgt.size(1))`。
- 训练损失：`CrossEntropy(verb) + CrossEntropy(noun) + CrossEntropy(action)`。
- 评估指标：`ClassMeanRecall@5`，**按 (V→V, V→L, L→V, L→L) 四象限分别统计并打印**。

### 0.4 张量形状速查（推断阶段）
| 名称 | 形状 (V context) | 形状 (L context) | 备注 |
|---|---|---|---|
| `video_ctx` | `[B, 3, 32, 384, 384]` | — | 仅 V |
| `video_tgt` | `[B, 3, 2, 384, 384]` | — | 仅 V |
| `text_ctx/tgt` | `List[str]` 长度 B | — | 仅 L |
| `z_ctx` (V) | `[B, 9216, 512]` | — | `9216 = (32/2)·(384/16)²` |
| `z_ctx` (L) | — | `[B, 64, 512]` | `max_length=64`（参 `SigLIPTextEncoder`） |
| `z_tgt` (V) | `[B, 576, 512]` | — | `(2/2)·24·24` |
| `z_tgt` (L) | — | `[B, 64, 512]` | — |
| `z_pred`  | `[B, n_tgt, 512]` | `[B, n_tgt, 512]` | `n_tgt = z_tgt.size(1)` |
| `concat_feat` | `[B, N_ctx+n_tgt, 512]` | 同 | 任务头唯一输入 |

> **注**：`SigLIPTextEncoder.max_length=64`，故 L 路径 `N = 64`（不是 77，不要照搬其它项目）。

---

## 1. 文件系统规划 (File System Plan)

### 1.1 工作目录
- 所有改动 / 新增 **必须**位于 `/data/vjepa2/multimodal_evals/ek_action_prediction/`。
- **禁止**修改 `/data/vjepa2/src/**`、`/data/vjepa2/multimodal_evals/action_anticipation_frozen/**`（视为只读依赖）。
- 允许 `import` 它们暴露的公共类 / 函数。

### 1.2 现有文件 → 处理动作
| 路径 | 状态 | 处置 | 说明 |
|---|---|---|---|
| `dataset.py` | 已存在 | **安全修改** | 修小问题：`__len__`，标签 dtype，`text_ctx` 缺省值，导出 `multi_pair_collate_fn`（详见 Part 3 § 5）。**禁止**改 `EKMultimodalDataset.MODALITY_PAIRS` 顺序。 |
| `models.py` | 已存在 | **安全修改** | 强化 ckpt 加载 (`strict=False`)、冻结逻辑、`init_classifier(num_heads, num_blocks)` 默认值与 yaml 对齐；加 verbose 日志。 |
| `eval.py` | 已存在 | **重写关键函数** | 仅替换 `compute_logits` / `train_one_epoch` / `evaluate` / `summarize_metrics` / `main`；保留导入与 `parse_args` 风格（详见 Part 3）。 |
| `ek_action_prediction.yaml` | 已存在 | **扩展字段** | 增加 `classifier.num_heads`、`optimization.warmup/start_lr/final_lr/final_wd`、`logging.folder`、`val_only`、`resume` 等（详见 Part 4）。 |

### 1.3 新建文件清单
| 路径 | 用途 | 行数上限 |
|---|---|---|
| `optim.py` | 复制最小化的 `WarmupCosineLRSchedule` / `CosineWDSchedule` / `init_opt`（与 `action_anticipation_frozen/utils.py` 一致），避免跨包污染。 | ≤ 120 |
| `metrics.py` | 包装 `ClassMeanRecall`：增加 `reset()`、`compute()`、`gather_state()`；屏蔽 `__call__` 内部的 `dist.all_reduce` 副作用（详见 Part 3 § 7）。 | ≤ 120 |
| `checkpoint.py` | 提供 `save_classifier_checkpoint(...)` / `load_classifier_checkpoint(...)`，只保存任务头权重。 | ≤ 80 |
| `EK_PLAN_PART{1..4}.md` | 本套蓝图。 | ≤ 200 each |

### 1.4 不冲突原则
- 复制粘贴 `action_anticipation_frozen/utils.py` 时**重命名为本目录** `optim.py`，避免后续被上游误删。
- 不得 `from evals.action_anticipation_frozen ...` 改为 `from multimodal_evals.action_anticipation_frozen ...`，
  注意当前已存在的 `multimodal_evals/action_anticipation_frozen/` 是镜像目录，可直接复用 `AttentiveClassifier`，
  但**不要**再次定义同名类。

---

## 2. 数据层规范 (Dataset & DataLoader)

### 2.1 `EKMultimodalDataset` 既有契约（保持不变）
- 每个底层样本被展开为 `len(MODALITY_PAIRS) = 4` 个虚拟样本：
  `__len__ = len(split_samples) * 4`，
  `(ctx_mod, tgt_mod) = MODALITY_PAIRS[idx % 4]`。
- `__getitem__` 返回字段必含：
  `video_ctx [3,32,H,W]`, `video_tgt [3,2,H,W]`, `text_ctx (str)`, `text_tgt (str)`,
  `verb_class (int)`, `noun_class (int)`, `clip_id (str)`, `action_narration (str)`,
  `ctx_mod (str)`, `tgt_mod (str)`。

### 2.2 必修小问题（安全修改）
1. **`build_label_maps(samples)`**：必须以 `train_dataset.samples`（整套 200 条）为输入；
   `samples` 字段应是 `EKSample` 列表，**不要**用 `split_samples`，否则 val 出现训练未见类导致 KeyError。
   - 校验：`assert set(s.verb_class for s in val_dataset.split_samples) <= verb_map.keys()`。
2. **`collate_fn`**：保留批级 `verb_class`/`noun_class` 为 `torch.long`；`ctx_mod`/`tgt_mod` 仍为 `List[str]`。
3. **空文本兜底**：若 `text_state_context/target` 在 jsonl 为空或缺失，`clean_text` 应返回 `""`；
   下游 tokenizer 仍能正常工作（验证：`SigLIPTextEncoder` 已支持 `padding="max_length"`）。
4. **DistributedSampler**：`world_size==1` 时返回 `None`，`shuffle` 走 `DataLoader` 自身。

> Part 2 将给出 `models.py` 与张量流转细节；Part 3 给出训练 / 评估循环；Part 4 给出 YAML + 入口 + 给 Codex 的纪律准则。
