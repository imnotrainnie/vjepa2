# Role: 资深 PyTorch 架构师与多模态大模型研究员 (Senior PyTorch Architect & Multi-modal AI Researcher)

## 【任务目标】
请你作为首席架构师，协助我完成一个复杂的多模态大模型魔改任务。
我们需要在 V-JEPA v2.1 的开源代码框架基础上，深度魔改并实现一个受 LeWorldModel (LeWM) 启发的多模态联合嵌入预测架构（Multi-modal JEPA）。
核心目标是：完全移除耗费显存的 EMA (指数移动平均) 机制，利用 SIGReg 实现纯文本与视频的跨空间、无条件 4 象限预测（V->V, V->L, L->V, L->L）。

你当前的任务**不是直接编写底层实现代码**，而是：
1. 阅读并理解指定的工作区代码架构。
2. 吸收我提供的【核心架构与开发约束】。
3. 思考具体的工程落地方案（张量形状变化、文件新建规划、类定义等）。
4. 输出一份极其详尽的 **《多模态 V-JEPA 实施蓝图 (Implementation Plan)》**。这份 Plan 文档将直接发送给下游的 Codex (代码生成模型) 进行逐行代码实现，因此逻辑必须严密、没有模糊地带。

---

## 【工作区与参考文件】
在开始设计之前，请仔细阅读（或调取记忆分析）以下代码库架构文档，严格遵循该项目的 DDP 写法与封装模式：
- `/data/vjepa2/src/README_CN.md`
- `/data/vjepa2/src/CODE_ANNOTATIONS_CN.md`
- `/data/vjepa2/CODE_OF_CONDUCT.md`
以及查看以下目录中的代码和文件：
/data/vjepa2 以及 /data/le-wm

---

## 【核心架构与开发约束 (CRITICAL ARCHITECTURE RULES)】

### 1. 编码器与投影层：非对称解耦设计
网络结构中需实现以下参数共享与独立逻辑：
- **统一编码器 (Encoders - 共享且冻结)**: 
  - `v_encoder`: 复用 V-JEPA 的 ViT (冻结, `requires_grad=False`)。Context 和 Target 侧的视频输入**完全共享**此编码器。
    *注意加载参考：*
    `ckpt = torch.load('/data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt', map_location='cpu')`
    `encoder.load_state_dict(ckpt['encoder'], strict=False)`
  - `l_encoder`: 引入 **SigLIP (Large 版本) 的纯文本编码器** (例如 `google/siglip-large-patch16-384` 的 text encoder，冻结，`requires_grad=False`)。Context 和 Target 的文本输入**完全共享**此编码器。
    *给 Codex 的指示提醒：* 在 Plan 中必须明确指示 Codex 使用 Hugging Face `transformers` 库的 `SiglipTextModel` 和 `AutoTokenizer` 来实现文本特征提取。需特别提醒 Codex 在 Tokenizer 处理时必须设置 `padding="max_length"`，并正确提取出序列特征 (Sequence output) 以适配 JEPA 的预测机制。具体代码交由 Codex 实现。
- **投影头 (Projectors - 结构相同但权重独立，需训练)**:
  - Context 侧: `v_proj_ctx` 和 `l_proj_ctx`。负责把 context 的 表征映射到预测共享空间 (shared_dim)。
  - Target 侧: `v_proj_tgt` 和 `l_proj_tgt`。负责把 target 的 表征映射为被预测的目标表征。**绝对不能**与 Context 侧共享权重。
  -不同模态输入的token数不一致，最后编码的token数也不一致，只是每个token或者patch的维度要共享，而不是将表征都转化成程度完全一致的向量。
    具体架构由你进一步设计。

### 2. Predictor 机制：天然匹配长度差异
- Context 视频是 32 帧，Target 是 2 帧。请在设计中**绝对不要**做全局平均池化或强行对齐序列长度。
- **机制**: Predictor 需要基于 v-jepa2.1 的设计以及当前需求进一步设计，需兼容多模态输入的序列特征。

### 3. 四个预测方向的统一实现：Modality Masking
- **禁止**设计四套不同的 Predictor 网络或臃肿的 IF/ELSE 分支逻辑。
- **实现方式**: 在每个 training step，随机采样一对 `(ctx_modality, tgt_modality) ∈ {V, L}²`。
- 构建统一的输入字典，利用 **Modality Mask** 机制将不需要的模态 token 遮蔽掉（丢弃或替换为零/特定 mask token），让同一个 Predictor 在不同的 Mask 配置下自动完成四个象限的计算，实现代码复用最大化。

### 4. 损失函数与去 EMA 逻辑 (LeWM SIGReg)
移除 V-JEPA 原版的 EMA + Momentum Encoder 逻辑。
- **前向传播拦截**: Target 侧的完整计算流（包括编码器和 Target Projectors）必须包裹在 `with torch.no_grad():` 中，实现整体 **stop-gradient**。Context 侧和 Predictor 正常记录计算图并反传。
- **损失函数**: `Loss = MSE(Z_pred, Z_target) + lambda * SIGReg_Loss`
  - `MSE` 约束 Predictor 还原目标表征。
  - `SIGReg_Loss` 作用在 **Target Projector 和 Context Projector 的输出表征上**。通过对 batch 内的特征施加方差正则项（基于随机投影的经验特征函数，迫使分布呈现各向同性高斯分布），防止多模态表征坍塌。

SIGReg 具体实现参考 `/data/le-wm` 中的代码。
原生代码中 SIGReg 的具体实现与调用路径整理：
1. 实现位置：`module.py`，核心类：`SIGReg`。
   - 输入：proj，形状为 `(T, B, D)`。
   - 逻辑：对高维特征做随机投影，使用 Epps-Pulley 统计量衡量与标准正态分布的差异。
2. 调用路径：`train.py` 中的 `lejepa_forward()`。
   - `output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))`
   - `output["loss"] = pred_loss + λ * sigreg_loss`
3. 作用范围：SIGReg 作用在projector模块输出的表征分布上，对其做正则约束。

### 5. 数据加载设计
- 数据源：`/data/eku/vjepa_state_transitions.jsonl`
- 利用 `torch.utils.data.DataLoader` 设计多进程高效加载模块。需包含图像加载与文本 tokenize。**请在 Plan 中指明要求 Codex 针对 SigLIP tokenizer 进行适配（如截断与 padding 策略）。**适配大规模情境。

---

## 【输出要求：生成 Plan 文档】
请输出一份 Markdown 格式的《实施蓝图》。该文档必须包含以下板块：

1. **文件修改策略 (File System Plan)**：
   - 明确列出哪些是新建文件（如 `src/models/multimodal_jepa.py` 等），哪些是安全修改已有文件（如 `dataset.py`）。
   - **原则**：尽可能采用非破坏性扩展，不影响原 V-JEPA 运行机制，注意文件路径不冲突。

2. **核心模块接口与张量流转 (Tensor Flow & Interface)**：
   - 详细规划 `Dataloader` 输出字典的具体结构和 shape。
   - 详细规划 `Forward` 函数的步骤，必须用伪代码和注释清晰标注从输入到输出**每一个步骤的 Tensor Shape 变化**（考虑到 32 帧/2 帧的视觉差异，以及 SigLIP Large 的文本 Token 序列维度整合）。

3. **Modality Masking 与 Predictor 设计图**：
   - 清晰阐述如何通过一个共用的 mask 生成函数，来控制 V->V, V->L, L->V, L->L 四种计算流。

4. **Loss Function 伪代码设计**：
   - 清晰规划 SIGReg Loss 的实现步骤，特别是如何将 Target 和 Context 的 projector 输出收集并 reshape 以满足 SIGReg 计算要求。

5. **给 Codex 的执行指令**：
   - 总结关键的防错提醒，作为后续 Codex 开发的纪律准则。
   - **必须包含：** `strict=False` 加载逻辑、`with torch.no_grad():` 的确切包裹范围。
   - **必须包含：** 指导 Codex 如何正确引入 `transformers` 的 `SiglipTextModel`，强调 `padding="max_length"` 与序列特征提取的必要性。