# 多模态 V-JEPA 实施蓝图 (Multi-modal JEPA Implementation Plan)

## 文档版本信息
- **创建日期**: 2026-05-12
- **目标**: 基于 V-JEPA 2.1 框架实现受 LeWorldModel 启发的多模态联合嵌入预测架构
- **核心创新**: 移除 EMA 机制，使用 SIGReg 实现 V->V, V->L, L->V, L->L 四象限预测

---

## 第一部分：文件系统规划 (File System Plan)

### 1.1 新建文件清单

#### 核心模型文件
```
/data/vjepa2/src/models/multimodal_jepa.py
```
**用途**: 多模态 JEPA 主模型类，整合视频编码器、文本编码器、投影器和预测器

```
/data/vjepa2/src/models/text_encoder.py
```
**用途**: SigLIP 文本编码器封装，处理文本 tokenization 和特征提取

```
/data/vjepa2/src/models/projectors.py
```
**用途**: 定义 Context 和 Target 侧的投影头（v_proj_ctx, l_proj_ctx, v_proj_tgt, l_proj_tgt）

```
/data/vjepa2/src/models/multimodal_predictor.py
```
**用途**: 多模态预测器，支持 Modality Masking 机制

#### 损失函数模块
```
/data/vjepa2/src/losses/sigreg.py
```
**用途**: 从 LeWM 移植的 SIGReg 损失实现

```
/data/vjepa2/src/losses/multimodal_loss.py
```
**用途**: 组合 MSE 和 SIGReg 的多模态损失函数

#### 数据加载模块
```
/data/vjepa2/src/datasets/multimodal_dataset.py
```
**用途**: 加载 JSONL 数据，处理视频帧和文本描述

```
/data/vjepa2/src/datasets/utils/text_transforms.py
```
**用途**: 文本预处理和 tokenization 工具

#### 训练脚本
```
/data/vjepa2/app/multimodal_jepa/train.py
```
**用途**: 多模态 JEPA 训练主脚本

```
/data/vjepa2/app/multimodal_jepa/configs/multimodal_base.yaml
```
**用途**: 训练配置文件

### 1.2 修改现有文件策略

#### 安全扩展（非破坏性）
```
/data/vjepa2/src/datasets/data_manager.py
```
**修改内容**: 在 `init_data()` 函数中添加 `multimodal` 数据集类型分支
**原则**: 使用 if-elif 结构扩展，不影响原有 imagenet 和 videodataset 逻辑

```
/data/vjepa2/src/utils/checkpoint_loader.py
```
**修改内容**: 添加 `load_multimodal_checkpoint()` 函数，支持加载预训练的 V-JEPA 和 SigLIP 权重
**原则**: 新增函数，不修改现有加载逻辑

### 1.3 文件组织原则
1. **模块化隔离**: 所有多模态相关代码放在独立文件中，避免污染原 V-JEPA 代码
2. **向后兼容**: 确保原 V-JEPA 训练和评估流程不受影响
3. **代码复用**: 最大化复用 V-JEPA 的 ViT、Predictor、Mask 生成等模块
4. **清晰命名**: 使用 `multimodal_` 前缀标识新增模块

---

## 第二部分：核心架构设计 (Core Architecture Design)

### 2.1 模型整体架构图

```
输入数据流:
┌──────────────────────────────────┐
│  Context 视频 (32 frames)    Target 视频 (2 frames)   │
│  Context 文本描述            Target 文本描述          │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────┐
│            Modality Sampling       │
│  随机采样: (ctx_modality, tgt_modality) ∈ {V, L}²  │
│  四种组合: V->V, V->L, L->V, L->L           │
└────────────────────────────────────┘
            ↓
        ┌────────────┴────────┐
      │            │
   Context 分支              Target 分支
   (需要梯度)        (stop-gradient)
      │                 │
        ↓                     ↓
┌─────────────┐          ┌─────────────┐
│ Encoder 选择 │       │ Encoder 选择 │
│ if V: v_encoder│        │ if V: v_encoder│
│ if L: l_encoder│        │ if L: l_encoder│
│ (共享, 冻结) │          │ (共享, 冻结) │
└─────────────┘          └──────────┘
        ↓                 ↓
┌─────────────┐          ┌─────────────┐
│ Projector 选择│         │ Projector 选择│
│ if V: v_proj_ctx│     │ if V: v_proj_tgt│
│ if L: l_proj_ctx│       │ if L: l_proj_tgt│
│ (独立, 可训练)│         │ (独立, 可训练)│
└─────────────┘          └─────────────┘
    ↓                ↓
   Z_ctx [B, N_ctx, D]    with torch.no_grad():
      ↓                  Z_tgt [B, N_tgt, D]
        ↓                       ↓
┌─────────────┐            │
│  Predictor  │              │
│  (可训练)   │                │
└─────────────┘                │
        ↓                   ↓
   Z_pred [B, N_tgt, D]        │
        ↓           ↓
        └──────┬───────────┘
                    ↓
         ┌───────────┐
         │ Loss Computation │
         │ MSE(Z_pred, Z_tgt)│
         │ + λ * SIGReg(Z_ctx,│
         │           Z_tgt)  │
         └─────────────┘
```

### 2.2 编码器与投影器详细设计

#### 2.2.1 视频编码器 (v_encoder)
- **类型**: V-JEPA ViT-B (从 vjepa2_1_vitb_dist_vitG_384.pt 加载)
- **参数**: 冻结 (`requires_grad=False`)
- **输入**: 
  - Context: `[B, 3, 32, H, W]` (32 帧)
  - Target: `[B, 3, 2, H, W]` (2 帧)
- **输出**:
  - Context: `[B, N_ctx_v, D_enc]` 其中 `N_ctx_v = (32/tubelet_size) * (H/patch_size) * (W/patch_size)`
  - Target: `[B, N_tgt_v, D_enc]` 其中 `N_tgt_v = (2/tubelet_size) * (H/patch_size) * (W/patch_size)`
- **关键参数**:
  - `tubelet_size = 2`
  - `patch_size = 16`
  - `img_size = 384`
  - `D_enc = 768` (ViT-B 嵌入维度)

#### 2.2.2 文本编码器 (l_encoder)
- **类型**: SigLIP Large Text Encoder (`google/siglip-large-patch16-384`)
- **参数**: 冻结 (`requires_grad=False`)
- **输入**: 
  - Context: 文本描述字符串
  - Target: 文本描述字符串
- **Tokenization 配置**:
  ```python
  tokenizer = AutoTokenizer.from_pretrained('google/siglip-large-patch16-384')
  inputs = tokenizer(
    text,
      padding="max_length",
      max_length=77,  # SigLIP 默认序列长度
      truncation=True,
      return_tensors="pt"
  )
  ```
- **输出**:
  - Sequence output: `[B, 77, D_text]` 其中 `D_text = 1024` (SigLIP Large)
  - **注意**: 使用完整序列特征，不使用 pooled output
- **关键参数**:
  - `max_length = 77`
  - `D_text = 1024`

#### 2.2.3 投影器设计 (Projectors)

**共享维度 (shared_dim)**:
```python
shared_dim = 512  # 预测共享空间维度
```

**Context 侧投影器**:
```python
# v_proj_ctx: 视频 Context 投影
v_proj_ctx = nn.Sequential(
    nn.Linear(768, 1024),      # D_enc -> hidden
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Linear(1024, shared_dim) # hidden -> shared_dim
)

# l_proj_ctx: 文本 Context 投影
l_proj_ctx = nn.Sequential(
    nn.Linear(1024, 1024),     # D_text -> hidden
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Linear(1024, shared_dim) # hidden -> shared_dim
)
```

**Target 侧投影器** (权重完全独立):
```python
# v_proj_tgt: 视频 Target 投影
v_proj_tgt = nn.Sequential(
    nn.Linear(768, 1024),
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Linear(1024, shared_dim)
)

# l_proj_tgt: 文本 Target 投影
l_proj_tgt = nn.Sequential(
    nn.Linear(1024, 1024),
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Linear(1024, shared_dim)
```

**关键约束**:
1. **不共享权重**: Context 和 Target 投影器必须参数独立
2. **输出维度统一**: 所有投影器输出 `shared_dim = 512`
3. **保留序列长度**: 投影器不做池化，保持 token 序列结构

---

## 第三部分：张量流转与接口设计 (Tensor Flow & Interface)

### 3.1 数据加载器输出格式

#### MultimodalDataset.__getitem__() 返回字典:
```python
{
    # 视频数据
    'video_ctx': torch.Tensor,    # [3, 32, H, W] - Context 视频
    'video_tgt': torch.Tensor,    # [3, 2, H, W] - Target 视频
    
    # 文本数据
    'text_ctx': str,              # Context 文本描述
    'text_tgt': str,              # Target 文本描述
    
    # 元数据
    'clip_id': str,               # 样本 ID
    'action_narration': str,      # 动作描述
}
```

#### DataLoader collate_fn 输出 (batch):
```python
{
    'video_ctx': torch.Tensor,    # [B, 3, 32, H, W]
    'video_tgt': torch.Tensor,    # [B, 3, 2, H, W]
    'text_ctx': List[str],        # 长度 B 的字符串列表
    'text_tgt': List[str],      # 长度 B 的字符串列表
    'clip_id': List[str],
    'action_narration': List[str],
}
```

### 3.2 Forward 函数详细流程

#### 伪代码与张量形状标注:
```python
def forward(self, batch, device):
    """
    完整的前向传播流程，包含 Modality Masking 和 SIGReg 损失计算
    """
    B = batch['video_ctx'].size(0)
    
    # ==================================
    # Step 1: 随机采样模态组合
    # ==============================
    # 从 {V, L}² 中随机选择一对 (ctx_modality, tgt_modality)
  modalities = ['V', 'L']
    ctx_mod = random.choice(modalities)  # 'V' 或 'L'
    tgt_mod = random.choice(modalities)  # 'V' 或 'L'
    
    # ========================
    # Step 2: Context 侧编码 (需要梯度)
    # ============================
    if ctx_mod == 'V':
        # 视频编码
        video_ctx = batch['video_ctx'].to(device)  # [B, 3, 32, H, W]
        with torch.no_grad():  # 编码器冻结
          h_ctx = self.v_encoder(video_ctx)      # [B, N_ctx_v, 768]
        # N_ctx_v = (32/2) * (384/16) * (384/16) = 16 * 24 * 24 = 9216
        z_ctx = self.v_proj_ctx(h_ctx)      # [B, 9216, 512]
        
    else:  # ctx_mod == 'L'
        # 文本编码
        text_ctx = batch['text_ctx']
        tokens = self.tokenizer(
            text_ctx,
            padding="max_length",
            max_length=77,
     truncation=True,
            return_tensors="pt"
        ).to(device)
      with torch.no_grad():  # 编码器冻结
          h_ctx = self.l_encoder(**tokens).last_hidden_state  # [B, 77, 1024]
        z_ctx = self.l_proj_ctx(h_ctx)                          # [B, 77, 512]
    
    # =======================================
    # Step 3: Target 侧编码 (stop-gradient)
    # ===========================
    with torch.no_grad():  # 整个 Target 分支不计算梯度
        if tgt_mod == 'V':
            # 视频编码
            video_tgt = batch['video_tgt'].to(device)  # [B, 3, 2, H, W]
            h_tgt = self.v_encoder(video_tgt)          # [B, N_tgt_v, 768]
            # N_tgt_v = (2/2) * (384/16) * (384/16) = 1 * 24 * 24 = 576
            z_tgt = self.v_proj_tgt(h_tgt)           # [B, 576, 512]
            
        else:  # tgt_mod == 'L'
        # 文本编码
       text_tgt = batch['text_tgt']
            tokens = self.tokenizer(
                text_tgt,
             padding="max_length",
       max_length=77,
           truncation=True,
                return_tensors="pt"
            ).to(device)
         h_tgt = self.l_encoder(**tokens).last_hidden_state  # [B, 77, 1024]
            z_tgt = self.l_proj_tgt(h_tgt)                  # [B, 77, 512]
    
    # ==============================
    # Step 4: Predictor 预测
    # ======================================
    # Predictor 需要处理不同长度的序列
    # z_ctx: [B, N_ctx, 512] - N_ctx 可能是 9216 (V) 或 77 (L)
    # z_tgt: [B, N_tgt, 512] - N_tgt 可能是 576 (V) 或 77 (L)
    
    z_pred = self.predictor(
    z_ctx,           # Context 表征
        ctx_mod,                 # Context 模态标识
        tgt_mod,                 # Target 模态标识
        N_tgt=z_tgt.size(1)      # Target 序列长度
    )  # [B, N_tgt, 512]
    
    # ========================
    # Step 5: 损失计算
    # ===========================
    # 5.1 MSE 损失
    loss_mse = F.mse_loss(z_pred, z_tgt.detach())  # 标量
    
    # 5.2 SIGReg 损失
    # 收集所有投影器输出用于正则化
    # 需要转置为 (T, B, D) 格式
    proj_outputs = []
    
    # Context 投影输出 (有梯度)
    proj_outputs.append(z_ctx.transpose(0, 1))  # [N_ctx, B, 512]
    
    # Target 投影输出 (无梯度，但参与 SIGReg 计算)
    proj_outputs.append(z_tgt.transpose(0, 1))  # [N_tgt, B, 512]
    
    # 拼接所有投影输出
    all_proj = torch.cat(proj_outputs, dim=0)   # [N_ctx + N_tgt, B, 512]
    
    loss_sigreg = self.sigreg(all_proj)         # 标量
    
    # 5.3 总损失
    lambda_sigreg = 0.1  # 超参数
    loss_total = loss_mse + lambda_sigreg * loss_sigreg
    
    return {
        'loss': loss_total,
        'loss_mse': loss_mse,
        'loss_sigreg': loss_sigreg,
        'z_pred': z_pred,
        'z_tgt': z_tgt,
        'z_ctx': z_ctx, 
        'ctx_mod': ctx_mod,
        'tgt_mod': tgt_mod,
    }
```

### 3.3 张量形状汇总表

| 阶段 | 变量名 | 形状 | 说明 |
|------|--------|------|------|
| **输入** | `video_ctx` | `[B, 3, 32, 384, 384]` | Context 视频 |
| | `video_tgt` | `[B, 3, 2, 384, 384]` | Target 视频 |
| | `text_ctx` | `List[str]` | Context 文本 |
| | `text_tgt` | `List[str]` | Target 文本 |
| **编码器输出** | `h_ctx` (V) | `[B, 9216, 768]` | 视频 Context 编码 |
| | `h_ctx` (L) | `[B, 77, 1024]` | 文本 Context 编码 |
| | `h_tgt` (V) | `[B, 576, 768]` | 视频 Target 编码 |
| | `h_tgt` (L) | `[B, 77, 1024]` | 文本 Target 编码 |
| **投影器输出** | `z_ctx` (V) | `[B, 9216, 512]` | 视频 Context 投影 |
| | `z_ctx` (L) | `[B, 77, 512]` | 文本 Context 投影 |
| | `z_tgt` (V) | `[B, 576, 512]` | 视频 Target 投影 |
| | `z_tgt` (L) | `[B, 77, 512]` | 文本 Target 投影 |
| **预测器输出** | `z_pred` | `[B, N_tgt, 512]` | 预测表征 |
| **SIGReg 输入** | `all_proj` | `[N_ctx+N_tgt, B, 512]` | 拼接的投影输出 |

---

## 第四部分：Modality Masking 与 Predictor 设计 (Modality Masking & Predictor)

### 4.1 Modality Masking 机制

#### 核心思想
通过在每个训练步骤随机采样模态组合，让单一 Predictor 网络学习处理所有四种预测任务，避免设计四套独立网络。

#### 实现策略
```python
class MultimodalPredictor(nn.Module):
    """
    统一的多模态预测器，通过模态标识自适应处理不同输入
    """
    def __init__(
    self,
      shared_dim=512,
        predictor_dim=384,
        depth=12,  # ✅ 修正：与 V-JEPA 2.1 ViT-B 对齐（原为 6）
        num_heads=8,
        mlp_ratio=4.0,
        use_3d_pos_for_video=True,  # ✅ 新增：视频使用 3D 位置编码
    ):
        super().__init__()
        
        # 模态嵌入：为 V 和 L 分配可学习的模态标识
        self.modality_embed = nn.Embedding(2, shared_dim)  # 0: V, 1: L
        
        # 输入投影
        self.input_proj = nn.Linear(shared_dim, predictor_dim)
        
        # 位置编码（动态生成，适配不同序列长度）
        # 不使用固定的 pos_embed，而是在 forward 中动态插值
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=predictor_dim,
         num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            use_rope=False,  # 使用 sincos 位置编码
          )
            for _ in range(depth)
        ])
        
        # 输出投影
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, shared_dim)
    
    def forward(self, z_ctx, ctx_mod, tgt_mod, N_tgt):
        """
      Args:
            z_ctx: [B, N_ctx, shared_dim] - Context 表征
            ctx_mod: str - Context 模态 ('V' 或 'L')
            tgt_mod: str - Target 模态 ('V' 或 'L')
        N_tgt: int - Target 序列长度
        
        Returns:
            z_pred: [B, N_tgt, shared_dim] - 预测的 Target 表征
        """
        B, N_ctx, D = z_ctx.shape
        
        # Step 1: 添加模态嵌入到 Context
        ctx_mod_id = 0 if ctx_mod == 'V' else 1
     ctx_mod_emb = self.modality_embed(
        torch.tensor([ctx_mod_id], device=z_ctx.device)
        )  # [1, shared_dim]
        z_ctx = z_ctx + ctx_mod_emb.unsqueeze(0)  # [B, N_ctx, shared_dim]
        
        # Step 2: 初始化 Target Mask Tokens
        tgt_mod_id = 0 if tgt_mod == 'V' else 1
        tgt_mod_emb = self.modality_embed(
            torch.tensor([tgt_mod_id], device=z_ctx.device)
        )  # [1, shared_dim]
        
        # 可学习的 mask token (初始化为零)
        mask_tokens = torch.zeros(B, N_tgt, D, device=z_ctx.device)
        mask_tokens = mask_tokens + tgt_mod_emb.unsqueeze(0)  # [B, N_tgt, shared_dim]
        
        # Step 3: 拼接 Context 和 Target tokens
        x = torch.cat([z_ctx, mask_tokens], dim=1)  # [B, N_ctx + N_tgt, shared_dim]
        
      # Step 4: 投影到 Predictor 维度
        x = self.input_proj(x)  # [B, N_ctx + N_tgt, predictor_dim]
        
        # Step 5: 添加位置编码
      pos_embed = self._get_pos_embed(N_ctx + N_tgt, x.device)  # [1, N_ctx + N_tgt, predictor_dim]
        x = x + pos_embed
        
        # Step 6: 通过 Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Step 7: 提取 Target 位置的预测
        x_tgt = x[:, N_ctx:, :]  # [B, N_tgt, predictor_dim]
        
        # Step 8: 投影回共享空间
     z_pred = self.output_proj(x_tgt)  # [B, N_tgt, shared_dim]
        
        return z_pred
    
   def _get_pos_embed(self, seq_len, ctx_mod, tgt_mod, device):
       """
       根据模态类型动态生成位置编码
       """
       embed_dim = self.input_proj.out_features

       # 判断是否需要 3D 位置编码
       if self.use_3d_pos_for_video and (ctx_mod == 'V' or tgt_mod == 'V'):
           # 视频：使用 3D sincos 位置编码
           # 需要根据 seq_len 推断 (T, H, W) 维度
           if seq_len == 9216:  # Context 视频: 32 帧
               grid_depth = 32 // self.tubelet_size  # 16
            grid_size = self.img_size // self.patch_size  # 24
           elif seq_len == 576:  # Target 视频: 2 帧
               grid_depth = 2 // self.tubelet_size  # 1
             grid_size = self.img_size // self.patch_size  # 24
           else:
            # 文本或其他长度，回退到 1D
               return self._get_1d_pos_embed(seq_len, embed_dim, device)

           # 生成 3D 位置编码
           from src.models.utils.pos_embs import get_3d_sincos_pos_embed
           pos_embed = get_3d_sincos_pos_embed(
             embed_dim, grid_size, grid_depth, cls_token=False
           )
           return torch.from_numpy(pos_embed).float().unsqueeze(0).to(device)
       else:
      # 文本：使用 1D sincos 位置编码
           return self._get_1d_pos_embed(seq_len, embed_dim, device)

   def _get_1d_pos_embed(self, seq_len, embed_dim, device):
       """1D sincos 位置编码"""
       pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
       dim_t = torch.arange(embed_dim, dtype=torch.float32, device=device)
       dim_t = 10000 ** (2 * (dim_t // 2) / embed_dim)

       pos_embed = pos / dim_t
       pos_embed[:, 0::2] = torch.sin(pos_embed[:, 0::2])
       pos_embed[:, 1::2] = torch.cos(pos_embed[:, 1::2])

       return pos_embed.unsqueeze(0)
```

### 4.2 Predictor 设计要点

#### 4.2.1 为什么不需要四套 Predictor？
1. **模态嵌入机制**: 通过可学习的模态标识 (V=0, L=1)，Predictor 能够区分不同模态的输入
2. **序列长度自适应**: 使用动态位置编码，自动适配 9216 (V) 和 77 (L) 的序列长度差异
3. **统一的表征空间**: 所有投影器输出到相同的 `shared_dim=512`，Predictor 只需学习一个映射

#### 4.2.2 与 V-JEPA 原版 Predictor 的区别
| 特性 | V-JEPA Predictor | Multimodal Predictor |
|------|----------|----------------------|
| 输入类型 | 仅视频 tokens | 视频或文本 tokens |
| 序列长度 | 固定 (基于 mask 策略) | 动态 (77 或 9216) |
| 位置编码 | 3D sincos (T, H, W) | 1D sincos (动态生成) |
| 模态区分 | 无 | 模态嵌入 (Embedding) |
| Mask Tokens | 固定数量 | 根据 N_tgt 动态生成 |

#### 4.2.3 关键设计决策
1. **混合位置编码策略** ✅ 修正: 
   - 视频输入使用 V-JEPA 的 3D sincos 位置编码 (T, H, W)，保留空间结构
   - 文本输入使用 1D sincos 位置编码，适配序列特性
2. **动态 Mask Tokens**: 根据 Target 模态和序列长度动态生成，而非预定义固定数量
3. **模态嵌入**: 作为额外的可学习信号，帮助 Predictor 理解当前处理的模态类型
4. **深度对齐** ✅ 修正: 使用 12 层 Transformer，与 V-JEPA 2.1 ViT-B 保持一致

---

## 第五部分：SIGReg 损失函数设计 (SIGReg Loss Function)

### 5.1 SIGReg 原理回顾

#### 来自 LeWM 的实现 (`/data/le-wm/module.py`)
```python
class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer"""
    
    def __init__(self, knots=17, num_proj=1024):
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

    def forward(self, proj):
        """
        proj: (T, B, D) - 投影器输出的表征
        """
        # 随机投影到低维空间
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        
        # 计算 Epps-Pulley 统计量
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
```

### 5.2 SIGReg 在多模态 JEPA 中的应用

#### 5.2.1 作用范围
SIGReg 作用在 **所有投影器的输出表征** 上，包括：
- Context 侧: `v_proj_ctx` 和 `l_proj_ctx` 的输出
- Target 侧: `v_proj_tgt` 和 `l_proj_tgt` 的输出

#### 5.2.2 为什么需要 SIGReg？
1. **防止表征坍塌**: 在没有 EMA 的情况下，投影器可能输出常数或低秩表征
2. **多模态对齐**: 强制视频和文本表征分布接近各向同性高斯分布，促进跨模态对齐
3. **替代 EMA**: SIGReg 通过分布约束提供稳定的训练信号，无需维护 Target Encoder

#### 5.2.3 输入准备
```python
def prepare_sigreg_input(z_ctx, z_tgt):
    """
    将 Context 和 Target 投影输出拼接为 SIGReg 输入格式
    
    Args:
        z_ctx: [B, N_ctx, D] - Context 投影输出
        z_tgt: [B, N_tgt, D] - Target 投影输出
    
    Returns:
        proj: [T, B, D] - SIGReg 输入，T = N_ctx + N_tgt
    """
    # 转置为 (T, B, D) 格式
    z_ctx_t = z_ctx.transpose(0, 1)  # [N_ctx, B, D]
    z_tgt_t = z_tgt.transpose(0, 1)  # [N_tgt, B, D]
    
    # 拼接
    proj = torch.cat([z_ctx_t, z_tgt_t], dim=0)  # [N_ctx + N_tgt, B, D]
    
    return proj
```

### 5.3 完整损失函数实现

```python
class MultimodalLoss(nn.Module):
    """
    多模态 JEPA 损失函数：MSE + SIGReg
    """
    def __init__(self, lambda_sigreg=0.1, sigreg_knots=17, sigreg_num_proj=1024):
        super().__init__()
        self.lambda_sigreg = lambda_sigreg
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
    
    def forward(self, z_pred, z_tgt, z_ctx):
        """
        Args:
            z_pred: [B, N_tgt, D] - Predictor 预测的 Target 表征
          z_tgt: [B, N_tgt, D] - 真实的 Target 表征 (已 detach)
            z_ctx: [B, N_ctx, D] - Context 表征
        
        Returns:
            loss_dict: 包含总损失和各项损失的字典
        """
        # 1. MSE 损失：预测准确性
        loss_mse = F.mse_loss(z_pred, z_tgt.detach())
        
        # 2. SIGReg 损失：表征分布正则化
        # 准备输入：拼接 Context 和 Target 投影输出
        proj = prepare_sigreg_input(z_ctx, z_tgt)  # [N_ctx + N_tgt, B, D]
        loss_sigreg = self.sigreg(proj)
        
        # 3. 总损失
     loss_total = loss_mse + self.lambda_sigreg * loss_sigreg
    
        return {
         'loss': loss_total,
            'loss_mse': loss_mse,
         'loss_sigreg': loss_sigreg,
        }
```

### 5.4 SIGReg 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_sigreg` | 0.1 | SIGReg 损失权重 |
| `knots` | 17 | Epps-Pulley 统计量的节点数 |
| `num_proj` | 1024 | 随机投影的维度数 |

**调优建议**:
- `lambda_sigreg` 过大会导致表征过度平滑，损失预测能力
- `lambda_sigreg` 过小无法有效防止坍塌
- 建议从 0.05 开始，根据验证集性能调整到 0.1-0.2 范围

---

## 第六部分：数据加载模块设计 (Data Loading Module)

### 6.1 数据格式分析

#### JSONL 文件结构 (`/data/eku/vjepa_state_transitions.jsonl`)
```json
{
    "clip_id": "P02_12_101",
    "action_narration": "close drawer",
    "verb_class": 4,
    "noun_class": 8,
    "text_state_context": "A hand reaches into a partially opened bag...",
    "visual_frame_context_paths": [
        "/nvme/vjepa_data_u/P02_12_101/frames/frame_0000017540.jpg",
     ...  // 32 帧路径
    ],
  "text_state_target": "A hand reaches into an open wooden drawer...",
    "visual_frame_target_paths": [
        "/nvme/vjepa_data_u/P02_12_101/frames/frame_0000017832.jpg",
        "/nvme/vjepa_data_u/P02_12_101/frames/frame_0000017833.jpg"
    ]
}
```

#### 关键字段映射
- `visual_frame_context_paths` → `video_ctx` (32 帧)
- `visual_frame_target_paths` → `video_tgt` (2 帧)
- `text_state_context` → `text_ctx`
- `text_state_target` → `text_tgt`

### 6.2 MultimodalDataset 实现

```python
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MultimodalDataset(Dataset):
    """
    多模态数据集：加载视频帧和文本描述
    """
    def __init__(
        self,
        jsonl_path,
        video_transform=None,
        img_size=384,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.img_size = img_size
        
        # 加载 JSONL 数据
        self.samples = []
      with open(jsonl_path, 'r') as f:
            for line in f:
          self.samples.append(json.loads(line))
        
        # 视频帧变换
        if video_transform is None:
            self.video_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
                transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
             )
        ])
        else:
            self.video_transform = video_transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载 Context 视频帧 (32 帧)
        ctx_frames = []
        for frame_path in sample['visual_frame_context_paths']:
            img = Image.open(frame_path).convert('RGB')
            img = self.video_transform(img)
            ctx_frames.append(img)
        video_ctx = torch.stack(ctx_frames, dim=1)  # [3, 32, H, W]
        
      # 加载 Target 视频帧 (2 帧)
        tgt_frames = []
        for frame_path in sample['visual_frame_target_paths']:
            img = Image.open(frame_path).convert('RGB')
         img = self.video_transform(img)
            tgt_frames.append(img)
        video_tgt = torch.stack(tgt_frames, dim=1)  # [3, 2, H, W]
      
        # 文本数据
        text_ctx = sample['text_state_context']
        text_tgt = sample['text_state_target']
        
        return {
            'video_ctx': video_ctx,
            'video_tgt': video_tgt,
            'text_ctx': text_ctx,
            'text_tgt': text_tgt,
            'clip_id': sample['clip_id'],
            'action_narration': sample['action_narration'],
      }
```

### 6.3 DataLoader 配置

```python
def create_multimodal_dataloader(
    jsonl_path,
    batch_size=16,
    num_workers=2,
    img_size=384,
    shuffle=True,
):
    """
    创建多模态数据加载器
    """
    dataset = MultimodalDataset(
        jsonl_path=jsonl_path,
        img_size=img_size,
    )
    
    # 自定义 collate_fn（处理文本列表）
    def collate_fn(batch):
        video_ctx = torch.stack([item['video_ctx'] for item in batch])
        video_tgt = torch.stack([item['video_tgt'] for item in batch])
        text_ctx = [item['text_ctx'] for item in batch]
        text_tgt = [item['text_tgt'] for item in batch]
        clip_id = [item['clip_id'] for item in batch]
        action_narration = [item['action_narration'] for item in batch]
        
        return {
        'video_ctx': video_ctx,
            'video_tgt': video_tgt,
            'text_ctx': text_ctx,
          'text_tgt': text_tgt,
            'clip_id': clip_id,
            'action_narration': action_narration,
        }
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # 确保批次大小一致
    )
    
    return dataloader
```

### 6.4 数据加载优化策略

#### 6.4.1 多进程加载
- `num_workers=2`: 使用 2 个进程并行加载图像
- `pin_memory=True`: 加速 CPU 到 GPU 的数据传输

#### 6.4.2 缓存策略（可选）
对于频繁访问的数据，可以实现内存缓存：
```python
class CachedMultimodalDataset(MultimodalDataset):
    def __init__(self, *args, cache_size=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = cache_size
    
    def __getitem__(self, idx):
        if idx in self.cache:
        return self.cache[idx]
        
        item = super().__getitem__(idx)
        
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
     
        return item
```

#### 6.4.3 预加载策略
对于小规模数据集，可以预加载所有数据到内存：
```python
class PreloadedMultimodalDataset(MultimodalDataset):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
        print("Preloading all data into memory...")
        self.preloaded_data = []
        for idx in range(len(self)):
          self.preloaded_data.append(super().__getitem__(idx))
        print(f"Preloaded {len(self.preloaded_data)} samples")
    
    def __getitem__(self, idx):
        return self.preloaded_data[idx]
```

---

## 第七部分：训练流程设计 (Training Pipeline)

### 7.1 训练主循环伪代码

```python
def train_multimodal_jepa(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    log_interval=100,
):
    """
    多模态 JEPA 训练主循环
    """
    model.to(device)
    loss_fn = MultimodalLoss(lambda_sigreg=0.1)
    
    for epoch in range(num_epochs):
        model.train()
     epoch_losses = {'loss': 0, 'loss_mse': 0, 'loss_sigreg': 0}
        
      for batch_idx, batch in enumerate(train_loader):
            # =====================
        # Forward Pass
            # =====================
            output = model(batch, device)
            
            # 计算损失
            loss_dict = loss_fn(
            z_pred=output['z_pred'],
             z_tgt=output['z_tgt'],
                z_ctx=output['z_ctx'],  # 需要在 model.forward 中返回
            )
            
            loss = loss_dict['loss']
            
       # =================
            # Backward Pass
            # =================
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # ===================
          # Logging
            # ===============
          for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
            
            if (batch_idx + 1) % log_interval == 0:
          print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"MSE: {loss_dict['loss_mse'].item():.4f} "
                 f"SIGReg: {loss_dict['loss_sigreg'].item():.4f} "
                   f"Modality: {output['ctx_mod']}->{output['tgt_mod']}")
        
    # ====================
        # Epoch Summary
        # ================
        for key in epoch_losses:
          epoch_losses[key] /= len(train_loader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  Avg Loss: {epoch_losses['loss']:.4f}")
        print(f"  Avg MSE: {epoch_losses['loss_mse']:.4f}")
        print(f"  Avg SIGReg: {epoch_losses['loss_sigreg']:.4f}")
        
     # ======================
        # Validation
        # ========================
        if (epoch + 1) % 5 == 0:
            val_loss = validate(model, val_loader, loss_fn, device)
            print(f"  Validation Loss: {val_loss:.4f}")
        
        # ====================
        # Learning Rate Scheduling
     # =========================
        scheduler.step()
        
        # ===========================
        # Checkpoint Saving
        # ===========================
        if (epoch + 1) % 10 == 0:
          save_checkpoint(model, optimizer, epoch, epoch_losses)

def validate(model, val_loader, loss_fn, device):
    """
    验证函数
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch, device)
         loss_dict = loss_fn(
                z_pred=output['z_pred'],
                z_tgt=output['z_tgt'],
                z_ctx=output['z_ctx'],
            )
            total_loss += loss_dict['loss'].item()
    
    return total_loss / len(val_loader)
```

### 7.2 优化器与学习率调度

```python
def create_optimizer_and_scheduler(model, num_epochs, steps_per_epoch):
    """
    创建优化器和学习率调度器
    """
    # 分组参数：投影器和预测器需要训练，编码器冻结
    trainable_params = []
    
    # 投影器参数
    trainable_params.append({
        'params': model.v_proj_ctx.parameters(),
      'lr': 1e-4,
        'name': 'v_proj_ctx'
    })
    trainable_params.append({
      'params': model.l_proj_ctx.parameters(),
        'lr': 1e-4,
        'name': 'l_proj_ctx'
    })
    trainable_params.append({
        'params': model.v_proj_tgt.parameters(),
        'lr': 1e-4,
        'name': 'v_proj_tgt'
    })
    trainable_params.append({
        'params': model.l_proj_tgt.parameters(),
        'lr': 1e-4,
        'name': 'l_proj_tgt'
    })
    
    # 预测器参数
    trainable_params.append({
        'params': model.predictor.parameters(),
        'lr': 5e-5,
        'name': 'predictor'
    })
    
    # 优化器：AdamW
    optimizer = torch.optim.AdamW(
        trainable_params,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )
    
    # 学习率调度器：Warmup + Cosine Annealing
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup 阶段：线性增长
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine Annealing 阶段
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler
```

### 7.3 检查点保存与加载

```python
def save_checkpoint(model, optimizer, epoch, losses, save_dir='checkpoints'):
    """
    保存训练检查点
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
      'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
      'losses': losses,
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载训练检查点
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch+1}")
  
    return epoch, losses
```

### 7.4 分布式训练支持（DDP）

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """
    初始化分布式训练环境
    """
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_multimodal_jepa_ddp(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    local_rank,
):
    """
    分布式训练主循环
    """
    # 包装模型为 DDP
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    loss_fn = MultimodalLoss(lambda_sigreg=0.1)
    
    for epoch in range(num_epochs):
        # 设置 epoch（用于 DistributedSampler）
        train_loader.sampler.set_epoch(epoch)
        
        model.train()
        epoch_losses = {'loss': 0, 'loss_mse': 0, 'loss_sigreg': 0}
        
        for batch_idx, batch in enumerate(train_loader):
          output = model.module(batch, local_rank)  # 注意使用 model.module
            
            loss_dict = loss_fn(
            z_pred=output['z_pred'],
         z_tgt=output['z_tgt'],
        z_ctx=output['z_ctx'],
          )
            
            loss = loss_dict['loss']
          
            optimizer.zero_grad()
            loss.backward()
         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 同步损失（用于日志）
          for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
        
        # 只在 rank 0 打印日志和保存检查点
        if local_rank == 0:
            for key in epoch_losses:
           epoch_losses[key] /= len(train_loader)
            
         print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
            print(f"  Avg Loss: {epoch_losses['loss']:.4f}")
          
            if (epoch + 1) % 10 == 0:
            save_checkpoint(model.module, optimizer, epoch, epoch_losses)
        
        scheduler.step()
    
    dist.destroy_process_group()
```

---

## 第八部分：给 Codex 的执行指令 (Instructions for Codex)

### 8.1 关键防错提醒

#### 8.1.1 模型加载
```python
# ✅ 正确：使用 strict=False 加载预训练权重
ckpt = torch.load('/data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt', map_location='cpu')
v_encoder.load_state_dict(ckpt['encoder'], strict=False)

# ❌ 错误：使用 strict=True 会因键不匹配而报错
v_encoder.load_state_dict(ckpt['encoder'], strict=True)
```

#### 8.1.2 梯度控制
```python
# ✅ 正确：将 no_grad 范围收窄到编码器，让投影器在 SIGReg 路径上保留梯度；仅在送入 MSE 时对 z_tgt 做 detach()。
# ✅ 修复后的 Target 分支
# 编码器冻结，投影器可训练
with torch.no_grad():
    if tgt_mod == 'V':
        h_tgt = self.v_encoder(video_tgt)   # [B, N_tgt_v, 768] 无梯度
    else:
        h_tgt = self.l_encoder(**tokens).last_hidden_state  # [B, 77, 1024] 无梯度

# 投影器在 no_grad 外，允许梯度回传到 v/l_proj_tgt
z_tgt = self.v_proj_tgt(h_tgt)   # [B, N_tgt, 512] ← 有梯度

# SIGReg: z_tgt 带梯度参与，更新 proj_tgt 参数
loss_sigreg = self.sigreg(prepare_sigreg_input(z_ctx, z_tgt))

# MSE: 只在这里 detach，阻止 predictor 的梯度绕道 z_tgt 影响 proj_tgt
loss_mse = F.mse_loss(z_pred, z_tgt.detach())


```

#### 8.1.3 SigLIP 文本编码器使用
```python
# ✅ 正确：使用 transformers 库加载 SigLIP
from transformers import AutoTokenizer, SiglipTextModel

tokenizer = AutoTokenizer.from_pretrained('google/siglip-large-patch16-384')
l_encoder = SiglipTextModel.from_pretrained('google/siglip-large-patch16-384')

# 冻结参数
for param in l_encoder.parameters():
    param.requires_grad = False

# Tokenization 必须设置 padding="max_length"
tokens = tokenizer(
    text_list,
    padding="max_length",  # 必须！
    max_length=77,
    truncation=True,
    return_tensors="pt"
)

# 提取序列特征（不是 pooled output）
output = l_encoder(**tokens)
text_features = output.last_hidden_state  # [B, 77, 1024]

# ❌ 错误：使用 pooler_output 会丢失序列信息
text_features = output.pooler_output  # 这是 [B, 1024]，丢失了序列维度！
```

#### 8.1.4 投影器权重独立性
```python
# ✅ 正确：Context 和 Target 投影器完全独立
v_proj_ctx = nn.Sequential(...)
v_proj_tgt = nn.Sequential(...)  # 独立的参数

# ❌ 错误：共享权重
v_proj_ctx = nn.Sequential(...)
v_proj_tgt = v_proj_ctx  # 错误！这会导致权重共享
```

#### 8.1.5 SIGReg 输入格式
```python
# ✅ 正确：输入格式为 (T, B, D)
z_ctx_t = z_ctx.transpose(0, 1)  # [B, N_ctx, D] -> [N_ctx, B, D]
z_tgt_t = z_tgt.transpose(0, 1)  # [B, N_tgt, D] -> [N_tgt, B, D]
proj = torch.cat([z_ctx_t, z_tgt_t], dim=0)  # [N_ctx + N_tgt, B, D]
loss_sigreg = self.sigreg(proj)

# ❌ 错误：直接传入 (B, T, D) 格式
proj = torch.cat([z_ctx, z_tgt], dim=1)  # [B, N_ctx + N_tgt, D]
loss_sigreg = self.sigreg(proj)  # 错误！维度不匹配
```

### 8.2 开发纪律准则

#### 8.2.1 代码组织
1. **模块化**: 每个功能独立成文件，避免单文件过长
2. **命名规范**: 使用描述性变量名，避免 `x`, `y`, `z` 等模糊命名
3. **注释**: 在关键张量操作处标注形状，例如 `# [B, N, D]`
4. **类型提示**: 使用 Python 类型提示增强代码可读性

#### 8.2.2 调试策略
1. **形状检查**: 在每个关键步骤后打印张量形状
   ```python
   print(f"z_ctx shape: {z_ctx.shape}")  # 应该是 [B, N_ctx, 512]
   ```
2. **梯度检查**: 验证哪些参数有梯度
   ```python
   for name, param in model.named_parameters():
       print(f"{name}: requires_grad={param.requires_grad}")
   ```
3. **损失监控**: 分别记录 MSE 和 SIGReg 损失，观察训练动态

#### 8.2.3 性能优化
1. **混合精度训练**: 使用 `torch.cuda.amp` 加速训练
2. **梯度累积**: 当 GPU 内存不足时，使用梯度累积模拟大批次
3. **数据预加载**: 使用 `pin_memory=True` 和多进程加载

### 8.3 测试检查清单

在开始完整训练前，Codex 必须验证以下项目：

- [ ] **模型初始化**: 所有模块正确初始化，编码器成功加载预训练权重
- [ ] **参数冻结**: 验证 `v_encoder` 和 `l_encoder` 的 `requires_grad=False`
- [ ] **前向传播**: 单个 batch 能够成功完成前向传播，输出形状正确
- [ ] **损失计算**: MSE 和 SIGReg 损失都能正常计算，数值合理
- [ ] **反向传播**: 梯度只在投影器和预测器中传播，编码器无梯度
- [ ] **四象限测试**: 分别测试 V->V, V->L, L->V, L->L 四种模态组合
- [ ] **数据加载**: DataLoader 能够正常迭代，无内存泄漏
- [ ] **检查点保存**: 能够保存和加载检查点，恢复训练状态

### 8.4 常见错误排查

| 错误现象 | 可能原因 | 解决方案 |
|----------|-----|----------|
| `RuntimeError: CUDA out of memory` | 批次大小过大 | 减小 `batch_size` 或使用梯度累积 |
| `KeyError: 'encoder'` | 检查点键名不匹配 | 使用 `strict=False` 加载 |
| 损失为 NaN | 学习率过大或梯度爆炸 | 降低学习率，添加梯度裁剪 |
| SIGReg 损失为 0 | 输入格式错误 | 检查输入是否为 `(T, B, D)` 格式 |
| 训练不收敛 | `lambda_sigreg` 过大 | 降低 SIGReg 权重到 0.05-0.1 |
| 文本编码器报错 | Tokenizer 配置错误 | 确保 `padding="max_length"` |

---

## 第九部分：实施路线图 (Implementation Roadmap)

### 9.1 开发阶段划分

#### Phase 1: 基础模块实现 
1. **SIGReg 损失函数** (`src/losses/sigreg.py`)
   - 从 LeWM 移植代码
   - 添加单元测试验证正确性

2. **文本编码器封装** (`src/models/text_encoder.py`)
   - 集成 SigLIP Text Encoder
   - 实现 Tokenizer 包装

3. **投影器模块** (`src/models/projectors.py`)
   - 实现四个独立投影器
   - 验证输出维度一致性

#### Phase 2: 核心模型构建 
4. **多模态预测器** (`src/models/multimodal_predictor.py`)
   - 实现模态嵌入机制
   - 动态位置编码生成
   - 单元测试四象限预测

5. **多模态 JEPA 主模型** (`src/models/multimodal_jepa.py`)
   - 整合所有子模块
   - 实现完整前向传播
   - 梯度流验证

#### Phase 3: 数据管道 
6. **多模态数据集** (`src/datasets/multimodal_dataset.py`)
   - JSONL 解析
   - 视频帧加载
   - 文本预处理

7. **DataLoader 集成** (`src/datasets/data_manager.py`)
   - 添加 `multimodal` 数据集类型
   - 自定义 collate_fn

#### Phase 4: 训练流程 
8. **训练脚本** (`app/multimodal_jepa/train.py`)
   - 实现训练主循环
   - 优化器和调度器配置
   - 日志和检查点管理

9. **分布式训练支持**
   - DDP 包装
   - 多 GPU 测试

#### Phase 5: 测试与调优 
10. **单元测试**
    - 每个模块的独立测试
    - 集成测试

11. **超参数调优**
    - `lambda_sigreg` 搜索
    - 学习率调整
    - 批次大小优化

### 9.2 里程碑检查点

| 里程碑 | 验收标准 | 预计时间 |
|--------|----------|------|
| M1: 基础模块完成 | 所有子模块通过单元测试 | Week 2 |
| M2: 模型前向传播 | 单 batch 成功完成四象限预测 | Week 4 |
| M3: 数据管道就绪 | DataLoader 正常迭代 | Week 5 |
| M4: 训练流程运行 | 完成 1 epoch 训练无错误 | Week 6 |
| M5: 分布式训练 | 多 GPU 训练稳定运行 | Week 7 |
| M6: 模型收敛 | 验证集损失持续下降 | Week 8 |

### 9.3 风险与应对

| 风险 | 影响 | 应对措施 |
|------|---|----------|
| SigLIP 模型下载失败 | 阻塞开发 | 提前下载并缓存模型权重 |
| 显存不足 | 无法训练 | 使用梯度累积或减小批次 |
| 训练不收敛 | 模型无效 | 调整 `lambda_sigreg`，检查梯度流 |
| 数据加载慢 | 训练效率低 | 增加 `num_workers`，使用缓存 |
| 多模态对齐差 | 跨模态预测失败 | 增加 SIGReg 权重，延长训练 |

---

## 第十部分：配置文件示例 (Configuration Example)

### 10.1 训练配置 YAML

```yaml
# app/multimodal_jepa/configs/multimodal_base.yaml

# 模型配置
model:
  # 编码器
  v_encoder:
    checkpoint_path: '/data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt'
    img_size: 384
    patch_size: 16
    tubelet_size: 2
    embed_dim: 768
    freeze: true
  
  l_encoder:
    model_name: 'google/siglip-large-patch16-384'
    max_length: 77
    embed_dim: 1024
    freeze: true
  
  # 投影器
  projectors:
    shared_dim: 512
    hidden_dim: 1024
  
  # 预测器
  predictor:
    predictor_dim: 384
    depth: 12  # ✅ 与 V-JEPA 2.1 ViT-B 对齐
    num_heads: 8
    mlp_ratio: 4.0
    use_3d_pos_for_video: true  # ✅ 视频使用 3D 位置编码

# 数据配置
data:
  train_jsonl: '/data/eku/vjepa_state_transitions.jsonl'
  val_split: 0.1
  img_size: 384
  batch_size: 16
  num_workers: 2
  pin_memory: true

# 训练配置
training:
  num_epochs: 100
  log_interval: 100
  val_interval: 5
  save_interval: 10
  
  # 优化器
  optimizer:
    type: 'AdamW'
    lr: 1e-4
    betas: [0.9, 0.999]
    weight_decay: 0.05
  
  # 学习率调度
  scheduler:
    type: 'WarmupCosine'
    warmup_ratio: 0.1
    min_lr: 1e-6
  
  # 损失函数
  loss:
    lambda_sigreg: 0.1
    sigreg_knots: 17
    sigreg_num_proj: 1024
  
  # 梯度裁剪
  grad_clip: 1.0

# 分布式训练
distributed:
  enabled: true
  backend: 'nccl'

# 日志与检查点
logging:
  save_dir: './checkpoints'
  log_dir: './logs'
  wandb:
    enabled: false
    project: 'multimodal-jepa'
    entity: 'your-entity'
```

---

## 总结：关键设计决策回顾

### 核心架构约束
1. ✅ **编码器冻结**: `v_encoder` 和 `l_encoder` 完全冻结，`requires_grad=False`
2. ✅ **投影器独立**: Context 和 Target 侧投影器权重完全独立，不共享
3. ✅ **去 EMA**: 移除 Momentum Encoder，Target 分支用 `torch.no_grad()` 包裹
4. ✅ **SIGReg 正则**: 作用在所有投影器输出上，防止表征坍塌
5. ✅ **Modality Masking**: 单一 Predictor 通过模态嵌入处理四象限预测
6. ✅ **序列长度保留**: 不做全局池化，保持 token 序列结构

### 张量形状关键点
- Context 视频: `[B, 3, 32, 384, 384]` → `[B, 9216, 512]`
- Target 视频: `[B, 3, 2, 384, 384]` → `[B, 576, 512]`
- Context 文本: `List[str]` → `[B, 77, 512]`
- Target 文本: `List[str]` → `[B, 77, 512]`
- SIGReg 输入: `[N_ctx + N_tgt, B, 512]`

### Codex 必须遵守的纪律
1. 使用 `strict=False` 加载预训练权重
2. Target 分支完整包裹在 `with torch.no_grad():`
3. SigLIP Tokenizer 必须设置 `padding="max_length"`
4. 提取 `last_hidden_state` 而非 `pooler_output`
5. SIGReg 输入格式为 `(T, B, D)`

---



---

## 附录 A：Predictor 架构修正说明 (2026-05-12)

### 修正背景
经过与 V-JEPA 2.1 ViT-B 384 预训练检查点的对比，发现原方案中 Predictor 深度设置不足。

### 关键修正

#### 1. Predictor 深度
- **原方案**: `depth=6` (6 层 Transformer)
- **修正后**: `depth=12` (12 层 Transformer) ✅
- **理由**: 与 V-JEPA 2.1 ViT-B 保持一致，提供足够的表征能力处理长序列（9216 tokens）

## 2. 位置编码策略
- **原方案**: 统一使用 1D sincos 位置编码
- **修正后**: 混合位置编码策略 ✅
  - 视频输入: 3D sincos (T, H, W) - 保留空间结构
  - 文本输入: 1D sincos - 适配序列特性
- **理由**: 视频的空间结构信息对预测至关重要，不应丢弃

#### 3. 完整配置对比

| 参数 | V-JEPA 2.1 ViT-B | 原方案 | 修正后 |
|------|---------|--------|--------|
| `predictor_dim` | 384 | 384 | 384 ✅ |
| `depth` | **12** | 6 ❌ | **12** ✅ |
| `num_heads` | 8 | 8 | 8 ✅ |
| `mlp_ratio` | 4.0 | 4.0 | 4.0 ✅ |
| 位置编码 | 3D sincos | 1D sincos ❌ | 混合 (3D+1D) ✅ |
| 模态嵌入 | 无 | 有 ✅ | 有 ✅ |

### 实现要点

#### Predictor 初始化
```python
class MultimodalPredictor(nn.Module):
    def __init__(
        self,
        shared_dim=512,
        predictor_dim=384,
        depth=12,  # ✅ 12 层
        num_heads=8,
        mlp_ratio=4.0,
        use_3d_pos_for_video=True,  # ✅ 新增参数
        img_size=384,
        patch_size=16,
        tubelet_size=2,
    ):
      super().__init__()
     
        # 模态嵌入
        self.modality_embed = nn.Embedding(2, shared_dim)
        
        # 输入投影
        self.input_proj = nn.Linear(shared_dim, predictor_dim)
        
        # 3D 位置编码参数（用于视频）
        self.use_3d_pos_for_video = use_3d_pos_for_video
        self.img_size = img_size
     self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        
        # 12 层 Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_rope=False,
            )
            for _ in range(depth)  # ✅ 12 层
        ])
        
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, shared_dim)
```

#### 动态位置编码生成
```python
def _get_pos_embed(self, seq_len, ctx_mod, tgt_mod, device):
    """
    根据模态类型动态生成位置编码
    """
    embed_dim = self.input_proj.out_features
    
    # 判断是否需要 3D 位置编码
    if self.use_3d_pos_for_video and (ctx_mod == 'V' or tgt_mod == 'V'):
        # 视频：使用 3D sincos 位置编码
        # 需要根据 seq_len 推断 (T, H, W) 维度
        if seq_len == 9216:  # Context 视频: 32 帧
            grid_depth = 32 // self.tubelet_size  # 16
         grid_size = self.img_size // self.patch_size  # 24
        elif seq_len == 576:  # Target 视频: 2 帧
            grid_depth = 2 // self.tubelet_size  # 1
          grid_size = self.img_size // self.patch_size  # 24
        else:
         # 文本或其他长度，回退到 1D
            return self._get_1d_pos_embed(seq_len, embed_dim, device)
        
        # 生成 3D 位置编码
        from src.models.utils.pos_embs import get_3d_sincos_pos_embed
        pos_embed = get_3d_sincos_pos_embed(
          embed_dim, grid_size, grid_depth, cls_token=False
        )
        return torch.from_numpy(pos_embed).float().unsqueeze(0).to(device)
    else:
   # 文本：使用 1D sincos 位置编码
        return self._get_1d_pos_embed(seq_len, embed_dim, device)

def _get_1d_pos_embed(self, seq_len, embed_dim, device):
    ""1D sincos 位置编码"""
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    dim_t = torch.arange(embed_dim, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / embed_dim)
    
    pos_embed = pos / dim_t
    pos_embed[:, 0::2] = torch.sin(pos_embed[:, 0::2])
    pos_embed[:, 1::2] = torch.cos(pos_embed[:, 1::2])
    
    return pos_embed.unsqueeze(0)
```

### 预期影响

#### 性能提升
- **表征能力**: 12 层深度提供更强的建模能力
- **空间感知**: 3D 位置编码保留视频的空间结构信息
- **跨模态对齐**: 更深的网络有助于学习视频-文本的共享表征空间

#### 计算成本
- **参数量**: 约增加 2 倍（6 层 → 12 层）
- **显存**: 增加约 30-40%（取决于批次大小）
- **训练时间**: 增加约 20-30%（Transformer 计算高效）

#### 缓解策略
如果资源受限，可以考虑：
1. 使用梯度检查点 (`use_activation_checkpointing=True`)
2. 减小批次大小
3. 使用混合精度训练 (`torch.cuda.amp`)
4. 折中方案：使用 8-10 层深度

### Codex 实现检查清单

在实现 Multimodal Predictor 时，Codex 必须确保：

- [ ] `depth=12`，不是 6
- [ ] 实现 `use_3d_pos_for_video` 参数
- [ ] 根据模态动态选择位置编码类型
- [ ] 从 `src.models.utils.pos_embs` 导入 `get_3d_sincos_pos_embed`
- [ ] 正确推断视频的 (T, H, W) 维度
- [ ] 文本回退到 1D 位置编码
- [ ] 所有 12 层 Block 使用相同的配置

---

**修正完成。本附录应与主文档一起交付给 Codex。**


