# V-JEPA 2 源代码中文注释版本

本文档提供了 V-JEPA 2 源代码各个模块的详细中文注释。

## 1. 视频数据集 (`video_dataset.py`)

```python
"""
视频数据集模块

功能: 从磁盘加载视频数据并进行预处理

核心类:
- VideoDataset: 视频数据集主类

关键特性:
1. 使用 decord 库高效加载视频
2. 支持多种帧采样策略
3. 支持图像和视频混合训练
4. 支持多数据集加权采样
"""

def make_videodataset(...):
    """
    创建视频数据集和数据加载器的工厂函数
    
    参数:
        data_paths: 数据路径列表 (CSV 或 NPY 文件)
        batch_size: 批次大小
        frames_per_clip: 每个视频片段的帧数
        frame_step: 帧间隔（采样步长）
        duration: 视频持续时间（秒）
        fps: 帧率
        num_clips: 每个视频采样的片段数
        random_clip_sampling: 是否随机采样片段
        allow_clip_overlap: 是否允许片段重叠
        transform: 数据增强变换
        datasets_weights: 数据集权重（用于混合训练）
    
    返回:
        dataset: 数据集对象
        data_loader: 数据加载器
        dist_sampler: 分布式采样器
    """

class VideoDataset(torch.utils.data.Dataset):
    """
    视频数据集类
    
    初始化参数:
        data_paths: 数据文件路径（CSV 或 NPY 格式）
        frames_per_clip: 每个片段的帧数
        frame_step: 帧采样间隔
        num_clips: 采样的片段数量
        transform: 每帧的数据增强
        shared_transform: 所有帧共享的增强
        random_clip_sampling: 随机采样还是均匀采样
        filter_short_videos: 是否过滤过短的视频
    
    数据格式:
        CSV: 第一列是视频路径，第二列是标签
        NPY: 视频路径数组
    """
    
    def __getitem__(self, index):
        """
        获取单个样本
        
        返回:
            buffer: 视频帧列表，每个片段一个张量 [T, C, H, W]
            label: 标签
            clip_indices: 帧索引列表
        """
    
    def loadvideo_decord(self, sample, fpc):
        """
        使用 decord 加载视频
        
        帧采样策略:
        1. 如果视频足够长，将其均匀分成 num_clips 段
        2. 在每段中随机或均匀选择起始位置
        3. 从起始位置采样 frames_per_clip 帧
        
        返回:
            buffer: 视频帧数组 [T, H, W, 3]
            clip_indices: 每个片段的帧索引
        """
```

## 2. Vision Transformer 编码器 (`vision_transformer.py`)

```python
"""
Vision Transformer 编码器模块

功能: 将视频编码为 token 表示

核心类:
- VisionTransformer: ViT 编码器

关键特性:
1. 支持 2D 图像和 3D 视频输入
2. Tubelet 时空分块嵌入
3. 可选的 RoPE 或 SinCos 位置编码
4. 支持遮蔽机制
"""

class VisionTransformer(nn.Module):
    """
    Vision Transformer 编码器
    
    架构:
    输入视频 [B, C, T, H, W]
        ↓
    Patch Embedding (3D Conv)
        ↓
    Token 序列 [B, N, D]  (N = T'*H'*W', D = embed_dim)
        ↓
    + 位置编码
        ↓
    应用遮蔽（可选）
        ↓
    L 个 Transformer Blocks
        ↓
    Layer Normalization
        ↓
    输出特征 [B, N, D]
    
    初始化参数:
        img_size: 输入尺寸 (H, W)
        patch_size: 空间 patch 大小
        num_frames: 输入帧数
        tubelet_size: 时间 tubelet 大小（几帧合成一个时间块）
        embed_dim: 嵌入维度
        depth: Transformer 层数
        num_heads: 注意力头数
        mlp_ratio: MLP 隐藏层比例
        use_rope: 是否使用旋转位置编码
    """
    
    def forward(self, x, masks=None):
        """
        前向传播
        
        参数:
            x: 输入视频 [B, C, T, H, W] 或图像 [B, C, H, W]
            masks: 遮蔽索引列表，指定要保留的 token 位置
        
        处理流程:
        1. Patch Embedding: 将视频转换为 token 序列
        2. 添加位置编码（如果不使用 RoPE）
        3. 应用遮蔽（如果提供）
        4. 通过 Transformer blocks
        5. Layer Normalization
        
        返回:
            编码后的特征 [B, N', D]  (N' 是遮蔽后的 token 数)
        """
    
    def interpolate_pos_encoding(self, x, pos_embed):
        """
        插值位置编码以适应不同尺寸的输入
        
        当输入尺寸与预训练尺寸不同时:
        - 使用三线性插值（3D）或双线性插值（2D）
        - 调整位置编码的空间和时间维度
        """
```

## 3. 预测器 (`predictor.py`)

```python
"""
预测器模块

功能: 根据上下文 token 预测被遮蔽位置的表示

核心类:
- VisionTransformerPredictor: 预测器网络
"""

class VisionTransformerPredictor(nn.Module):
    """
    Vision Transformer 预测器
    
    架构:
    上下文 tokens [B, N_ctx, D_enc]
        ↓
    线性映射到预测器维度 [B, N_ctx, D_pred]
        ↓
    + 位置编码
        ↓
    拼接 Mask Tokens [B, N_ctx + N_tgt, D_pred]
        ↓
    排序（按位置索引）
        ↓
    L 个 Transformer Blocks
        ↓
    提取目标位置的输出
        ↓
    投影到目标维度 [B, N_tgt, D_out]
    
    初始化参数:
        embed_dim: 编码器嵌入维度
        predictor_embed_dim: 预测器内部维度
        out_embed_dim: 输出维度（默认等于编码器维度）
        depth: 预测器层数
        use_mask_tokens: 是否使用可学习的 mask tokens
        num_mask_tokens: mask token 数量
    """
    
    def forward(self, x, masks_x, masks_y, mask_index=1, has_cls=False):
        """
        前向传播
        
        参数:
            x: 上下文 tokens [B, N_ctx, D]
            masks_x: 上下文 token 的位置索引
            masks_y: 目标 token 的位置索引
            mask_index: 使用哪个 mask token（多任务时）
            has_cls: 是否包含 CLS token
        
        处理流程:
        1. 将上下文映射到预测器维度
        2. 为上下文添加位置编码
        3. 初始化目标位置的 mask tokens 并添加位置编码
        4. 拼接上下文和目标 tokens
        5. 按位置索引排序
        6. 通过 Transformer blocks
        7. 提取目标位置的预测
        8. 投影到输出维度
        
        返回:
            目标位置的预测表示 [B, N_tgt, D_out]
        """
```

## 4. 遮蔽生成器 (`masks/multiseq_multiblock3d.py`)

```python
"""
遮蔽生成模块

功能: 生成训练时使用的遮蔽策略

核心类:
- MaskCollator: 遮蔽收集器（用于 DataLoader）
- _MaskGenerator: 遮蔽生成器
"""

class MaskCollator:
    """
    遮蔽收集器
    
    功能:
    1. 作为 DataLoader 的 collate_fn
    2. 对每个批次生成遮蔽
    3. 支持不同帧数的样本
    """
    
    def __call__(self, batch):
        """
        处理批次数据
        
        输入:
            batch: 视频样本列表
        
        处理流程:
        1. 按帧数分组样本
        2. 对每组生成遮蔽
        3. 返回分组后的批次和遮蔽
        
        返回:
            fpc_collations: 按帧数分组的列表
                每个元素: (collated_batch, masks_enc, masks_pred)
        """

class _MaskGenerator:
    """
    遮蔽生成器
    
    遮蔽策略:
    1. 在 3D 时空网格中采样矩形块
    2. 矩形块作为预测目标
    3. 剩余区域作为上下文
    
    参数:
        spatial_pred_mask_scale: 空间遮蔽比例范围 (min, max)
        temporal_pred_mask_scale: 时间遮蔽比例范围 (min, max)
        aspect_ratio: 遮蔽块宽高比范围 (min, max)
        npred: 预测块数量
        max_context_frames_ratio: 上下文最大时间范围比例
    """
    
    def __call__(self, batch_size):
        """
        生成遮蔽
        
        步骤:
        1. 使用种子采样遮蔽块大小 (t, h, w)
        2. 为每个样本随机采样块位置
        3. 生成编码器遮蔽（上下文索引）
        4. 生成预测器遮蔽（目标索引）
        5. 截断到最小长度以便批处理
        
        返回:
            collated_masks_enc: 编码器遮蔽 [B, N_ctx]
            collated_masks_pred: 预测器遮蔽 [B, N_tgt]
        """
```

## 5. 训练流程 (`app/vjepa/train.py`)

```python
"""
V-JEPA 2 训练脚本

核心函数:
- main(): 主训练函数
"""

def main(args, resume_preempt=False):
    """
    主训练函数
    
    训练流程:
    
    1. 初始化阶段:
       - 解析配置
       - 初始化分布式环境
       - 创建模型（编码器、预测器、目标编码器）
       - 创建数据加载器和遮蔽生成器
       - 初始化优化器和学习率调度器
    
    2. 训练循环 (每个 epoch):
       for itr in range(iterations_per_epoch):
           
           # 加载数据
           clips, masks_enc, masks_pred = load_batch()
           
           # 前向传播 - 目标编码器（无梯度）
           with torch.no_grad():
               h = target_encoder(clips)  # 完整视频编码
               h = layer_norm(h)
           
           # 前向传播 - 上下文编码和预测
           z = encoder(clips, masks_enc)  # 编码上下文
           z = predictor(z, masks_enc, masks_pred)  # 预测目标
           
           # 计算损失
           h_masked = apply_masks(h, masks_pred)  # 提取目标位置
           loss = |z - h_masked|^p / p  # L1/L2 损失
           
           # 反向传播
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
           
           # 动量更新目标编码器
           for param_q, param_k in zip(encoder.params, target_encoder.params):
               param_k = m * param_k + (1 - m) * param_q
           
           # 更新学习率和动量
           scheduler.step()
           m = next(momentum_scheduler)
    
    3. 保存检查点
       save_checkpoint(encoder, predictor, target_encoder, optimizer)
    
    损失函数:
        loss = mean(|predictor_output - target_representation|^loss_exp) / loss_exp
        - loss_exp=1: L1 损失
        - loss_exp=2: L2 损失
    
    EMA 动量更新:
        target_encoder = m * target_encoder + (1-m) * encoder
        - m 从 ema[0] 线性增加到 ema[1]
        - 典型值: ema = [0.996, 1.0]
    """
```

## 6. 评估流程 (`evals/video_classification_frozen/eval.py`)

```python
"""
视频分类评估脚本

功能: 在视频分类任务上评估预训练编码器
"""

def main(args_eval, resume_preempt=False):
    """
    评估主函数
    
    评估流程:
    
    1. 初始化:
       - 加载预训练编码器
       - 冻结编码器参数
       - 初始化注意力分类器
    
    2. 训练分类器:
       for epoch in range(num_epochs):
           for batch in train_loader:
               # 提取特征（无梯度）
               with torch.no_grad():
                   features = encoder(videos)
               
               # 分类
               logits = classifier(features)
               loss = cross_entropy(logits, labels)
               
               # 更新分类器
               loss.backward()
               optimizer.step()
    
    3. 验证:
       for batch in val_loader:
           # 多视角测试
           all_logits = []
           for segment in segments:
               for view in views:
                   features = encoder(crop(video, segment, view))
                   logits = classifier(features)
                   all_logits.append(logits)
           
           # 平均预测
           final_logits = mean(all_logits)
           predictions = argmax(final_logits)
       
       # 计算准确率
       top1_acc = accuracy(predictions, labels, k=1)
       top5_acc = accuracy(predictions, labels, k=5)
    """
```

## 7. 基础模块 (`models/utils/modules.py`)

```python
"""
基础网络模块

包含 Transformer 的基本构建块
"""

class Block(nn.Module):
    """
    Transformer 基础块
    
    架构:
    x → LayerNorm → Attention → + → LayerNorm → MLP → +
    ↑__________________________|    ↑_______________|
    
    组件:
    - norm1: 第一个 LayerNorm
    - attn: 注意力层（Attention 或 RoPEAttention）
    - drop_path: 随机深度
    - norm2: 第二个 LayerNorm  
    - mlp: MLP 层（标准 MLP 或 SwiGLU）
    """

class RoPEAttention(nn.Module):
    """
    旋转位置编码注意力
    
    特点:
    1. 将位置信息编码到 Q 和 K 的旋转中
    2. 支持 3D 位置（时间、高度、宽度）
    3. 每个维度独立旋转
    
    位置编码:
    - 深度维度: 旋转 Q[:d_dim] 和 K[:d_dim]
    - 高度维度: 旋转 Q[d_dim:d_dim+h_dim] 和 K[...]
    - 宽度维度: 旋转 Q[d_dim+h_dim:...] 和 K[...]
    """

def rotate_queries_or_keys(x, pos):
    """
    旋转查询或键向量
    
    RoPE 公式:
    x' = x * cos(θ) + rotate(x) * sin(θ)
    
    其中:
    - θ = pos * ω
    - ω_i = 1 / 10000^(2i/d)
    - rotate(x) 是将相邻元素配对并旋转 90°
    """

class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络
    
    公式:
    output = fc3(SiLU(fc1(x)) * fc2(x))
    
    相比标准 FFN:
    - 使用门控机制
    - SiLU 激活函数
    - 通常效果更好
    """
```
