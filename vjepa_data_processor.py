import pandas as pd
import numpy as np
import os
import json
import shutil

# ==========================================
# 1. 路径与参数配置 (建议根据实际情况修改)
# ==========================================
# 输入文件
CSV_PATH = '/data/epic-kitchens-100-annotations/EPIC_100_validation.csv'
VISOR_JSON_DIR = '/data/visor_assets/GroundTruth-SparseAnnotations/val' ###改！！！！！！

# 输出目录 (强烈建议设在 NVMe 分区以加速后续 V-JEPA 训练)
OUTPUT_BASE_DIR = '/nvme/vjepa_ready_data' 
MANIFEST_PATH = '/data/vjepa_final_manifest.csv'

# 动作类别定义 (基于你提供的 verb-class)
TARGET_VERB_CLASSES = [0, 1, 2, 3, 4]  # take, put, wash, open, close
CLIPS_PER_CLASS = 40
TOTAL_FRAMES = 64  # V-JEPA 所需的总帧数

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def generate_manifest():
    print(">>> 正在生成 V-JEPA 数据抽取清单...")
    df = pd.read_csv(CSV_PATH)
    
    # 基础过滤：
    # 1. 属于目标动作类
    # 2. 动作长度必须足够进行 64 帧采样（建议至少 64 帧）
    df['duration_frames'] = df['stop_frame'] - df['start_frame']
    filtered = df[
        (df['verb_class'].isin(TARGET_VERB_CLASSES)) & 
        (df['duration_frames'] >= TOTAL_FRAMES)
    ].copy()

    # 检查哪些视频在 VISOR 中有 Mask
    available_visor_videos = [f.replace('.json', '') for f in os.listdir(VISOR_JSON_DIR) if f.endswith('.json')]
    filtered = filtered[filtered['video_id'].isin(available_visor_videos)]

    final_selection = []

    for v_class in TARGET_VERB_CLASSES:
        class_pool = filtered[filtered['verb_class'] == v_class]
        
        # 采样策略：取前 40 条
        # 如果数据充沛，可以按 duration_frames 排序选最紧凑的动作
        samples = class_pool.head(CLIPS_PER_CLASS)
        
        for _, row in samples.iterrows():
            # 【等距均匀采样逻辑】
            # 在动作的 start 和 stop 之间均匀取出 64 个点
            sampled_indices = np.linspace(row['start_frame'], row['stop_frame'], num=TOTAL_FRAMES).astype(int)
            
            final_selection.append({
                'narration_id': row['narration_id'],
                'video_id': row['video_id'],
                'verb_class': v_class,
                'verb': row['verb'],
                'noun': row['noun'],
                'sampled_frames': sampled_indices.tolist(),
                'context_start_frame': sampled_indices[0],
                'target_stop_frame': sampled_indices[-1]
            })

    manifest_df = pd.DataFrame(final_selection)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    print(f">>> 清单已保存至: {MANIFEST_PATH}")
    return manifest_df

def process_pipeline(manifest_df):
    print("\n>>> 开始数据抽取流水线 (按 Video ID 分组以节省空间)...")
    
    # 按 Video ID 分组，这样每个几十GB的大包只需要下载/解压一次
    grouped = manifest_df.groupby('video_id')
    
    for video_id, clips in grouped:
        print(f"\n[任务] 正在处理视频: {video_id} (包含 {len(clips)} 个目标片段)")
        
        # -------------------------------------------------------------------
        # 步骤 1: 下载与解压 (此处为逻辑说明，需配合官方下载工具)
        # -------------------------------------------------------------------
        # print(f"  -> 请执行: python epic_downloader.py --ids {video_id} --parts rgb")
        
        # 假设图片已下载到 /data/rgb_frames/{video_id}/ 目录下
        video_source_dir = f"/data/rgb_frames/{video_id}"
        
        # 步骤 2: 遍历该视频下的所有目标 Clip
        for _, clip in clips.iterrows():
            n_id = clip['narration_id']
            frame_list = clip['sampled_frames']
            
            # 创建输出文件夹 (在 NVMe)
            target_dir = os.path.join(OUTPUT_BASE_DIR, n_id)
            os.makedirs(target_dir, exist_ok=True)
            
            print(f"    -> 提取片段 {n_id}: 共 {len(frame_list)} 帧")
            
            # 步骤 3: 物理拷贝/提取帧图片
            for i, f_num in enumerate(frame_list):
                # 构造文件名，EPIC 通常是 frame_0000000XXX.jpg 格式
                frame_filename = f"frame_{str(f_num).zfill(10)}.jpg"
                src_path = os.path.join(video_source_dir, frame_filename)
                dst_path = os.path.join(target_dir, f"vjepa_frame_{str(i).zfill(2)}.jpg")
                
                # if os.path.exists(src_path):
                #     shutil.copy(src_path, dst_path)
            
            # 步骤 4: 同步处理 VISOR Mask
            # 此处应调用你手头的 P01_107.json 等文件，提取这 64 帧对应的坐标并生成 PNG
            # print(f"    -> 正在从 {video_id}.json 提取 Mask...")

        # -------------------------------------------------------------------
        # 步骤 5: 该视频处理完毕，清理空间
        # -------------------------------------------------------------------
        # print(f"  -> 处理完毕，建议删除源码: rm -rf {video_source_dir}")

if __name__ == "__main__":
    # 执行清单生成
    manifest = generate_manifest()
    
    # 打印执行计划预览
    process_pipeline(manifest)
    
    print("\n>>> 脚本运行结束。")
    print("提示：请根据生成的清单，使用官方工具下载对应的 Video ID 包，然后执行帧提取。")