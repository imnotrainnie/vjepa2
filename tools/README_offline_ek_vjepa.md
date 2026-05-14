# Offline V-JEPA 2.1 EK Pipeline

## Files

- `extract_features.py`: 从 RGB 帧目录提取全局特征并保存为 `.pt`。
- `eval_ek_probe_offline.py`: 按 V-JEPA2.1 的 encoder + predictor + probe 设计做 EK `recall@5` 测试。

## 1) Feature Extraction

```bash
python /data/vjepa2/tools/extract_features.py \
  --input-root /nvme/vjepa_data_32f_rgb \
  --manifest /data/ek/ek_manifest_test_v2.csv \
  --weights /data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  --output-root /nvme/vjepa_features \
  --encoder-model vit_base \
  --batch-size 4 \
  --num-workers 4 \
  --precision bf16
```

快速冒烟（只跑 1 条）：

```bash
python /data/vjepa2/tools/extract_features.py --max-clips 1
```

## 2) EK Probe Recall@5

```bash
python /data/vjepa2/tools/eval_ek_probe_offline.py \
  --input-root /nvme/vjepa_data_32f_rgb \
  --manifest /data/ek/ek_manifest_test_v2.csv \
  --weights /data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  --probe /data/vjepa2/ek100-vitg-384.pt \
  --ek-train-csv /data/epic-kitchens-100-annotations/EPIC_100_train.csv \
  --max-clips 200 \
  --precision bf16
```

## 兼容性说明

- 当前脚本按 V-JEPA 2.1 的原生 token 流程实现（context + 2 future-frame tokens -> probe）。
- `ek100-vitg-384.pt` 是 `vitG` probe；若 backbone embed dim 与 probe 不一致，脚本会报错并提示更换匹配权重/probe。
