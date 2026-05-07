# test_load.py
import os
import subprocess
import json
import sys
import torch
import torch.nn.functional as F  # 修复：补充 F 的导入

current_dir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, current_dir)

# 统一设备设置：优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Classifier Pretrained weights loaded with msg: {}".format(msg))

# ----------------- 1. 初始化 Encoder -----------------
from src.models.vision_transformer import vit_giant_xformers

# 将 Encoder 部署到正确的设备
encoder = vit_giant_xformers(img_size=[224, 224], patch_size=16).to(device)

# 加载权重，建议先 map 到 CPU，再由 .to(device) 管理，防止显存冗余
ckpt = torch.load('/data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt', map_location='cpu')
encoder.load_state_dict(ckpt['encoder'], strict=False)
encoder.eval()

# ----------------- 2. 初始化 Classifier -----------------
from src.models.attentive_pooler import AttentiveClassifier
classifier_model_path = "/data/vjepa2/ssv2-vitg-384-64x2x3.pt"

# ⚠️ 注意：vit_huge 默认输出维度是 1280，所以这里的 embed_dim 必须设为 1280，
# 否则接下来 encoder 的输出送进 classifier 时会报 size mismatch。
classifier = AttentiveClassifier(
    embed_dim=1408, 
    num_heads=16, 
    depth=4, 
    num_classes=174
).to(device).eval()

load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

# ----------------- 3. 准备标签数据 -----------------
ssv2_classes_path = "ssv2_classes.json"
if not os.path.exists(ssv2_classes_path):
    command = [
        "wget",
        "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json",
        "-O",
        "ssv2_classes.json",
    ]
    subprocess.run(command)
    print("Downloading SSV2 classes")

def get_vjepa_video_classification_results(classifier, out_patch_features_pt):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features_pt)

    print(f"\nClassifier output shape: {out_classifier.shape}")
    print("Top 5 predicted class names:")
    
    top5_indices = out_classifier.topk(5).indices[0]
    # 修复：明确指定 dim=-1 以消除警告
    top5_probs = F.softmax(out_classifier.topk(5).values[0], dim=-1) * 100.0  
    
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob:.2f}%)")

# ----------------- 4. 模拟输入并运行 -----------------
# 修复：将随机输入也转移到同一设备（GPU），并将 batch 设为 1
x = torch.randn(2, 3, 224, 224).to(device)

with torch.no_grad():
    out = encoder(x)

print(f"\nEncoder output shape: {out.shape}")  # 应该输出 [1, 196, 1280]

get_vjepa_video_classification_results(classifier, out)