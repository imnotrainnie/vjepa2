# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    # # 👇 ===== 加在这里 ===== 👇
    # print(f"\n[DEBUG] PT Model Missing keys: {len(msg.missing_keys)} 个")
    # if len(msg.missing_keys) > 0:
    #     print(f"[DEBUG] 缺失的键前 10 个: {msg.missing_keys[:10]}\n")
    # # 👆 ==================== 👆

def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform


def get_video():
    vr = VideoReader("sample_video.mp4")
    # choosing some frames here, you can define more complex sampling strategy
    frame_idx = np.arange(0, 128, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    return video


def forward_vjepa_video(model_hf, model_pt, hf_transform, pt_transform):
    with torch.inference_mode():
        video = get_video()  
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  
        
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        
# 👇 ===== 修改后的数据对比代码 ===== 👇
        print(f"\n[DEBUG] PT Transform 数据均值: {x_pt.mean().item():.4f}, 标准差: {x_pt.std().item():.4f}")
        print(f"[DEBUG] HF Transform 数据均值: {x_hf.mean().item():.4f}, 标准差: {x_hf.std().item():.4f}")
        print(f"[DEBUG] PT Shape: {x_pt.shape}, HF Shape: {x_hf.shape}")
        
        # 将 HF 的 [B, T, C, H, W] 转换为 [B, C, T, H, W] 以对齐 PT
        x_hf_aligned = x_hf.permute(0, 2, 1, 3, 4)
        
        # 对比像素级差异
        input_diff = torch.abs(x_pt - x_hf_aligned).sum()
        print(f"[DEBUG] 输入数据绝对值差异: {input_diff:.4f}\n")
        # 👆 ========================================= 👆

        out_patch_features_pt = model_pt(x_pt)
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    return out_patch_features_hf, out_patch_features_pt


def get_vjepa_video_classification_results(classifier, out_patch_features_pt):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features_pt)

    print(f"Classifier output shape: {out_classifier.shape}")

    print("Top 5 predicted class names:")
    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0]) * 100.0  # convert to percentage
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")

    return


def run_sample_inference():
    # HuggingFace model repo name
    hf_model_name = (
        "facebook/vjepa2-vitg-fpc64-384"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )
    # Path to local PyTorch weights
    pt_model_path = "/data/vjepa2/vitg-384.pt"

    sample_video_path = "sample_video.mp4"
    # Download the video if not yet downloaded to local path
    if not os.path.exists(sample_video_path):
        video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
        command = ["wget", video_url, "-O", sample_video_path]
        subprocess.run(command)
        print("Downloading video")

    # Initialize the HuggingFace model, load pretrained weights
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.cuda().eval()

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    img_size = hf_transform.crop_size["height"]  # E.g. 384, 256, etc.

    # Initialize the PyTorch model, load pretrained weights
    model_pt = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
    model_pt.cuda().eval()
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)

    # Build PyTorch preprocessing transform
    pt_video_transform = build_pt_video_transform(img_size=img_size)

    # Inference on video
    out_patch_features_hf, out_patch_features_pt = forward_vjepa_video(
        model_hf, model_pt, hf_transform, pt_video_transform
    )
    # fake_input = torch.ones(1, 3, 64, 384, 384).cuda()
    # # 👇 ===== 加在这里：对比第一层输出 ===== 👇
    # # with torch.no_grad():
    # #     try:
    # #         pt_patch = model_pt.patch_embed(fake_input)
    # #         hf_patch = model_hf.embeddings.patch_embeddings(fake_input) 
    # #         print(f"[DEBUG] FAKE Patch Embed Diff: {torch.abs(pt_patch - hf_patch).sum():.6f}")
    # #     except Exception as e:
    # #         print(f"[DEBUG] Patch 对比失败，可能是 HF 属性名不同: {e}")
    # # 👆 =================================== 👆
    # out_pt = model_pt(fake_input)
    # out_hf = model_hf.get_vision_features(fake_input) # 确保 HF 接收张量
    # print(f"FAKE Absolute difference sum:  {torch.abs(out_pt - out_hf).sum():.6f}")
    print(
        f"""
        Inference results on video:
        HuggingFace output shape: {out_patch_features_hf.shape}
        PyTorch output shape:     {out_patch_features_pt.shape}
        Absolute difference sum:  {torch.abs(out_patch_features_pt - out_patch_features_hf).sum():.6f}
        Close: {torch.allclose(out_patch_features_pt, out_patch_features_hf, atol=1e-3, rtol=1e-3)}
        """
    )

    # Initialize the classifier
    classifier_model_path = "/data/vjepa2/ssv2-vitg-384-64x2x3.pt"
    classifier = (
        AttentiveClassifier(embed_dim=model_pt.embed_dim, num_heads=16, depth=4, num_classes=174).cuda().eval()
    )
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # Download SSV2 classes if not already present
    ssv2_classes_path = "ssv2_classes.json"
    if not os.path.exists(ssv2_classes_path):
        command = [
            "wget",
            "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/"
            "something-something-v2-id2label.json",
            "-O",
            "ssv2_classes.json",
        ]
        subprocess.run(command)
        print("Downloading SSV2 classes")

    get_vjepa_video_classification_results(classifier, out_patch_features_pt)


if __name__ == "__main__":
    # Run with: `python -m notebooks.vjepa2_demo`
    run_sample_inference()
