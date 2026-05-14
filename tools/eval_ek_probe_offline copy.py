#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class EKFrameDataset(Dataset):
    def __init__(self, clips_root: Path, manifest_csv: Path, num_frames: int = 32, image_size: int = 384) -> None:
        self.num_frames = num_frames
        self.df = pd.read_csv(manifest_csv)
        for col in ["clip_id", "verb_class", "noun_class"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in {manifest_csv}")

        self.items: List[Dict[str, int | str]] = []
        for _, row in self.df.iterrows():
            clip_id = str(row["clip_id"])
            frames_dir = clips_root / clip_id / "frames"
            frame_paths = sorted(frames_dir.glob("*.jpg"))
            if len(frame_paths) < num_frames:
                continue
            self.items.append(
                {
                    "clip_id": clip_id,
                    "verb_class": int(row["verb_class"]),
                    "noun_class": int(row["noun_class"]),
                    "frames_dir": str(frames_dir),
                }
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        frames_dir = Path(str(item["frames_dir"]))
        frame_paths = sorted(frames_dir.glob("*.jpg"))[: self.num_frames]
        frames_tchw = []
        for fp in frame_paths:
            with Image.open(fp) as img:
                frames_tchw.append(self.transform(img.convert("RGB")))
        clip = torch.stack(frames_tchw, dim=0).permute(1, 0, 2, 3).contiguous()
        return str(item["clip_id"]), int(item["verb_class"]), int(item["noun_class"]), str(frames_dir), clip


def collate_fn(batch):
    clip_ids = [x[0] for x in batch]
    verbs = torch.tensor([x[1] for x in batch], dtype=torch.long)
    nouns = torch.tensor([x[2] for x in batch], dtype=torch.long)
    frame_dirs = [x[3] for x in batch]
    clips = torch.stack([x[4] for x in batch], dim=0)
    return clip_ids, verbs, nouns, frame_dirs, clips


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
        out[k] = v
    return out


def infer_probe_depth(sd: Dict[str, torch.Tensor]) -> int:
    block_indices = []
    for k in sd.keys():
        m = re.match(r"module\.pooler\.blocks\.(\d+)\.", k)
        if m is not None:
            block_indices.append(int(m.group(1)))
    if not block_indices:
        return 1
    return max(block_indices) + 2


def build_label_maps(train_csv: Path):
    tdf = pd.read_csv(train_csv)
    actions = set((int(v), int(n)) for v, n in zip(tdf["verb_class"].values, tdf["noun_class"].values))
    verbs = set(int(v) for v in tdf["verb_class"].values)
    nouns = set(int(n) for n in tdf["noun_class"].values)
    verb_map = {k: i for i, k in enumerate(verbs)}
    noun_map = {k: i for i, k in enumerate(nouns)}
    action_map = {k: i for i, k in enumerate(actions)}
    return verb_map, noun_map, action_map


def resolve_autocast_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32

def _detect_scale_hint(weights_name: str) -> Dict[str, str]:
    name = weights_name.lower()
    if "vjepa2_1_vitb" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-B/16",
            "encoder_fn": "vit_base",
            "checkpoint_key": "ema_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "768",
        }
    if "vjepa2_1_vitl" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-L/16",
            "encoder_fn": "vit_large",
            "checkpoint_key": "ema_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1024",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-G/16",
            "encoder_fn": "vit_gigantic_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1664",
        }
    if "vitg-384" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vitl" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-L/16",
            "encoder_fn": "vit_large",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1024",
        }
    if "vith" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-H/16",
            "encoder_fn": "vit_huge",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1280",
        }
    if "vitg" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1408",
        }
    return {
        "family": "Unknown",
        "scale": "Unknown",
        "encoder_fn": "Unknown",
        "checkpoint_key": "ema_encoder/encoder/target_encoder",
        "module": "src.models.vision_transformer or app.vjepa_2_1.models.vision_transformer",
        "embed_dim": "Unknown",
    }
"""
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitg_384" in name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vjepa2_1_vitG" in weights_name:
        return {
            "family": "V-JEPA 2.1",
            "scale": "ViT-G/16",
            "encoder_fn": "vit_gigantic_xformers",
            "checkpoint_key": "target_encoder",
            "module": "app.vjepa_2_1.models.vision_transformer",
            "embed_dim": "1664",
        }
    if "vitg-384" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1408",
        }
    if "vitl" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-L/16",
            "encoder_fn": "vit_large",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1024",
        }
    if "vith" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-H/16",
            "encoder_fn": "vit_huge",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1280",
        }
    if "vitg" in name:
        return {
            "family": "V-JEPA 2",
            "scale": "ViT-g/16",
            "encoder_fn": "vit_giant_xformers",
            "checkpoint_key": "target_encoder",
            "module": "src.models.vision_transformer",
            "embed_dim": "1408",
        }
    return {
        "family": "Unknown",
        "scale": "Unknown",
        "encoder_fn": "Unknown",
        "checkpoint_key": "ema_encoder/encoder/target_encoder",
        "module": "src.models.vision_transformer or app.vjepa_2_1.models.vision_transformer",
        "embed_dim": "Unknown",
    }
"""


def _infer_ckpt_embed_dim(state_dict: Dict[str, torch.Tensor]) -> int | None:
    candidate_keys = [
        "module.backbone.blocks.0.attn.qkv.weight",
        "blocks.0.attn.qkv.weight",
    ]
    for key in candidate_keys:
        if key in state_dict:
            tensor = state_dict[key]
            if tensor.ndim == 2:
                return int(tensor.shape[1])
    return None


def _print_checklist(
    args: argparse.Namespace,
    ckpt: Dict[str, object],
    enc_ckey: str,
    probe_embed_dim: int | None,
) -> None:
    hint = _detect_scale_hint(args.weights.name)
    ckpt_keys = sorted(list(ckpt.keys()))
    ckpt_embed_dim = None
    if enc_ckey in ckpt and isinstance(ckpt[enc_ckey], dict):
        ckpt_embed_dim = _infer_ckpt_embed_dim(ckpt[enc_ckey])

    print("\n[CHECKLIST] Checkpoint/Model/Probe compatibility")
    print("-" * 72)
    print(f"weights_path         : {args.weights}")
    print(f"weights_keys         : {ckpt_keys}")
    print(f"detected_family      : {hint['family']}")
    print(f"detected_scale       : {hint['scale']}")
    print(f"recommended_module   : {hint['module']}")
    print(f"recommended_encoder  : {hint['encoder_fn']}")
    print(f"recommended_ckpt_key : {hint['checkpoint_key']}")
    print(f"recommended_embeddim : {hint['embed_dim']}")
    print(f"cli_encoder_model    : {args.encoder_model}")
    print(f"resolved_ckpt_key    : {enc_ckey}")
    print(f"ckpt_embed_dim       : {ckpt_embed_dim}")
    if probe_embed_dim is not None:
        compatible = (ckpt_embed_dim is not None) and (ckpt_embed_dim == probe_embed_dim)
        print(f"probe_path           : {args.probe}")
        print(f"probe_embed_dim      : {probe_embed_dim}")
        print(f"probe_compatible     : {compatible}")
    print("-" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline EK probe test for V-JEPA 2.1 frame clips")
    parser.add_argument("--input-root", type=Path, default=Path("/nvme/vjepa_data_32f"))
    parser.add_argument("--manifest", type=Path, default=Path("/data/ek/ek_manifest_test_v2.csv"))
    parser.add_argument("--weights", type=Path, default=Path("/data/vjepa2/vitg-384.pt"))
    parser.add_argument("--probe", type=Path, default=Path("/data/vjepa2/evals/ek100-vitg-384.pt"))
    parser.add_argument("--ek-train-csv", type=Path, default=Path("/data/epic-kitchens-100-annotations/EPIC_100_train.csv"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-clips", type=int, default=200)
    parser.add_argument("--frames-per-clip", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--anticipation-time-sec", type=float, default=1.0)
    parser.add_argument("--encoder-model", type=str, default="vit_giant")
    parser.add_argument("--predictor-depth", type=int, default=12)
    parser.add_argument("--predictor-num-heads", type=int, default=12)
    parser.add_argument("--predictor-embed-dim", type=int, default=384)
    parser.add_argument("--predictor-num-frames", type=int, default=64)
    parser.add_argument("--predictor-num-mask-tokens", type=int, default=8)
    parser.add_argument("--teacher-embed-dim", type=int, default=1664)
    parser.add_argument("--num-output-distillation", type=int, default=1)
    parser.add_argument("--num-output-frames", type=int, default=2)
    parser.add_argument("--probe-num-heads", type=int, default=16)
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging for intermediate steps")
    parser.add_argument("--debug-batches", type=int, default=2, help="Number of initial batches to print debug details for")
    parser.add_argument("--debug-topk", type=int, default=5, help="Top-k classes to print for debug predictions")
    parser.add_argument("--print-check-table", action="store_true", default=True, help="Print startup compatibility checklist table")
    parser.add_argument("--no-print-check-table", action="store_false", dest="print_check_table", help="Disable startup compatibility checklist table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import app.vjepa_2_1.models.predictor as app_pred
    import app.vjepa_2_1.models.vision_transformer as app_vit
    from evals.action_anticipation_frozen.models import AttentiveClassifier

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    def dprint(message: str) -> None:
        if args.debug:
            print(f"[DEBUG] {message}")

    dprint(
        f"args: input_root={args.input_root}, manifest={args.manifest}, weights={args.weights}, "
        f"probe={args.probe}, batch_size={args.batch_size}, device={device}, precision={args.precision}"
    )

    dataset = EKFrameDataset(
        clips_root=args.input_root,
        manifest_csv=args.manifest,
        num_frames=args.frames_per_clip,
        image_size=args.resolution,
    )
    if args.max_clips > 0:
        dataset.items = dataset.items[: args.max_clips]
    dprint(f"dataset_size={len(dataset)}")
    if args.debug and len(dataset) > 0:
        preview = dataset.items[: min(3, len(dataset))]
        for idx, item in enumerate(preview):
            dprint(
                f"sample[{idx}]: clip_id={item['clip_id']} verb={item['verb_class']} noun={item['noun_class']} "
                f"frames_dir={item['frames_dir']}"
            )
    if len(dataset) == 0:
        raise RuntimeError("No valid clips found for evaluation")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    ckpt = torch.load(args.weights, map_location="cpu")
    enc_ckey = "ema_encoder" if "ema_encoder" in ckpt else "encoder"
    if enc_ckey not in ckpt and "target_encoder" in ckpt:
        enc_ckey = "target_encoder"
    dprint(f"encoder_ckpt_key={enc_ckey}")
    encoder = app_vit.__dict__[args.encoder_model](
        img_size=(args.resolution, args.resolution),
        num_frames=args.frames_per_clip,
        patch_size=16,
        tubelet_size=2,
        use_rope=True,
        uniform_power=True,
        img_temporal_dim_size=1,
    ).to(device)
    encoder_sd = strip_prefix(ckpt[enc_ckey], prefixes=["module.", "backbone."])
    enc_filtered = {k: v for k, v in encoder_sd.items() if k in encoder.state_dict() and encoder.state_dict()[k].shape == v.shape}
    enc_msg = encoder.load_state_dict(enc_filtered, strict=False)
    dprint(
        f"encoder_load: matched={len(enc_filtered)} missing={len(enc_msg.missing_keys)} unexpected={len(enc_msg.unexpected_keys)} "
        f"embed_dim={encoder.embed_dim}"
    )
    encoder.eval()

    predictor = app_pred.vit_predictor(
        img_size=(args.resolution, args.resolution),
        embed_dim=encoder.embed_dim,
        patch_size=encoder.patch_size,
        tubelet_size=encoder.tubelet_size,
        num_frames=args.predictor_num_frames,
        depth=args.predictor_depth,
        num_heads=args.predictor_num_heads,
        predictor_embed_dim=args.predictor_embed_dim,
        num_mask_tokens=args.predictor_num_mask_tokens,
        teacher_embed_dim=args.teacher_embed_dim,
        n_output_distillation=args.num_output_distillation,
        return_all_tokens=True,
        img_temporal_dim_size=1,
        uniform_power=True,
        use_mask_tokens=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=False,
        use_rope=True,
    ).to(device)
    pred_sd = strip_prefix(ckpt["predictor"], prefixes=["module.", "backbone."])
    pred_filtered = {k: v for k, v in pred_sd.items() if k in predictor.state_dict() and predictor.state_dict()[k].shape == v.shape}
    pred_msg = predictor.load_state_dict(pred_filtered, strict=False)
    dprint(
        f"predictor_load: matched={len(pred_filtered)} missing={len(pred_msg.missing_keys)} unexpected={len(pred_msg.unexpected_keys)}"
    )
    predictor.eval()

    probe_path = args.probe
    if not probe_path.exists():
        fallback_probe_path = Path("/data/vjepa2/ek100-vitg-384.pt")
        if fallback_probe_path.exists():
            probe_path = fallback_probe_path
        else:
            raise FileNotFoundError(f"Probe checkpoint not found: {args.probe}")

    probe_ckpt = torch.load(probe_path, map_location="cpu")
    probe_sd = probe_ckpt["classifiers"][0]
    probe_depth = infer_probe_depth(probe_sd)
    num_verb = probe_sd["module.verb_classifier.weight"].shape[0]
    num_noun = probe_sd["module.noun_classifier.weight"].shape[0]
    num_action = probe_sd["module.action_classifier.weight"].shape[0]
    probe_embed_dim = probe_sd["module.pooler.query_tokens"].shape[-1]

    if args.print_check_table:
        _print_checklist(args=args, ckpt=ckpt, enc_ckey=enc_ckey, probe_embed_dim=probe_embed_dim)

    if probe_embed_dim != encoder.embed_dim:
        raise RuntimeError(
            f"Probe embed_dim={probe_embed_dim} but encoder embed_dim={encoder.embed_dim}. "
            f"Current probe likely targets a different backbone scale than the provided encoder checkpoint."
        )

    probe = AttentiveClassifier(
        verb_classes={i: i for i in range(num_verb)},
        noun_classes={i: i for i in range(num_noun)},
        action_classes={i: i for i in range(num_action)},
        embed_dim=probe_embed_dim,
        num_heads=args.probe_num_heads,
        depth=probe_depth,
        use_activation_checkpointing=False,
    ).to(device)
    probe.load_state_dict(strip_prefix(probe_sd, prefixes=["module."]), strict=True)
    dprint(
        f"probe_load: depth={probe_depth} num_verb={num_verb} num_noun={num_noun} num_action={num_action} embed_dim={probe_embed_dim}"
    )
    probe.eval()

    verb_map, noun_map, action_map = build_label_maps(args.ek_train_csv)
    dprint(f"label_maps: verbs={len(verb_map)} nouns={len(noun_map)} actions={len(action_map)}")

    autocast_dtype = resolve_autocast_dtype(args.precision)
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}
    tubelet = int(getattr(encoder, "tubelet_size", 2))
    grid = args.resolution // 16
    n_pred = int((grid * grid) * (args.num_output_frames // tubelet))
    dprint(f"token_setup: tubelet={tubelet} grid={grid} n_pred={n_pred}")

    total = 0
    hit5 = 0
    skipped = 0

    with torch.no_grad():
        for batch_idx, (clip_ids, verbs_raw, nouns_raw, frame_dirs, clips) in enumerate(
            tqdm(loader, total=len(loader), desc="ek-probe-eval")
        ):
            if args.debug and batch_idx < args.debug_batches:
                dprint(f"batch={batch_idx} raw_batch_size={len(clip_ids)} clips_shape={tuple(clips.shape)}")
                for i in range(min(len(clip_ids), 3)):
                    dprint(
                        f"raw_sample[{i}]: clip_id={clip_ids[i]} frame_dir={frame_dirs[i]} "
                        f"verb={int(verbs_raw[i])} noun={int(nouns_raw[i])}"
                    )

            labels = []
            keep = []
            for i, (v, n) in enumerate(zip(verbs_raw.tolist(), nouns_raw.tolist())):
                if v not in verb_map or n not in noun_map or (v, n) not in action_map:
                    if args.debug and batch_idx < args.debug_batches:
                        dprint(f"skip_sample[{i}] clip_id={clip_ids[i]} reason=label_not_in_train verb={v} noun={n}")
                    skipped += 1
                    continue
                labels.append(action_map[(v, n)])
                keep.append(i)

            if args.debug and batch_idx < args.debug_batches:
                dprint(f"batch={batch_idx} kept={len(keep)} skipped_in_batch={len(clip_ids)-len(keep)}")
            if len(keep) == 0:
                continue

            clips = clips[keep].to(device, non_blocking=True)
            action_labels = torch.tensor(labels, dtype=torch.long, device=device)
            kept_clip_ids = [clip_ids[i] for i in keep]
            kept_frame_dirs = [frame_dirs[i] for i in keep]

            if args.debug and batch_idx < args.debug_batches:
                dprint(f"batch={batch_idx} kept_clips_shape={tuple(clips.shape)} action_labels={action_labels.tolist()}")
                for i in range(min(len(kept_clip_ids), 3)):
                    v = int(verbs_raw[keep[i]])
                    n = int(nouns_raw[keep[i]])
                    gt_action = int(action_map[(v, n)])
                    dprint(
                        f"kept_sample[{i}]: clip_id={kept_clip_ids[i]} frame_dir={kept_frame_dirs[i]} "
                        f"verb={v}->{verb_map[v]} noun={n}->{noun_map[n]} action_gt={gt_action}"
                    )

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                x_full = encoder(clips)
                bsz, n_ctx, d_full = x_full.shape
                embed_dim = encoder.embed_dim
                if d_full > embed_dim:
                    x_ctx = x_full[:, :, -embed_dim:]
                else:
                    x_ctx = x_full

                x_acc = x_ctx.clone()
                ctxt_positions = torch.arange(n_ctx, device=device).unsqueeze(0).repeat(bsz, 1)

                anticipation_times = torch.full((bsz,), args.anticipation_time_sec, device=device)
                anticipation_steps = (anticipation_times * args.fps / tubelet).to(torch.int64)
                skip_positions = n_ctx + (grid * grid) * anticipation_steps
                tgt_positions = torch.arange(n_pred, device=device).unsqueeze(0).repeat(bsz, 1)
                tgt_positions = tgt_positions + skip_positions.unsqueeze(1)

                if args.debug and batch_idx < args.debug_batches:
                    dprint(
                        f"forward_shapes: x_full={tuple(x_full.shape)} x_ctx={tuple(x_ctx.shape)} "
                        f"ctxt_positions={tuple(ctxt_positions.shape)} tgt_positions={tuple(tgt_positions.shape)}"
                    )
                    dprint(
                        f"anticipation_steps={anticipation_steps.tolist()} skip_positions={skip_positions.tolist()} "
                        f"tgt_positions_first5={tgt_positions[0, :5].tolist() if tgt_positions.numel() > 0 else []}"
                    )

                pred_out = predictor(x_full, masks_x=ctxt_positions, masks_y=tgt_positions)
                x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
                if x_pred_full.size(-1) > embed_dim:
                    x_pred = x_pred_full[:, :, -embed_dim:]
                else:
                    x_pred = x_pred_full
                x_acc = torch.cat([x_acc, x_pred], dim=1)

                logits = probe(x_acc)["action"]

                if args.debug and batch_idx < args.debug_batches:
                    dprint(
                        f"predictor_probe_shapes: x_pred_full={tuple(x_pred_full.shape)} x_pred={tuple(x_pred.shape)} "
                        f"x_acc={tuple(x_acc.shape)} logits={tuple(logits.shape)}"
                    )
                    debug_k = min(args.debug_topk, logits.shape[1])
                    debug_topk_vals, debug_topk_idx = torch.topk(logits, k=debug_k, dim=1)
                    for i in range(min(logits.shape[0], 3)):
                        gt = int(action_labels[i].item())
                        pred_list = [int(x) for x in debug_topk_idx[i].detach().cpu().tolist()]
                        val_list = [float(x) for x in debug_topk_vals[i].detach().cpu().tolist()]
                        dprint(
                            f"probe_out[{i}] clip_id={kept_clip_ids[i]} gt={gt} top{debug_k}_idx={pred_list} top{debug_k}_logits={val_list}"
                        )

            top5 = logits.topk(k=5, dim=1).indices
            matched = (top5 == action_labels.unsqueeze(1)).any(dim=1)
            hit5 += int(matched.sum().item())
            total += int(action_labels.numel())

            if args.debug and batch_idx < args.debug_batches:
                dprint(
                    f"batch={batch_idx} matched={matched.detach().cpu().tolist()} "
                    f"running_hit5={hit5} running_total={total}"
                )

    recall5 = (100.0 * hit5 / total) if total > 0 else 0.0
    print(f"[RESULT] evaluated={total} skipped={skipped} recall@5={recall5:.4f}%")


if __name__ == "__main__":
    main()
