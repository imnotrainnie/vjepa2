#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SelectedClipDataset(Dataset):
    def __init__(self, clips_root: Path, clip_ids: Sequence[str], num_frames: int, image_size: int) -> None:
        self.clips_root = clips_root
        self.num_frames = num_frames
        self.items: List[Tuple[str, Path]] = []
        for clip_id in clip_ids:
            frames_dir = clips_root / clip_id / "frames"
            frame_paths = self._list_frame_paths(frames_dir)
            if len(frame_paths) >= num_frames:
                self.items.append((clip_id, frames_dir))

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def _list_frame_paths(frames_dir: Path) -> List[Path]:
        paths: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            paths.extend(frames_dir.glob(pattern))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        clip_id, frames_dir = self.items[idx]
        frame_paths = self._list_frame_paths(frames_dir)[: self.num_frames]
        frames_tchw: List[torch.Tensor] = []
        for fp in frame_paths:
            with Image.open(fp) as img:
                frames_tchw.append(self.transform(img.convert("RGB")))
        clip = torch.stack(frames_tchw, dim=0).permute(1, 0, 2, 3).contiguous()
        return clip_id, clip


def collate_fn(batch: Sequence[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    clip_ids = [b[0] for b in batch]
    clips = torch.stack([b[1] for b in batch], dim=0)
    return clip_ids, clips


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = value
    return out


def resolve_autocast_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def build_clip_table(manifest: Path, input_root: Path, num_frames: int) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    required = {"clip_id", "video_id", "verb_class", "noun_class"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"manifest missing columns: {sorted(miss)}")

    rows = []
    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])
        frames_dir = input_root / clip_id / "frames"
        count = 0
        if frames_dir.exists():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                count += len(list(frames_dir.glob(pattern)))
        if count >= num_frames:
            rows.append(
                {
                    "clip_id": clip_id,
                    "video_id": str(row["video_id"]),
                    "verb_class": int(row["verb_class"]),
                    "noun_class": int(row["noun_class"]),
                    "action_label": f"{int(row['verb_class'])}_{int(row['noun_class'])}",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid clips with enough frames were found")
    return out


def choose_pairs(df: pd.DataFrame, case: str, n_pairs: int, rng: random.Random) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    if case == "same_action_same_bg":
        for (_, _), group in df.groupby(["action_label", "video_id"]):
            clips = group["clip_id"].tolist()
            if len(clips) < 2:
                continue
            for a, b in itertools.combinations(clips, 2):
                pairs.append((a, b))

    elif case == "same_action_diff_bg":
        for _, group in df.groupby("action_label"):
            vids = group["video_id"].unique().tolist()
            if len(vids) < 2:
                continue
            by_vid = {vid: group[group["video_id"] == vid]["clip_id"].tolist() for vid in vids}
            for v1, v2 in itertools.combinations(vids, 2):
                for a in by_vid[v1]:
                    for b in by_vid[v2]:
                        pairs.append((a, b))

    elif case == "diff_action_same_bg":
        for _, group in df.groupby("video_id"):
            actions = group["action_label"].unique().tolist()
            if len(actions) < 2:
                continue
            by_action = {act: group[group["action_label"] == act]["clip_id"].tolist() for act in actions}
            for a1, a2 in itertools.combinations(actions, 2):
                for c1 in by_action[a1]:
                    for c2 in by_action[a2]:
                        pairs.append((c1, c2))
    elif case == "diff_action_diff_bg":
        records = df[["clip_id", "action_label", "video_id"]].to_dict("records")
        for i in range(len(records)):
            r1 = records[i]
            for j in range(i + 1, len(records)):
                r2 = records[j]
                if r1["action_label"] != r2["action_label"] and r1["video_id"] != r2["video_id"]:
                    pairs.append((str(r1["clip_id"]), str(r2["clip_id"])))
    else:
        raise ValueError(f"unknown case={case}")

    dedup = []
    seen = set()
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((a, b))

    if len(dedup) < n_pairs:
        raise RuntimeError(f"Case {case}: only found {len(dedup)} pairs, need {n_pairs}")

    rng.shuffle(dedup)
    return dedup[:n_pairs]


def load_encoder_predictor(args: argparse.Namespace, device: torch.device):
    try:
        import app.vjepa_2_1.models.predictor as app_pred
        import app.vjepa_2_1.models.vision_transformer as app_vit
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Failed to import V-JEPA 2.1 modules: {error}. "
            "Please install required deps, e.g. `pip install einops timm`."
        ) from error

    ckpt = torch.load(args.weights, map_location="cpu")
    enc_key = "ema_encoder" if "ema_encoder" in ckpt else ("encoder" if "encoder" in ckpt else "target_encoder")

    encoder = app_vit.__dict__[args.encoder_model](
        img_size=(args.resolution, args.resolution),
        num_frames=args.num_frames,
        patch_size=16,
        tubelet_size=2,
        use_rope=True,
        uniform_power=True,
        img_temporal_dim_size=1,
    ).to(device)
    enc_sd = strip_prefix(ckpt[enc_key], prefixes=["module.", "backbone."])
    enc_filtered = {k: v for k, v in enc_sd.items() if k in encoder.state_dict() and encoder.state_dict()[k].shape == v.shape}
    encoder.load_state_dict(enc_filtered, strict=False)
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
    predictor.load_state_dict(pred_filtered, strict=False)
    predictor.eval()

    return encoder, predictor


def extract_fjoint(
    args: argparse.Namespace,
    device: torch.device,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    clip_ids: Sequence[str],
) -> Dict[str, torch.Tensor]:
    dataset = SelectedClipDataset(args.input_root, clip_ids, args.num_frames, args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    autocast_dtype = resolve_autocast_dtype(args.precision)
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}

    tubelet = int(getattr(encoder, "tubelet_size", 2))
    grid = args.resolution // 16
    n_pred = int((grid * grid) * (args.num_output_frames // tubelet))

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for batch_clip_ids, clips in tqdm(loader, total=len(loader), desc="extract_fjoint"):
            clips = clips.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                x_full = encoder(clips)
                bsz, n_ctx, d_full = x_full.shape
                embed_dim = encoder.embed_dim
                x_ctx = x_full[:, :, -embed_dim:] if d_full > embed_dim else x_full

                ctxt_positions = torch.arange(n_ctx, device=device).unsqueeze(0).repeat(bsz, 1)
                anticipation_times = torch.full((bsz,), args.anticipation_time_sec, device=device)
                anticipation_steps = (anticipation_times * args.fps / tubelet).to(torch.int64)
                skip_positions = n_ctx + (grid * grid) * anticipation_steps
                tgt_positions = torch.arange(n_pred, device=device).unsqueeze(0).repeat(bsz, 1)
                tgt_positions = tgt_positions + skip_positions.unsqueeze(1)

                pred_out = predictor(x_full, masks_x=ctxt_positions, masks_y=tgt_positions)
                x_pred_full = pred_out[0] if isinstance(pred_out, tuple) else pred_out
                x_pred = x_pred_full[:, :, -embed_dim:] if x_pred_full.size(-1) > embed_dim else x_pred_full

                x_joint = torch.cat([x_ctx, x_pred], dim=1)
                f_joint = x_joint.mean(dim=1)

            for clip_id, feat in zip(batch_clip_ids, f_joint.detach().float().cpu()):
                out[clip_id] = feat
    return out


def cosine_for_pairs(pairs: Sequence[Tuple[str, str]], feature_map: Dict[str, torch.Tensor]) -> List[float]:
    values: List[float] = []
    for a, b in pairs:
        fa = feature_map[a].unsqueeze(0)
        fb = feature_map[b].unsqueeze(0)
        sim = F.cosine_similarity(fa, fb, dim=1).item()
        values.append(float(sim))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze background contamination in F_joint via cosine similarity")
    parser.add_argument("--manifest", type=Path, default=Path("/data/ek/ek_manifest_test_v2.csv"))
    parser.add_argument("--input-root", type=Path, default=Path("/nvme/vjepa_data_32f"))
    parser.add_argument("--weights", type=Path, default=Path("/data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"))
    parser.add_argument("--encoder-model", type=str, default="vit_giant")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--anticipation-time-sec", type=float, default=1.0)
    parser.add_argument("--predictor-depth", type=int, default=12)
    parser.add_argument("--predictor-num-heads", type=int, default=12)
    parser.add_argument("--predictor-embed-dim", type=int, default=384)
    parser.add_argument("--predictor-num-frames", type=int, default=64)
    parser.add_argument("--predictor-num-mask-tokens", type=int, default=8)
    parser.add_argument("--teacher-embed-dim", type=int, default=1664)
    parser.add_argument("--num-output-distillation", type=int, default=1)
    parser.add_argument("--num-output-frames", type=int, default=2)

    parser.add_argument("--num-groups", type=int, default=50, help="Number of pairs sampled for each case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--save-pairs-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    clip_df = build_clip_table(args.manifest, args.input_root, args.num_frames)
    case1_pairs = choose_pairs(clip_df, "same_action_same_bg", args.num_groups, rng)
    case2_pairs = choose_pairs(clip_df, "same_action_diff_bg", args.num_groups, rng)
    case3_pairs = choose_pairs(clip_df, "diff_action_same_bg", args.num_groups, rng)
    case4_pairs = choose_pairs(clip_df, "diff_action_diff_bg", args.num_groups, rng)

    all_clip_ids = sorted(set([x for p in (case1_pairs + case2_pairs + case3_pairs + case4_pairs) for x in p]))
    print(
        f"[INFO] selected_pairs: case1={len(case1_pairs)} case2={len(case2_pairs)} "
        f"case3={len(case3_pairs)} case4={len(case4_pairs)}"
    )
    print(f"[INFO] unique_clips_for_feature_extraction={len(all_clip_ids)}")

    encoder, predictor = load_encoder_predictor(args, device)
    feature_map = extract_fjoint(args, device, encoder, predictor, all_clip_ids)

    c1 = cosine_for_pairs(case1_pairs, feature_map)
    c2 = cosine_for_pairs(case2_pairs, feature_map)
    c3 = cosine_for_pairs(case3_pairs, feature_map)
    c4 = cosine_for_pairs(case4_pairs, feature_map)

    meta_map: Dict[str, Dict[str, object]] = {}
    for _, row in clip_df.iterrows():
        clip_id = str(row["clip_id"])
        meta_map[clip_id] = {
            "video_id": str(row["video_id"]),
            "verb_class": int(row["verb_class"]),
            "noun_class": int(row["noun_class"]),
            "action_label": str(row["action_label"]),
        }

    def avg(x: Sequence[float]) -> float:
        return float(sum(x) / len(x)) if x else float("nan")

    report = {
        "config": {
            "manifest": str(args.manifest),
            "input_root": str(args.input_root),
            "weights": str(args.weights),
            "encoder_model": args.encoder_model,
            "num_groups": args.num_groups,
            "fps": args.fps,
            "anticipation_time_sec": args.anticipation_time_sec,
        },
        "scores": {
            "case1_same_action_same_bg": {"mean_cosine": avg(c1), "values": c1, "pairs": case1_pairs},
            "case2_same_action_diff_bg": {"mean_cosine": avg(c2), "values": c2, "pairs": case2_pairs},
            "case3_diff_action_same_bg": {"mean_cosine": avg(c3), "values": c3, "pairs": case3_pairs},
            "case4_diff_action_diff_bg": {"mean_cosine": avg(c4), "values": c4, "pairs": case4_pairs},
        },
    }

    print("\n===== F_joint 背景污染分析报表 =====")
    print(f"Case1 同动作+同背景 mean cosine: {report['scores']['case1_same_action_same_bg']['mean_cosine']:.6f}")
    print(f"Case2 同动作+异背景 mean cosine: {report['scores']['case2_same_action_diff_bg']['mean_cosine']:.6f}")
    print(f"Case3 异动作+同背景 mean cosine: {report['scores']['case3_diff_action_same_bg']['mean_cosine']:.6f}")
    print(f"Case4 异动作+异背景 mean cosine: {report['scores']['case4_diff_action_diff_bg']['mean_cosine']:.6f}")

    m1 = report["scores"]["case1_same_action_same_bg"]["mean_cosine"]
    m2 = report["scores"]["case2_same_action_diff_bg"]["mean_cosine"]
    m3 = report["scores"]["case3_diff_action_same_bg"]["mean_cosine"]
    m4 = report["scores"]["case4_diff_action_diff_bg"]["mean_cosine"]
    print("\n===== 逻辑判读 =====")
    print(f"Case2-Case1 差值: {m2 - m1:+.6f} (若显著为负，支持背景污染假设)")
    print(f"Case3 相对 Case2: {m3 - m2:+.6f} (若偏高，说明同背景信号可能主导)")
    print(f"Case4 相对 Case3: {m4 - m3:+.6f} (通常应更低，作为双异条件参考)")

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[INFO] report saved to {args.save_json}")

    if args.save_pairs_csv is not None:
        rows: List[Dict[str, object]] = []

        def append_case_rows(case_name: str, pairs: Sequence[Tuple[str, str]], sims: Sequence[float]) -> None:
            for pair_index, ((clip_a, clip_b), cosine_sim) in enumerate(zip(pairs, sims), start=1):
                meta_a = meta_map[clip_a]
                meta_b = meta_map[clip_b]
                rows.append(
                    {
                        "case": case_name,
                        "pair_index": pair_index,
                        "clip_a": clip_a,
                        "clip_b": clip_b,
                        "video_id_a": meta_a["video_id"],
                        "video_id_b": meta_b["video_id"],
                        "verb_class_a": meta_a["verb_class"],
                        "verb_class_b": meta_b["verb_class"],
                        "noun_class_a": meta_a["noun_class"],
                        "noun_class_b": meta_b["noun_class"],
                        "action_label_a": meta_a["action_label"],
                        "action_label_b": meta_b["action_label"],
                        "cosine_similarity": float(cosine_sim),
                    }
                )

        append_case_rows("case1_same_action_same_bg", case1_pairs, c1)
        append_case_rows("case2_same_action_diff_bg", case2_pairs, c2)
        append_case_rows("case3_diff_action_same_bg", case3_pairs, c3)
        append_case_rows("case4_diff_action_diff_bg", case4_pairs, c4)

        args.save_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.save_pairs_csv, index=False)
        print(f"[INFO] pairs csv saved to {args.save_pairs_csv}")


if __name__ == "__main__":
    main()
