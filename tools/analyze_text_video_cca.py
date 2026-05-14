#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class SampleRecord:
    clip_id: str
    video_id: str
    verb: str
    noun: str
    action_text: str


class ClipDataset(Dataset):
    def __init__(self, clips_root: Path, clip_ids: Sequence[str], num_frames: int, image_size: int) -> None:
        self.num_frames = num_frames
        self.items: List[Tuple[str, Path]] = []
        for clip_id in clip_ids:
            frames_dir = clips_root / clip_id / "frames"
            frames = self._list_frames(frames_dir)
            if len(frames) >= num_frames:
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
    def _list_frames(frames_dir: Path) -> List[Path]:
        paths: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            paths.extend(frames_dir.glob(pattern))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        clip_id, frames_dir = self.items[idx]
        frame_paths = self._list_frames(frames_dir)[: self.num_frames]
        frames_tchw: List[torch.Tensor] = []
        for fp in frame_paths:
            with Image.open(fp) as img:
                frames_tchw.append(self.transform(img.convert("RGB")))
        clip = torch.stack(frames_tchw, dim=0).permute(1, 0, 2, 3).contiguous()
        return clip_id, clip


def collate_fn(batch: Sequence[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    clip_ids = [x[0] for x in batch]
    clips = torch.stack([x[1] for x in batch], dim=0)
    return clip_ids, clips


def strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = val
    return out


def resolve_autocast_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    token = token.replace(":", " ")
    token = token.replace("_", " ")
    token = token.replace("-", " ")
    token = re.sub(r"\s+", " ", token)
    return token


def build_records(manifest: Path, input_root: Path, num_frames: int, text_source: str) -> List[SampleRecord]:
    df = pd.read_csv(manifest)
    required = {"clip_id", "video_id", "verb", "noun"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"manifest missing required columns: {sorted(miss)}")

    records: List[SampleRecord] = []
    for _, row in df.iterrows():
        clip_id = str(row["clip_id"])
        frames_dir = input_root / clip_id / "frames"
        frames = ClipDataset._list_frames(frames_dir)
        if len(frames) < num_frames:
            continue

        verb = normalize_token(str(row.get("verb", "")))
        noun = normalize_token(str(row.get("noun", "")))
        narration = normalize_token(str(row.get("narration", "")))

        if text_source == "narration" and narration:
            action_text = narration
        else:
            action_text = f"{verb} {noun}".strip()

        records.append(
            SampleRecord(
                clip_id=clip_id,
                video_id=str(row["video_id"]),
                verb=verb,
                noun=noun,
                action_text=action_text,
            )
        )

    records = sorted(records, key=lambda x: x.clip_id)
    if not records:
        raise RuntimeError("No valid clips found after frame availability filtering")
    return records


def load_vjepa_encoder(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    try:
        import app.vjepa_2_1.models.vision_transformer as app_vit
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Failed to import V-JEPA 2.1 modules: {error}. Please install deps: `pip install einops timm`."
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
    sd = strip_prefix(ckpt[enc_key], prefixes=["module.", "backbone."])
    filtered = {k: v for k, v in sd.items() if k in encoder.state_dict() and encoder.state_dict()[k].shape == v.shape}
    msg = encoder.load_state_dict(filtered, strict=False)
    print(f"[INFO] encoder_ckpt_key={enc_key}, loaded={len(filtered)}, missing={len(msg.missing_keys)}")
    encoder.eval()
    return encoder


def extract_video_features(args: argparse.Namespace, records: Sequence[SampleRecord], device: torch.device) -> np.ndarray:
    encoder = load_vjepa_encoder(args, device)
    dataset = ClipDataset(args.input_root, [r.clip_id for r in records], args.num_frames, args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    feat_by_clip: Dict[str, np.ndarray] = {}
    autocast_dtype = resolve_autocast_dtype(args.precision)
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}

    with torch.no_grad():
        for clip_ids, clips in tqdm(loader, total=len(loader), desc="video_features"):
            clips = clips.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                latent = encoder(clips)
                pooled = latent.mean(dim=1)
            pooled = pooled.detach().float().cpu().numpy()
            for clip_id, feat in zip(clip_ids, pooled):
                feat_by_clip[clip_id] = feat

    mat = np.stack([feat_by_clip[r.clip_id] for r in records], axis=0)
    return mat


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-6)
    return summed / counts


def extract_text_features(args: argparse.Namespace, records: Sequence[SampleRecord], device: torch.device) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Failed to import transformers: {error}. Please install with `pip install transformers`"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.text_model_path, trust_remote_code=True).to(device)
    model.eval()

    texts = [r.action_text for r in records]
    all_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), args.text_batch_size), desc="text_features"):
            batch_text = texts[start : start + args.text_batch_size]
            batch = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=args.text_max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            all_embeddings.append(pooled.detach().float().cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def save_feature_cache(cache_dir: Path, records: Sequence[SampleRecord], video_features: np.ndarray, text_features: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "video_features.npy", video_features)
    np.save(cache_dir / "text_features.npy", text_features)

    meta = pd.DataFrame(
        {
            "clip_id": [r.clip_id for r in records],
            "video_id": [r.video_id for r in records],
            "verb": [r.verb for r in records],
            "noun": [r.noun for r in records],
            "action_text": [r.action_text for r in records],
        }
    )
    meta.to_csv(cache_dir / "meta.csv", index=False)


def load_feature_cache(cache_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta = pd.read_csv(cache_dir / "meta.csv")
    v = np.load(cache_dir / "video_features.npy")
    t = np.load(cache_dir / "text_features.npy")
    return meta, v, t


def maybe_get_features(args: argparse.Namespace, records: Sequence[SampleRecord], device: torch.device) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if args.use_cache and (args.cache_dir / "meta.csv").exists() and (args.cache_dir / "video_features.npy").exists() and (args.cache_dir / "text_features.npy").exists():
        meta, v, t = load_feature_cache(args.cache_dir)
        clip_ids_current = [r.clip_id for r in records]
        if meta["clip_id"].tolist() == clip_ids_current and len(v) == len(records) and len(t) == len(records):
            print(f"[INFO] loaded cached features from {args.cache_dir}")
            return meta, v, t
        print("[WARN] cache exists but clip ordering/size mismatch; recomputing features")

    v = extract_video_features(args, records, device)
    t = extract_text_features(args, records, device)
    meta = pd.DataFrame(
        {
            "clip_id": [r.clip_id for r in records],
            "video_id": [r.video_id for r in records],
            "verb": [r.verb for r in records],
            "noun": [r.noun for r in records],
            "action_text": [r.action_text for r in records],
        }
    )
    if args.save_cache:
        save_feature_cache(args.cache_dir, records, v, t)
        print(f"[INFO] cached features saved to {args.cache_dir}")
    return meta, v, t


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_kfold_cca(
    video_features: np.ndarray,
    text_features: np.ndarray,
    n_splits: int,
    n_components: int,
    random_state: int,
) -> Dict[str, np.ndarray | float]:
    from sklearn.cross_decomposition import CCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    n = len(video_features)
    if n < n_splits:
        raise ValueError(f"n_samples={n} < n_splits={n_splits}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_corrs: List[np.ndarray] = []
    sample_pair_similarity = np.full((n,), np.nan, dtype=np.float64)

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(video_features), start=1):
        xv_train, xv_test = video_features[train_idx], video_features[test_idx]
        xt_train, xt_test = text_features[train_idx], text_features[test_idx]

        scaler_v = StandardScaler().fit(xv_train)
        scaler_t = StandardScaler().fit(xt_train)
        xv_train = scaler_v.transform(xv_train)
        xv_test = scaler_v.transform(xv_test)
        xt_train = scaler_t.transform(xt_train)
        xt_test = scaler_t.transform(xt_test)

        k_eff = min(n_components, xv_train.shape[1], xt_train.shape[1], len(train_idx) - 1)
        if k_eff < 1:
            raise RuntimeError("effective CCA components < 1")

        cca = CCA(n_components=k_eff, max_iter=2000)
        cca.fit(xv_train, xt_train)
        uv_test, ut_test = cca.transform(xv_test, xt_test)

        corrs = np.array([_safe_corr(uv_test[:, i], ut_test[:, i]) for i in range(k_eff)], dtype=np.float64)
        padded = np.full((n_components,), np.nan, dtype=np.float64)
        padded[:k_eff] = corrs
        fold_corrs.append(padded)

        uv_norm = np.linalg.norm(uv_test, axis=1)
        ut_norm = np.linalg.norm(ut_test, axis=1)
        denom = np.clip(uv_norm * ut_norm, a_min=1e-8, a_max=None)
        sample_cos = np.sum(uv_test * ut_test, axis=1) / denom
        sample_pair_similarity[test_idx] = sample_cos

        print(f"[INFO] fold={fold_id}, k_eff={k_eff}, mean_corr={np.nanmean(corrs):.6f}")

    fold_corrs_mat = np.stack(fold_corrs, axis=0)
    mean_corrs = np.nanmean(fold_corrs_mat, axis=0)
    global_score = float(np.nanmean(mean_corrs))

    return {
        "mean_corrs": mean_corrs,
        "fold_corrs": fold_corrs_mat,
        "global_score": global_score,
        "sample_pair_similarity": sample_pair_similarity,
    }


def compute_group_scores(meta: pd.DataFrame, sample_pair_similarity: np.ndarray, verb_heavy_verbs: Sequence[str], noun_heavy_nouns: Sequence[str]) -> Dict[str, float]:
    verb_set = {normalize_token(v) for v in verb_heavy_verbs if v.strip()}
    noun_set = {normalize_token(n) for n in noun_heavy_nouns if n.strip()}

    group = []
    for _, row in meta.iterrows():
        verb = normalize_token(str(row["verb"]))
        noun_tokens = normalize_token(str(row["noun"]))
        noun_hit = any(tok in noun_set for tok in noun_tokens.split())
        verb_hit = verb in verb_set

        if verb_hit and not noun_hit:
            group.append("verb_heavy")
        elif noun_hit and not verb_hit:
            group.append("noun_heavy")
        elif verb_hit and noun_hit:
            group.append("both")
        else:
            group.append("other")

    meta = meta.copy()
    meta["group"] = group
    meta["sample_pair_similarity"] = sample_pair_similarity

    scores = {}
    for key in ["verb_heavy", "noun_heavy", "both", "other"]:
        vals = meta.loc[meta["group"] == key, "sample_pair_similarity"].dropna().values
        scores[key] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return scores


def plot_canonical_corrs(mean_corrs: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    valid = mean_corrs[~np.isnan(mean_corrs)]
    if len(valid) == 0:
        raise RuntimeError("No valid canonical correlations to plot")

    sorted_vals = np.sort(valid)[::-1]
    xs = np.arange(1, len(sorted_vals) + 1)

    plt.figure(figsize=(10, 4))
    plt.bar(xs, sorted_vals)
    plt.xlabel("Canonical Component (sorted)")
    plt.ylabel("Correlation")
    plt.title("Canonical Correlations (Video vs Text)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_group_comparison(group_scores: Dict[str, float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["verb_heavy", "noun_heavy"]
    vals = [group_scores.get("verb_heavy", float("nan")), group_scores.get("noun_heavy", float("nan"))]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.ylabel("Average CCA-space Pair Similarity")
    plt.title("Verb-heavy vs Noun-heavy Alignment")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCA analysis for text-video alignment (V-JEPA 2.1 + Jina text encoder)")
    parser.add_argument("--manifest", type=Path, default=Path("/data/ek/ek_manifest_test_v2.csv"))
    parser.add_argument("--input-root", type=Path, default=Path("/nvme/vjepa_data_32f"))

    parser.add_argument("--weights", type=Path, default=Path("/data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"))
    parser.add_argument("--encoder-model", type=str, default="vit_base")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--text-model-path", type=Path, default=Path("/data/jina-v4-local"))
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--text-source", type=str, default="verb_noun", choices=["verb_noun", "narration"])

    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cache-dir", type=Path, default=Path("/nvme/vjepa_cca_cache"))
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-use-cache", action="store_false", dest="use_cache")
    parser.add_argument("--save-cache", action="store_true", default=True)
    parser.add_argument("--no-save-cache", action="store_false", dest="save_cache")

    parser.add_argument("--output-dir", type=Path, default=Path("/data/ek/cca_alignment_outputs"))
    parser.add_argument("--report-json", type=Path, default=Path("/data/ek/cca_alignment_outputs/cca_report.json"))

    parser.add_argument("--verb-heavy-verbs", type=str, default="mix,stir,wash,knead,chop,slice,cut,peel")
    parser.add_argument("--noun-heavy-nouns", type=str, default="plate,cup,bowl,pan,dish,mug,glass,bottle,tray")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    records = build_records(args.manifest, args.input_root, args.num_frames, args.text_source)
    meta, video_features, text_features = maybe_get_features(args, records, device)

    cca_out = run_kfold_cca(
        video_features=video_features,
        text_features=text_features,
        n_splits=args.n_splits,
        n_components=args.n_components,
        random_state=args.seed,
    )

    verb_heavy = [x.strip() for x in args.verb_heavy_verbs.split(",")]
    noun_heavy = [x.strip() for x in args.noun_heavy_nouns.split(",")]
    group_scores = compute_group_scores(meta, cca_out["sample_pair_similarity"], verb_heavy, noun_heavy)

    corr_plot = args.output_dir / "canonical_correlations.png"
    grp_plot = args.output_dir / "verb_vs_noun_heavy.png"
    plot_canonical_corrs(cca_out["mean_corrs"], corr_plot)
    plot_group_comparison(group_scores, grp_plot)

    report = {
        "config": {
            "manifest": str(args.manifest),
            "input_root": str(args.input_root),
            "weights": str(args.weights),
            "text_model_path": str(args.text_model_path),
            "n_components": args.n_components,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "num_samples": int(len(meta)),
        },
        "global_alignment_score": float(cca_out["global_score"]),
        "canonical_correlations": [float(x) if not np.isnan(x) else None for x in cca_out["mean_corrs"]],
        "group_scores": group_scores,
        "plots": {
            "canonical_correlations": str(corr_plot),
            "verb_vs_noun_heavy": str(grp_plot),
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== CCA Alignment Report =====")
    print(f"Global score (mean canonical corr): {report['global_alignment_score']:.6f}")
    print(f"Verb-heavy mean: {group_scores.get('verb_heavy', float('nan')):.6f}")
    print(f"Noun-heavy mean: {group_scores.get('noun_heavy', float('nan')):.6f}")
    print(f"Report JSON: {args.report_json}")
    print(f"Plot #1: {corr_plot}")
    print(f"Plot #2: {grp_plot}")


if __name__ == "__main__":
    main()
