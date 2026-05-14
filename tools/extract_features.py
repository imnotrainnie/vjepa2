#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


class VJEPAFrameDataset(Dataset):
    def __init__(
        self,
        clips_root: Path,
        manifest_csv: Path,
        num_frames: int = 32,
        image_size: int = 384,
    ) -> None:
        self.clips_root = clips_root
        self.num_frames = num_frames
        self.image_size = image_size
        self.df = pd.read_csv(manifest_csv)
        if "clip_id" not in self.df.columns:
            raise ValueError(f"Missing required column 'clip_id' in {manifest_csv}")

        self.missing_frames_dirs: List[str] = []
        self.insufficient_frames: List[str] = []
        self.items: List[Dict[str, str]] = []
        for _, row in self.df.iterrows():
            clip_id = str(row["clip_id"])
            frames_dir = self.clips_root / clip_id / "frames"
            frame_paths = self._list_frame_paths(frames_dir)
            if not frames_dir.exists():
                self.missing_frames_dirs.append(str(frames_dir))
                continue
            if len(frame_paths) < self.num_frames:
                self.insufficient_frames.append(f"{frames_dir} ({len(frame_paths)})")
                continue
            self.items.append(
                {
                    "clip_id": clip_id,
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

    @staticmethod
    def _list_frame_paths(frames_dir: Path) -> List[Path]:
        frame_paths: List[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            frame_paths.extend(frames_dir.glob(pattern))
        return sorted(frame_paths)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        item = self.items[idx]
        clip_id = item["clip_id"]
        frames_dir = Path(item["frames_dir"])
        frame_paths = self._list_frame_paths(frames_dir)[: self.num_frames]

        frames_tchw: List[torch.Tensor] = []
        for fp in frame_paths:
            with Image.open(fp) as img:
                img = img.convert("RGB")
                frames_tchw.append(self.transform(img))

        video_tchw = torch.stack(frames_tchw, dim=0)
        video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()
        return clip_id, video_cthw


def collate_fn(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    clip_ids = [b[0] for b in batch]
    clips = torch.stack([b[1] for b in batch], dim=0)
    return clip_ids, clips


def strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        out[key] = val
    return out


def load_encoder(
    checkpoint_path: Path,
    model_name: str,
    resolution: int,
    num_frames: int,
    device: torch.device,
) -> torch.nn.Module:
    try:
        import src.models.vision_transformer as vit
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Failed to import V-JEPA modules: {error}. "
            "Please install required deps, e.g. `pip install einops timm`."
        ) from error

    if model_name not in vit.__dict__:
        raise ValueError(f"Unknown encoder model name: {model_name}")

    model = vit.__dict__[model_name](
        img_size=(resolution, resolution),
        num_frames=num_frames,
        patch_size=16,
        tubelet_size=2,
        use_rope=True,
        uniform_power=True,
    ).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    key_candidates = ["ema_encoder", "encoder", "target_encoder"]
    ck_key = next((k for k in key_candidates if k in ckpt), None)
    if ck_key is None:
        raise ValueError(f"Cannot find encoder key in checkpoint. Tried {key_candidates}")

    encoder_state = strip_state_dict_prefix(ckpt[ck_key])
    model_state = model.state_dict()
    filtered = {k: v for k, v in encoder_state.items() if k in model_state and model_state[k].shape == v.shape}
    msg = model.load_state_dict(filtered, strict=False)
    print(f"[INFO] encoder_ckpt_key={ck_key} loaded={len(filtered)} missing={len(msg.missing_keys)}")
    return model


def resolve_autocast_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline feature extraction for V-JEPA 2.1 frame clips")
    parser.add_argument("--input-root", type=Path, default=Path("/nvme/vjepa_data_32f"))
    parser.add_argument("--manifest", type=Path, default=Path("/data/ek/ek_manifest_test_v2.csv"))
    parser.add_argument("--weights", type=Path, default=Path("/data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("/nvme/vjepa_features"))
    parser.add_argument("--encoder-model", type=str, default="vit_base")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-clips", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)

    dataset = VJEPAFrameDataset(
        clips_root=args.input_root,
        manifest_csv=args.manifest,
        num_frames=args.num_frames,
        image_size=args.resolution,
    )
    if args.max_clips > 0:
        dataset.items = dataset.items[: args.max_clips]
    if len(dataset) == 0:
        missing_count = len(dataset.missing_frames_dirs)
        insufficient_count = len(dataset.insufficient_frames)
        sample_missing = dataset.missing_frames_dirs[:3]
        sample_insufficient = dataset.insufficient_frames[:3]
        raise RuntimeError(
            "No valid clips found. "
            f"input_root={args.input_root}, manifest={args.manifest}, "
            f"missing_frames_dirs={missing_count}, insufficient_frames={insufficient_count}, "
            f"sample_missing={sample_missing}, sample_insufficient={sample_insufficient}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = load_encoder(
        checkpoint_path=args.weights,
        model_name=args.encoder_model,
        resolution=args.resolution,
        num_frames=args.num_frames,
        device=device,
    )

    autocast_dtype = resolve_autocast_dtype(args.precision)
    use_autocast = device.type == "cuda" and args.precision in {"bf16", "fp16"}

    with torch.no_grad():
        pbar = tqdm(loader, total=len(loader), desc="extract")
        for clip_ids, clips in pbar:
            clips = clips.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                patch_tokens = model(clips)
                global_features = patch_tokens.mean(dim=1)

            global_features = global_features.detach().float().cpu()
            for clip_id, feat in zip(clip_ids, global_features):
                out_path = args.output_root / f"{clip_id}.pt"
                torch.save(feat, out_path)

    print(f"[DONE] Saved {len(dataset)} feature files into {args.output_root}")


if __name__ == "__main__":
    main()
