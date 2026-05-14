import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from src.datasets.utils.text_transforms import clean_text


class MultimodalDataset(Dataset):
    """JSONL dataset for context/target video frames and text state descriptions."""

    def __init__(
        self,
        jsonl_path: str,
        video_transform=None,
        img_size: int = 384,
        strict_frames: bool = True,
    ):
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.img_size = img_size
        self.strict_frames = strict_frames
        self.samples = self._load_jsonl(self.jsonl_path)
        self.video_transform = video_transform or transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_jsonl(jsonl_path: Path) -> List[Dict[str, object]]:
        samples = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {jsonl_path}") from exc
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        video_ctx = self._load_frames(sample.get("visual_frame_context_paths", []), expected_frames=32)
        video_tgt = self._load_frames(sample.get("visual_frame_target_paths", []), expected_frames=2)

        return {
            "video_ctx": video_ctx,
            "video_tgt": video_tgt,
            "text_ctx": clean_text(sample.get("text_state_context")),
            "text_tgt": clean_text(sample.get("text_state_target")),
            "clip_id": str(sample.get("clip_id", idx)),
            "action_narration": clean_text(sample.get("action_narration")),
        }

    def _load_frames(self, frame_paths: Iterable[str], expected_frames: int) -> torch.Tensor:
        paths = list(frame_paths)
        if len(paths) != expected_frames and self.strict_frames:
            raise ValueError(f"Expected {expected_frames} frames, got {len(paths)}")

        frames = []
        for frame_path in paths:
            with Image.open(frame_path) as image:
                frames.append(self.video_transform(image.convert("RGB")))

        if not frames:
            raise ValueError("Frame path list is empty")
        return torch.stack(frames, dim=1)


def multimodal_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "video_ctx": torch.stack([item["video_ctx"] for item in batch]),
        "video_tgt": torch.stack([item["video_tgt"] for item in batch]),
        "text_ctx": [item["text_ctx"] for item in batch],
        "text_tgt": [item["text_tgt"] for item in batch],
        "clip_id": [item["clip_id"] for item in batch],
        "action_narration": [item["action_narration"] for item in batch],
    }


def make_multimodal_dataset(
    jsonl_path: str,
    batch_size: int,
    img_size: int = 384,
    video_transform=None,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    world_size: int = 1,
    rank: int = 0,
    persistent_workers: bool = False,
) -> tuple[MultimodalDataset, DataLoader, Optional[DistributedSampler]]:
    dataset = MultimodalDataset(jsonl_path=jsonl_path, video_transform=video_transform, img_size=img_size)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=multimodal_collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    return dataset, loader, sampler
