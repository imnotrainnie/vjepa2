# Implements EK_PLAN_PART1 §2
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from src.datasets.utils.text_transforms import clean_text


ModalityPair = Tuple[str, str]


@dataclass
class EKSample:
    clip_id: str
    verb_class: int
    noun_class: int
    text_state_context: str
    text_state_target: str
    visual_frame_context_paths: List[str]
    visual_frame_target_paths: List[str]
    action_narration: str


class EKMultimodalDataset(Dataset):
    """Epic-Kitchens EK dataset expanded into four modality pairs per sample."""

    MODALITY_PAIRS: Sequence[ModalityPair] = (("V", "V"), ("V", "L"), ("L", "V"), ("L", "L"))

    def __init__(
        self,
        jsonl_path: str,
        split: str = "train",
        val_split: float = 0.1,
        split_seed: int = 0,
        img_size: int = 384,
        strict_frames: bool = True,
        video_transform=None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        self.jsonl_path = Path(jsonl_path)
        self.samples = self._load_jsonl(self.jsonl_path)
        self.img_size = img_size
        self.strict_frames = strict_frames

        rng = random.Random(split_seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1.0 - val_split))
        if split == "train":
            split_indices = indices[:split_idx]
        else:
            split_indices = indices[split_idx:]

        self.split_samples = [self.samples[i] for i in split_indices]

        self.video_transform = video_transform or transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_jsonl(jsonl_path: Path) -> List[EKSample]:
        samples: List[EKSample] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {jsonl_path}") from exc

                samples.append(
                    EKSample(
                        clip_id=str(payload.get("clip_id", line_number)),
                        verb_class=int(payload["verb_class"]),
                        noun_class=int(payload["noun_class"]),
                        text_state_context=clean_text(payload.get("text_state_context")),
                        text_state_target=clean_text(payload.get("text_state_target")),
                        visual_frame_context_paths=list(payload.get("visual_frame_context_paths", [])),
                        visual_frame_target_paths=list(payload.get("visual_frame_target_paths", [])),
                        action_narration=clean_text(payload.get("action_narration")),
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.split_samples) * len(self.MODALITY_PAIRS)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        base_idx = idx // len(self.MODALITY_PAIRS)
        pair_idx = idx % len(self.MODALITY_PAIRS)
        ctx_mod, tgt_mod = self.MODALITY_PAIRS[pair_idx]
        sample = self.split_samples[base_idx]

        video_ctx = self._load_frames(sample.visual_frame_context_paths, expected_frames=32)
        video_tgt = self._load_frames(sample.visual_frame_target_paths, expected_frames=2)

        return {
            "video_ctx": video_ctx,
            "video_tgt": video_tgt,
            "text_ctx": sample.text_state_context,
            "text_tgt": sample.text_state_target,
            "verb_class": sample.verb_class,
            "noun_class": sample.noun_class,
            "clip_id": sample.clip_id,
            "action_narration": sample.action_narration,
            "ctx_mod": ctx_mod,
            "tgt_mod": tgt_mod,
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


def build_label_maps(samples: Sequence[EKSample]) -> Tuple[Dict[int, int], Dict[int, int], Dict[Tuple[int, int], int]]:
    verbs = sorted({s.verb_class for s in samples})
    nouns = sorted({s.noun_class for s in samples})
    actions = sorted({(s.verb_class, s.noun_class) for s in samples})

    verb_map = {verb: idx for idx, verb in enumerate(verbs)}
    noun_map = {noun: idx for idx, noun in enumerate(nouns)}
    action_map = {action: idx for idx, action in enumerate(actions)}
    return verb_map, noun_map, action_map


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "video_ctx": torch.stack([item["video_ctx"] for item in batch]),
        "video_tgt": torch.stack([item["video_tgt"] for item in batch]),
        "text_ctx": [item["text_ctx"] for item in batch],
        "text_tgt": [item["text_tgt"] for item in batch],
        "verb_class": torch.tensor([item["verb_class"] for item in batch], dtype=torch.long),
        "noun_class": torch.tensor([item["noun_class"] for item in batch], dtype=torch.long),
        "clip_id": [item["clip_id"] for item in batch],
        "action_narration": [item["action_narration"] for item in batch],
        "ctx_mod": [item["ctx_mod"] for item in batch],
        "tgt_mod": [item["tgt_mod"] for item in batch],
    }


multi_pair_collate_fn = collate_fn


def create_dataloaders(
    jsonl_path: str,
    batch_size: int,
    img_size: int = 384,
    val_split: float = 0.1,
    split_seed: int = 0,
    strict_frames: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[EKMultimodalDataset, EKMultimodalDataset, DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    train_dataset = EKMultimodalDataset(
        jsonl_path=jsonl_path,
        split="train",
        val_split=val_split,
        split_seed=split_seed,
        img_size=img_size,
        strict_frames=strict_frames,
    )
    val_dataset = EKMultimodalDataset(
        jsonl_path=jsonl_path,
        split="val",
        val_split=val_split,
        split_seed=split_seed,
        img_size=img_size,
        strict_frames=strict_frames,
    )

    train_sampler = None
    val_sampler = None
    shuffle = True
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler
