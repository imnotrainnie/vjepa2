import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from multimodal_evals.action_anticipation_frozen.metrics import ClassMeanRecall
from multimodal_evals.ek_action_prediction.dataset import (
    EKMultimodalDataset,
    build_label_maps,
    create_dataloaders,
)
from multimodal_evals.ek_action_prediction.models import init_classifier, init_module
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter
import yaml


ModalityPair = Tuple[str, str]
PAIR_NAMES: Dict[ModalityPair, str] = {
    ("V", "V"): "V->V",
    ("V", "L"): "V->L",
    ("L", "V"): "L->V",
    ("L", "L"): "L->L",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EK multimodal action prediction (linear probe)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "ek_action_prediction.yaml",
    )
    parser.add_argument("--val_only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_distributed() -> Tuple[int, int]:
    world_size, rank = init_distributed()
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "37129")
        dist.init_process_group(backend=backend, rank=0, world_size=1)
        world_size, rank = 1, 0
    return world_size, rank


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_pair(ctx_mods: Sequence[str], tgt_mods: Sequence[str]) -> Dict[ModalityPair, List[int]]:
    indices: Dict[ModalityPair, List[int]] = {}
    for idx, (ctx_mod, tgt_mod) in enumerate(zip(ctx_mods, tgt_mods)):
        indices.setdefault((ctx_mod, tgt_mod), []).append(idx)
    return indices


def select_batch(batch: Dict[str, object], indices: List[int]) -> Dict[str, object]:
    return {
        "video_ctx": batch["video_ctx"][indices],
        "video_tgt": batch["video_tgt"][indices],
        "text_ctx": [batch["text_ctx"][i] for i in indices],
        "text_tgt": [batch["text_tgt"][i] for i in indices],
    }


def compute_logits(
    model,
    classifier,
    batch: Dict[str, object],
    indices: List[int],
    ctx_mod: str,
    tgt_mod: str,
    device: torch.device,
):
    sub_batch = select_batch(batch, indices)
    with torch.no_grad():
        z_ctx = model.encode_context(sub_batch, ctx_mod, device)
        z_tgt = model.encode_target(sub_batch, tgt_mod, device)
        z_pred = model.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=z_tgt.size(1))
        concat_feat = torch.cat([z_ctx, z_pred], dim=1)
    return classifier(concat_feat)


def train_one_epoch(
    epoch: int,
    model,
    classifiers,
    optimizer,
    data_loader,
    device: torch.device,
    use_bfloat16: bool,
    verb_map: Dict[int, int],
    noun_map: Dict[int, int],
    action_map: Dict[Tuple[int, int], int],
    criterion,
    log_interval: int,
):
    for clf in classifiers:
        clf.train(True)

    loss_meter = AverageMeter()
    data_iter = iter(data_loader)
    num_batches = len(data_loader)

    for itr in range(num_batches):
        try:
            batch = next(data_iter)
        except Exception:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        ctx_mods = batch["ctx_mod"]
        tgt_mods = batch["tgt_mod"]
        pair_indices = split_by_pair(ctx_mods, tgt_mods)

        verb_labels = torch.tensor([verb_map[int(v)] for v in batch["verb_class"]], device=device)
        noun_labels = torch.tensor([noun_map[int(n)] for n in batch["noun_class"]], device=device)
        action_labels = torch.tensor(
            [action_map[(int(v), int(n))] for v, n in zip(batch["verb_class"], batch["noun_class"])],
            device=device,
        )

        total_loss = 0.0
        total_count = 0

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            for pair, indices in pair_indices.items():
                if not indices:
                    continue
                ctx_mod, tgt_mod = pair
                outputs = [
                    compute_logits(model, clf, batch, indices, ctx_mod, tgt_mod, device)
                    for clf in classifiers
                ]
                idx_tensor = torch.tensor(indices, device=device)
                v_labels = verb_labels.index_select(0, idx_tensor)
                n_labels = noun_labels.index_select(0, idx_tensor)
                a_labels = action_labels.index_select(0, idx_tensor)

                for out in outputs:
                    loss = (
                        criterion(out["verb"], v_labels)
                        + criterion(out["noun"], n_labels)
                        + criterion(out["action"], a_labels)
                    )
                    total_loss += loss * len(indices)
                    total_count += len(indices)

        if total_count > 0:
            total_loss = total_loss / total_count
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            loss_meter.update(total_loss.item())

        if itr % log_interval == 0 and dist.get_rank() == 0:
            print(f"[Epoch {epoch}] iter={itr} loss={loss_meter.avg:.4f}")


def evaluate(
    model,
    classifiers,
    data_loader,
    device: torch.device,
    use_bfloat16: bool,
    verb_map: Dict[int, int],
    noun_map: Dict[int, int],
    action_map: Dict[Tuple[int, int], int],
):
    for clf in classifiers:
        clf.train(False)

    metrics_by_pair = {
        pair: ClassMeanRecall(num_classes=len(action_map), device=device, k=5) for pair in PAIR_NAMES
    }

    with torch.no_grad():
        for batch in data_loader:
            ctx_mods = batch["ctx_mod"]
            tgt_mods = batch["tgt_mod"]
            pair_indices = split_by_pair(ctx_mods, tgt_mods)

            action_labels = torch.tensor(
                [action_map[(int(v), int(n))] for v, n in zip(batch["verb_class"], batch["noun_class"])],
                device=device,
            )

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                for pair, indices in pair_indices.items():
                    if not indices:
                        continue
                    ctx_mod, tgt_mod = pair
                    outputs = compute_logits(model, classifiers[0], batch, indices, ctx_mod, tgt_mod, device)
                    idx_tensor = torch.tensor(indices, device=device)
                    a_labels = action_labels.index_select(0, idx_tensor)
                    metrics_by_pair[pair](outputs["action"], a_labels)

    if dist.get_rank() == 0:
        for pair, metric in metrics_by_pair.items():
            name = PAIR_NAMES[pair]
            recall, accuracy = summarize_metrics(metric)
            print(f"[VAL] {name}: recall@5={recall:.2f} acc@5={accuracy:.2f}")


def summarize_metrics(metric: ClassMeanRecall) -> Tuple[float, float]:
    tp = metric.TP.clone()
    fn = metric.FN.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tp)
        dist.all_reduce(fn)
    denom = (tp + fn)
    nch = torch.sum(denom > 0)
    recall = 100.0 * torch.sum(tp / (denom + 1e-8)) / torch.clamp(nch, min=1)
    accuracy = 100.0 * tp.sum() / torch.clamp(denom.sum(), min=1)
    return float(recall.item()), float(accuracy.item())


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    world_size, rank = ensure_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    data_cfg = cfg["data"]
    train_dataset, val_dataset, train_loader, val_loader, train_sampler, _ = create_dataloaders(
        jsonl_path=data_cfg["jsonl_path"],
        batch_size=data_cfg.get("batch_size", 4),
        img_size=data_cfg.get("img_size", 384),
        val_split=data_cfg.get("val_split", 0.1),
        split_seed=data_cfg.get("split_seed", 0),
        strict_frames=data_cfg.get("strict_frames", True),
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("persistent_workers", False),
        world_size=world_size,
        rank=rank,
    )

    verb_map, noun_map, action_map = build_label_maps(train_dataset.samples)

    model_cfg = cfg["model"]
    model = init_module(model_cfg, device)
    classifiers = init_classifier(
        embed_dim=model_cfg["projectors"].get("shared_dim", 512),
        num_heads=cfg["classifier"].get("num_heads", 8),
        num_blocks=cfg["classifier"].get("num_blocks", 2),
        num_classifiers=1,
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        device=device,
    )

    if world_size > 1:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]

    optim_cfg = cfg["optimization"]
    optimizer = torch.optim.AdamW(
        [p for clf in classifiers for p in clf.parameters() if p.requires_grad],
        lr=optim_cfg.get("lr", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 0.05),
    )

    num_epochs = int(cfg.get("num_epochs", 5))
    log_interval = int(cfg.get("log_interval", 10))
    val_interval = int(cfg.get("val_interval", 1))
    use_bfloat16 = bool(cfg.get("use_bfloat16", True))
    criterion = torch.nn.CrossEntropyLoss()

    if args.val_only:
        evaluate(model, classifiers, val_loader, device, use_bfloat16, verb_map, noun_map, action_map)
        return

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            epoch,
            model,
            classifiers,
            optimizer,
            train_loader,
            device,
            use_bfloat16,
            verb_map,
            noun_map,
            action_map,
            criterion,
            log_interval,
        )
        if (epoch + 1) % val_interval == 0:
            evaluate(model, classifiers, val_loader, device, use_bfloat16, verb_map, noun_map, action_map)


if __name__ == "__main__":
    main()
