# Implements EK_PLAN_PART3 §5
import argparse
import logging
import os
import pprint
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel

from multimodal_evals.ek_action_prediction.checkpoint import (
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from multimodal_evals.ek_action_prediction.dataset import build_label_maps, create_dataloaders
from multimodal_evals.ek_action_prediction.metrics import LocalClassMeanRecall
from multimodal_evals.ek_action_prediction.models import _log_trainable_params, init_classifier, init_module
from multimodal_evals.ek_action_prediction.optim import init_opt
from src.utils.distributed import init_distributed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ModalityPair = Tuple[str, str]
PAIR_NAMES: Dict[ModalityPair, str] = {
    ("V", "V"): "V->V",
    ("V", "L"): "V->L",
    ("L", "V"): "L->V",
    ("L", "L"): "L->L",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EK multimodal action prediction")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "ek_action_prediction.yaml")
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
    if dist.is_available() and dist.is_initialized():
        world_size, rank = dist.get_world_size(), dist.get_rank()
    return world_size, rank


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_pair(ctx_mods: Sequence[str], tgt_mods: Sequence[str]) -> Dict[ModalityPair, List[int]]:
    indices: Dict[ModalityPair, List[int]] = {pair: [] for pair in PAIR_NAMES}
    for idx, (ctx_mod, tgt_mod) in enumerate(zip(ctx_mods, tgt_mods)):
        indices[(ctx_mod, tgt_mod)].append(idx)
    return indices


def select_batch(batch: Dict[str, object], indices: List[int]) -> Dict[str, object]:
    return {
        "video_ctx": batch["video_ctx"][indices],
        "video_tgt": batch["video_tgt"][indices],
        "text_ctx": [batch["text_ctx"][i] for i in indices],
        "text_tgt": [batch["text_tgt"][i] for i in indices],
    }


def compute_concat_feat(model, batch: Dict[str, object], indices: List[int], ctx_mod: str, tgt_mod: str, device: torch.device):
    sub_batch = select_batch(batch, indices)
    with torch.no_grad():
        z_ctx = model.encode_context(sub_batch, ctx_mod, device)
        z_tgt = model.encode_target(sub_batch, tgt_mod, device)
        z_pred = model.predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=z_tgt.size(1))
        concat_feat = torch.cat([z_ctx, z_pred], dim=1).detach()
    return concat_feat


def compute_logits(model, classifier, batch: Dict[str, object], indices: List[int], ctx_mod: str, tgt_mod: str, device: torch.device):
    concat_feat = compute_concat_feat(model, batch, indices, ctx_mod, tgt_mod, device)
    return classifier(concat_feat)


def train_one_epoch(
    epoch: int,
    model,
    classifiers: List[nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[object],
    wd_schedulers: List[object],
    scalers: List[object],
    data_loader,
    device: torch.device,
    use_bfloat16: bool,
    verb_map: Dict[int, int],
    noun_map: Dict[int, int],
    action_map: Dict[Tuple[int, int], int],
    criterion: nn.Module,
    log_interval: int,
    rank: int,
) -> Dict[str, float]:
    for classifier in classifiers:
        classifier.train(True)

    running_loss = 0.0
    running_batches = 0
    for itr, batch in enumerate(data_loader):
        for scheduler in schedulers:
            scheduler.step()
        for wd_scheduler in wd_schedulers:
            wd_scheduler.step()

        verb_labels = torch.tensor([verb_map[int(v)] for v in batch["verb_class"]], device=device)
        noun_labels = torch.tensor([noun_map[int(n)] for n in batch["noun_class"]], device=device)
        action_labels = torch.tensor(
            [action_map[(int(v), int(n))] for v, n in zip(batch["verb_class"], batch["noun_class"])],
            device=device,
        )

        pair_indices = split_by_pair(batch["ctx_mod"], batch["tgt_mod"])
        b_total = sum(len(indices) for indices in pair_indices.values())
        if b_total == 0:
            continue

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        pair_loss_log = {}
        batch_loss_value = 0.0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            for pair, indices in pair_indices.items():
                if not indices:
                    continue
                ctx_mod, tgt_mod = pair
                idx_tensor = torch.tensor(indices, device=device)
                concat_feat = compute_concat_feat(model, batch, indices, ctx_mod, tgt_mod, device)
                v_lbl = verb_labels.index_select(0, idx_tensor)
                n_lbl = noun_labels.index_select(0, idx_tensor)
                a_lbl = action_labels.index_select(0, idx_tensor)

                for classifier in classifiers:
                    logits = classifier(concat_feat)
                    loss = (
                        criterion(logits["verb"], v_lbl)
                        + criterion(logits["noun"], n_lbl)
                        + criterion(logits["action"], a_lbl)
                    )
                    weight = len(indices) / b_total
                    (loss * weight).backward()
                    pair_loss_log[pair] = loss.detach().item()
                    batch_loss_value += loss.detach().item() * weight

        for optimizer in optimizers:
            optimizer.step()

        running_loss += batch_loss_value
        running_batches += 1
        if itr % log_interval == 0 and rank == 0:
            msg = " | ".join(f"{pair[0]}->{pair[1]}: {value:.3f}" for pair, value in pair_loss_log.items())
            logger.info("[Train E%d it%d] %s", epoch, itr, msg)

    return {"loss": running_loss / max(1, running_batches)}


def evaluate(
    model,
    classifiers: List[nn.Module],
    data_loader,
    device: torch.device,
    use_bfloat16: bool,
    verb_map: Dict[int, int],
    noun_map: Dict[int, int],
    action_map: Dict[Tuple[int, int], int],
    rank: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    for classifier in classifiers:
        classifier.train(False)

    class_counts = {"verb": len(verb_map), "noun": len(noun_map), "action": len(action_map)}
    metrics = {
        pair: {head: LocalClassMeanRecall(class_counts[head], 5, device) for head in class_counts}
        for pair in PAIR_NAMES
    }

    with torch.no_grad():
        for batch in data_loader:
            pair_indices = split_by_pair(batch["ctx_mod"], batch["tgt_mod"])
            verb_labels = torch.tensor([verb_map[int(v)] for v in batch["verb_class"]], device=device)
            noun_labels = torch.tensor([noun_map[int(n)] for n in batch["noun_class"]], device=device)
            action_labels = torch.tensor(
                [action_map[(int(v), int(n))] for v, n in zip(batch["verb_class"], batch["noun_class"])],
                device=device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                for pair, indices in pair_indices.items():
                    if not indices:
                        continue
                    ctx_mod, tgt_mod = pair
                    idx_tensor = torch.tensor(indices, device=device)
                    logits = compute_logits(model, classifiers[0], batch, indices, ctx_mod, tgt_mod, device)
                    metrics[pair]["verb"].update(logits["verb"], verb_labels.index_select(0, idx_tensor))
                    metrics[pair]["noun"].update(logits["noun"], noun_labels.index_select(0, idx_tensor))
                    metrics[pair]["action"].update(logits["action"], action_labels.index_select(0, idx_tensor))

    return summarize_metrics(metrics, rank)


def summarize_metrics(metrics, rank: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pair, head_map in metrics.items():
        pair_name = PAIR_NAMES[pair]
        report[pair_name] = {}
        for head, metric in head_map.items():
            recall, acc = metric.compute(reduce=True)
            report[pair_name][head] = {"recall@5": recall, "acc@5": acc}
            if rank == 0:
                logger.info("[VAL] %-5s %-6s: recall@5=%6.2f  acc@5=%6.2f", pair_name, head, recall, acc)
    return report


def _output_folder(cfg: Dict[str, object]) -> str:
    folder = cfg.get("folder", "/data/vjepa2/logs/ek_action_prediction")
    tag = cfg.get("tag")
    return os.path.join(folder, tag) if tag else folder


def validate_config(cfg: Dict[str, object]) -> None:
    model_ckpt = Path(cfg["model"]["checkpoint"])
    if not model_ckpt.exists():
        raise FileNotFoundError(model_ckpt)
    data_path = Path(cfg["data"]["jsonl_path"])
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    shared_dim = int(cfg["model"]["projectors"]["shared_dim"])
    num_heads = int(cfg["classifier"]["num_heads"])
    assert shared_dim % num_heads == 0, "classifier.num_heads must divide projectors.shared_dim"
    opt_kwargs = cfg["optimization"].get("multihead_kwargs")
    if not isinstance(opt_kwargs, list) or not opt_kwargs:
        raise ValueError("optimization.multihead_kwargs must be a non-empty list")


def _ddp_device_ids(device: torch.device):
    if device.type != "cuda":
        return None
    return [torch.cuda.current_device()]


def main(args_eval=None, resume_preempt: bool = False):
    if args_eval is None:
        args = parse_args()
        cfg = load_config(args.config)
        cfg["val_only"] = bool(cfg.get("val_only", False) or args.val_only)
    else:
        cfg = args_eval

    validate_config(cfg)
    set_seed(int(cfg.get("seed", 0)))

    world_size, rank = ensure_distributed()
    if rank != 0:
        logger.setLevel(logging.ERROR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())

    output_folder = _output_folder(cfg)
    if rank == 0:
        os.makedirs(output_folder, exist_ok=True)
        logger.info("EK action prediction config:\n%s", pprint.pformat(cfg))

    train_dataset, val_dataset, train_loader, val_loader, train_sampler, _ = create_dataloaders(
        **cfg["data"],
        world_size=world_size,
        rank=rank,
    )
    verb_map, noun_map, action_map = build_label_maps(train_dataset.samples)

    model = init_module(cfg["model"], device, rank=rank)
    num_classifiers = len(cfg["optimization"]["multihead_kwargs"])
    classifiers = init_classifier(
        embed_dim=cfg["model"]["projectors"]["shared_dim"],
        num_heads=cfg["classifier"]["num_heads"],
        num_blocks=cfg["classifier"]["num_blocks"],
        num_classifiers=num_classifiers,
        verb_classes=verb_map,
        noun_classes=noun_map,
        action_classes=action_map,
        device=device,
    )
    _log_trainable_params(model, classifiers, rank)

    if world_size > 1:
        classifiers = [DistributedDataParallel(classifier, device_ids=_ddp_device_ids(device), static_graph=True) for classifier in classifiers]

    optimizers, scalers, schedulers, wd_schedulers = init_opt(
        classifiers=classifiers,
        opt_kwargs=cfg["optimization"]["multihead_kwargs"],
        iterations_per_epoch=len(train_loader),
        num_epochs=int(cfg["num_epochs"]),
        use_bfloat16=bool(cfg["use_bfloat16"]),
    )

    latest_path = os.path.join(output_folder, "latest.pt")
    start_epoch = 0
    resume_path = cfg.get("resume_checkpoint") or latest_path
    if cfg.get("resume", False) and os.path.exists(resume_path):
        start_epoch = load_classifier_checkpoint(resume_path, classifiers, optimizers, map_location=device)

    use_bfloat16 = bool(cfg.get("use_bfloat16", True))
    if cfg.get("val_only", False):
        evaluate(model, classifiers, val_loader, device, use_bfloat16, verb_map, noun_map, action_map, rank)
        return

    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, int(cfg["num_epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            epoch,
            model,
            classifiers,
            optimizers,
            schedulers,
            wd_schedulers,
            scalers,
            train_loader,
            device,
            use_bfloat16,
            verb_map,
            noun_map,
            action_map,
            criterion,
            int(cfg["log_interval"]),
            rank,
        )
        if (epoch + 1) % int(cfg.get("val_interval", 1)) == 0:
            evaluate(model, classifiers, val_loader, device, use_bfloat16, verb_map, noun_map, action_map, rank)
        if (epoch + 1) % int(cfg.get("save_interval", 5)) == 0:
            save_classifier_checkpoint(latest_path, classifiers, optimizers, epoch + 1, rank, world_size)

    save_classifier_checkpoint(latest_path, classifiers, optimizers, int(cfg["num_epochs"]), rank, world_size)


if __name__ == "__main__":
    main()
