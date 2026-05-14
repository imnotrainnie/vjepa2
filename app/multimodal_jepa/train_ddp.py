import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split

from src.datasets.multimodal_dataset import MultimodalDataset, multimodal_collate_fn
from src.losses.multimodal_loss import MultimodalLoss
from src.models.multimodal_jepa import MultimodalJEPA, build_vjepa2_1_vitb_encoder
from src.models.text_encoder import SigLIPTextEncoder
import torch.multiprocessing as mp

def load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_ready() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_ready() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(config: Dict[str, object], requested_device: str) -> Tuple[torch.device, bool]:
    dist_cfg = config.get("distributed", {})
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = bool(dist_cfg.get("enabled", False) or env_world_size > 1)
    if not distributed:
        return torch.device(requested_device), False

    backend = dist_cfg.get("backend", "nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
    dist.init_process_group(backend=backend)
    return device, True


def cleanup_distributed() -> None:
    if is_dist_ready():
        dist.destroy_process_group()


def create_model(config: Dict[str, object]) -> MultimodalJEPA:
    model_cfg = config["model"]
    v_cfg = model_cfg["v_encoder"]
    l_cfg = model_cfg["l_encoder"]
    projector_cfg = model_cfg["projectors"]
    predictor_cfg = model_cfg["predictor"]

    v_encoder = build_vjepa2_1_vitb_encoder(
        checkpoint_path=v_cfg.get("checkpoint_path"),
        img_size=v_cfg.get("img_size", 384),
        patch_size=v_cfg.get("patch_size", 16),
        tubelet_size=v_cfg.get("tubelet_size", 2),
        num_frames=v_cfg.get("num_frames", 32),
    )
    l_encoder = SigLIPTextEncoder(
        model_name=l_cfg.get("model_name", "google/siglip-large-patch16-384"),
        max_length=l_cfg.get("max_length", 77),
        freeze=l_cfg.get("freeze", True),
        local_files_only=l_cfg.get("local_files_only", False),
    )

    return MultimodalJEPA(
        v_encoder=v_encoder,
        l_encoder=l_encoder,
        video_dim=v_cfg.get("embed_dim", 768),
        text_dim=l_cfg.get("embed_dim", 1024),
        shared_dim=projector_cfg.get("shared_dim", 512),
        projector_hidden_dim=projector_cfg.get("hidden_dim", 1024),
        predictor_dim=predictor_cfg.get("predictor_dim", 384),
        predictor_depth=predictor_cfg.get("depth", 12),
        predictor_num_heads=predictor_cfg.get("num_heads", 8),
        img_size=v_cfg.get("img_size", 384),
        patch_size=v_cfg.get("patch_size", 16),
        tubelet_size=v_cfg.get("tubelet_size", 2),
        freeze_encoders=v_cfg.get("freeze", True) and l_cfg.get("freeze", True),
    )


def split_dataset(dataset: MultimodalDataset, val_split: float, seed: int):
    if val_split <= 0:
        return dataset, None
    if not 0 < val_split < 1:
        raise ValueError("data.val_split must be in [0, 1)")
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no training samples")
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def create_loader(dataset, config: Dict[str, object], training: bool, distributed: bool):
    data_cfg = config["data"]
    sampler = None
    shuffle = training
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=training)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=training,
        collate_fn=multimodal_collate_fn,
        persistent_workers=data_cfg.get("persistent_workers", False) and data_cfg.get("num_workers", 2) > 0,
    )
    return loader, sampler


def create_loaders(config: Dict[str, object], distributed: bool):
    data_cfg = config["data"]
    train_dataset = MultimodalDataset(
        jsonl_path=data_cfg["train_jsonl"],
        img_size=data_cfg.get("img_size", 384),
        strict_frames=data_cfg.get("strict_frames", True),
    )
    if data_cfg.get("val_jsonl"):
        val_dataset = MultimodalDataset(
            jsonl_path=data_cfg["val_jsonl"],
            img_size=data_cfg.get("img_size", 384),
            strict_frames=data_cfg.get("strict_frames", True),
        )
    else:
        train_dataset, val_dataset = split_dataset(
            train_dataset,
            val_split=float(data_cfg.get("val_split", 0.0)),
            seed=int(data_cfg.get("split_seed", 0)),
        )

    train_loader, train_sampler = create_loader(train_dataset, config, training=True, distributed=distributed)
    val_loader, val_sampler = (None, None)
    if val_dataset is not None:
        val_loader, val_sampler = create_loader(val_dataset, config, training=False, distributed=distributed)
    return train_loader, train_sampler, val_loader, val_sampler


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def create_optimizer_and_scheduler(model: MultimodalJEPA, config: Dict[str, object], steps_per_epoch: int):
    training_cfg = config["training"]
    optim_cfg = training_cfg["optimizer"]
    scheduler_cfg = training_cfg["scheduler"]
    base_model = unwrap_model(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_model.v_proj_ctx.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "v_proj_ctx"},
            {"params": base_model.l_proj_ctx.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "l_proj_ctx"},
            # {"params": base_model.v_proj_tgt.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "v_proj_tgt"},
            # {"params": base_model.l_proj_tgt.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "l_proj_tgt"},
            {"params": base_model.predictor.parameters(), "lr": optim_cfg.get("predictor_lr", 5e-5), "name": "predictor"},
        ],
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=optim_cfg.get("weight_decay", 0.05),
    )

    total_steps = max(1, training_cfg["num_epochs"] * steps_per_epoch)
    warmup_steps = int(scheduler_cfg.get("warmup_ratio", 0.1) * total_steps)
    min_lr_ratio = scheduler_cfg.get("min_lr", 1e-6) / optim_cfg.get("lr", 1e-4)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def reduce_loss_totals(loss_totals: Dict[str, float], num_batches: int, device: torch.device) -> Dict[str, float]:
    values = torch.tensor(
        [loss_totals.get("loss", 0.0), loss_totals.get("loss_mse", 0.0), loss_totals.get("loss_sigreg", 0.0), num_batches],
        dtype=torch.float64,
        device=device,
    )
    if is_dist_ready():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denominator = max(1.0, values[-1].item())
    return {
        "loss": values[0].item() / denominator,
        "loss_mse": values[1].item() / denominator,
        "loss_sigreg": values[2].item() / denominator,
    }


def build_loss_fn(config: Dict[str, object], device: torch.device) -> MultimodalLoss:
    loss_cfg = config["training"]["loss"]
    return MultimodalLoss(
        lambda_sigreg=loss_cfg.get("lambda_sigreg", 0.1),
        sigreg_knots=loss_cfg.get("sigreg_knots", 17),
        sigreg_num_proj=loss_cfg.get("sigreg_num_proj", 1024),
    ).to(device)


def init_swanlab(config: Dict[str, object], enabled: Optional[bool] = None):
    swan_cfg = config.get("logging", {}).get("swanlab", {})
    if enabled is None:
        enabled = bool(swan_cfg.get("enabled", False))
    if not enabled:
        return None
    if not bool(swan_cfg.get("log_on_all_ranks", False)) and not is_main_process():
        return None

    try:
        import swanlab
    except ImportError as exc:
        raise ImportError("SwanLab logging is enabled but `swanlab` is not installed.") from exc

    init_kwargs = {
        "project": swan_cfg.get("project", "multimodal-jepa"),
        "workspace": swan_cfg.get("workspace"),
        "experiment_name": swan_cfg.get("experiment_name"),
        "description": swan_cfg.get("description"),
        "group": swan_cfg.get("group"),
        "tags": swan_cfg.get("tags"),
        "config": config,
        "logdir": swan_cfg.get("logdir", config.get("logging", {}).get("log_dir", "./logs")),
        "mode": swan_cfg.get("mode", "cloud"),
        "id": swan_cfg.get("id") or os.environ.get("SWANLAB_RUN_ID"),
        "resume": swan_cfg.get("resume"),
    }
    if bool(swan_cfg.get("log_on_all_ranks", False)) and get_world_size() > 1:
        init_kwargs["parallel"] = swan_cfg.get("parallel", "shared")
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    return swanlab.init(**init_kwargs)


def log_swanlab(run, data: Dict[str, object], step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def save_checkpoint(model, optimizer, scheduler, epoch: int, losses: Dict[str, float], save_dir: str, name: str) -> None:
    if not is_main_process():
        return
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(save_dir) / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "losses": losses,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def load_training_checkpoint(model, optimizer, scheduler, checkpoint_path: str, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", -1)) + 1


def run_epoch(
    model,
    loader: DataLoader,
    loss_fn: MultimodalLoss,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    config: Optional[Dict[str, object]] = None,
    run=None,
    global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "loss_mse": 0.0, "loss_sigreg": 0.0}
    training_cfg = config.get("training", {}) if config else {}
    log_interval = training_cfg.get("log_interval", 100)

    for batch_idx, batch in enumerate(loader):
        with torch.set_grad_enabled(training):
            output = model(batch, device=device)
            loss_dict = loss_fn(output["z_pred"], output["z_tgt"], output["z_ctx"])

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            # ====== [Debug Phase 3: Gradient Flow Check] ======
            # if batch_idx == 0:  # 仅检查一次避免刷屏
            #     print("\n====== Gradient Flow Report ======")
            #     # 1. 验证 Encoder 是否真的被冻结 (不应有梯度)
            #     v_encoder_grads = [p.grad for p in model.module.v_encoder.parameters() if p.grad is not None]
            #     print(f"V-Encoder gradients found: {len(v_encoder_grads)} (Expected: 0)")
                
            #     # 2. 验证可训练层的梯度大小
            #     for name, param in model.module.named_parameters():
            #         if param.requires_grad:
            #             if param.grad is None:
            #                 print(f"⚠️ NO GRADIENT: {name} (Is it disconnected from the computational graph?)")
            #             else:
            #                 grad_norm = param.grad.norm().item()
            #                 if grad_norm == 0:
            #                     print(f"⚠️ ZERO GRADIENT: {name}")
            #                 elif math.isnan(grad_norm) or math.isinf(grad_norm):
            #                     print(f"❌ EXPLODED GRADIENT: {name}")
            #                 # 可选：打印正常梯度范数
            #                 # else:
            #                 #     print(f"✅ Grad OK: {name} | Norm: {grad_norm:.4f}")
            #     print("===================================\n")
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), training_cfg.get("grad_clip", 1.0))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        for key in totals:
            totals[key] += loss_dict[key].item()

        if training and is_main_process() and (batch_idx + 1) % log_interval == 0:
            metrics = {
                "train/loss": loss_dict["loss"].item(),
                "train/loss_mse": loss_dict["loss_mse"].item(),
                "train/loss_sigreg": loss_dict["loss_sigreg"].item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch + 1,
                "train/ctx_is_video": 1 if output["ctx_mod"] == "V" else 0,
                "train/tgt_is_video": 1 if output["tgt_mod"] == "V" else 0,
            }
            log_swanlab(run, metrics, global_step)
            print(
                f"Epoch [{epoch + 1}/{training_cfg.get('num_epochs', '?')}] "
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss_dict['loss'].item():.4f} "
                f"MSE: {loss_dict['loss_mse'].item():.4f} "
                f"SIGReg: {loss_dict['loss_sigreg'].item():.4f} "
                f"Modality: {output['ctx_mod']}->{output['tgt_mod']}"
            )

    return reduce_loss_totals(totals, len(loader), device), global_step


def train(config: Dict[str, object], device: torch.device, distributed: bool, resume: Optional[str] = None) -> None:
    model = create_model(config).to(device)
    train_loader, train_sampler, val_loader, _ = create_loaders(config, distributed=distributed)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, len(train_loader))

    if distributed:
        ddp_cfg = config.get("distributed", {})
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=ddp_cfg.get("find_unused_parameters", True),
        )

    start_epoch = load_training_checkpoint(model, optimizer, scheduler, resume, device) if resume else 0
    
    loss_fn = build_loss_fn(config, device)
    run = init_swanlab(config)
    # if is_dist_ready():
    #     dist.barrier()
    training_cfg = config["training"]
    global_step = start_epoch * len(train_loader)
    #在 train_loader 创建后，进入 epoch 前
    # print("====== [Debug Phase 1: Data Loader Sanity Check] ======")
    # sample_batch = next(iter(train_loader))
    # print(f"Batch Keys: {list(sample_batch.keys())}")
    
    # video_ctx = sample_batch["video_ctx"]
    # # 期望形状通常是 [B, T, C, H, W] 或 [B, C, T, H, W]，视你底层 ViT 的要求而定
    # print(f"video_ctx shape: {video_ctx.shape}, dtype: {video_ctx.dtype}")
    # print(f"video_ctx Value Range: Min={video_ctx.min().item():.3f}, Max={video_ctx.max().item():.3f}")
    # assert not torch.isnan(video_ctx).any(), "Found NaN in video inputs!"
    
    # text_ctx = sample_batch["text_ctx"]
    # print(f"text_ctx sample (first 2): {text_ctx[:2]}")
    # print("=====================111==================================")
    for epoch in range(start_epoch, training_cfg["num_epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # def trace_shapes_hook(module, input, output):
        #     print("=====================111==================================")
        # # 仅在第一个 batch 打印
        #     if not getattr(module, "_has_printed", False):
        #         in_shape = input[0].shape if isinstance(input, tuple) and len(input) > 0 else "N/A"
        #         out_shape = output.shape if isinstance(output, torch.Tensor) else "Dict/Tuple"
        #         print(f"[Data Flow] {module.__class__.__name__} | In: {in_shape} -> Out: {out_shape}")
        #         module._has_printed = True

        # # 挂载到关键组件上
        # model.module.v_proj_ctx.register_forward_hook(trace_shapes_hook)
        # model.module.l_proj_ctx.register_forward_hook(trace_shapes_hook)
        # model.module.predictor.register_forward_hook(trace_shapes_hook)
        # print("=======================================================")
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            run=run,
            global_step=global_step,
        )
        if is_main_process():
            print(f"Epoch [{epoch + 1}] train: {train_metrics}")
            log_swanlab(run, {f"epoch_train/{key}": value for key, value in train_metrics.items()}, global_step)

        val_metrics = None
        if val_loader is not None and (epoch + 1) % training_cfg.get("val_interval", 1) == 0:
            val_metrics, _ = run_epoch(model, val_loader, loss_fn, device, epoch=epoch, config=config)
            if is_main_process():
                print(f"Epoch [{epoch + 1}] val: {val_metrics}")
                log_swanlab(run, {f"val/{key}": value for key, value in val_metrics.items()}, global_step)

        if (epoch + 1) % training_cfg.get("save_interval", 10) == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics or train_metrics,
                config["logging"]["save_dir"],
                name=f"checkpoint_epoch_{epoch + 1}.pt",
            )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        training_cfg["num_epochs"] - 1,
        train_metrics,
        config["logging"]["save_dir"],
        "checkpoint_last.pt",
    )


def evaluate(config: Dict[str, object], device: torch.device, distributed: bool, checkpoint: Optional[str]) -> Dict[str, float]:
    model = create_model(config).to(device)
    _, _, val_loader, _ = create_loaders(config, distributed=distributed)
    if val_loader is None:
        raise ValueError("Evaluation requires data.val_jsonl or data.val_split > 0")
    if checkpoint:
        load_training_checkpoint(model, None, None, checkpoint, device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    loss_fn = build_loss_fn(config, device)
    run = init_swanlab(config)
    metrics, _ = run_epoch(model, val_loader, loss_fn, device, config=config)
    if is_main_process():
        print(f"Evaluation: {metrics}")
        log_swanlab(run, {f"eval/{key}": value for key, value in metrics.items()}, step=0)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate multimodal JEPA with DDP and SwanLab")
    parser.add_argument("--config", default="app/multimodal_jepa/configs/multimodal_base.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None, help="Path to a training checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for eval-only mode")
    return parser.parse_args()


def main() -> None:
    # try:
    #     mp.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = load_config(args.config)
    device, distributed = setup_distributed(config, args.device)
    try:
        if args.eval_only:
            evaluate(config, device, distributed, args.checkpoint or args.resume)
        else:
            train(config, device, distributed, resume=args.resume)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
