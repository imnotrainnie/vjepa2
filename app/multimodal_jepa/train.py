import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from src.datasets.multimodal_dataset import make_multimodal_dataset
from src.losses.multimodal_loss import MultimodalLoss
from src.models.multimodal_jepa import MultimodalJEPA, build_vjepa2_1_vitb_encoder
from src.models.text_encoder import SigLIPTextEncoder
import torch.multiprocessing as mp

def load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
        max_length=l_cfg.get("max_length", 64),
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


def create_loaders(config: Dict[str, object]):
    data_cfg = config["data"]
    _, train_loader, train_sampler = make_multimodal_dataset(
        jsonl_path=data_cfg["train_jsonl"],
        batch_size=data_cfg.get("batch_size", 16),
        img_size=data_cfg.get("img_size", 384),
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
        shuffle=True,
    )
    return train_loader, train_sampler


def create_optimizer_and_scheduler(model: MultimodalJEPA, config: Dict[str, object], steps_per_epoch: int):
    training_cfg = config["training"]
    optim_cfg = training_cfg["optimizer"]
    scheduler_cfg = training_cfg["scheduler"]
    num_epochs = training_cfg["num_epochs"]

    optimizer = torch.optim.AdamW(
        [
            {"params": model.v_proj_ctx.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "v_proj_ctx"},
            {"params": model.l_proj_ctx.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "l_proj_ctx"},
            # {"params": model.v_proj_tgt.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "v_proj_tgt"},
            # {"params": model.l_proj_tgt.parameters(), "lr": optim_cfg.get("lr", 1e-4), "name": "l_proj_tgt"},
            {"params": model.predictor.parameters(), "lr": optim_cfg.get("predictor_lr", 5e-5), "name": "predictor"},
        ],
        betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=optim_cfg.get("weight_decay", 0.05),
    )

    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = int(scheduler_cfg.get("warmup_ratio", 0.1) * total_steps)
    min_lr_ratio = scheduler_cfg.get("min_lr", 1e-6) / optim_cfg.get("lr", 1e-4)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, epoch: int, losses: Dict[str, float], save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def train(config: Dict[str, object], device: torch.device) -> None:
    model = create_model(config).to(device)
    train_loader, train_sampler = create_loaders(config)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, len(train_loader))

    training_cfg = config["training"]
    loss_cfg = training_cfg["loss"]
    loss_fn = MultimodalLoss(
        lambda_sigreg=loss_cfg.get("lambda_sigreg", 0.1),
        sigreg_knots=loss_cfg.get("sigreg_knots", 17),
        sigreg_num_proj=loss_cfg.get("sigreg_num_proj", 1024),
    ).to(device)

    # 在 train_loader 创建后，进入 epoch 前
    print("====== [Debug Phase 1: Data Loader Sanity Check] ======")
    sample_batch = next(iter(train_loader))
    print(f"Batch Keys: {list(sample_batch.keys())}")
    
    video_ctx = sample_batch["video_ctx"]
    # 期望形状通常是 [B, T, C, H, W] 或 [B, C, T, H, W]，视你底层 ViT 的要求而定
    print(f"video_ctx shape: {video_ctx.shape}, dtype: {video_ctx.dtype}")
    print(f"video_ctx Value Range: Min={video_ctx.min().item():.3f}, Max={video_ctx.max().item():.3f}")
    assert not torch.isnan(video_ctx).any(), "Found NaN in video inputs!"
    
    text_ctx = sample_batch["text_ctx"]
    print(f"text_ctx sample (first 2): {text_ctx[:2]}")
    print("=====================111==================================")

    for epoch in range(training_cfg["num_epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        def trace_shapes_hook(module, input, output):
            print("=====================111==================================")
        # 仅在第一个 batch 打印
            if not getattr(module, "_has_printed", False):
                in_shape = input[0].shape if isinstance(input, tuple) and len(input) > 0 else "N/A"
                out_shape = output.shape if isinstance(output, torch.Tensor) else "Dict/Tuple"
                print(f"[Data Flow] {module.__class__.__name__} | In: {in_shape} -> Out: {out_shape}")
                module._has_printed = True

        # 挂载到关键组件上
        model.v_proj_ctx.register_forward_hook(trace_shapes_hook)
        model.l_proj_ctx.register_forward_hook(trace_shapes_hook)
        model.predictor.register_forward_hook(trace_shapes_hook)
        print("=======================================================")
        model.train()
        epoch_losses = {"loss": 0.0, "loss_mse": 0.0, "loss_sigreg": 0.0}

        for batch_idx, batch in enumerate(train_loader):
            output = model(batch, device=device)
            loss_dict = loss_fn(output["z_pred"], output["z_tgt"], output["z_ctx"])

            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.get("grad_clip", 1.0))
            optimizer.step()
            scheduler.step()

            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()

            if (batch_idx + 1) % training_cfg.get("log_interval", 100) == 0:
                print(
                    f"Epoch [{epoch + 1}/{training_cfg['num_epochs']}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss_dict['loss'].item():.4f} "
                    f"MSE: {loss_dict['loss_mse'].item():.4f} "
                    f"SIGReg: {loss_dict['loss_sigreg'].item():.4f} "
                    f"Modality: {output['ctx_mod']}->{output['tgt_mod']}"
                )

        for key in epoch_losses:
            epoch_losses[key] /= max(1, len(train_loader))
        print(f"Epoch [{epoch + 1}] Summary: {epoch_losses}")

        if (epoch + 1) % training_cfg.get("save_interval", 10) == 0:
            save_checkpoint(model, optimizer, epoch, epoch_losses, config["logging"]["save_dir"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal JEPA")
    parser.add_argument("--config", default="app/multimodal_jepa/configs/multimodal_base.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    # 1. 关键解药：强制设置进程启动方式为 spawn，必须放在所有逻辑的最前面！
    # try:
    #     mp.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass
    
    #os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from app.multimodal_jepa.train_ddp import main

    main()
