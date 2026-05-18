# 多模态 V-JEPA EK 动作预测下游任务 实施蓝图 (Part 4A / 4)
> 本部分：**YAML 配置完整字段、`main()` 入口装配、`main_distributed.py` 兼容性。**

---

## 11. YAML 配置 (`ek_action_prediction.yaml`) 终版字段

> 在原 yaml 之上**追加 / 显式化**字段；保持 `model.*` 子树不变。

```yaml
eval_name: multimodal_evals.ek_action_prediction       # 供 scaffold.py 转发
seed: 0
num_epochs: 20
log_interval: 5
val_interval: 1
save_interval: 5
use_bfloat16: true
val_only: false
resume: false

folder: /data/vjepa2/logs/ek_action_prediction         # 日志与 latest.pt 落盘根目录
tag: linear_probe_run1                                 # 可空；非空则拼到 folder 下

model:
  checkpoint: /data/vjepa2/checkpoints/checkpoint_epoch_100.pt
  v_encoder:
    checkpoint_path: /data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
    img_size: 384
    patch_size: 16
    tubelet_size: 2
    num_frames: 32
    embed_dim: 768
    freeze: true
  l_encoder:
    model_name: "google/siglip-large-patch16-384"
    max_length: 64
    embed_dim: 1024
    freeze: true
    local_files_only: false
  projectors:
    shared_dim: 512
    hidden_dim: 1024
  predictor:
    predictor_dim: 384
    depth: 12
    num_heads: 8

classifier:
  num_blocks: 2
  num_heads: 8                                          # 必须能整除 shared_dim=512

optimization:
  multihead_kwargs:
    - lr: 0.001
      start_lr: 0.0001
      final_lr: 0.00001
      warmup: 1                                         # warmup epochs
      weight_decay: 0.05
      final_weight_decay: 0.05

data:
  jsonl_path: /data/eku/vjepa_state_transitions.jsonl
  val_split: 0.1
  split_seed: 0
  img_size: 384
  batch_size: 4
  num_workers: 2
  pin_memory: true
  persistent_workers: false
  strict_frames: true
```

### 11.1 字段校验（Codex 必须实现）
启动时校验，失败 `raise`：
1. `model.checkpoint` 存在；不存在则 `FileNotFoundError`。
2. `classifier.num_heads` 整除 `projectors.shared_dim`；否则 `assert`。
3. `optimization.multihead_kwargs` 是非空 list；长度与 `init_classifier(num_classifiers=...)` 一致。
4. `data.jsonl_path` 文件存在。

---

## 12. `main()` 入口装配 (`eval.py` 重写)

### 12.1 顶层流程伪代码（按此顺序，逐步严格）
```text
def main(args_eval=None, resume_preempt=False):
    # ① 兼容两种调用：scaffold.py(args_eval) / 直接命令行
    if args_eval is None:
        args = parse_args()
        cfg = load_config(args.config)
        cfg["val_only"] = cfg.get("val_only", False) or args.val_only
    else:
        cfg = args_eval

    set_seed(int(cfg.get("seed", 0)))

    # ② 分布式
    world_size, rank = init_distributed()
    if not (dist.is_available() and dist.is_initialized()):
        ensure_distributed()                       # 单进程模式启用 gloo / nccl 1×1
        world_size, rank = 1, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)

    # ③ 数据
    train_dataset, val_dataset, train_loader, val_loader, train_sampler, _ = create_dataloaders(
        **cfg["data"], world_size=world_size, rank=rank,
    )
    verb_map, noun_map, action_map = build_label_maps(train_dataset.samples)

    # ④ 模型
    model = init_module(cfg["model"], device)                 # frozen, eval()
    classifiers = init_classifier(
        embed_dim=cfg["model"]["projectors"]["shared_dim"],
        num_heads=cfg["classifier"]["num_heads"],
        num_blocks=cfg["classifier"]["num_blocks"],
        num_classifiers=len(cfg["optimization"]["multihead_kwargs"]),
        verb_classes=verb_map, noun_classes=noun_map, action_classes=action_map,
        device=device,
    )
    _log_trainable_params(model, classifiers, rank)

    if world_size > 1:
        classifiers = [DistributedDataParallel(c, device_ids=[device.index], static_graph=True)
                       for c in classifiers]

    # ⑤ 优化器 / scheduler
    iterations_per_epoch = len(train_loader)
    optimizers, scalers, schedulers, wd_schedulers = init_opt(
        classifiers=classifiers,
        opt_kwargs=cfg["optimization"]["multihead_kwargs"],
        iterations_per_epoch=iterations_per_epoch,
        num_epochs=int(cfg["num_epochs"]),
        use_bfloat16=bool(cfg["use_bfloat16"]),
    )

    # ⑥ 续训 / 仅推断
    start_epoch = 0
    latest_path = os.path.join(cfg["folder"], cfg.get("tag", ""), "latest.pt")
    if cfg.get("resume", False) and os.path.exists(latest_path):
        start_epoch = load_classifier_checkpoint(latest_path, classifiers, optimizers)

    if cfg.get("val_only", False):
        evaluate(model, classifiers, val_loader, device,
                 cfg["use_bfloat16"], verb_map, noun_map, action_map, rank)
        return

    # ⑦ 训练
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, int(cfg["num_epochs"])):
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, classifiers, optimizers, schedulers, wd_schedulers,
                        scalers, train_loader, device, cfg["use_bfloat16"],
                        verb_map, noun_map, action_map, criterion, int(cfg["log_interval"]), rank)

        if (epoch + 1) % int(cfg.get("val_interval", 1)) == 0:
            evaluate(model, classifiers, val_loader, device, cfg["use_bfloat16"],
                     verb_map, noun_map, action_map, rank)

        if (epoch + 1) % int(cfg.get("save_interval", 5)) == 0:
            save_classifier_checkpoint(latest_path, classifiers, optimizers,
                                       epoch + 1, rank, world_size)
```

### 12.2 与 `main.py` / `main_distributed.py` / `scaffold.py` 的兼容性
- `scaffold.py` 通过 `cfg["eval_name"]` 找到 `evals.{eval_name}.eval.main` 或 `app.{...}.eval.main`，
  本任务希望从 `multimodal_evals.ek_action_prediction.eval.main` 加载——
  **不在允许修改范围内**改造 `scaffold.py`。Codex 必须采用以下方案：

#### 12.2.1 方案（默认）：自带 launcher
- 新建 `/data/vjepa2/multimodal_evals/ek_action_prediction/main.py`（允许，本目录之下）。
- 仿造 `multimodal_evals/main.py` 的多进程模式：
  - `argparse` 解析 `--fname`/`--devices`/`--debugmode`/`--val_only`/`--checkpoint`/`--batch_size`；
  - `process_main` 内 `os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank].split(":")[-1]`；
  - 调用 `init_distributed(rank_and_world_size=(rank, world_size))`；
  - 直接 `from multimodal_evals.ek_action_prediction.eval import main as ek_main`
    并 `ek_main(args_eval=params)`。
- 启动示例：
  ```bash
  python -m multimodal_evals.ek_action_prediction.main \
         --fname multimodal_evals/ek_action_prediction/ek_action_prediction.yaml \
         --devices cuda:0 cuda:1 cuda:2 cuda:3 \
         --debugmode False
  ```

#### 12.2.2 调试模式
- `--debugmode True` 时单进程运行，等价于 `process_main(rank=0, world_size=1, devices=["cuda:0"])`。
- 该路径必须能跑通完整的 train + eval，便于 Codex 自验。

### 12.3 错误日志
- 启动后 rank==0 打印一次完整 `cfg`（用 `pprint`），便于复现。
- `ckpt missing/unexpected keys` 打印前 20 条；其它 rank 静默。
