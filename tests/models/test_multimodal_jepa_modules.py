import json

import torch
from PIL import Image

from src.datasets.multimodal_dataset import MultimodalDataset, multimodal_collate_fn
from src.losses.multimodal_loss import MultimodalLoss, prepare_sigreg_input
from src.models.multimodal_predictor import MultimodalPredictor
from src.models.projectors import MultimodalProjectors


def test_prepare_sigreg_input_shape():
    z_ctx = torch.randn(2, 5, 16)
    z_tgt = torch.randn(2, 3, 16)

    proj = prepare_sigreg_input(z_ctx, z_tgt)

    assert proj.shape == (8, 2, 16)


def test_multimodal_loss_backward():
    z_ctx = torch.randn(2, 5, 16, requires_grad=True)
    z_tgt = torch.randn(2, 3, 16, requires_grad=True)
    z_pred = torch.randn(2, 3, 16, requires_grad=True)
    loss_fn = MultimodalLoss(lambda_sigreg=0.1, sigreg_num_proj=8)

    loss = loss_fn(z_pred, z_tgt, z_ctx)["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert z_pred.grad is not None
    assert z_ctx.grad is not None
    assert z_tgt.grad is not None


def test_projectors_are_independent():
    projectors = MultimodalProjectors(video_dim=8, text_dim=10, hidden_dim=12, shared_dim=6)

    assert projectors.v_proj_ctx is not projectors.v_proj_tgt
    assert projectors.l_proj_ctx is not projectors.l_proj_tgt
    assert projectors.v_proj_ctx(torch.randn(2, 4, 8)).shape == (2, 4, 6)
    assert projectors.l_proj_tgt(torch.randn(2, 5, 10)).shape == (2, 5, 6)


def test_multimodal_predictor_four_quadrants():
    predictor = MultimodalPredictor(shared_dim=32, predictor_dim=32, depth=1, num_heads=4, img_size=32)
    cases = [
        ("V", "V", 64, 4),
        ("V", "L", 64, 77),
        ("L", "V", 77, 4),
        ("L", "L", 77, 77),
    ]

    for ctx_mod, tgt_mod, n_ctx, n_tgt in cases:
        z_ctx = torch.randn(2, n_ctx, 32)
        z_pred = predictor(z_ctx, ctx_mod, tgt_mod, n_tgt=n_tgt)
        assert z_pred.shape == (2, n_tgt, 32)


def test_multimodal_dataset_and_collate(tmp_path):
    frame_paths = []
    for idx in range(34):
        frame_path = tmp_path / f"frame_{idx:03d}.jpg"
        Image.new("RGB", (8, 8), color=(idx, 0, 0)).save(frame_path)
        frame_paths.append(str(frame_path))

    jsonl_path = tmp_path / "samples.jsonl"
    sample = {
        "clip_id": "clip-1",
        "action_narration": "open drawer",
        "text_state_context": "drawer closed",
        "text_state_target": "drawer open",
        "visual_frame_context_paths": frame_paths[:32],
        "visual_frame_target_paths": frame_paths[32:],
    }
    jsonl_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    dataset = MultimodalDataset(str(jsonl_path), img_size=16)
    item = dataset[0]
    batch = multimodal_collate_fn([item, item])

    assert item["video_ctx"].shape == (3, 32, 16, 16)
    assert item["video_tgt"].shape == (3, 2, 16, 16)
    assert batch["video_ctx"].shape == (2, 3, 32, 16, 16)
    assert batch["text_ctx"] == ["drawer closed", "drawer closed"]
