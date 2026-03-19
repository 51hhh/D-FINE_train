import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.solver.det_engine import build_synthetic_negative_batch, compute_synthetic_negative_weight
from src.solver.det_solver import _get_bbox_ap50_95
from src.zoo.dfine.dfine_criterion import DFINECriterion


class DummyMatcher:
    def __call__(self, outputs, targets):
        indices = []
        for target in targets:
            num_gt = len(target["labels"])
            src = torch.arange(num_gt, dtype=torch.int64)
            tgt = torch.arange(num_gt, dtype=torch.int64)
            indices.append((src, tgt))
        return {"indices": indices}


def test_compute_synthetic_negative_weight():
    assert compute_synthetic_negative_weight(0.0) == 0.0
    assert compute_synthetic_negative_weight(0.5) == 1.0
    assert compute_synthetic_negative_weight(1.0) == 4.0
    assert compute_synthetic_negative_weight(0.5, coeff=0.25) == 0.25
    assert compute_synthetic_negative_weight(1.0, coeff=0.5) == 2.0
    assert compute_synthetic_negative_weight(0.5, coeff=-1.0) == 0.0


def test_get_bbox_ap50_95_handles_missing_stats():
    assert _get_bbox_ap50_95({}) == 0.0
    assert _get_bbox_ap50_95({"coco_eval_bbox": []}) == 0.0
    assert _get_bbox_ap50_95({"coco_eval_bbox": [0.83, 0.9]}) == 0.83


def test_build_synthetic_negative_batch_creates_empty_target_images():
    torch.manual_seed(0)
    samples = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
            "orig_size": torch.tensor([32, 32]),
            "image_id": torch.tensor([1]),
        }
    ]
    cfg = {
        "synthetic_neg_fill_background_prob": 0.0,
        "synthetic_neg_fill_noise_prob": 1.0,
        "synthetic_neg_fill_solid_prob": 0.0,
        "synthetic_neg_expand_ratio": 1.0,
        "synthetic_neg_max_trials": 20,
    }

    neg_samples, neg_targets, stats = build_synthetic_negative_batch(samples, targets, cfg)
    assert neg_samples is not None
    assert neg_samples.shape == samples.shape
    assert len(neg_targets) == 1
    assert neg_targets[0]["boxes"].shape == (0, 4)
    assert neg_targets[0]["labels"].shape == (0,)
    assert stats["generated_images"] == 1.0
    assert stats["filled_boxes"] == 1.0
    assert not torch.allclose(neg_samples, samples)


def test_loss_synthetic_negative_uses_topk_softplus_and_weight():
    criterion = DFINECriterion(
        matcher=DummyMatcher(),
        weight_dict={"loss_vfl": 1, "loss_bbox": 1, "loss_giou": 1, "loss_fgl": 1, "loss_ddf": 1},
        losses=["vfl", "boxes", "local"],
        num_classes=1,
    )
    outputs = {
        "pred_logits": torch.tensor([[[4.0], [1.0], [-2.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]], dtype=torch.float32),
    }
    loss_dict = criterion.loss_synthetic_negative(outputs, loss_weight=1.0, topk=2)
    expected = (F.softplus(torch.tensor(4.0)) + F.softplus(torch.tensor(1.0))) / 2
    assert math.isclose(loss_dict["loss_synth_neg"].item(), expected.item(), rel_tol=1e-5)
    assert criterion.latest_synth_neg_stats["topk_queries"] == 2.0


def test_loss_synthetic_negative_zero_weight_returns_zero():
    criterion = DFINECriterion(
        matcher=DummyMatcher(),
        weight_dict={"loss_vfl": 1, "loss_bbox": 1, "loss_giou": 1, "loss_fgl": 1, "loss_ddf": 1},
        losses=["vfl", "boxes", "local"],
        num_classes=1,
    )
    outputs = {
        "pred_logits": torch.tensor([[[4.0], [1.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]], dtype=torch.float32),
    }
    loss_dict = criterion.loss_synthetic_negative(outputs, loss_weight=0.0, topk=2)
    assert loss_dict["loss_synth_neg"].item() == 0.0


def test_loss_synthetic_negative_handles_zero_topk():
    criterion = DFINECriterion(
        matcher=DummyMatcher(),
        weight_dict={"loss_vfl": 1, "loss_bbox": 1, "loss_giou": 1, "loss_fgl": 1, "loss_ddf": 1},
        losses=["vfl", "boxes", "local"],
        num_classes=1,
    )
    outputs = {
        "pred_logits": torch.tensor([[[4.0], [1.0]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]], dtype=torch.float32),
    }
    loss_dict = criterion.loss_synthetic_negative(outputs, loss_weight=1.0, topk=0)
    assert loss_dict["loss_synth_neg"].item() == 0.0
    assert criterion.latest_synth_neg_stats["topk_queries"] == 0.0


def test_loss_synthetic_negative_multi_class_uses_max_class_logit():
    criterion = DFINECriterion(
        matcher=DummyMatcher(),
        weight_dict={"loss_vfl": 1, "loss_bbox": 1, "loss_giou": 1, "loss_fgl": 1, "loss_ddf": 1},
        losses=["vfl", "boxes", "local"],
        num_classes=2,
    )
    outputs = {
        "pred_logits": torch.tensor([[[0.5, 3.0], [2.5, -1.0], [0.1, 0.2]]], dtype=torch.float32),
        "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]], dtype=torch.float32),
    }
    loss_dict = criterion.loss_synthetic_negative(outputs, loss_weight=1.0, topk=2)
    expected = (F.softplus(torch.tensor(3.0)) + F.softplus(torch.tensor(2.5))) / 2
    assert math.isclose(loss_dict["loss_synth_neg"].item(), expected.item(), rel_tol=1e-5)
    assert math.isclose(criterion.latest_synth_neg_stats["max_logit"], 3.0, rel_tol=1e-5)
    assert math.isclose(criterion.latest_synth_neg_stats["mean_logit"], 1.9, rel_tol=1e-5)


def test_loss_synthetic_negative_keeps_zero_grad_path_when_topk_disabled():
    criterion = DFINECriterion(
        matcher=DummyMatcher(),
        weight_dict={"loss_vfl": 1, "loss_bbox": 1, "loss_giou": 1, "loss_fgl": 1, "loss_ddf": 1},
        losses=["vfl", "boxes", "local"],
        num_classes=1,
    )
    pred_logits = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32, requires_grad=True)
    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]], dtype=torch.float32),
    }
    loss = criterion.loss_synthetic_negative(outputs, loss_weight=1.0, topk=0)["loss_synth_neg"]
    loss.backward()
    assert pred_logits.grad is not None
    assert torch.allclose(pred_logits.grad, torch.zeros_like(pred_logits.grad))


if __name__ == "__main__":
    test_compute_synthetic_negative_weight()
    test_get_bbox_ap50_95_handles_missing_stats()
    test_build_synthetic_negative_batch_creates_empty_target_images()
    test_loss_synthetic_negative_uses_topk_softplus_and_weight()
    test_loss_synthetic_negative_zero_weight_returns_zero()
    test_loss_synthetic_negative_handles_zero_topk()
    test_loss_synthetic_negative_multi_class_uses_max_class_logit()
    test_loss_synthetic_negative_keeps_zero_grad_path_when_topk_disabled()
    print("ok")
