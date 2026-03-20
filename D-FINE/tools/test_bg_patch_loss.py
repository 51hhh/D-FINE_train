import math
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.solver import det_engine
from src.solver.det_engine import build_synthetic_negative_batch, compute_synthetic_negative_weight, train_one_epoch
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


class DummyOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]
        self.step_calls = 0
        self.zero_grad_calls = []

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls.append(set_to_none)
        return None

    def step(self):
        self.step_calls += 1
        return None


class DummyScaledLoss:
    def __init__(self, loss):
        self.loss = loss

    def backward(self):
        self.loss.backward()


class DummyScaler:
    def __init__(self):
        self.step_calls = 0
        self.update_calls = 0
        self.unscale_calls = 0

    def scale(self, loss):
        return DummyScaledLoss(loss)

    def unscale_(self, optimizer):
        self.unscale_calls += 1
        return None

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1
        return None


class DummyAutocast:
    entered = []
    current_enabled = False

    def __init__(self, device_type=None, cache_enabled=None, enabled=True):
        self.enabled = enabled
        self.prev_enabled = None

    def __enter__(self):
        DummyAutocast.entered.append(self.enabled)
        self.prev_enabled = DummyAutocast.current_enabled
        DummyAutocast.current_enabled = self.enabled
        return self

    def __exit__(self, exc_type, exc, tb):
        DummyAutocast.current_enabled = self.prev_enabled
        return False


class DummySmoothedValue:
    def __init__(self, window_size=1, fmt="{value:.6f}"):
        self.total = 0.0
        self.count = 0
        self.global_avg = 0.0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n
        self.global_avg = self.total / self.count


class DummyMetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = {}

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, data_loader, print_freq, header):
        for batch in data_loader:
            yield batch

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if key not in self.meters:
                self.meters[key] = DummySmoothedValue()
            self.meters[key].update(float(value))

    def synchronize_between_processes(self):
        return None

    def __str__(self):
        return str({k: v.global_avg for k, v in self.meters.items()})


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, targets=None):
        self.calls.append(DummyAutocast.current_enabled)
        batch_size = samples.shape[0]
        pred_logits = self.param.view(1, 1, 1).expand(batch_size, 2, 1)
        pred_boxes = self.param.view(1, 1, 1).expand(batch_size, 2, 4)
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class FakeOOMError(RuntimeError):
    pass


class RecordingModel(nn.Module):
    def __init__(self, fail_indices=None, fail_predicate=None, failure_exc_factory=None):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))
        self.fail_indices = set(fail_indices or [])
        self.fail_predicate = fail_predicate
        self.failure_exc_factory = failure_exc_factory or (lambda call: FakeOOMError("fake oom"))
        self.calls = []

    def forward(self, samples, targets=None):
        call = {
            "index": len(self.calls),
            "phase": "synth" if targets and all(t.get("synthetic_negative", False) for t in targets) else "main",
            "batch_size": int(samples.shape[0]),
            "size": int(samples.shape[-1]),
            "autocast": DummyAutocast.current_enabled,
        }
        self.calls.append(call)
        should_fail = call["index"] in self.fail_indices
        if self.fail_predicate is not None:
            should_fail = should_fail or bool(self.fail_predicate(call))
        if should_fail:
            raise self.failure_exc_factory(call)

        batch_size = samples.shape[0]
        pred_logits = self.param.view(1, 1, 1).expand(batch_size, 2, 1)
        pred_boxes = self.param.view(1, 1, 1).expand(batch_size, 2, 4)
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class DummyEMA:
    def __init__(self):
        self.update_calls = 0

    def update(self, model):
        self.update_calls += 1


class DummyWarmup:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class DummyCriterionForTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_called = False
        self.latest_synth_neg_stats = {}
        self.loss_calls = 0
        self.synth_loss_calls = 0

    def train(self):
        self.train_called = True
        return self

    def __call__(self, outputs, targets, **metas):
        self.loss_calls += 1
        return {"loss_main": outputs["pred_logits"].sum() * 0.0}

    def loss_synthetic_negative(self, outputs, loss_weight=0.0, topk=5):
        self.synth_loss_calls += 1
        return {"loss_synth_neg": outputs["pred_logits"].sum() * 0.0}


def make_batch(batch_size, size=8):
    samples = torch.ones((batch_size, 3, size, size), dtype=torch.float32)
    targets = []
    for image_id in range(batch_size):
        targets.append(
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
                "orig_size": torch.tensor([size, size], dtype=torch.int64),
                "size": torch.tensor([size, size], dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
            }
        )
    return samples, targets


def run_train_one_epoch_for_test(model, criterion, data_loader, optimizer, **kwargs):
    build_synth = kwargs.pop("build_synth", None)
    use_dummy_autocast = kwargs.pop("use_dummy_autocast", False)

    patchers = [
        patch.object(det_engine, "MetricLogger", DummyMetricLogger),
        patch.object(det_engine, "SmoothedValue", DummySmoothedValue),
        patch.object(det_engine.dist_utils, "is_main_process", lambda: False),
    ]
    if build_synth is not None:
        patchers.append(patch.object(det_engine, "build_synthetic_negative_batch", build_synth))
    if use_dummy_autocast:
        DummyAutocast.entered = []
        DummyAutocast.current_enabled = False
        patchers.append(patch.object(det_engine.torch, "autocast", DummyAutocast))

    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        return train_one_epoch(
            model,
            criterion,
            data_loader,
            optimizer,
            torch.device("cpu"),
            epoch=kwargs.pop("epoch", 0),
            use_wandb=False,
            **kwargs,
        )


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


def test_train_one_epoch_drops_step_when_micro_step_ooms():
    model = RecordingModel(fail_indices={0, 2})
    criterion = DummyCriterionForTrain()
    optimizer = DummyOptimizer()
    ema = DummyEMA()
    warmup = DummyWarmup()
    data_loader = [make_batch(4, 8), make_batch(4, 8)]

    with patch.object(det_engine, "_is_memory_error", side_effect=lambda exc: isinstance(exc, FakeOOMError)):
        stats = run_train_one_epoch_for_test(
            model,
            criterion,
            data_loader,
            optimizer,
            ema=ema,
            lr_warmup_scheduler=warmup,
            yaml_cfg={"synthetic_neg_weight_coeff": 0.0, "synthetic_neg_topk": 5},
            prev_ap=0.0,
        )

    assert [call["batch_size"] for call in model.calls if call["phase"] == "main"] == [4, 2, 2]
    assert optimizer.step_calls == 0
    assert ema.update_calls == 0
    assert warmup.step_calls == 0
    assert stats["oom_drop_steps"] == 2.0


def test_train_one_epoch_shrinks_micro_batch_after_oom():
    model = RecordingModel(fail_indices={0})
    criterion = DummyCriterionForTrain()
    optimizer = DummyOptimizer()
    data_loader = [make_batch(4, 8), make_batch(4, 8)]

    with patch.object(det_engine, "_is_memory_error", side_effect=lambda exc: isinstance(exc, FakeOOMError)):
        stats = run_train_one_epoch_for_test(
            model,
            criterion,
            data_loader,
            optimizer,
            yaml_cfg={"synthetic_neg_weight_coeff": 0.0, "synthetic_neg_topk": 5},
            prev_ap=0.0,
        )

    assert [call["batch_size"] for call in model.calls if call["phase"] == "main"] == [4, 2, 2]
    assert optimizer.step_calls == 1
    assert stats["train_micro_batch_size"] == 2.0


def test_train_one_epoch_shrinks_runtime_size_when_micro_batch_is_one():
    model = RecordingModel(fail_predicate=lambda call: call["phase"] == "main" and call["size"] > 64)
    criterion = DummyCriterionForTrain()
    optimizer = DummyOptimizer()
    data_loader = [make_batch(1, 96), make_batch(1, 96)]

    with patch.object(det_engine, "_is_memory_error", side_effect=lambda exc: isinstance(exc, FakeOOMError)):
        stats = run_train_one_epoch_for_test(
            model,
            criterion,
            data_loader,
            optimizer,
            yaml_cfg={"synthetic_neg_weight_coeff": 0.0, "synthetic_neg_topk": 5},
            prev_ap=0.0,
        )

    assert [call["size"] for call in model.calls if call["phase"] == "main"] == [96, 64]
    assert optimizer.step_calls == 1
    assert stats["train_runtime_max_size"] == 64.0


def test_train_one_epoch_keeps_synth_training_in_micro_batch_mode():
    model = RecordingModel(fail_indices={0})
    criterion = DummyCriterionForTrain()
    optimizer = DummyOptimizer()
    data_loader = [make_batch(4, 8), make_batch(4, 8)]
    build_calls = []

    def fake_build(samples, targets, cfg):
        build_calls.append(int(samples.shape[0]))
        synth_samples, synth_targets, _ = build_synthetic_negative_batch(samples, targets, cfg)
        return synth_samples, synth_targets, {"generated_images": float(samples.shape[0]), "filled_boxes": float(samples.shape[0])}

    with patch.object(det_engine, "_is_memory_error", side_effect=lambda exc: isinstance(exc, FakeOOMError)):
        stats = run_train_one_epoch_for_test(
            model,
            criterion,
            data_loader,
            optimizer,
            build_synth=fake_build,
            yaml_cfg={"synthetic_neg_weight_coeff": 1.0, "synthetic_neg_topk": 5},
            prev_ap=0.5,
        )

    synth_batch_sizes = [call["batch_size"] for call in model.calls if call["phase"] == "synth"]
    assert build_calls == [2, 2]
    assert synth_batch_sizes == [2, 2]
    assert criterion.synth_loss_calls == 2
    assert optimizer.step_calls == 1
    assert stats["train_micro_batch_size"] == 2.0


def test_train_one_epoch_restores_workload_after_stable_steps():
    model = RecordingModel(fail_indices={0})
    criterion = DummyCriterionForTrain()
    optimizer = DummyOptimizer()
    data_loader = [make_batch(4, 8) for _ in range(5)]

    with patch.object(det_engine, "_is_memory_error", side_effect=lambda exc: isinstance(exc, FakeOOMError)):
        stats = run_train_one_epoch_for_test(
            model,
            criterion,
            data_loader,
            optimizer,
            yaml_cfg={"synthetic_neg_weight_coeff": 0.0, "synthetic_neg_topk": 5},
            prev_ap=0.0,
        )

    main_batch_sizes = [call["batch_size"] for call in model.calls if call["phase"] == "main"]
    assert main_batch_sizes == [4, 2, 2, 2, 2, 2, 2, 4]
    assert optimizer.step_calls == 4
    assert stats["train_micro_batch_size"] == 4.0
    assert stats["train_workload_recoveries"] == 1.0


def test_train_one_epoch_skips_building_synth_batch_when_weight_is_zero():
    model = DummyModel()
    criterion = DummyCriterionForTrain()
    optimizer = DummyOptimizer()
    data_loader = [
        (
            torch.ones((1, 3, 8, 8), dtype=torch.float32),
            [{"boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32), "labels": torch.tensor([0], dtype=torch.int64)}],
        )
    ]

    called = {"count": 0}

    def fail_if_called(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("build_synthetic_negative_batch should not be called when weight is zero")

    run_train_one_epoch_for_test(
        model,
        criterion,
        data_loader,
        optimizer,
        build_synth=fail_if_called,
        yaml_cfg={"synthetic_neg_weight_coeff": 1.0, "synthetic_neg_topk": 5},
        prev_ap=0.0,
    )

    assert called["count"] == 0


def test_train_one_epoch_runs_synth_forward_inside_autocast_when_amp_enabled():
    model = DummyModel()
    criterion = DummyCriterionForTrain()
    optimizer = DummyOptimizer()
    data_loader = [
        (
            torch.ones((1, 3, 8, 8), dtype=torch.float32),
            [{"boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32), "labels": torch.tensor([0], dtype=torch.int64)}],
        )
    ]

    run_train_one_epoch_for_test(
        model,
        criterion,
        data_loader,
        optimizer,
        scaler=DummyScaler(),
        build_synth=lambda samples, targets, cfg: (
            samples.clone(),
            [{"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64), "synthetic_negative": True}],
            {"generated_images": 1.0, "filled_boxes": 1.0},
        ),
        use_dummy_autocast=True,
        yaml_cfg={"synthetic_neg_weight_coeff": 1.0, "synthetic_neg_topk": 5},
        prev_ap=0.5,
        epoch=1,
    )

    assert len(model.calls) == 2
    assert model.calls == [True, True]
    assert DummyAutocast.entered[:2] == [True, False]


if __name__ == "__main__":
    test_compute_synthetic_negative_weight()
    test_get_bbox_ap50_95_handles_missing_stats()
    test_build_synthetic_negative_batch_creates_empty_target_images()
    test_loss_synthetic_negative_uses_topk_softplus_and_weight()
    test_loss_synthetic_negative_zero_weight_returns_zero()
    test_loss_synthetic_negative_handles_zero_topk()
    test_loss_synthetic_negative_multi_class_uses_max_class_logit()
    test_loss_synthetic_negative_keeps_zero_grad_path_when_topk_disabled()
    test_train_one_epoch_drops_step_when_micro_step_ooms()
    test_train_one_epoch_shrinks_micro_batch_after_oom()
    test_train_one_epoch_shrinks_runtime_size_when_micro_batch_is_one()
    test_train_one_epoch_keeps_synth_training_in_micro_batch_mode()
    test_train_one_epoch_restores_workload_after_stable_steps()
    test_train_one_epoch_skips_building_synth_batch_when_weight_is_zero()
    test_train_one_epoch_runs_synth_forward_inside_autocast_when_amp_enabled()
    print("ok")
