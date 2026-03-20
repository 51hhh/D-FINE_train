"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def compute_synthetic_negative_weight(ap_value: float, coeff: float = 1.0) -> float:
    ap_value = max(0.0, min(1.0, float(ap_value)))
    coeff = max(0.0, float(coeff))
    return coeff * (2.0 * ap_value) ** 2


def _boxes_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
        ],
        dim=-1,
    )


def _sample_background_patch(image: torch.Tensor, mask: torch.Tensor, patch_hw, max_trials: int):
    _, height, width = image.shape
    patch_h, patch_w = int(patch_hw[0]), int(patch_hw[1])
    patch_h = max(1, min(patch_h, height))
    patch_w = max(1, min(patch_w, width))
    if patch_h >= height or patch_w >= width:
        return None
    for _ in range(max_trials):
        top = torch.randint(0, height - patch_h + 1, (1,), device=image.device).item()
        left = torch.randint(0, width - patch_w + 1, (1,), device=image.device).item()
        region = mask[top:top + patch_h, left:left + patch_w]
        if not region.any():
            return image[:, top:top + patch_h, left:left + patch_w].clone()
    return None


def build_synthetic_negative_batch(samples: torch.Tensor, targets, cfg):
    generated_images = []
    generated_targets = []
    generated_count = 0.0
    filled_boxes = 0.0
    background_prob = float(cfg.get("synthetic_neg_fill_background_prob", 0.5))
    noise_prob = float(cfg.get("synthetic_neg_fill_noise_prob", 0.25))
    solid_prob = float(cfg.get("synthetic_neg_fill_solid_prob", 0.25))
    expand_ratio = float(cfg.get("synthetic_neg_expand_ratio", 1.15))
    max_trials = int(cfg.get("synthetic_neg_max_trials", 30))
    mode_total = background_prob + noise_prob + solid_prob
    if mode_total <= 0:
        background_prob, noise_prob, solid_prob = 1.0, 0.0, 0.0
        mode_total = 1.0
    probs = torch.tensor([background_prob, noise_prob, solid_prob], dtype=torch.float32, device=samples.device)
    probs = probs / probs.sum()

    for sample, target in zip(samples, targets):
        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None or len(boxes) == 0:
            continue

        synth = sample.clone()
        _, height, width = synth.shape
        mask = torch.zeros((height, width), dtype=torch.bool, device=synth.device)
        xyxy = _boxes_cxcywh_to_xyxy(boxes.to(dtype=synth.dtype)).clamp(0.0, 1.0)
        expanded = xyxy.clone()
        centers = (expanded[:, :2] + expanded[:, 2:]) / 2
        half_sizes = (expanded[:, 2:] - expanded[:, :2]) * 0.5 * expand_ratio
        expanded[:, :2] = (centers - half_sizes).clamp(0.0, 1.0)
        expanded[:, 2:] = (centers + half_sizes).clamp(0.0, 1.0)

        changed = False
        for box in expanded:
            x1 = int(torch.floor(box[0] * width).item())
            y1 = int(torch.floor(box[1] * height).item())
            x2 = int(torch.ceil(box[2] * width).item())
            y2 = int(torch.ceil(box[3] * height).item())
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            mask[y1:y2, x1:x2] = True

        for box in expanded:
            x1 = int(torch.floor(box[0] * width).item())
            y1 = int(torch.floor(box[1] * height).item())
            x2 = int(torch.ceil(box[2] * width).item())
            y2 = int(torch.ceil(box[3] * height).item())
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            patch_h, patch_w = y2 - y1, x2 - x1
            mode_idx = torch.multinomial(probs, 1).item()
            if mode_idx == 0:
                fill_patch = _sample_background_patch(synth, mask, (patch_h, patch_w), max_trials)
                if fill_patch is None:
                    fill_patch = torch.rand((3, patch_h, patch_w), device=synth.device, dtype=synth.dtype)
            elif mode_idx == 1:
                fill_patch = torch.rand((3, patch_h, patch_w), device=synth.device, dtype=synth.dtype)
            else:
                color = torch.rand((3, 1, 1), device=synth.device, dtype=synth.dtype)
                fill_patch = color.expand(3, patch_h, patch_w)
            synth[:, y1:y2, x1:x2] = fill_patch
            changed = True
            filled_boxes += 1.0

        if not changed:
            continue

        neg_target = {}
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                neg_target[key] = value.clone()
            else:
                neg_target[key] = value
        neg_target["boxes"] = torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device)
        neg_target["labels"] = torch.zeros((0,), dtype=labels.dtype, device=labels.device)
        if "area" in neg_target:
            neg_target["area"] = torch.zeros((0,), dtype=neg_target["area"].dtype, device=neg_target["area"].device)
        if "iscrowd" in neg_target:
            neg_target["iscrowd"] = torch.zeros((0,), dtype=neg_target["iscrowd"].dtype, device=neg_target["iscrowd"].device)
        neg_target["synthetic_negative"] = True
        generated_images.append(synth)
        generated_targets.append(neg_target)
        generated_count += 1.0

    if not generated_images:
        return None, [], {"generated_images": 0.0, "filled_boxes": 0.0}

    return torch.stack(generated_images, dim=0), generated_targets, {
        "generated_images": generated_count,
        "filled_boxes": filled_boxes,
    }


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        synth_cfg = kwargs.get("yaml_cfg", {})
        prev_ap = kwargs.get("prev_ap", 0.0)
        synth_weight_coeff = float(synth_cfg.get("synthetic_neg_weight_coeff", 1.0))
        synth_weight = compute_synthetic_negative_weight(prev_ap, coeff=synth_weight_coeff)
        synth_samples, synth_targets, synth_stats = None, [], {"generated_images": 0.0, "filled_boxes": 0.0}
        if synth_weight > 0:
            synth_samples, synth_targets, synth_stats = build_synthetic_negative_batch(samples, targets, synth_cfg)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
                synth_outputs = None
                if synth_samples is not None and synth_weight > 0:
                    synth_outputs = model(synth_samples, targets=synth_targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
                if synth_outputs is not None:
                    loss_dict.update(
                        criterion.loss_synthetic_negative(
                            synth_outputs,
                            loss_weight=synth_weight,
                            topk=int(synth_cfg.get("synthetic_neg_topk", 5)),
                        )
                    )

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            if synth_samples is not None and synth_weight > 0:
                synth_outputs = model(synth_samples, targets=synth_targets)
                loss_dict.update(
                    criterion.loss_synthetic_negative(
                        synth_outputs,
                        loss_weight=synth_weight,
                        topk=int(synth_cfg.get("synthetic_neg_topk", 5)),
                    )
                )

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)
            writer.add_scalar("Train/synth_neg_dynamic_weight", float(synth_weight), global_step)
            for k, v in synth_stats.items():
                writer.add_scalar(f"Train/synth_neg_{k}", float(v), global_step)
            synth_neg_stats = getattr(criterion, "latest_synth_neg_stats", None)
            if synth_neg_stats:
                for k, v in synth_neg_stats.items():
                    writer.add_scalar(f"Train/synth_neg_{k}", float(v), global_step)

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(  # from model input size to original img size
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-1], samples[idx].shape[-2]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator


@torch.no_grad()
def evaluate_overactivation(
    model: torch.nn.Module,
    postprocessor,
    negative_img_dir: str,
    device,
    input_size: int = 640,
    conf_threshold: float = 0.3,
) -> Dict[str, float]:
    """在纯负样本图片上计算 Over-Activation 指标（FPPI / Clean Rate / Max Score）。
    仅在主进程调用，无需分布式同步。
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    neg_dir = Path(negative_img_dir)
    if not neg_dir.exists():
        print(f"[OA] negative_img_dir not found: {neg_dir}")
        return {}

    img_files = sorted(f for f in neg_dir.iterdir() if f.is_file() and f.suffix.lower() in img_extensions)
    if not img_files:
        print(f"[OA] No images found in {neg_dir}")
        return {}

    tfm = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
    model.eval()

    total_fp = 0
    clean_images = 0
    max_score = 0.0

    for img_path in img_files:
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            tensor = tfm(image).unsqueeze(0).to(device)
            orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)

            outputs = model(tensor)
            results = postprocessor(outputs, orig_size)

            # 兼容 list[dict] 和 tuple 两种输出格式
            if isinstance(results, (tuple, list)) and len(results) == 3 and torch.is_tensor(results[0]):
                scores = results[2][0].cpu().tolist()
            else:
                scores = results[0]["scores"].cpu().tolist()

            fp = sum(1 for s in scores if s >= conf_threshold)
            total_fp += fp
            if fp == 0:
                clean_images += 1
            if scores:
                max_score = max(max_score, max(scores))
        except Exception as e:
            print(f"[OA] skip {img_path.name}: {e}")

    n = len(img_files)
    return {
        "oa_fppi": round(total_fp / n, 4),
        "oa_clean_rate": round(clean_images / n, 4),
        "oa_max_score": round(max_score, 4),
    }
