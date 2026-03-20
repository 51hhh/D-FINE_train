"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from ..zoo.dfine.box_ops import box_cxcywh_to_xyxy

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


class _DistributedMemoryError(RuntimeError):
    pass


class _DistributedStepAbort(RuntimeError):
    pass


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
        xyxy = box_cxcywh_to_xyxy(boxes.to(dtype=synth.dtype)).clamp(0.0, 1.0)
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


def _get_oom_runtime_state(state, full_batch_size: int, current_size: int):
    if state is None:
        state = {}
    state.setdefault("micro_batch_size", max(1, int(full_batch_size)))
    state.setdefault("runtime_max_size", max(1, int(current_size)))
    state.setdefault("stable_steps", 0)
    state.setdefault("consecutive_oom", 0)
    state.setdefault("oom_drop_steps", 0)
    state.setdefault("last_peak_allocated_mb", 0.0)
    state.setdefault("last_peak_ratio", 0.0)
    state.setdefault("workload_recoveries", 0)
    state.setdefault("recovery_window", 3)
    state.setdefault("recovery_peak_ratio_threshold", 0.85)
    state["micro_batch_size"] = max(1, int(state["micro_batch_size"]))
    state["runtime_max_size"] = max(1, int(state["runtime_max_size"]))
    return state


def _is_memory_error(exc: Exception) -> bool:
    if isinstance(exc, (_DistributedMemoryError, torch.cuda.OutOfMemoryError)):
        return True
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cublas_status_internal_error" in message
        or "cublas_status_alloc_failed" in message
    )


def _iter_micro_batches(samples: torch.Tensor, targets, micro_batch_size: int) -> Iterator[Tuple[torch.Tensor, List[Dict]]]:
    batch_size = len(targets)
    micro_batch_size = max(1, min(int(micro_batch_size), batch_size))
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        yield samples[start:end], targets[start:end]


def _move_targets_to_device(targets, device: torch.device):
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]


def _move_batch_to_device(samples: torch.Tensor, targets, device: torch.device):
    return samples.to(device), _move_targets_to_device(targets, device)


def _align_size_down(value: int, multiple: int = 32) -> int:
    value = max(1, int(value))
    if value < multiple:
        return value
    return max(multiple, (value // multiple) * multiple)


def _align_size_up(value: int, multiple: int = 32) -> int:
    value = max(1, int(value))
    if value < multiple:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _apply_runtime_size_cap(samples: torch.Tensor, targets, runtime_max_size: int):
    runtime_max_size = max(1, int(runtime_max_size))
    height, width = samples.shape[-2:]
    longest_side = max(height, width)
    if longest_side <= runtime_max_size:
        return samples, targets

    scale = float(runtime_max_size) / float(longest_side)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    if runtime_max_size >= 32:
        if new_height >= 32:
            new_height = min(runtime_max_size, _align_size_down(new_height))
        if new_width >= 32:
            new_width = min(runtime_max_size, _align_size_down(new_width))
    if new_height == height and new_width == width:
        return samples, targets

    resized_samples = F.interpolate(samples, size=(new_height, new_width), mode="bilinear", align_corners=False)
    resized_targets = []
    for target in targets:
        resized_target = dict(target)
        size_tensor = resized_target.get("size")
        if isinstance(size_tensor, torch.Tensor):
            resized_target["size"] = size_tensor.new_tensor([new_height, new_width])
        resized_targets.append(resized_target)
    return resized_samples, resized_targets


def _shrink_workload(state, full_batch_size: int, current_size: int) -> str:
    current_micro_batch_size = max(1, min(int(state["micro_batch_size"]), int(full_batch_size)))
    if current_micro_batch_size > 1:
        state["micro_batch_size"] = max(1, current_micro_batch_size // 2)
        return "micro_batch_size"

    current_cap = max(1, int(state.get("runtime_max_size", current_size)))
    if current_cap > 32:
        state["runtime_max_size"] = _align_size_down(current_cap - 32)
    else:
        state["runtime_max_size"] = max(1, current_cap - 1)
    return "runtime_max_size"


def _recover_workload(state, full_batch_size: int, target_size: int) -> bool:
    if int(state["stable_steps"]) < int(state["recovery_window"]):
        return False
    if float(state["last_peak_ratio"]) >= float(state["recovery_peak_ratio_threshold"]):
        return False

    target_size = max(1, int(target_size))
    current_cap = max(1, int(state["runtime_max_size"]))
    if current_cap < target_size:
        next_cap = min(target_size, _align_size_up(current_cap + 32))
        if next_cap != current_cap:
            state["runtime_max_size"] = next_cap
            state["stable_steps"] = 0
            state["workload_recoveries"] += 1
            return True

    current_micro_batch_size = max(1, int(state["micro_batch_size"]))
    if current_micro_batch_size < int(full_batch_size):
        state["micro_batch_size"] = min(int(full_batch_size), max(current_micro_batch_size + 1, current_micro_batch_size * 2))
        state["stable_steps"] = 0
        state["workload_recoveries"] += 1
        return True

    return False


def _collect_cuda_peak_stats(device: torch.device, total_memory: float = 0.0) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"peak_allocated_mb": 0.0, "peak_ratio": 0.0}

    peak_allocated = float(torch.cuda.max_memory_allocated(device))
    if total_memory <= 0:
        total_memory = float(torch.cuda.get_device_properties(device).total_memory)
    peak_ratio = 0.0 if total_memory <= 0 else peak_allocated / total_memory
    return {
        "peak_allocated_mb": peak_allocated / (1024.0 * 1024.0),
        "peak_ratio": peak_ratio,
    }


def _all_reduce_grads_if_needed(model: torch.nn.Module):
    if not dist_utils.is_dist_available_and_initialized() or dist_utils.get_world_size() < 2:
        return

    world_size = dist_utils.get_world_size()
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def _sync_step_status(local_exception: Exception, phase_name: str):
    if not dist_utils.is_dist_available_and_initialized() or dist_utils.get_world_size() < 2:
        if local_exception is not None:
            raise local_exception
        return

    local_status = {
        "success": local_exception is None,
        "memory_error": _is_memory_error(local_exception) if local_exception is not None else False,
    }
    gathered_status = dist_utils.all_gather(local_status)
    all_success = all(item["success"] for item in gathered_status)
    any_memory_error = any(item["memory_error"] for item in gathered_status)

    if local_exception is not None:
        raise local_exception
    if not all_success:
        if any_memory_error:
            raise _DistributedMemoryError(f"distributed memory error during {phase_name}")
        raise _DistributedStepAbort(f"distributed step aborted during {phase_name}")


def _accumulate_loss_dict(accumulated_loss_dict, loss_dict):
    if accumulated_loss_dict is None:
        return {key: value.detach() for key, value in loss_dict.items()}
    for key, value in loss_dict.items():
        if key in accumulated_loss_dict:
            accumulated_loss_dict[key] = accumulated_loss_dict[key] + value.detach()
        else:
            accumulated_loss_dict[key] = value.detach()
    return accumulated_loss_dict


def _dump_nan_state(model: torch.nn.Module):
    dist_utils.save_on_master({"model": dist_utils.de_parallel(model).state_dict()}, "./NaN.pth")


def _update_runtime_metrics(metric_logger: MetricLogger, optimizer, runtime_state):
    metric_logger.update(
        lr=optimizer.param_groups[0]["lr"],
        oom_drop_steps=float(runtime_state["oom_drop_steps"]),
        oom_consecutive=float(runtime_state["consecutive_oom"]),
        train_micro_batch_size=float(runtime_state["micro_batch_size"]),
        train_runtime_max_size=float(runtime_state["runtime_max_size"]),
        train_peak_mem_mb=float(runtime_state["last_peak_allocated_mb"]),
        train_peak_mem_ratio=float(runtime_state["last_peak_ratio"]),
        train_workload_recoveries=float(runtime_state["workload_recoveries"]),
    )


def _log_runtime_metrics(writer: SummaryWriter, runtime_state, global_step: int):
    writer.add_scalar("Train/oom_drop_steps", float(runtime_state["oom_drop_steps"]), global_step)
    writer.add_scalar("Train/oom_consecutive", float(runtime_state["consecutive_oom"]), global_step)
    writer.add_scalar("Train/train_micro_batch_size", float(runtime_state["micro_batch_size"]), global_step)
    writer.add_scalar("Train/train_runtime_max_size", float(runtime_state["runtime_max_size"]), global_step)
    writer.add_scalar("Train/train_peak_mem_mb", float(runtime_state["last_peak_allocated_mb"]), global_step)
    writer.add_scalar("Train/train_peak_mem_ratio", float(runtime_state["last_peak_ratio"]), global_step)
    writer.add_scalar("Train/train_workload_recoveries", float(runtime_state["workload_recoveries"]), global_step)


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
    runtime_state = None

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    synth_cfg = kwargs.get("yaml_cfg", {})
    prev_ap = kwargs.get("prev_ap", 0.0)
    synth_weight_coeff = float(synth_cfg.get("synthetic_neg_weight_coeff", 1.0))
    synth_topk = int(synth_cfg.get("synthetic_neg_topk", 5))
    is_distributed = dist_utils.is_dist_available_and_initialized() and dist_utils.get_world_size() > 1
    can_defer_grad_sync = is_distributed and hasattr(model, "no_sync")
    cuda_total_memory = 0.0
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_total_memory = float(torch.cuda.get_device_properties(device).total_memory)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        full_batch_size = len(targets)
        target_runtime_size = int(max(samples.shape[-2:]))
        runtime_state = _get_oom_runtime_state(runtime_state, full_batch_size, target_runtime_size)
        effective_micro_batch_size = min(int(runtime_state["micro_batch_size"]), full_batch_size)
        synth_weight = compute_synthetic_negative_weight(prev_ap, coeff=synth_weight_coeff)
        synth_stats = {"generated_images": 0.0, "filled_boxes": 0.0}
        accumulated_loss_dict = None
        current_runtime_size = target_runtime_size

        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)

        micro_batches = None
        micro_samples_cpu = None
        micro_targets_cpu = None
        micro_samples = None
        micro_targets = None
        outputs = None
        synth_outputs = None
        synth_samples = None
        synth_targets = None
        loss_dict = None
        loss = None
        try:
            num_micro_batches = math.ceil(full_batch_size / effective_micro_batch_size)
            micro_batches = _iter_micro_batches(samples, targets, effective_micro_batch_size)
            for micro_index, (micro_samples_cpu, micro_targets_cpu) in enumerate(micro_batches):
                micro_samples = None
                micro_targets = None
                outputs = None
                synth_outputs = None
                synth_samples = None
                synth_targets = None
                loss_dict = None
                loss = None
                micro_samples_cpu, micro_targets_cpu = _apply_runtime_size_cap(
                    micro_samples_cpu,
                    micro_targets_cpu,
                    runtime_state["runtime_max_size"],
                )
                current_runtime_size = int(max(micro_samples_cpu.shape[-2:]))
                transfer_exception = None
                try:
                    micro_samples, micro_targets = _move_batch_to_device(micro_samples_cpu, micro_targets_cpu, device)
                except Exception as exc:
                    transfer_exception = exc
                _sync_step_status(transfer_exception, "device_transfer")
                micro_scale = float(len(micro_targets)) / float(full_batch_size)
                is_last_micro_batch = micro_index + 1 == num_micro_batches
                sync_context = model.no_sync() if can_defer_grad_sync and not is_last_micro_batch else nullcontext()
                with sync_context:
                    micro_synth_stats = {"generated_images": 0.0, "filled_boxes": 0.0}

                    forward_exception = None
                    try:
                        forward_context = (
                            torch.autocast(device_type=device.type, cache_enabled=True) if scaler is not None else nullcontext()
                        )
                        with forward_context:
                            outputs = model(micro_samples, targets=micro_targets)
                            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                                print(outputs["pred_boxes"])
                                _dump_nan_state(model)

                            if synth_weight > 0:
                                synth_samples, synth_targets, micro_synth_stats = build_synthetic_negative_batch(
                                    micro_samples, micro_targets, synth_cfg
                                )
                                if synth_samples is not None:
                                    synth_outputs = model(synth_samples, targets=synth_targets)
                    except Exception as exc:
                        forward_exception = exc
                    _sync_step_status(forward_exception, "forward")

                    for key, value in micro_synth_stats.items():
                        synth_stats[key] += float(value)

                    backward_exception = None
                    try:
                        loss_context = (
                            torch.autocast(device_type=device.type, enabled=False) if scaler is not None else nullcontext()
                        )
                        with loss_context:
                            loss_dict = criterion(outputs, micro_targets, **metas)
                            if synth_outputs is not None:
                                loss_dict.update(
                                    criterion.loss_synthetic_negative(
                                        synth_outputs,
                                        loss_weight=synth_weight,
                                        topk=synth_topk,
                                    )
                                )
                        loss_dict = {key: value * micro_scale for key, value in loss_dict.items()}
                        loss = sum(loss_dict.values())
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    except Exception as exc:
                        backward_exception = exc
                    _sync_step_status(backward_exception, "backward")

                accumulated_loss_dict = _accumulate_loss_dict(accumulated_loss_dict, loss_dict)

            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        except Exception as exc:
            peak_stats = _collect_cuda_peak_stats(device, total_memory=cuda_total_memory)
            runtime_state["last_peak_allocated_mb"] = peak_stats["peak_allocated_mb"]
            runtime_state["last_peak_ratio"] = peak_stats["peak_ratio"]

            if _is_memory_error(exc):
                runtime_state["stable_steps"] = 0
                runtime_state["consecutive_oom"] += 1
                runtime_state["oom_drop_steps"] += 1
                _shrink_workload(runtime_state, full_batch_size, current_runtime_size)
                optimizer.zero_grad(set_to_none=True)
                del exc
                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _update_runtime_metrics(metric_logger, optimizer, runtime_state)
                if writer and dist_utils.is_main_process() and global_step % 10 == 0:
                    _log_runtime_metrics(writer, runtime_state, global_step)
                continue
            raise

        peak_stats = _collect_cuda_peak_stats(device, total_memory=cuda_total_memory)
        runtime_state["last_peak_allocated_mb"] = peak_stats["peak_allocated_mb"]
        runtime_state["last_peak_ratio"] = peak_stats["peak_ratio"]
        runtime_state["consecutive_oom"] = 0
        runtime_state["stable_steps"] += 1
        _recover_workload(runtime_state, full_batch_size, target_runtime_size)

        loss_dict_reduced = dist_utils.reduce_dict(accumulated_loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        _update_runtime_metrics(metric_logger, optimizer, runtime_state)

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
            _log_runtime_metrics(writer, runtime_state, global_step)

    if use_wandb:
        wandb.log(
            {
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "train/loss": float(np.mean(losses)) if losses else 0.0,
            }
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if runtime_state is not None:
        stats.update(
            {
                "oom_drop_steps": float(runtime_state["oom_drop_steps"]),
                "oom_consecutive": float(runtime_state["consecutive_oom"]),
                "train_micro_batch_size": float(runtime_state["micro_batch_size"]),
                "train_runtime_max_size": float(runtime_state["runtime_max_size"]),
                "train_peak_mem_mb": float(runtime_state["last_peak_allocated_mb"]),
                "train_peak_mem_ratio": float(runtime_state["last_peak_ratio"]),
                "train_workload_recoveries": float(runtime_state["workload_recoveries"]),
            }
        )
    return stats


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
