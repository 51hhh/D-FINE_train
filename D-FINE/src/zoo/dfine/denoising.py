"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh
from .utils import inverse_sigmoid


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    num_neg_random=0,
    neg_iou_threshold=0.3,
):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        dn_meta = {"dn_positive_idx": None, "dn_num_group": 0, "dn_num_split": [0, num_queries]}
        return None, None, None, dn_meta

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    # --- Method D: random background DN queries ---
    num_denoising_orig = num_denoising
    if num_neg_random > 0:
        rand_cx = torch.rand(bs, num_neg_random, 1, device=device)
        rand_cy = torch.rand(bs, num_neg_random, 1, device=device)
        rand_w = torch.rand(bs, num_neg_random, 1, device=device) * 0.23 + 0.02
        rand_h = torch.rand(bs, num_neg_random, 1, device=device) * 0.23 + 0.02
        rand_boxes = torch.cat([rand_cx, rand_cy, rand_w, rand_h], dim=-1)

        # Filter boxes overlapping with GT
        for b in range(bs):
            if num_gts[b] > 0:
                gt_xyxy = box_cxcywh_to_xyxy(targets[b]["boxes"])
                rand_xyxy = box_cxcywh_to_xyxy(rand_boxes[b])
                iou_matrix, _ = box_iou(rand_xyxy, gt_xyxy)
                max_iou = iou_matrix.max(dim=1)[0]
                rand_boxes[b][max_iou >= neg_iou_threshold] = 0

        rand_labels = torch.full([bs, num_neg_random], num_classes, dtype=torch.int32, device=device)
        rand_logits = class_embed(rand_labels)
        rand_bbox_unact = inverse_sigmoid(rand_boxes.clamp(min=1e-6, max=1 - 1e-6))

        input_query_logits = torch.cat([input_query_logits, rand_logits], dim=1)
        input_query_bbox_unact = torch.cat([input_query_bbox_unact, rand_bbox_unact], dim=1)
        num_denoising = num_denoising + num_neg_random

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising_orig,
            ] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[
                max_gt_num * 2 * i : max_gt_num * 2 * (i + 1),
                max_gt_num * 2 * (i + 1) : num_denoising_orig,
            ] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    # Random background queries: isolate from original DN, visible to each other
    if num_neg_random > 0:
        bg_start = num_denoising_orig
        bg_end = num_denoising
        attn_mask[bg_start:bg_end, :num_denoising_orig] = True
        attn_mask[:num_denoising_orig, bg_start:bg_end] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
        "num_neg_random": num_neg_random,
    }

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
