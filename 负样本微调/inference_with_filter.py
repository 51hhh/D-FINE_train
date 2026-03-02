"""推理时的后处理过滤规则，减少误检"""
import torch
import numpy as np

def filter_false_positives(boxes, scores, labels, img_width, img_height):
    """
    根据排球的物理特征过滤误检
    
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        scores: [N]
        labels: [N]
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        保留的索引
    """
    keep_indices = []
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # 规则1：排球近似圆形，长宽比应在 0.7-1.4 之间
        if aspect_ratio < 0.6 or aspect_ratio > 1.6:
            continue
        
        # 规则2：排球大小合理（相对图片的占比）
        relative_area = area / (img_width * img_height)
        if relative_area < 0.0001 or relative_area > 0.3:  # 0.01% ~ 30%
            continue
        
        # 规则3：边界框不能太靠边缘（可能是噪声）
        margin = 5
        if x1 < margin or y1 < margin or x2 > img_width - margin or y2 > img_height - margin:
            # 置信度要求更高
            if score < 0.6:
                continue
        
        # 规则4：置信度基本阈值
        if score < 0.3:
            continue
        
        keep_indices.append(i)
    
    return keep_indices


# 使用示例
def inference_with_filtering(model, postprocessor, image_tensor, img_size):
    """推理并应用后处理过滤"""
    with torch.no_grad():
        outputs = model(image_tensor)
        results = postprocessor(outputs, torch.tensor([[img_size[1], img_size[0]]]))
    
    if len(results) > 0:
        result = results[0]
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        
        # 应用过滤
        keep = filter_false_positives(boxes, scores, labels, img_size[1], img_size[0])
        
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        }
    
    return {"boxes": [], "scores": [], "labels": []}
