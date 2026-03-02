# FiftyOne 简化评估项目（独立于 D-FINE）

这个目录提供一个精简版流程：  
读取 COCO 标注 + D-FINE 模型推理 + FiftyOne 可视化评估（`ground_truth` vs `predictions`）。

## 1. 环境建议

建议在你的 `dfine` 环境中运行，不要使用 `base`。  
你之前遇到的错误是 NumPy 2.x 与已编译扩展不兼容，先执行：

```cmd
conda activate dfine
python -m pip install -U "numpy<2" matplotlib fiftyone tqdm
```

## 2. 运行示例（你的排球数据）

在项目根目录 `C:\Users\Rick\Desktop\python\ObjectDetection` 执行：

```cmd
conda activate dfine
python fo_project/run_fo_eval.py ^
  --repo-dir ./D-FINE ^
  -c ./config/volleyball_s_transfer_balance.yml ^
  -r ./D-FINE/output/dfine_hgnetv2_s_obj2custom/best_stg1.pth ^
  --img-root ./coco/images ^
  --ann-file ./coco/converted/annotations/val.json ^
  --dataset-name volleyball-val-s-b1 ^
  --device cuda:0 ^
  --input-size 640 ^
  --conf-thres 0.25 ^
  --overwrite
```

运行后会：
1. 创建 FiftyOne 数据集
2. 写入 `ground_truth` 和 `predictions`
3. 计算 mAP
4. 自动打开 FiftyOne App

## 3. 常用参数

- `--limit 100`：仅评估前 100 张，快速调试
- `--no-app`：只计算指标，不打开 UI
- `--input-size 512`：更快，适合边缘模型对比
- `--conf-thres 0.35`：提高置信度阈值，减少低分框

## 4. 对比不同模型建议

分别运行多次并使用不同 `--dataset-name`，例如：

- `volleyball-val-n-fast`
- `volleyball-val-s-balance`
- `volleyball-val-n-scratch`

然后在 FiftyOne 中对比误检/漏检分布；同时记录每次打印的 mAP。

## 5. 一体化评估工作台（推荐）

如果你要同时做：
- `evaluate_detections` 算 mAP
- 快速筛选 FP/FN/低置信度/低 IoU
- 同一数据集多模型对比（`pred_n` vs `pred_s`）
- 质量检查（脏标注、重复图、难例）

直接用：

```cmd
python fo_project/fo_workbench.py
```

脚本为顶部硬编码配置，按需改 `SETTINGS` 即可。

脚本会自动写入的关键字段（可在 FiftyOne 左侧筛选）：
- `has_fp_pred_*`：误检样本
- `has_fn_pred_*`：漏检样本
- `has_low_conf_pred_*`：低置信度样本
- `has_low_iou_pred_*`：低 IoU 样本
- `cmp_better_a / cmp_better_b`：模型对比结果
- `dirty_gt`：疑似脏标注
- `is_duplicate`：重复图
- `hard_sample`：难例
