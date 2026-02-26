# D-FINE 与 RT-DETR 训练执行手册（含配置映射）

本文是直接可执行的流程文档。你只需要切换 `benchmark/scripts/train.py` 顶部的 `PROFILE_FILE`，按顺序跑即可。

## 1. 先跑哪些配置（训练顺序）

按这个顺序执行：

1. 阶段A 主基线（预训练，必须）
2. 阶段B 从零训练对照（必须）
3. 阶段C 扩展模型（可选）
4. 阶段D 预训练策略对照（可选）

对应 profile（按顺序）：

1. `01_dfine_n_coco_pretrain.yaml`
2. `02_rtdetrv2_r18_coco_pretrain.yaml`
3. `03_dfine_s_coco_pretrain.yaml`
4. `04_rtdetrv2_r34_coco_pretrain.yaml`
5. `11_dfine_n_scratch.yaml`
6. `12_rtdetrv2_r18_scratch.yaml`
7. `13_dfine_s_scratch.yaml`
8. `14_rtdetrv2_r34_scratch.yaml`
9. `21_dfine_m_coco_pretrain.yaml`（可选）
10. `22_rtdetrv2_r50_coco_pretrain.yaml`（可选）
11. `23_dfine_s_obj365_finetune.yaml`（可选）
12. `24_rtdetrv2_r18_obj365_finetune.yaml`（可选）

## 2. 每一步怎么跑

每次只改一个地方：
- [train.py](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/scripts/train.py) 中 `PROFILE_FILE`

运行：

```bash
python benchmark/scripts/train.py
```

输出自动递增，不会覆盖：
- `benchmark/experiments/<output_group>/run_0001`
- `benchmark/experiments/<output_group>/run_0002`

## 3. 怎么对比效果（必须统一）

每个配置训练后都做同一套评估（同数据、同阈值、同脚本）：

1. 小目标召回
- `recall_small@0.5`

2. 稳定性
- `center_jerk_norm_mean`
- `area_log_std_mean`

3. 抗运动模糊
- `recall_drop_vs_clear`（`mild/heavy`）

4. 常规检测
- `mAP50-95`
- `mAP50`
- `Recall`

建议评分权重：
- 小目标召回 40%
- 稳定性 35%
- 抗模糊 25%

## 4. 评判与下一步修正方向

## 情况A：小目标召回不够
优先动作：
1. 从 A 组升级到 B 组（`N/R18 -> S/R34`）
2. 再升级到扩展组（`M/R50`）

## 情况B：框抖动严重
优先动作：
1. 保持模型规模不变，先比较预训练与从零训练结果（通常预训练更稳）
2. 若 D-FINE 抖动明显低于 RT-DETR，后续优先 D-FINE 路线

## 情况C：模糊场景掉点严重
优先动作：
1. 先看 B 组是否改善
2. 再尝试 D-FINE Objects365 预训练对照（`23_...`）

## 情况D：预训练收益不明显
动作：
1. 对比同模型 scratch vs pretrain（11/12/13/14 对 01/02/03/04）
2. 若差异 < 1-2% 且稳定性无提升，可简化为 scratch 路线

## 5. profile 文件与实验目的对照表

1. `01_dfine_n_coco_pretrain.yaml`  
用途：A组主基线，轻量 + 预训练

2. `02_rtdetrv2_r18_coco_pretrain.yaml`  
用途：A组主基线，轻量 + 预训练

3. `03_dfine_s_coco_pretrain.yaml`  
用途：B组主基线，中等规模 + 预训练

4. `04_rtdetrv2_r34_coco_pretrain.yaml`  
用途：B组主基线，中等规模 + 预训练

5. `11/12/13/14 *_scratch.yaml`  
用途：从零训练对照，验证预训练收益

6. `21/22 *_pretrain.yaml`  
用途：扩展规模验证上限

7. `23/24 *_obj365_finetune.yaml`  
用途：预训练策略对照（COCO vs Objects365）

## 6. 推荐最小实验集（先跑这 8 个）

1. `01` vs `02`（轻量预训练对比）
2. `03` vs `04`（中等预训练对比）
3. `11` vs `01`（D-FINE-N 预训练收益）
4. `12` vs `02`（RT-DETR-R18 预训练收益）
5. `13` vs `03`（D-FINE-S 预训练收益）
6. `14` vs `04`（RT-DETR-R34 预训练收益）

跑完这 8 个就能决定主方向。

## 7. 参考资料

- D-FINE 官方仓库  
  https://github.com/Peterande/D-FINE
- RT-DETR 官方仓库  
  https://github.com/lyuwenyu/RT-DETR
- RT-DETRv2 PyTorch README  
  https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/README.md
