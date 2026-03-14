# D-FINE 单卡训练（Windows CMD）

本项目数据为 COCO 格式自定义数据集（1 类：`volleyball`），路径如下：

- 图片目录：`C:/Users/Rick/Desktop/python/ObjectDetection/coco/images`
- 标注目录：`C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations`
- 训练/验证标注：`train.json`、`val.json`

注意：Windows 下单卡训练建议直接使用 `python train.py`，不使用 `torchrun`。

## 1. 环境准备（CMD）

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE

conda create -n dfine python=3.11.9 -y
conda activate dfine
pip install -r requirements.txt

python -c "import torch; print(torch.__version__)"
```

## 2. 迁移训练（推荐，S 模型）

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
mkdir weights

curl -L "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj365.pth" -o weights\dfine_s_obj365.pth

set IMG=C:/Users/Rick/Desktop/python/ObjectDetection/coco/images
set ANN=C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations

python train.py ^
  -c configs/dfine/custom/objects365/dfine_hgnetv2_s_obj2custom.yml ^
  --use-amp --seed 0 ^
  -t weights/dfine_s_obj365.pth ^
  -u num_classes=1 remap_mscoco_category=False ^
     train_dataloader.dataset.img_folder="%IMG%" ^
     train_dataloader.dataset.ann_file="%ANN%/train.json" ^
     val_dataloader.dataset.img_folder="%IMG%" ^
     val_dataloader.dataset.ann_file="%ANN%/val.json"
```

## 3. 从零开始训练（N 模型）

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE

set IMG=C:/Users/Rick/Desktop/python/ObjectDetection/coco/images
set ANN=C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations

python train.py ^
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml ^
  --use-amp --seed 0 ^
  -u HGNetv2.pretrained=False num_classes=1 remap_mscoco_category=False ^
     train_dataloader.dataset.img_folder="%IMG%" ^
     train_dataloader.dataset.ann_file="%ANN%/train.json" ^
     val_dataloader.dataset.img_folder="%IMG%" ^
     val_dataloader.dataset.ann_file="%ANN%/val.json"
```

## 4. 评估（test-only）

优先使用 `best_stg2.pth`，如果不存在可改成 `best_stg1.pth`。

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE

set IMG=C:/Users/Rick/Desktop/python/ObjectDetection/coco/images
set ANN=C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations

python train.py ^
  -c configs/dfine/custom/objects365/dfine_hgnetv2_s_obj2custom.yml ^
  --test-only -r output/dfine_hgnetv2_s_obj2custom/best_stg2.pth ^
  -u num_classes=1 remap_mscoco_category=False ^
     val_dataloader.dataset.img_folder="%IMG%" ^
     val_dataloader.dataset.ann_file="%ANN%/val.json"
```

## 5. 常见问题

1. 报错：`'torchrun' 不是内部或外部命令`
- 原因：环境中没有 `torchrun` 可执行文件或 PATH 未生效。
- 处理：单卡下直接使用本文档的 `python train.py`。

2. 报错：`use_libuv was requested but PyTorch was build without libuv support`
- 原因：`python -m torch.distributed.run` 在 Windows 环境触发 rendezvous/libuv 问题。
- 处理：单卡下直接使用 `python train.py`。

## 6. 训练现象说明（你看到的颜色异常/缩放）

`D-FINE/output/.../train_samples` 保存的是训练批次可视化，不是原图。  
训练阶段默认包含数据增强，因此出现颜色变化、裁剪、缩放是正常现象，主要来自：

- `RandomPhotometricDistort`（颜色扰动）
- `RandomZoomOut`
- `RandomIoUCrop`
- `Resize`

如果想看接近原图分布的结果，可查看 `val_samples` 或者单独做 `--test-only` 推理可视化。

## 7. 模型保存频率与 TensorBoard

### 7.1 模型多久保存一次

当前配置默认 `checkpoint_freq: 12`（来自 `configs/runtime.yml`）：

- 每轮都会更新 `last.pth`（在训练前半段）
- 每 12 轮额外保存 `checkpointxxxx.pth`
- 验证指标提升时保存 `best_stg1.pth` / `best_stg2.pth`

可在命令行改保存频率：

```cmd
python train.py ... -u checkpoint_freq=5
```

### 7.2 TensorBoard 是否可用

可以，默认日志目录在：

- `D-FINE/output/<实验目录>/summary`

启动方式（CMD）：

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
python -m tensorboard.main --logdir output --port 6006
```

浏览器打开 `http://localhost:6006`。  
如果缺少命令：`pip install tensorboard`

## 8. 多模型对比实验（单目标数据集）

目标：对比不同模型规模（N/S/M/L/X）、是否预训练、以及训练策略，结合评估指标选型。

### 8.1 建议实验维度

1. 模型规模：`N`、`S`（先做），资源允许再加 `M/L/X`
2. 初始化方式：
- 迁移学习（`-t weights/...pth`，推荐）
- 从零训练（`HGNetv2.pretrained=False`）
3. 固定训练条件：
- 相同数据划分（同一份 train/val）
- 相同 `seed`
- 尽量相同 epoch / batch / 输入尺寸

### 8.2 推荐评估指标（判断模型）

1. 精度：`AP50:95`（主指标）、`AP50`（单目标场景常看）
2. 速度：单张推理时延（ms）或 FPS
3. 资源：参数量、显存占用、GFLOPs（可用 benchmark 工具）
4. 稳定性：收敛速度、是否过拟合（看 train/val 曲线分叉）

实践上可按优先级：`AP50:95` > 推理速度 > 显存/模型大小。

### 8.3 为每个实验创建独立配置文件（推荐）

建议在 `D-FINE/configs/dfine/custom/` 下新建专用配置，例如：

- `volleyball_s_transfer.yml`
- `volleyball_n_scratch.yml`
- `volleyball_s_scratch.yml`

并确保每个配置有不同 `output_dir`，防止结果互相覆盖。

### 8.4 不改配置文件时的命令模板（快速跑实验）

S 模型迁移训练：

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
python train.py ^
  -c configs/dfine/custom/objects365/dfine_hgnetv2_s_obj2custom.yml ^
  --use-amp --seed 0 ^
  -t weights/dfine_s_obj365.pth ^
  --output-dir output/exp_s_transfer ^
  -u num_classes=1 remap_mscoco_category=False ^
     train_dataloader.dataset.img_folder="C:/Users/Rick/Desktop/python/ObjectDetection/coco/images" ^
     train_dataloader.dataset.ann_file="C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations/train.json" ^
     val_dataloader.dataset.img_folder="C:/Users/Rick/Desktop/python/ObjectDetection/coco/images" ^
     val_dataloader.dataset.ann_file="C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations/val.json"
```

N 模型从零训练：

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
python train.py ^
  -c configs/dfine/custom/dfine_hgnetv2_n_custom.yml ^
  --use-amp --seed 0 ^
  --output-dir output/exp_n_scratch ^
  -u HGNetv2.pretrained=False num_classes=1 remap_mscoco_category=False ^
     train_dataloader.dataset.img_folder="C:/Users/Rick/Desktop/python/ObjectDetection/coco/images" ^
     train_dataloader.dataset.ann_file="C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations/train.json" ^
     val_dataloader.dataset.img_folder="C:/Users/Rick/Desktop/python/ObjectDetection/coco/images" ^
     val_dataloader.dataset.ann_file="C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations/val.json"
```

评估模板（替换配置和权重路径）：

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
python train.py ^
  -c configs/dfine/custom/objects365/dfine_hgnetv2_s_obj2custom.yml ^
  --test-only -r output/exp_s_transfer/best_stg2.pth ^
  -u num_classes=1 remap_mscoco_category=False ^
     val_dataloader.dataset.img_folder="C:/Users/Rick/Desktop/python/ObjectDetection/coco/images" ^
     val_dataloader.dataset.ann_file="C:/Users/Rick/Desktop/python/ObjectDetection/coco/converted/annotations/val.json"
```

### 8.5 对比记录表（建议）

可在项目根目录维护一张实验表（CSV/Markdown）：

| 实验名 | 模型 | 预训练 | 配置文件 | 最优权重 | AP50:95 | AP50 | 参数量 | 推理时延(ms) | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---|
| exp_s_transfer | S | 是 | dfine_hgnetv2_s_obj2custom.yml | best_stg2.pth |  |  |  |  |  |
| exp_n_scratch | N | 否 | dfine_hgnetv2_n_custom.yml | best_stg2.pth |  |  |  |  |  |

## 9. 边缘部署推荐实验（已生成配置）

以下配置文件已放在项目根目录 `config/`，适合单类排球任务做模型对比：

- `volleyball_n_transfer_fast.yml`：N 模型，速度优先（512 输入）
- `volleyball_s_transfer_balance.yml`：S 模型，精度/速度平衡（640 输入）
- `volleyball_n_scratch_compare.yml`：N 模型从零训练，对照组（512 输入）

训练命令（CMD，在 `D-FINE` 目录执行）：

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE

python train.py -c ../config/volleyball_n_transfer_fast.yml --use-amp --seed 0
python train.py -c ../config/volleyball_s_transfer_balance.yml --use-amp --seed 0 -t weights/dfine_s_obj365.pth
python train.py -c ../config/volleyball_n_scratch_compare.yml --use-amp --seed 0
```

## 10. 如何查看训练指标并判断“收敛/完成”

### 10.1 训练过程中看什么

1. `train_loss`（训练损失）
- 初期快速下降，后期变缓是正常收敛特征。

2. `test_coco_eval_bbox`
- 下标 `0`：`AP50:95`（主指标）
- 下标 `1`：`AP50`（单目标任务常用）
- 下标 `2`：`AP75`

3. `best_stg1.pth / best_stg2.pth`
- 指标提升时自动更新，是你最终优先使用的权重。

### 10.2 TensorBoard 查看曲线

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
python -m tensorboard.main --logdir output --port 6006
```

浏览器打开：`http://localhost:6006`

重点看：
- `Loss/total` 是否趋于平稳
- `Test/coco_eval_bbox_0`（AP50:95）是否进入平台期
- `Test/coco_eval_bbox_1`（AP50）是否进入平台期

### 10.3 从 log.txt 快速读取最新/最佳指标

查看最新 epoch 指标：

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
python -c "import json,pathlib; p=pathlib.Path('output/exp_s_transfer_balance/log.txt'); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]; r=rows[-1]; print('epoch=',r['epoch'],'AP50:95=',r['test_coco_eval_bbox'][0],'AP50=',r['test_coco_eval_bbox'][1],'train_loss=',r['train_loss'])"
```

查看历史最佳 AP50:95：

```cmd
cd /d C:\Users\Rick\Desktop\python\ObjectDetection\D-FINE
python -c "import json,pathlib; p=pathlib.Path('output/exp_s_transfer_balance/log.txt'); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]; b=max(rows,key=lambda r:r['test_coco_eval_bbox'][0]); print('best_epoch=',b['epoch'],'best_AP50:95=',b['test_coco_eval_bbox'][0],'AP50=',b['test_coco_eval_bbox'][1])"
```

### 10.4 怎样算“收敛”

可用这个实操标准（经验阈值）：

1. 连续 `8~12` 个 epoch，`AP50:95` 增益很小（如 `< 0.002`）
2. `train_loss` 波动变小，整体不再明显下降
3. 新的 `best_stg2.pth` 长时间不再刷新

满足以上 2~3 条，通常可判定已收敛。

### 10.5 怎样算“训练完成”

1. 到达配置中的 `epochs` 上限，控制台出现 `Training time ...`
2. 输出目录中存在 `best_stg2.pth`（或 `best_stg1.pth`）和 `log.txt`
3. 用 `--test-only` 跑一次验证，确认最终指标并记录到实验表



## 快速开始
cd D-FINE

python train.py -c ../config/volleyball_s_transfer_balance.yml --use-amp --seed 0 -t weights/dfine_s_obj365.pth
python train.py -c ../config/volleyball_s_transfer.yml --use-amp --seed 0 -r output/exp_s_transfer/last.pth

+ -c 接config配置文件路径
 + -t 从预训练模型路径迁移tuning
 + -r 从last模型开始继续训练resume

---

## 11. Over-Activation 误检抑制：新增参数与 TensorBoard 指标

### 11.1 背景

D-FINE 等 DETR 系列模型在单类微调时容易出现 **Query Over-Activation** 问题：预训练的 300 条 query 在单类任务下退化为"是否有目标"的二元分类器，导致对任何物体都以高置信度误检。

本项目通过以下机制抑制该问题：
- **Method A**：训练集中加入空标注负样本图片
- **Method D**：DN 随机负框（每张图注入 50-100 个随机背景框）
- **bg_loss_weight**：放大背景查询的 VFL loss 权重

---

### 11.2 配置方式（YAML）

在实验配置文件中添加以下参数（不声明则默认关闭，对训练零影响）：

```yaml
# 背景 loss 权重（默认 1.0，建议 2.0）
DFINECriterion:
  bg_loss_weight: 2.0

# 训练验证时自动评估 Over-Activation 指标
eval_overactivation: True
negative_img_dir: "../coco/images/negative_samples"  # 负样本图片目录
oa_conf_threshold: 0.3                               # 评估阈值，默认 0.3
```

参考配置文件：`config/volleyball_s_obj2coco_neg_d_bg2.yml`

---

### 11.3 TensorBoard 指标说明

开启 `eval_overactivation: True` 后，每轮验证后会在 TensorBoard 中出现以下三个指标：

#### `Test/oa_fppi` — False Positives Per Image

**每张负样本图的平均误检数**（置信度 >= `oa_conf_threshold`）

| 值 | 含义 |
|---|---|
| 0.04 | 100 张背景图共 4 个误检，几乎无误检（生产级） |
| 2.15 | 100 张背景图共 215 个误检，严重过激活 |

- 值域：0 → ∞
- 理想值：< 0.05
- 趋势期望：随训练**持续下降**

#### `Test/oa_clean_rate` — Clean Image Rate

**完全无误检的负样本图片比例**

| 值 | 含义 |
|---|---|
| 0.96 | 100 张中 96 张完全干净，4 张有误检 |
| 0.00 | 所有图片都有至少一个误检（最差情况） |

- 值域：0.0 → 1.0
- 理想值：> 0.95
- 趋势期望：随训练**持续上升**

#### `Test/oa_max_score` — Max Detection Score

**所有负样本图中最高的单个误检置信度**

| 值 | 含义 |
|---|---|
| 0.417 | 最严重误检置信度 41.7%，用 0.5 阈值可完全过滤（安全） |
| 0.930 | 最严重误检置信度 93%，模型对误检极度自信（危险） |

- 值域：0.0 → 1.0
- 理想值：< 0.5
- 趋势期望：随训练**持续下降**

---

### 11.4 部署安全阈值参考

```
oa_fppi       < 0.05  → 生产级
oa_clean_rate > 0.95  → 生产级
oa_max_score  < 0.50  → 0.5 推理阈值可完全过滤误检
```

### 11.5 各模型指标对比（实验记录）

| 模型 | 预训练 | 负样本 | Method D | FPPI@0.3 | Clean@0.3 | Max FP |
|------|--------|--------|---------|----------|-----------|--------|
| transfer_aug_2000 | obj365 | 否 | 否 | 2.149 | 0% | 0.930 |
| transfer_objtococo_2000 | obj2coco | 否 | 否 | 1.109 | 27.7% | 0.938 |
| finetune_neg_aug | obj365 | 是 | 否 | 2.356 | 0% | 0.885 |
| finetune_neg_objtococo | obj2coco | 是 | 否 | 0.931 | 36.6% | 0.925 |
| **neg_d** | obj2coco | 是 | 是 | **0.040** | **96%** | **0.417** |
| neg_d_q100 | obj2coco | 是 | 是(q=100) | 0.050 | 99% | 0.698 |
