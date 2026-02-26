# 本地对比训练指南（D-FINE vs RT-DETR）

目标：在 x86 + NVIDIA 平台，先完成 D-FINE 与 RT-DETR 的本地训练与对比，为后续 Jetson AGX Orin 部署做准备。

当前数据目录（已检测到）：
- 图片：`coco/images`
- 标注：`coco/Annotations`（VOC XML）

下面流程全部使用 Python 脚本，不使用 PowerShell 脚本。

## 1. 安装依赖

先安装基础依赖：

```bash
pip install -r benchmark/requirements.txt
```

再按需安装模型依赖：

```bash
pip install -r third_party/D-FINE/requirements.txt
pip install -r third_party/RT-DETR/rtdetrv2_pytorch/requirements.txt
pip install ultralytics
```

## 2. 准备 COCO 训练标注（从 XML 转换）

先打开 [prepare_dataset.py](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/scripts/prepare_dataset.py) 修改脚本头部配置（如有需要）：
- `IMAGES_DIR`
- `XML_DIR`
- `OUTPUT_DIR`
- `TRAIN_RATIO / VAL_RATIO / TEST_RATIO`
- `SEED`

然后直接执行：

```bash
python benchmark/scripts/prepare_dataset.py
```

脚本会生成：
- `coco/converted/annotations/train.json`
- `coco/converted/annotations/val.json`
- `coco/converted/annotations/test.json`
- `coco/converted/splits/train.txt`
- `coco/converted/splits/val.txt`
- `coco/converted/splits/test.txt`
- `coco/converted/ultralytics_data.yaml`

## 3. 开始训练（统一脚本 + YAML 配置）

### 3.1 先选训练配置文件

在 `benchmark/configs/profiles/` 中选择一个：
- 主对比（预训练）：
  - [01_dfine_n_coco_pretrain.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/01_dfine_n_coco_pretrain.yaml)
  - [02_rtdetrv2_r18_coco_pretrain.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/02_rtdetrv2_r18_coco_pretrain.yaml)
  - [03_dfine_s_coco_pretrain.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/03_dfine_s_coco_pretrain.yaml)
  - [04_rtdetrv2_r34_coco_pretrain.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/04_rtdetrv2_r34_coco_pretrain.yaml)
- 从零训练对照：
  - [11_dfine_n_scratch.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/11_dfine_n_scratch.yaml)
  - [12_rtdetrv2_r18_scratch.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/12_rtdetrv2_r18_scratch.yaml)
  - [13_dfine_s_scratch.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/13_dfine_s_scratch.yaml)
  - [14_rtdetrv2_r34_scratch.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/14_rtdetrv2_r34_scratch.yaml)
- 扩展组（可选）：
  - [21_dfine_m_coco_pretrain.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/21_dfine_m_coco_pretrain.yaml)
  - [22_rtdetrv2_r50_coco_pretrain.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/22_rtdetrv2_r50_coco_pretrain.yaml)
  - [23_dfine_s_obj365_finetune.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/23_dfine_s_obj365_finetune.yaml)
  - [24_rtdetrv2_r18_obj365_finetune.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/profiles/24_rtdetrv2_r18_obj365_finetune.yaml)

数据配置统一在：
- [dataset.yaml](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/configs/dataset.yaml)

### 3.2 在统一脚本头部切换 PROFILE

打开 [train.py](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/scripts/train.py)，修改：
- `PROFILE_FILE`（指向你要跑的 profile yaml）
- `DATASET_FILE`（通常保持默认）

### 3.3 启动训练

```bash
python benchmark/scripts/train.py
```

## 4. 训练输出位置

每次训练会自动创建独立目录，不会混在一起：

- `benchmark/experiments/<output_group>/run_0001/`
- `benchmark/experiments/<output_group>/run_0002/`
- ...

每个 `run_xxxx` 内固定结构：
- `configs/`：本次训练实际使用的 YAML（含覆盖后的 dataset/train 配置）
- `logs/train.log`：完整训练日志
- `model_output/`：模型输出（checkpoint、events 等）
- `tensorboard.txt`：TensorBoard 启动命令
- `metadata.json`：本次训练元信息（命令、配置、时间）

TensorBoard 查看：

```bash
tensorboard --logdir benchmark/experiments --port 6006
```

## 5. GPU 评估/推理（官方成熟配置）

评估脚本会调用官方仓库的 `--test-only` 流程，配置来源是：
1. profile 中的 `base_config`（官方成熟配置）
2. dataset 覆盖配置（你的数据路径和类别数）

使用步骤：
1. 打开 [eval.py](c:/Users/Rick/Desktop/python/ObjectDetection/benchmark/scripts/eval.py)
2. 设置 `PROFILE_FILE`
3. 设置 `CHECKPOINT`
4. 运行：

```bash
python benchmark/scripts/eval.py
```

评估输出目录：
- `benchmark/evaluations/<output_group>/eval_0001/`
- `benchmark/evaluations/<output_group>/eval_0002/`

并且会自动使用 GPU：
- 环境变量 `CUDA_VISIBLE_DEVICES` 来自 profile 的 `device` 字段

## 5. 下一步（对比评测）

训练完成后，你可以继续用：
- `benchmark/metrics/metric_suite.py`

做三维对比：
1. 小目标召回率
2. 框稳定性（抖动）
3. 抗运动模糊能力
