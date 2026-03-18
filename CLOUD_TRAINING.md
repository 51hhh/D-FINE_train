# GradMotion 云训练使用指南

## 容器启动问题解决方案

### 问题根因
平台 Git checkout 目录结构不可控，导致相对路径失效。

### 解决方案
在 `startScript` 中手动 `git clone` 到固定位置 `/workspace/dfine`。

---

## 推荐配置

### 1. Smoke Test（必须先做）

**文件**: `task_smoke_test_final.json`

**用途**: 验证 Git clone、数据路径、Python 环境

**创建并运行**:
```bash
gm --base-url "https://spaces.gradmotion.com/dev-api" task create --file task_smoke_test_final.json
# 获取返回的 taskId
gm --base-url "https://spaces.gradmotion.com/dev-api" task run --task-id "TASK_xxx"
gm --base-url "https://spaces.gradmotion.com/dev-api" task logs --task-id "TASK_xxx" --follow
```

**预期输出**:
```
OK: cloud_train.sh
OK: D-FINE/train.py
OK: config/volleyball_s_obj2coco_d_bg2.yml
OK: /personal/personal/images.zip
torch=2.7.0
SMOKE_TEST_SUCCESS
```

---

### 2. 正式训练（no-A 主配置）

**文件**: `task_cloud_train_no_A.json`

**配置**: `config/volleyball_s_obj2coco_d_bg2.yml`
- Method D: `num_neg_random=100`
- `bg_loss_weight=2.0`
- **不使用** Method A（无 `train_with_negatives.json`）
- OA 评估: `eval_overactivation=True`

**创建并运行**:
```bash
gm --base-url "https://spaces.gradmotion.com/dev-api" task create --file task_cloud_train_no_A.json
gm --base-url "https://spaces.gradmotion.com/dev-api" task run --task-id "TASK_xxx"
```

---

## 历史实验配置（不推荐）

### E2 配置（已废弃）
- **文件**: `config/volleyball_s_E2_lr_noscale_bg6.yml`
- **问题**:
  - 仍使用 Method A (`train_with_negatives.json`)
  - `bg_loss_weight=6.0` 可能过度补偿
  - 移除了 `dn_vfl_extra_scale` 但缺乏实验验证
- **状态**: 仅作历史参考，不作为主训练入口

---

## 关键路径

| 路径 | 说明 |
|------|------|
| `/workspace/dfine/` | 代码根目录（手动 clone） |
| `/workspace/dfine/D-FINE/` | D-FINE 训练代码 |
| `/workspace/dfine/config/` | 配置文件目录 |
| `/personal/personal/images.zip` | 训练数据（平台挂载） |
| `/personal/personal/negative_samples.zip` | 负样本数据（可选） |

---

## 常见问题

### Q: 容器 podName=null，没有启动？
**A**: 检查 `startScript` 语法，确保使用手动 `git clone` 方式。

### Q: 找不到文件 `/workspace/isaaclab/benchmark/...`？
**A**: 不要依赖平台 Git checkout 的目录结构，使用手动 clone 到固定路径。

### Q: 日志为空？
**A**: 先运行 smoke test 验证环境，再运行正式训练。

---

## 下一步

1. ✅ 运行 `task_smoke_test_final.json` 验证环境
2. ✅ 确认 smoke test 输出 `SMOKE_TEST_SUCCESS`
3. ✅ 运行 `task_cloud_train_no_A.json` 开始训练
4. ✅ 监控 TensorBoard 指标（AP、OA）
