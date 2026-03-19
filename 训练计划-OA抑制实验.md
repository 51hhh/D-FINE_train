# D-FINE 训练计划 - Query Over-Activation 抑制实验

**创建时间**: 2026-03-19
**当前分支**: exp/E2-no-a-train
**目标**: 解决单类排球检测的Query Over-Activation误检问题

---

## 当前训练状态

### 正在运行的训练
- **配置**: `config/volleyball_s_obj2coco_d_bg2.yml`
- **进度**: Epoch 63/160
- **最佳AP**: 0.8875 @ epoch 61
- **趋势**: epoch 57-61持续上升(0.881→0.888)，epoch 62-63略有波动
- **输出目录**: `./output/exp_s_obj2coco_d_bg2_no_A`

### 配置特点
- num_queries: 100 (300→100，减少过激活槽位)
- num_neg_random: 100 (DN随机负框)
- bg_loss_weight: 2.0
- 训练集: train.json (no-A，不包含负样本)
- eval_overactivation: False (暂未启用OA评估)

### 问题
- ❌ 无OA评估数据（eval_overactivation=False）
- ❌ 负样本误检FPs=194（每张图约2个误检）

---

## 决策与行动计划

### 阶段1: 继续当前训练 (当前 → epoch 80)

**决策**: 继续bg2训练到epoch 80

**理由**:
- AP趋势健康，仍在上升期
- q100配置首次独立验证（历史与freeze绑定）
- 需要完整数据评估q100效果

**判断标准**:
- ✅ 若AP持续≥0.885：继续到120
- ⚠️ 若连续10 epoch无提升：提前停止
- ❌ 若出现明显过拟合：停止

### 阶段2: 并行启动bg4训练 (立即)

**配置**: `config/volleyball_s_obj2coco_d_bg4.yml`

**关键参数**:
```yaml
bg_loss_weight: 4.0          # 从2.0提升到4.0，更激进抑制背景
num_queries: 100
num_neg_random: 100
eval_overactivation: False   # 暂不启用
```

**目的**:
- 验证bg权重敏感度
- 与bg2形成对照组
- 渐进式增强背景惩罚

**训练命令**:
```bash
cd D-FINE
python train.py -c ../config/volleyball_s_obj2coco_d_bg4.yml --use-amp --seed 0 -t weights/dfine_s_obj2coco.pth
```

### 阶段3: Epoch 80评估与决策

**评估内容**:
1. 对比bg2 vs bg4的AP曲线
2. 分析loss收敛速度和稳定性
3. 决定是否启用OA评估

**决策路径**:
```
bg2 @ epoch 80
    ├─ AP≥0.89 且稳定 → 继续到120 + 启用OA评估
    ├─ AP=0.87-0.89 → 等bg4结果，决定是否切换
    └─ AP<0.87 → 考虑q50或调整DN

bg4 @ epoch 80
    ├─ bg4 > bg2 (AP差距≥0.005) → 主线切换到bg4
    ├─ bg2 > bg4 → 继续bg2，尝试DN=150
    └─ 相近 → 选bg4（理论上更抑制OA）
```

### 阶段4: 备选方案 (根据epoch 80结果触发)

#### 方案A: q50配置（更激进减少queries）
**触发条件**: q100在epoch 80时AP<0.89 或 OA问题未改善

**配置**: `volleyball_s_obj2coco_d_q50_bg4.yml`
```yaml
num_queries: 50              # 减半
num_neg_random: 50           # 同步减半
bg_loss_weight: 4.0
```

#### 方案B: DN负框数量调整
**触发条件**: bg4效果好但OA仍有问题

**配置**: `volleyball_s_obj2coco_d_bg4_dn150.yml`
```yaml
num_neg_random: 150          # 从100提升到150
bg_loss_weight: 4.0
```

---

## OA评估策略

### 当前状态
- OA门控逻辑已修复（commit 47a05f1 + 最新修复）
- 配置中eval_overactivation=False（暂未启用）

### 启用时机
**建议**: epoch 80后，根据AP表现决定

**启用条件**:
- bg2或bg4的AP稳定在0.89+
- 需要收集OA基线数据

**配置修改**:
```yaml
eval_overactivation: True
oa_ap_min: 0.84              # AP门控阈值
oa_conf_threshold: 0.3
```

### OA指标目标
- oa_fppi < 0.05 (每张图误检<0.05个)
- oa_clean_rate > 0.95 (95%图片无误检)
- oa_max_score < 0.5 (最高误检置信度<50%)

---

## 实验记录表

| 实验名 | 配置 | 状态 | 最佳AP | Epoch | OA指标 | 备注 |
|--------|------|------|--------|-------|--------|------|
| bg2_q100_no_A | volleyball_s_obj2coco_d_bg2.yml | 运行中 | 0.8875 | 61/160 | 未评估 | 当前主线 |
| bg4_q100_no_A | volleyball_s_obj2coco_d_bg4.yml | 待启动 | - | - | - | 并行对照组 |

---

## 关键文件路径

### 配置文件
- 当前bg2: `config/volleyball_s_obj2coco_d_bg2.yml`
- 新建bg4: `config/volleyball_s_obj2coco_d_bg4.yml`
- 历史A+D+bg2: `config/volleyball_s_obj2coco_neg_d_bg2.yml`

### 训练输出
- bg2输出: `D-FINE/output/exp_s_obj2coco_d_bg2_no_A/`
- bg4输出: `D-FINE/output/exp_s_obj2coco_d_bg4_no_A/`

### 日志
- 当前训练日志: `log2` (服务器上)

### 代码修改
- OA门控逻辑: `D-FINE/src/solver/det_solver.py:125-175`

---

## 停止/继续训练判断标准

### 停止训练
- ❌ 连续15 epoch无AP提升（当前best=0.888）
- ❌ 出现严重过拟合（train AP - val AP > 8%）
- ❌ 训练不稳定（loss震荡>20%）

### 继续训练
- ✅ AP每10 epoch提升≥0.002
- ✅ loss曲线平滑下降
- ✅ 未达到预期收敛点（AP<0.90）

---

## 下一步行动清单

### 立即执行
- [x] 修复OA门控逻辑bug
- [x] 创建bg4配置文件
- [x] 创建训练计划文档
- [ ] 提交代码修改到git
- [ ] 在服务器上启动bg4训练

### Epoch 80前
- [ ] 监控bg2训练进度
- [ ] 分析bg2的AP和loss曲线
- [ ] 准备OA评估配置

### Epoch 80时
- [ ] 对比bg2 vs bg4结果
- [ ] 决定是否启用OA评估
- [ ] 决定是否需要q50或DN调整

---

## 参考文档
- 模型误识别调试总结: `模型误识别调试总结.md`
- OA门控逻辑修改: `OA门控逻辑修改说明.md`
- GPT-5.4代码审查: `GPT5.4代码审查请求.md`
