# D-FINE OA/AP 研究结论

## 当前结论摘要

1. **AP best 不等于业务 best。**
   - 对“无球时不能误检”这个目标，不能只看 AP。
2. **当前用户不再希望继续使用 Method A。**
   - 原因是 OA 测试集负样本不能与训练负样本混用，否则 OA 结论失真。
3. **OA 评估集必须独立。**
   - 训练负样本、选模负样本、最终盲测负样本应隔离。
4. **当前代码中的 `best_oa.pth` 只保存，不会作为继续训练的恢复点。**
5. **当前更合理的主线，是不依赖外部 A 负样本的可信评估 + 再验证当前分支。**

## 08 轮的关键信息

历史目录：
- `E:\数据集\model\08_method_D_bg2_latest`

已确认：
- 08 的归档实验名更接近 `exp_s_obj2coco_neg_d_bg2`
- AP 峰值约为 `0.894`

但同时：
- OA 不能认为同样优秀
- 不能因为 AP 高就直接认为适合业务上线

## 关于 `best_oa.pth`

代码证据：
- `D-FINE/src/solver/det_solver.py:75-79`
  - stage 切换时加载的是 `best_stg1.pth`
- `D-FINE/src/solver/det_solver.py:208-215`
  - `best_oa.pth` 只在满足条件时保存

因此：
- 当前代码机制下，`best_oa` 不是继续训练起点
- 只是额外保存的业务候选权重

## 当前不再接受的前提

- 不再接受把 OA 测试负样本直接并入训练。
- 不再接受用 A 路线的 OA 结果作为可信上线依据。

## 后续更可信的方向

1. 不用外部 A 负样本直接训练。
2. OA 测试集独立保留。
3. 若需要训练负监督，可考虑：
   - 在现有正样本图中，随机裁切非排球区域作为局部负样本
   - 或从模型机制上抑制过度自信
4. 先验证当前分支，而不是直接押 `bg_loss_weight=6` 为唯一解。

## 当前对 E2 的判断

配置文件：
- `config/volleyball_s_E2_lr_noscale_bg6.yml`

其特征：
- `num_neg_random: 100`
- `neg_iou_threshold: 0.2`
- `bg_loss_weight: 6.0`
- `oa_ap_min: 0.84`
- 仍然使用 `train_with_negatives.json`

因此，E2 仍带 A，不符合当前新的路线偏好。
