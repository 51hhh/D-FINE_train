# GradMotion Smoke Test 方案

## 目标

在不浪费计费时间的前提下，一次性验证以下事实：

1. 平台 Git checkout 是否稳定可用。
2. 实际工作目录与代码落点是否明确。
3. `/personal/personal/images.zip` 是否可读。
4. `/personal/personal/negative_samples.zip` 是否可读。
5. 关键训练文件是否存在：
   - `cloud_train.sh`
   - `D-FINE/train.py`
   - 目标配置文件
6. 训练入口至少能被解释器加载（可选：`python train.py --help`）

## 为什么先做这一步

正式训练失败的两个已知原因都不是模型本身，而是平台路径问题：
- 工作目录假设错误
- `/personal` 路径判断错误

所以在修清这些之前，不应开长时间训练。

## 当前建议任务参数

### 项目与资源
- `projectId`: `PRO_20260317_040`
- `goodsId`: `ESKU000001`
- `imageId`: `BJX00000178`
- `imageVersion`: `V000220`
- `personalDataPath`: `/personal`

### Git 代码
- 仓库：`https://github.com/51hhh/D-FINE_train.git`
- 分支：`exp/E1-lr-fix-and-scale-removal`
- `codeType`: `2`

## 推荐 smoke test startScript

```bash
set -eu
echo '=== SMOKE TEST START ==='

echo '[pwd]'
pwd

echo '[whoami]'
whoami

echo '[list /code]'
ls -la /code

echo '[git dir]'
test -d /code/.git || echo 'INFO: /code/.git not exposed by platform'

echo '[key files]'
test -f /code/cloud_train.sh
test -f /code/D-FINE/train.py
test -f /code/config/volleyball_s_E2_lr_noscale_bg6.yml

echo '[config list]'
ls -1 /code/config/*.yml

echo '[personal mount]'
test -d /personal
test -d /personal/personal

echo '[zip files]'
test -f /personal/personal/images.zip
test -f /personal/personal/negative_samples.zip

echo '[python]'
python3 --version || python --version

echo '[optional parser sanity]'
cd /code/D-FINE && python train.py --help >/dev/null

echo 'SMOKE_TEST_OK'
echo '=== SMOKE TEST END ==='
```

## 通过标准

日志中必须能确认：
- Git checkout 成功
- `/code` 下有项目文件
- `/personal/personal/images.zip` 存在
- `/personal/personal/negative_samples.zip` 存在
- `cloud_train.sh` 存在
- `D-FINE/train.py` 存在
- 目标配置文件存在

## 不放进 smoke test 的内容

为了节约时间，以下步骤不应放进当前 smoke test：
- 解压完整数据集
- 生成 COCO 标注
- `pip install -r requirements.txt`
- 正式训练
- 长时间 GPU 占用

## 下一步

如果 smoke test 通过，再进入“极短训练闭环”阶段：
- 只验证数据解压、训练入口、首轮迭代与日志
- 不直接开 70–90 epoch 的正式实验
