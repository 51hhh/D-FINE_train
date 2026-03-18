# 历史云任务失败记录

## 已确认的失败根因

### 1. `TASK_20260317_238` — `E2-lr-noscale-bg6-v1`

#### 现象
- 任务很快失败。
- 使用旧镜像 `BJX00000001 / V000021`。

#### 关键信息
- `startScript`:
  ```bash
  cd /code && bash cloud_train.sh volleyball_s_E2_lr_noscale_bg6.yml
  ```

#### 根因
- 日志明确报错：
  - `cd: /code: No such file or directory`
- 说明当时任务实际没有把代码放到 `/code`，或者该任务配置并未正确使用平台 Git checkout 目录。

#### 结论
- 失败原因是 **工作目录假设错误**。
- 不能直接假设平台总会提供 `/code`。
- 后续必须通过 smoke test 先验证 checkout 后的真实工作目录。

---

### 2. `TASK_20260318_007` — `E2-lr-noscale-bg6-v2`

#### 现象
- 改为手工 `git clone` 后仍然失败。
- 使用目标镜像 `BJX00000178 / V000220`。

#### 关键信息
- `startScript`:
  ```bash
  git clone --branch exp/E1-lr-fix-and-scale-removal --depth 1 https://github.com/51hhh/D-FINE_train.git /workspace/dfine && cd /workspace/dfine && bash cloud_train.sh volleyball_s_E2_lr_noscale_bg6.yml
  ```

#### 日志中的直接报错
- `WARNING: /personal/images.zip not found!`
- `WARNING: /personal/negative_samples.zip not found!`
- `FileNotFoundError: No xml files found in /workspace/dfine/coco/Annotations`

#### 根因
不是 Git，也不是镜像环境，而是 **`/personal` 路径判断错误**。

诊断任务证明：
- `/personal` 存在
- `/personal` 目录下只有一个符号链接：
  - `personal -> /tos/prod/3140/personal`
- 实际 zip 位于：
  - `/personal/personal/images.zip`
  - `/personal/personal/negative_samples.zip`

也就是说，`cloud_train.sh` 当前的判断：
```bash
if [ -f "/personal/images.zip" ]; then ...
```
是错误的。

#### 结论
- 平台数据已挂载。
- 失败原因是 **脚本对挂载路径理解错误**。

---

### 3. `TASK_20260318_011` — `diag-filesystem`

#### 作用
- 这是一次成功的诊断任务，不是训练失败任务。
- 它提供了最重要的平台事实证据。

#### 已确认事实
- 当前镜像里工作目录示例为 `/workspace/isaaclab`
- 根目录存在 `/personal`
- `/personal` 是符号链接结构，不是直接文件目录
- `images.zip` 可在 `/tos/prod/3140/personal/images.zip` 找到
- GPU、Python 环境正常

#### 价值
- 为后续 smoke test 与正式训练提供了最关键的路径依据。

## 后续避免方式

1. 先做 smoke test，再做正式训练。
2. 不再假设 `/code` 一定存在；要先验证真实 checkout 路径。
3. 不再假设 `/personal/*.zip` 直接存在；要按真实挂载结构验证。
4. 长训练前，先用日志验证：
   - repo 已 checkout
   - 关键文件在
   - `/personal/personal/*.zip` 在

## 2026-03-18 补充：遥测链路问题

### 4. `TASK_20260318_073` — `dfine-mini-telemetry-v1`

#### 现象
- 任务创建与启动成功。
- 最终 `taskStatus = 6`。
- 但：
  - `gm task logs` 为空
  - `gm task data keys` 为空
  - `gm task model list` 为空
  - `argoLogDownloadUrl = null`

#### 结论
- 这轮没有证明出平台能为自定义 D-FINE 任务正常暴露日志/图表/模型。

---

### 5. `TASK_20260318_074` — `dfine-mini-telemetry-v2`

#### 现象
- 任务启动成功后长时间停留在 `taskStatus = 2`。
- 一直没有 `podName` / `startTime`。
- 日志仍为空。

#### 结论
- 该写法没有真正进入有效运行阶段，不能作为后续正式训练模板。

---

### 6. `TASK_20260318_086` — `dfine-mini-telemetry-v3`

#### 现象
- 任务真正运行过：
  - 有 `podName`
  - 有 `startTime`
  - `runtime = 137`
- 但：
  - `gm task logs` 仍为空
  - `gm task data keys` 仍为空
  - `gm task model list` 仍为空
  - `argoLogDownloadUrl = null`
- 同时：
  - `gm task data download` 返回了 TensorBoard 事件包下载链接 `rl-tfevents-logs.tgz`
  - 但实际下载到的是 OSS `NoSuchKey` 错误 XML，说明对象本身不存在

#### 结论
- 这说明平台对该任务返回了一个“看起来成功”的事件包 URL，但后端对象并没有真正落盘。
- 对比官方任务 `TASK_20260306_194`：其事件包可以正常下载、解包，并读取到与 `gm task data keys` 一致的 31 个 scalar。
- 因此当前瓶颈已经不只是训练脚本本身，而是 **GradMotion 对这类自定义任务的遥测产物落盘/索引兼容性**。

---

### 7. `TASK_20260318_087` — `gm-telemetry-probe-v1`

#### 现象
- 这是一个比 D-FINE 更小的自定义 Git 探针任务。
- `startScript` 仅为：
  - `gm-run benchmark/scripts/gm_telemetry_probe.py --steps 6 --sleep 1`
- 脚本只做：
  - `print(...)`
  - `SummaryWriter.add_scalar(...)`
  - 写完成标记文件
- 后续补查真实运行信息后发现：
  - 任务其实真正跑起来过
  - 有 `podName`
  - 有 `startTime`
  - 有 `runtime=519`
  - 有 `argoLogDownloadUrl`
- 原始日志的直接报错是：
  - `/workspace/isaaclab/_isaac_sim/kit/python/bin/python3: can't open file '/workspace/isaaclab/benchmark/scripts/gm_telemetry_probe.py': [Errno 2] No such file or directory`

#### 根因
- 这一轮的首要失败原因不是“完全没有日志链路”，而是 **运行时脚本路径错误**。
- `gm-run` 实际按 `/workspace/isaaclab` 作为扫描/执行根目录之一，但目标脚本并不在该位置。
- 同时本地代码里 `benchmark/scripts/gm_telemetry_probe.py` 还是未跟踪文件，云端 Git checkout 也不会拿到它。

#### 结论
- 当前已经能确认：
  - 自定义 Git 任务并非一律没有日志；至少该任务拿到了真实 Argo 日志。
  - 但路径解析与代码落点仍然不稳定，不能直接照搬本地相对路径假设。
- 因此下一步应优先验证：
  - 真实 checkout 根目录
  - `gm-run` 的脚本解析规则
  - 使用绝对路径或运行时生成脚本后，是否能稳定产生日志 / 图表 / 模型索引。

---

### 8. `TASK_20260318_118` — `gm-telemetry-probe-v2`

#### 现象
- 这是针对 `087` 的修正版最小探针。
- 不再依赖仓库中的未跟踪脚本。
- `startScript` 会先在 `/workspace/isaaclab` 动态写入一个极小 Python 文件，再执行：
  - `gm-run /workspace/isaaclab/gm_probe_v2.py --steps 6 --sleep 1`
- 但结果是：
  - `taskStatus = 6`
  - `podName = null`
  - `startTime = null`
  - `runtime = ""`
  - `commitId = null`
  - `argoLogDownloadUrl = null`
  - `gm task logs / data keys / model list / data download / env get` 全为空

#### 根因判断
- 这一轮没有证据表明容器真正启动过。
- 因此它和 `087` 不同，不像是“进入运行态后脚本报错”，而更像是 **更早阶段就未进入有效运行态**。

#### 结论
- 自定义 Git 任务当前至少存在两类不稳定现象：
  1. **真正启动后因路径解析失败而终止**（如 `087`）
  2. **任务创建/结束成功，但没有真正进入运行态**（如 `118`）
- 这进一步说明：当前问题不只是单一脚本路径错误，而是平台对自定义 Git 任务的运行态进入与遥测接入都还不稳定。
