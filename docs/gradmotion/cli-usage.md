# GradMotion CLI 使用笔记

## 已确认可用

- CLI 命令：`gm`
- 本机版本：`0.1.2`
- 常用命令：
  - `gm --help`
  - `gm task --help`
  - `gm project --help`
  - `gm auth status`
  - `gm auth whoami`
  - `gm task list --page 1 --limit 20`
  - `gm project list --page 1 --limit 20`
  - `gm task image official`
  - `gm task image versions --image-id "..."`
  - `gm task resource list --goods-back-category 3 --page-num 1 --page-size 20`
  - `gm task storage list --folder-path "personal/"`
  - `gm task info --task-id "..."`
  - `gm task logs --task-id "..." --raw --no-request-log`

## 本账号当前关键事实

### 正确 API 入口

当前 API key 在下面这个入口可正常工作：

```bash
gm --base-url "https://spaces.gradmotion.com/dev-api" ...
```

说明：
- 之前用 `prod-api` 时，部分命令返回 `response is not valid JSON`。
- 切到 `dev-api` 后，`whoami / task list / project list / image list / storage list` 均正常。

### 认证状态

- API key 已写入 keychain。
- `gm auth status` 能正常返回本地状态。
- `gm auth whoami` 在 `dev-api` 下可正确返回用户信息。

## 关键平台对象

### 项目

当前训练项目：
- `projectId`: `PRO_20260317_040`
- `projectName`: `D-FINE Volleyball Detection`

### 训练资源

当前使用的 4090D 训练资源：
- `goodsId`: `ESKU000001`
- `goodsBackId`: `SKUSL000002`
- `goodsName`: `1*4090D*24G`

注意：
- 创建任务时填的是 `goodsId`。
- `goodsBackId` 不能误填到 `taskBaseInfo.goodsId`。

### 目标镜像

当前已确认的目标镜像：
- `imageId`: `BJX00000178`
- `imageVersion`: `V000220`
- `versionCode`: `isaac-lab2.3.2-v1`
- 描述：`IsaacSim:5.1 | IsaacLab:2.3.2 | python:3.11.13 | PyTorch:2.7.0 | Ubuntu 22.04.5 LTS`

注意：
- 创建任务时应填 `imageVersion: V000220`
- 不能把 `versionCode` 当 `imageVersion` 用。

## 官方示例的真实结构

官方示例任务 `TASK_20260306_195` 证明：

- 支持 `codeType=2` 的 Git 方式。
- `codeUrl` 是 JSON 字符串，可包含多个仓库。
- `startScript` 可以直接写 `gm-run ...`。

示例特征：
- `codeUrl`: 两个 GitHub 仓库
- `startScript`: `gm-run limx_legged_gym-main/.../train.py ...`

## 当前对本项目最重要的经验

1. 正式训练任务创建前，先做 smoke test。
2. 优先使用平台的 Git checkout 方式，不要在 `startScript` 里重复 clone，除非确有必要。
3. `gm task edit` 不能只提交局部字段，必须先 `task info` 再全量回写。
4. 对日志命令，`--raw --no-request-log` 最适合拿纯日志正文。
5. 不要在任务 JSON 中传 Git 凭证；私有仓库需在平台 Web 端配置 Git 信息。

## 2026-03-18 遥测验证补充

### 官方示例任务能正常显示哪些内容

以 `TASK_20260306_194` 为证：

- `gm task logs --raw --no-request-log` 可直接拿到完整训练日志。
- `gm task data keys --task-id "TASK_20260306_194"` 可返回图表 key。
- `gm task model list --task-id "TASK_20260306_194"` 可返回 checkpoint / 视频。
- `gm task hp get --task-id "TASK_20260306_194"` 可返回完整超参文件内容。
- `gm task env get --task-id "TASK_20260306_194"` 可返回运行环境描述。

重要结论：
- `argoLogDownloadUrl = null` **不等于没有日志**。
- 官方任务里即使 `argoLogDownloadUrl` 为 `null`，CLI 仍可正常拿到日志、图表、模型。

### D-FINE 自定义任务的当前现象

#### 1. `TASK_20260318_066` — smoke-git-checkout-preflight
- 目标镜像 / Git checkout / commit 记录正常。
- `taskCodeInfo.commitId` 已记录为：`37ddf626871758333d6ed89cf64ad702aef127d0`。
- 但日志接口仍为空，`argoLogDownloadUrl` 也为空。

#### 2. `TASK_20260318_073` / `TASK_20260318_074`
- 都是最小 D-FINE 遥测试验。
- 日志、图表 key、模型列表均为空。
- 其中 `074` 长时间停在 `taskStatus = 2`，没有 `podName` / `startTime`。

#### 3. `TASK_20260318_086` — dfine-mini-telemetry-v3
- 这是目前最有价值的一次验证：
  - 有 `podName`
  - 有 `startTime`
  - 有运行时长 `runtime=137`
  - 说明任务**确实真正启动过容器**
- 但同时：
  - `gm task logs` 返回空串
  - `gm task data keys` 返回空数组
  - `gm task model list` 返回空列表
  - `gm task hp get` / `gm task env get` 也没有有效内容
  - `argoLogDownloadUrl` 仍为 `null`
- 不过：
  - `gm task data download --task-id "TASK_20260318_086"` 返回了 `rl-tfevents-logs.tgz` 的下载链接
  - 但实际下载结果是 OSS `NoSuchKey` XML，不是真实 tgz 文件

进一步验证：
- 官方任务 `TASK_20260306_194` 的 `rl-tfevents-logs.tgz` 可正常下载并解包。
- 其事件文件里可读出 `31` 个 scalar tag，和 `gm task data keys` 返回结果一致。
- `TASK_20260318_086` 则属于：**平台返回了下载 URL，但对象实际不存在**。

这说明：
- 官方任务的 TensorBoard → 平台图表索引链路是正常的。
- 自定义 D-FINE 任务在当前平台上，至少存在 **事件包对象未落盘或索引信息伪成功** 的问题。
- 因此平台侧对该任务的**日志展示 / 图表 key 索引 / checkpoint 索引**目前没有像官方示例那样正常暴露出来。

### 额外定位：最小自定义 gm-run 探针

我又补了一轮更小的探针任务：`TASK_20260318_087`。

其特征：
- 不跑 D-FINE 训练栈
- 只运行仓库内一个极小 Python 脚本
- 脚本只做三件事：
  - `print(...)`
  - `SummaryWriter.add_scalar(...)`
  - 写一个本地完成标记文件
- `startScript` 仍按官方模式使用：
  - `gm-run benchmark/scripts/gm_telemetry_probe.py --steps 6 --sleep 1`

后续补查真实结果后，定位又前进了一步：
- 该任务最终其实真正运行过，并不是一直卡死：
  - 有 `podName`
  - 有 `startTime`
  - 有 `runtime=519`
  - 有 `argoLogDownloadUrl`
- `gm task logs --raw --no-request-log` 拿到的真实报错是：
  - `/workspace/isaaclab/_isaac_sim/kit/python/bin/python3: can't open file '/workspace/isaaclab/benchmark/scripts/gm_telemetry_probe.py': [Errno 2] No such file or directory`

这说明：
- 至少这类自定义 Git 任务**不是完全没有运行/日志链路**。
- 更具体的问题是：`gm-run` 实际以 `/workspace/isaaclab` 作为扫描/执行根目录之一，但该路径下并没有我们期望的 `benchmark/scripts/gm_telemetry_probe.py`。
- 也就是说，这一轮失败首先是 **运行时根目录 / 代码落点 / 脚本路径不匹配**，而不是单纯“平台完全不给日志”。
- 同时还要注意一个更直接的问题：`benchmark/scripts/gm_telemetry_probe.py` 目前是本地未跟踪文件，云端 Git checkout 本身也不可能拉到它。

### 2026-03-18 新增验证：`TASK_20260318_118` — `gm-telemetry-probe-v2`

我又做了一轮更强约束的探针：
- 不再依赖仓库里的未跟踪脚本
- 直接在 `startScript` 里把一个极小 Python 脚本写入 `/workspace/isaaclab/gm_probe_v2.py`
- 再用绝对路径执行：
  - `gm-run /workspace/isaaclab/gm_probe_v2.py --steps 6 --sleep 1`

结果：
- 任务最终状态仍为 `taskStatus = 6`
- 但：
  - `podName = null`
  - `startTime = null`
  - `runtime = ""`
  - `commitId = null`
  - `argoLogDownloadUrl = null`
  - `gm task logs` 为空
  - `gm task data keys` 为空
  - `gm task model list` 为空
  - `gm task data download` 为空
  - `gm task env get` 为空

这说明：
- 即使把“未跟踪文件不存在”这个因素排掉，并显式使用绝对路径，平台仍可能在更早阶段就没有真正进入容器运行态。
- 也就是说，当前平台现象并不统一：
  - `TASK_20260318_087` 是“真正启动后因脚本路径失败”
  - `TASK_20260318_118` 则更像是“创建成功但未真正进入有效运行态”

### 当前实际结论

1. 目标镜像 + Git 方式 + `/personal/personal/*.zip` 路径已经基本摸清。
2. 自定义 Git 任务的行为**并不稳定一致**：有的会真正启动并给出 Argo 日志，有的会在更早阶段结束且没有有效运行痕迹。
3. 问题范围已经从“D-FINE 训练代码”收缩到“自定义 Git 任务在 GradMotion 平台上的运行态进入条件 + 遥测/索引兼容性”。
4. 在这个问题收敛前，直接开长训练仍然不稳，前端日志 / 图表 / checkpoint 展示也仍不可靠。
