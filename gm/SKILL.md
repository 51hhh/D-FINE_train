---
name: gm-cli
description: Operates gm-cli (gm): auth/config/profile/project/task workflows, safe execution patterns, and troubleshooting. Use when the user mentions gm, gm-cli, Gradmotion, API key, base_url, profile, auth login/logout/whoami/status, project list/create/edit/delete/info, task create/edit/copy/list/info/run/stop/delete/logs/resource/image/storage/data/hp/env/params/tag/batch, or wants CLI automation.
---

# gm-cli

## 适用范围
用 `gm` 完成以下工作流：
- 认证：`gm auth login/logout/status/whoami`
- 配置与 profile：`gm config ...`
- 项目管理：`gm project ...`（list/create/edit/delete/info）
- 任务管理：`gm task ...`（create/edit/copy/list/info/run/stop/delete/logs/resource/image/storage/data/hp/env/params/tag/batch）

## 安全与约束
- 不要在对话输出中回显用户的 `api-key` 或完整密钥内容。
- 高风险操作（`project delete`、`task stop/delete`、`task batch stop/delete`）默认需要二次确认；只有用户明确要求无人值守时才加 `--yes`。
- 涉及文件路径一律使用相对路径（例如 `--file ./payload.json`）。
- Agent 生成的临时 JSON 文件（如 `create-*.json`、`edit-*.json`、`copy-*.json` 等），在对应的 CLI 命令执行成功后应**立即删除**，避免在工作目录中残留过期的临时文件。

## 帮助与版本
- 总览：`gm --help`
- 子命令：`gm <command> --help`
- 版本：`gm --version`（或 `gm -v`，不使用 `gm version`）

## 执行前快速探测（固定步骤）
在执行任何写操作前，先跑以下 3 条命令确认 CLI 能力与命令可用性：
1. `gm --help`
2. `gm task --help`
3. `gm project --help`

建议：
- 若任一命令报错，先不要继续执行创建/编辑/删除类操作。
- 若子命令缺失，优先按当前 CLI 版本能力降级执行或提示用户升级。

## 配置优先级
按优先级覆盖：`CLI flags > 环境变量 > 配置文件`。

常用环境变量：
- `GM_PROFILE`
- `GM_BASE_URL` / `GM_API_KEY` / `GM_TIMEOUT` / `GM_RETRY` / `GM_CONCURRENCY`

说明：
- 默认：CLI 请求 `base_url + /api + endpoint`。
- 绝对路径模式：个别命令（如 `gm task storage list`）会直接请求 `base_url + endpoint`（不自动拼 `/api`）。

## 首次上手（推荐步骤）
1. 设置 base_url（落到当前 profile）：
   - `gm config set base_url "https://YOUR-HOST/prod-api"`
2. 登录保存 API Key（优先 Keychain，失败回落 config）：
   - `gm auth login --api-key "<YOUR_KEY>"`
3. 验证：
   - `gm auth status`（本地）
   - `gm auth whoami`（请求服务端）

如需临时覆盖且不落盘：
- `gm --base-url "https://..." --api-key "<KEY>" auth whoami`

## Profile（多环境）
- 列表：`gm config profile list`
- 创建/更新：
  - `gm config profile set dev --base-url "https://..." --timeout 30s --retry 3 --concurrency 4`
- 切换：`gm config profile use dev`
- 临时指定：`gm --profile dev task list` 或 `GM_PROFILE=dev gm task list`

## Task 常用操作
- 列表：`gm task list --page 1 --limit 50`
- 详情：`gm task info --task-id "task_xxx"`
- Checkpoint 列表：`gm task model list --task-id "task_xxx" --page-num 1 --page-size 20`
- 复制：`gm task copy --file ./copy.json`
- 运行：`gm task run --task-id "task_xxx"`
- 停止：`gm task stop --task-id "task_xxx"`（会二次确认）
- 删除：`gm task delete --task-id "task_xxx"`（会二次确认）

日志：
- 单次：`gm task logs --task-id "task_xxx"`
- 追踪：`gm task logs --task-id "task_xxx" --follow --interval 2s --timeout 1m`
- 仅输出日志正文（不包 JSON）：`gm task logs --task-id "task_xxx" --raw`；管道/重定向时常用。
- 不向 stderr 打请求元数据：`gm task logs ... --no-request-log`；与 `--raw` 搭配可得到纯净日志流。

资源/镜像/存储：
- 资源列表：`gm task resource list --goods-back-category 3 --page-num 1 --page-size 10`
  > 返回字段说明：`goodsId` 对应 `taskBaseInfo.goodsId`（填任务时使用此值）；`goodsBackId` 是后台 SKU 标识，**不可**混用。
- 官方镜像：`gm task image official`
- 个人镜像：`gm task image personal --version-status 1 --page-num 1 --page-size 50`
- 镜像版本：`gm task image versions --image-id "img_xxx"`
  > 返回字段说明：`id` 对应 `taskBaseInfo.imageVersion`（填任务时使用此值），`versionCode` 仅为可读标识，**不可**用于 `imageVersion` 字段。
- 个人存储：`gm task storage list --folder-path "personal/"`

图表/超参/环境：
- 图表 keys：`gm task data keys --task-id "task_xxx"`
- 图表数据：`gm task data get --task-id "task_xxx" --data-key "train/loss"`
- 图表下载：`gm task data download --task-id "task_xxx"`
- 超参读取：`gm task hp get --task-id "task_xxx"`
- 运行环境：`gm task env get --task-id "task_xxx"`

任务打标签：
- 更新标签：`gm task tag update --task-id "task_xxx" --tags "tag1,tag2"` 或 `--file ./tag.json`
- 查看任务标签：`gm task tag get --task-id "task_xxx"`
- 用户历史标签列表：`gm task tag list --limit 200`

> **说明**：任务分享（生成分享链接）需在 Web 端操作，CLI 未提供对应命令。

## Project 常用操作
- 列表：`gm project list --page 1 --limit 50`
- 创建：`gm project create --file ./project-create.json`
- 编辑（如修改名称）：`gm project edit --file ./project-edit.json` 或 `--data '{"projectId":"proj_xxx","projectName":"新名称"}'`
- 删除：`gm project delete --project-id "proj_xxx"`（会二次确认，可加 `--yes` 跳过）
- 详情：`gm project info --project-id "proj_xxx"`

## create/edit/params 的请求体输入（JSON）
这些命令通过 `--data` 或 `--file` 提供 JSON（两者二选一）。
- `gm task create --data '{"...":"..."}'`
- `gm task create --file ./create.json`
- `gm task edit --file ./edit.json`

超参：
- `gm task params submit --task-id "task_xxx" --file ./params.json`
- `gm task params update --task-id "task_xxx" --data '{"...":"..."}'`

## 最小可运行 JSON 模版（训练任务）
下面是一个“可创建并可运行”的最小模板（请替换示例值）：

```json
{
  "taskBaseInfo": {
    "projectId": "proj_xxx",
    "taskType": "1",
    "trainType": "1",
    "taskName": "mvp-train-task",
    "taskDescription": "created by gm-cli",
    "taskTag": [],
    "goodsId": "goods_xxx",
    "imageId": "BJX00000001",
    "imageVersion": "V000057",
    "personalDataPath": "/personal"
  },
  "taskCodeInfo": {
    "codeType": "2",
    "codeUrl": "[{\"codeUrl\":\"https://github.com/your-org/your-repo.git\",\"versionType\":\"1\",\"versionName\":\"main\"}]",
    "mainCodeUri": "train.py",
    "hparamsPath": "configs/train.yaml",
    "startScript": "gm-run train.py --headless",
    "isOpen": "1"
  },
  "runtimeReminderConfig": {
    "enableRuntimeReminder": false,
    "reminderDurations": []
  }
}
```

建议创建与运行步骤：
- `gm task create --file ./create-train.json`
- 从返回结果取 `taskId`，执行 `gm task run --task-id "task_xxx"`
- `gm task logs --task-id "task_xxx" --follow`

## 最小可运行 JSON 模版（恢复训练任务）
下面是一个“基于已有任务 checkpoint 恢复训练”的最小模板（请替换示例值）：

```json
{
  "taskBaseInfo": {
    "projectId": "proj_xxx",
    "taskType": "1",
    "trainType": "2",
    "taskName": "resume-train-task",
    "taskDescription": "resume from existing checkpoint",
    "taskTag": [],
    "goodsId": "goods_xxx",
    "imageId": "BJX00000001",
    "imageVersion": "V000057",
    "personalDataPath": "/personal"
  },
  "taskCodeInfo": {
    "codeType": "2",
    "codeUrl": "[{\"codeUrl\":\"https://github.com/your-org/your-repo.git\",\"versionType\":\"1\",\"versionName\":\"main\"}]",
    "mainCodeUri": "train.py",
    "hparamsPath": "configs/train.yaml",
    "startScript": "gm-run train.py --headless --max_iterations=4000",
    "isOpen": "1",
    "checkPointFilePath": "upload/2026/3/17/model_3000_xxx.pt",
    "checkPointMountPath": "your-project-root/",
    "resumeFromTaskId": "task_source_xxx",
    "resumeFromTaskName": "source-task-name",
    "resumeFromCheckPoint": "3000"
  },
  "runtimeReminderConfig": {
    "enableRuntimeReminder": false,
    "reminderDurations": []
  }
}
```

建议恢复步骤：
- **先获取任务列表**：执行 `gm task list --page 1 --limit 50`，展示任务列表供用户选择。
- **用户选择源任务**：根据列表中的 `taskId`、任务名称、任务状态等，让用户确认或选择要恢复的源任务（得到 `task_source_xxx`）。
- **再获取该任务的 checkpoint**：执行 `gm task model list --task-id "task_source_xxx" --page-num 1 --page-size 20`，找到目标 checkpoint。
- 将返回结果中的 `policUrl` 填入 `taskCodeInfo.checkPointFilePath`
- 将 `resumeFromTaskId` / `resumeFromTaskName` / `resumeFromCheckPoint` 与源任务、源 checkpoint 保持一致
- `gm task create --file ./create-resume-train.json`
- 如需启动，再执行 `gm task run --task-id "task_xxx"`

恢复训练任务的软提示：
1. 优先复用源任务的 `goodsId`、`imageId`、`imageVersion`、`codeUrl`、`mainCodeUri`、`hparamsPath`，避免环境不一致导致恢复失败。
2. `checkPointFilePath` 应优先使用 `gm task model list` 返回的 `policUrl`，不要手写猜测路径。
3. 若用户只是想“基于 checkpoint 新建恢复任务”，默认先 `create`，不要自动 `run`；只有用户明确要求时才执行运行。

### 关于本地 zip 压缩包上传（`codeType=1`）—— 不支持

> **重要限制：Agent / gm-cli 不支持本地上传 zip 压缩包的方式创建训练任务。**

原因：
- `codeType=1` 要求代码以 zip 包形式先上传至平台对象存储（OSS），再将返回的 OSS 路径填入 `codeUrl`。
- 这一上传过程依赖 Web 端的文件上传接口，gm-cli 及 Agent 均未实现本地文件上传到 OSS 的能力。
- 因此，即使在 JSON 中指定 `codeType=1`，Agent 也无法帮用户完成本地压缩包的上传流程。

**Agent 应遵循以下规则：**
1. 当用户要求以本地 zip/压缩包方式创建训练任务时，应**明确告知不支持**，并建议用户改用 Git 仓库方式（`codeType=2`）。
2. 若用户坚持使用 zip 方式，应引导用户前往 Web 平台手动上传。
3. 创建训练任务时，**默认且唯一推荐的代码方式为 Git 仓库**（`codeType=2`），参见上方模板。

### 私有 Git 仓库需要配置账号和 Token

> **重要提示：当用户使用私有 Git 仓库（GitHub / GitLab）创建训练任务时，必须确认已在平台配置 Git 凭证。**

背景：
- 任务运行时，后端会自动从用户资料中读取 `github_name` + `github_token`（或 `gitlab_name` + `gitlab_token`）来拉取私有仓库代码。
- 这些凭证通过 Web 平台的「个人设置 → Git 信息」页面配置（对应后端接口 `POST /api/user/editGitInfo`）。
- **gm-cli 当前没有管理 Git 凭证的命令**，无法通过 CLI 直接设置。

**Agent 应遵循以下规则：**
1. 当用户提供的 `codeUrl` 为私有仓库时，**必须主动询问**用户是否已在平台配置对应的 Git 账号和 Token。
2. 若用户尚未配置，应引导其前往 Web 平台「个人设置 → Git 信息」页面填写：
   - GitHub 仓库需填写：`GitHub 账号名` 和 `GitHub Token`（Personal Access Token）
   - GitLab 仓库需填写：`GitLab 账号名` 和 `GitLab Token`
3. **不要在 task create 的 JSON 中传递 Git 凭证**——凭证不属于任务请求体，由后端在运行时自动关联。
4. 若用户不确定仓库是否为私有，建议先确认仓库可访问性后再创建任务，避免任务运行时因拉取代码失败而报错。

### `gm task edit` 必须提交完整数据（全量更新）

> **重要限制：后端 `edit_task_dao` 采用全字段覆盖更新（`model_dump()` → `UPDATE`），不支持部分字段更新。**
> **如果只提交用户想改的字段，其余未传字段会被 Pydantic 默认值（`None`/`''`）覆盖，导致 `task_status`、`userId`、`queue` 等关键数据丢失，接口返回 500 或数据损坏。**

**Agent 执行 `gm task edit` 的强制流程：**

1. **先读后改**——执行编辑前，必须先通过 `gm task info --task-id "task_xxx"` 获取任务当前完整数据。
2. **合并生成完整 JSON**——以 `task info` 返回的 `taskBaseInfo` + `taskCodeInfo` 为基础，仅替换用户明确要求修改的字段，其余字段保持原值。
3. **补齐 `taskCodeInfo.taskId`**——编辑 JSON 的 `taskCodeInfo` 中必须包含 `taskId`（与 `taskBaseInfo.taskId` 一致），否则后端代码表更新会因主键为空而失效。
4. **写出完整 JSON 文件**——将合并后的完整 payload 写入 `--file`，再执行 `gm task edit --file ./edit.json`。

**合并时需特别注意的字段：**
- `taskBaseInfo.taskStatus`：必须保留原值，绝不能被默认值覆盖
- `taskBaseInfo.userId`：必须保留原值
- `taskBaseInfo.goodsId` / `goodsBackId`：保留原值或替换为用户指定的新值
- `taskBaseInfo.imageId` / `imageVersion`：保留原值或替换为用户指定的新值
- `taskBaseInfo.taskTag`：保留原值（数组格式）
- `taskCodeInfo` 的所有字段：保留原值，仅覆盖用户明确修改的部分

**示例流程（Agent 修改算力资源和镜像）：**
```bash
# 1. 读取现有任务
gm task info --task-id "TASK_xxx"

# 2. Agent 在本地合并 JSON：以 info 返回为基础，替换 goodsId/imageId/imageVersion
# 3. 写出完整 edit JSON 文件（包含所有原有字段 + 修改字段）

# 4. 执行编辑
gm task edit --file ./edit-task.json
```

## Task 参数软限制
说明：
- 这是 **skill 软限制**（执行前检查并提示），不是后端强校验的替代。
- 字段命名优先使用后端 alias（小驼峰），如 `taskBaseInfo`/`taskCodeInfo`/`taskId`。
- 后端在 task 流程中并未统一显式调用 `validate_fields()`；因此本节采用：
  - `STRICT`：强烈建议拦截（明显会失败或风险极高）
  - `WARN`：仅提示（业务建议或 Agent 侧保护）

### 1) `gm task create` -> `POST /api/task/create` -> `TaskCreateModel`
`STRICT`：
- 顶层必须有：`taskBaseInfo`（object）、`taskCodeInfo`（object）
- `taskBaseInfo.projectId`：必填，长度 `1..20`
- `taskBaseInfo.taskName`：必填，长度 `1..100`
- `taskCodeInfo.codeType`：必填，长度 `1..2`

`WARN`：
- `taskBaseInfo.taskDescription`：可选，最大 `1000`
- `taskBaseInfo.goodsId`：建议必填，最大 `20`（业务侧常见硬依赖）
- `taskBaseInfo.userId`：可选；服务端会按当前登录用户覆盖，不建议依赖请求体值
- `taskCodeInfo.codeUrl`：可选，最大 `1000`
- `taskCodeInfo.mainCodeUri`：可选，最大 `255`
- `taskCodeInfo.runParams`：可选，最大 `255`
- `taskCodeInfo.urdfPath` / `hparamsPath`：可选，最大 `255`
- `taskCodeInfo.checkPointFilePath`：可选，最大 `1000`；若传入建议先确认对象存在

### 2) `gm task edit` -> `POST /api/task/edit` -> `TaskEditModel`
`STRICT`（在 create 基础上追加）：
- `taskBaseInfo.taskId`：必填，长度 `1..20`
- `taskCodeInfo.taskId`：必填，须与 `taskBaseInfo.taskId` 一致（后端不会自动同步，缺失会导致代码表更新失效）
- **必须提交完整字段**：编辑前先 `gm task info` 获取原数据，合并后再提交（参见上方「`gm task edit` 必须提交完整数据」章节）

`WARN`：
- `taskCodeInfo.isCopyHparams`：可选，建议 `1/2`

### 3) `gm task list` -> `POST /api/task/list` -> `TaskPageQueryModel`
`STRICT`：
- 无必须拦截项（可空请求体，CLI 会补 page）

`WARN`：
- `pageNum`：建议 `>=1`
- `pageSize`：建议 `1..200`（Agent 侧保护阈值，非后端硬限制）
- 若带过滤字段，建议遵循：`projectId <= 20`、`taskName <= 100`、`taskDescription <= 1000`

### 4) `gm task info` -> `GET /api/task/info/{task_id}`
`STRICT`：
- `task-id`：必填，长度 `1..20`

### 5) `gm task model list` -> `POST /api/task/model/info` -> `TaskModelPagesModel`
`STRICT`：
- `task-id`：必填，长度 `1..20`

`WARN`：
- `page-num`：建议 `>=1`
- `page-size`：建议 `1..200`
- `checkpoint`：可选筛选条件

### 6) `gm task run` -> `POST /api/task/run` -> `TaskOperation`
`STRICT`：
- 请求体 `task_id`：必填，长度 `1..20`

`WARN`（执行前建议先 `task info` 预检）：
- 任务状态应为草稿态（后端要求 `task_status == "0"`）
- 任务应已绑定可用资源（如 `goodsId/imageId/goodsBackId` 完整）

### 7) `gm task stop` -> `POST /api/task/stop` -> `TaskOperation`
`STRICT`：
- 请求体 `task_id`：必填，长度 `1..20`

`WARN`（执行前建议先 `task info` 预检）：
- 后端不允许停止状态 `0/5/6` 的任务

### 8) `gm task delete` -> `POST /api/task/del` -> `TaskDelModel`
`STRICT`：
- 请求体 `task_id`：必填，长度 `1..20`

`WARN`（执行前建议先 `task info` 预检）：
- 后端仅允许状态集合 `{0,1,2,5,6}` 删除

### 9) `gm task logs` -> `POST /api/task/console/log` -> `ConsoleLogUp`
`STRICT`：
- 请求体 `task_id`：必填，长度 `1..20`

### 10) `gm task params submit` -> `POST /api/task/hp/up` -> `TaskHpModel`
`STRICT`：
- `task_id`：必填，长度 `1..20`

`WARN`：
- `hp_file_name`：建议提供，最大 `1000`
- `hp_file_uri`：建议提供，最大 `1000`
- `hp_save_file_uri`：建议提供，最大 `1000`

### 11) `gm task params update` -> `POST /api/task/hp/edit` -> `EditTaskHpModel`
`STRICT`：
- `task_id`：必填，长度 `1..20`

`WARN`：
- `hp_file_content`：建议必填，最大 `20000`

### 12) `gm task batch stop` -> `POST /api/task/batch/stop` -> `BatchTaskOperation`
`STRICT`：
- `task_ids`：必填，非空数组

`WARN`：
- 每个 `task_id` 建议长度 `1..20`

### 13) `gm task batch delete` -> `POST /api/task/batch/delete` -> `BatchTaskDelete`
`STRICT`：
- `task_ids`：必填，非空数组

`WARN`：
- 每个 `task_id` 建议长度 `1..20`

### 14) `gm task copy` -> `POST /api/task/copy` -> `CopyTaskModel`
`STRICT`：
- 建议请求体包含：`taskId`、`projectId`、`taskName`

`WARN`：
- `taskDescription`：可选，最大 `1000`

### 15) `gm task resource list` -> `GET /api/task/goods/list-by-category`
`STRICT`：
- `goods-back-category`：必填，建议值 `3`（训练）或 `4`（开发机）

`WARN`：
- `page-num >= 1`
- `page-size >= 1`

### 16) `gm task image official` -> `GET /api/images/official/list`
`STRICT`：
- 无必须参数

### 17) `gm task image personal` -> `GET /api/images/personal/list`
`STRICT`：
- 无必须参数

`WARN`：
- `version-status` 默认 `1`
- 分页建议：`page-num >=1`, `page-size >=1`

### 18) `gm task image versions` -> `GET /api/task/getImageVersion`
`STRICT`：
- `image-id`：必填
- 返回的 `id` 字段（如 `V000057`）才是 `taskBaseInfo.imageVersion` 的正确填值；`versionCode`（如 `isaac-gym-v17`）是可读标识，**不能**用于 `imageVersion`。

### 19) `gm task storage list` -> `GET /gm/storage/list`（绝对路径模式）
`STRICT`：
- 无必须参数

`WARN`：
- 查询参数使用 `folderPath`（CLI flag: `--folder-path`）

### 20) `gm task data keys` -> `GET /api/task/data/keys/{task_id}`
`STRICT`：
- `task-id`：必填，长度 `1..20`

### 21) `gm task data get` -> `POST /api/task/data/info` -> `GetDataInfoModel`
`STRICT`：
- `task_id`：必填，长度 `1..20`
- `data_key`：必填

`WARN`：
- `sampling_mode`：建议 `precise|accelerate`
- `max_data_points`：建议正整数

### 22) `gm task data download` -> `GET /api/task/data/download/{task_id}`
`STRICT`：
- `task-id`：必填，长度 `1..20`

### 23) `gm task hp get` -> `GET /api/task/hp/info/{task_id}`
`STRICT`：
- `task-id`：必填，长度 `1..20`

### 24) `gm task env get` -> `GET /api/task/run/env/{task_id}`
`STRICT`：
- `task-id`：必填，长度 `1..20`

### 25) `gm project list` -> `POST /api/project/list`
`STRICT`：
- 无必须拦截项（可空请求体，CLI 会补 page）

`WARN`：
- `pageNum >= 1`
- `pageSize >= 1`

### 26) `gm project create` -> `POST /api/project/create`
`STRICT`：
- 建议请求体至少包含：`projectName`

### 27) `gm project info` -> `GET /api/project/info/{project_id}`
`STRICT`：
- `project-id`：必填，长度建议 `1..20`

### 28) `gm project edit` -> `POST /api/project/edit` -> `ProjectEditModel`
`STRICT`：
- `projectId`：必填，长度 `1..20`

`WARN`：
- `projectName`：可选，最大 `100`
- `projectDescription`：可选，最大 `1000`

### 29) `gm project delete` -> `POST /api/project/del` -> `ProjectDelModel`
`STRICT`：
- `projectId`：必填，长度 `1..20`

`WARN`：
- 会级联删除项目下所有任务，执行前建议确认；默认需二次确认，可用 `--yes` 跳过。

### 30) `gm task tag update` -> `POST /api/task/updateTag` -> `UpdateTaskTagModel`
`STRICT`：
- `taskId`：必填，长度 `1..20`

`WARN`：
- `taskTag`：数组，可为空（清空标签）

### 31) `gm task tag get` -> `POST /api/task/getTaskTag` -> `TaskIdModel`
`STRICT`：
- `task-id`：必填，长度 `1..20`

### 32) `gm task tag list` -> `POST /api/task/getUserTag`
`STRICT`：
- 无必须参数

`WARN`：
- `limit`：查询参数，默认 `200`

## 软限制执行规则（给 Agent）
- 先按 `STRICT` 做预检，不通过则先提示修复示例后再执行。
- `WARN` 只提醒，不阻塞；用户明确要求可带风险继续执行。
- 对 `create/edit` 优先建议 `--file ./payload.json`，避免 shell 转义造成 JSON 结构错误。
- 对 `project create/copy/data get` 同样优先建议 `--file ./payload.json`。
- 对 `run/stop/delete` 优先建议先执行 `gm task info --task-id ...` 做状态预检。
- 默认不自动补全高风险业务字段（如 `goodsId`）；需向用户确认或回读既有任务信息后再填充。

## 批量操作（batch）
- `gm task batch stop --task-ids "t1,t2,t3"`（高风险，默认确认）
- `gm task batch delete --task-ids "t1,t2,t3"`（高风险，默认确认）

无人值守脚本（用户明确要求时才用）：
- `gm --yes project delete --project-id "proj_xxx"`
- `gm --yes task stop --task-id "task_xxx"`
- `gm --yes task delete --task-id "task_xxx"`
- `gm --yes task batch stop --task-ids "t1,t2"`

## 输出与调试
- 默认 stdout：JSON；`--human` 人类可读；`--quiet` 仅关键字段
- `--debug` 开启调试日志
- `--log-file ./gm.log` 将 stderr JSONL 写入文件

## 常见报错快速处理
- base_url 为空：先 `gm config set base_url "..."` 或设置 `GM_BASE_URL`
- api key 为空：先 `gm auth login --api-key ...` 或设置 `GM_API_KEY`
- 不确定参数：先跑 `gm <command> --help`

## task edit 关键注意事项

> **严重警告：后端 edit 接口会用提交的数据整体覆盖记录，未传的字段会被置为 null。**

**Agent 执行 `gm task edit` 前必须遵守以下规则：**
1. **先执行 `gm task info`** 获取任务完整数据，作为编辑的基准。
2. 编辑 JSON 的 `taskBaseInfo` 中**必须包含以下字段**（即使不修改也要从 info 中原样回传）：
   - `taskId`、`projectId`、`taskType`、`trainType`、`taskName`
   - `taskStatus`（关键！缺失会导致页面不可见）
   - `userId`（关键！缺失会导致任务变为"无主"，且后续无法编辑/删除）
   - `goodsId`、`gpuNum`、`imageId`、`imageVersion`、`personalDataPath`、`source`
3. 编辑 JSON 的 `taskCodeInfo` 中**必须包含以下字段**：
   - `taskId`、`userId`、`codeType`、`codeUrl`、`mainCodeUri`、`mainCodeType`
   - `startScript`、`isOpen`、`runParams`
4. 仅修改需要变更的字段值，其余字段从 `task info` 返回中原样保留。
5. 若 edit 返回 500 错误，**禁止盲目重试**，应先分析原因。
