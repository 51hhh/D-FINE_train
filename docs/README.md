# Docs Index

本目录用于沉淀当前项目的训练研究、GradMotion 平台使用方法与历史排障记录，避免信息散落在对话和临时日志中。

## 目录

- `gradmotion/cli-usage.md`
  - gm CLI 的可用命令、正确 base_url、关键字段与安全使用方式。
- `gradmotion/cloud-task-failures.md`
  - 历史云任务失败原因、已确认根因、后续避免方式。
- `gradmotion/smoke-test-plan.md`
  - 当前最小 smoke test 方案、任务 payload 设计与验证目标。
- `dfine/oa-ap-findings.md`
  - 本地日志、历史模型、官方 issue 调研后的核心结论。
- `dfine/no-A-roadmap.md`
  - 不再使用 Method A 后的后续训练/评估路线。

## 当前共识

- OA 测试负样本必须与训练负样本隔离。
- 当前用户不希望继续使用 Method A（外部 negative images 直接并入训练）。
- 云平台已确认可用，但需要用正确 API 入口与正确的 `/personal` 路径。
- 正式长训练前，先做最小 smoke test，验证 Git checkout、工作目录、数据挂载与关键文件路径。
