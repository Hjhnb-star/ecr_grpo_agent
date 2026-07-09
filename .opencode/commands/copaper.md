<!-- CoPaper managed: @copaper/opencode; command=copaper; schemaVersion=1 -->
---
description: 显示 CoPaper 项目仪表盘
---

调用 `copaper_dashboard` 工具。每次调用 CoPaper 工具后，必须在最终给用户的回复中展示工具返回的人类可读 markdown 正文和表格，默认不要展示 fenced JSON block。不要只总结工具结果或用摘要替代工具输出。用户明确要求 JSON、debug 或原始输出时才展示 JSON；用户要求完整工具输出时也可包含 JSON。

如果 Dashboard 显示项目需要初始化，必须使用 question tool 询问是否初始化，并收集项目名称和研究领域、确认初始化细节。如果缺少项目名称或研究领域，继续使用 question tool 询问缺失字段。只有用户明确确认且项目名称和研究领域都已知后，才可调用 `copaper_init_apply` 工具。不要在用户未确认时调用初始化工具。

如果 Dashboard 显示项目已就绪，先调用 `copaper_artifact_status` 展示只读工件状态、就绪证据和建议；再调用 `copaper_workflow_status` 展示进度、阶段和下一步，并调用 `copaper_workflow_log` 查看最近工作流记录。

如果用户表达「找相关工作 / 跑 relatedwork / 检索文献 / 下载 PDF / 写摘要 / 注册摘要 / 建跨文献索引 / 同步 BibTeX / 清理文献条目」等意图，或想推进 literature 阶段，请引导用户使用专用的 `/copaper-relatedwork` 斜杠命令，不要在本模板中直接调用 relatedwork 工具。

分派工作时使用专用 CoPaper subagents：`@copaper-coordinator`、`@copaper-storyline`、`@copaper-writer`、`@copaper-reviewer`、`@copaper-recorder` 和 `@copaper-literature`。如果出现 agent profile warning 或 diagnostic，不要忽略；需要时先运行 `/copaper-doctor` 再委派。

`copaper_artifact_status` 是只读工具。不得直接写入状态、安装技能、运行 relatedwork/checker/report/git 命令，或在没有单独明确用户请求和确认时改变阶段。

如果用户明确要求记录工件就绪度，必须先复述 artifact、status、confidence 和 reason，然后等待用户确认。只有用户明确确认后，才可调用 `copaper_artifact_record`。该工具会写入 artifact readiness state 并追加事件，但不会自动推进 phase，也不会运行 checker/relatedwork/report/git/skills 动作。

在修改阶段状态前，必须复述阶段、状态，以及 status 为 `skipped` 时的原因，然后等待用户确认。只有用户明确确认后，才可调用 `copaper_workflow_set_phase`；不得进行未经确认的阶段状态修改。

如果工具不可用，请告诉用户：
- 运行 `/copaper-doctor` 进行诊断
- 或在终端运行：`bunx -p @copaper/opencode copaper-opencode doctor`

不要编造 CoPaper 状态。只能使用工具返回的信息，同时按上文要求默认省略 fenced JSON block。
