<!-- CoPaper managed: @copaper/opencode; command=copaper-doctor; schemaVersion=1 -->
---
description: 诊断 CoPaper OpenCode 插件安装
---

运行此诊断，并原样显示输出：

!`bunx -p @copaper/opencode copaper-opencode doctor --format markdown 2>&1 || true`

这是一个便捷包装命令。权威诊断请在终端运行：
`bunx -p @copaper/opencode copaper-opencode doctor`

输出包含 agent profile diagnostics，用于检查 CoPaper subagent 注入、同名冲突和 permission profile 警告。

不要解读或修改诊断输出。
