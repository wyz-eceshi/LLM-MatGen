---
name: llm-matgen
description: 用本地MatGen生成材料和吸附候选结构、处理 MOF 键图与片段、构建并核对横向或垂直异质结、检查修订原子、导出格式并制作离线球棍预览。
user-invocable: true
---

# MatGen 原生文件技能

用户可以用 `/llm-matgen` 调用。本技能适配 DeepSeek Harness filesystem Skill，
不注册MCP、不提供HTTP服务、不需要DeepSeek API密钥。不得修改Harness配置。
先读 [请求说明](references/requests.md)。资源路径相对本Skill目录解析。

从本Skill目录的 `runtime.json` 读取用户安装时明确选定的Python。
将请求保存为UTF-8 JSON，调用（PowerShell；将变量换成实际绝对路径）：

```powershell
& $Python -B "$SkillDir/scripts/invoke.py" --workspace $Project --request $RequestJson
```

工作项目必须显式指定；所有输入输出都留在该目录内。不调用任意shell回调。
先运行build-info，并核JSON `verification` 为 `verified`；解释器、包或来源绑定变化再重核。
默认保持历史策略off；MP搜索/下载只在本次用户明确请求后设置allow_mp=true；禁止擅自联网，不开放SSH入口。
结构检查成功、候选生成、revision导入不等于科学批准，不伪造审查人或批准字段。
`inspect` 即使exit0也要阅读 `validation_issues`；`validate` 有问题返回非零。
生成保留manifest、结构和HTML。大于2000原子的预览仅提示，结构仍可保存。
已有主项目结构必须经其正式revision API保存，本离线Skill不得直接改科学记录。


## MOF 与通用异质结（0.4.0.dev1）

使用 `operation: "mof"` 和 `subcommand` 选择周期键图与沿键路径、无序清洗、键图比较、取层或链、客体扫描、金属节点替换、片段删除或模板替换。位点编号从 1 开始。桥联和端位窗口支持参考键长及碰撞下限。

换环可逐支保留或删除父环取代基；删配体可按电荷和原羧酸位点联合封端。放置支持单供体或双锚点、显式移走客体片段和按原子指定排斥距离。用 sidecar 核对原子去向、组成、形式电荷与键变化。

使用 `operation: "hetero"` 测量板片，按 vertical 或 lateral 构建候选，或从坐标核对两个界面。检查报告包含逐元素对距离下限和来源，以及每个界面的最近距离和周期接触数。生成目录带结构、预览和记录原子映射的 sidecar。退出码 2 时读取候选列表、违规列表或参数错误。各操作的请求字段和配方示例见 [请求说明](references/requests.md)。
