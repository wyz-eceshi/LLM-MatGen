---
name: llm-matgen
description: 使用 MatGen 生成材料结构、检查本地结构、可追溯地修订单原子坐标，并读取实际构建身份。
---

# LLM-MatGen

使用实际公开 CLI 或当前服务明确列出的工具。生成命令包括
`vacancy`、`interstitial`、`doping`、`solid-solution`、`surface`、
`grain-boundary`、`interface`、`stacking-fault`、`dislocation`、`adsorption` 和
`symmetry-crystal`。

生成时给出输入路径（界面使用 `film` 和 `substrate`）、明确参数、指定输出目录，
随机生成给出 `seed`。吸附使用 `--slab`、`--adsorbate` 和从 1 开始的 `--anchor`。
只有用户授权历史案例读取时才启用相应历史策略；纯本地生成可用 `--history-policy off`。
结构生成、轻量检查通过与 revision 导入均不等于科学批准，后续计算需要独立审查。

本地检查与修订由 MatGen 自身实现，不依赖旧 builder：

```bash
llm-matgen structure inspect --input POSCAR
llm-matgen structure validate --input POSCAR --min-distance 0.7
llm-matgen structure adjust-coordinate --input POSCAR --index 2 --delta-cartesian 0 0 0.2 --reason "人工校正" --output POSCAR.revised
llm-matgen structure relaxation-audit --initial POSCAR --final CONTCAR --label failed --reason "晶胞失控" --output relaxation-audit.json
llm-matgen revision import --revised POSCAR.revised --sidecar POSCAR.revised.revision.json --output-root revisions
llm-matgen build-info
```

空间群晶体使用工作目录内的版本化配方：

```bash
llm-matgen generate symmetry-crystal --recipe recipe.yaml --output-root output
```

显式模式可直接使用；自动搜索需要 `llm-matgen[symmetry]`。先核对晶胞制度、Hall
设置、最终常规晶胞元素数、轨道重复数、距离和配位限制。保持空间群不等于优化稳定；
只有几何检查合格时标记 `geometry_status=passed`，弛豫前始终保留
`relaxation_status=unknown`。不得把候选描述成已通过 DFT 稳定性验证。

编号从 1 开始；`--set-cartesian X Y Z`、`--set-fractional U V W` 与
`--delta-cartesian DX DY DZ` 恰选一种，`--reason` 必填且不能为空。
输入、已有输出和 sidecar 不得覆盖；不使用强制覆盖参数。检查输出 JSON 中的
`validation_issues`；`inspect` 汇总问题，`validate` 在发现问题时返回非零。
修订默认生成 `<output>.revision.json`，可用 `--sidecar` 指定另一个新文件。
核对其中父/输出 SHA256、原子编号、元素、前后坐标、位移、原因与时间。
revision import 保留旧 v1 sidecar 兼容性，并要求输入与输出位于当前工作目录内；
旧 sidecar 未记录可验证父路径时显式提供 `--parent`。

主项目通过 `llm_matgen.build_identity.get_build_info()` 或 `llm-matgen build-info`
读取包版本、上游提交、本地变更哈希及 `verification`。仅 `verified` 表示实际包源码
与交付证据一致；`modified` 或 `unavailable` 需要重新核对来源。
这些新能力是 CLI/Python 接口；不得假称它们已注册为 MCP 工具。
保留生成返回的 manifest、结构与预览文件；失败时依据实际错误处理，不伪造成功产物。
