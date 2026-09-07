# LLM-MatGen

基于原项目 [Xinyang-Li666/LLM-MatGen](https://github.com/Xinyang-Li666/LLM-MatGen) 扩展，保留 MIT 许可证。

[下载 v0.2.1 安装包](https://github.com/wyz-eceshi/LLM-MatGen/releases/download/v0.2.1/llm_matgen-0.2.1-py3-none-any.whl) · [查看发布记录](https://github.com/wyz-eceshi/LLM-MatGen/releases/tag/v0.2.1)

项目名称已恢复为 LLM-MatGen，主命令为 `llm-matgen`。为兼容改名期间的使用方式，保留 `dft-structure-building` 命令和 `python -m dft_structure_building` 入口。


Provider-neutral crystal-structure generation for ten generator families.
LLM-MatGen exposes deterministic generators through CLI, Python, and MCP,
performs lightweight structural checks, and exports POSCAR, CIF, or LAMMPS
data files.

LLM-MatGen 是一个由任意兼容客户端驱动的材料晶体结构生成工具包。项目负责生成结构、记录参数与来源并执行默认轻量检查；模型与凭据由外部客户端管理。

> 当前版本：`0.2.1`。生成结果不代表结构已经完成弛豫，也不保证热力学稳定性、动力学稳定性、可合成性或发表质量。

## 功能

| 生成器  | CLI 名称           | 用途                                  |
| ---- | ---------------- | ----------------------------------- |
| 空位   | `vacancy`        | 按数量或浓度删除指定元素                        |
| 间隙原子 | `interstitial`   | 在候选间隙位点插入元素                         |
| 取代掺杂 | `doping`         | 用掺杂元素替换基体元素                         |
| 固溶体  | `solid-solution` | 按组成生成随机或 SQS 固溶体                    |
| 表面   | `surface`        | 按 Miller 指数生成带真空层的表面                |
| 晶界   | `grain-boundary` | 按旋转轴与角度生成晶界                         |
| 界面   | `interface`      | 匹配薄膜和基底并构造界面                        |
| 层错   | `stacking-fault` | 按滑移面和位移矢量生成层错                       |
| 位错   | `dislocation`    | 使用各向同性弹性位移场生成 edge、screw 或 mixed 位错 |

支持读取常见晶体结构文件，并可输出：

- VASP POSCAR（默认）
- CIF
- LAMMPS data

LAMMPS data 仅包含结构和原子类型，不包含势函数或计算参数。

## 安装

需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/wyz-eceshi/LLM-MatGen.git
cd LLM-MatGen
python -m pip install -e .
```

按需安装可选功能：

```bash
# MCP 服务
python -m pip install -e ".[mcp]"

# SQS 固溶体
python -m pip install -e ".[sqs]"

# 开发与测试
python -m pip install -e ".[dev]"
```

验证安装：

```bash
llm-matgen --help
llm-matgen generate --help
```

如果终端找不到 `llm-matgen`，可使用：

```bash
python -m llm_matgen --help
```

## 命令索引

| 命令 | 用途 |
| --- | --- |
| `llm-matgen search` | 搜索 Materials Project 结构 |
| `llm-matgen download` | 下载 Materials Project 结构 |
| `llm-matgen properties` | 查询材料性质 |
| `llm-matgen substrates` | 查询薄膜结构对应的基底候选 |
| `llm-matgen generate` | 使用十类生成器构造结构 |
| `llm-matgen check` | 执行轻量结构检查 |
| `llm-matgen export` | 转换 POSCAR、CIF 或 LAMMPS data |
| `llm-matgen db` | 管理本地数据库快照 |
| `llm-matgen mcp` | 启动 MCP 服务 |
| `llm-matgen config` | 管理非敏感的本地配置 |

使用 `llm-matgen <命令> --help` 查看对应参数。API Key 应通过环境变量或外部客户端管理，不写入本地配置。

## 已有材料先从 Materials Project 获取

默认模型工作流为：**MP 搜索 → 确定材料编号和晶型 → 下载母结构 → 生成 → 查看球棍模型**。
MCP 的 `search` 与 `download` 已接通 MPCollector；仅从环境变量读取 MP_API_KEY。
多个晶型匹配时先展示候选，不默认取第一条；没有匹配、接口不可用或缺少密钥时明确提示。
用户明确提供本地结构或要求离线时仍可直接使用本地输入。
表面、缺陷及吸附构型由下载的母结构继续生成，不将它们标注为 MP 原始结构。

```powershell
llm-matgen search --formula Si --limit 5
# 确定所需晶型后，使用搜索返回的材料编号下载：
llm-matgen download mp-149 --output-dir downloads
llm-matgen generate vacancy --input downloads/mp-149.cif --target-element Si --count 1 --output-root output --open
```

## 五分钟快速开始

准备一个包含 Si 的 CIF 文件，并将下面的 `path/to/si.cif` 替换为实际路径。该命令生成一个 Si 空位并写出 POSCAR：

```bash
llm-matgen generate vacancy \
  --input path/to/si.cif \
  --target-element Si \
  --count 1 \
  --variants 1 \
  --seed 7 \
  --format poscar \
  --output-root output
```

Windows PowerShell 可写为一行：

```powershell
llm-matgen generate vacancy --input path/to/si.cif --target-element Si --count 1 --variants 1 --seed 7 --format poscar --output-root output
```

对已有结构执行轻量检查：

```bash
llm-matgen check path/to/si.cif
```

转换输出格式：

```bash
llm-matgen export path/to/si.cif --format cif --output-root output
llm-matgen export path/to/si.cif --format lammps-data --output-root output
```

每次生成都会保存结构文件及可复现信息，包括实际参数、结构哈希、随机种子、来源和 manifest。

## 生成后直接查看结构

参考 dft-structure-viewer 的交互方式，全部十类生成流程现在默认在结果目录生成 `viewer.html`。
页面内置 3Dmol.js，可离线打开，支持球棍模型、旋转/缩放/平移、候选切换、晶胞显示、
1 起始原子编号、点击多选、原子坐标与固定状态、XY/XZ/YZ 视角及重置。
吸附结果还提供干净表面作为可切换的参考结构。

```powershell
llm-matgen generate vacancy --input Si.cif --target-element Si --count 1 --variants 3 --seed 7 --output-root output --open
```

`--open` 在生成成功后调用系统默认浏览器。未加此参数时仍生成页面，命令输出 JSON 中的
`viewer` 字段给出绝对路径。Python 返回对象提供 `viewer_path`；MCP `generate` 返回
查看页面的路径及 `artifact://` 引用，客户端可展示链接或打开 HTML。客户端是否自动内嵌取决于其 HTML 支持。

页面不改变坐标或原子顺序；连接仅按共价半径估计，暂不画跨晶胞边界的连接。
预览支持有序结构，单结构最多 20000 原子、整页最多 200000 原子；无法预览时保留结构文件，
并在 manifest 中记录原因。页面 SHA256 登记在 manifest 的 `artifacts` 中。

## 历史案例驱动的表面吸附

LLM-MatGen 是结构生成、校验、谱系、序列化和 DFT 交接的唯一生产核心。`vasp-structure-builder` 仅用于紧急情况下手工修订一个 POSCAR 原子坐标；其结果必须经 `llm-matgen revision import` 审计后才成为正式 revision。

```powershell
llm-matgen cases scan
llm-matgen cases status
llm-matgen cases query --features query.json
llm-matgen cases inspect REVISION_ID

llm-matgen generate adsorption `
  --slab clean-POSCAR `
  --adsorbate OH.xyz `
  --anchor 1 `
  --reference-axis 0,0,1 `
  --history-policy prefer `
  --surface-side top `
  --output-root output

llm-matgen revision import `
  --parent parent-POSCAR `
  --revised revised-POSCAR `
  --sidecar revised-POSCAR.revision.json
```

新 sidecar 会记录父结构绝对路径和哈希；该路径仍是可读普通文件且哈希一致时可省略 `--parent`。旧 sidecar、相对路径、符号链接、路径失效或哈希不一致时必须显式提供父 POSCAR。

`cases scan` 是唯一会访问远程案例目录的命令，且只允许只读扫描 `/public/home/zhangwy01/culuyao`。生成命令不会触发扫描。`prefer` 在索引不可用或无匹配时保留原因并退回几何生成；`require` 则明确失败。

吸附包固定包含候选 POSCAR、可完整恢复 site properties 的 MSON、RetrievalTrace、reference artifacts、manifest 和 DFT 比较矩阵。所有候选只表示“初始构型”。本项目不训练模型、不自动运行或提交 VASP、不生成 POTCAR、不从单个总能计算吸附能，也不宣称最低能位点。

## Materials Project

搜索、下载和性质查询需要用户自己的 Materials Project API Key。请只通过环境变量提供，不要写入代码、脚本、README 或提交记录。

Linux/macOS：

```bash
export MP_API_KEY="your-mp-api-key"
llm-matgen search --formula Si
llm-matgen download mp-149 --output-dir downloads
```

Windows PowerShell：

```powershell
$env:MP_API_KEY = "your-mp-api-key"
llm-matgen search --formula Si
llm-matgen download mp-149 --output-dir downloads
```

`.env.example` 只提供变量名和虚假占位符。真实密钥、下载缓存和生成结果不应进入公开仓库。

固溶体支持随机替换和可选 SQS。SQS 需要额外安装 `.[sqs]`，并可用 `--sqs-iterations N` 控制优化迭代次数。SQS 依赖缺失或优化失败时，命令会给出原因并建议使用 `--method random` 继续生成；不会静默回退。

## 任意客户端与 MCP

安装 MCP 可选依赖后启动服务：

```bash
llm-matgen mcp --output-root output
```

MCP 客户端负责选择模型、管理模型凭据并把自然语言请求转换为工具调用。LLM-MatGen 只提供确定性的结构生成、查询、检查和导出工具，因此不绑定特定模型厂商。

既有生成器的自然语言工作流示例见 [自然语言工作流](docs/examples/natural-language-workflows.md)，吸附命令见上节，MCP 接口说明见 [MCP 文档](docs/mcp.md)。

## Python 接口

生成器、参数模型、检查器和导出器均可直接作为 Python 库使用。公共接口与最小示例见 [API 文档](docs/api.md)。

## 默认轻量检查

生成流程默认检查格式可读性、晶格有效性、坐标有限性和异常近邻距离。检查结果分为提示、警告和错误：

- 警告用于提醒可能需要人工复核的结构，不等同于计算失败。
- 错误表示文件或几何存在明显问题。
- 轻量检查不是能量计算、结构弛豫、声子计算或稳定性判定。

## 可复现性

- 随机生成器接受 `seed`。
- manifest 记录输入、实际参数、结构哈希和输出文件。
- 达到 `max_structures` 时会截断候选并产生警告。
- 达到 `max_atoms` 时会拒绝继续构建超大结构。
- 相同输入、参数和随机种子应得到确定性结果。

## 责任边界

本项目只负责结构生成与轻量检查。用户需要自行负责：

- 选择合理的材料、晶格、滑移系、界面和边界条件；
- 执行结构弛豫、能量和稳定性计算；
- 选择势函数、DFT 参数及收敛标准；
- 判断结构是否适合实验、工程应用或学术发表。

## 文档

- [中文用户手册](docs/user-guide.zh-CN.md)
- [Python API](docs/api.md)
- [MCP 接口](docs/mcp.md)
- [自然语言工作流](docs/examples/natural-language-workflows.md)
- [项目设计](docs/llm-matgen-design.md)

## 测试

```bash
python -m pytest --import-mode=importlib -q
python -m compileall -q llm_matgen integrations
```
