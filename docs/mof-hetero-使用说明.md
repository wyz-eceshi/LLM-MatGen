# MOF 与异质结建模 · 0.4.0.dev1

在安装本预发布 wheel 的 Python 环境中运行。下表给出 `<解释器> -m llm_matgen` 后面的完整参数示例。先用同一解释器核对 `llm_matgen.__file__` 和 `__version__`。位点从 1 编号，长度单位为 Å，配方为 UTF-8 JSON。

| 操作及用途 | 常用参数 | 示例 |
|---|---|---|
| 周期键图与沿键路径 | `--expand-symmetry` 展开 CIF；`--bond-path` 接收两个位点 | `mof bond-graph --input aqua.cif --bond-path 3 6 --output-dir graph` |
| 清洗无序结构与客体 | `--choose` 选构象或移除游离客体 | `mof clean --input 1.cif --expand-symmetry --choose remove-guests --output-dir clean` |
| 比较增删键、配位和维数 | `--parent`、`--revised`、`--atom-map` 指向匹配的父子结构和映射 | `mof graph-diff --parent 1.cif --expand-symmetry --revised clean/structure.cif --atom-map clean/sidecar.json` |
| 提取层或链 | `--recipe` 指定维数、分量和真空；`--cap` 指定封端 | `mof extract-subnet --input clean/structure.cif --recipe Co-HHTP.json --cap none --output-dir layer` |
| 在孔、金属位或桥联位放置分子 | `--guest` 给出模板；桥联可用 `--distance-window`、`--prefer-distance` | `mof scan-placement --input host.cif --guest H2O.xyz --recipe bridge-window.json --distance-window 1.90,2.60 --output-dir bridge` |
| 枚举金属节点替换 | 配方给出 `sites`、`species`；`--max-structures` 限制枚举总数 | `mof node-substitute --input layer-63.cif --recipe metals.json --max-structures 6 --output-dir permutations` |
| 按片段编号或邻居条件删除 | 配方给出 `fragment` 或 `predicate`；可选 OH、H₂O 封端 | `mof remove-fragment --input layer/structure.cif --recipe remove-O.json --cap none --output-dir cut` |
| 替换环或端位配体 | `--guest` 使用 V2000 SDF；配方指定锚点和模式 | `mof replace-fragment --input aqua.cif --guest imidazole.sdf --recipe terminal.json --output-dir replaced` |
| 测量板片、边界层和配位指纹 | `--axis a/b/c` 指定测量法线 | `hetero measure --input TiO2.cif --axis b` |
| 构建横向结 | `--mode lateral`；`--fix-registry` 冻结相对平移 | `hetero build --mode lateral --substrate Cu2O.cif --film CuO.cif --recipe loose.json --fix-registry dy=0.40 --output-dir lateral` |
| 构建垂直结 | `--mode vertical`；`--accept-strain` 显式放宽主应变上限 | `hetero build --mode vertical --substrate CeO2.vasp --film ZIF67.vasp --recipe Ce-ZIF.json --accept-strain 0.08 --output-dir vertical` |
| 复核界面碰撞与接触 | 用 `--sidecar` 回读产物，或用 `--split` 和 `--axis` 标明原件的块边界；`--pair-floor` 可重复 | `hetero check --input tight.cif --split 72 --axis a` |

**读取结果。** 成功返回 0。构建会写入新的输出目录，包含 POSCAR、带键表的 CIF、预览和 sidecar。核对父结构摘要后，可以用 sidecar 将子位点追溯到父位点。界面检查分别报告两侧的接触数和最近距离。默认碰撞下限取块内最短同类确定键；缺少同类键时取共价半径和的 0.75 倍，可用 `--radius-factor` 调整。

**退出码 2。** 先看输出原因。构象或候选未选定时明确选择后重跑；应变超限或全部碰撞时查看最接近候选或取舍前沿；参数冲突和输出目录已存在时修正请求。组合验证的组成和原子数只用于核对候选模型。

技能请求和配方字段见[请求示例](../skills/llm-matgen/references/requests.md)。

## 取代基、封端与双锚点

换环时以 `--substituent keep:<编号>` 或 `drop:<编号>` 声明外接支链，用 `--map 父环位:模板环位` 指明连接位置。删除配体时，`--cap-plan auto-charge` 按空出的配位位点和电荷分配封端。

指定供体时用 `--site donor --host-atom <编号> --guest-anchor <编号> --distance-window 下限,上限`。用 `--displace <片段名>` 先移除配方声明的片段。需要同时满足两个窗口时，两次传入 `--anchors 客体编号~宿主编号:下限,上限`。每一步的组成、电荷与键图差异都可以从 sidecar 中核对。
