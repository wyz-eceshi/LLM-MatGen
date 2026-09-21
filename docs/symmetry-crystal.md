# 空间群约束晶体生成

`symmetry-crystal` 用空间群、固定晶胞、整数化学计量和 Wyckoff 轨道生成三维原子晶体。它覆盖普通空间群 1–230，不处理磁空间群、部分占位、无序结构、分子晶体或二维层群。

生成结果只通过几何和格式检查。`geometry_status=passed` 表示配方、空间群、距离和往返检查合格；`relaxation_status=unknown` 表示尚未证明 DFT 优化稳定。

## 安装

显式轨道展开属于基础安装。自动搜索额外需要 PyXtal：

```bash
python -m pip install -e .
python -m pip install -e ".[symmetry]"
```

## 显式配方

下面的配方生成常规立方晶胞中的 Na₄Cl₄。输入的元素计数始终指最终导出的常规晶胞。

```yaml
schema: llm-matgen-symmetry-crystal
version: 1
mode: explicit

space_group:
  number: 225
  hall_number: null

cell:
  setting: conventional
  parameters: [5.6, 5.6, 5.6, 90, 90, 90]

composition:
  reduced: {Na: 1, Cl: 1}
  formula_units: 4
  formula_units_range: null

constraints:
  symprec: 0.001
  pair_min_A: {Na-Cl: 2.0}
  coordination: []

explicit:
  orbits:
    - element: Na
      representative_fractional: [0, 0, 0]
      expected_multiplicity: 4
      wyckoff: 4a
    - element: Cl
      representative_fractional: [0.5, 0.5, 0.5]
      expected_multiplicity: 4
      wyckoff: 4b
```

执行：

```bash
llm-matgen generate symmetry-crystal --recipe recipe.yaml --output-root output
```

未指定 `--format` 时，每个候选同时写出 POSCAR、CIF 和 MSON，并生成 `viewer.html`。显式模式会检查代表坐标展开后的轨道重复数、Wyckoff 字母、精确元素计数和空间群恢复；任一不符都会停止。

## 自动搜索配方

```yaml
schema: llm-matgen-symmetry-crystal
version: 1
mode: search
space_group: {number: 225, hall_number: null}
cell:
  setting: conventional
  parameters: [5.6, 5.6, 5.6, 90, 90, 90]
composition:
  reduced: {Na: 1, Cl: 1}
  formula_units: null
  formula_units_range: [1, 8]
constraints:
  symprec: 0.001
  pair_min_A: {Na-Cl: 2.0}
  coordination: []
search:
  seed: 20260921
  candidates: 5
  max_attempts: 5000
  fixed_orbits: []
```

程序先用 PyXtal 检查元素计数是否能由该空间群的 Wyckoff 重复数组合，从原子数较少的化学式规模开始，选择第一个兼容规模。`max_attempts` 是整次搜索的总预算。固定种子、固定配方和相同后端版本会给出一致的结构哈希。

候选按硬性检查、最小距离裕量、配位偏差和结构哈希排序。去重固定使用：

```text
primitive_cell=False, scale=False, attempt_supercell=False
ltol=0.2, stol=0.3, angle_tol=5
```

## 晶胞和 Hall 设置

程序按晶系检查长度和角度。例如立方必须满足 `a=b=c` 和三个直角，六方必须满足 `a=b`、`alpha=beta=90`、`gamma=120`。单斜默认唯一轴为 b；选择 a 或 c 时必须给 Hall 编号。R 点阵采用菱方轴设置时也必须给 Hall 编号。Hall 编号必须属于给出的国际空间群编号，并与选定轴向或 H/R 设置一致。

中心化平移直接由空间群对称操作产生，适用于 P、A、B、C、I、F 和 R 点阵，不使用写死的 FCC 平移表。

## 几何筛选与清单

每个合格候选记录：

- 空间群编号、符号和 Hall 编号；
- 晶胞设置、Wyckoff 轨道和中心化平移残差；
- 配方 SHA256、结构哈希、后端名称和版本；
- 随机种子、实际尝试次数、拒绝原因统计和候选排名；
- 分元素对最短距离、距离裕量、`geometry_status` 和 `relaxation_status`。

POSCAR 和 CIF 回读后必须与内存结构保持相同几何；MSON 必须保持完整语义。`pair_min_A` 和 `coordination` 属于材料配方，不写入通用核心默认值。

## 优化结果审计

```bash
llm-matgen structure relaxation-audit \
  --initial POSCAR \
  --final CONTCAR \
  --label failed \
  --reason "ISIF=3 后晶胞失控" \
  --output relaxation-audit.json
```

审计比较元素数、晶格、角度、体积、密度、空间群、扣除均匀晶胞变化后的原子位移、分元素对最短距离和近邻网络。异常代码包括：

- `large_cell_change`：任一晶格长度变化超过 10%；
- `runaway_cell`：任一晶格长度变化超过 25%，或体积变化超过 100%；
- `symmetry_changed`：初末空间群不同；
- `large_nonaffine_displacement`：出现明显非均匀重排；
- `contact_collapse`：形成异常短接触；
- `network_dilution`：近邻距离整体显著增大。

异常代码不替代人工判断。命令完整保留用户给出的 `label` 和 `reason`，也不会据此训练稳定性模型或生成、提交 DFT 任务。
