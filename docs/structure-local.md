# 本地结构与构建身份

`llm-matgen structure inspect|validate|adjust-coordinate|relaxation-audit` 是 MatGen 自身的公开入口。
`python -m llm_matgen` 具有同样的命令树。结构处理使用 ASE；检查最短原子距离使用
逐步扩大截断的邻居列表，不建立全原子平方距离矩阵，默认重叠报警阈值为 0.7 Å。

## 单原子修订

必填 `--input`、`--output`、`--index`（从 1 开始）、`--reason`。
以下恰选一种，均接受三个数字：

- `--set-cartesian X Y Z`：设置笛卡尔坐标，单位 Å。
- `--set-fractional U V W`：设置分数坐标，不主动折回晶胞。
- `--delta-cartesian DX DY DZ`：增加笛卡尔位移，单位 Å。

默认 sidecar 是 `<output>.revision.json`，也可用 `--sidecar` 指定新路径。
任何已有输出或 sidecar 均拒绝覆盖。先读取输入字节快照，绑定父哈希并解析同一快照；
写临时结构后回读核对原子顺序、晶胞、未选原子和约束，再发布结构与 sidecar。
发布要求目标文件系统支持同目录硬链接；运行时发布失败会清理本次创建的结果并返回非零。
两个文件不能作为一个文件系统事务同时发布，进程被强制终止时仍应检查文件对是否完整，
仅有结构文件不能视为成功修订。

sidecar 沿用 `vasp-structure-revision` v1，包含 `parent_sha256`、`output_sha256`、
`index_1based`、`element`、前后两种坐标、笛卡尔位移、原因、带时区的 UTC 时间和父路径。
`revision import` 验证旧、新 sidecar 后建立新的 revision，保留
`status=formal_structure_revision` 兼容字段，并明确 `scientific_approval=required`。
几何检查或导入成功不能代替科学批准。单原子坐标没有实际变化时，既有 importer 会拒绝导入。

## 主项目读取构建身份

```python
from llm_matgen.build_identity import get_build_info

identity = get_build_info()
assert identity['verification'] == 'verified'
```

等效命令为 `llm-matgen build-info`，stdout 是 JSON。接口无需 Git、联网或读案例库。
随包资源 `llm_matgen/_build_identity.json` 记录固定上游归档的 SHA256、完整上游提交、
上游生产包逐文件 SHA256 和本次交付的生产包逐文件 SHA256。安装 wheel 后该资源仍随包存在。

| 字段 | 含义 |
| --- | --- |
| `package_version` | 实际导入包的 `__version__`；当前源码为 0.3.0 |
| `upstream_commit` | 已实际下载、逐文件比对的上游基线，不等于本地 Git HEAD |
| `upstream_archive_sha256` | 本次固定提交源码 ZIP 的 SHA256 |
| `source_sha256` | 当前两个生产包文件路径及内容哈希表的 SHA256 |
| `local_changes_sha256` | 当前生产包相对上游的新增、修改、删除列表的 SHA256 |
| `local_changed_files` | 上述差异涉及的包内文件 |
| `verification` | `verified`：当前源码与交付证据一致；`modified`：交付后变动；`unavailable`：资源缺失 |
| `resource` | 当前环境实际读取的资源路径 |

哈希表以排序键的紧凑 JSON 编码后取 SHA256。范围为 `llm_matgen` 与
`dft_structure_building` 两个生产包，排除字节码、`__pycache__` 和身份资源自身，
包含预览静态资源；文档、测试、环境和 Git 状态不计入运行包身份。它是源码完整性与
来源记录，不是签名或科学审批。后续修改源码必须重新核对并生成构建证据，不能仅换提交常量。

既有 `GenerationService`、生成 CLI、吸附专用入口、预览 HTML 与 revision importer 保持可用。
这四个结构子命令和构建身份命令没有新增 MCP 注册；客户端应使用实际存在的 CLI/Python 接口。

`relaxation-audit` 对比初末结构的元素数、晶胞、空间群、非均匀位移和近邻网络，
并原样保存人工 `label` 与 `reason`。详细阈值和异常代码见
[空间群约束晶体生成](symmetry-crystal.md#优化结果审计)。
