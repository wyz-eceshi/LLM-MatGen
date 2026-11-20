# 自有请求合同 v1

这是本包定义的JSON到MatGen CLI适配合同，不是Harness官方JSON RPC。
请求UTF-8、最多64KiB。version=1；operation为下列白名单。
arguments键与真实CLI长选项一致（去掉--）。未知键、open、run-id、history-policy拒绝。
所有路径相对显式workspace或为其内部绝对路径；拒绝越界/链接逃逸/C0/Windows设备名/备用数据流。
输出必须新路径，保留原件。请求本身由调用者显式指定，不作为结构输入。

示例（均保存为单个JSON文件）：

```json
{"version":1,"operation":"build-info"}
```
```json
{"version":1,"operation":"inspect","arguments":{"input":"POSCAR"}}
```
```json
{"version":1,"operation":"validate","arguments":{"input":"POSCAR","min-distance":0.7}}
```
```json
{"version":1,"operation":"adjust-coordinate","arguments":{"input":"POSCAR","index":2,"delta-cartesian":[0,0,0.2],"reason":"用户校正第二原子","output":"POSCAR.revised.vasp"}}
```
```json
{"version":1,"operation":"revision-import","arguments":{"parent":"POSCAR","revised":"POSCAR.revised.vasp","sidecar":"POSCAR.revised.vasp.revision.json","output-root":"revisions-new"}}
```
```json
{"version":1,"operation":"export","paths":["POSCAR"],"arguments":{"format":["cif","mson"],"output-root":"export-new"}}
```
```json
{"version":1,"operation":"viewer","structures":["POSCAR","POSCAR.revised.vasp"],"output":"preview-new.html"}
```
```json
{"version":1,"operation":"generate","generator":"vacancy","arguments":{"input":"POSCAR","target-element":["Si"],"count":[1],"variants":1,"seed":7,"max-structures":1,"max-atoms":100,"output-root":"vacancy-new"}}
```

11种generator：vacancy、interstitial、doping、solid-solution、surface、grain-boundary、interface、stacking-fault、dislocation、adsorption、symmetry-crystal。
具体所需参数依赖安装wheel真实CLI，使用所选Python `-B -m llm_matgen generate <名称> --help` 查看。
可重复选项为数组，如target-element/count/miller/height/format；CLI逗号三元值以字符串传，例如miller:["1,0,0"]；
仅坐标修订的set-cartesian/set-fractional/delta-cartesian为3个数字数组。
开关如conventional使用JSON布尔值；数值不可NaN/Infinity。

index和吸附anchor严格从1开始，不再次加1或减1；本接口不接受主项目0-based原子索引。
吸附需slab周期结构、adsorbate独立XYZ/MSON分子、anchor；不得把分子当周期Structure。
adsorption的history-policy固定off，其余generator没有此选项。
导出格式支持poscar/cif/mson/lammps-data。symmetry自动搜索另需额外依赖，未默认安装。

退出码沿真实MatGen：0调用完成、2参数/结构校验问题、3部分失败、4系统错误。
任何退出码都不构成DFT稳定性或科学批准；阅读JSON、manifest、validation_issues和原错误。

MP联网入口仅本次真实用户明确要求搜索/下载时使用allow_mp=true，不把它当科学批准：
```json
{"version":1,"operation":"mp-search","allow_mp":true,"arguments":{"formula":"Si","limit":2}}
```
```json
{"version":1,"operation":"mp-download","allow_mp":true,"material_ids":["mp-149"],"arguments":{"output-dir":"mp-download-new"}}
```
缺授权默认拒绝。凭据由MatGen既有安全配置消费，本包不读取、保存或伪造密钥。
本次只离线核验授权门和真实CLI映射，未执行MP网络查询/下载。


## MOF 周期键分析与清洗

```json
{"version":1,"operation":"mof","subcommand":"bond-graph","arguments":{"input":"parent.cif","expand-symmetry":true,"bond-path":[1,77],"output-dir":"graph"}}
```
```json
{"version":1,"operation":"mof","subcommand":"clean","arguments":{"input":"parent.cif","expand-symmetry":true,"choose":["remove-guests"],"output-dir":"cleaned"}}
```
```json
{"version":1,"operation":"mof","subcommand":"graph-diff","arguments":{"parent":"parent.cif","revised":"cleaned/structure.cif","atom-map":"cleaned/sidecar.json","expand-symmetry":true}}
```

清洗时省略 choose 会列出候选并返回 2；可按 group:组装编号:构象 选择无序组。键图将临界距离的推断键单列，含 H 默认用 ±0.05 Å，其余默认用 ±0.15 Å，维数和配位统计基于确定键。bond-path 返回到终点任一周期像的最短路径，逐键记录跨胞方向。graph-diff 的删除项使用父编号，新增项使用子编号。

## 提取子网与放置客体

extract-subnet 接收 input、recipe 和新的 output-dir。recipe 可指定 dimension（层 2 或链 1）、miller（父胞指数）、primitive、basis（相对参考胞的幺模整数矩阵）、vacuum 和 component（从 1 开始）。省略 component 时，多解返回 2 并列出种子父编号。cut_edges 使用参考胞编号与周期像；cap 取 none、H2O 或 OH，实际封端还须给出 uncapped_charge 和 cap_distance。

scan-placement 另需 guest（XYZ 或 MSON）。site.kind 取 pore、atom、metal 或 bridge；位点编号从 1 开始，image 为整数晶胞平移。rotations 是 xyz 欧拉角（度），translations 是笛卡尔平移（Å）；桥联 azimuths 沿两目标距离的交圆取点。min_nonbond 是非键碰撞下限（Å）。top 指报告候选数，select 指显式选择排名。全部碰撞或多个可行候选未选择时返回 2。

## 桥联窗口和片段操作

窗口模式的请求示例（bridge-window.json 的 site 中省略 distances）：

```json
{"version":1,"operation":"mof","subcommand":"scan-placement","arguments":{"input":"host.cif","guest":"H2O.xyz","recipe":"bridge-window.json","distance-window":"1.90,2.60","output-dir":"bridge-window"}}
```
```json
{"site":{"kind":"bridge","atoms":[{"index":2},{"index":1,"image":[0,0,1]}]},"rotations":[[0,0,0],[90,0,0],[180,0,0]],"min_nonbond":1.65,"select":1}
```

窗口和两条目标距离互斥。先按碰撞下限筛选，再按配位距离与各自参考键长的绝对偏差之和升序排列；同偏差取最小非键距离较大者。参考值为两元素的 ASE 共价半径之和，也可用 prefer-distance 指定（Å）。端位替换沿用相同窗口排序，可在 scan 中配置 min_margin，在桥联扫描中配置 min_nonbond。无解返回 2，并报告已扫描位置中约束违反量最小者。

P3 子命令均接收 input、recipe、output-dir。node-substitute 的 recipe 例为 {"sites":[1,2,3],"species":["Co","Ni","Zn"]}，可用 fragments 或 coordination 筛选，max-structures 限制全部排列数；每种排列具有独立 sidecar 和对称轨道号。remove-fragment 使用 {"predicate":"element=O & neighbors(Co)>=1 & neighbors(C)=0","cap":"none"} 或 {"fragment":1}。H2O/OH 封端需 cap_distance、uncapped_charge。

replace-fragment 用 guest 指定 V2000 SDF。整配体示例为 {"mode":"terminal","metal":3,"anchor":2}；有多个端配体时须选 fragment。环替换 recipe 中 ring 为依次相连的父环编号，template_ring 为对应模板重原子编号，anchors 为保留的父原子，ports 为新环与锚点结合的位置（从 1 起）。orientation 可取 lowest-cost，明确选取通过碰撞检查后拟合代价最小的构型。

## 取层与片段操作的完整请求

从清洗后的 Co-HHTP 取层，在项目中提供取层配方 Co-HHTP.json。

```json
{"version":1,"operation":"mof","subcommand":"extract-subnet","arguments":{"input":"cleaned/structure.cif","recipe":"Co-HHTP.json","cap":"none","output-dir":"layer"}}
```

删除层上与 Co 相连且没有 C 邻居的轴向 O。remove-O.json 中填写上述 predicate。

```json
{"version":1,"operation":"mof","subcommand":"remove-fragment","arguments":{"input":"layer/structure.cif","recipe":"remove-O.json","cap":"none","output-dir":"layer-63"}}
```

在金属位点上枚举 Co、Ni、Zn。metals.json 可使用 {"sites":[1,2,3],"species":["Co","Ni","Zn"]}。

```json
{"version":1,"operation":"mof","subcommand":"node-substitute","arguments":{"input":"layer-63/structure.cif","recipe":"metals.json","max-structures":6,"output-dir":"permutations"}}
```

用咪唑替换端位水。terminal.json 可使用 {"mode":"terminal","metal":3,"anchor":2}。

```json
{"version":1,"operation":"mof","subcommand":"replace-fragment","arguments":{"input":"aqua-framework.cif","guest":"imidazole.sdf","recipe":"terminal.json","distance-window":"1.90,2.60","output-dir":"imidazole"}}
```

## 取代基、封端与双锚点

请参阅[命令行示例](../../../docs/mof-hetero-使用说明.md)。请求的 arguments 使用对应长参数名，重复选项写成数组。

## 异质结

沿 b 法线测量边界层、终止层和配位。

```json
{"version":1,"operation":"hetero","subcommand":"measure","arguments":{"input":"inputs/TiO2.cif","axis":"b"}}
```

横向构建。loose.json 给出法线、晶胞长度、扫描网格和成键窗口。

```json
{"version":1,"operation":"hetero","subcommand":"build","arguments":{"mode":"lateral","substrate":"inputs/Cu2O.cif","film":"inputs/CuO.cif","recipe":"loose.json","fix-registry":"dy=0.40","output-dir":"loose"}}
```

垂直构建。vertical.json 指定整数超胞搜索范围、配准、间隙与真空。

```json
{"version":1,"operation":"hetero","subcommand":"build","arguments":{"mode":"vertical","substrate":"inputs/CeO2.vasp","film":"inputs/ZIF67.vasp","accept-strain":0.08,"recipe":"vertical.json","output-dir":"vertical"}}
```

回读产物，按实际周期像分别统计两个界面，并与两块输入核对。

```json
{"version":1,"operation":"hetero","subcommand":"check","arguments":{"input":"loose/structure.cif","sidecar":"loose/sidecar.json","substrate":"inputs/Cu2O.cif","film":"inputs/CuO.cif"}}
```

单独检查原始成品时，用 split 给出第一块的原子数，用 axis 指定法线。逐元素对报告碰撞下限。优先采用配套 sidecar 的构建规则，其次采用显式下限（例如 pair-floor 数组中的 "Cu-O=1.90"）。默认取两块中最短的同类确定键。缺少同类键时取共价半径和乘以 radius-factor，默认 0.75。
