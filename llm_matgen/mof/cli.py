"""MOF 三个基础操作的命令行参数。"""
import json
from pathlib import Path


HELP = {"node-substitute":"枚举金属节点排列","remove-fragment":"删除片段或匹配位点","replace-fragment":"替换锚定环或端位配体","extract-subnet":"按周期连通性提取层或链","scan-placement":"枚举周期位置和取向放置客体","bond-graph":"构建周期键图并分析片段和配位","graph-diff":"按位点映射比较父子周期键图",
        "clean":"报告或按显式选择清洗无序位点与客体"}


def configure(parser, operation):
    parser.add_argument("--expand-symmetry",action="store_true",help="展开 CIF 中的全部等价位置")
    parser.add_argument("--tolerance",type=float,default=1.15,help="共价半径和乘数")
    parser.add_argument("--metal-tolerance",type=float,default=1.25,help="金属与 O/N 的半径和乘数")
    parser.add_argument("--h-window",type=float,default=.05)
    parser.add_argument("--metal-window",type=float,default=.15)
    parser.add_argument("--other-window",type=float,default=.15)
    if operation in {"extract-subnet","scan-placement","node-substitute","remove-fragment","replace-fragment"}:
        parser.add_argument("--recipe",required=True,help="UTF-8 JSON 参数文件")
    if operation in {"extract-subnet","remove-fragment"}:
        parser.add_argument("--cap",choices=("none","OH","H2O"))
    if operation=="remove-fragment":
        parser.add_argument("--cap-plan",metavar="方案",help="按电荷自动封端 auto-charge，或 site:金属编号=OH,H2O;…")
    if operation=="scan-placement":
        parser.add_argument("--site",choices=("donor",),help="将客体锚点放在指定宿主原子附近")
        parser.add_argument("--host-atom",metavar="编号或判据")
        parser.add_argument("--guest-anchor",type=int,metavar="客体原编号")
        parser.add_argument("--displace",action="append",metavar="客体片段编号",help="放置前移除指定的完整片段；多个片段逐项指定")
        parser.add_argument("--anchors",nargs=2,metavar=("客体~宿主:lo,hi","客体~宿主:lo,hi"),help="同时满足两组配位窗口，编号从 1 开始")
        parser.add_argument("--exclude-contact",action="append",metavar="选择器:距离",help="按最短周期像排除近接，可重复")
    if operation=="node-substitute":
        parser.add_argument("--max-structures",type=int,default=1000)
    if operation in {"scan-placement","replace-fragment"}:
        parser.add_argument("--distance-window",metavar="lo,hi",help="端位、桥联或供体配位距离闭区间（Å）")
        parser.add_argument("--prefer-distance",type=float,help="窗口模式参考键长（Å），默认取共价半径之和")
    if operation in {"scan-placement","replace-fragment"}:
        parser.add_argument("--guest",required=True,help="放置使用 XYZ/MSON；配体替换使用 SDF V2000")
    if operation=="replace-fragment":
        parser.add_argument("--substituent",action="append",metavar="keep:编号|drop:编号",
                            help="保留或删除父环的外接取代基，指定直接成键的环外原子；可重复")
        parser.add_argument("--map",dest="ring_map",action="append",metavar="父环编号:模板环编号",
                            help="保留取代基时，明确给出新旧环位的对应关系；可重复")
    if operation=="graph-diff":
        parser.add_argument("--parent",required=True)
        parser.add_argument("--revised",required=True)
        parser.add_argument("--atom-map",help="子→父映射 JSON，或带该映射的 sidecar")
    else:
        parser.add_argument("--input",required=True)
        parser.add_argument("--output-dir",help="写入新目录；省略时只打印分析")
    if operation=="clean":
        parser.add_argument("--choose",action="append",help="可重复，例如 remove-guests 或 group:A")
    if operation=="bond-graph":
        parser.add_argument("--bond-path",nargs=2,type=int,metavar=("起点","终点"))
    parser.set_defaults(_handler=run)


def _paired_bond_model(poscar, sidecar, metadata):
    """用同一次导出的显式键表解释已匹配摘要的 POSCAR。"""
    import numpy as np
    from .cif import read_model
    from .sidecar import sha256
    cif = Path(sidecar).with_name("structure.cif")
    if cif.is_symlink() or not cif.is_file() or sha256(cif) != metadata["product_sha256"]:
        raise ValueError("缺少与 sidecar 摘要一致的 structure.cif，无法确认 POSCAR 的周期键")
    model = read_model(cif)
    structure = read_model(poscar).structure
    a,b = model.structure,structure
    if (model.explicit is None or a.species != b.species or
            not np.allclose(a.lattice.metric_tensor,b.lattice.metric_tensor,rtol=1e-8,atol=1e-7)):
        raise ValueError("配套 CIF 和 POSCAR 的键表、位点顺序或晶胞不匹配")
    delta = a.frac_coords-b.frac_coords
    if not np.allclose(delta-np.rint(delta),0,atol=1e-8):
        raise ValueError("配套 CIF 和 POSCAR 的坐标不匹配")
    return model


def run(args):
    from .cif import read_model
    from .clean import clean
    from .diff import compare
    from .graph import BondGraph,bond_path
    from .sidecar import publish,sha256
    def load(path):
        return BondGraph(read_model(path,args.expand_symmetry),args.tolerance,args.metal_tolerance,args.h_window,args.metal_window,args.other_window)
    params = dict(expand_symmetry=args.expand_symmetry,tolerance=args.tolerance,metal_tolerance=args.metal_tolerance,h_window=args.h_window,metal_window=args.metal_window,other_window=args.other_window)
    if args.operation=="graph-diff":
        mapping = None
        parent_options = {}
        parent_expand = args.expand_symmetry
        paired_model = None
        frame = None
        if args.atom_map:
            data = json.loads(Path(args.atom_map).read_text(encoding="utf-8-sig"))
            if isinstance(data,dict) and "atom_map" in data:
                if "parent_sha256" in data and sha256(args.parent)!=data["parent_sha256"]:
                    raise ValueError("映射与父文件摘要不符")
                if "product_sha256" in data:
                    digest = sha256(args.revised)
                    if digest == data.get("poscar_sha256"):
                        paired_model = _paired_bond_model(args.revised,args.atom_map,data)
                    elif digest != data["product_sha256"]:
                        raise ValueError("映射与子文件摘要不符")
                recorded = data.get("params",{})
                parent_expand = recorded.get("expand_symmetry",args.expand_symmetry)
                parent_options = {key:recorded.get(key,params[key]) for key in
                                  ("tolerance","metal_tolerance","h_window","metal_window","other_window")}
                mapping = data["atom_map"]
                frame = data.get("comparison_frame")
            else:
                mapping = data
        child = BondGraph(paired_model) if paired_model is not None else load(args.revised)
        parent = BondGraph(read_model(args.parent,parent_expand),**parent_options) if parent_options else load(args.parent)
        if frame is not None:
            from .subnet import subnet_diff
            report = subnet_diff(parent,child,mapping,frame)
        else:
            report = compare(parent,child,mapping)
    else:
        graph = load(args.input)
        assumptions = []
        if args.operation in ("extract-subnet","scan-placement","node-substitute","remove-fragment","replace-fragment"):
            recipe=json.loads(Path(args.recipe).read_text("utf-8-sig"))
            if any(o!=1 for o in graph.model.occupancy):
                raise ValueError("请先清洗部分占位结构")
            if args.operation=="scan-placement":
                site=recipe.setdefault("site",{})
                if not isinstance(site,dict):
                    raise ValueError("site 须为包含 kind 的对象")
                if args.site is not None:
                    if site.get("kind",args.site)!=args.site:
                        raise ValueError("配方与命令行的位点类型冲突")
                    site["kind"]=args.site
                if args.anchors is not None:
                    if "anchors" in recipe:
                        raise ValueError("配方与命令行重复指定 anchors")
                    recipe["anchors"]=args.anchors
                if "anchors" in recipe:
                    if site.setdefault("kind","donor")!="donor":
                        raise ValueError("双锚点仅适用于 donor 位点")
                for key in ("host_atom","guest_anchor","displace","exclude_contact"):
                    value=getattr(args,key)
                    if value is not None:
                        if key in recipe:
                            raise ValueError(f"配方与命令行重复指定 {key}")
                        if key=="displace":
                            value=value[0] if len(value)==1 else {"fragments":value}
                        recipe[key]=value
                    if key in recipe and site.get("kind")!="donor":
                        raise ValueError(f"{key} 需要 donor 位点")
                if not site.get("kind"):
                    raise ValueError("请指定放置位点类型")
            if args.operation=="replace-fragment":
                for key, values in (("substituent",args.substituent),("map",args.ring_map)):
                    if values is not None:
                        if key in recipe:
                            raise ValueError(f"配方与命令行重复指定 {key}")
                        recipe[key]=values
                    if key in recipe and recipe.get("mode")!="ring":
                        raise ValueError("取代基和环位映射参数仅用于换环模式")
            if args.operation in {"scan-placement","replace-fragment"} and (args.distance_window is not None or args.prefer_distance is not None):
                settings=recipe
                if args.operation=="replace-fragment":
                    if recipe.get("mode")!="terminal":
                        raise ValueError("配位距离参数仅用于端位替换")
                    settings=recipe.setdefault("scan",{})
                if args.distance_window is not None:
                    if "distance_window" in settings:
                        raise ValueError("距离窗口重复")
                    settings["distance_window"]=[float(v) for v in args.distance_window.split(",")]
                if args.prefer_distance is not None:
                    if "prefer_distance" in settings:
                        raise ValueError("参考距离重复")
                    settings["prefer_distance"]=args.prefer_distance
            if args.operation=="node-substitute":
                from .substitute import enumerate_nodes,generate
                from .sidecar import write_json
                from tempfile import TemporaryDirectory
                if args.max_structures<1:
                    raise ValueError("max-structures 须为正整数")
                report,choices=enumerate_nodes(graph,recipe,args.max_structures)
                if not choices:
                    print(json.dumps(report,ensure_ascii=False))
                    return 2
                if not args.output_dir:
                    raise ValueError("需要新的输出目录")
                dest=Path(args.output_dir).resolve()
                if dest.exists():
                    raise ValueError("输出目录已存在")
                dest.parent.mkdir(parents=True,exist_ok=True)
                with TemporaryDirectory(dir=dest.parent) as tmp:
                    batch=Path(tmp)/"batch"
                    batch.mkdir()
                    for number,assignment in enumerate(choices,1):
                        child,mapping=generate(graph,report["sites"],assignment)
                        selected=report["assignments"][number-1]
                        publish(batch/f"candidate-{number:03}",args.input,args.operation,graph,child,mapping,
                                dict(params,recipe=recipe,max_structures=args.max_structures,assignment=assignment,
                                     sites=report["sites"]),
                                selected,["仅改变选定节点的元素"])
                    write_json(batch/"report.json",report)
                    batch.rename(dest)
                print(json.dumps(report,ensure_ascii=False))
                return 0
            if args.operation=="remove-fragment":
                from .fragment_ops import remove
                if args.cap is not None:
                    recipe["cap"]=args.cap
                if args.cap_plan is not None:
                    if "cap_plan" in recipe:
                        raise ValueError("配方与命令行重复指定 cap_plan")
                    recipe["cap_plan"]=args.cap_plan
                if "cap_plan" in recipe:
                    from .ledger import source_charge
                    if "cap" in recipe:
                        raise ValueError("cap 和 cap-plan 只能选择一项")
                    declared=recipe.setdefault("formal_charges",{})
                    if not isinstance(declared,dict):
                        raise ValueError("formal_charges 须为分项电荷")
                    if "parent" not in declared:
                        declared["parent"]=source_charge(graph.structure,args.input)
                report,result=remove(graph,recipe)
                assumptions.append("封端选择："+recipe.get("cap_plan",recipe.get("cap","none")))
            elif args.operation=="replace-fragment":
                from .fragment_ops import replace_fragment
                report,result=replace_fragment(graph,args.guest,recipe)
                params["guest_sha256"]=sha256(args.guest)
                assumptions.append("锚点固定并保留原宿主键表")
            elif args.operation=="extract-subnet":
                from .subnet import extract
                if args.cap is not None:
                    recipe["cap"]=args.cap
                report,result=extract(graph,recipe)
                assumptions.append("封端选择："+recipe.get("cap","none"))
            else:
                from .placement import scan
                report,result=scan(graph,args.guest,recipe)
                params["guest_sha256"]=sha256(args.guest)
                assumptions.append("刚性客体；指定的配位键为几何模型假设")
                if recipe.get("displace") is not None:
                    removed=[r["index"] for r in report.get("template_discarded",[])]
                    assumptions.append(f"放置前移除客体片段 {recipe['displace']}，原模板编号 {removed}")
            params.update(report.get("window_params",{}))
            params["recipe"]=recipe
            p5c_mode=(args.operation=="remove-fragment" and "cap_plan" in recipe
                      or args.operation=="scan-placement" and recipe["site"]["kind"]=="donor")
            if p5c_mode and "assumptions" in recipe:
                notes=recipe["assumptions"]
                if not isinstance(notes,list) or any(not isinstance(s,str) or not s.strip() for s in notes):
                    raise ValueError("建模假设须为非空文本的列表")
                assumptions.extend(notes)
            if result is None:
                print(json.dumps(report,ensure_ascii=False))
                return 2
            if not args.output_dir:
                raise ValueError("生成候选需要 --output-dir")
            product,mapping=result
        elif args.operation=="clean":
            report, result = clean(graph,args.choose)
            if result is None:
                print(json.dumps(report,ensure_ascii=False))
                return 2
            if not args.output_dir:
                raise ValueError("清洗产物需要 --output-dir")
            product,mapping = result
            params["choose"] = args.choose
            assumptions = list(args.choose)
        else:
            report = graph.report()
            product,mapping = graph,list(range(1,len(graph.elements)+1))
            if args.bond_path:
                params["bond_path"] = args.bond_path
                report["bond_path"] = bond_path(graph,*args.bond_path)
        if args.output_dir:
            publish(args.output_dir,args.input,args.operation,graph,product,mapping,params,report,assumptions,
                    template=getattr(args,"guest",None))
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))
    return 0
