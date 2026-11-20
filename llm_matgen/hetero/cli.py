"""通用异质结命令及双父结构导出。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .check import check,collision_limits,pair_key
from .measure import measure
from llm_matgen.mof.cif import read_model
from llm_matgen.mof.sidecar import write_json,sha256,bonded_cif


HELP={"measure":"测量板片晶格、层和配位","build":"从两块板片构建异质结","check":"从产物坐标核对两个界面"}


def configure(parser,operation):
    parser.set_defaults(_handler=run)
    parser.add_argument("--recipe",help="UTF-8 JSON 几何与判据参数")
    if operation!="measure":
        parser.add_argument("--radius-factor",type=float,help="块内无同类键时，共价半径和的乘数（默认 0.75）")
        parser.add_argument("--pair-floor",action="append",metavar="Cu-O=距离",help="元素对跨界面碰撞下限，单位 Å，可重复")
        parser.add_argument("--cross-floor",type=float,help="全部跨界面元素对的距离下限（Å）")
        parser.add_argument("--watch-pair",action="append",metavar="左标签,右标签")
        parser.add_argument("--bond-window",metavar="下限,上限")
    if operation=="build":
        parser.add_argument("--mode",required=True,choices=["vertical","lateral"])
        parser.add_argument("--fix-registry",metavar="dy=平移量")
        parser.add_argument("--no-reorigin",action="store_true")
        parser.add_argument("--allow-shear",action="store_true")
        parser.add_argument("--target-distance",type=float)
        parser.add_argument("--accept-strain",type=float)
        parser.add_argument("--output-dir",required=True)
        parser.add_argument("--archive",action="store_true",help="将候选与双父快照写入指定库")
        parser.add_argument("--archive-db",help="显式指定 SQLite 库路径")
    else:
        parser.add_argument("--input",required=True)
        parser.add_argument("--axis",choices=["a","b","c"])
    if operation in ("build","check"):
        parser.add_argument("--substrate",required=operation=="build")
        parser.add_argument("--film",required=operation=="build")
    if operation=="check":
        parser.add_argument("--split",type=int,help="第一块的原子数")
        parser.add_argument("--sidecar")


def fragment_violations(checked,settings):
    bad=[]
    for block,counts in settings.get("required_fragments",{}).items():
        actual=checked["molecular_fragments"].get(str(block),{}).get("formulas",{})
        if actual!=counts:
            bad.append(f"块 {block} 的片段要求为 {counts}，从坐标得到 {actual}")
    return bad


def dump(obj):
    print(json.dumps(obj,ensure_ascii=False,allow_nan=False))


def publish(destination,paths,parents,recipe,mode,structure,mapping,report,archive_db=None):
    from pymatgen.io.vasp.inputs import Poscar
    from llm_matgen.viewer import write_viewer
    from llm_matgen.mof.ledger import verify_output
    from .audit import graphs, parent_inventory
    axis=recipe["axis"]
    parent_symbols, expected_sites, blocks, charges = parent_inventory(parents, paths, mapping, report, recipe)
    product,diff=graphs(parents,structure,mapping,report,axis)
    checked=report["check"]
    header=dict(方向={char:("界面法线" if i==axis else "真空" if i==2 else "面内共格") for i,char in enumerate("xyz")},
                原子数=len(structure),组分=checked["composition"],跨界面最近距离=checked["cross_minimum"],
                两个界面=checked["interfaces"])
    for field in ["selected","widths","strains","common_period","match","registry","gap"]:
        if field in report:
            header[field]=report[field]
    notes=["切面可能有悬空键，需先弛豫。"]
    if "fix_registry" in recipe:
        notes.append("冻结共格方向的相对平移，保留输入的手动配准。")
    if mode=="vertical" and recipe.get("accept_strain",.05)>.05:
        notes.append(f"显式放宽主应变上限至 {recipe['accept_strain']:.2%}。")
    dest=Path(destination)
    if dest.exists():
        raise ValueError("输出目录已存在")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as scratch:
        out=Path(scratch)/"candidate"
        out.mkdir()
        comments="".join("# "+k+" "+json.dumps(v,ensure_ascii=False)+"\n" for k,v in header.items())
        (out/"structure.cif").write_text(comments+"# "+notes[0]+"\n"+bonded_cif(product),encoding="utf8")
        (out/"POSCAR").write_text(Poscar(structure).get_str(),encoding="utf8")
        accounting = verify_output(out, parent_symbols, [], [], charges, expected_sites=expected_sites)
        accounting["composition_ledger"]["parent_blocks"] = blocks
        write_viewer([("异质结",structure)],out/"preview.html")
        write_json(out/"sidecar.json",dict(operation="hetero build",params=dict(recipe,mode=mode),
            parent_sha256=[sha256(p) for p in paths],product_sha256=sha256(out/"structure.cif"),
            poscar_sha256=sha256(out/"POSCAR"),atom_map=mapping,atom_map_basis="按子编号依次记录父块、父位点、原胞像及绕回产物的位移",
            assumptions=notes,status="unoptimized_candidate",graph_diff=diff,**accounting))
        if archive_db is not None:
            from llm_matgen.adsorption.revision import RevisionImporter
            report["archive_snapshot_id"]=RevisionImporter.archive_interface(archive_db,paths,out)
        write_json(out/"report.json",report)
        out.rename(dest)


def run(args):
    recipe=json.loads(Path(args.recipe).read_text("utf8")) if args.recipe else {}
    if not isinstance(recipe,dict):
        raise ValueError("参数文件须为 JSON 对象")
    explicit={}
    for item in getattr(args,"pair_floor",None) or []:
        key,separator,value=item.partition("=")
        if not separator or len(key.split("-"))!=2:
            raise ValueError("--pair-floor 需要 元素-元素=距离")
        explicit.setdefault("pair_floors",{})[pair_key(*key.split("-"))]=float(value)
    if getattr(args,"cross_floor",None) is not None:
        explicit["cross_floor"]=args.cross_floor
    if getattr(args,"radius_factor",None) is not None:
        explicit["radius_factor"]=args.radius_factor
    layers=[("命令行",explicit),("配方",recipe)]
    meta=None
    if getattr(args,"sidecar",None):
        meta=json.loads(Path(args.sidecar).read_text("utf8"))
        if sha256(args.input) not in (meta["product_sha256"],meta["poscar_sha256"]):
            raise ValueError("输入结构与 sidecar 摘要不符")
        layers.insert(0,("sidecar",meta["params"]))
        recipe={**meta["params"],**recipe}
    settings=recipe.setdefault("check",{})
    if getattr(args,"watch_pair",None):
        settings["watch_pairs"]=args.watch_pair
    if getattr(args,"bond_window",None):
        window=list(map(float,args.bond_window.split(",")))
        if len(window)!=2 or not 0<=window[0]<=window[1]:
            raise ValueError("接触距离窗口无效")
        settings["bond_window"]=window
    if getattr(args,"axis",None):
        recipe["axis"]="abc".index(args.axis)
    if args.operation=="measure":
        dump(measure(read_model(args.input),recipe.get("axis",2),settings))
        return 0
    paths=[args.substrate,args.film]
    if any(paths) and not all(paths):
        raise ValueError("核对父结构需要同时给出两侧路径")
    parents=[read_model(p) for p in paths] if all(paths) else []
    if any(occ!=1 for p in parents for occ in p.occupancy):
        raise ValueError("请先明确选择部分占位")
    from .audit import compare_blocks
    if args.operation=="check":
        model=read_model(args.input)
        if meta:
            mapping=meta["atom_map"]
            if parents and [sha256(p) for p in paths]!=meta["parent_sha256"]:
                raise ValueError("父结构与 sidecar 摘要不符")
            split=sum(e["parent"]==1 for e in mapping)
            labels=[e["label"] for e in mapping]
        else:
            split=args.split or recipe.get("split") or (len(parents[0].structure) if parents else None)
            labels=sum((p.labels for p in parents),[]) if parents else model.labels
            mapping=[dict(parent=k,site=i+1) for k,p in enumerate(parents,1) for i in range(len(p.labels))]
        if split is None or "axis" not in recipe or len(labels)!=len(model.structure):
            raise ValueError("请指定块边界、法线与对应标签，或提供配套 sidecar")
        if not 0<split<len(model.structure):
            raise ValueError("父块边界无效")
        blocks=parents
        if not blocks:
            blocks=[]
            for start,end in ((0,split),(split,len(model.structure))):
                edges=None if model.explicit is None else {(i-start,j-start,t) for i,j,t in model.explicit
                                                           if start<=i<end and start<=j<end}
                blocks.append(model.slice(range(start,end),edges))
        limits=collision_limits(blocks,settings,layers)
        checked=check(model.structure,split,recipe["axis"],settings,labels,limits=limits)
        if parents:
            checked["comparison"],block_violations=compare_blocks(parents,model.structure,mapping,recipe["axis"],settings,checked["blocks"],recipe)
            checked["violations"].extend(block_violations)
        checked["violations"].extend(fragment_violations(checked,settings))
        dump(checked)
        return 2 if checked.get("violations") else 0
    if bool(args.archive)!=bool(args.archive_db):
        raise ValueError("入库须同时给出 --archive 和 --archive-db")
    if Path(args.output_dir).exists():
        raise ValueError("输出目录已存在")
    if args.mode=="vertical":
        if args.accept_strain is None and recipe.get("accept_strain",.05)>.05:
            raise ValueError("超过 5% 须显式给出 --accept-strain")
        from .vertical import build
        recipe["axis"]=2
        if args.accept_strain is not None:
            recipe["accept_strain"]=args.accept_strain
    else:
        from .lateral import build
        if recipe.get("axis") not in (0,1):
            raise ValueError("横向模式须指定面内法线 axis=0 或 1")
        if args.no_reorigin: recipe["reorigin"]=False
        if args.allow_shear: recipe["allow_shear"]=True
        if args.fix_registry is not None:
            key,sep,value=args.fix_registry.rpartition("=")
            if sep and key!=("dy","dx")[recipe["axis"]]:
                raise ValueError("冻结方向应平行于界面")
            recipe["fix_registry"]=float(value)
        if args.target_distance is not None: recipe["target_distance"]=args.target_distance
    limits=collision_limits(parents,settings,layers)
    recipe["resolved_floors"]=limits
    effective={k:r["minimum"] for k,r in limits.items() if r["minimum"] is not None}
    report,result=build(*parents,dict(recipe,pair_floors=effective,cross_floor=0.))
    if result is None:
        dump(report)
        return 2
    structure,mapping=result
    for e in mapping:
        e["label"]=parents[e["parent"]-1].labels[e["site"]-1]
    split=sum(e["parent"]==1 for e in mapping)
    axis=recipe["axis"]
    checked=check(structure,split,axis,settings,[e["label"] for e in mapping],limits=limits)
    checked["comparison"],block_violations=compare_blocks(parents,structure,mapping,axis,settings,checked["blocks"],recipe)
    checked["violations"].extend(fragment_violations(checked,settings))
    checked["violations"].extend(block_violations)
    report["check"]=checked
    if checked["violations"]:
        dump(report)
        return 2
    publish(args.output_dir,paths,parents,recipe,args.mode,structure,mapping,report,args.archive_db)
    dump(dict(output=args.output_dir,atoms=len(structure),cross_minimum=checked["cross_minimum"]))
    return 0
