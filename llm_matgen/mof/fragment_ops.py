"""按片段或局部配位选择原子，在显式周期图中执行删除和替换。"""
import operator
import re
from collections import Counter
from pathlib import Path
import numpy as np
from pymatgen.core import Element
from .cif import edge
from .graph import BondGraph
from .subnet import model_at
from .placement import unit


def selected(g, recipe):
    selectors=set(recipe)&{"predicate","fragment"}
    if len(selectors)!=1:
        raise ValueError("须指定一个片段编号或配位判据")
    if "fragment" in selectors:
        return next(([i-1 for i in f["atoms"]] for f in g.fragments() if f["id"]==recipe["fragment"]),[])
    conditions=[]
    ops={"=":operator.eq,">=":operator.ge,"<=":operator.le,">":operator.gt,"<":operator.lt}
    for clause in recipe["predicate"].split("&"):
        term=clause.strip()
        m=re.fullmatch(r"element\s*=\s*([A-Z][a-z]?)",term)
        if m:
            conditions.append(("element",operator.eq,Element(m[1]).symbol))
            continue
        m=re.fullmatch(r"neighbors\(([A-Z][a-z]?)\)\s*(>=|<=|=|>|<)\s*(\d+)",term)
        if not m:
            raise ValueError("无法识别的配位条件 "+term)
        conditions.append((Element(m[1]).symbol,ops[m[2]],int(m[3])))
    result=[]
    for i,symbol in enumerate(g.symbols):
        neighbors=Counter(g.symbols[j] for j,_ in g.neighbors[i])
        if all(op(symbol if field=="element" else neighbors[field],value) for field,op,value in conditions):
            result.append(i)
    return result


def omit(g, drop):
    gone=set(drop)
    keep=[i for i in range(len(g.elements)) if i not in gone]
    if not keep:
        raise ValueError("不能删除全部位点")
    ix={i:j for j,i in enumerate(keep)}
    retained={(ix[i],ix[j],t) for i,j,t in g.edges if i in ix and j in ix}
    return BondGraph(g.model.slice(keep,retained)),[i+1 for i in keep]


def remove(g, recipe):
    drop=selected(g,recipe)
    report={"removed_atoms":[i+1 for i in drop]}
    if not drop:
        report["reason"]="没有匹配位点"
        return report,None
    host,mapping=omit(g,drop)
    if "cap_plan" in recipe:
        if recipe.get("cap","none") != "none":
            raise ValueError("cap 与 cap_plan 互斥")
        from .cap_plan import place_caps
        return place_caps(g,host,mapping,drop,recipe)
    cap=recipe.get("cap","none")
    if cap=="none":
        return report,(host,mapping)
    if cap not in ("H2O","OH"):
        raise ValueError("未知封端")
    ix={p-1:i for i,p in enumerate(mapping)}
    ends=[(ix[i],(g.coords[j]+t-g.coords[i])@g.cell) for j in drop for i,rev in g.neighbors[j]
          if i in ix and g.elements[i].is_metal for t in [-np.array(rev)]]
    charge=recipe.get("uncapped_charge")
    if charge is None or charge!=(len(ends) if cap=="OH" else 0):
        raise ValueError("封端数量与给定裸结构电荷不平衡")
    length=recipe.get("cap_distance",0.)
    if not np.isfinite(length) or length<=0:
        raise ValueError("请指定正的金属–O 键长")
    coords=list(host.coords)
    symbols=list(host.symbols)
    bonds=set(host.edges)
    for i,direction in ends:
        u=unit(direction)
        v=unit(np.cross(u,np.eye(3)[np.argmin(abs(u))]))
        o=host.coords[i]@host.cell+length*u
        points=[o,o+.96*(.612*u+.791*v)]
        if cap=="H2O":
            points.append(o+.96*(.612*u-.791*v))
        frac=np.asarray(points)@np.linalg.inv(host.cell)
        wraps=np.floor(frac).astype(int)
        first=len(coords)
        bonds.add(edge(i,first,wraps[0]))
        for j in range(1,len(points)):
            bonds.add(edge(first,first+j,wraps[j]-wraps[0]))
        coords.extend(frac-wraps)
        symbols+=["O"]+["H"]*(len(points)-1)
        mapping += [None]*len(points)
    report["caps"]=len(ends)
    return report,(BondGraph(model_at(host.cell,symbols,coords,bonds)),mapping)


def sdf_template(path):
    lines=Path(path).read_text("utf-8").splitlines()
    if len(lines)<4 or "V2000" not in lines[3]:
        raise ValueError("需要 SDF V2000 模板")
    n,m=int(lines[3][:3]),int(lines[3][3:6])
    atoms=lines[4:4+n]
    coords=np.array([[float(row[k:k+10]) for k in (0,10,20)] for row in atoms])
    symbols=[Element(row[31:34].strip()).symbol for row in atoms]
    pairs=[(int(row[:3])-1,int(row[3:6])-1) for row in lines[4+n:4+n+m]]
    if len(atoms)!=n or len(pairs)!=m or any(not 0<=i<n or not 0<=j<n for i,j in pairs):
        raise ValueError("模板被截断或键端点越界")
    return symbols,coords,pairs


def append_template(host,mapping,symbols,positions,pairs,attachments):
    """模板使用连续笛卡尔坐标。attachments 中的整数像偏移对应宿主端点。"""
    n=len(host.symbols)
    frac=np.asarray(positions)@np.linalg.inv(host.cell)
    wraps=np.floor(frac).astype(int)
    bonds=set(host.edges)
    for i,j in pairs:
        bonds.add(edge(n+i,n+j,wraps[j]-wraps[i]))
    for i,image,j in attachments:
        bonds.add(edge(i,n+j,wraps[j]-image))
    child=BondGraph(model_at(host.cell,host.symbols+symbols,np.vstack([host.coords,frac-wraps]),bonds))
    return child,mapping+[None]*len(symbols)


def replace_fragment(g,path,recipe):
    if recipe.get("mode")=="ring":
        from .ring import replace_ring
        return replace_ring(g,path,recipe)
    if recipe.get("mode")!="terminal":
        raise ValueError("替换模式必须是 ring 或 terminal")
    from .placement import scan
    metal=recipe["metal"]-1
    if not 0<=metal<len(g.elements) or not g.elements[metal].is_metal:
        raise ValueError("金属编号无效")
    choices=[]
    for frag in g.fragments():
        if frag["type"]=="metal_node":
            continue
        atoms={j-1 for j in frag["atoms"]}
        bound=[(i,j,t) for i in range(len(g.symbols)) if i not in atoms for j,t in g.neighbors[i] if j in atoms]
        if len(bound)==1 and bound[0][0]==metal:
            choices.append((frag,bound[0]))
    chosen=[v for v in choices if v[0]["id"]==recipe.get("fragment",v[0]["id"])]
    if len(chosen)!=1:
        return {"candidates":[f for f,_ in choices]},None
    frag,(_,j,t)=chosen[0]
    host,mapping=omit(g,[i-1 for i in frag["atoms"]])
    direction=(g.coords[j]+t-g.coords[metal])@g.cell
    settings=dict(recipe.get("scan",{}),anchor=recipe["anchor"],site=dict(kind="terminal-axis",
                  direction=direction.tolist(),index=mapping.index(metal+1)+1))
    report,result=scan(host,path,settings)
    report["removed_atoms"]=frag["atoms"]
    report["scan_site"]=settings["site"]
    if result is None:
        return report,None
    child,indices=result
    return report,(child,[mapping[i-1] if i is not None else None for i in indices])
