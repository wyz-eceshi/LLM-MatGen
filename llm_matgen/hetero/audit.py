"""在同一超胞和位点编号中比较两个独立父图与界面图。"""
from collections import Counter
from dataclasses import replace
import numpy as np
from pymatgen.core import Structure
from llm_matgen.mof.cif import Model,edge
from llm_matgen.mof.graph import BondGraph
from llm_matgen.mof.diff import compare
from .check import continuous,height,fingerprint


def parent_inventory(parents, paths, mapping, report, recipe):
    """从原始双父结构和选中的整数换胞矩阵确定每个父位点应出现的次数。"""
    from llm_matgen.mof.ledger import charge_ledger, source_charge

    parent_symbols = []
    expected_sites = []
    blocks = []
    cursor = 0
    expanded_charges = []
    for number, (name, model) in enumerate(zip(("substrate", "film"), parents, strict=True), 1):
        matrix = np.asarray(report.get("match", {}).get(name, [[1, 0], [0, 1]]))
        if matrix.shape != (2, 2) or not np.issubdtype(matrix.dtype, np.integer):
            raise ValueError("组成账需要二维整数换胞矩阵")
        copies = abs(int(matrix[0, 0]) * int(matrix[1, 1]) - int(matrix[1, 0]) * int(matrix[0, 1]))
        source = [s.specie.symbol for s in model.structure]
        count = copies * len(source)
        entries = mapping[cursor:cursor + count]
        if (copies == 0 or any(e["parent"] != number for e in entries)
                or Counter(e["site"] for e in entries) != dict.fromkeys(range(1, len(source) + 1), copies)):
            raise ValueError(f"块 {number} 的子→父映射与整数换胞的预期位点数不符")
        # 周期像的陪集唯一性由下游 transported() 校验。
        parent_symbols.extend(source * copies)
        expected_sites.extend(source[e["site"] - 1] for e in entries)
        cursor += count
        known_charge = source_charge(model.structure, paths[number - 1])
        charge = recipe.get("formal_charges", {}).get(name)
        if charge is None:
            charge = known_charge
        if charge is not None and type(charge) is not int:
            raise ValueError(f"块 {number} 的形式电荷须为整数或 null")
        if known_charge is not None and charge != known_charge:
            raise ValueError(f"块 {number} 的配方形式电荷与父结构记录不符")
        expanded_charges.append(None if charge is None else charge * copies)
        blocks.append(dict(parent=number, composition=dict(Counter(source)), copies=copies,
                           matrix=matrix.tolist(), formal_charge=charge))
    if cursor != len(mapping):
        raise ValueError("产物映射含有双父结构之外的原子")
    total = None if None in expanded_charges else sum(expanded_charges)
    return parent_symbols, expected_sites, blocks, charge_ledger(total, 0, 0)


def transported(graph,entries,matrix,offset=0):
    """父原胞边按整数超胞陪集展开，再补偿产物绕回胞内的偏移。"""
    inv=np.linalg.inv(matrix)
    def key(site,image):
        frac=np.asarray(image)@inv
        return site,tuple(np.rint((frac-np.floor(frac+1e-8))*abs(round(np.linalg.det(matrix)))).astype(int))
    lookup={key(e["site"],e["image"]):i for i,e in enumerate(entries)}
    if len(lookup)!=len(entries):
        raise ValueError("超胞映射重复")
    for i,e in enumerate(entries):
        for parent_j,t in graph.neighbors[e["site"]-1]:
            endpoint=np.array(e["image"])+t
            j=lookup[key(parent_j+1,endpoint)]
            other=entries[j]
            shift=np.rint((endpoint-other["image"])@inv).astype(int)
            image=shift+other["wrap"]-np.array(e["wrap"])
            yield edge(offset+i,offset+j,image),shift


def graphs(parents,structure,mapping,report,axis):
    n=len(structure)
    model=Model(structure,[f"{s.specie.symbol}{i}" for i,s in enumerate(structure,1)],["."]*n,[1.]*n,None,["."]*n)
    inferred=BondGraph(model)
    parent_edges=set()
    product_edges=set(inferred.edges)
    offset=0
    for k,parent in enumerate(parents):
        entries=[e for e in mapping if e["parent"]==k+1]
        matrix=np.eye(3,dtype=int)
        if "match" in report:
            matrix[:2,:2]=report["match"][("substrate","film")[k]]
        if parent.explicit is not None:
            product_edges={e for e in product_edges if not offset<=e[0]<=e[1]<offset+len(entries)}
        for e,t in transported(BondGraph(parent),entries,matrix,offset):
            parent_edges.add(e)
            if parent.explicit is not None and t[axis]==0:
                product_edges.add(e)
        offset+=len(entries)
    reference=BondGraph(replace(model,explicit=parent_edges))
    product=BondGraph(replace(model,explicit=product_edges))
    diff=compare(reference,product,list(range(1,n+1)))
    diff["basis"]="两个父图展开后共用产物编号；周期像已按原点和绕胞平移校正。"
    return product,diff


def same_block_lengths(parent,block,entries,axis,cutoffs,shear):
    # 把原胞也沿共格方向施加同一应变。切面可以少键，保留下来的每条键须能一一匹配。
    from collections import defaultdict
    from .lateral import block as orient
    tangent=1-axis
    common=block.lattice.abc[tangent]
    xy,_=orient(parent,axis,common,False,shear)
    cell=parent.lattice.matrix
    along=cell[tangent]/parent.lattice.abc[tangent]
    normal=np.cross(*cell[:2]);normal/=np.linalg.norm(normal)
    perp=np.cross(along,normal) if axis==0 else np.cross(normal,along)
    strained=np.empty((3,3))
    strained[:,axis]=cell@perp
    strained[:,tangent]=(cell@along)*common/parent.lattice.abc[tangent]
    if shear:
        strained[:,tangent]=0
        strained[tangent,tangent]=common
    strained[:,2]=cell@normal
    ref=Structure(strained,parent.species,xy,coords_are_cartesian=True)
    def lengths(s,ids):
        pairs=defaultdict(list)
        names=[site.specie.symbol for site in s]
        for i,j,t,d in zip(*s.get_neighbor_list(max(cutoffs.values()))):
            if edge(i,j,t)!=(i,j,tuple(t)): continue
            for label,limit in cutoffs.items():
                if d<=limit and sorted(label.split("-"))==sorted((names[i],names[j])):
                    pairs[(label,*sorted([ids[i],ids[j]]))].append(float(d))
        return pairs
    available=lengths(ref,list(range(1,len(ref)+1)))
    actual=lengths(block,[entry["site"] for entry in entries])
    bad=[]
    for key,distances in actual.items():
        original=available[key]
        for d in distances:
            match=next((i for i,x in enumerate(original) if abs(x-d)<=1e-4),None)
            if match is None:
                bad.append(dict(pair=key,product_length=d))
            else:
                original.pop(match)
    return bad


def compare_blocks(parents,structure,mapping,axis,settings,actual,recipe=None):
    from collections import Counter
    if len(mapping)!=len(structure):
        raise ValueError("产物原子数与双父映射长度不符")
    blocks=[entry["parent"] for entry in mapping]
    if sorted(set(blocks))!=[1,2] or blocks!=sorted(blocks):
        raise ValueError("两个父块的记录须按产物位点顺序连续排列")
    summaries=[]
    errors=[]
    for k,parent in enumerate(parents,1):
        ids=[i for i,e in enumerate(mapping) if e["parent"]==k]
        counts=Counter(mapping[i]["site"] for i in ids)
        if set(counts)!=set(range(1,len(parent.structure)+1)) or len(set(counts.values()))!=1:
            raise ValueError("原子数或父位点重复次数不符")
        if axis!=2 and len(ids)!=len(parent.structure):
            raise ValueError("横向结的原子数须等于两侧输入之和")
        if any(structure[i].species!=parent.structure[mapping[i]["site"]-1].species for i in ids):
            errors.append("原子映射与元素不符")
        original=fingerprint(parent.structure,settings.get("cutoffs"))
        z=continuous(structure.frac_coords[ids,axis])*height(structure.lattice.matrix,axis)
        stats={}
        for pair,row in actual[k-1].items():
            ref=original[pair]
            if not row["range"] and not ref["range"]:
                continue
            cutoff=(settings.get("cutoffs") or {}).get(pair)
            if cutoff is None:
                from ase.data import covalent_radii,atomic_numbers
                cutoff=1.2*sum(covalent_radii[atomic_numbers[a]] for a in pair.split("-"))
            lost=[j for j,i in enumerate(ids) if row["site_counts"][j]<ref["site_counts"][mapping[i]["site"]-1]]
            inside=[ids[j]+1 for j in lost if min(z[j]-min(z),max(z)-z[j])>cutoff+1e-6]
            if inside:
                errors.append(f"块 {k} 的 {pair} 在切面以外发生配位下降")
            stats[pair]=dict(parent_range=ref["range"],product_range=row["range"],
                             parent_coordination=ref["coordination"],product_coordination=row["coordination"],
                             decreased=[ids[j]+1 for j in lost],interior_decreased=inside)
        if axis!=2:
            limits=settings.get("cutoffs")
            if not limits:
                from ase.data import covalent_radii,atomic_numbers
                limits={key:1.2*sum(covalent_radii[atomic_numbers[a]] for a in key.split("-")) for key in original}
            bad=same_block_lengths(parent.structure,Structure.from_sites([structure[i] for i in ids]),
                                   [mapping[i] for i in ids],axis,limits,(recipe or {}).get("allow_shear",False))
            if bad:
                errors.append(f"块 {k} 有 {len(bad)} 条键与应变后的输入不符")
            stats["unmatched_bonds"]=bad
        summaries.append(stats)
    return summaries,errors
