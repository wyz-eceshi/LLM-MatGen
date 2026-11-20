"""按周期连通分量抽层或抽链。所有位点编号以展开后的输入为准。"""
import numpy as np
from pymatgen.core import Structure
from .cif import Model, edge
from .graph import BondGraph, adjacency, components
from .diff import compare


def model_at(cell, symbols, coords, edges):
    n = len(symbols)
    return Model(Structure(cell,symbols,coords), [f"{s}{i}" for i,s in enumerate(symbols,1)],
                 ["."]*n,[1.]*n,edges,["."]*n)


def reference(g, primitive):
    """按原胞平移折叠，保留每个代表位点对应的全部父编号。"""
    cell = g.structure.get_primitive_structure(tolerance=.01).lattice.matrix if primitive else g.cell
    transform = g.cell @ np.linalg.inv(cell)
    if not np.allclose(transform,np.rint(transform),atol=1e-7):
        raise ValueError("原胞与源晶胞无整数超胞关系")
    xyz = g.coords @ transform
    shifts = np.floor(xyz+1e-8).astype(int)
    coords = xyz-shifts
    table, groups, ids, sites = {},[],[],[]
    for i,(s,v) in enumerate(zip(g.symbols,coords)):
        key = (s,*np.round(v,7))
        if key not in table:
            table[key]=len(groups)
            groups.append([])
            sites.append(i)
        k=table[key]
        groups[k].append(i+1)
        ids.append(k)
    multiplicity=round(abs(np.linalg.det(transform)))
    if any(len(group)!=multiplicity for group in groups):
        raise ValueError("原胞位点与父结构对应不完整")
    edges={edge(ids[i],ids[j],np.rint(t@transform).astype(int)+shifts[j]-shifts[i])
           for i,j,t in g.edges}
    if len(edges)*multiplicity!=len(g.edges):
        raise ValueError("原胞平移不保持父结构键表")
    ref=BondGraph(model_at(cell,[g.symbols[i] for i in sites],coords[sites],edges))
    return ref,groups


def walk(graph, atoms):
    root=atoms[0]
    offsets={root:np.zeros(3,int)}
    cycles=set()
    stack=[root]
    while stack:
        i=stack.pop()
        for j,t in graph.neighbors[i]:
            nxt=offsets[i]+t
            if j not in offsets:
                offsets[j]=nxt
                stack.append(j)
            elif np.any(nxt!=offsets[j]):
                cycles.add(tuple(nxt-offsets[j]))
    return offsets,np.array(sorted(cycles))


def integer_basis(cycles, rank, cell, specified):
    if specified is not None:
        mat=np.asarray(specified)
        if mat.shape!=(3,3) or not np.issubdtype(mat.dtype,np.integer):
            raise ValueError("basis 必须为三行整数")
    else:
        # 利用闭环平移求独立的短矢量，再补齐三个独立方向。
        vectors=sorted(cycles.tolist(),key=lambda v:np.linalg.norm(v@cell))
        rows=[]
        for v in vectors+np.eye(3,dtype=int).tolist():
            if np.linalg.matrix_rank(rows+[v])>len(rows):
                rows.append(v)
            if len(rows)==3:
                break
        mat=np.array(rows,int)
    if abs(round(np.linalg.det(mat)))!=1:
        raise ValueError("请给出与周期子网相容的幺模整数 basis")
    winds=cycles@np.linalg.inv(mat)
    if not np.allclose(winds[:,rank:],0,atol=1e-8):
        raise ValueError("所选基矢未张成子网周期方向")
    return mat


def subnet_diff(parent,child,mapping,frame):
    ref,groups=reference(parent,frame["primitive"])
    lookup={p:i+1 for i,g in enumerate(groups) for p in g}
    mapped=[lookup[p] if p is not None else None for p in mapping]
    transform=np.asarray(frame["child_to_reference"])
    coords=child.coords@transform + frame["origin"]
    edges=set()
    for i,j,t in child.edges:
        winding=t@transform
        if not np.allclose(winding,np.rint(winding),atol=1e-7):
            raise ValueError("子网在真空方向存在周期键")
        edges.add(edge(i,j,np.rint(winding).astype(int)))
    aligned=BondGraph(model_at(ref.cell,child.symbols,coords,edges))
    result=compare(ref,aligned,mapped)
    result["parent_numbering"]="比较参考胞的一起编号，对应展开父编号见 reference_parent_groups"
    result["reference_parent_groups"]=groups
    return result


def extract(g, recipe):
    if any(o!=1 for o in g.model.occupancy):
        raise ValueError("请先处理部分占位")
    primitive=recipe.get("primitive",False)
    ref,groups=reference(g,primitive)
    rank=recipe.get("dimension",2)
    if rank not in (1,2):
        raise ValueError("dimension 应为 1 或 2")
    vacuum=float(recipe.get("vacuum",20))
    if not np.isfinite(vacuum) or vacuum<=0:
        raise ValueError("真空必须为正")
    edges=ref.edges.copy()
    cuts=[]
    for a,b,t in recipe.get("cut_edges",[]):
        if not (1<=a<=len(ref.symbols) and 1<=b<=len(ref.symbols)):
            raise ValueError("断键编号越界")
        if len(t)!=3 or any(type(v) is not int for v in t):
            raise ValueError("断键周期像必须为三个整数")
        k=edge(a-1,b-1,t)
        if k not in edges or ref.elements[a-1].is_metal==ref.elements[b-1].is_metal:
            raise ValueError("只允许断开现有金属–配体键")
        cuts.append(k)
        edges.remove(k)
    cut=BondGraph(model_at(ref.cell,ref.symbols,ref.coords,edges))
    candidates=[]
    for atoms,d in cut.components:
        if d!=rank:
            continue
        offsets,cycles=walk(cut,atoms)
        if "miller" in recipe and np.any(abs(cycles@ref.cell@np.linalg.inv(g.cell)@recipe["miller"])>1e-7):
            continue
        candidates.append((atoms,offsets,cycles))
    summary=[dict(component=i,atoms=len(a),parent_seed=groups[a[0]][0]) for i,(a,_,_) in enumerate(candidates,1)]
    choice=recipe.get("component",1 if len(candidates)==1 else None)
    if type(choice)!=int or not 1<=choice<=len(candidates):
        return {"candidates":summary},None
    atoms,offsets,cycles=candidates[choice-1]
    basis=integer_basis(cycles,rank,ref.cell,recipe.get("basis"))
    oriented=ref.structure.copy()
    oriented.make_supercell(basis)
    cell=oriented.lattice.matrix.copy()
    xyz=np.array([(ref.coords[i]+offsets[i])@ref.cell for i in atoms])
    symbols=[ref.symbols[i] for i in atoms]
    mapping=[groups[i][0] for i in atoms]
    ix={i:n for n,i in enumerate(atoms)}
    cap=recipe.get("cap","none")
    if cap not in ("none","H2O","OH"):
        raise ValueError("cap 仅支持 none、H2O、OH")
    caps=[]
    for i,j,t in cuts:
        if not ref.elements[i].is_metal:
            i,j,t=j,i,-np.array(t)
        if i in ix and cap!="none":
            caps.append((ix[i],(ref.coords[j]+t-ref.coords[i])@ref.cell))
    if caps:
        if not np.isfinite(recipe.get("cap_distance",0)) or recipe.get("cap_distance",0)<=0:
            raise ValueError("封端需要明确的正金属–O 距离 cap_distance")
        charge=recipe.get("uncapped_charge")
        if charge is None or charge-(len(caps) if cap=="OH" else 0)!=0:
            raise ValueError("封端数与显式裸子网电荷不平衡")
    cap_edges=[]
    for metal,direction in caps:
        u=direction/np.linalg.norm(direction)
        o=xyz[metal]+u*float(recipe["cap_distance"])
        v=np.cross(u,np.eye(3)[np.argmin(abs(u))])
        v/=np.linalg.norm(v)
        points=[o,o+.96*(.612*u+.791*v)]
        if cap=="H2O":
            points.append(o+.96*(.612*u-.791*v))
        first=len(xyz)
        xyz=np.vstack([xyz,points])
        symbols+=["O"]+["H"]*(len(points)-1)
        mapping += [None]*len(points)
        cap_edges.extend([(metal,first)]+[(first,first+j) for j in range(1,len(points))])
    normal=np.cross(cell[0],cell[1])
    normal/=np.linalg.norm(normal)
    directions=[normal] if rank==2 else [np.cross(normal,cell[0])/np.linalg.norm(cell[0]),normal]
    origin=np.zeros(3)
    for axis,u in enumerate(directions,rank):
        z=xyz@u
        cell[axis]=u*(np.ptp(z)+vacuum)
        origin+=u*(z.min()-vacuum/2)
    raw=(xyz-origin)@np.linalg.inv(cell)
    wrap=np.floor(raw+1e-9).astype(int)
    bonds=set()
    for i,j,t in edges:
        if i in ix and j in ix:
            w=(offsets[i]+t-offsets[j])@np.linalg.inv(basis)
            bonds.add(edge(ix[i],ix[j],np.rint(w).astype(int)+wrap[ix[j]]-wrap[ix[i]]))
    bonds.update(edge(i,j,wrap[j]-wrap[i]) for i,j in cap_edges)
    child=BondGraph(model_at(cell,symbols,raw-wrap,bonds))
    frame=dict(primitive=primitive,child_to_reference=(cell@np.linalg.inv(ref.cell)).tolist(),
               origin=(origin@np.linalg.inv(ref.cell)).tolist())
    report=dict(candidates=summary,component=choice,integer_basis=basis.tolist(),vacuum=vacuum,
                atom_count=len(symbols),retained_edges=len(bonds)-len(cap_edges),dimension=child.dimension,
                lattice=child.structure.lattice.parameters,comparison_frame=frame)
    return report,(child,mapping)
