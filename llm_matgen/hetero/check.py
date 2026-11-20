"""从周期坐标分别量取两个接面的距离和接触数。"""
from collections import Counter
import numpy as np
from pymatgen.core import Lattice,Structure
from ase.data import covalent_radii,atomic_numbers


def continuous(values):
    wrapped=np.asarray(values)%1.
    ordered=np.sort(wrapped)
    gaps=np.diff(np.r_[ordered,ordered[0]+1])
    start=ordered[(int(gaps.argmax())+1)%len(ordered)]
    return (wrapped-start+1e-10)%1+start-1e-10


def height(cell,axis):
    return 1/np.linalg.norm(np.linalg.inv(cell)[:,axis])


def fingerprint(structure,cutoffs=None):
    names=[a.specie.symbol for a in structure]
    if cutoffs is None:
        cutoffs={a+"-"+b:1.2*(covalent_radii[atomic_numbers[a]]+covalent_radii[atomic_numbers[b]])
                 for a in set(names) for b in set(names) if a<=b}
    neighbors=structure.get_neighbor_list(max(cutoffs.values()))
    result={}
    for key,limit in cutoffs.items():
        a,b=key.split("-")
        counts=np.zeros(len(names),int)
        lengths=[]
        for i,j,t,d in zip(*neighbors):
            if d>limit or not ((names[i]==a and names[j]==b) or (names[i]==b and names[j]==a)):
                continue
            counts[i]+=1
            lengths.append(float(d))
        result[key]=dict(range=[min(lengths),max(lengths)] if lengths else None,
            coordination={s:dict(Counter(map(int,counts[np.array(names)==s]))) for s in sorted({a,b})},
            site_counts=counts.tolist())
    return result


def pair_key(a,b):
    return "-".join(sorted((a,b)))


def collision_limits(blocks,settings,layers=()):
    """对每种跨块元素对按来源选取第一个适用下限。"""
    from llm_matgen.mof.cif import Model
    from llm_matgen.mof.graph import BondGraph

    models=[]
    for block in blocks:
        if isinstance(block,Model):
            models.append(block)
        else:
            n=len(block)
            models.append(Model(block,[s.label for s in block],["."]*n,[1.]*n,None,["."]*n))
    pairs={pair_key(a.symbol,b.symbol) for a in models[0].structure.composition.elements
           for b in models[1].structure.composition.elements}
    factor=next((config["radius_factor"] for _,config in layers if "radius_factor" in config),
                settings.get("radius_factor",.75))
    if not np.isfinite(factor) or factor<=0:
        raise ValueError("共价半径系数须为正的有限数")
    limits={}
    for source,config in layers:
        saved=config.get("resolved_floors",{}) if source=="sidecar" else {}
        floors={pair_key(*k.split("-")):float(v) for k,v in config.get("pair_floors",{}).items()}
        common=config.get("cross_floor")
        if common is not None:
            common=float(common)
        if any(not np.isfinite(v) or v<0 for v in [*floors.values(),*([common] if common is not None else [])]):
            raise ValueError("碰撞下限须为非负有限距离")
        for key in sorted(pairs-limits.keys()):
            if key in saved and saved[key]["minimum"] is not None:
                limits[key]=dict(minimum=saved[key]["minimum"],source=source,recorded_source=saved[key]["source"])
            elif key in floors or common is not None:
                limits[key]=dict(minimum=max(floors.get(key,0.),common or 0.),source=source)
    missing=pairs-limits.keys()
    if missing:
        shortest={}
        for model in models:
            graph=BondGraph(model)
            for i,j,image in graph.edges:
                key=pair_key(graph.symbols[i],graph.symbols[j])
                if key in missing:
                    shortest[key]=min(shortest.get(key,np.inf),graph.distance(i,j,image))
        for key in sorted(missing):
            if key in shortest:
                limits[key]=dict(minimum=shortest[key],source="块内最短同类键")
            else:
                radii=sum(covalent_radii[atomic_numbers[e]] for e in key.split("-"))
                limits[key]=dict(minimum=float(factor*radii),source="共价半径规则",factor=factor)
    return limits


def interface_frame(structure,split,axis):
    # 在法向圆周上找块 2 → 块 1 的过渡，沿正方向展开两块。
    coords=structure.frac_coords.copy()
    z=coords[:,axis]%1
    order=np.argsort(z,kind="stable")
    successor=np.roll(order,-1)
    cuts=(order>=split)&(successor<split)
    gaps=(z[successor]-z[order])%1
    cut=np.flatnonzero(cuts)[np.argmax(gaps[cuts])]
    origin=z[successor[cut]]
    coords[:,axis]=(z-origin+1.e-12)%1+origin-1.e-12
    left,right=coords[:split,axis],coords[split:,axis]
    intervals=np.array([[left.max(),right.min()],[right.max(),left.min()+1]])
    return coords,intervals


def midpoint_face(midpoint,intervals,towards_right):
    # 对跨过切面的深层原子，中点可能在块内。此时用这两个像之间跨过的间隙。
    origin=intervals[1,1]-1
    z=(midpoint-origin)%1+origin
    for face,(start,end) in enumerate(intervals):
        if start-1.e-12<=z<=end+1.e-12:
            return face
    return 0 if towards_right else 1


def check(structure,split,axis,settings,labels=None,*,limits=None):
    if not 0<split<len(structure) or axis not in (0,1,2):
        raise ValueError("块边界或接面方向无效")
    coords,intervals=interface_frame(structure,split,axis)
    symbols=np.array([s.specie.symbol for s in structure])
    left,right=coords[:split],coords[split:]
    # 面内取最短周期像，法线保留有符号距离，避免库先将三轴都折回胞内。
    cell=structure.lattice.matrix
    plane=np.delete(cell,axis,axis=0)
    normal=np.cross(*plane)
    normal/=np.linalg.norm(normal)
    flat=Lattice(np.vstack([plane,normal]))
    def face_dist(l,r):
        l=flat.get_fractional_coords(l@cell)
        r=flat.get_fractional_coords(r@cell)
        dz=l[:,None,2]-r[None,:,2]
        l[:,2]=0
        r[:,2]=0
        return np.hypot(flat.get_all_distances(l,r),dz)
    ds=[face_dist(left,right),face_dist(left,right-np.eye(3)[axis])]
    pairs=settings.get("contact_pairs",[])
    mask=np.ones(ds[0].shape,bool) if not pairs else np.zeros(ds[0].shape,bool)
    def pair_mask(a,b):
        return ((symbols[:split,None]==a)&(symbols[None,split:]==b))|((symbols[:split,None]==b)&(symbols[None,split:]==a))
    for pair in pairs:
        mask|=pair_mask(*pair.split("-"))
    lo,hi=settings.get("bond_window",[0,2.35])
    if not np.isfinite([lo,hi]).all() or not 0<=lo<=hi:
        raise ValueError("接触窗口须为非负有限闭区间")
    interfaces=[]
    for face,d in enumerate(ds):
        i,j=np.unravel_index(d.argmin(),d.shape)
        interfaces.append(dict(gap_fractional=intervals[face].tolist(),minimum=float(d[i,j]),
                               pair=[int(i)+1,int(j)+split+1],contacts=0))
    distances=structure.lattice.get_all_distances(left,right)
    if limits is None:
        blocks=[Structure.from_sites(part) for part in (structure[:split],structure[split:])]
        limits=collision_limits(blocks,settings,[("显式参数",settings)])
    violations=[]
    if intervals[0,1]<intervals[0,0]-1.e-10:
        violations.append("两个块在法向上的范围交叠")
    for key,rule in limits.items():
        floor=rule["minimum"]
        if floor is None:
            continue
        selected=pair_mask(*key.split("-"))
        shortest=np.where(selected,distances,np.inf)
        i,j=np.unravel_index(shortest.argmin(),shortest.shape)
        if shortest[i,j]+1.e-8<floor:
            violations.append(dict(type="跨界面碰撞",elements=key,atoms=[int(i)+1,int(j)+split+1],
                                   distance=float(shortest[i,j]),floor=floor,source=rule["source"]))
    # 邻居表从块 1 指向块 2，每行有自己的整数像；反向项不重复计入。
    ii,jj,images,lengths=structure.get_neighbor_list(hi+1.e-10,exclude_self=False)
    contacts=(ii<split)&(jj>=split)&(lengths>=lo-1.e-10)&(lengths<=hi+1.e-10)
    for i,j,image in zip(ii[contacts],jj[contacts],images[contacts]):
        if not mask[i,j-split]:
            continue
        step=structure[j].frac_coords[axis]+image[axis]-structure[i].frac_coords[axis]
        mid=coords[i,axis]+.5*step
        face=midpoint_face(mid,intervals,step>=0)
        interfaces[face]["contacts"]+=1
    watches={}
    for pair in settings.get("watch_pairs",[]):
        a,b=pair.split(",")
        def resolve(token,block,names):
            if token.isdigit():
                idx=int(token)-1
                if not 0<=idx<len(block):
                    raise ValueError("监视编号越界")
                return [idx]
            hits=[i for i,(label,s) in enumerate(zip(block,names)) if token in (label,s)]
            if not hits:
                raise ValueError("监视标签不存在 "+token)
            return hits
        if labels is None:
            labels=[str(i+1) for i in range(split)]+[str(j+1) for j in range(len(right))]
        ids=np.ix_(resolve(a,labels[:split],symbols[:split]),resolve(b,labels[split:],symbols[split:]))
        watches[pair]=dict(minimum=float(distances[ids].min()),interfaces=[float(d[ids].min()) for d in ds])
    molecular_fragments={}
    block_stats=[]
    terminations=[]
    global_min=float(distances.min())
    for i,part in enumerate([structure[:split],structure[split:]]):
        block=Structure.from_sites(part)
        z=(left if i==0 else right)[:,axis]*height(cell,axis)
        names=np.array([site.specie.symbol for site in block])
        tol=settings.get("layer_tol",.14)
        terminations.append([dict(Counter(names[z<=z.min()+tol])),dict(Counter(names[z>=z.max()-tol]))])
        if any(x.specie.is_metal for x in block) and "C" in names:
            from llm_matgen.mof.cif import Model
            from llm_matgen.mof.graph import BondGraph
            n=len(block)
            g=BondGraph(Model(block,list(names),["."]*n,[1.]*n,None,["."]*n))
            fragments=[f for f in g.fragments() if f["type"]!="metal_node"]
            molecular_fragments[str(i+1)]=dict(formulas=dict(Counter(f["formula"] for f in fragments)),
                                                 fragments=fragments,borderline=g.borderline,
                                                 numbering="当前块内编号，从 1 开始")
        matrix=block.distance_matrix
        np.fill_diagonal(matrix,np.inf)
        global_min=min(global_min,float(matrix.min()),float(min(block.lattice.get_lll_reduced_lattice().abc)))
        block_stats.append(fingerprint(block,settings.get("cutoffs")))
    return dict(collision_floors=limits,violations=violations,atom_count=len(structure),composition=dict(Counter(symbols)),cell=structure.lattice.parameters,
                vacuum=float((1-np.ptp(continuous(coords[:,2])))*height(structure.lattice.matrix,2)),
                molecular_fragments=molecular_fragments,face_compositions=terminations,interfaces=interfaces,global_minimum=global_min,cross_minimum=float(distances.min()),
                contacts=sum(x["contacts"] for x in interfaces),blocks=block_stats,watch_pairs=watches,
                pair_minima={pair:float(distances[pair_mask(*pair.split("-"))].min()) for pair in
                             settings.get("pair_minima",[]) if pair_mask(*pair.split("-")).any()})
