"""沿共格方向施加单轴应变，再将两个有限宽板块并排。"""
from itertools import product
import numpy as np
from pymatgen.core import Structure
from .check import height
from .measure import layers


def block(s,axis,common,reorigin,shear=False):
    along=1-axis
    frac=s.frac_coords.copy()%1
    cell=s.lattice.matrix
    tangent=cell[along]/np.linalg.norm(cell[along])
    z=np.cross(*cell[:2]);z/=np.linalg.norm(z)
    if abs(np.dot(cell[2]/s.lattice.c,z))<.999999:
        raise ValueError("真空矢量须垂直于板片")
    normal=np.cross(tangent,z) if axis==0 else np.cross(z,tangent)
    slab_layers=layers(s,axis)
    shift=0.
    if reorigin and any(r["boundary"] for r in slab_layers) and len(slab_layers)>1:
        period=height(cell,axis)
        pos=[r["position"] for r in slab_layers]
        shift=.5*np.median(np.diff(np.r_[pos,pos[0]+period]))/period
        frac[:,axis]=(frac[:,axis]+shift)%1
    xyz=frac@cell
    points=np.zeros_like(xyz)
    points[:,axis]=xyz@normal
    points[:,along]=(xyz@tangent)*common/np.linalg.norm(cell[along])
    if shear:
        points[:,along]=frac[:,along]*common
    points[:,2]=xyz@z
    return points,float(shift)


def build(left,right,recipe):
    axis=recipe["axis"]
    if axis not in (0,1):
        raise ValueError("横向结法线须在面内")
    along=1-axis
    a,b=left.structure,right.structure
    lengths=[s.lattice.abc[along] for s in (a,b)]
    common=recipe.get("period",np.mean(lengths))
    pa,sa=block(a,axis,common,recipe.get("reorigin",True),recipe.get("allow_shear",False))
    pb,sb=block(b,axis,common,recipe.get("reorigin",True),recipe.get("allow_shear",False))
    origin=recipe.get("base_registry",[0.,0.])
    pb[:,along]+=origin[0]
    pb[:,2]+=origin[1]
    widths=[float(np.ptp(x[:,axis])) for x in (pa,pb)]
    vacuum=max(a.lattice.c,b.lattice.c)
    settings=recipe["check"]
    sizes=recipe.get("sizes")
    gaps=recipe.get("gaps")
    if sizes is None and gaps is None:
        raise ValueError("须给出法向胞长或每侧间距")
    if sizes is None:
        sizes=[sum(widths)+2*g for g in gaps]
    if gaps is not None and (len(gaps)!=len(sizes) or not np.allclose(sizes,np.array(gaps)*2+sum(widths),atol=1e-7,rtol=0)):
        raise ValueError("法向胞长与两侧间隙不自洽")
    if not sizes or min(sizes)<=sum(widths):
        raise ValueError("块宽超过晶胞可容纳的宽度")
    shifts=recipe.get("shifts",np.arange(0,common,recipe.get("step",.2)).tolist())
    if "fix_registry" in recipe:
        shifts=[recipe["fix_registry"]]
    dzs=recipe.get("dz",[0.])
    symbols=[x.specie.symbol for x in a]+[x.specie.symbol for x in b]
    lhs,rhs=np.array(symbols[:len(a)])[:,None],np.array(symbols[len(a):])[None,:]
    def mask(pair):
        x,y=pair.split("-")
        return ((lhs==x)&(rhs==y))|((lhs==y)&(rhs==x))
    contacts=np.zeros((len(a),len(b)),bool)
    for pair in settings.get("contact_pairs",[]):
        contacts|=mask(pair)
    if not settings.get("contact_pairs"):
        contacts[:]=True
    floors=np.full(contacts.shape,float(recipe.get("cross_floor",0)))
    specific=recipe.get("pair_floors")
    if specific is None:
        from .check import collision_limits
        limits=collision_limits([left,right],settings,[("配方",recipe)])
        specific={pair:r["minimum"] for pair,r in limits.items()}
    for pair,floor in specific.items():
        floors[mask(pair)]=np.maximum(floors[mask(pair)],floor)
    def positions(size,dy,dz):
        gap=(size-sum(widths))/2
        one,two=pa.copy(),pb.copy()
        # 沿法线去掉块的空白边缘，共格方向保留输入配准。
        if recipe.get("center_normal",axis==0):
            one[:,axis]-=one[:,axis].min()
        two[:,axis]+=one[:,axis].max()+gap-two[:,axis].min()
        two[:,along]+=dy
        two[:,2]+=dz
        xyz=np.vstack([one,two])
        if recipe.get("align_bottom",axis==0):
            xyz[:,2]-=xyz[:,2].min()
        cell=np.ones(3)
        cell[axis],cell[along],cell[2]=size,common,vacuum
        return xyz,cell
    # 扫描只计算跨块的周期距离；原块内键永远不能触发跨块碰撞。
    base=pb[None,:,:]-pa[:,None,:]
    base[:,:,axis]+=pa[:,axis].max()-pb[:,axis].min()
    lo,hi=settings.get("bond_window",[0,2.35])
    target=recipe.get("target_distance",1.95)
    special=mask(recipe["tradeoff_pair"]) if "tradeoff_pair" in recipe else None
    best=None
    by_size={}
    total=0
    feasible=0
    for size in sizes:
        gap=(size-sum(widths))/2
        dx=base[:,:,axis]+gap
        normal2=np.minimum(dx*dx,(dx-size)**2)
        for dy,dz in product(shifts,dzs):
            total+=1
            y=base[:,:,along]+dy
            z=base[:,:,2]+dz
            plane2=(y-common*np.rint(y/common))**2+(z-vacuum*np.rint(z/vacuum))**2
            ds=np.sqrt(normal2+plane2)
            nearest=float(ds.min())
            if np.any(ds+1e-8<floors):
                continue
            if 2*hi<min(size,common,vacuum):
                count=np.sum(contacts&(ds>=lo)&(ds<=hi))
            else:
                # 小胞可能在同一窗口纳入同一对原子的多个像。
                # 以最近像为中心，只枚举可能进入 cutoff 的相邻周期。
                periods=np.array([size,common,vacuum])
                bounds=np.floor(hi/periods+.5).astype(int)
                offsets=[range(-n,n+1) for n in bounds]
                wrapped=[v-p*np.rint(v/p) for v,p in zip([dx,y,z],periods)]
                count=0
                for image in product(*offsets):
                    squared=sum((v+k*p)**2 for v,k,p in zip(wrapped,image,periods))
                    count+=np.sum(contacts&(squared>=lo**2)&(squared<=hi**2))
            score=(int(count),-abs(nearest-target))
            row=dict(size=float(size),gap=float(gap),shift=float(dy),dz=float(dz),contacts=score[0],cross_minimum=nearest)
            if special is not None:
                row["pair_minimum"]=float(ds[special].min())
                row["length_deviation"]=abs(size-sum([height(s.lattice.matrix,axis) for s in (a,b)]))
                if size not in by_size or row["pair_minimum"]>by_size[size]["pair_minimum"]:
                    by_size[size]=row
                if row["pair_minimum"]<recipe["pair_minimum"] or row["length_deviation"]>recipe["length_tolerance"]:
                    continue
            feasible+=1
            if best is None or score>best[0]:
                best=score,row
    front=[]
    for row in sorted(by_size.values(),key=lambda r:(r["length_deviation"],-r["pair_minimum"])):
        if not front or row["pair_minimum"]>front[-1]["pair_minimum"]:
            front.append(row)
    report=dict(scanned=total,feasible=feasible,frontier=front,widths=widths,reorigin_fraction=[sa,sb],
                strains=[common/x-1 for x in lengths],common_period=float(common),pair_floors=specific)
    if best is None:
        return report,None
    row=best[1]
    xyz,cell=positions(row["size"],row["shift"],row["dz"])
    struct=Structure(np.diag(cell),symbols,xyz,coords_are_cartesian=True,to_unit_cell=True)
    mapping=[dict(parent=k,site=i+1,image=[0,0,0]) for k,slab in enumerate((a,b),1) for i in range(len(slab))]
    for i,(site,entry) in enumerate(zip(struct,mapping)):
        k=entry["parent"]-1
        slab=(a,b)[k]
        raw=slab[entry["site"]-1].frac_coords.copy()
        raw[axis]+=(sa,sb)[k]
        entry["image"]=(-np.floor(raw)).astype(int).tolist()
        entry["wrap"]=np.rint(xyz[i]/cell-site.frac_coords).astype(int).tolist()
    report["selected"]=row
    return report,(struct,mapping)
