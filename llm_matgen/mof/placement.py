"""周期宿主中的刚性客体位置与取向枚举。平移用 Å，转角用度。"""
import json
from itertools import product
from pathlib import Path
import numpy as np
from pymatgen.core import Molecule
from scipy.spatial.transform import Rotation
from ase.data import atomic_numbers,covalent_radii
from .cif import edge
from .graph import BondGraph
from .subnet import model_at


def unit(v):
    norm=np.linalg.norm(v)
    if norm<1e-10:
        raise ValueError("方向向量为零")
    return v/norm


def window_parameters(recipe, metals, donor):
    preferred=recipe.get("prefer_distance")
    if preferred is not None and (not np.isfinite(preferred) or preferred<=0):
        raise ValueError("参考键长须为正的有限数")
    values=[float(preferred) if preferred is not None else
            float(covalent_radii[atomic_numbers[m]]+covalent_radii[atomic_numbers[donor]]) for m in metals]
    return dict(distance_window=recipe["distance_window"],reference_distances=values,
                reference_pairs=[f"{m}-{donor}" for m in metals],
                reference_source="prefer_distance" if preferred is not None else "ASE covalent_radii sum",
                ranking=["reference_deviation ASC","min_nonbond DESC"])


def window_rank(row):
    # 相同几何的浮点舍入差不能抢在真正的非键净距之前。
    return round(row["reference_deviation"],12),-row["min_nonbond"]


def scan(host, guest_path, recipe):
    if "prefer_distance" in recipe and recipe.get("distance_window") is None and "anchors" not in recipe:
        raise ValueError("参考键长仅用于距离窗口模式")
    if recipe["site"]["kind"] == "donor":
        from .donor import scan_donor
        return scan_donor(host, guest_path, recipe)
    if recipe.get("distance_window") is not None and recipe["site"]["kind"] not in ("bridge","terminal-axis"):
        raise ValueError("距离窗口仅用于桥联位")
    if recipe["site"]["kind"]=="terminal-axis":
        from .terminal import scan_axis
        return scan_axis(host,guest_path,recipe)
    if Path(guest_path).suffix.lower() not in (".xyz",".mson",".json"):
        raise ValueError("客体只接受 XYZ 或 MSON")
    mol=(Molecule.from_dict(json.loads(Path(guest_path).read_text("utf-8")))
         if Path(guest_path).suffix.lower() in (".json",".mson") else Molecule.from_file(guest_path))
    symbols=[s.specie.symbol for s in mol]
    anchor=recipe.get("anchor",1)-1
    if not 0<=anchor<len(mol):
        raise ValueError("客体锚点越界")
    local=mol.cart_coords-mol.cart_coords[anchor]
    site=recipe["site"]
    mode=site["kind"]
    lattice=host.structure.lattice
    inv=np.linalg.inv(host.cell)
    targets=[]
    for item in site.get("atoms",[]):
        i=item["index"]-1
        t=np.asarray(item.get("image",[0,0,0]))
        if not 0<=i<len(host.elements) or t.shape!=(3,) or not np.issubdtype(t.dtype,np.integer):
            raise ValueError("位点或周期像无效")
        targets.append((i,t,(host.coords[i]+t)@host.cell))
    window=recipe.get("distance_window")
    if window is not None and mode!="bridge":
        raise ValueError("距离窗口仅用于桥联位")
    section=[(0.,0.)]
    azimuths=[0.]
    if mode=="bridge":
        if len(targets)!=2:
            raise ValueError("桥联位需要两个带周期像的端点")
        a,b=[v for _,_,v in targets]
        axis=unit(b-a)
        d=np.linalg.norm(b-a)
        if (window is None)==("distances" not in site):
            raise ValueError("桥联需要且只能选择目标距离或距离窗口之一")
        if window is None:
            r,s=site["distances"]
            if min(r,s)<=0 or not abs(r-s)<=d<=r+s:
                raise ValueError("桥联距离无法构成三角形")
            axial=(d*d+r*r-s*s)/(2*d)
            center=a+axial*axis
            section=[(0.,np.sqrt(max(0,r*r-axial*axial)))]
        else:
            ranking=window_parameters(recipe,[host.symbols[i] for i,_,_ in targets],symbols[anchor])
            lo,hi=window
            if not np.isfinite([lo,hi]).all() or not 0<lo<=hi:
                raise ValueError("距离窗口必须是正的有限闭区间")
            center=(a+b)/2
            extent=max(0,hi-d/2)
            radius=np.sqrt(max(0,hi*hi-(d/2)**2))
            section=list(product(np.unique(np.linspace(-extent,extent,5)),np.unique(np.linspace(0,radius,5))))
        u=unit(np.cross(axis,np.eye(3)[np.argmin(abs(axis))]))
        v=np.cross(axis,u)
        azimuths=recipe.get("azimuths",list(range(0,360,30)))
    elif mode in ("atom","metal"):
        if len(targets)!=1 or (mode=="metal" and not host.elements[targets[0][0]].is_metal):
            raise ValueError("原子或金属位点不符")
        center=targets[0][2]
    elif mode=="pore":
        center=np.array(site.get("fractional",[.5,.5,.5]))@host.cell
    else:
        raise ValueError("未知放置位点")
    if any(o!=1 for o in host.model.occupancy):
        raise ValueError("请先处理宿主的部分占位")
    for name in ("rotations","translations"):
        grid=np.asarray(recipe.get(name,[[0,0,0]]),dtype=float)
        if grid.ndim!=2 or grid.shape[1]!=3 or not np.isfinite(grid).all():
            raise ValueError(name+" 须为有限三维向量列表")
    coordinated=mode=="bridge" or (mode=="metal" and "distances" in site)
    def geometry(angle,rot,shift,axial,radial):
        pos=center+np.asarray(shift)
        if mode=="bridge":
            rad=np.radians(angle)
            pos=pos+axial*axis+radial*(np.cos(rad)*u+np.sin(rad)*v)
        return (Rotation.from_euler("xyz",rot,degrees=True).apply(local)+pos)@inv
    minimum=float(recipe["min_nonbond"])
    top=int(recipe.get("top",3))
    if not np.isfinite(minimum) or minimum<=0 or top<1:
        raise ValueError("碰撞下限和候选数须为正")
    # 对另一周期像的近接也执行筛选，即使最近像正是指定配位端点。
    def evaluate(frac):
        distances=lattice.get_all_distances(frac,host.coords)
        lengths=[]
        if coordinated:
            for i,t,xyz in targets:
                lengths.append(float(np.linalg.norm(frac[anchor]@host.cell-xyz)))
                _,nearest=lattice.get_distance_and_image(frac[anchor],host.coords[i])
                if np.array_equal(nearest,t):
                    cutoff=max(minimum,lengths[-1])+max(lattice.abc)
                    other=lattice.get_points_in_sphere([host.coords[i]%1],frac[anchor]@host.cell,cutoff)
                    excluded=t+np.floor(host.coords[i]).astype(int)
                    distances[anchor,i]=min((float(row[1]) for row in other
                                              if not np.array_equal(row[3],excluded)),default=np.inf)
        clearance=float(distances.min())
        # 客体与其周期像之间也可能碰撞；胞内既有分子几何由模板给定。
        image_radius=minimum if window is None else min(clearance,min(lattice.abc))+1e-12
        for point in frac:
            for _,d,j,img in lattice.get_points_in_sphere(frac%1,point@host.cell,image_radius):
                if np.any(img!=np.floor(frac[int(j)])):
                    clearance=min(clearance,float(d))
        error=float(np.linalg.norm(np.asarray(lengths)-site["distances"])) if coordinated and window is None else 0.
        return clearance,lengths,error
    rows=[]
    rejected=0
    collisions=0
    coordination_rejected=0
    total=0
    closest=None
    for (axial,radial),angle,rot,shift in product(section,azimuths,recipe.get("rotations",[[0,0,0]]),recipe.get("translations",[[0,0,0]])):
        if window is not None and radial==0 and angle!=azimuths[0]:
            continue
        total+=1
        frac=geometry(angle,rot,shift,axial,radial)
        clearance,lengths,error=evaluate(frac)
        row=dict(axial=axial,radial=radial,azimuth=angle,rotation=rot,translation=shift,min_nonbond=clearance,distances=lengths,
                 score=error-clearance,guest_fractional=frac.tolist())
        outside=0.
        if window is not None:
            outside=sum(max(lo-v,0.,v-hi) for v in lengths)
            row.pop("score")
            row["reference_deviation"]=sum(abs(v-r) for v,r in zip(lengths,ranking["reference_distances"],strict=True))
            row["violation"]=outside+max(0.,minimum-clearance)
        if outside>1e-12 or clearance<minimum or error>float(recipe.get("distance_tolerance",.02)):
            rejected+=1
            if clearance<minimum:
                collisions+=1
            else:
                coordination_rejected+=1
            if closest is None or ((row["violation"],-clearance)<(closest["violation"],-closest["min_nonbond"])
                                   if window is not None else clearance>closest["min_nonbond"]):
                closest=row
        else:
            rows.append(row)
            rows.sort(key=window_rank if window is not None else (lambda x:x["score"]))
            del rows[top:]
    if not total:
        raise ValueError("扫描网格为空")
    report=dict(total=total,rejected=rejected,collisions=collisions,coordination_rejected=coordination_rejected,top=rows)
    if window is not None:
        report["window_params"]=ranking
    if not rows:
        report["closest"]=closest
        return report,None
    selection=recipe.get("select",1 if total-rejected==1 else None)
    if type(selection)!=int or not 1<=selection<=len(rows):
        return report,None
    selected=rows[selection-1]
    frac=geometry(selected["azimuth"],selected["rotation"],selected["translation"],selected["axial"],selected["radial"])
    n=len(host.coords)
    wrap=np.floor(frac).astype(int)
    bonds=host.edges.copy()
    for j,s in enumerate(symbols):
        for k in range(j):
            cutoff=1.15*(covalent_radii[atomic_numbers[s]]+covalent_radii[atomic_numbers[symbols[k]]])
            if np.linalg.norm(local[j]-local[k])<=cutoff:
                bonds.add(edge(n+k,n+j,wrap[j]-wrap[k]))
    if coordinated:
        for i,t,_ in targets:
            bonds.add(edge(i,n+anchor,wrap[anchor]-t))
    model=model_at(host.cell,host.symbols+symbols,np.vstack([host.coords,frac-wrap]),bonds)
    report["selected"]=selection
    return report,(BondGraph(model),list(range(1,n+1))+[None]*len(mol))
