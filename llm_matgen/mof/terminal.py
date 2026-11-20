"""端位配体的金属轴向扫描，距离、倾角、方位角和自转角共同枚举。"""
from itertools import product
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .placement import unit,window_parameters,window_rank
from .fragment_ops import sdf_template,append_template


FLOORS={tuple(sorted(k.split("-"))):v for k,v in {
    "H-H":1.60,"H-C":1.90,"H-N":1.75,"H-O":1.65,"H-Cu":1.85,"C-C":2.64,"C-N":2.60,
    "C-O":2.57,"C-Cu":2.53,"N-N":2.56,"N-O":2.53,"N-Cu":2.65}.items()}


def scan_axis(host,path,recipe):
    symbols,coords,bonds=sdf_template(path)
    site=recipe["site"]
    metal=site["index"]-1
    if not 0<=metal<len(host.symbols) or not host.elements[metal].is_metal:
        raise ValueError("端位需要金属位点")
    donor=recipe["anchor"]-1
    if not 0<=donor<len(symbols):
        raise ValueError("模板供体越界")
    near=[b if a==donor else a for a,b in bonds if donor in (a,b)]
    if not near:
        raise ValueError("供体无模板键")
    x=unit(coords[near].mean(axis=0)-coords[donor])
    heavy=coords[[s!="H" for s in symbols]]
    normal=np.linalg.svd(heavy-heavy.mean(axis=0))[2][-1]
    y=unit(np.cross(normal,x))
    local=(coords-coords[donor])@np.stack([x,y,np.cross(x,y)]).T
    axis=unit(np.array(site["direction"]))
    up=unit(np.cross(host.cell[0],host.cell[1]))
    if abs(up@axis)>.999:
        up=np.eye(3)[np.argmin(abs(axis))]
    tx=unit(np.cross(axis,up))
    ty=np.cross(axis,tx)
    center=host.coords[metal]@host.cell
    lattice=host.structure.lattice
    radii=recipe.get("lengths",[2.12,2.18,2.24,2.28,2.32,2.36])
    window=recipe.get("distance_window")
    if window is not None:
        if "lengths" in recipe or len(window)!=2 or not 0<window[0]<=window[1]:
            raise ValueError("端配体窗口与距离列表互斥")
        ranking=window_parameters(recipe,[host.symbols[metal]],symbols[donor])
        radii=np.linspace(*window,15).tolist()
    threshold=recipe.get("min_margin",0.)
    if not radii or min(radii)<=0 or not np.isfinite(radii).all() or not np.isfinite(threshold) or threshold<0:
        raise ValueError("配位距离或间距下限无效")
    radius=max(np.linalg.norm(local,axis=1))
    search_distance=5.+threshold
    if window is not None:
        # 任一客体原子与自身最近胞像的距离至多为最短胞矢长。
        # 全部候选的次排序键都能在这个半径内精确求出。
        search_distance=max(search_distance,min(lattice.abc))
    cloud=lattice.get_points_in_sphere(host.coords%1,center,radius+max(radii)+search_distance)
    pos=np.array([row[0] for row in cloud])@host.cell
    elems=np.array([host.symbols[int(row[2])] for row in cloud])
    types=set(symbols)
    groups=[]
    for typ in set(elems):
        mask=elems==typ
        tree=cKDTree(pos[mask])
        others=pos[mask & (np.linalg.norm(pos-center,axis=1)>1e-7)]
        for s in types:
            ids=np.flatnonzero(np.array(symbols)==s)
            floor=FLOORS.get(tuple(sorted((s,typ))),recipe.get("min_nonbond"))
            if floor is None:
                raise ValueError("需为未登记元素对指定 min_nonbond")
            if donor in ids and typ==host.symbols[metal]:
                if len(others):
                    groups.append(([donor],cKDTree(others),floor))
                ids=ids[ids!=donor]
            if len(ids):
                groups.append((ids,tree,floor))
    images=[np.array(im)@host.cell for _,_,_,im in lattice.get_points_in_sphere([[0,0,0]],[0,0,0],2*radius+(search_distance if window is not None else 2.5+threshold)) if np.any(im)]
    spins=recipe.get("spins",list(range(0,360,10)))
    def geometry(length,tilt,azimuth,spins):
        az=np.radians(azimuth)
        ang=np.radians(tilt)
        align=np.cos(ang)*axis+np.sin(ang)*(np.cos(az)*tx+np.sin(az)*ty)
        ey=unit(np.cross(align,tx))
        base=local@np.array([align,ey,np.cross(align,ey)])
        rotate=Rotation.from_rotvec(np.radians(spins)[:,None]*align).as_matrix()
        return np.einsum("sij,nj->sni",rotate,base)+center+length*align
    total=rejected=0
    closest=None
    best=None
    for length,tilt in product(radii,recipe.get("tilts",[0,7,14,21,28])):
        for az in (recipe.get("azimuths",list(range(0,360,45))) if tilt else [0]):
            points=geometry(length,tilt,az,spins)
            gap=np.full(len(spins),np.inf)
            clearance=np.full(len(spins),np.inf)
            for ids,tree,floor in groups:
                nearest=tree.query(points[:,ids,:])[0].min(axis=1)
                gap=np.minimum(gap,nearest-floor)
                clearance=np.minimum(clearance,nearest)
            for image in images:
                distances=np.linalg.norm(points[:,:,None,:]-points[:,None,:,:]-image,axis=3)
                nearest=distances.min(axis=(1,2))
                gap=np.minimum(gap,nearest-2.)
                clearance=np.minimum(clearance,nearest)
            for spin,margin,nonbond in zip(spins,gap,clearance,strict=True):
                total+=1
                row=dict(length=length,tilt=tilt,azimuth=az,spin=spin,margin=float(margin))
                if window is None:
                    row["score"]=float(min(margin,.45)-.0018*tilt-.20*abs(length-2.24))
                else:
                    row.update(min_nonbond=float(nonbond),reference_deviation=abs(length-ranking["reference_distances"][0]))
                if closest is None or margin>closest["margin"]:
                    closest=row
                if margin<threshold:
                    rejected+=1
                elif best is None or (window_rank(row)<window_rank(best) if window is not None else row["score"]>best["score"]):
                    best=row
    report=dict(total=total,rejected=rejected,closest=closest,selected=best)
    if window is not None:
        report["window_params"]=ranking
    if best is None:
        return report,None
    xyz=geometry(best["length"],best["tilt"],best["azimuth"],[best["spin"]])[0]
    return report,append_template(host,list(range(1,len(host.symbols)+1)),symbols,xyz,bonds,[(metal,np.zeros(3,int),donor)])
