"""羧酸缺陷封端锥扫。每个碰撞/距离约束只涉及一个或两个封端。"""
from itertools import combinations
import math

import numpy as np
from scipy.spatial.distance import cdist

from .cap_plan import _positions, _worst
from .cif import edge
from .graph import BondGraph
from .placement import unit
from .subnet import model_at


def _poses(site, origin):
    axis = np.asarray(site["direction"])
    v = unit(np.cross(axis,np.eye(3)[np.argmin(abs(axis))]))
    w = np.cross(axis,v)
    settings,points = [],[]
    for tilt in range(0,21,5):
        for az in (range(0,360,30) if tilt else [0]):
            a,b = np.radians([tilt,az])
            direction = np.cos(a)*axis+np.sin(a)*(np.cos(b)*v+np.sin(b)*w)
            for spin in range(0,360 if site["cap"]=="OH" else 180,30):
                settings.append((tilt,az,spin,direction))
                points.append(_positions(dict(site,direction=direction),origin,spin))
    return dict(settings=settings,points=np.asarray(points),origin=origin,
                symbols=["O","H"] if site["cap"]=="OH" else ["O","H","H"])


def _images(lattice, first, second, radius):
    """返回使第二个中心落入第一个中心给定球内的全部整数平移。"""
    frac = lattice.get_fractional_coords(second)
    wrap = np.floor(frac).astype(int)
    return [np.asarray(t,dtype=int)-wrap for _,_,_,t in
            lattice.get_points_in_sphere([frac%1],first,radius)]


def _local(checker,host,site,data):
    # 取球半径覆盖所有网格点可能达到的距离上界及原子相对金属的最大偏移。
    points = data["points"]
    extent = np.linalg.norm(points-data["origin"],axis=2).max()
    data["extent"] = extent
    near = checker.lattice.get_points_in_sphere(host.coords%1,data["origin"],checker.radius+extent)
    host_wraps = np.floor(host.coords).astype(int)
    metal = site["metal_child"]-1
    oxygen_exclude = {(j,tuple(t)) for j,t in host.neighbors[metal]}
    d = np.full(len(points),checker.radius)
    margin = d.copy()
    for k,s in enumerate(data["symbols"]):
        retained = [(f,int(j)) for f,_,j,t in near if
                    (int(j),tuple(t-host_wraps[int(j)])) not in
                    ({(metal,(0,0,0))} | (oxygen_exclude if k==0 else set()))]
        if retained:
            frac,indices = zip(*retained)
            dist = cdist(points[:,k],np.asarray(frac) @ host.cell)
            floors = [checker.floors[s,host.symbols[j]] for j in indices]
            d = np.minimum(d,dist.min(axis=1))
            margin = np.minimum(margin,(dist-floors).min(axis=1))
    for shift in _images(checker.lattice,data["origin"],data["origin"],checker.radius+2*extent):
        if not shift.any():
            continue
        for k,s in enumerate(data["symbols"]):
            for l,t in enumerate(data["symbols"]):
                dist = np.linalg.norm(points[:,k]-points[:,l]-shift@host.cell,axis=1)
                d = np.minimum(d,dist)
                margin = np.minimum(margin,dist-checker.floors[s,t])
    return {"distance":d,"margin":margin}


def _pair(checker,first,second,p,q):
    a,b = p["points"],q["points"]
    d = np.full((len(a),len(b)),checker.radius)
    margin = d.copy()
    limit = checker.radius + p["extent"]+q["extent"]
    for image in _images(checker.lattice,p["origin"],q["origin"],limit):
        shift = image @ checker.lattice.matrix
        for k,s in enumerate(p["symbols"]):
            for l,t in enumerate(q["symbols"]):
                if k==l==0 and first["metal_child"]==second["metal_child"] and not image.any():
                    continue
                dist = cdist(a[:,k],b[:,l]+shift)
                np.minimum(d,dist,out=d)
                np.minimum(margin,dist-checker.floors[s,t],out=margin)
    table = {"distance":d,"margin":margin}
    if "carboxyl_parent" in first and first["carboxyl_parent"]==second.get("carboxyl_parent"):
        shift = (np.array(first["carboxyl_image"])-second["carboxyl_image"]) @ checker.lattice.matrix
        oo = cdist(a[:,0],b[:,0]+shift)
        table["oo"] = -np.maximum.reduce([2.5-oo,oo-2.8,np.zeros_like(oo)])
        # 近似氢键不设截断；朝向连续变化时，所有 H 均参与择优。
        angle = np.zeros_like(d)
        all_angle = np.zeros_like(d)
        nearest = np.full_like(d,np.inf)
        for donor,acceptor,offset,reverse in ((a,b,shift,False),(b,a,-shift,True)):
            for k in range(1,donor.shape[1]):
                other = acceptor[None,:,0,:]+offset-donor[:,None,k,:]
                ha = np.linalg.norm(other,axis=-1)
                oh = donor[:,0]-donor[:,k]
                cosines = np.einsum("ik,ijk->ij",oh,other) / (
                    np.linalg.norm(oh,axis=1)[:,None]*np.maximum(ha,1e-12))
                deg = np.degrees(np.arccos(np.clip(cosines,-1.,1.)))
                score = deg.T if reverse else deg
                angle = np.maximum(angle,score)
                all_angle += score
                nearest = np.minimum(nearest,ha.T if reverse else ha)
        table["hbond"] = angle
        table["mean_DHA"] = all_angle/(a.shape[1]+b.shape[1]-2)
        table["near_HA"] = -nearest
        table["oxygen_distance"] = oo
    return table


def _assignment(local,pairs,limits):
    def mask(table):
        return np.logical_and.reduce([table[k] >= limit-1e-10 for k,limit in limits.items() if k in table])
    domains = {i:np.flatnonzero(mask(t)) for i,t in enumerate(local)}
    allowed = {ij:mask(t) for ij,t in pairs.items()}
    def search(domains):
        # 删除在任一其余位点都找不到兼容取向的值，直到不再变化。
        domains = domains.copy()
        changed = True
        while changed:
            if any(not len(ids) for ids in domains.values()):
                return None
            changed = False
            for (i,j),matrix in allowed.items():
                a,b = domains[i],domains[j]
                support = matrix[np.ix_(a,b)]
                good_a,good_b = a[support.any(axis=1)], b[support.any(axis=0)]
                if not len(good_a) or not len(good_b):
                    return None
                if len(good_a)<len(a) or len(good_b)<len(b):
                    domains.update({i:good_a,j:good_b})
                    changed = True
        remaining = [(len(v),i) for i,v in domains.items() if len(v)>1]
        if not remaining:
            return [int(domains[i][0]) for i in range(len(domains))]
        _,i = min(remaining)
        for v in domains[i]:
            child = domains.copy()
            child[i] = np.array([v])
            answer = search(child)
            if answer is not None:
                return answer
        return None
    return search(domains)


def _maximize(key,local,pairs,limits):
    # 瓶颈只可能取某个单点或点对表中的值；二分每个离散阈值并完整求解兼容性。
    arrays = [t[key] for t in [*local,*pairs.values()] if key in t]
    upper = min(float(x.max()) for x in arrays)
    values = np.unique(np.concatenate([x[x<=upper+1e-10] for x in arrays]))
    low,high = 0,len(values)-1
    best = None
    while low<=high:
        mid = (low+high)//2
        idx = _assignment(local,pairs,dict(limits,**{key:values[mid]}))
        if idx is None:
            high = mid-1
        else:
            best = float(values[mid]),idx
            low = mid+1
    if best is None:
        raise ValueError("封端几何离散约束无共同解")
    return best


def search_caps(host,mapping,drop,sites,charge,checker,scale):
    poses,local = [],[]
    first = len(mapping)+1
    host_xyz = host.coords @ host.cell
    for site in sites:
        m = mapping.index(site["metal_parent"])
        p = _poses(site,host_xyz[m])
        ids = list(range(first,first+len(p["symbols"])))
        first += len(ids)
        site.update(metal_child=m+1,cap_indices=ids)
        poses.append(p)
        local.append(_local(checker,host,site,p))
    pairs = {(i,j):_pair(checker,sites[i],sites[j],poses[i],poses[j]) for i,j in combinations(range(len(sites)),2)}
    limits = {"margin":0.}
    feasible = _assignment(local,pairs,limits) is not None
    if feasible:
        targets = ["distance"]
        if any("oo" in table for table in pairs.values()):
            targets.extend(("oo","hbond","mean_DHA","near_HA"))
        for key in targets:
            limits[key],selected = _maximize(key,local,pairs,limits)
    else:
        limits["margin"],selected = _maximize("margin",local,pairs,{})
    symbols,coords,bonds = list(host.symbols),list(host.coords),set(host.edges)
    host_min,self_min,between_min = None,None,None
    chosen = []
    for site,p,idx in zip(sites,poses,selected,strict=True):
        tilt,az,spin,direction = p["settings"][idx]
        site.update(tilt_degrees=tilt,azimuth_degrees=az,angle_degrees=spin,actual_direction=direction.tolist())
        xyz = _positions(dict(site,direction=direction),p["origin"],spin)
        chosen.append(xyz)
        frac = checker.lattice.get_fractional_coords(xyz)
        wraps = np.floor(frac).astype(int)
        m = site["metal_child"]-1
        ids = site["cap_indices"]
        k = len(coords)
        bonds.add(edge(m,k,wraps[0]))
        bonds.update(edge(k,k+j,wraps[j]-wraps[0]) for j in range(1,len(xyz)))
        symbols.extend(p["symbols"])
        coords.extend(frac-wraps)
        excluded = {(number,m+1,(0,0,0)) for number in ids}
        excluded.update((ids[0],j+1,tuple(t)) for j,t in host.neighbors[m])
        host_min = _worst(host_min,checker.minimum(xyz,ids,p["symbols"],host_xyz,list(range(1,len(host_xyz)+1)),
                                                   host.symbols,excluded,"封端–宿主"))
        own = {(a,b,(0,0,0)) for a in ids for b in ids}
        self_min = _worst(self_min,checker.minimum(xyz,ids,p["symbols"],xyz,ids,p["symbols"],own,"封端自身周期像"))
    grouping = []
    for (i,j),table in pairs.items():
        s,t = sites[i],sites[j]
        excluded = {(s["cap_indices"][0],t["cap_indices"][0],(0,0,0))} if s["metal_child"]==t["metal_child"] else set()
        between_min = _worst(between_min,checker.minimum(chosen[i],s["cap_indices"],poses[i]["symbols"],
                            chosen[j],t["cap_indices"],poses[j]["symbols"],excluded,"不同封端"))
        if "oo" in table:
            key = selected[i],selected[j]
            image = np.array(s["carboxyl_image"])-t["carboxyl_image"]
            hs = []
            for donor,acceptor,shift in ((i,j,image),(j,i,-image)):
                xyz = chosen[donor]
                acc = chosen[acceptor][0] + shift @ host.cell
                hydrogen_ids = sites[donor]["cap_indices"][1:]
                for hydrogen,number in zip(xyz[1:],hydrogen_ids,strict=True):
                    ha = acc-hydrogen
                    da = xyz[0]-hydrogen
                    angle = np.degrees(np.arccos(np.clip(unit(da) @ unit(ha),-1,1)))
                    hs.append(dict(donor_O=sites[donor]["cap_indices"][0],H=number,
                                   acceptor_O=sites[acceptor]["cap_indices"][0],
                                   HA_distance=float(np.linalg.norm(ha)),DHA_degrees=float(angle),
                                   acceptor_pose_image=shift.tolist()))
            grouping.append(dict(carboxyl_parent=s["carboxyl_parent"],metal_parent=[s["metal_parent"],t["metal_parent"]],
                                 caps=[s["cap"],t["cap"]],oxygen_distance=float(table["oxygen_distance"][key]),
                                 best_DHA_degrees=float(table["hbond"][key]),hydrogens=hs))
    closest = _worst(host_min,self_min,between_min)
    if feasible and closest["margin"] < -1e-9:
        raise ValueError("重建后的碰撞检查与锥扫不同")
    report = dict(removed_atoms=sorted(i+1 for i in drop),caps=len(sites),cap_sites=sites,cap_charge=charge,
                  cap_geometry=dict(vdw_source="pymatgen Element.van_der_waals_radius",vdw_scale=scale,feasible=feasible,
                     radial_step_degrees=5,max_tilt_degrees=20,azimuth_step_degrees=30,rotation_step_degrees=30,
                     variants=[len(p["points"]) for p in poses],joint_combinations=math.prod(len(p["points"]) for p in poses),
                     limits=limits,objective="无碰撞时最大化最短非键距离；并列时依次选择各对 O···O 偏离 2.5–2.8 Å 的最大值最小、各对最佳 DHA 角最小值最大、各对平均 DHA 角最小值最大、各对最近 H···O 距离最大值最小",
                     carboxyl_pairs=grouping,closest=closest,host=host_min,self_image=self_min,between_caps=between_min))
    if not feasible:
        report["reason"] = "20° 锥扫的全部组合均有非键碰撞"
        report["closest"] = dict(contact=closest,
                                 angles=[[s["tilt_degrees"],s["azimuth_degrees"],s["angle_degrees"]] for s in sites])
        return report,None
    return report,(BondGraph(model_at(host.cell,symbols,coords,bonds)),mapping+[None]*(len(coords)-len(mapping)))
