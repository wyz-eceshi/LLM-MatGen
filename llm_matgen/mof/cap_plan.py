"""按实际断开的金属配位键放置中性水或 OH⁻，并检查周期非键接触。"""
from collections import defaultdict
import math
import re
from itertools import product

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from pymatgen.core import Element

from .ledger import _removed_charge, charge_ledger
from .placement import unit


def _explicit_caps(spec, sites, graph):
    groups = defaultdict(list)
    for i, site in enumerate(sites):
        groups[site["metal_parent"]].append(i)
    result = [None]*len(sites)
    for item in spec.split(";"):
        match = re.fullmatch(r"(?:([A-Z][a-z]?))?([1-9][0-9]*)=(OH|H2O)((?:,(?:OH|H2O))*)",item.strip())
        if not match:
            raise ValueError("封端计划应为 site:金属编号=OH,H2O;...")
        symbol, idx, first, rest = match.groups()
        idx = int(idx)
        ids = groups.pop(idx, [])
        caps = (first+rest).split(",")
        if len(caps) != len(ids) or (symbol and graph.symbols[idx-1] != symbol):
            raise ValueError("指定的金属、空位数或封端数与断键清单不符")
        for i,cap in zip(ids,caps,strict=True):
            result[i] = cap
    if groups:
        raise ValueError("仍有空位未指定封端")
    return result


def describe_plan(graph, drop, recipe):
    removed = set(drop)
    declared = recipe.get("formal_charges", {})
    if not isinstance(declared,dict):
        raise ValueError("formal_charges 须为分项电荷")
    parent_q = declared.get("parent")
    known = _removed_charge(graph,{i+1 for i in removed})
    removed_q = declared.get("removed",known)
    if known is not None and known != removed_q:
        raise ValueError("删除部分的电荷与片段不符")
    if type(parent_q) is not int or type(removed_q) is not int:
        raise ValueError("混合封端需要已知的父结构电荷和删除片段电荷")
    bare = parent_q-removed_q
    if "uncapped_charge" in recipe and recipe["uncapped_charge"] != bare:
        raise ValueError("指定裸结构电荷与删除片段电荷不符")
    sites = []
    for i in sorted(removed):
        for metal, rev in graph.neighbors[i]:
            if metal in removed or not graph.elements[metal].is_metal:
                continue
            image = -np.asarray(rev,dtype=int)
            vector = (graph.coords[i]+image-graph.coords[metal]) @ graph.cell
            original = float(np.linalg.norm(vector))
            if "cap_distance" in recipe:
                length, source = float(recipe["cap_distance"]), "cap_distance"
            elif graph.symbols[i] == "O":
                length, source = original, "原金属–O 键"
            else:
                length = float(covalent_radii[atomic_numbers[graph.symbols[metal]]]+covalent_radii[atomic_numbers["O"]])
                source = "ASE 金属和氧的共价半径和"
            if not math.isfinite(length) or length <= 0:
                raise ValueError("封端参考键长须为正的有限数")
            sites.append(dict(metal_parent=metal+1, removed_parent=i+1, removed_image=image.tolist(),
                              direction=unit(vector).tolist(), broken_distance=original,
                              reference_distance=length, reference_source=source))
    sites.sort(key=lambda s:(s["metal_parent"],s["removed_parent"],s["removed_image"]))
    if not sites:
        raise ValueError("删除操作没有空出金属配位位点")
    # 以被删羧酸碳为参照。每个金属所在胞观察到的羧酸碳胞像可不同。
    carbon_sites = defaultdict(list)
    for idx, site in enumerate(sites):
        o = site["removed_parent"]-1
        if graph.symbols[o] != "O":
            continue
        carbons = [(j,t) for j,t in graph.neighbors[o] if j in removed and graph.symbols[j]=="C"]
        if len(carbons) == 1:
            c,offset = carbons[0]
            carbon_sites[c].append((idx,(np.array(site["removed_image"])+offset).tolist()))
    pairs = []
    for c,rows in sorted(carbon_sites.items()):
        oxygen_edges = [(j,t) for j,t in graph.neighbors[c] if graph.symbols[j]=="O"]
        if (len(rows) != 2 or len(oxygen_edges) != 2 or
                len({sites[i]["removed_parent"] for i,_ in rows}) != 2):
            continue
        pairs.append([i for i,_ in rows])
        for i,image in rows:
            sites[i].update(carboxyl_parent=c+1,carboxyl_image=image)
    plan = recipe["cap_plan"]
    if plan == "auto-charge":
        if not 0 <= bare <= len(sites):
            raise ValueError(f"裸结构电荷 {bare} 与 {len(sites)} 个空位无法用 OH⁻/H₂O 配平")
        caps = ["H2O"]*len(sites)
        unpaired = [i for i,s in enumerate(sites) if "carboxyl_parent" not in s]
        if not 0 <= bare-len(pairs) <= len(unpaired):
            raise ValueError("羧酸端每对一个 OH 和一个水无法与其余空位配平电荷")
        for i in [p[0] for p in pairs] + unpaired[:bare-len(pairs)]:
            caps[i] = "OH"
    elif isinstance(plan,str) and plan.startswith("site:"):
        caps = _explicit_caps(plan[5:],sites,graph)
    else:
        raise ValueError("未知封端计划")
    charge = charge_ledger(parent_q,removed_q,-caps.count("OH"))
    if charge["child"] != 0:
        raise ValueError(f"封端形式电荷未配平，产物电荷 {charge['child']}")
    for site,cap in zip(sites,caps,strict=True):
        site.update(cap=cap,formal_charge=-1 if cap=="OH" else 0)
        if graph.symbols[site["metal_parent"]-1] == "Zr" and "cap_distance" not in recipe:
            site.update(reference_distance=2.20 if cap == "OH" else 2.30,
                        reference_source="Zr–OH 2.20 Å / Zr–OH₂ 2.30 Å")
    charge["uncapped"] = bare
    return sites,charge


def _positions(site, origin, angle):
    u = np.array(site["direction"])
    v = unit(np.cross(u,np.eye(3)[np.argmin(abs(u))]))
    radians = np.radians(angle)
    transverse = np.cos(radians)*v + np.sin(radians)*np.cross(u,v)
    oxygen = origin + site["reference_distance"]*u
    radial = np.cos(np.radians(104.5/2))
    lateral = np.sin(np.radians(104.5/2))
    points = [oxygen,oxygen+.96*(radial*u+lateral*transverse)]
    if site["cap"] == "H2O":
        points.append(oxygen+.96*(radial*u-lateral*transverse))
    return np.array(points)


class Contacts:
    def __init__(self,lattice,symbols,scale):
        self.lattice = lattice
        radii = {s:Element(s).van_der_waals_radius for s in set(symbols)|{"H","O"}}
        if any(v is None for v in radii.values()):
            raise ValueError("封端碰撞检查缺少元素的范德华半径")
        self.floors = {(a,b):float(scale*(radii[a]+radii[b])) for a in radii for b in radii}
        # 任一封端原子与自身相邻胞像给出余量上界。
        self.radius = min(lattice.abc)+max(self.floors.values())

    def minimum(self,points,ids,elements,other_points,other_ids,other_elements,excluded,kind):
        """周期像相对两端传入的连续坐标，用同一坐标系判断拓扑排除。"""
        frac = self.lattice.get_fractional_coords(other_points)
        wraps = np.floor(frac).astype(int)
        best = None
        for p,number,symbol in zip(points,ids,elements,strict=True):
            origin_image = np.floor(self.lattice.get_fractional_coords(p)).astype(int)
            near = self.lattice.get_points_in_sphere(frac%1,p,self.radius)
            for _,distance,j,image in near:
                j = int(j)
                partner = other_ids[j]
                shift = tuple(map(int,image-wraps[j]))
                if (number,partner,shift) in excluded:
                    continue
                floor = self.floors[symbol,other_elements[j]]
                margin = float(distance)-floor
                if best is None or margin < best["margin"]:
                    best = dict(category=kind, atoms=[number,partner], image=(image-origin_image).astype(int).tolist(),
                                pose_image=list(shift),
                                elements=[symbol,other_elements[j]],distance=float(distance),limit=floor,margin=margin)
        return best


def _worst(*items):
    return min((r for r in items if r is not None),key=lambda r:r["margin"],default=None)


def place_caps(graph,host,mapping,drop,recipe):
    sites,charge = describe_plan(graph,drop,recipe)
    scale = float(recipe.get("cap_vdw_scale",.75))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("范德华碰撞系数须为正的有限数")
    checker = Contacts(host.structure.lattice,host.symbols,scale)
    from .cap_cone import search_caps
    groups = defaultdict(list)
    if recipe["cap_plan"] == "auto-charge":
        for i,site in enumerate(sites):
            if "carboxyl_parent" in site:
                groups[site["carboxyl_parent"]].append(i)
    alternatives = []
    best_key = best = None
    for swap in product((False,True),repeat=len(groups)):
        assigned = [s.copy() for s in sites]
        for change,(i,j) in zip(swap,groups.values(),strict=True):
            if change:
                for field in ("cap","reference_distance","formal_charge"):
                    assigned[i][field],assigned[j][field] = assigned[j][field],assigned[i][field]
        result = search_caps(host,mapping,drop,assigned,charge,checker,scale)
        geometry = result[0]["cap_geometry"]
        limits = geometry["limits"]
        keys = ("distance","oo","hbond","mean_DHA","near_HA") if geometry["feasible"] else ("margin",)
        rank = (geometry["feasible"],*(round(limits[k],10) for k in keys if k in limits))
        alternatives.append(dict(caps=[s["cap"] for s in assigned],feasible=geometry["feasible"],
                                 limits=limits,combinations=geometry["joint_combinations"]))
        if best_key is None or rank > best_key:
            best_key,best = rank,result
    geometry = best[0]["cap_geometry"]
    geometry["assignments"] = alternatives
    geometry["all_combinations"] = sum(a["combinations"] for a in alternatives)
    return best
