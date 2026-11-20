"""平面环拟合；残差权重沿用 Cu-FDC 建模方法。"""
import numpy as np
from scipy.optimize import least_squares
from .fragment_ops import omit,sdf_template,append_template
from .placement import unit
from .ring_substituents import parent_substituents


def _substituents(order, neighbors, omitted):
    """按模板键划分环外片段，记录各片段连接的环位。"""
    ring_sites = set(order)
    pending = set(range(len(neighbors))) - ring_sites - omitted
    groups = []
    while pending:
        members = {pending.pop()}
        stack = list(members)
        roots = set()
        while stack:
            for other in neighbors[stack.pop()]:
                if other in ring_sites:
                    roots.add(other)
                elif other in pending:
                    pending.remove(other)
                    members.add(other)
                    stack.append(other)
        groups.append((sorted(members), sorted(roots)))
    return groups


def _proper_rotation(source, target):
    """行向量的最佳刚性旋转，不镜像环外手性中心。"""
    u, _, vt = np.linalg.svd(source.T @ target)
    u[:, -1] *= np.linalg.det(u @ vt)
    return u @ vt


def _fit_ring_candidates(g,path,recipe):
    """给出两种几何的测量结果，以及按同一拟合参数重新构建的函数。"""
    old=[i-1 for i in recipe["ring"]]
    anchors=[i-1 for i in recipe["anchors"]]
    order=[i-1 for i in recipe["template_ring"]]
    ports=[i-1 for i in recipe["ports"]]
    n=len(old)
    syms,local,pairs=sdf_template(path)
    if (n<3 or len(order)!=n or len(set(old))!=n or len(set(order))!=n or
        len(ports)!=2 or len(set(ports))!=2 or len(anchors)!=2 or
        any(not 0<=i<len(g.symbols) for i in old+anchors) or
        any(not 0<=i<len(syms) for i in order) or any(not 0<=i<n for i in ports)):
        raise ValueError("环或锚点编号无效")
    def link(i,j):
        offsets=[np.asarray(t) for k,t in g.neighbors[i] if k==j]
        if len(offsets)!=1:
            raise ValueError("环路径或锚点键不唯一")
        return offsets[0]
    images=[np.zeros(3,int)]
    for i,j in zip(old,old[1:]):
        images.append(images[-1]+link(i,j))
    if np.any(images[-1]+link(old[-1],old[0])):
        raise ValueError("所选路径跨胞后未闭合")
    oldxyz=(g.coords[old]+images)@g.cell
    image_a=np.array([images[k]+link(old[k],a) for k,a in zip(ports,anchors)])
    anc=(g.coords[anchors]+image_a)@g.cell
    drop, branches = parent_substituents(g,old,anchors,ports,images,order,recipe)
    kept = [branch for branch in branches if branch.action == "keep"]
    host,mapping=omit(g,drop)
    neighbors=[set() for _ in syms]
    for a,b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    if any(order[(k+1)%n] not in neighbors[i] for k,i in enumerate(order)):
        raise ValueError("模板环不闭合")
    # 两个成键端口上的单键 H 让位于父结构锚点；逐个给出模板编号与原因。
    omitted = {h: f"模板环位 {order[k]+1} 的 H 由父结构锚点键替代"
               for k in ports for h in neighbors[order[k]]
               if syms[h] == "H" and neighbors[h] == {order[k]}}
    handling = []
    for branch in branches:
        entry = dict(action=branch.action, parent_root=branch.root+1,
                     parent_ring=branch.parent_ring+1, atoms=[i+1 for i in branch.members],
                     parent_image=branch.parent_image.tolist())
        if branch.action == "keep":
            site = branch.template_ring
            outside = neighbors[site] - set(order)
            if len(outside) > 1 or any(syms[j] != "H" or neighbors[j] != {site} for j in outside):
                raise ValueError("保留取代基的模板环位已有取代基或离去 H 不唯一")
            for h in outside:
                omitted[h] = f"模板环位 {site+1} 的 H 由保留的父取代基 {branch.root+1} 替代"
            entry.update(template_ring=site+1, discarded_template_H=[j+1 for j in sorted(outside)])
        handling.append(entry)
    groups = _substituents(order, neighbors, set(omitted))
    # 已支持的芳香环和其 H 保持旧的产物编号；新增的环外片段随后按模板序写出。
    direct_h = [h for i in order for h in sorted(neighbors[i])
                if syms[h] == "H" and neighbors[h] == {i} and h not in omitted and h not in order]
    prefix = order + direct_h
    remaining = set(range(len(syms))) - set(prefix) - set(omitted)
    written = prefix + sorted(remaining)
    ix = {atom: child for child, atom in enumerate(written)}
    symbols = [syms[i] for i in written]
    links = [(ix[a],ix[b]) for a,b in pairs if a in ix and b in ix]
    origin=oldxyz.mean(axis=0)
    plane=np.linalg.svd(oldxyz-origin)[2][:2].T
    target=(oldxyz-origin)@plane
    raw=local[order]-local[order].mean(axis=0)
    template=raw@np.linalg.svd(raw)[2][:2].T
    def lengths(x):
        return np.linalg.norm(x-np.roll(x,-1,axis=0),axis=1)
    def angles(x):
        return np.array([unit(x[k-1]-x[k])@unit(x[(k+1)%n]-x[k]) for k in range(n)])
    ref_len=lengths(template)
    ref_ang=angles(template)
    goals=np.linalg.norm(oldxyz[ports]-anc,axis=1)
    lattice=g.structure.lattice
    extra=float(recipe.get("extra_clearance",0.))
    def build(reverse):
        # 反向拟合需将模板的第一个端口对到父环第二个端口，反之亦然。
        seq=[(sum(ports)-k)%n for k in range(n)] if reverse else list(range(n))
        ends=anchors[::-1] if reverse else anchors
        shifts=image_a[::-1] if reverse else image_a
        fixed=anc[::-1] if reverse else anc
        desired=goals[::-1] if reverse else goals
        tar=target[seq]
        u,_,v=np.linalg.svd(template.T@tar)
        guess=template@u@v+tar.mean(axis=0)
        def residual(flat):
            x=flat.reshape(n,2)
            xyz=x@plane.T+origin
            end=xyz[ports]
            axis=[unit(a-xyz[k])@unit((xyz[k-1]+xyz[(k+1)%n])/2-xyz[k]) for a,k in zip(fixed,ports)]
            return np.r_[8*(lengths(x)-ref_len),1.6*(angles(x)-ref_ang),
                         7*(np.linalg.norm(end-fixed,axis=1)-desired),1.6*(np.array(axis)+1),.35*(x-tar).ravel()]
        fit=least_squares(residual,guess.ravel(),max_nfev=450)
        ring=fit.x.reshape(n,2)@plane.T+origin
        rotation = _proper_rotation(raw, ring - ring.mean(axis=0))
        positioned = np.empty_like(local)
        positioned[order] = ring
        for members, roots in groups:
            if len(roots) == 1 and len(members) == 1 and syms[members[0]] == "H":
                # 沿用芳香环 H 在相邻两根环键外角平分线上的约定。
                k = ix[roots[0]]
                outward = -unit(unit(ring[k-1]-ring[k])+unit(ring[(k+1)%n]-ring[k]))
                positioned[members[0]] = ring[k] + outward*np.linalg.norm(local[members[0]]-local[roots[0]])
            else:
                # 环可轻微变形；环外片段刚性随动，以成键环位为原点。
                reference = roots or order
                src = local[reference].mean(axis=0)
                dst = positioned[reference].mean(axis=0)
                positioned[members] = (local[members] - src) @ rotation + dst
        points = positioned[written]
        free=np.asarray(points)@np.linalg.inv(g.cell)
        bridges = list(zip(ports,ends,shifts))
        for branch in kept:
            bridges.append((ix[branch.template_ring],branch.root,branch.parent_image))
        matches = all(old[seq[ix[branch.template_ring]]] == branch.parent_ring for branch in kept)
        keep_lengths = [float(np.linalg.norm(positioned[b.template_ring] -
                        (g.coords[b.root]+b.parent_image) @ g.cell)) for b in kept]
        min_gap=float("inf")
        closest_contact = None
        for k,(s,pos) in enumerate(zip(symbols,points)):
            exclude=set()
            for j,a,t in bridges:
                if k==j or written[k] in neighbors[written[j]]:
                    exclude.add((a,tuple(t)))
                if k==j:
                    exclude.update((b,tuple(t+d)) for b,d in g.neighbors[a] if b not in drop)
            for _,dist,j,t in lattice.get_points_in_sphere(host.coords,pos,4.+max(extra,0)):
                parent=mapping[int(j)]-1
                if (parent,tuple(t)) in exclude:
                    continue
                other=g.symbols[parent]
                floor=(1.50 if other=="H" else 1.64) if s=="H" else (1.64 if other=="H" else 2.56 if other=="Cu" else 2.18)
                margin = float(dist) - floor - extra
                if margin < min_gap:
                    min_gap = margin
                    closest_contact = dict(template_index=written[k]+1, parent_index=parent+1,
                                           parent_image=np.asarray(t,dtype=int).tolist(),
                                           elements=[s,other], distance=float(dist), limit=floor+extra)
            for _,dist,j,t in lattice.get_points_in_sphere(free%1,pos,4.+max(extra,0)):
                if np.any(t!=np.floor(free[int(j)])):
                    other=symbols[int(j)]
                    floor=1.5 if s==other=="H" else 1.64 if "H" in (s,other) else 2.18
                    margin = float(dist) - floor - extra
                    if margin < min_gap:
                        min_gap = margin
                        closest_contact = dict(template_index=written[k]+1,
                                               image_template_index=written[int(j)]+1,
                                               relative_image=(np.asarray(t,dtype=int)-np.floor(free[int(j)]).astype(int)).tolist(),
                                               elements=[s,other], distance=float(dist), limit=floor+extra)
        nit=[i for i,s in enumerate(symbols) if s=="N"]
        cu=[i for i,s in enumerate(host.symbols) if s=="Cu"]
        nearest=float(lattice.get_all_distances(free[nit],host.coords[cu]).min()) if cu and nit else None
        error=float(max(abs(lengths(ring)-ref_len)))
        row=dict(reverse=reverse,cost=float(fit.cost),fit_error=error,min_margin=min_gap,closest_contact=closest_contact,
                 anchor_lengths=np.linalg.norm(ring[ports]-fixed,axis=1).tolist(),nearest_Cu_N=nearest,
                 feasible=bool(matches and fit.success and min_gap>=0 and error<=recipe.get("fit_limit",.025)))
        if kept:
            row.update(keep_mapping_matches=matches, keep_bond_lengths=keep_lengths)
        attachments=[(mapping.index(a+1),t,k) for k,a,t in bridges]
        return row,(points,attachments)
    rows=[build(reverse)[0] for reverse in (False,True)]
    report=dict(candidates=rows,removed_atoms=[i+1 for i in sorted(drop)],closest=max(rows,key=lambda r:r["min_margin"]))
    report["template_indices"] = [i+1 for i in written]
    report["template_discarded"] = [
        {"index": h+1, "reason": reason}
        for h, reason in sorted(omitted.items())
    ]
    if handling:
        report["substituents"] = handling
    def rebuild(row):
        _, (points,attachments) = build(row["reverse"])
        return append_template(host,mapping,symbols,points,links,attachments)
    return report, rebuild


def replace_ring(g,path,recipe):
    report,rebuild = _fit_ring_candidates(g,path,recipe)
    valid = [r for r in report["candidates"] if r["feasible"]]
    if not valid:
        return report,None
    chosen=recipe.get("orientation")
    if chosen is None and len(valid)>1:
        return report,None
    if chosen=="lowest-cost" or chosen is None:
        best=min(valid,key=lambda r:r["cost"])
    else:
        best=next((r for r in valid if r["reverse"]==chosen),None)
        if best is None:
            return report,None
    report["selected"]=best
    return report,rebuild(best)
