"""两个距离窗口下的周期刚体放置。球面交线给出位置，模板内部距离不变。"""
from itertools import combinations, product

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .donor import (align_to, atom_ids, displaced_parts, exclusions, guest_template,
                    scan_directions)
from .fragment_ops import append_template
from .placement import unit, window_parameters, window_rank


def anchor_specs(recipe):
    raw = recipe["anchors"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("anchors 需要两组 客体编号~宿主编号:下限,上限")
    result = []
    for item in raw:
        if isinstance(item, str):
            atoms, window = item.split(":")
            guest, host = atoms.split("~")
            item = dict(guest=guest, host=host, window=window.split(","))
        if not isinstance(item, dict) or not {"guest","host","window"} <= item.keys():
            raise ValueError("每组 anchors 都须给出 guest、host 和 window")
        window = np.asarray(item["window"], float)
        if window.shape != (2,) or not np.isfinite(window).all() or not 0 < window[0] <= window[1]:
            raise ValueError("每个锚点窗口须为正的有限闭区间")
        guest, host = (atom_ids(item[k], 10**10) for k in ("guest","host"))
        if len(guest) != 1 or len(host) != 1:
            raise ValueError("每组锚点只允许一个宿主和一个客体编号")
        row = dict(guest=guest[0], host=host[0], window=window.tolist())
        if "image" in item:
            img = np.asarray(item["image"])
            if img.shape != (3,) or not np.issubdtype(img.dtype, np.integer):
                raise ValueError("锚点周期像须是三个整数")
            row["image"] = img.tolist()
        result.append(row)
    if (all(result[0][k] == result[1][k] for k in ("guest","host"))
            and result[0].get("image",[0,0,0]) == result[1].get("image",[0,0,0])):
        raise ValueError("两组锚点不能是同一个宿主、客体和周期像")
    return result


def reachable_gap(h, g, windows):
    (lo, hi), (low, high) = windows
    return max(0., abs(h-g)-hi-high, lo-high-h-g, low-hi-h-g)


def host_images(host, anchors, g):
    first, second = anchors
    a, b = first["host"]-1, second["host"]-1
    lattice = host.structure.lattice
    origin = np.asarray(first.get("image", [0,0,0]), int)
    f = host.coords[a]+origin
    if "image" in second:
        return [(origin, np.asarray(second["image"], int))]
    windows = [item["window"] for item in anchors]
    near, near_image = lattice.get_distance_and_image(f, host.coords[b])
    repeat_pair = (a == b and first["guest"] == second["guest"])
    if repeat_pair:
        distinct = [r for r in lattice.get_points_in_sphere([[0,0,0]],[0,0,0],min(lattice.abc)+1e-8)
                    if any(r[3])]
        nearest_other = min(distinct,key=lambda r:r[1])
        near, near_image = nearest_other[1], origin+np.asarray(nearest_other[3],int)
    limit = windows[0][1]+g+windows[1][1]
    # 最短像提供一个已知的误差上界，超过该半径的像不能改善最近失败姿势。
    limit += reachable_gap(near, g, windows)
    cloud = lattice.get_points_in_sphere([host.coords[b] % 1], f @ host.cell, limit+1e-8)
    images = sorted((float(r[1]), tuple(np.asarray(r[3], int)-np.floor(host.coords[b]).astype(int))) for r in cloud)
    if repeat_pair:
        images = [r for r in images if r[1] != tuple(origin)]
    possible = [entry for entry in images if reachable_gap(entry[0], g, windows) < 1e-12]
    chosen = possible or images or [(near, tuple(near_image))]
    return [(origin, np.asarray(im, int)) for _, im in chosen]


def length_pairs(h, g, windows, refs, count):
    (lo, hi), (low, high) = windows
    grids = [np.unique(np.r_[np.linspace(*w, count), np.clip(r, *w)]) for w, r in zip(windows, refs)]
    pairs = list(product(*grids))
    # 可达区域是两个窗口的矩形被三角不等式裁切所得的凸多边形。
    # 加入边界和参考键长线的交点，可取到只在切线处成立的解。
    lines = [(1,0,v) for v in (lo,hi,np.clip(refs[0],lo,hi))]
    lines += [(0,1,v) for v in (low,high,np.clip(refs[1],low,high))]
    lines += [(1,1,abs(h-g)), (1,-1,h+g), (-1,1,h+g)]
    for (a,b,c),(d,e,f) in combinations(lines, 2):
        det = a*e-b*d
        if det:
            x, y = (c*e-b*f)/det, (a*f-c*d)/det
            if lo-1e-12 <= x <= hi+1e-12 and low-1e-12 <= y <= high+1e-12:
                pairs.append((np.clip(x,lo,hi), np.clip(y,low,high)))
    pairs = sorted(set((float(x),float(y)) for x,y in pairs))
    gaps = [reachable_gap(h, g, [(r,r),(s,s)]) for r,s in pairs]
    best = min(gaps)
    return [p for p, gap in zip(pairs, gaps) if gap <= best+1e-12]


def intersections(a, b, ra, rb, angles):
    """两个球面的交点或交圆；重合的球面取球面方向网格。"""
    axis = b-a
    d = np.linalg.norm(axis)
    if d < 1e-12:
        if ra < 1e-12:
            return a[None,:]
        return a + ra*scan_directions({"direction_count":len(angles)})
    axis = axis/d
    x = (ra*ra-rb*rb+d*d)/(2*d)
    x = np.clip(x, -ra, ra)
    radius = np.sqrt(max(0., (ra-x)*(ra+x)))
    center = a + x*axis
    if radius < 1e-8:
        return center[None,:]
    u = unit(np.cross(axis, np.eye(3)[np.argmin(abs(axis))]))
    phase = np.deg2rad(angles)[:,None]
    return center + radius*(np.cos(phase)*u+np.sin(phase)*np.cross(axis,u))


def positions(first, second, g, radii, diagonal_count, azimuths, hinges):
    r1,r2 = radii
    h = np.linalg.norm(second-first)
    # s 为宿主 2 到客体 1 的对角线，同时受两个三角形限制。
    lo, hi = max(abs(h-r1), abs(g-r2)), min(h+r1, g+r2)
    if lo > hi:
        # 不可达时保持刚体和第一条长度，构造第二条最接近窗口的姿势。
        s_values = [np.clip(abs(g-r2) if h+r1 < abs(g-r2) else g+r2, abs(h-r1), h+r1)]
    else:
        s_values = np.unique(np.linspace(lo, hi, diagonal_count))
    for s in s_values:
        actual_r2 = np.clip(r2, abs(s-g), s+g)
        for p in intersections(first, second, r1, s, azimuths):
            if g == 0:
                yield p, p
            else:
                for q in intersections(p, second, g, actual_r2, hinges):
                    yield p, q


class PeriodicContacts:
    """宿主周期像、客体自身周期像及角色排斥；只豁免声明的两条边。"""
    def __init__(self, host, kept, anchors, images, local, rules, center, extent):
        lattice = host.structure.lattice
        radius = float(np.linalg.norm(local,axis=1).max())
        search = max(.5*sum(lattice.abc), min(lattice.abc))
        cloud = lattice.get_points_in_sphere(host.coords % 1, center, extent+radius+search+1e-7)
        xyz = np.asarray([row[0] for row in cloud]) @ host.cell
        indices = np.asarray([row[2] for row in cloud], int)
        offsets = np.asarray([row[3] for row in cloud], int)
        self.host = []
        special = {a["guest"] for a in anchors}
        ordinary = [i for i,k in enumerate(kept) if k not in special]
        if ordinary:
            self.host.append((cKDTree(xyz), ordinary))
        for k in special:
            allowed = np.ones(len(xyz), bool)
            for a, image in zip(anchors, images):
                if a["guest"] == k:
                    j = a["host"]-1
                    absolute = np.floor(host.coords[j]).astype(int)+image
                    allowed &= ~((indices == j) & np.all(offsets == absolute, axis=1))
            self.host.append((cKDTree(xyz[allowed]), [kept.index(k)]))
        self.excluded = [(cKDTree(xyz[np.isin(indices+1, r["host"])]), [kept.index(v) for v in r["guest"]])
                         for r in rules]
        self.periods = [(np.asarray(p[3]) @ host.cell, p[1]-2*radius)
                        for p in lattice.get_points_in_sphere([[0,0,0]], [0,0,0], 2*radius+min(lattice.abc)+1e-7)
                        if tuple(p[3]) > (0,0,0)]

    def evaluate(self, poses):
        clearance = np.minimum.reduce([tree.query(poses[:,ids])[0].min(axis=1) for tree,ids in self.host])
        delta = None
        for shift, lower_bound in self.periods:
            if lower_bound < clearance.max():
                if delta is None:
                    delta = poses[:,:,None,:]-poses[:,None,:,:]
                clearance = np.minimum(clearance, np.linalg.norm(delta-shift,axis=3).min(axis=(1,2)))
        exclusions = [tree.query(poses[:,ids])[0].min(axis=1) for tree,ids in self.excluded]
        return clearance, np.array(exclusions).reshape(len(exclusions), len(poses)).T


def pose_batches(first, second, local, gvec, configs, spins, vacancies, count):
    rigid_axis = unit(gvec) if np.linalg.norm(gvec) > 0 else None
    base_axis = unit(local[np.argmax(np.linalg.norm(local,axis=1))]) if np.linalg.norm(local) else np.eye(3)[0]
    if rigid_axis is None and len(local)>1:
        free_axes = scan_directions({"direction_count":count})
        free = [Rotation.from_rotvec(np.deg2rad(spins)[:,None]*u) * align_to(base_axis,u) for u in free_axes]
        free_matrices = Rotation.concatenate(free).as_matrix()
    matrices, origins, size = [], [], 0
    for p,q in configs:
        if rigid_axis is not None:
            u = unit(q-p)
            rot = Rotation.from_rotvec(np.deg2rad(spins)[:,None]*u) * align_to(rigid_axis,u)
            current = rot.as_matrix()
        elif len(local) == 1:
            current = np.eye(3)[None,:,:]
        else:
            rotations = [free_matrices]
            # η² 模板的两个腾空方向同时对准两个供体，另保留通用刚体方向网格。
            for v,w in combinations(vacancies, 2):
                dest = np.array([unit(first-p), unit(second-p)])
                if np.linalg.norm(np.cross(dest[0],dest[1])) < 1e-9 or np.linalg.norm(np.cross(v,w)) < 1e-9:
                    continue
                for a,b in ((v,w),(w,v)):
                    rot = Rotation.align_vectors(dest, [a,b])[0]
                    rotations.append(rot.as_matrix()[None,:,:])
            current = np.concatenate(rotations)
        matrices.append(current)
        origins.append(np.tile(p,(len(current),1)))
        size += len(current)
        if size >= 256:
            yield np.concatenate(matrices), np.concatenate(origins)
            matrices, origins, size = [], [], 0
    if matrices:
        yield np.concatenate(matrices), np.concatenate(origins)


def rebuild_selected(host, path, recipe, chosen):
    """从原分子重建刚体并独立复查硬约束；候选快照仅供展示。"""
    specs = anchor_specs(recipe)
    symbols, xyz, bonds = guest_template(path, recipe)
    ids = [a["guest"] for a in specs]
    gone = {i for p in displaced_parts(symbols,bonds,recipe,ids) for i in p["template_indices"]}
    kept = [i for i in range(1,len(symbols)+1) if i not in gone]
    local = xyz[np.asarray(kept)-1]-xyz[ids[0]-1]
    pose = chosen["rigid_transform"]
    matrix = np.asarray(pose["rotation"])
    if not np.allclose(matrix @ matrix.T,np.eye(3),rtol=0,atol=1e-11) or abs(np.linalg.det(matrix)-1)>1e-11:
        raise ValueError("所选候选的旋转矩阵不保持刚体")
    points = local @ matrix.T + pose["anchor_cartesian"]
    images = chosen["host_images"]
    centers = np.array([(host.coords[a["host"]-1]+im) @ host.cell for a,im in zip(specs,images)])
    indices = [kept.index(i) for i in ids]
    lengths = np.linalg.norm(points[indices]-centers,axis=1)
    windows = np.asarray([a["window"] for a in specs])
    rules = exclusions(host,symbols,kept,recipe.get("exclude_contact"))
    contacts = PeriodicContacts(host,kept,specs,images,local,rules,centers[0],windows[0,1])
    clearance, distances = contacts.evaluate(points[None,:,:])
    tol = 64*np.finfo(float).eps*(1+np.max(abs(centers))+np.linalg.norm(local[indices[1]])+windows.max())
    if (np.any(lengths < windows[:,0]-tol) or np.any(lengths > windows[:,1]+tol)
            or clearance[0] < recipe["min_nonbond"]
            or any(d<r["min"] or (r["strict"] and d<=r["min"]) for r,d in zip(rules,distances[0]))):
        raise ValueError("从原模板重建的候选未通过双窗口、碰撞或排斥复查")
    chosen["guest_cartesian"] = points.tolist()
    chosen["reconstructed"] = dict(distances=lengths.tolist(),min_nonbond=float(clearance[0]),
                                   exclusion_distances=distances[0].tolist())
    links = [(a["host"]-1,im,j) for a,im,j in zip(specs,images,indices)]
    edges = [(kept.index(i+1),kept.index(j+1)) for i,j in bonds if i+1 not in gone and j+1 not in gone]
    return append_template(host,list(range(1,len(host.symbols)+1)),[symbols[i-1] for i in kept],points,edges,links)


def angle_grid(recipe, key, step):
    angles = np.asarray(recipe.get(key, np.arange(0,360,step)), float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError(key+" 须是非空有限角度列表")
    return angles


def scan_dual(host, path, recipe):
    if any(key in recipe for key in ("host_atom","guest_anchor","distance_window")):
        raise ValueError("双锚点与单锚点参数互斥")
    if any(o != 1 for o in host.model.occupancy):
        raise ValueError("双锚点放置前请先处理部分占位")
    anchors = anchor_specs(recipe)
    symbols, xyz, bonds = guest_template(path, recipe)
    for a in anchors:
        atom_ids(a["guest"],len(symbols))
        atom_ids(a["host"],len(host.symbols))
    ids = [a["guest"] for a in anchors]
    parts = displaced_parts(symbols, bonds, recipe, ids)
    discarded = [dict(index=i, reason="displace", fragment=p["name"]) for p in parts for i in p["template_indices"]]
    gone = {row["index"] for row in discarded}
    kept = [i for i in range(1,len(symbols)+1) if i not in gone]
    local = xyz[np.array(kept)-1]-xyz[ids[0]-1]
    gvec = xyz[ids[1]-1]-xyz[ids[0]-1]
    g = np.linalg.norm(gvec)
    if ids[0] != ids[1] and g < 1e-8:
        raise ValueError("不同客体锚原子的模板坐标重合")
    windows = np.asarray([a["window"] for a in anchors])
    preferences = recipe.get("prefer_distances",[recipe.get("prefer_distance")]*2)
    if not isinstance(preferences, (list,tuple)) or len(preferences) != 2:
        raise ValueError("prefer_distances 需要两项参考键长")
    prefs = [window_parameters({"distance_window":a["window"],"prefer_distance":pref},
                               [host.symbols[a["host"]-1]],symbols[a["guest"]-1]) for a,pref in zip(anchors,preferences)]
    refs = np.array([r["reference_distances"][0] for r in prefs])
    minimum = float(recipe.get("min_nonbond",0))
    top = recipe.get("top",3)
    counts = [recipe.get("distance_samples",3), recipe.get("diagonal_samples",3), recipe.get("orientation_count",6), top]
    if not np.isfinite(minimum) or minimum <= 0 or any(type(k) is not int or k <= 0 for k in counts):
        raise ValueError("碰撞下限、候选数和采样数须为正")
    az, hinge, spin = [angle_grid(recipe,s,n) for s,n in (("azimuths_deg",45),("hinges_deg",60),("spins_deg",45))]
    rules = exclusions(host, symbols, kept, recipe.get("exclude_contact"))
    floors = np.array([r["min"] for r in rules])
    strict = np.array([r["strict"] for r in rules],bool)
    guest_indices = [kept.index(i) for i in ids]
    vacancies = [unit(xyz[p["cut_bond"][1]-1]-xyz[ids[0]-1]) for p in parts if p["cut_bond"][0] == ids[0]]
    wp = dict(distance_windows=windows.tolist(), reference_distances=refs.tolist(),
              reference_source=[r["reference_source"] for r in prefs], reference_pairs=[r["reference_pairs"][0] for r in prefs],
              ranking=["reference_deviation ASC","min_nonbond DESC"])
    report = dict(anchors=anchors, window_params=wp, total=0, rejected=0, collisions=0, exclusion_rejected=0,
                  coordination_rejected=0, top=[], template_indices=kept, template_discarded=discarded,
                  displaced_fragments=parts, exclude_contact=rules, image_reachability=[])
    closest_key = None
    for images in host_images(host,anchors,g):
        centers = np.array([(host.coords[a["host"]-1]+im) @ host.cell for a,im in zip(anchors,images)])
        h = np.linalg.norm(centers[1]-centers[0])
        image_list = [im.tolist() for im in images]
        report["image_reachability"].append(dict(host_images=image_list, host_separation=float(h),
                 guest_separation=float(g), minimum_window_deviation=float(reachable_gap(h,g,windows))))
        contact = PeriodicContacts(host, kept, anchors, images, local, rules, centers[0], windows[0,1])
        for radii in length_pairs(h,g,windows,refs,counts[0]):
            configs = positions(*centers, g, radii, counts[1], az, hinge)
            for matrices, origins in pose_batches(*centers, local,gvec,configs,spin,vacancies,counts[2]):
                poses = (matrices @ local.T).transpose(0,2,1)+origins[:,None,:]
                lengths = np.linalg.norm(poses[:,guest_indices]-centers,axis=2)
                window_gap = np.maximum(0., np.maximum(windows[:,0]-lengths,lengths-windows[:,1]))
                # 消除解析交圆坐标在浮点计算时的数个 ULP，不放宽排斥或碰撞下限。
                window_gap[window_gap < np.finfo(float).eps*64*(1+np.max(abs(centers))+g+windows.max())] = 0
                refdev = abs(lengths-refs).sum(axis=1)
                clear, separation = contact.evaluate(poses)
                collision = np.maximum(0.,minimum-clear)
                excludes = np.maximum(0.,floors-separation)
                strict_failed = (separation <= floors) & strict
                error = window_gap.sum(axis=1)+collision+excludes.sum(axis=1)
                valid = (error == 0) & ~strict_failed.any(axis=1)
                report["total"] += len(poses)
                report["rejected"] += int((~valid).sum())
                report["collisions"] += int((collision>0).sum())
                report["coordination_rejected"] += int((window_gap>0).any(axis=1).sum())
                report["exclusion_rejected"] += int(((excludes>0)|strict_failed).any(axis=1).sum())
                def row(j):
                    return dict(host_images=image_list, distances=lengths[j].tolist(), min_nonbond=float(clear[j]),
                                reference_deviation=float(refdev[j]), exclusion_distances=separation[j].tolist(),
                                violation=float(error[j]), guest_cartesian=poses[j].tolist(),
                                rigid_transform=dict(anchor_cartesian=origins[j].tolist(),rotation=matrices[j].tolist()),
                                violations=dict(distance_windows=window_gap[j].tolist(), min_nonbond=float(collision[j]),
                                    exclude_contact=excludes[j].tolist(), strict_exclusions_failed=(np.flatnonzero(strict_failed[j])+1).tolist()))
                good = np.flatnonzero(valid)
                if len(good):
                    order = np.lexsort((-clear[good], np.round(refdev[good],12)))
                    report["top"].extend(row(i) for i in good[order[:top]])
                    report["top"].sort(key=window_rank)
                    del report["top"][top:]
                bad = np.flatnonzero(~valid)
                if len(bad):
                    ix = bad[np.lexsort((-clear[bad],np.round(refdev[bad],12),error[bad]))[0]]
                    key = (error[ix], round(refdev[ix],12), -clear[ix])
                    if closest_key is None or key < closest_key:
                        closest_key, report["closest"] = key, row(ix)
    if not report["top"]:
        report["reason"] = "没有同时满足两条窗口、碰撞和排斥约束的姿势"
        return report,None
    select = recipe.get("select", 1 if report["total"]-report["rejected"] == 1 else None)
    if type(select) is not int or not 1 <= select <= len(report["top"]):
        report["reason"] = "请用 select 选择候选"
        return report,None
    chosen = report["top"][select-1]
    report["selected"] = select
    return report,rebuild_selected(host,path,recipe,chosen)
