"""单宿主位点与指定客体锚点的周期刚体扫描。对外编号均从 1 开始。"""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from pymatgen.core import Molecule
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .fragment_ops import append_template, sdf_template, selected
from .placement import unit, window_parameters, window_rank


def atom_ids(value, size):
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    elif isinstance(value, int):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("须提供非空的原子编号")
    if any(type(v) is not int and not (isinstance(v, str) and v.isdecimal()) for v in value):
        raise ValueError("原子编号必须为正整数")
    ids = [int(v) for v in value]
    if len(set(ids)) != len(ids) or min(ids) < 1 or max(ids) > size:
        raise ValueError("原子编号越界或重复")
    return ids


def guest_template(path, recipe):
    ext = Path(path).suffix.lower()
    if ext == ".sdf":
        symbols, xyz, pairs = sdf_template(path)
    else:
        if ext == ".xyz":
            mol = Molecule.from_file(path)
        elif ext in (".json", ".mson"):
            mol = Molecule.from_dict(json.loads(Path(path).read_text("utf-8")))
        else:
            raise ValueError("供体扫描只接受 XYZ、MSON 或 V2000 模板")
        symbols = [site.specie.symbol for site in mol]
        xyz = np.asarray(mol.cart_coords)
        pairs = []
        # XYZ 没有键表。氢仅接最近的可成键原子，避免金属的半径重复认领水氢。
        radii = covalent_radii[[atomic_numbers[s] for s in symbols]]
        distances = np.linalg.norm(xyz[:, None] - xyz, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.argmin(distances, axis=1)
        for i, j in zip(*np.tril_indices(len(symbols), -1), strict=True):
            if distances[i, j] > (radii[i] + radii[j]) * 1.15:
                continue
            if (symbols[i] == "H" and nearest[i] != j) or (symbols[j] == "H" and nearest[j] != i):
                continue
            pairs.append((int(j), int(i)))
    if "guest_bonds" in recipe:
        pairs = []
        for pair in recipe["guest_bonds"]:
            a, b = atom_ids(pair, len(symbols))
            pairs.append((a-1, b-1))
    return symbols, np.asarray(xyz), pairs


def displaced_part(symbols, pairs, recipe, request, protected):
    fragments = recipe.get("guest_fragments", {})
    if not isinstance(fragments, dict):
        raise ValueError("guest_fragments 须是片段编号或名称到原子清单的映射")
    if isinstance(request, (int, str)) and str(request) in fragments:
        name = str(request)
        request = fragments[name]
    elif isinstance(request, (list, dict)) or (isinstance(request, str) and "," in request):
        name = "explicit"
    else:
        raise ValueError("未知客体片段；请在 guest_fragments 登记该片段的模板原子编号")
    spec = request if isinstance(request, dict) else {"atoms": request}
    name = str(spec.get("fragment", name))
    ids = atom_ids(spec.get("atoms"), len(symbols))
    if set(protected).intersection(ids):
        raise ValueError("移除的客体片段不可包含锚点")
    gone = {i-1 for i in ids}
    neighbors = [set() for _ in symbols]
    for i, j in pairs:
        neighbors[i].add(j)
        neighbors[j].add(i)
    stack = [min(gone)]
    connected = set()
    while stack:
        i = stack.pop()
        if i not in connected:
            connected.add(i)
            stack.extend((neighbors[i] & gone) - connected)
    boundary = [(i, j) for i in gone for j in neighbors[i] if j not in gone]
    if connected != gone or len(boundary) != 1:
        raise ValueError("displace 须指定完整连通的端位客体片段，且只有一个连接端")
    i, j = boundary[0]
    if symbols[i] == "H" or symbols[j] == "H":
        raise ValueError("displace 不可切断片段内部的氢键接关系")
    charge = spec.get("formal_charge")
    counts = dict(sorted(Counter(symbols[i] for i in gone).items()))
    if charge is None and counts == {"H": 2, "O": 1}:
        charge = 0
    if charge is not None and type(charge) is not int:
        raise ValueError("客体片段的形式电荷须为整数")
    return dict(name=name, template_indices=sorted(ids), composition=counts, formal_charge=charge,
                cut_bond=[j+1, i+1])


def displaced_parts(symbols, pairs, recipe, protected):
    request = recipe.get("displace")
    if request is None:
        return []
    requests = request.get("fragments", [request]) if isinstance(request, dict) else [request]
    if not isinstance(requests, list) or not requests:
        raise ValueError("displace 的 fragments 必须是非空片段清单")
    parts = [displaced_part(symbols, pairs, recipe, item, protected) for item in requests]
    ids = [i for part in parts for i in part["template_indices"]]
    if len(ids) != len(set(ids)):
        raise ValueError("移走的客体片段包含重复模板原子")
    return parts


def resolve_displacements(path, recipe):
    """直接从原模板与配方复核移除的原子及电荷，供写出账本独立调用。"""
    if recipe.get("displace") is None:
        return []
    symbols, _, pairs = guest_template(path, recipe)
    if "anchors" in recipe:
        from .dual import anchor_specs
        protected = [a["guest"] for a in anchor_specs(recipe)]
    else:
        protected = atom_ids(recipe.get("guest_anchor"), len(symbols))
    return displaced_parts(symbols, pairs, recipe, protected)


def resolve_displacement(path, recipe):
    """旧的单片段接口。多片段调用方使用 resolve_displacements。"""
    parts = resolve_displacements(path, recipe)
    if len(parts) > 1:
        raise ValueError("多个客体片段须逐项记账")
    return parts[0] if parts else None


def host_ids(host, selector):
    if isinstance(selector, str) and not all(s.strip().isdecimal() for s in selector.split(",")):
        ids = [i+1 for i in selected(host, {"predicate": selector})]
        if not ids:
            raise ValueError("宿主判据没有匹配原子：" + selector)
        return ids
    return atom_ids(selector, len(host.symbols))


def exclusions(host, symbols, retained, raw):
    if raw is None:
        return []
    rules = []
    for item in raw if isinstance(raw, list) else [raw]:
        if isinstance(item, str):
            selector, threshold = item.rsplit(":", 1)
            if selector.startswith("guest=") and "@host=" in selector:
                guest, selector = selector[6:].split("@host=", 1)
            else:
                guest = None
            item = {"host": selector, "guest": guest, "min": threshold.lstrip(">"),
                    "strict": threshold.startswith(">")}
        if not isinstance(item, dict) or "host" not in item or "min" not in item:
            raise ValueError("每条 exclude-contact 约束都须指定宿主原子和距离下限")
        guest = item.get("guest")
        if guest is None:
            members = retained
        elif isinstance(guest, str) and guest.startswith("element="):
            members = [i for i in retained if symbols[i-1] == guest[8:]]
        else:
            members = atom_ids(guest, len(symbols))
        if not members or not set(members) <= set(retained):
            raise ValueError("排斥约束的客体原子必须存在于保留的片段中")
        floor = float(item["min"])
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("排斥下限须为有限正数")
        strict = item.get("strict", False)
        if type(strict) is not bool:
            raise ValueError("排斥下限的 strict 必须是布尔值")
        rules.append(dict(host=host_ids(host, item["host"]), guest=members, min=floor, strict=strict))
    return rules


def scan_directions(recipe):
    explicit = recipe.get("directions")
    if explicit is not None:
        vectors = np.asarray(explicit, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != 3 or not len(vectors) or not np.isfinite(vectors).all():
            raise ValueError("directions 须是非空的有限笛卡尔方向列表")
        return np.array([unit(v) for v in vectors])
    count = recipe.get("direction_count", 48)
    if type(count) is not int or count < 1:
        raise ValueError("direction_count 须为正整数")
    k = np.arange(count)
    z = 1 - 2*(k+.5)/count
    phi = np.pi*(3-np.sqrt(5))*k
    r = np.sqrt(1-z*z)
    return np.vstack((np.eye(3), -np.eye(3), np.column_stack((r*np.cos(phi), r*np.sin(phi), z))))


def align_to(vector, direction):
    axis = np.cross(vector, direction)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        if vector @ direction > 0:
            return Rotation.identity()
        axis = np.cross(vector, np.eye(3)[np.argmin(abs(vector))])
        return Rotation.from_rotvec(np.pi * unit(axis))
    return Rotation.from_rotvec(np.arctan2(sine, vector @ direction) * axis / sine)


def scan_donor(host, path, recipe):
    if "anchors" in recipe:
        from .dual import scan_dual
        return scan_dual(host, path, recipe)
    if any(o != 1 for o in host.model.occupancy):
        raise ValueError("供体扫描前请先处理部分占位")
    hosts = host_ids(host, recipe.get("host_atom"))
    if len(hosts) != 1:
        raise ValueError(f"供体位点须唯一；匹配的宿主编号为 {hosts}")
    source = hosts[0]-1
    image = np.asarray(recipe["site"].get("image", [0, 0, 0]))
    if image.shape != (3,) or not np.issubdtype(image.dtype, np.integer):
        raise ValueError("宿主周期像必须是三个整数")
    symbols, xyz, pairs = guest_template(path, recipe)
    anchors = atom_ids(recipe.get("guest_anchor"), len(symbols))
    if len(anchors) != 1:
        raise ValueError("供体位点只接受一个客体锚点")
    anchor = anchors[0]-1
    parts = displaced_parts(symbols, pairs, recipe, anchors)
    removed = parts[0] if len(parts) == 1 else None
    gone = {i for part in parts for i in part["template_indices"]}
    kept = [i for i in range(1, len(symbols)+1) if i not in gone]
    donor = kept.index(anchor+1)
    rules = exclusions(host, symbols, kept, recipe.get("exclude_contact"))
    window = np.asarray(recipe.get("distance_window"), dtype=float)
    if window.shape != (2,) or not np.isfinite(window).all() or not 0 < window[0] <= window[1]:
        raise ValueError("供体配位距离窗口须为正的有限闭区间")
    minimum = float(recipe.get("min_nonbond", 0))
    top = recipe.get("top", 3)
    if not np.isfinite(minimum) or minimum <= 0 or type(top) is not int or top < 1:
        raise ValueError("请指定正的 min_nonbond 和候选数")
    ranking = window_parameters(recipe, [host.symbols[source]], symbols[anchor])
    ref = ranking["reference_distances"][0]
    default_radii = np.unique(np.r_[np.linspace(*window, 15), np.clip(ref, *window)])
    radii = np.asarray(recipe.get("radii", default_radii), dtype=float)
    spins = np.asarray(recipe.get("spins_deg", np.arange(0, 360, 30)), dtype=float)
    if any(a.ndim != 1 or not a.size or not np.isfinite(a).all() for a in (radii, spins)) or min(radii) <= 0:
        raise ValueError("候选距离和自转角须是有限数列表，距离须为正")
    local = xyz[np.asarray(kept)-1] - xyz[anchor]
    vacancies = [part["cut_bond"][1]-1 for part in parts if part["cut_bond"][0] == anchor+1]
    if vacancies:
        # 配位空位（移出的片段方向）指向宿主，客体内侧朝相反方向。
        inward = xyz[anchor] - xyz[vacancies].mean(axis=0)
        alignment = "removed attachment toward host"
    else:
        near = [j if i == anchor else i for i, j in pairs if anchor in (i, j) and i+1 not in gone and j+1 not in gone]
        inward = xyz[near].mean(axis=0) - xyz[anchor] if near else np.zeros(3)
        alignment = "retained neighbors inward"
    if np.linalg.norm(inward) < 1e-8:
        inward = local[np.argmax(np.linalg.norm(local, axis=1))]
        alignment = "farthest retained atom inward" if len(kept) > 1 else "single atom"
    inward = unit(inward) if np.linalg.norm(inward) > 1e-8 else np.array([1., 0., 0.])
    if len(kept) == 1:
        spins = np.array([0.])
    lattice = host.structure.lattice
    center = (host.coords[source]+image) @ host.cell
    radius = float(np.linalg.norm(local, axis=1).max())
    # 半个胞矢长度之和为任意点到给定宿主编号某个周期像的距离上界。
    search = max(.5*sum(lattice.abc), min(lattice.abc), minimum)
    cloud = lattice.get_points_in_sphere(host.coords % 1, center, radius+max(radii)+search+1e-7)
    coords = np.asarray([p[0] for p in cloud]) @ host.cell
    indices = np.asarray([p[2] for p in cloud], int)
    offsets = np.asarray([p[3] for p in cloud], int)
    target = (indices == source) & np.all(offsets == np.floor(host.coords[source]).astype(int)+image, axis=1)
    host_tree = cKDTree(coords)
    other_tree = cKDTree(coords[~target])
    other_atoms = [i for i in range(len(kept)) if i != donor]
    constraint_trees = [(cKDTree(coords[np.isin(indices+1, rule["host"])]),
                         [kept.index(i) for i in rule["guest"]]) for rule in rules]
    periodic = [np.asarray(p[3]) @ host.cell for p in lattice.get_points_in_sphere(
        [[0, 0, 0]], [0, 0, 0], 2*radius+min(lattice.abc)+1e-7)
        if tuple(p[3]) > (0, 0, 0)]
    report = dict(total=0, rejected=0, collisions=0, exclusion_rejected=0, coordination_rejected=0,
                  top=[], window_params=ranking, host_atom=source+1, host_image=image.tolist(),
                  guest_anchor=anchor+1, template_indices=kept, displaced_fragment=removed, displaced_fragments=parts,
                  orientation=dict(method=alignment, template_inward_unit=inward.tolist()),
                  template_discarded=[dict(index=i, reason="displace", fragment=p["name"])
                                      for p in parts for i in p["template_indices"]],
                  exclude_contact=rules)
    best_rejected = None
    for direction in scan_directions(recipe):
        base = align_to(inward, direction).apply(local)
        rotations = Rotation.from_rotvec(np.deg2rad(spins)[:, None]*direction).as_matrix()
        oriented = np.einsum("sij,nj->sni", rotations, base)
        points = (center + radii[:, None, None, None]*direction + oriented[None, :, :, :])
        clear = other_tree.query(points[:, :, donor, :])[0]
        if other_atoms:
            clear = np.minimum(clear, host_tree.query(points[:, :, other_atoms, :])[0].min(axis=2))
        delta = oriented[:, :, None, :] - oriented[:, None, :, :]
        for shift in periodic:
            clear = np.minimum(clear, np.linalg.norm(delta-shift, axis=3).min(axis=(1, 2)))
        separations = [tree.query(points[:, :, members, :])[0].min(axis=2) for tree, members in constraint_trees]
        for ir, length in enumerate(radii):
            for it, spin in enumerate(spins):
                nonbond = float(clear[ir, it])
                strict_fail = [k+1 for k, (rule, d) in enumerate(zip(rules, separations))
                               if rule["strict"] and d[ir, it] <= rule["min"]]
                penal = dict(distance_window=max(0., float(window[0]-length), float(length-window[1])),
                             min_nonbond=max(0., minimum-nonbond),
                             exclude_contact=[max(0., rule["min"]-float(d[ir, it])) for rule, d in zip(rules, separations)],
                             strict_exclusions_failed=strict_fail)
                error = penal["distance_window"]+penal["min_nonbond"]+sum(penal["exclude_contact"])
                row = dict(direction=direction.tolist(), spin=float(spin), distances=[float(length)],
                           reference_deviation=float(abs(length-ref)), min_nonbond=nonbond,
                           exclusion_distances=[float(d[ir, it]) for d in separations], violations=penal, violation=error)
                report["total"] += 1
                if error > 0 or strict_fail:
                    report["rejected"] += 1
                    report["collisions"] += int(penal["min_nonbond"] > 0)
                    report["exclusion_rejected"] += int(bool(strict_fail) or any(v > 0 for v in penal["exclude_contact"]))
                    report["coordination_rejected"] += int(penal["distance_window"] > 0)
                    key = (error, *window_rank(row))
                    if best_rejected is None or key < best_rejected:
                        best_rejected = key
                        report["closest"] = row
                    continue
                report["top"].append(row)
                report["top"].sort(key=window_rank)
                del report["top"][top:]
    if not report["top"]:
        report["reason"] = "没有同时满足配位窗口、碰撞下限与排斥约束的姿势"
        return report, None
    selection = recipe.get("select", 1 if report["total"]-report["rejected"] == 1 else None)
    if type(selection) is not int or not 1 <= selection <= len(report["top"]):
        report["reason"] = "请用 select 明确选择候选"
        return report, None
    row = report["top"][selection-1]
    direction = np.asarray(row["direction"])
    rot = Rotation.from_rotvec(np.deg2rad(row["spin"]) * direction) * align_to(inward, direction)
    positions = rot.apply(local) + center + row["distances"][0]*direction
    kept_pairs = [(kept.index(i+1), kept.index(j+1)) for i, j in pairs if i+1 not in gone and j+1 not in gone]
    report["selected"] = selection
    return report, append_template(host, list(range(1, len(host.symbols)+1)), [symbols[i-1] for i in kept],
                                   positions, kept_pairs, [(source, image, donor)])
