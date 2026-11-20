"""按父环的实际周期键解析显式声明的外接取代基。"""
from dataclasses import dataclass
import re

import numpy as np


@dataclass
class Branch:
    action: str
    root: int
    parent_ring: int
    parent_image: np.ndarray
    members: list[int]
    template_ring: int | None


def _entries(recipe, name, pattern):
    entries = recipe.get(name, [])
    if not isinstance(entries, list):
        raise ValueError(f"{name} 须为声明列表")
    for entry in entries:
        match = re.fullmatch(pattern, entry) if isinstance(entry, str) else None
        if match is None:
            raise ValueError(f"无效的 {name} 声明 {entry}")
        yield match.groups()


def _branch_members(graph, root, parent_ring, image, ring_images, anchors):
    """仅接受通过一条边连接到所选环的有限片段。"""
    seen = {root: image}
    pending = [root]
    while pending:
        atom = pending.pop()
        for other, relative in graph.neighbors[atom]:
            position = seen[atom] + relative
            if other in ring_images:
                if atom == root and other == parent_ring and np.array_equal(position, ring_images[other]):
                    continue
                raise ValueError("取代基不只通过一条边连接到所选环")
            if other in anchors:
                raise ValueError("取代基与父结构锚点相通")
            if other not in seen:
                seen[other] = position
                pending.append(other)
            elif not np.array_equal(position, seen[other]):
                raise ValueError("取代基沿周期无限延伸")
    return sorted(seen)


def parent_substituents(graph, old, anchors, ports, images, order, recipe):
    actions = {}
    for action, number in _entries(recipe, "substituent", r"(keep|drop):([1-9][0-9]*)"):
        root = int(number)-1
        if root in actions:
            raise ValueError("外接原子存在重复处理声明")
        actions[root] = action
    targets = {}
    for parent, template in _entries(recipe, "map", r"([1-9][0-9]*):([1-9][0-9]*)"):
        site, target = int(parent)-1, int(template)-1
        if site not in old or target not in order or site in targets or target in targets.values():
            raise ValueError("环位映射重复或端点不属于指定环")
        if order.index(target) in ports:
            raise ValueError("保留取代基的模板环位已被锚点占用")
        targets[site] = target

    ring_images = dict(zip(old, images, strict=True))
    anchor_bonds = {(old[k], a) for k,a in zip(ports, anchors, strict=True)}
    drop = set(old)
    roots = {}
    for site in old:
        for other, offset in graph.neighbors[site]:
            if other in ring_images:
                if not np.array_equal(ring_images[site]+offset, ring_images[other]):
                    raise ValueError("所选父环与另一周期像有未声明的外部键")
                continue
            if (site,other) in anchor_bonds:
                continue
            if graph.symbols[other] == "H" and len(graph.neighbors[other]) == 1:
                drop.add(other)
            elif other in roots or other in anchors:
                raise ValueError("环外接点有多个连接或属于父结构锚点")
            else:
                roots[other] = site, ring_images[site] + offset
    if roots.keys() - actions.keys():
        raise ValueError("替换会切断未声明的外部键")
    if actions.keys() - roots.keys():
        raise ValueError("声明的原子不是本次替换环的外接点")
    keep_sites = [roots[j][0] for j in roots if actions[j] == "keep"]
    if len(set(keep_sites)) != len(keep_sites) or set(keep_sites) != targets.keys():
        raise ValueError("每个保留取代基均须显式指定一个唯一环位映射")
    branches = []
    for root, (site,image) in roots.items():
        members = _branch_members(graph,root,site,image,ring_images,anchors)
        if actions[root] == "drop":
            drop.update(members)
        branches.append(Branch(actions[root],root,site,image,members,
                               targets[site] if actions[root] == "keep" else None))
    return drop, branches
