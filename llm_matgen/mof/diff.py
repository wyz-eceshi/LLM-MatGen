"""在父结构编号中匹配周期键；增加的键用子结构编号报告。"""
import numpy as np
from .cif import edge
from .graph import public_edge


def compare(parent, child, atom_map=None):
    n, m = len(parent.elements), len(child.elements)
    if atom_map is None:
        if n != m:
            raise ValueError("原子数发生变化，需要子→父 atom_map")
        old_labels,new_labels = parent.model.labels,child.model.labels
        delta = child.coords-parent.coords
        relative = delta - delta[0]
        if parent.symbols==child.symbols and np.allclose(relative-np.rint(relative),0,atol=1e-5):
            atom_map = list(range(1,n+1))
        elif (parent.model.explicit is not None and child.model.explicit is not None
              and len(set(old_labels))==n and set(old_labels)==set(new_labels)):
            lookup = {label:i for i,label in enumerate(old_labels,1)}
            atom_map = [lookup[label] for label in new_labels]
        else:
            raise ValueError("无法确认原子对应关系，需要 atom_map")
    if isinstance(atom_map,dict):
        if {str(k) for k in atom_map} != {str(i) for i in range(1,m+1)}:
            raise ValueError("atom_map 子编号不完整")
        atom_map = [atom_map.get(str(i),atom_map.get(i)) for i in range(1,m+1)]
    if not isinstance(atom_map,list) or len(atom_map)!=m:
        raise ValueError("atom_map 长度错误")
    parents = [p for p in atom_map if p is not None]
    if any(type(p) is not int or not 1<=p<=n for p in parents) or len(parents)!=len(set(parents)):
        raise ValueError("atom_map 父编号越界或重复")
    mapped_sites = [i for i,p in enumerate(atom_map) if p is not None]
    shifts = {}
    if mapped_sites:
        delta = child.coords[mapped_sites] - parent.coords[np.asarray(parents)-1]
        # 先在周期坐标上估计同一个原点平移，再确定各位点绕回了几个胞。
        phase = np.exp(2j*np.pi*delta).mean(axis=0)
        origin = np.where(abs(phase)>1e-8, np.angle(phase)/(2*np.pi), 0.)
        shifts = dict(zip(mapped_sites, np.rint(delta-origin).astype(int), strict=True))
    common, added = set(), []
    for key in sorted(child.edges):
        i,j,t = key
        if i in shifts and j in shifts:
            mapped = edge(atom_map[i]-1,atom_map[j]-1,np.array(t)+shifts[j]-shifts[i])
            if mapped in parent.edges:
                common.add(mapped)
                continue
        added.append(public_edge(key))

    def signature(fragment, ids):
        return (fragment["type"],fragment["formula"],fragment["dimension"],
                tuple(sorted(ids[i-1] for i in fragment["atoms"])))
    original = {signature(f,list(range(1,n+1))):f for f in parent.fragments()}
    # 新原子的临时负号仅用于集合比对，输出仍为子图的一起编号。
    ids = [p if p is not None else -i for i,p in enumerate(atom_map,1)]
    revised = {signature(f,ids):f for f in child.fragments()}
    old_coord,new_coord = parent.coordination(),child.coordination()
    coord_changes = []
    for j,p in enumerate(atom_map,1):
        before,after = old_coord.get(p),new_coord.get(j)
        if before!=after:
            coord_changes.append(dict(parent=p,child=j,before=before,after=after))
    for p in sorted(old_coord.keys()-set(parents)):
        coord_changes.append(dict(parent=p,child=None,before=old_coord[p],after=None))
    return dict(preserved=len(common),added=added,
                removed=[public_edge(k) for k in sorted(parent.edges-common)],
                dimension=[parent.dimension,child.dimension],
                fragments_removed=[original[k] for k in sorted(original.keys()-revised.keys())],
                fragments_added=[revised[k] for k in sorted(revised.keys()-original.keys())],
                coordination_changes=coord_changes)
