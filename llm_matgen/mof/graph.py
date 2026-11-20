"""周期键图，图内索引从零开始，报告从一开始。"""
from collections import Counter
import math

import numpy as np
from ase.data import covalent_radii
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from pymatgen.core import Composition
from .cif import edge


def public_edge(key):
    i, j, shift = key
    return [i + 1, j + 1, list(shift)]


def adjacency(count, edges):
    neighbors = [[] for _ in range(count)]
    for i, j, offset in sorted(edges):
        neighbors[i].append((j, offset))
        neighbors[j].append((i, tuple(-v for v in offset)))
    return neighbors


def components(neighbors):
    """沿树传递晶格平移；非树边的闭环平移张成周期子空间。"""
    visited = set()
    for root in range(len(neighbors)):
        if root in visited:
            continue
        offsets = {root: np.zeros(3, dtype=int)}
        stack, cycles = [root], []
        while stack:
            i = stack.pop()
            for j, delta in neighbors[i]:
                expected = offsets[i] + delta
                if j not in offsets:
                    offsets[j] = expected
                    stack.append(j)
                elif np.any(expected != offsets[j]):
                    cycles.append(expected - offsets[j])
        visited.update(offsets)
        yield sorted(offsets), int(np.linalg.matrix_rank(cycles)) if cycles else 0


class BondGraph:
    def __init__(self, model, tolerance=1.15, metal_tolerance=1.25, h_window=.05, metal_window=.15, other_window=.15):
        if not all(math.isfinite(v) and v > 0 for v in (tolerance,metal_tolerance)):
            raise ValueError("成键容差必须为正的有限数")
        self.windows = h_window,metal_window,other_window
        if not all(math.isfinite(w) and w>=0 for w in self.windows):
            raise ValueError("临界窗口必须是非负有限数")
        self.model = model
        self.structure = model.structure
        self.coords = self.structure.frac_coords
        self.cell = self.structure.lattice.matrix
        self.elements = [s.specie for s in self.structure]
        self.symbols = [el.symbol for el in self.elements]
        self.borderline = []
        self.edges = model.explicit
        if self.edges is None:
            self.edges = self._infer(tolerance, metal_tolerance)
        self.neighbors = adjacency(len(self.structure), self.edges)
        self.components = list(components(self.neighbors))
        self.dimension = max(d for _,d in self.components)

    def _infer(self, tolerance, metal_tolerance):
        radii = [covalent_radii[e.Z] for e in self.elements]
        cutoff = 2*max(radii)*max(tolerance, metal_tolerance) + max(self.windows)
        edges, seen = set(), set()
        for i,j,offset,d in zip(*self.structure.get_neighbor_list(cutoff),strict=True):
            key = edge(int(i),int(j),offset)
            if key in seen:
                continue
            seen.add(key)
            a,b = self.elements[i],self.elements[j]
            metal_on = (a.is_metal and b.symbol in {"O","N"}) or (b.is_metal and a.symbol in {"O","N"})
            limit = (radii[i]+radii[j]) * (metal_tolerance if metal_on else tolerance)
            width = self.windows[0] if "H" in (a.symbol,b.symbol) else self.windows[1] if metal_on else self.windows[2]
            if abs(d-limit) <= width:
                self.borderline.append(dict(edge=public_edge(key),distance=float(d),threshold=limit))
            elif d < limit:
                edges.add(key)
        return edges

    def distance(self, i, j, offset):
        return float(np.linalg.norm((self.coords[j]+offset-self.coords[i]) @ self.cell))

    def coordination(self):
        return {i+1:dict(sorted(Counter(self.symbols[j] for j,_ in self.neighbors[i]).items()))
                for i,el in enumerate(self.elements) if el.is_metal}

    def fragments(self):
        metals = {i for i,e in enumerate(self.elements) if e.is_metal}
        cut = {k for k in self.edges if (k[0] in metals)==(k[1] in metals)}
        frames = {i for nodes,rank in self.components if rank or any(n in metals for n in nodes) for i in nodes}
        out = []
        for atoms,dim in components(adjacency(len(self.elements),cut)):
            counts = Counter(self.symbols[i] for i in atoms)
            formula = Composition(counts).formula.replace(" ", "")
            charge = None
            if atoms[0] in metals:
                kind = "metal_node"
            elif atoms[0] not in frames:
                kind = "free_guest"
            elif set(counts) <= {"O","H"} and counts["O"]==1:
                kind = "coordinated_solvent"
                if counts["H"] in (1,2):
                    charge = counts["H"]-2
            else:
                kind = "linker"
            out.append(dict(id=len(out)+1,type=kind,atoms=[i+1 for i in atoms],formula=formula,
                            dimension=dim,formal_charge=charge))
        return out

    def report(self):
        return dict(atom_count=len(self.elements),edges=[public_edge(k) for k in sorted(self.edges)],
                    dimension=self.dimension,fragments=self.fragments(),coordination=self.coordination(),
                    partial_sites=[dict(atom=i+1,label=self.model.labels[i],occupancy=occ)
                                   for i,occ in enumerate(self.model.occupancy) if occ!=1],
                    borderline=self.borderline,source="CIF" if self.model.explicit is not None else "radii")


def bond_path(graph, start, end):
    """到目标位点任一周期像的最短键长和，同时返回实际跨过的晶胞。"""
    if not all(1 <= i <= len(graph.elements) for i in (start,end)):
        raise ValueError("路径端点超出原子编号范围")
    origin, target = start-1,end-1
    # 同一位点对可能有多条跨胞键；稀疏矩阵只放最短的那条，保留其平移。
    links = {}
    for i,neighbors in enumerate(graph.neighbors):
        for j,shift in neighbors:
            key = i,j
            candidate = (graph.distance(i,j,shift),shift)
            if key not in links or candidate < links[key]:
                links[key] = candidate
    rows = [i for i,j in links]
    cols = [j for i,j in links]
    matrix = csr_matrix(([w for w,_ in links.values()],(rows,cols)),
                        shape=(len(graph.elements),len(graph.elements)))
    distances,previous = dijkstra(matrix,indices=origin,return_predecessors=True)
    if not np.isfinite(distances[target]):
        raise ValueError("端点不连通")
    path = []
    j = target
    while j != origin:
        i = int(previous[j])
        path.append([i+1,j+1,list(links[i,j][1])])
        j = i
    return {"length":float(distances[target]),"edges":path[::-1]}
