"""保留 CIF 位点顺序、占位和显式周期键。"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from pymatgen.core import Element, Lattice, Structure, SymmOp
from pymatgen.io.cif import CifFile, str2float
from pymatgen.io.vasp.inputs import Poscar


@dataclass
class Model:
    structure: Structure
    labels: list
    groups: list
    occupancy: list
    explicit: set | None
    assemblies: list

    def slice(self, keep, edges):
        return Model(Structure.from_sites([self.structure[i] for i in keep]),
                     [self.labels[i] for i in keep], ["."] * len(keep),
                     [1.] * len(keep), edges, ["."] * len(keep))


def edge(i, j, shift):
    shift = tuple(int(x) for x in shift)
    return min((i, j, shift), (j, i, tuple(-x for x in shift)))


def symmetry_point(code, xyz, operations):
    if code == ".":
        return xyz
    match = re.fullmatch(r"(\d+)(?:_(\d{3}))?", code)
    if not match or match[1] not in operations:
        raise ValueError(f"无法解释键的对称编码 {code}")
    offset = np.array(list(match[2] or "555"), dtype=int) - 5
    return operations[match[1]].operate(xyz) + offset


def read_model(path, expand=False):
    path = Path(path)
    if path.suffix.lower() != ".cif":
        if path.suffix.lower() in {".json", ".mson"}:
            import json
            structure = Structure.from_dict(json.loads(path.read_text("utf-8")))
        else:
            structure = Poscar.from_file(path, check_for_potcar=False).structure
        if not structure.is_ordered:
            raise ValueError("部分占位结构请提供带位点记录的 CIF")
        n = len(structure)
        return Model(structure, [f"{s.specie.symbol}{i}" for i, s in enumerate(structure, 1)],
                     ["."] * n, [1.] * n, None, ["."] * n)
    blocks = list(CifFile.from_file(path).data.values())
    if len(blocks) != 1:
        raise ValueError("请选择单个结构 CIF")
    data = {k.lower(): v for k, v in blocks[0].data.items()}
    labels = data["_atom_site_label"]
    if len(set(labels)) != len(labels):
        raise ValueError("CIF 原子标签重复")
    lattice = Lattice.from_parameters(*(str2float(data["_cell_"+key]) for key in
        ("length_a","length_b","length_c","angle_alpha","angle_beta","angle_gamma")))
    sym = data.get("_space_group_symop_operation_xyz", data.get("_symmetry_equiv_pos_as_xyz"))
    if sym is None:
        from pymatgen.io.cif import CifParser
        ops = CifParser(path).get_symops(blocks[0])
    else:
        ops = [SymmOp.from_xyz_str(x) for x in ([sym] if isinstance(sym,str) else sym)]
    if len(ops) > 1 and not expand:
        raise ValueError("非 P1 输入需要 --expand-symmetry")
    ids = data.get("_space_group_symop_id",data.get("_symmetry_equiv_pos_site_id",map(str,range(1,len(ops)+1))))
    operations = dict(zip(ids, ops, strict=True))
    raw = dict(zip(labels, np.array([[str2float(v) for v in data["_atom_site_fract_"+a]]
                                     for a in "xyz"]).T, strict=True))
    coords, species, names, occupancy, groups, assemblies = [], [], [], [], [], []
    copies = {}
    types = data.get("_atom_site_type_symbol", labels)
    for row, name in enumerate(labels):
        symbol = re.match(r"[A-Z][a-z]?", types[row]).group()
        Element(symbol)
        copies[name] = []
        for op in ops:
            point = op.operate(raw[name])
            if any(np.max(np.abs((point-coords[i])-np.rint(point-coords[i]))) < 1e-5
                   for i in copies[name]):
                continue
            copies[name].append(len(coords))
            coords.append(point % 1)
            species.append(symbol)
            names.append(name)
            occupancy.append(str2float(data.get("_atom_site_occupancy", ["1"]*len(labels))[row]))
            groups.append(data.get("_atom_site_disorder_group", ["."]*len(labels))[row])
            assemblies.append(data.get("_atom_site_disorder_assembly", ["."]*len(labels))[row])
    def locate(name, point):
        for i in copies[name]:
            delta = point - coords[i]
            shift = np.rint(delta).astype(int)
            if np.max(np.abs(delta-shift)) < 1e-5:
                return i, shift
        raise ValueError(f"键端点未映射到展开坐标 {name}")
    bonds = set() if data.get("_matgen_explicit_bond_count") == "0" else None
    if "_geom_bond_atom_site_label_1" in data:
        bonds = set()
        first = data["_geom_bond_atom_site_label_1"]
        second = data["_geom_bond_atom_site_label_2"]
        codes = [data.get("_geom_bond_site_symmetry_"+n, ["."]*len(first)) for n in ("1","2")]
        for left,right,c1,c2 in zip(first,second,*codes,strict=True):
            one = symmetry_point(c1, raw[left], operations)
            two = symmetry_point(c2, raw[right], operations)
            for op in ops:
                i, offset_i = locate(left, op.operate(one))
                j, offset_j = locate(right, op.operate(two))
                bonds.add(edge(i,j,offset_j-offset_i))
    return Model(Structure(lattice,species,coords),names,groups,occupancy,bonds,assemblies)
