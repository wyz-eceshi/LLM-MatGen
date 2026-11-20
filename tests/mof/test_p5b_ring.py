"""用跨胞芳香 C–OH 检验换环保留、映射及 CLI 拒绝路径。"""
import json
from math import cos, pi, sin

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from llm_matgen.__main__ import main
from llm_matgen.mof.cif import read_model
from llm_matgen.mof.graph import BondGraph
from llm_matgen.mof.sidecar import bonded_cif


def ray(k, length):
    return length * np.array([cos(k*pi/3), sin(k*pi/3), 0.])


@pytest.fixture
def phenol_ring(tmp_path,request):
    # 父结构编号：两端锚点 1/2，羟基 O3/H4，环 C5–C10，三个芳环 H11–H13。
    # 正方胞 y 边界恰好落在父环 C6 与 O3 之间。
    ring = [ray(k, 1.40) for k in range(6)]
    aromatic_h = [ray(k, 2.48) for k in (2,4,5)]
    oxygen = ray(1, 2.76)
    hydroxy_h = oxygen + ray(1, .28) + np.array([0,0,.92])
    parent_xyz = [ray(0,2.88),ray(3,2.88),oxygen,hydroxy_h,*ring,*aromatic_h]
    parent = tmp_path/"POSCAR"
    Structure(
        Lattice.cubic(30), ["C","C","O","H"] + ["C"]*6 + ["H"]*3,
        np.asarray(parent_xyz) + [15.,28.,15.], coords_are_cartesian=True,
        to_unit_cell=True
    ).to(filename=str(parent),fmt="poscar")

    # SDF 编号从芳环的另一端开始。与父 C6 对应的是模板 C4，既非父编号也非环序号 2。
    template_ring = [3,4,5,6,1,2]
    h_sites = (1,2,4,5) if getattr(request,"param",None) == "protonated" else (2,4,5)
    symbols = ["C"]*6 + ["H"]*len(h_sites)
    template_xyz = [ring[k] for k in (4,5,0,1,2,3)] + [ray(k,2.48) for k in h_sites]
    bonds = [(template_ring[k],template_ring[(k+1)%6],1+k%2) for k in range(6)]
    bonds += [(template_ring[k],index,1) for index,k in enumerate(h_sites,7)]
    lines = ["羟基接回测试", "", "", f"{len(symbols):3}{len(bonds):3}  0  0  0  0            999 V2000"]
    lines += [f"{x:10.4f}{y:10.4f}{z:10.4f} {s:<3} 0  0  0  0  0  0"
              for (x,y,z),s in zip(template_xyz,symbols,strict=True)]
    lines += [f"{a:3}{b:3}{bond:3}  0  0  0" for a,b,bond in bonds]
    template = tmp_path/"ring.sdf"
    template.write_text("\n".join([*lines,"M  END","$$$$",""]), encoding="utf-8")
    recipe = tmp_path/"recipe.json"
    recipe.write_text(json.dumps(dict(mode="ring", ring=list(range(5,11)),
                          anchors=[1,2], template_ring=template_ring, ports=[1,4],
                          orientation="lowest-cost")), encoding="utf-8")
    dest = tmp_path/"product"
    args = ["mof","replace-fragment","--input",str(parent),"--guest",str(template),
            "--recipe",str(recipe),"--output-dir",str(dest)]
    return parent, dest, args


def test_keep_reconnects_mapped_ring_site_across_periodic_boundary(phenol_ring):
    source, dest, args = phenol_ring
    assert main(args+["--substituent","keep:3","--map","6:4"]) == 0
    side = json.loads((dest/"sidecar.json").read_text("utf-8"))
    graph = BondGraph(read_model(dest/"structure.cif"))
    parent_map = side["atom_map"]
    template_map = side["template_integrity"]["atom_map"]
    new_site, = [int(i) for i,t in template_map.items() if t==4]
    o = parent_map.index(3)
    h = parent_map.index(4)

    # 确认 O 连到正确的模板原子、其原有 H，以及正确的胞像。
    assert {j for j,_ in graph.neighbors[o]} == {new_site-1,h}
    shift, = [t for j,t in graph.neighbors[o] if j==new_site-1]
    assert tuple(shift) == (0,-1,0)
    assert graph.distance(o,new_site-1,shift) == pytest.approx(1.36,abs=.03)
    h_shift, = [t for j,t in graph.neighbors[o] if j==h]
    assert graph.distance(o,h,h_shift) == pytest.approx(np.hypot(.28,.92),abs=.001)
    assert graph.symbols[o] == "O" and graph.symbols[h] == "H"
    assert side["status"] == "unoptimized_candidate"

    # 新旧环总体组成相同，旧 OH 内部键是唯一一条保留边。
    ledger = side["composition_ledger"]
    assert ledger["actual_child"] == ledger["expected_child"] == ledger["parent"] == {"C":8,"H":4,"O":1}
    assert {r["index"] for r in ledger["removed"]} == set(range(5,14))
    assert {r["template_index"] for r in ledger["added"]} == set(range(1,10))
    diff = side["graph_diff"]
    assert diff["preserved"] == 1 and len(diff["removed"]) == len(diff["added"]) == 12
    assert [3,6,[0,-1,0]] in diff["removed"]
    assert [o+1,new_site,[0,-1,0]] in diff["added"]
    assert main(["mof","graph-diff","--parent",str(source),"--revised",str(dest/"POSCAR"),
                 "--atom-map",str(dest/"sidecar.json")]) == 0


@pytest.mark.parametrize("phenol_ring",["protonated"],indirect=True)
def test_drop_removes_oxygen_and_its_hydrogen_and_preserves_anchors(phenol_ring):
    _, dest, args = phenol_ring
    assert main(args+["--substituent","drop:3"]) == 0
    side = json.loads((dest/"sidecar.json").read_text("utf-8"))
    ledger = side["composition_ledger"]
    assert ledger["expected_child"] == ledger["actual_child"] == {"C":8,"H":4}
    assert {r["index"] for r in ledger["removed"]} == set(range(3,14))
    assert {i for i in side["atom_map"] if i is not None} == {1,2}
    assert len(side["graph_diff"]["removed"]) == 13
    assert len(side["graph_diff"]["added"]) == 12
    assert side["graph_diff"]["preserved"] == 0


@pytest.mark.parametrize("flags", [
    pytest.param([], id="undeclared_branch"),
    pytest.param(["--substituent","keep:3"], id="missing_ring_map"),
    pytest.param(["--substituent","keep:3","--substituent","drop:3","--map","6:4"],
                 id="conflicting_branch_declarations"),
    pytest.param(["--substituent","keep:3","--map","6:4","--map","6:5"],
                 id="contradictory_ring_map"),
])
def test_external_substituent_requires_consistent_declaration(phenol_ring, flags):
    _, dest, args = phenol_ring
    assert main(args+flags) == 2
    assert not dest.exists()


def test_rejects_unaccounted_bond_to_periodic_copy_of_same_ring(phenol_ring, capsys):
    parent, dest, args = phenol_ring
    graph = BondGraph(read_model(parent))
    # 本胞 C5 与下一原胞的 C8 相连；这条边不属于已选的六边环。
    graph.edges.add((4, 7, (1, 0, 0)))
    network = parent.with_name("crosslinked.cif")
    network.write_text(bonded_cif(graph), encoding="utf-8")
    args[args.index("--input") + 1] = str(network)
    assert main(args + ["--substituent", "keep:3", "--map", "6:4"]) == 2
    output = capsys.readouterr()
    assert "另一周期像" in output.err + output.out
    assert not dest.exists()
