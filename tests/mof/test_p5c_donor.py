"""G2 单锚点供体、客体置换和多角色排斥的合成回归。"""
import json

import numpy as np
import pytest

from llm_matgen.__main__ import main
from llm_matgen.mof.donor import resolve_displacement
from llm_matgen.mof.graph import BondGraph
from llm_matgen.mof.placement import scan
from llm_matgen.mof.subnet import model_at


def host_graph(symbols, xyz, length=16):
    return BondGraph(model_at(np.eye(3)*length, symbols, np.asarray(xyz)/length, set()))


def guest(tmp_path, rows):
    path = tmp_path/"guest.xyz"
    path.write_text(f"{len(rows)}\nsynthetic\n"+"\n".join(rows)+"\n", encoding="utf-8")
    return path


def test_g2_nonmetal_donor_displaces_whole_guest_water_across_boundary(tmp_path):
    host = host_graph(["O", "Zr", "Zr"], [[15, 8, 8], [7, 4, 8], [9, 8, 5]])
    mol = guest(tmp_path, ["U 0 0 0", "F 2.25 0 0", "F -2.25 0 0",
                           "O 0 2.45 0", "H 0.75 3.05 0", "H -0.75 3.05 0"])
    recipe = dict(site={"kind":"donor"}, host_atom="element=O", guest_anchor=1, displace="1",
                  guest_fragments={"1":{"atoms":[4, 5, 6], "formal_charge":0}},
                  exclude_contact=["guest=element=F@host=element=Zr:2.82"],
                  distance_window=[2.2, 2.7], radii=[2.21, 2.5], prefer_distance=2.23,
                  min_nonbond=1., directions=[[1, 0, 0]], spins_deg=[0, 90], select=1)
    report, (child, mapping) = scan(host, mol, recipe)
    assert report["total"] == 4 and report["rejected"] == 0
    assert report["selected"] == 1 and report["top"][0]["distances"] == [2.21]
    assert report["template_indices"] == [1, 2, 3]
    assert [row["index"] for row in report["template_discarded"]] == [4, 5, 6]
    part = resolve_displacement(mol, recipe)
    assert part == report["displaced_fragment"]
    assert part["cut_bond"] == [1, 4] and part["composition"] == {"H":2, "O":1}
    assert part["formal_charge"] == 0
    assert mapping == [1, 2, 3, None, None, None]
    assert child.symbols == host.symbols + ["U", "F", "F"]
    assert (0, 3, (1, 0, 0)) in child.edges and len(child.edges) == 3
    shortest = child.structure.lattice.get_all_distances(child.coords[[4,5]], child.coords[[1,2]]).min()
    assert report["top"][0]["exclusion_distances"] == pytest.approx([shortest])
    assert report["top"][0]["min_nonbond"] >= report["top"][1]["min_nonbond"]
    with pytest.raises(ValueError, match="片段"):
        resolve_displacement(mol, dict(recipe, displace=[4, 5]))
    with pytest.raises(ValueError, match="锚点"):
        resolve_displacement(mol, dict(recipe, displace=[1, 2]))


def test_g2_exclusion_tests_guest_uranium_even_when_fluorine_is_anchor(tmp_path):
    host = host_graph(["Zr", "O"], [[3,8,8], [9,8,8]])
    mol = guest(tmp_path, ["U 0 0 0", "F 2 0 0"])
    recipe = dict(site={"kind":"donor"}, host_atom=1, guest_anchor=2, distance_window=[2, 2],
                  directions=[[1,0,0]], spins_deg=[0], exclude_contact=["guest=1@host=2:>3.0"],
                  min_nonbond=.8, select=1)
    report, child = scan(host, mol, recipe)
    assert child is None and report["exclusion_rejected"] == 1
    assert report["closest"]["exclusion_distances"] == pytest.approx([2.])
    assert report["closest"]["violations"]["exclude_contact"] == pytest.approx([1.])
    report, child = scan(host, mol, dict(recipe, directions=[[-1,0,0]]))
    assert child and report["top"][0]["exclusion_distances"][0] > 3.


def test_g2_cli_strict_exclusion_rejects_equality_and_reports_deviations(tmp_path, capsys):
    host = host_graph(["N", "Zr"], [[0,0,0], [6,0,0]])
    inp = tmp_path/"POSCAR"
    host.structure.to(filename=str(inp), fmt="poscar")
    mol = guest(tmp_path, ["F 0 0 0"])
    cfg = tmp_path/"recipe.json"
    out = tmp_path/"out"
    settings = dict(radii=[3], directions=[[1,0,0]], min_nonbond=.1, select=1)
    cfg.write_text(json.dumps(settings), encoding="utf-8")
    args = ["mof","scan-placement","--input",str(inp),"--guest",str(mol),"--recipe",str(cfg),
            "--site","donor","--host-atom","element=N","--guest-anchor","1","--distance-window","3,3",
            "--exclude-contact","guest=1@host=element=Zr:>3","--output-dir",str(out)]
    assert main(args) == 2
    row = json.loads(capsys.readouterr().out)["closest"]
    assert row["exclusion_distances"] == [3.]
    assert row["violations"] == dict(distance_window=0., min_nonbond=0.,
                                      exclude_contact=[0.], strict_exclusions_failed=[1])
    assert not out.exists()
    settings["min_nonbond"] = 3.25
    cfg.write_text(json.dumps(settings), encoding="utf-8")
    args[args.index("3,3")] = "1,2"
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)["closest"]["violations"] == dict(
        distance_window=1., min_nonbond=.25, exclude_contact=[0.], strict_exclusions_failed=[1])
    assert not out.exists()


def test_g2_nonbond_floor_includes_other_guest_periodic_images(tmp_path):
    host = host_graph(["O"], [[0,0,0]], length=8)
    mol = guest(tmp_path, ["C 0 0 0", "C 7.8 0 0"])
    recipe = dict(site={"kind":"donor"}, host_atom=1, guest_anchor=1, distance_window=[2,2],
                  directions=[[1,0,0]], spins_deg=[0], min_nonbond=1.4)
    report, product = scan(host, mol, recipe)
    assert product is None
    assert report["closest"]["min_nonbond"] == pytest.approx(.2)
    assert report["closest"]["violations"]["min_nonbond"] == pytest.approx(1.2)
