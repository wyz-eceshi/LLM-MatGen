"""混合封端。金属编号 1 的两条配位键中，一条跨过 x 胞界。"""
import json

import numpy as np
import pytest

from llm_matgen.__main__ import main
from llm_matgen.mof.cif import edge, read_model
from llm_matgen.mof.fragment_ops import remove
from llm_matgen.mof.graph import BondGraph
from llm_matgen.mof.sidecar import bonded_cif
from llm_matgen.mof.subnet import model_at


def parent_graph(*, short_axis=False, obstacle=False):
    cell = np.diag([30.,30.,1.8 if short_axis else 30.])
    xyz = [[29,12,.6],[1,12,.6],[27,12,.6]]
    symbols = ["Zr","O","O"]
    if obstacle:
        symbols.append("C")
        xyz.append([27.,12.4,.6])
    bonds = {edge(0,1,(1,0,0)),edge(0,2,(0,0,0))}
    return BondGraph(model_at(cell,symbols,np.asarray(xyz) @ np.linalg.inv(cell),bonds))


def cap_recipe():
    return dict(predicate="element=O",formal_charges={"parent":0,"removed":-1})


@pytest.mark.parametrize("plan,expected",[("auto-charge",["OH","H2O"]),("site:Zr1=H2O,OH",["H2O","OH"])])
def test_g3_mixed_cap_each_broken_periodic_bond_is_accounted(tmp_path,plan,expected):
    source = tmp_path/"parent.cif"
    source.write_text(bonded_cif(parent_graph()),encoding="utf8")
    recipe = tmp_path/"recipe.json"
    recipe.write_text(json.dumps(cap_recipe()),encoding="utf8")
    dest = tmp_path/"out"
    assert main(["mof","remove-fragment","--input",str(source),"--recipe",str(recipe),
                 "--cap-plan",plan,"--output-dir",str(dest)]) == 0
    side = json.loads((dest/"sidecar.json").read_text("utf8"))
    sites = side["cap_sites"]
    assert {p.name for p in dest.iterdir()} == {"POSCAR","structure.cif","sidecar.json","preview.html"}
    assert [s["cap"] for s in sites] == expected
    assert [s["removed_image"] for s in sites] == [[1,0,0],[0,0,0]]
    assert [s["reference_distance"] for s in sites] == pytest.approx([2.2 if c=="OH" else 2.3 for c in expected])
    assert side["formal_charge_ledger"]["child"] == 0 and side["formal_charge_ledger"]["added"] == -1
    assert side["composition_ledger"]["actual_child"] == {"Zr":1,"H":3,"O":2}
    child = BondGraph(read_model(dest/"structure.cif"))
    assert {(j+1,t) for j,t in child.neighbors[0]} == {
        (sites[0]["cap_indices"][0],(1,0,0)),(sites[1]["cap_indices"][0],(0,0,0))}
    assert side["cap_geometry"]["closest"]["margin"] >= 0


@pytest.mark.parametrize("plan",["site:1=OH,OH","site:1=H2O","site:2=OH,H2O"])
def test_g3_invalid_site_or_charge_rejected(plan):
    with pytest.raises(ValueError):
        remove(parent_graph(),dict(cap_recipe(),cap_plan=plan))


@pytest.mark.parametrize("case,category",[("short_axis","self_image"),("obstacle","host")])
def test_g3_collision_chooses_closest_and_refuses_structure(case,category):
    report, candidate = remove(parent_graph(**{case: True}), dict(cap_recipe(),cap_plan="auto-charge"))
    assert candidate is None
    assert report["closest"]["contact"]["margin"] < 0
    assert report["cap_geometry"][category]["margin"] < 0
    assert len(report["closest"]["angles"]) == 2


def test_g3_caps_on_distinct_metals_collide_across_boundary():
    # 两金属隔胞相距 0.5 Å；即便各自偏转 20°，这对 O 也必须受碰撞约束。
    cell = np.diag([20.,20.,20.])
    xyz = [[.25,5.,5.],[.25,7.,5.],[19.75,5.,5.],[19.75,7.,5.]]
    graph = BondGraph(model_at(cell,["Zr","O","Zr","O"],np.asarray(xyz) @ np.linalg.inv(cell),
                               {edge(0,1,(0,0,0)),edge(2,3,(0,0,0))}))
    report,product = remove(graph,dict(cap_recipe(),cap_plan="auto-charge"))
    assert product is None
    between = report["cap_geometry"]["between_caps"]
    assert between["margin"] < 0 and any(between["image"])


def test_g3_cone_pairs_two_oxygens_via_periodic_carboxyl_carbon():
    cell = np.diag([20.,20.,20.])
    xyz = [[1.,5.,5.],[1.,7.,5.],[19.,5.,5.],[19.,7.,5.],[0.,7.8,5.]]
    bonds = {edge(0,1,(0,0,0)),edge(2,3,(0,0,0)),edge(1,4,(0,0,0)),edge(3,4,(1,0,0))}
    graph = BondGraph(model_at(cell,["Zr","O","Zr","O","C"],np.asarray(xyz) @ np.linalg.inv(cell),bonds))
    frag = next(f for f in graph.fragments() if 5 in f["atoms"])
    report,product = remove(graph,dict(fragment=frag["id"],formal_charges={"parent":0,"removed":-1},cap_plan="auto-charge"))
    assert product is not None
    sites = report["cap_sites"]
    assert [s["carboxyl_parent"] for s in sites] == [5,5]
    assert [s["carboxyl_image"] for s in sites] == [[0,0,0],[1,0,0]]
    geom = report["cap_geometry"]
    assert {tuple(a["caps"]) for a in geom["assignments"]} == {("OH","H2O"),("H2O","OH")}
    assert geom["closest"]["margin"] >= 0
    assert any(0 < s["tilt_degrees"] <= 20 for s in sites)
    assert len(geom["carboxyl_pairs"][0]["hydrogens"]) == 3
    assert geom["limits"]["distance"] == pytest.approx(max(a["limits"]["distance"] for a in geom["assignments"]))


def test_g3_joint_threshold_search_agrees_with_exhaustive_small_grid():
    from itertools import product
    from llm_matgen.mof.cap_cone import _maximize
    rng = np.random.default_rng(27)
    local = [dict(distance=rng.random(4),margin=np.ones(4)) for _ in range(3)]
    pairs = {(i,j):dict(distance=rng.random((4,4)),margin=rng.random((4,4))-.3)
             for i,j in ((0,1),(0,2),(1,2))}
    def value(ids):
        return min(*[t["distance"][a] for t,a in zip(local,ids)],
                   *[t["distance"][ids[i],ids[j]] for (i,j),t in pairs.items()])
    valid = [ids for ids in product(range(4),repeat=3)
             if all(t["margin"][ids[i],ids[j]]>=0 for (i,j),t in pairs.items())]
    limit,chosen = _maximize("distance",local,pairs,{"margin":0})
    assert limit == value(chosen) == max(map(value,valid))
