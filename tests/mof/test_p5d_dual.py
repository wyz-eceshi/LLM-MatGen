"""双距离窗口、同一原子的双齿配位、多片段去除及周期像。"""
import numpy as np
import pytest

from llm_matgen.mof.donor import resolve_displacements
from llm_matgen.mof.graph import BondGraph
from llm_matgen.mof.placement import scan
from llm_matgen.mof.subnet import model_at


def model(symbols, points):
    return BondGraph(model_at(16*np.eye(3),symbols,np.array(points)/16,set()))


def recipe(anchors, **kwargs):
    return dict(site={"kind":"donor"}, anchors=anchors, min_nonbond=.6, select=1,
                azimuths_deg=[0,90,180,270], hinges_deg=[0,90,180,270], spins_deg=[0,90], **kwargs)


def test_dual_distinct_guest_anchors_build_two_edges_to_actual_periodic_images(tmp_path):
    host = model(["O","Zr"], [[13,8,8],[3,8,8]])
    guest = tmp_path/"UF.xyz"
    guest.write_text("2\nsynthetic\nU 0 0 0\nF 2 0 0\n",encoding="utf-8")
    r, (child, _) = scan(host,guest,recipe(["1~1:2,2","2~2:2,2"]))
    row = r["top"][0]
    assert row["distances"] == pytest.approx([2,2])
    assert row["host_images"] == [[0,0,0],[1,0,0]]
    assert {tuple(sorted((a,b))) for a,b,_ in child.edges} == {(0,2),(1,3),(2,3)}
    for h,j in ((0,2),(1,3)):
        i,k,im = next(edge for edge in child.edges if edge[:2] == (h,j))
        assert child.distance(i,k,im) == pytest.approx(2.)


def test_dual_one_guest_atom_binds_both_donors_and_all_displaced_fragments_are_accounted(tmp_path):
    host = model(["O","N"], [[4,8,8],[7,8,8]])
    guest = tmp_path/"cluster.xyz"
    guest.write_text("8\nsynthetic\nU 0 0 0\nF -2.2 0 0\nO 0 2.45 0\nH .75 3.04 0\nH -.75 3.04 0\n"
                     "O 2.45 0 0\nH 3.04 .75 0\nH 3.04 -.75 0\n",encoding="utf-8")
    cfg = recipe(["1~1:2.2,2.7","1~2:2.2,2.7"], prefer_distances=[2.29,2.51],
                 guest_fragments={"w1":{"atoms":[3,4,5],"formal_charge":0},"w2":{"atoms":[6,7,8],"formal_charge":0}},
                 displace={"fragments":["w1","w2"]})
    r, (child, _) = scan(host,guest,cfg)
    assert r["top"][0]["distances"] == pytest.approx([2.29,2.51])
    assert r["displaced_fragments"] == resolve_displacements(guest,cfg)
    assert r["template_indices"] == [1,2]
    assert [(x["index"],x["fragment"]) for x in r["template_discarded"]] == [(i,"w1" if i<6 else "w2") for i in range(3,9)]
    assert {edge[:2] for edge in child.edges} == {(0,2),(1,2),(2,3)}
    with pytest.raises(ValueError, match="重复"):
        resolve_displacements(guest,dict(cfg,displace={"fragments":["w1","w1"]}))


def test_dual_geometric_failure_reports_real_closest_pose_and_separate_deficits(tmp_path):
    host = model(["O","Zr"], [[1,0,0],[8,0,0]])
    guest = tmp_path/"UF.xyz"
    guest.write_text("2\ntest\nU 0 0 0\nF 2 0 0\n",encoding="utf-8")
    r, failed = scan(host,guest,recipe(["1~1:2,2","2~2:2,2"]))
    assert failed is None and r["rejected"] == r["total"] > 0
    row = r["closest"]
    assert row["distances"] == pytest.approx([2,3])
    assert row["violations"]["distance_windows"] == pytest.approx([0,1])
    assert np.linalg.norm(np.diff(row["guest_cartesian"],axis=0)) == pytest.approx(2.)
    assert min(x["minimum_window_deviation"] for x in r["image_reachability"]) == pytest.approx(1.)


def test_dual_tangent_solution_without_any_requested_direction_pointing_at_it(tmp_path):
    host = model(["O","N"], [[1,2,3],[3,4,5]])
    guest = tmp_path/"M.xyz"
    guest.write_text("1\ntest\nCu 0 0 0\n",encoding="utf-8")
    rad = np.sqrt(3)
    r, child = scan(host,guest,recipe([{"guest":1,"host":i,"window":[rad,rad]} for i in (1,2)]))
    assert child is not None
    assert r["top"][0]["guest_cartesian"][0] == pytest.approx([2,3,4],abs=1e-7)


def test_dual_publication_rebuilds_from_template_and_transform_despite_corrupt_snapshot(tmp_path,monkeypatch):
    from llm_matgen.mof import dual
    host = model(["O","N"], [[4,8,8],[7,8,8]])
    mol = tmp_path/"M.xyz"
    mol.write_text("1\nsynthetic\nCu 0 0 0\n",encoding="utf-8")
    rebuild = dual.rebuild_selected
    def corrupt_snapshot(host,path,cfg,row):
        row["guest_cartesian"] = [[0.,0.,0.]]
        return rebuild(host,path,cfg,row)
    monkeypatch.setattr(dual,"rebuild_selected",corrupt_snapshot)
    report, (child,_) = scan(host,mol,recipe(["1~1:2,2","1~2:2.2,2.2"]))
    row = report["top"][0]
    assert child.structure.cart_coords[-1] == pytest.approx(row["guest_cartesian"][0])
    assert row["guest_cartesian"] != [[0.,0.,0.]]
    assert row["reconstructed"]["distances"] == pytest.approx([2,2.2])
    row["rigid_transform"]["anchor_cartesian"] = [0.,0.,0.]
    with pytest.raises(ValueError,match="复查"):
        rebuild(host,mol,recipe(["1~1:2,2","1~2:2.2,2.2"]),row)


@pytest.mark.parametrize("anchors", [
    ["1,2~1:2,3","2~2:2,3"],
    ["1~1,2:2,3","1~2:2,3"],
    ["1~1:2,3",{"guest":1,"host":1,"window":[2,3],"image":[0,0,0]}],
])
def test_dual_rejects_multivalued_or_effectively_duplicate_anchor(anchors):
    from llm_matgen.mof.dual import anchor_specs
    with pytest.raises(ValueError):
        anchor_specs({"anchors":anchors})
