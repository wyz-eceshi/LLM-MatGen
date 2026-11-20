"""§5 七个横向结回归，每项使用独立小晶胞。"""
import json
import numpy as np
import pytest
from pymatgen.core import Structure
from llm_matgen.__main__ import main
from llm_matgen.hetero.check import check
from llm_matgen.hetero.lateral import build
from llm_matgen.mof.cif import Model,read_model


def slab(points,species=None):
    species=species or ["C"]*len(points)
    s=Structure(np.diag([10.,10.,20.]),species,points,coords_are_cartesian=True)
    return Model(s,[f"{el}{i+1}" for i,el in enumerate(species)],["."]*len(s),[1.]*len(s),None,["."]*len(s))


def params(**kw):
    return dict(dict(axis=0,sizes=[4.],reorigin=False,shifts=[0.],dz=[0.],pair_floors={},
                     check=dict(bond_window=[1.99,2.01],cutoffs={"C-C":1.25})),**kw)


def test_pitfall_1_signed_layer_translation():
    left=slab([[1,1,5],[3,1,5]])
    right=slab([[2,1,6],[4,1,6]])
    r,s=build(left,right,params(sizes=[8.],dz=[-1.]))
    xyz=s[0].cart_coords
    assert np.allclose(xyz[:,0],[0,2,4,6]) and np.ptp(xyz[:,2])==pytest.approx(0)
    assert r["selected"]["cross_minimum"]==pytest.approx(2)


def test_pitfall_2_rebuild_selected_cell_before_export(tmp_path):
    model=slab([[1,2,5]])
    source=tmp_path/"slab.json"
    source.write_text(json.dumps(model.structure.as_dict()),encoding="utf8")
    recipe=tmp_path/"recipe.json"
    recipe.write_text(json.dumps(params(sizes=[4.,6.])),encoding="utf8")
    out=tmp_path/"best"
    assert main(["hetero","build","--mode","lateral","--substrate",str(source),"--film",str(source),"--recipe",str(recipe),"--output-dir",str(out)])==0
    actual=read_model(out/"structure.cif").structure
    row=json.loads((out/"report.json").read_text("utf8"))["selected"]
    header=next(s for s in (out/"structure.cif").read_text("utf8").splitlines() if s.startswith("# selected "))
    assert json.loads(header.removeprefix("# selected "))==row
    assert actual.lattice.a==row["size"]==4 and actual.get_distance(0,1)==pytest.approx(2)


def test_pitfall_3_inconsistent_gap_and_cell():
    s=slab([[1,2,5]])
    with pytest.raises(ValueError,match="不自洽"):
        build(s,s,params(gaps=[3.]))


def test_pitfall_4_cross_collision_ignores_short_internal_bond():
    s=slab([[1,2,5],[1,3,5]])
    recipe=params(cross_floor=1.9)
    _,built=build(s,s,recipe)
    measured=check(built[0],2,0,recipe["check"])
    assert measured["global_minimum"]==pytest.approx(1)
    assert measured["cross_minimum"]==pytest.approx(2)


def test_pitfall_5_search_full_coherent_period():
    left,right=slab([[1,7,5]]),slab([[1,1,5]])
    recipe=params(step=.5)
    del recipe["shifts"]
    r,_=build(left,right,recipe)
    assert r["scanned"]==20 and r["selected"]["shift"]==6


def test_pitfall_6_reorigin_keeps_boundary_plane_together():
    s=slab([[.01,1,5],[9.99,2,5],[5.01,1,5],[4.99,2,5]],["C","O","C","O"])
    r,built=build(s,s,params(sizes=[18.],reorigin=True))
    measured=check(built[0],4,0,{})
    assert np.allclose(r["reorigin_fraction"],[.25,.25])
    assert measured["face_compositions"]==[[{"C":1,"O":1}]*2]*2


def test_pitfall_7_frozen_registry_preserves_manual_translation():
    left,right=slab([[1,2,5]]),slab([[1,5,5]])
    r,built=build(left,right,params(fix_registry=.4,shifts=[-3.,0.,1.]))
    assert r["scanned"]==1
    assert built[0].cart_coords[1,1]-built[0].cart_coords[0,1]==pytest.approx(3.4)
