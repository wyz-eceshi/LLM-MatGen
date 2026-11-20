"""通过正式 CLI 验证形式电荷、读回组成和新增原子的模板来源。"""
import json

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

from llm_matgen.__main__ import main


def placement(tmp_path):
    host = tmp_path / "POSCAR"
    Structure(Lattice.cubic(15), ["C"], [[0,0,0]]).to(filename=str(host), fmt="poscar")
    guest = tmp_path / "ion.xyz"
    guest.write_text("1\nion\nNa 0 0 0\n", encoding="utf-8")
    cfg = tmp_path / "recipe.json"
    cfg.write_text(json.dumps({"site":{"kind":"pore"}, "min_nonbond":1.5,
                   "formal_charges":{"parent":0,"added":1}}), encoding="utf-8")
    out = tmp_path / "placed"
    args = ["mof","scan-placement","--input",str(host),"--guest",str(guest),
            "--recipe",str(cfg),"--output-dir",str(out)]
    return args, out


def test_charged_guest_maps_to_template_and_composition(tmp_path):
    args, out = placement(tmp_path)
    assert main(args) == 0
    meta = json.loads((out/"sidecar.json").read_text("utf-8"))
    assert meta["template_integrity"]["atom_map"] == {"2":1}
    ledger = meta["composition_ledger"]
    assert ledger["parent"] == {"C":1} and ledger["removed"] == []
    assert ledger["expected_child"] == ledger["actual_child"] == {"C":1,"Na":1}
    assert ledger["added"] == [{"index":2,"element":"Na","source":"guest","template_index":1}]
    assert meta["formal_charge_ledger"] == {"parent":0,"removed":0,"added":1,"child":1}


@pytest.mark.parametrize("output_file", ["POSCAR","structure.cif"])
def test_output_readback_rejects_wrong_species(tmp_path,monkeypatch,output_file):
    from llm_matgen.mof import sidecar
    args, out = placement(tmp_path)
    if output_file == "POSCAR":
        writer = Poscar.get_str
        monkeypatch.setattr(Poscar,"get_str",lambda self,*a,**k: writer(self,*a,**k).replace("Na","Mg"))
    else:
        writer = sidecar.bonded_cif
        monkeypatch.setattr(sidecar,"bonded_cif",lambda g: writer(g).replace("Na","Mg"))
    assert main(args) == 2
    assert not out.exists()
