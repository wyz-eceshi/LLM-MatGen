"""用合成 Zr 位点贯通去 OH、同位点补 F 与两步的电荷账。"""
import json
import numpy as np
import pytest

from llm_matgen.__main__ import main
from llm_matgen.mof.cif import edge, read_model
from llm_matgen.mof.graph import BondGraph
from llm_matgen.mof.sidecar import bonded_cif
from llm_matgen.mof.subnet import model_at


def test_terminal_oh_exchanged_for_f_on_same_periodic_metal(tmp_path):
    cell = np.diag([20.,20.,20.])
    graph = BondGraph(model_at(cell, ["Zr","O","H"], [[.95,.5,.5],[.05,.5,.5],[.075,.54,.5]],
                               {edge(0,1,(1,0,0)),edge(1,2,(0,0,0))}))
    source = tmp_path/"host.cif"
    source.write_text(bonded_cif(graph),encoding="utf-8")
    group, = [f for f in graph.fragments() if f["formal_charge"] == -1]
    config = tmp_path/"remove.json"
    config.write_text(json.dumps({"fragment":group["id"],"formal_charges":{"parent":0}}),encoding="utf-8")
    bare = tmp_path/"bare"
    assert main(["mof","remove-fragment","--input",str(source),"--recipe",str(config),"--output-dir",str(bare)]) == 0
    first = json.loads((bare/"sidecar.json").read_text("utf-8"))
    assert first["formal_charge_ledger"] == dict(parent=0,removed=-1,added=0,child=1)
    assert first["atom_map"] == [1]
    cfg = tmp_path/"place.json"
    cfg.write_text(json.dumps(dict(directions=[[1,0,0]],radii=[2.1],min_nonbond=1.65,
                                   guest_charge=-1,select=1)),encoding="utf-8")
    guest = tmp_path/"F.xyz"
    guest.write_text("1\nfluoride\nF 0 0 0\n",encoding="utf-8")
    product = tmp_path/"exchange"
    assert main(["mof","scan-placement","--site","donor","--host-atom","1","--guest-anchor","1",
                 "--distance-window","1.95,2.25","--input",str(bare/"structure.cif"),"--recipe",str(cfg),
                 "--guest",str(guest),"--output-dir",str(product)]) == 0
    meta = json.loads((product/"sidecar.json").read_text("utf-8"))
    assert meta["formal_charge_ledger"] == dict(parent=1,removed=0,added=-1,child=0)
    assert meta["composition_ledger"]["actual_child"] == {"Zr":1,"F":1}
    assert not meta["graph_diff"]["removed"] and len(meta["graph_diff"]["added"]) == 1
    result = BondGraph(read_model(product/"structure.cif"))
    (fluorine,image), = result.neighbors[0]
    assert fluorine == 1 and tuple(image) == (1,0,0)
    assert result.distance(0,fluorine,image) == pytest.approx(2.1)
