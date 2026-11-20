from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from llm_matgen.hetero.audit import parent_inventory


def test_hetero_copies_and_charge_come_from_both_parent_cells():
    substrate = SimpleNamespace(structure=Structure(Lattice.cubic(5), ["O"], [[0, 0, 0]]))
    film = SimpleNamespace(structure=Structure(Lattice.cubic(5), ["Cu"], [[0, 0, 0]]))
    mapping = [dict(parent=1, site=1, image=i) for i in [[0, 0, 0], [1, 0, 0]]]
    mapping.append(dict(parent=2, site=1, image=[0, 0, 0]))
    report = {"match": {"substrate": [[2, 0], [0, 1]], "film": [[1, 0], [0, 1]]}}
    recipe = {"formal_charges": {"substrate": -1, "film": 1}}
    args = ([substrate, film], ["substrate", "film"])
    parent, expected, blocks, charges = parent_inventory(*args, mapping, report, recipe)
    assert parent == expected == ["O", "O", "Cu"]
    assert charges["parent"] == charges["child"] == -1
    assert [b["copies"] for b in blocks] == [2, 1]
    # 即使保留的每个父位点重复数均相同，整体遗漏一个超胞仍须拒绝。
    with pytest.raises(ValueError, match="预期位点数"):
        parent_inventory(*args, mapping[1:], report, recipe)
    with pytest.raises(ValueError, match="形式电荷须为整数"):
        parent_inventory(*args, mapping, report, {"formal_charges": {"substrate": 0.5}})
