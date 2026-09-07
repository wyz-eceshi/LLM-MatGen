from __future__ import annotations

import json

import pytest
from monty.json import MontyEncoder
from pymatgen.core import Lattice, Molecule, Structure


def test_read_molecule_xyz_and_mson_preserve_order_charge_and_spin(tmp_path):
    from llm_matgen.io.readers import read_molecule

    molecule = Molecule(["O", "H"], [[0, 0, 0], [0, 0, 1]], charge=-1, spin_multiplicity=1)
    xyz = tmp_path / "oh.xyz"
    mson = tmp_path / "oh.mson.json"
    molecule.to(filename=str(xyz))
    mson.write_text(json.dumps(molecule, cls=MontyEncoder), encoding="utf-8")
    from_xyz = read_molecule(xyz, charge=-1, spin_multiplicity=1)
    from_mson = read_molecule(mson)
    assert [str(site.specie) for site in from_xyz] == ["O", "H"]
    assert int(round(float(from_xyz.charge))) == -1
    assert from_mson.as_dict() == molecule.as_dict()


def test_read_molecule_rejects_periodic_structure_mson(tmp_path):
    from llm_matgen.io.readers import MoleculeReadError, read_molecule

    path = tmp_path / "slab.mson.json"
    path.write_text(
        json.dumps(Structure(Lattice.cubic(4), ["Cu"], [[0, 0, 0]]), cls=MontyEncoder),
        encoding="utf-8",
    )
    with pytest.raises(MoleculeReadError, match="periodic Structure"):
        read_molecule(path)
