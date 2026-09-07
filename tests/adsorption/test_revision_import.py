from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision_files(tmp_path):
    parent = Structure(
        Lattice.cubic(8.0),
        ["Cu", "O"],
        [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]],
        site_properties={"selective_dynamics": [[False] * 3, [True] * 3]},
    )
    revised = parent.copy()
    revised.translate_sites([1], [0.0, 0.0, 0.125], frac_coords=True, to_unit_cell=False)
    parent_path = tmp_path / "parent-POSCAR"
    revised_path = tmp_path / "revised-POSCAR"
    Poscar(parent).write_file(parent_path)
    Poscar(revised).write_file(revised_path)
    old = np.asarray(Poscar.from_file(parent_path).structure[1].coords)
    new = np.asarray(Poscar.from_file(revised_path).structure[1].coords)
    sidecar = {
        "schema": "vasp-structure-revision",
        "version": 1,
        "parent_sha256": _sha(parent_path),
        "output_sha256": _sha(revised_path),
        "index_1based": 2,
        "element": "O",
        "old_cartesian_angstrom": old.tolist(),
        "new_cartesian_angstrom": new.tolist(),
        "old_fractional": [0.5, 0.5, 0.5],
        "new_fractional": [0.5, 0.5, 0.625],
        "cartesian_delta_angstrom": (new - old).tolist(),
        "reason": "manual correction",
        "created_utc": "2026-09-01T00:00:00+00:00",
        "input_path_absolute": str(parent_path.resolve()),
        "input_filename": parent_path.name,
        "output_filename": revised_path.name,
    }
    sidecar_path = tmp_path / "revision.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return parent_path, revised_path, sidecar_path


def test_parent_path_can_be_resolved_only_from_a_verified_absolute_sidecar_path(tmp_path):
    from llm_matgen.adsorption.revision import RevisionImporter

    parent, _revised, sidecar = revision_files(tmp_path)
    assert RevisionImporter.parent_from_sidecar(sidecar) == parent.resolve()

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["input_path_absolute"] = parent.name
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not absolute"):
        RevisionImporter.parent_from_sidecar(sidecar)

    payload.pop("input_path_absolute")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provide --parent"):
        RevisionImporter.parent_from_sidecar(sidecar)


def test_parent_path_fallback_rejects_hash_or_filename_mismatch(tmp_path):
    from llm_matgen.adsorption.revision import RevisionImporter

    parent, _revised, sidecar = revision_files(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["parent_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        RevisionImporter.parent_from_sidecar(sidecar)

    payload["parent_sha256"] = _sha(parent)
    payload["input_filename"] = "different-POSCAR"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        RevisionImporter.parent_from_sidecar(sidecar)


def test_revision_import_validates_and_publishes_immutable_revision(tmp_path):
    from llm_matgen.adsorption.revision import RevisionImporter

    parent, revised, sidecar = revision_files(tmp_path)
    result = RevisionImporter().import_revision(
        parent, revised, sidecar, output_root=tmp_path / "store"
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "formal_structure_revision"
    assert manifest["changed_atom_index_1based"] == 2
    assert (result.root / "POSCAR").read_bytes() == revised.read_bytes()
    assert (result.root / "revision.sidecar.json").read_bytes() == sidecar.read_bytes()


def test_revision_import_rejects_sidecar_mismatch(tmp_path):
    from llm_matgen.adsorption.revision import RevisionImporter

    parent, revised, sidecar = revision_files(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["index_1based"] = 1
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="old coordinate|declared atom"):
        RevisionImporter().import_revision(
            parent, revised, sidecar, output_root=tmp_path / "store"
        )


@pytest.mark.parametrize("field", ["old_fractional", "new_fractional", "created_utc"])
def test_revision_import_rejects_fractional_or_time_audit_mismatch(tmp_path, field):
    from llm_matgen.adsorption.revision import RevisionImporter

    parent, revised, sidecar = revision_files(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload[field] = "not-a-time" if field == "created_utc" else [9.0, 9.0, 9.0]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fractional|created_utc"):
        RevisionImporter().import_revision(
            parent, revised, sidecar, output_root=tmp_path / "store"
        )
