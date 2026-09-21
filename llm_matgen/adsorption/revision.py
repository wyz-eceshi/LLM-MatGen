"""Audited import of a single-coordinate manual POSCAR revision."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from monty.json import MontyEncoder
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from llm_matgen.utils.structure import versioned_structure_hashes
from llm_matgen.config import default_data_root

DEFAULT_REVISION_ROOT = default_data_root() / "revisions"


@dataclass(frozen=True)
class ImportedRevision:
    revision_id: str
    root: Path
    manifest_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _triplet(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"sidecar {name} must contain three finite numbers")
    return array


class RevisionImporter:
    @staticmethod
    def parent_from_sidecar(sidecar_path: Path | str) -> Path:
        """Resolve a sidecar-recorded parent only when identity is verifiable."""

        sidecar_path = Path(sidecar_path).resolve()
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("manual revision sidecar is not readable JSON") from exc
        if sidecar.get("schema") != "vasp-structure-revision" or sidecar.get("version") != 1:
            raise ValueError("unsupported manual revision sidecar schema")
        recorded = sidecar.get("input_path_absolute")
        if not isinstance(recorded, str) or not recorded.strip():
            raise ValueError(
                "sidecar has no absolute parent path; provide --parent explicitly"
            )
        raw_parent = Path(recorded)
        if not raw_parent.is_absolute():
            raise ValueError(
                "sidecar parent path is not absolute; provide --parent explicitly"
            )
        if raw_parent.is_symlink():
            raise ValueError(
                "sidecar parent path is a symbolic link; provide --parent explicitly"
            )
        try:
            parent_path = raw_parent.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                "sidecar parent path is not safely readable; provide --parent explicitly"
            ) from exc
        if not parent_path.is_file():
            raise ValueError(
                "sidecar parent path is not a file; provide --parent explicitly"
            )
        if sidecar.get("input_filename") != parent_path.name:
            raise ValueError("sidecar parent path filename does not match input_filename")
        expected_hash = sidecar.get("parent_sha256")
        if not isinstance(expected_hash, str) or expected_hash != _sha256(parent_path):
            raise ValueError("sidecar parent path hash does not match parent_sha256")
        return parent_path

    def import_revision(
        self,
        parent_path: Path | str,
        revised_path: Path | str,
        sidecar_path: Path | str,
        *,
        output_root: Path | str = DEFAULT_REVISION_ROOT,
    ) -> ImportedRevision:
        parent_path = Path(parent_path).resolve()
        revised_path = Path(revised_path).resolve()
        sidecar_path = Path(sidecar_path).resolve()
        if len({parent_path, revised_path, sidecar_path}) != 3:
            raise ValueError("parent, revised structure, and sidecar must be different files")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("schema") != "vasp-structure-revision" or sidecar.get("version") != 1:
            raise ValueError("unsupported manual revision sidecar schema")
        try:
            created = datetime.fromisoformat(str(sidecar.get("created_utc")))
        except (TypeError, ValueError) as exc:
            raise ValueError("sidecar created_utc must be an ISO timestamp") from exc
        if created.tzinfo is None:
            raise ValueError("sidecar created_utc must include a timezone")
        if sidecar.get("input_filename") != parent_path.name:
            raise ValueError("sidecar input filename does not match the parent file")
        if sidecar.get("output_filename") != revised_path.name:
            raise ValueError("sidecar output filename does not match the revised file")
        if sidecar.get("parent_sha256") != _sha256(parent_path):
            raise ValueError("sidecar parent hash does not match the parent file")
        if sidecar.get("output_sha256") != _sha256(revised_path):
            raise ValueError("sidecar output hash does not match the revised file")
        parent = Poscar.from_file(parent_path, check_for_potcar=False).structure
        revised = Poscar.from_file(revised_path, check_for_potcar=False).structure
        if len(parent) != len(revised):
            raise ValueError("manual revision changed the atom count")
        if [str(site.specie) for site in parent] != [str(site.specie) for site in revised]:
            raise ValueError("manual revision changed element order")
        if not np.allclose(parent.lattice.matrix, revised.lattice.matrix, atol=1.0e-8):
            raise ValueError("manual revision changed the lattice")
        if tuple(parent.lattice.pbc) != tuple(revised.lattice.pbc):
            raise ValueError("manual revision changed periodic boundary conditions")
        parent_constraints = parent.site_properties.get("selective_dynamics")
        revised_constraints = revised.site_properties.get("selective_dynamics")
        if (parent_constraints is None) != (revised_constraints is None) or (
            parent_constraints is not None
            and not np.array_equal(parent_constraints, revised_constraints)
        ):
            raise ValueError("manual revision changed selective dynamics")
        index = int(sidecar.get("index_1based", 0)) - 1
        if not 0 <= index < len(parent):
            raise ValueError("sidecar atom index is outside the structure")
        if sidecar.get("element") != str(parent[index].specie):
            raise ValueError("sidecar element does not match the declared atom")
        old_cart = _triplet(sidecar.get("old_cartesian_angstrom"), "old coordinate")
        new_cart = _triplet(sidecar.get("new_cartesian_angstrom"), "new coordinate")
        delta = _triplet(sidecar.get("cartesian_delta_angstrom"), "delta")
        old_fractional = _triplet(sidecar.get("old_fractional"), "old fractional coordinate")
        new_fractional = _triplet(sidecar.get("new_fractional"), "new fractional coordinate")
        if not np.allclose(old_cart, parent[index].coords, atol=1.0e-7):
            raise ValueError("sidecar old coordinate does not match the parent")
        if not np.allclose(new_cart, revised[index].coords, atol=1.0e-7):
            raise ValueError("sidecar new coordinate does not match the revised structure")
        if not np.allclose(new_cart - old_cart, delta, atol=1.0e-7):
            raise ValueError("sidecar coordinate delta is inconsistent")
        if not np.allclose(old_fractional, parent[index].frac_coords, atol=1.0e-7):
            raise ValueError("sidecar old fractional coordinate does not match the parent")
        if not np.allclose(new_fractional, revised[index].frac_coords, atol=1.0e-7):
            raise ValueError("sidecar new fractional coordinate does not match the revised structure")
        changed = [
            atom_index
            for atom_index in range(len(parent))
            if not np.allclose(parent[atom_index].coords, revised[atom_index].coords, atol=1.0e-7)
        ]
        if changed != [index]:
            raise ValueError("manual revision must change only the declared atom")
        if not str(sidecar.get("reason", "")).strip():
            raise ValueError("manual revision reason must not be empty")

        revision_id = "revision-" + hashlib.sha256(
            (sidecar["parent_sha256"] + sidecar["output_sha256"] + str(index + 1)).encode()
        ).hexdigest()[:20]
        root = Path(output_root).resolve()
        target = root / revision_id
        if target.exists():
            raise FileExistsError(f"revision already imported: {revision_id}")
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".{revision_id}.tmp-{uuid4().hex}"
        staging.mkdir()
        try:
            shutil.copyfile(parent_path, staging / "parent-POSCAR")
            shutil.copyfile(revised_path, staging / "POSCAR")
            shutil.copyfile(sidecar_path, staging / "revision.sidecar.json")
            (staging / "structure.mson.json").write_text(
                json.dumps(revised, cls=MontyEncoder, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            manifest = {
                "schema": "llm-matgen-structure-revision",
                "version": 1,
                "revision_id": revision_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parent_file_sha256": sidecar["parent_sha256"],
                "revised_file_sha256": sidecar["output_sha256"],
                "parent_structure_hashes": versioned_structure_hashes(parent),
                "revised_structure_hashes": versioned_structure_hashes(revised),
                "parent_child_lineage": {
                    "parent_v2": versioned_structure_hashes(parent)["v2"],
                    "child_v2": versioned_structure_hashes(revised)["v2"],
                },
                "changed_atom_index_1based": index + 1,
                "reason": sidecar["reason"],
                "audit_sidecar": "revision.sidecar.json",
                "status": "formal_structure_revision",
                "scientific_approval": "required",
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ImportedRevision(revision_id, target, target / "manifest.json")
