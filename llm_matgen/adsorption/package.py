"""Atomic, auditable output packages for adsorption initial configurations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from monty.json import MontyEncoder
from pymatgen.io.vasp import Poscar

from llm_matgen.generators.adsorption import AdsorptionGenerationResult
from llm_matgen.utils.structure import versioned_structure_hashes


@dataclass(frozen=True)
class AdsorptionPackage:
    root: Path
    manifest_path: Path
    viewer_path: Path | None = None


def _write_mson(path: Path, value) -> str:
    text = json.dumps(value, cls=MontyEncoder, ensure_ascii=False, sort_keys=True, indent=2)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdsorptionPackageWriter:
    def write(
        self,
        result: AdsorptionGenerationResult,
        output_root: Path | str,
        *,
        run_id: str | None = None,
    ) -> AdsorptionPackage:
        if not result.generated or result.clean_slab is None or result.adsorbate is None:
            raise ValueError("adsorption result is incomplete or contains no accepted candidate")
        resolved_id = run_id or f"adsorption-{uuid4().hex[:12]}"
        if not resolved_id or resolved_id in {".", ".."} or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in resolved_id
        ):
            raise ValueError("run_id must be a safe relative directory name")
        root = Path(output_root).resolve()
        target = root / resolved_id
        if target.exists():
            raise FileExistsError(f"adsorption package already exists: {target}")
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".{resolved_id}.tmp-{uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        try:
            candidates_dir = staging / "candidates"
            references_dir = staging / "references"
            candidates_dir.mkdir()
            references_dir.mkdir()
            _write_mson(references_dir / "clean-slab.mson.json", result.clean_slab)
            _write_mson(references_dir / "adsorbate.mson.json", result.adsorbate)
            if result.gas_reference is None:
                raise ValueError("adsorption result has no gas reference")
            _write_mson(references_dir / "gas-reference.mson.json", result.gas_reference)
            candidates = []
            for index, generated in enumerate(result.generated, start=1):
                name = f"{index:04d}-{generated.record.structure_id[:12]}"
                candidate_dir = candidates_dir / name
                candidate_dir.mkdir()
                poscar = candidate_dir / "POSCAR"
                Poscar(generated.structure, sort_structure=False).write_file(poscar, direct=True)
                mson = candidate_dir / "structure.mson.json"
                mson_sha = _write_mson(mson, generated.structure)
                candidates.append(
                    {
                        "structure_id": generated.record.structure_id,
                        "hashes": versioned_structure_hashes(generated.structure),
                        "parent_structure_id": generated.record.parent_structure_id,
                        "poscar_path": poscar.relative_to(staging).as_posix(),
                        "poscar_sha256": hashlib.sha256(poscar.read_bytes()).hexdigest(),
                        "mson_path": mson.relative_to(staging).as_posix(),
                        "mson_sha256": mson_sha,
                        "actual_parameters": generated.record.actual_parameters,
                    }
                )
            manifest = {
                "schema": "llm-matgen-adsorption-package",
                "version": 1,
                "run_id": resolved_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "configuration_status": "initial_configuration",
                "hash_schema_versions": ["v1", "v2"],
                "runtime_versions": {
                    name: importlib.metadata.version(name)
                    for name in ("llm-matgen", "pymatgen", "ase", "scipy")
                },
                "clean_slab_hashes": versioned_structure_hashes(result.clean_slab),
                "retrieval_trace": result.retrieval_trace.model_dump(mode="json")
                if result.retrieval_trace is not None
                else None,
                "dft_handoff": result.dft_handoff.model_dump(mode="json")
                if result.dft_handoff is not None
                else None,
                "proposal_audit": result.proposal_audit.__dict__
                if result.proposal_audit is not None
                else None,
                "warnings": result.warnings,
                "candidates": candidates,
            }
            try:
                from llm_matgen.viewer import write_viewer
                preview = write_viewer(
                    [(f"候选 {index} · {item.record.formula}", item.structure)
                     for index, item in enumerate(result.generated, 1)]
                    + [("参考：干净表面", result.clean_slab)], staging / "viewer.html")
                manifest["artifacts"] = [{"format": "html", "path": "viewer.html",
                                          "sha256": preview.sha256, "role": "structure_viewer"}]
                manifest["viewer"] = "viewer.html"
            except Exception as exc:
                manifest["warnings"] = [*manifest["warnings"], f"structure viewer unavailable: {exc}"]
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return AdsorptionPackage(root=target, manifest_path=target / "manifest.json",
                                 viewer_path=target / "viewer.html" if "viewer" in manifest else None)
