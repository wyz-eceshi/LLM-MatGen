"""Deterministic generation, check, export, and manifest pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from uuid import uuid4

from monty.json import MontyEncoder
from pymatgen.core import Structure

from llm_matgen.checks.checker import LightStructureChecker
from llm_matgen.checks.models import CheckIssue, CheckReport
from llm_matgen.generators.models import GenerationResult, JsonValue, OutputFormat
from llm_matgen.io.exporters import ExportArtifact, ExportOptions, StructureExporter
from llm_matgen.io.manifest import (
    ManifestArtifact,
    ManifestStore,
    ManifestStructure,
    RunManifest,
)
from llm_matgen.utils.structure import versioned_structure_hashes


@dataclass
class PipelineResult:
    generation: GenerationResult
    check_reports: dict[str, CheckReport]
    artifacts: list[ExportArtifact]
    manifest_path: Path
    ok: bool
    errors: list[str] = field(default_factory=list)
    viewer_path: Path | None = None


class GenerationPipeline:
    def __init__(
        self,
        output_root: Path,
        *,
        checker: LightStructureChecker | None = None,
        exporter: StructureExporter | None = None,
        software_version: str = "0.2.1",
    ):
        self.output_root = Path(output_root).resolve()
        self.checker = checker or LightStructureChecker()
        self.exporter = exporter or StructureExporter()
        self.software_version = software_version

    def run(
        self,
        generator,
        structure,
        params,
        export_options: ExportOptions,
        *,
        run_id: str | None = None,
    ) -> PipelineResult:
        generation = generator.generate(structure, params)
        resolved_run_id = run_id or f"run-{uuid4().hex[:12]}"
        run_dir = self.output_root / resolved_run_id
        structures_dir = run_dir / "structures"
        structures_dir.mkdir(parents=True, exist_ok=False)
        formats = list(export_options.formats)
        if generation.defect_type == "adsorption":
            formats = list(dict.fromkeys([OutputFormat.POSCAR, OutputFormat.MSON, *formats]))
        options = export_options.model_copy(
            update={"output_dir": structures_dir, "formats": formats}
        )

        reports: dict[str, CheckReport] = {}
        artifacts: list[ExportArtifact] = []
        errors: list[str] = []
        manifest_structures: list[ManifestStructure] = []
        manifest_artifacts: list[ManifestArtifact] = []
        reference_artifacts: list[dict[str, JsonValue]] = []
        reference_ids: dict[str, str] = {}

        reference = structure if isinstance(structure, Structure) else getattr(structure, "substrate", None)
        for generated in generation.generated:
            record = generated.record
            report = self.checker.check(generated.structure, reference=reference)
            reports[record.structure_id] = report
            manifest_structures.append(
                ManifestStructure(
                    structure_id=record.structure_id,
                    parent_structure_id=record.parent_structure_id,
                    parent_structure_ids=record.parent_structure_ids,
                    formula=record.formula,
                    n_atoms=record.n_atoms,
                    actual_parameters=record.actual_parameters,
                    site_mapping=record.site_mapping,
                    site_lineage=record.site_lineage,
                    identity_hashes=versioned_structure_hashes(generated.structure),
                    identity_version="v2",
                    configuration_status=record.actual_parameters.get("configuration_status"),
                    check_issues=[issue.model_dump(mode="json") for issue in report.issues],
                )
            )
            if not report.can_export:
                errors.append(f"{record.structure_id}: lightweight check contains errors")
                continue
            try:
                exported = self.exporter.export_structure(
                    generated.structure,
                    record.structure_id,
                    options,
                )
            except Exception as exc:
                errors.append(f"{record.structure_id}: export failed: {exc}")
                continue
            artifacts.extend(exported.artifacts)
            for artifact in exported.artifacts:
                manifest_artifacts.append(
                    ManifestArtifact(
                        structure_id=artifact.structure_id,
                        format=artifact.format.value,
                        path=artifact.path.relative_to(run_dir).as_posix(),
                        sha256=artifact.sha256,
                        metadata=artifact.metadata,
                    )
                )

        if generation.defect_type == "adsorption":
            references_dir = run_dir / "references"
            references_dir.mkdir(exist_ok=True)
            for role, value in (
                ("clean_slab", getattr(generation, "clean_slab", None)),
                ("adsorbate", getattr(generation, "adsorbate", None)),
                ("gas_reference", getattr(generation, "gas_reference", None)),
            ):
                if value is None:
                    errors.append(f"adsorption reference is missing: {role}")
                    continue
                path = references_dir / f"{role.replace('_', '-')}.mson.json"
                temporary = path.with_suffix(path.suffix + ".tmp")
                try:
                    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                        json.dump(value, handle, cls=MontyEncoder, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    reference_ids[role] = (
                        versioned_structure_hashes(value)["v2"]
                        if isinstance(value, Structure)
                        else digest
                    )
                    reference_artifacts.append(
                        {
                            "role": role,
                            "path": path.relative_to(run_dir).as_posix(),
                            "sha256": digest,
                            "format": "mson",
                        }
                    )
                except Exception as exc:
                    temporary.unlink(missing_ok=True)
                    errors.append(f"adsorption reference export failed for {role}: {exc}")

        viewer_path = None
        exported_ids = {artifact.structure_id for artifact in artifacts}
        preview_structures = [(f"候选 {index} · {item.record.formula}", item.structure)
                              for index, item in enumerate(generation.generated, 1)
                              if item.record.structure_id in exported_ids]
        if preview_structures:
            try:
                from llm_matgen.viewer import write_viewer
                preview = write_viewer(preview_structures, run_dir / "viewer.html")
                viewer_path = preview.path
                manifest_artifacts.append(ManifestArtifact(
                    structure_id="run-viewer", format="html", path="viewer.html",
                    sha256=preview.sha256, metadata={"role": "structure_viewer"}))
            except Exception as exc:
                generation.warnings.append(f"structure viewer unavailable: {exc}")
        provenance = generation.provenance
        manifest = RunManifest(
            run_id=resolved_run_id,
            created_at=datetime.now(timezone.utc),
            software_version=self.software_version,
            input_source=provenance.input_source if provenance else "unknown",
            parameters=provenance.parameters if provenance else {},
            structures=manifest_structures,
            artifacts=manifest_artifacts,
            warnings=[*generation.warnings, *errors],
            reference_ids=reference_ids,
            reference_artifacts=reference_artifacts,
            runtime_versions=(
                {
                    name: importlib.metadata.version(name)
                    for name in ("llm-matgen", "pymatgen", "ase", "scipy")
                }
                if generation.defect_type == "adsorption"
                else {}
            ),
            retrieval_trace=(
                generation.retrieval_trace.model_dump(mode="json")
                if getattr(generation, "retrieval_trace", None) is not None
                else None
            ),
            dft_handoff=(
                generation.dft_handoff.model_dump(mode="json")
                if getattr(generation, "dft_handoff", None) is not None
                else None
            ),
        )
        manifest_path = ManifestStore(self.output_root).write_atomic(
            manifest,
            allow_existing_dir=True,
        )
        return PipelineResult(
            generation=generation,
            check_reports=reports,
            artifacts=artifacts,
            manifest_path=manifest_path,
            ok=not errors and len(artifacts) > 0,
            errors=errors,
            viewer_path=viewer_path,
        )
