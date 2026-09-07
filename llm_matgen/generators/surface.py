"""Surface slab generation using pymatgen's local crystallographic tools."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import numpy as np
from pydantic import Field, PositiveFloat, PositiveInt, field_validator
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from llm_matgen.generators.extended import MillerIndex, normalize_miller
from llm_matgen.generators.models import (
    BaseGenerationParams,
    GeneratedStructure,
    GenerationResult,
    Provenance,
    StructureRecord,
)
from llm_matgen.utils.structure import assign_site_ids, structure_sha256


class SurfaceParams(BaseGenerationParams):
    miller_indices: list[MillerIndex] = Field(min_length=1)
    min_slab_size: PositiveFloat
    min_vacuum_size: PositiveFloat
    center_slab: bool = True
    primitive: bool = True
    max_normal_search: PositiveInt | None = None
    freeze_bottom_layers: int = Field(default=0, ge=0)
    layer_tolerance: PositiveFloat = 0.35

    @field_validator("miller_indices")
    @classmethod
    def validate_millers(cls, value: list[MillerIndex]) -> list[MillerIndex]:
        return list(dict.fromkeys(normalize_miller(index) for index in value))


class SurfaceGenerator:
    defect_name = "surface"
    generator_version = "0.1.0"

    @staticmethod
    def _freeze_layers(slab, count: int, tolerance: float):
        if count == 0:
            return slab, ()
        lattice = np.asarray(slab.lattice.matrix, dtype=float)
        normal = np.cross(lattice[0], lattice[1])
        normal /= np.linalg.norm(normal)
        if float(np.dot(normal, lattice[2])) < 0.0:
            normal *= -1.0
        ordered = sorted(
            (float(np.dot(site.coords, normal)), index)
            for index, site in enumerate(slab)
        )
        layers = [[ordered[0][1]]]
        previous = ordered[0][0]
        for projection, index in ordered[1:]:
            if projection - previous > tolerance:
                layers.append([])
            layers[-1].append(index)
            previous = projection
        selected = tuple(tuple(sorted(layer)) for layer in layers[:count])
        fixed = {index for layer in selected for index in layer}
        result = slab.copy()
        existing = result.site_properties.get("selective_dynamics")
        flags = [
            [False, False, False]
            if index in fixed
            else ([bool(item) for item in existing[index]] if existing is not None else [True, True, True])
            for index in range(len(result))
        ]
        if existing is not None:
            result.remove_site_property("selective_dynamics")
        result.add_site_property("selective_dynamics", flags)
        return result, selected

    @staticmethod
    def _metadata(source: Structure, slab, miller, parent_sites, child_id):
        lattice = np.asarray(slab.lattice.matrix, dtype=float)
        normal = np.cross(lattice[0], lattice[1])
        normal = normal / np.linalg.norm(normal)
        if float(np.dot(normal, lattice[2])) < 0.0:
            normal *= -1.0
        projections = np.asarray(slab.cart_coords, dtype=float) @ normal
        material_span = float(np.max(projections) - np.min(projections))
        repeat = abs(float(np.dot(lattice[2], normal)))
        layer_tolerance = 0.35
        top_indices = [
            index for index, value in enumerate(projections)
            if float(np.max(projections) - value) <= layer_tolerance
        ]
        bottom_indices = [
            index for index, value in enumerate(projections)
            if float(value - np.min(projections)) <= layer_tolerance
        ]

        def composition(indices):
            counts = {}
            for index in indices:
                symbol = str(slab[index].specie)
                counts[symbol] = counts.get(symbol, 0) + 1
            return {key: counts[key] for key in sorted(counts)}

        surface_composition = {
            "top": composition(top_indices),
            "bottom": composition(bottom_indices),
        }
        termination_payload = {
            "miller": list(miller),
            "surface_composition": surface_composition,
            "top_fractional": sorted(
                [
                    [str(slab[index].specie), *np.mod(slab[index].frac_coords[:2], 1.0).round(8).tolist()]
                    for index in top_indices
                ]
            ),
            "bottom_fractional": sorted(
                [
                    [str(slab[index].specie), *np.mod(slab[index].frac_coords[:2], 1.0).round(8).tolist()]
                    for index in bottom_indices
                ]
            ),
        }
        digest = sha256(
            json.dumps(termination_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        child_sites = assign_site_ids(slab, child_id)
        try:
            groups = SpacegroupAnalyzer(source).get_symmetrized_structure().equivalent_indices
        except Exception:
            groups = [[index] for index in range(len(source))]
        parent_to_class = {
            parent_index: class_index
            for class_index, group in enumerate(groups)
            for parent_index in group
        }
        bulk_equivalent = slab.site_properties.get("bulk_equivalent", [])
        mapping = {}
        for parent_index, parent_site_id in enumerate(parent_sites):
            class_index = parent_to_class.get(parent_index, parent_index)
            mapping[parent_site_id] = [
                child_sites[index]
                for index, equivalent in enumerate(bulk_equivalent)
                if int(equivalent) == class_index
            ]
        is_symmetric = bool(slab.is_symmetric())
        is_polar = bool(slab.is_polar())
        return {
            "surface_normal": [float(item) for item in normal],
            "normal_cell_repeat": repeat,
            "material_span": material_span,
            "vacuum_estimate": max(0.0, repeat - material_span),
            "surface_composition": surface_composition,
            "is_symmetric": is_symmetric,
            "is_polar": is_polar,
            "dipole_risk": is_polar or not is_symmetric or surface_composition["top"] != surface_composition["bottom"],
            "termination_fingerprint": f"term-v1-{digest}",
            "bulk_to_slab_site_mapping": mapping,
            "mapping_semantics": "parent symmetry class to generated slab site IDs",
        }

    def generate(self, structure: Structure, params: SurfaceParams) -> GenerationResult:
        source = structure.copy()
        parent_id = structure_sha256(source)
        parent_sites = assign_site_ids(source, parent_id)
        generated: list[GeneratedStructure] = []
        warnings: list[str] = []
        matcher = StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=False)

        for miller in params.miller_indices:
            builder = SlabGenerator(
                source,
                miller,
                float(params.min_slab_size),
                float(params.min_vacuum_size),
                center_slab=params.center_slab,
                primitive=params.primitive,
                max_normal_search=params.max_normal_search,
            )
            slabs = builder.get_slabs(symmetrize=False)
            if not slabs:
                warnings.append(f"no slab generated for Miller index {miller}")
                continue
            for termination, slab in enumerate(slabs):
                if len(slab) > params.max_atoms_per_structure:
                    raise ValueError(
                        f"surface atom limit exceeded: {len(slab)} > {params.max_atoms_per_structure}"
                    )
                slab, frozen_layers = self._freeze_layers(
                    slab, params.freeze_bottom_layers, float(params.layer_tolerance)
                )
                child_id = structure_sha256(slab)
                if any(matcher.fit(slab, item.structure) for item in generated):
                    warnings.append(
                        f"duplicate surface skipped for Miller index {miller}, termination {termination}"
                    )
                    continue
                metadata = self._metadata(source, slab, miller, parent_sites, child_id)
                generated.append(
                    GeneratedStructure(
                        structure=slab,
                        record=StructureRecord(
                            structure_id=child_id,
                            parent_structure_id=parent_id,
                            formula=slab.composition.reduced_formula,
                            n_atoms=len(slab),
                            actual_parameters={
                                "miller_index": list(miller),
                                "termination": metadata["termination_fingerprint"],
                                "termination_index": termination,
                                "cell_height": float(slab.lattice.c),
                                **metadata,
                                "freeze_bottom_layers": params.freeze_bottom_layers,
                                "frozen_layer_atom_indices": [list(layer) for layer in frozen_layers],
                            },
                            site_mapping={
                                site_id: (metadata["bulk_to_slab_site_mapping"][site_id] or [None])[0]
                                for site_id in parent_sites
                            },
                            site_lineage=metadata["bulk_to_slab_site_mapping"],
                        ),
                    )
                )
                if len(generated) >= params.max_structures:
                    warnings.append("surface variants truncated by max_structures")
                    break
            if len(generated) >= params.max_structures:
                break

        if not generated:
            raise ValueError("surface generation produced no structures")
        return GenerationResult(
            defect_type=self.defect_name,
            input_count=1,
            skipped_count=0,
            generated=generated,
            warnings=warnings,
            provenance=Provenance(
                generator=self.defect_name,
                generator_version=self.generator_version,
                input_source="in-memory",
                input_structure_hash=parent_id,
                parameters=params.model_dump(mode="json"),
                seed=None,
                created_at=datetime.now(timezone.utc),
            ),
        )
