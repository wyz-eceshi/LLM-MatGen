import pytest
from pymatgen.core import Lattice, Structure
from llm_matgen.generators.models import OutputFormat
from llm_matgen.io.exporters import ExportOptions
from llm_matgen.pipeline import GenerationPipeline


def silicon_structure() -> Structure:
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def test_surface_generates_slab_with_vacuum_without_mutating_input():
    from llm_matgen.generators.surface import SurfaceGenerator, SurfaceParams

    source = silicon_structure()
    original = source.copy()
    result = SurfaceGenerator().generate(
        source,
        SurfaceParams(
            miller_indices=[(0, 0, 1)],
            min_slab_size=6.0,
            min_vacuum_size=8.0,
        ),
    )
    assert result.generated_count >= 1
    slab = result.generated[0]
    assert slab.structure.lattice.c > 14.0
    assert slab.record.parent_structure_id == result.provenance.input_structure_hash
    assert slab.record.actual_parameters["miller_index"] == [0, 0, 1]
    assert source == original

    metadata = slab.record.actual_parameters
    assert metadata["termination_fingerprint"].startswith("term-v1-")
    assert len(metadata["surface_normal"]) == 3
    assert metadata["normal_cell_repeat"] == pytest.approx(
        metadata["material_span"] + metadata["vacuum_estimate"]
    )
    assert set(metadata["surface_composition"]) == {"top", "bottom"}
    assert "dipole_risk" in metadata
    assert metadata["bulk_to_slab_site_mapping"]


def test_surface_normalizes_duplicate_miller_indices():
    from llm_matgen.generators.surface import SurfaceGenerator, SurfaceParams

    result = SurfaceGenerator().generate(
        silicon_structure(),
        SurfaceParams(
            miller_indices=[(0, 0, 1), (0, 0, 2)],
            min_slab_size=5.0,
            min_vacuum_size=5.0,
        ),
    )
    assert {tuple(item.record.actual_parameters["miller_index"]) for item in result.generated} == {(0, 0, 1)}


def test_surface_enforces_atom_limit():
    from llm_matgen.generators.surface import SurfaceGenerator, SurfaceParams

    with pytest.raises(ValueError, match="atom limit"):
        SurfaceGenerator().generate(
            silicon_structure(),
            SurfaceParams(
                miller_indices=[(0, 0, 1)],
                min_slab_size=20.0,
                min_vacuum_size=5.0,
                max_atoms_per_structure=1,
            ),
        )


def test_surface_deduplicates_symmetry_equivalent_cubic_planes():
    from llm_matgen.generators.surface import SurfaceGenerator, SurfaceParams

    result = SurfaceGenerator().generate(
        silicon_structure(),
        SurfaceParams(
            miller_indices=[(1, 0, 0), (0, 1, 0)],
            min_slab_size=5.0,
            min_vacuum_size=5.0,
        ),
    )
    assert result.generated_count == 1
    assert any("duplicate" in warning for warning in result.warnings)


def test_surface_pipeline_exports_all_formats(tmp_path):
    from llm_matgen.generators.surface import SurfaceGenerator, SurfaceParams

    result = GenerationPipeline(tmp_path).run(
        SurfaceGenerator(),
        silicon_structure(),
        SurfaceParams(
            miller_indices=[(0, 0, 1)],
            min_slab_size=5.0,
            min_vacuum_size=5.0,
        ),
        ExportOptions(formats=list(OutputFormat)),
        run_id="surface",
    )
    assert result.ok
    assert {artifact.format for artifact in result.artifacts} == set(OutputFormat)


def test_surface_is_public_generator_api():
    from llm_matgen.generators import SurfaceGenerator, SurfaceParams

    assert SurfaceGenerator is not None
    assert SurfaceParams is not None


def test_surface_freezes_only_requested_true_normal_bottom_layer():
    from llm_matgen.generators.surface import SurfaceGenerator, SurfaceParams

    generated = SurfaceGenerator().generate(
        silicon_structure(),
        SurfaceParams(
            miller_indices=[(0, 0, 1)],
            min_slab_size=8.0,
            min_vacuum_size=8.0,
            freeze_bottom_layers=1,
        ),
    ).generated[0]
    frozen = generated.record.actual_parameters["frozen_layer_atom_indices"]
    flags = generated.structure.site_properties["selective_dynamics"]
    assert len(frozen) == 1
    assert all(flags[index] == [False, False, False] for index in frozen[0])
    assert all(
        flags[index] == [True, True, True]
        for index in range(len(flags)) if index not in frozen[0]
    )
    assert generated.record.site_lineage
