from __future__ import annotations

import json
from pathlib import Path

import pytest
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup


def rocksalt_recipe(*, mode: str = "explicit") -> dict:
    recipe = {
        "schema": "llm-matgen-symmetry-crystal",
        "version": 1,
        "mode": mode,
        "space_group": {"number": 225, "hall_number": None},
        "cell": {"setting": "conventional", "parameters": [5.6, 5.6, 5.6, 90, 90, 90]},
        "composition": {"reduced": {"Na": 1, "Cl": 1}, "formula_units": 4, "formula_units_range": None},
        "constraints": {"symprec": 0.001, "pair_min_A": {"Na-Cl": 2.0}, "coordination": []},
    }
    if mode == "explicit":
        recipe["explicit"] = {"orbits": [
            {"element": "Na", "representative_fractional": [0, 0, 0], "expected_multiplicity": 4, "wyckoff": "4a"},
            {"element": "Cl", "representative_fractional": [0.5, 0.5, 0.5], "expected_multiplicity": 4, "wyckoff": "4b"},
        ]}
    else:
        recipe["search"] = {"seed": 19, "candidates": 5, "max_attempts": 20}
    return recipe


def test_explicit_recipe_expands_orbits_and_records_geometry_status():
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalGenerator, SymmetryCrystalParams

    result = SymmetryCrystalGenerator().generate(None, SymmetryCrystalParams(recipe=rocksalt_recipe()))
    assert result.generated_count == 1
    candidate = result.generated[0]
    assert candidate.structure.composition.as_dict() == {"Na": 4.0, "Cl": 4.0}
    assert SpacegroupAnalyzer(candidate.structure, symprec=1e-3).get_space_group_number() == 225
    assert candidate.record.actual_parameters["geometry_status"] == "passed"
    assert candidate.record.actual_parameters["relaxation_status"] == "unknown"
    assert candidate.record.actual_parameters["space_group_number"] == 225
    assert candidate.record.actual_parameters["rank"] == 1
    assert len(candidate.record.actual_parameters["centering_translations"]) == 3


def test_explicit_recipe_is_deterministic():
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalGenerator, SymmetryCrystalParams
    from llm_matgen.utils.structure import structure_sha256

    params = SymmetryCrystalParams(recipe=rocksalt_recipe())
    first = SymmetryCrystalGenerator().generate(None, params).generated[0].structure
    second = SymmetryCrystalGenerator().generate(None, params).generated[0].structure
    assert structure_sha256(first) == structure_sha256(second)


def test_shipped_yaml_explicit_recipe_loads_and_generates():
    from llm_matgen.generators.symmetry_crystal import (
        SymmetryCrystalGenerator,
        SymmetryCrystalParams,
        load_recipe_file,
    )

    recipe_path = Path(__file__).parents[1] / "examples" / "symmetry-crystal" / "rocksalt-explicit.yaml"
    recipe, digest = load_recipe_file(recipe_path)
    result = SymmetryCrystalGenerator().generate(
        None,
        SymmetryCrystalParams(
            recipe=recipe,
            recipe_source=str(recipe_path),
            recipe_sha256=digest,
        ),
    )
    assert result.generated_count == 1
    assert result.provenance.input_structure_hash == digest


def test_cubic_space_group_rejects_incompatible_lattice():
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalParams

    recipe = rocksalt_recipe()
    recipe["cell"]["parameters"][1] = 5.7
    with pytest.raises(ValueError, match="cubic"):
        SymmetryCrystalParams(recipe=recipe)


def test_nonstandard_monoclinic_axis_requires_matching_hall_number():
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalParams

    recipe = rocksalt_recipe(mode="search")
    recipe["space_group"] = {"number": 3, "hall_number": None}
    recipe["cell"] = {
        "setting": "conventional",
        "parameters": [5.1, 5.7, 6.2, 90, 90, 103],
        "unique_axis": "c",
    }
    with pytest.raises(ValueError, match="Hall"):
        SymmetryCrystalParams(recipe=recipe)

    recipe["space_group"]["hall_number"] = 4
    assert SymmetryCrystalParams(recipe=recipe).recipe.space_group.hall_number == 4


def test_rhombohedral_setting_requires_matching_hall_number():
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalParams

    recipe = rocksalt_recipe(mode="search")
    recipe["space_group"] = {"number": 166, "hall_number": None}
    recipe["cell"] = {
        "setting": "rhombohedral",
        "parameters": [5.1, 5.1, 5.1, 73, 73, 73],
    }
    with pytest.raises(ValueError, match="Hall"):
        SymmetryCrystalParams(recipe=recipe)

    recipe["space_group"]["hall_number"] = 459
    assert SymmetryCrystalParams(recipe=recipe).recipe.space_group.hall_number == 459


def test_explicit_recipe_rejects_wrong_orbit_multiplicity():
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalGenerator, SymmetryCrystalParams

    recipe = rocksalt_recipe()
    recipe["explicit"]["orbits"][0]["expected_multiplicity"] = 8
    with pytest.raises(ValueError, match="multiplicity"):
        SymmetryCrystalGenerator().generate(None, SymmetryCrystalParams(recipe=recipe))


def test_explicit_recipe_enforces_atom_limit():
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalGenerator, SymmetryCrystalParams

    with pytest.raises(ValueError, match="atom limit"):
        SymmetryCrystalGenerator().generate(
            None,
            SymmetryCrystalParams(recipe=rocksalt_recipe(), max_atoms_per_structure=7),
        )


def test_search_reports_optional_backend_install_command(monkeypatch):
    from llm_matgen.generators import symmetry_crystal
    from llm_matgen.generators.backends import OptionalDependencyError

    def unavailable():
        raise ImportError("missing")

    monkeypatch.setattr(symmetry_crystal, "_import_pyxtal", unavailable)
    with pytest.raises(OptionalDependencyError, match=r"llm-matgen\[symmetry\]"):
        symmetry_crystal.SymmetryCrystalGenerator().generate(
            None, symmetry_crystal.SymmetryCrystalParams(recipe=rocksalt_recipe(mode="search"))
        )


def test_search_attempt_limit_is_global_across_formula_unit_range(monkeypatch):
    from llm_matgen.generators import symmetry_crystal

    calls = []

    class FakeCrystal:
        valid = False

        def from_random(self, *args, **kwargs):
            calls.append((args, kwargs))

    class FakeLattice:
        @staticmethod
        def from_para(*args, **kwargs):
            return object()

    monkeypatch.setattr(symmetry_crystal, "_import_pyxtal", lambda: (FakeCrystal, FakeLattice))
    recipe = rocksalt_recipe(mode="search")
    recipe["composition"]["formula_units"] = None
    recipe["composition"]["formula_units_range"] = [1, 3]
    recipe["search"]["max_attempts"] = 2

    with pytest.raises(ValueError, match="no valid candidates"):
        symmetry_crystal.SymmetryCrystalGenerator().generate(
            None, symmetry_crystal.SymmetryCrystalParams(recipe=recipe)
        )
    assert len(calls) == 2


def test_real_pyxtal_search_smoke_when_optional_backend_is_installed():
    pytest.importorskip("pyxtal")
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalGenerator, SymmetryCrystalParams
    from llm_matgen.utils.structure import structure_sha256

    recipe = rocksalt_recipe(mode="search")
    recipe["composition"]["formula_units"] = None
    recipe["composition"]["formula_units_range"] = [1, 8]
    recipe["constraints"]["pair_min_A"] = {}
    recipe["search"].update({"candidates": 1, "max_attempts": 50})
    result = SymmetryCrystalGenerator().generate(
        None, SymmetryCrystalParams(recipe=recipe)
    )
    repeated = SymmetryCrystalGenerator().generate(
        None, SymmetryCrystalParams(recipe=recipe)
    )
    assert result.generated_count == 1
    metadata = result.generated[0].record.actual_parameters
    assert metadata["space_group_number"] == 225
    assert 1 <= metadata["search_statistics"]["attempts"] <= 50
    assert metadata["search_statistics"]["selected_formula_units"] == 4
    assert metadata["search_statistics"]["rejected"]["orbit_incompatible"] == 3
    assert metadata["backend"]["name"] == "pyxtal"
    assert metadata["backend"]["version"].startswith("1.")
    assert structure_sha256(result.generated[0].structure) == structure_sha256(
        repeated.generated[0].structure
    )


def test_generation_service_runs_source_free_symmetry_generator(tmp_path):
    from llm_matgen.services.generation import ExecutionLimits, GenerationRequest, GenerationService

    class RejectingSource:
        def get(self, reference):
            raise AssertionError("source-free generator must not resolve a parent structure")

    request = GenerationRequest(
        generator="symmetry-crystal", input_refs=[], parameters={"recipe": rocksalt_recipe()},
        limits=ExecutionLimits(output_root=tmp_path),
    )
    result = GenerationService(source=RejectingSource()).run(request)
    assert result.ok
    assert result.runs[0].generation.generated_count == 1
    manifest = json.loads(result.runs[0].manifest_path.read_text(encoding="utf-8"))
    assert manifest["parameters"]["recipe"]["space_group"]["number"] == 225


def test_cli_builds_source_free_recipe_request(tmp_path: Path, monkeypatch):
    from llm_matgen.__main__ import build_generation_request, build_parser

    monkeypatch.chdir(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(rocksalt_recipe()), encoding="utf-8")
    args = build_parser().parse_args([
        "generate", "symmetry-crystal", "--recipe", str(recipe_path),
        "--format", "cif", "--output-root", "runs",
    ])
    request = build_generation_request(args)
    assert request.generator == "symmetry-crystal"
    assert request.input_refs == []
    assert request.parameters["recipe"]["mode"] == "explicit"
    assert request.parameters["recipe_sha256"]
    assert request.export_options.formats[0].value == "cif"


def test_symmetry_crystal_cli_defaults_to_poscar_cif_and_mson(tmp_path: Path, monkeypatch):
    from llm_matgen.__main__ import build_generation_request, build_parser

    monkeypatch.chdir(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(rocksalt_recipe()), encoding="utf-8")
    args = build_parser().parse_args([
        "generate", "symmetry-crystal", "--recipe", str(recipe_path),
    ])
    request = build_generation_request(args)
    assert [fmt.value for fmt in request.export_options.formats] == ["poscar", "cif", "mson"]


def test_symmetry_crystal_cli_writes_three_formats_manifest_and_viewer(tmp_path: Path, monkeypatch):
    from llm_matgen.__main__ import main

    monkeypatch.chdir(tmp_path)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(rocksalt_recipe()), encoding="utf-8")
    assert main([
        "generate", "symmetry-crystal", "--recipe", str(recipe_path),
        "--output-root", "runs",
    ]) == 0

    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    structures = run_dir / "structures"
    assert len(list(structures.glob("*.vasp"))) == 1
    assert len(list(structures.glob("*.cif"))) == 1
    assert len(list(structures.glob("*.mson.json"))) == 1
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "viewer.html").is_file()


def compatible_cell(space_group_number: int) -> list[float]:
    system = str(SpaceGroup.from_int_number(space_group_number).crystal_system)
    return {
        "triclinic": [5.1, 5.7, 6.2, 77, 83, 71],
        "monoclinic": [5.1, 5.7, 6.2, 90, 103, 90],
        "orthorhombic": [5.1, 5.7, 6.2, 90, 90, 90],
        "tetragonal": [5.1, 5.1, 6.2, 90, 90, 90],
        "trigonal": [5.1, 5.1, 8.2, 90, 90, 120],
        "hexagonal": [5.1, 5.1, 8.2, 90, 90, 120],
        "cubic": [5.1, 5.1, 5.1, 90, 90, 90],
    }[system]


@pytest.mark.parametrize("number", range(1, 231))
def test_all_230_space_groups_resolve_with_compatible_cells(number):
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalParams, _symmetry_operations

    recipe = rocksalt_recipe(mode="search")
    recipe["space_group"] = {"number": number, "hall_number": None}
    recipe["cell"]["parameters"] = compatible_cell(number)
    parsed = SymmetryCrystalParams(recipe=recipe).recipe
    assert len(_symmetry_operations(parsed)) >= 1


@pytest.mark.parametrize("number", range(1, 231))
def test_all_230_space_groups_expand_general_orbit_and_roundtrip(number, tmp_path):
    from llm_matgen.generators.models import OutputFormat
    from llm_matgen.generators.symmetry_crystal import (
        SymmetryCrystalGenerator,
        SymmetryCrystalParams,
        _symmetry_operations,
        _unique_fractional,
    )
    from llm_matgen.io.exporters import ExportOptions, StructureExporter

    representatives = [
        [0.1234567, 0.2345678, 0.3456789],
        [0.1717171, 0.3191919, 0.4131313],
        [0.2718281, 0.1414213, 0.5623730],
    ]
    probe = rocksalt_recipe(mode="search")
    probe["space_group"] = {"number": number, "hall_number": None}
    probe["cell"]["parameters"] = compatible_cell(number)
    parsed_probe = SymmetryCrystalParams(recipe=probe).recipe
    multiplicities = [
        len(_unique_fractional([
            operation.operate(representative) for operation in _symmetry_operations(parsed_probe)
        ]))
        for representative in representatives
    ]
    recipe = {
        "schema": "llm-matgen-symmetry-crystal",
        "version": 1,
        "mode": "explicit",
        "space_group": {"number": number, "hall_number": None},
        "cell": {"setting": "conventional", "parameters": compatible_cell(number)},
        "composition": {
            "reduced": {"Si": 1},
            "formula_units": sum(multiplicities),
            "formula_units_range": None,
        },
        "constraints": {"symprec": 0.001, "pair_min_A": {}, "coordination": []},
        "explicit": {"orbits": [
            {
                "element": "Si",
                "representative_fractional": representative,
                "expected_multiplicity": multiplicity,
                "wyckoff": None,
            }
            for representative, multiplicity in zip(representatives, multiplicities, strict=True)
        ]},
    }
    candidate = SymmetryCrystalGenerator().generate(
        None, SymmetryCrystalParams(recipe=recipe)
    ).generated[0].structure
    artifacts = StructureExporter().export_structure(
        candidate,
        f"sg-{number}",
        ExportOptions(formats=[OutputFormat.POSCAR, OutputFormat.CIF], output_dir=tmp_path),
    ).artifacts
    assert len(candidate) == sum(multiplicities)
    assert len(artifacts) == 2


@pytest.mark.parametrize(
    "number,hall_number,expected_nonzero_translations",
    [
        (1, None, 0),
        (21, 120, 1),  # A centered nonstandard setting
        (21, 121, 1),  # B centered nonstandard setting
        (15, None, 1),
        (139, None, 1),
        (166, None, 2),
        (225, None, 3),
    ],
)
def test_centering_translations_are_derived_from_space_group(
    number, hall_number, expected_nonzero_translations
):
    from llm_matgen.generators.symmetry_crystal import SymmetryCrystalParams, _centering_translations

    recipe = rocksalt_recipe(mode="search")
    recipe["space_group"] = {"number": number, "hall_number": hall_number}
    recipe["cell"]["parameters"] = compatible_cell(number)
    parsed = SymmetryCrystalParams(recipe=recipe).recipe
    assert len(_centering_translations(parsed)) == expected_nonzero_translations
