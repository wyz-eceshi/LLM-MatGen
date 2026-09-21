def test_package_exposes_version():
    import llm_matgen

    assert llm_matgen.__version__ == "0.3.0"


def test_public_subpackages_export_core_types():
    from llm_matgen.checks import CheckReport, LightStructureChecker
    from llm_matgen.generators import GenerationResult, OutputFormat
    from llm_matgen.io import ExportOptions, StructureExporter

    assert OutputFormat.POSCAR.value == "poscar"
    assert GenerationResult(defect_type="test").generated_count == 0
    assert CheckReport(n_atoms=0, formula="").can_export is True
    assert LightStructureChecker is not None
    assert ExportOptions().formats
    assert StructureExporter is not None


def test_full_extra_installs_every_runtime_optional_feature():
    from pathlib import Path
    import tomllib

    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    extras = project["optional-dependencies"]

    assert "pyxtal>=1.1,<2" in extras["symmetry"]
    assert set(extras["sqs"]) <= set(extras["full"])
    assert set(extras["symmetry"]) <= set(extras["full"])
    assert set(extras["all-llm"]) <= set(extras["full"])
