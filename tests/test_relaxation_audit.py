from __future__ import annotations

import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar


FIXTURES = Path(__file__).parent / "fixtures" / "relaxation_audit"


def write_pair(tmp_path, initial_a: float, final_a: float):
    species = ["Na"] * 4 + ["Cl"] * 4
    coords = [
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
        [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
    ]
    initial = tmp_path / "POSCAR"
    final = tmp_path / "CONTCAR"
    Poscar(Structure(Lattice.cubic(initial_a), species, coords)).write_file(initial)
    Poscar(Structure(Lattice.cubic(final_a), species, coords)).write_file(final)
    return initial, final


def test_relaxation_audit_flags_runaway_cell_without_symmetry_loss(tmp_path):
    from llm_matgen.structure import audit_relaxation

    initial, final = write_pair(tmp_path, 10.0, 15.0)
    report = audit_relaxation(initial, final, label="failed", reason="cell expanded")
    assert report["label"] == "failed"
    assert report["metrics"]["lattice_change_percent"] == pytest.approx([50, 50, 50])
    assert report["metrics"]["volume_change_percent"] == pytest.approx(237.5)
    assert "runaway_cell" in report["anomalies"]
    assert "network_dilution" in report["anomalies"]
    assert "symmetry_changed" not in report["anomalies"]


def test_relaxation_audit_accepts_uniform_five_percent_contraction(tmp_path):
    from llm_matgen.structure import audit_relaxation

    initial, final = write_pair(tmp_path, 10.0, 9.5)
    report = audit_relaxation(initial, final, label="passed", reason="usable relaxed cell")
    assert "large_cell_change" not in report["anomalies"]
    assert "runaway_cell" not in report["anomalies"]
    assert report["metrics"]["nonaffine_rms_displacement_A"] == pytest.approx(0.0, abs=1e-10)


def test_relaxation_audit_handles_single_atom_periodic_structure(tmp_path):
    from llm_matgen.structure import audit_relaxation

    initial = tmp_path / "single-POSCAR"
    final = tmp_path / "single-CONTCAR"
    Poscar(Structure(Lattice.cubic(4.0), ["Si"], [[0.2, 0.3, 0.4]])).write_file(initial)
    Poscar(Structure(Lattice.cubic(4.1), ["Si"], [[0.2, 0.3, 0.4]])).write_file(final)
    report = audit_relaxation(initial, final, label="passed", reason="single atom smoke")

    assert report["initial"]["pair_min_A"]["Si-Si"] == pytest.approx(4.0)
    assert report["final"]["pair_min_A"]["Si-Si"] == pytest.approx(4.1)
    assert report["metrics"]["pair_distance_ratios"]["Si-Si"] == pytest.approx(1.025)


def test_relaxation_audit_rejects_blank_reason_and_composition_change(tmp_path):
    from llm_matgen.structure import audit_relaxation

    initial, final = write_pair(tmp_path, 10.0, 10.0)
    with pytest.raises(ValueError, match="reason"):
        audit_relaxation(initial, final, label="passed", reason="   ")
    changed = Structure.from_file(final)
    changed.replace(0, "K")
    Poscar(changed).write_file(final)
    with pytest.raises(ValueError, match="composition"):
        audit_relaxation(initial, final, label="failed", reason="changed")


def test_relaxation_audit_cli_writes_new_json(tmp_path):
    from llm_matgen.__main__ import main

    initial, final = write_pair(tmp_path, 10.0, 15.0)
    output = tmp_path / "audit.json"
    code = main([
        "structure", "relaxation-audit", "--initial", str(initial), "--final", str(final),
        "--label", "failed", "--reason", "runaway expansion", "--output", str(output),
    ])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "llm-matgen-relaxation-audit"
    assert payload["label"] == "failed"
    assert "runaway_cell" in payload["anomalies"]


def test_relaxation_audit_refuses_to_overwrite(tmp_path):
    from llm_matgen.__main__ import main

    initial, final = write_pair(tmp_path, 10.0, 15.0)
    output = tmp_path / "audit.json"
    output.write_text("existing", encoding="utf-8")
    code = main([
        "structure", "relaxation-audit", "--initial", str(initial), "--final", str(final),
        "--label", "failed", "--reason", "runaway expansion", "--output", str(output),
    ])
    assert code != 0
    assert output.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize(
    ("stem", "label", "space_group", "axis_change", "must_include", "must_exclude"),
    [
        (
            "success_f43m",
            "passed",
            216,
            -5.643,
            set(),
            {"large_cell_change", "runaway_cell", "symmetry_changed"},
        ),
        (
            "success_fd3m",
            "passed",
            227,
            -4.988,
            set(),
            {"large_cell_change", "runaway_cell", "symmetry_changed"},
        ),
        (
            "failed_fd3m",
            "failed",
            227,
            55.227,
            {"large_cell_change", "runaway_cell", "network_dilution"},
            {"symmetry_changed"},
        ),
        (
            "failed_f43m",
            "failed",
            216,
            101.925,
            {"large_cell_change", "runaway_cell", "network_dilution"},
            {"symmetry_changed"},
        ),
    ],
)
def test_mg_nd_o_regression_detects_cell_runaway_even_when_space_group_is_retained(
    stem,
    label,
    space_group,
    axis_change,
    must_include,
    must_exclude,
):
    from llm_matgen.structure import audit_relaxation

    report = audit_relaxation(
        FIXTURES / f"{stem}_initial.vasp",
        FIXTURES / f"{stem}_final.vasp",
        label=label,
        reason="人工确认的 Mg-Nd-O 优化结果",
    )

    assert report["label"] == label
    assert report["initial"]["space_group"]["number"] == space_group
    assert report["final"]["space_group"]["number"] == space_group
    assert report["metrics"]["lattice_change_percent"][0] == pytest.approx(axis_change, abs=0.002)
    assert must_include <= set(report["anomalies"])
    assert must_exclude.isdisjoint(report["anomalies"])
