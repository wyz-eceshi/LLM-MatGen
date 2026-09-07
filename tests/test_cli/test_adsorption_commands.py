from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.io.vasp import Poscar


def test_parser_exposes_cases_adsorption_and_revision_commands():
    from llm_matgen.__main__ import build_parser

    parser = build_parser()
    assert parser.parse_args(["cases", "status"]).cases_command == "status"
    assert parser.parse_args([
        "generate", "adsorption", "--slab", "POSCAR", "--adsorbate", "H.xyz", "--anchor", "1"
    ]).generator == "adsorption"
    assert parser.parse_args([
        "revision", "import", "--parent", "a", "--revised", "b", "--sidecar", "c"
    ]).revision_command == "import"
    fallback = parser.parse_args([
        "revision", "import", "--revised", "b", "--sidecar", "c"
    ])
    assert fallback.revision_command == "import"
    assert fallback.parent is None


def test_generate_adsorption_cli_writes_complete_package(tmp_path, monkeypatch, capsys):
    from llm_matgen.__main__ import main

    monkeypatch.chdir(tmp_path)
    slab_path = tmp_path / "POSCAR"
    molecule_path = tmp_path / "H.xyz"
    Poscar(
        Structure(Lattice.from_parameters(3, 3, 18, 90, 90, 90), ["Pt"], [[0, 0, 0.5]])
    ).write_file(slab_path)
    Molecule(["H"], [[0, 0, 0]]).to(filename=str(molecule_path))
    code = main([
        "generate", "adsorption",
        "--slab", str(slab_path),
        "--adsorbate", str(molecule_path),
        "--anchor", "1",
        "--history-policy", "off",
        "--site-type", "top",
        "--max-structures", "1",
        "--max-proposal-attempts", "2",
        "--output-root", "out",
        "--run-id", "cli-ads",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["generated"] == 1
    assert Path(payload["manifest"]).is_file()


def test_cases_status_uses_explicit_local_store(tmp_path, capsys):
    from llm_matgen.__main__ import main

    assert main(["cases", "status", "--store-root", str(tmp_path / "cases")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["index_revision"] == 0
    assert payload["schema_version"] >= 1
