from __future__ import annotations

import json

from monty.json import MontyDecoder
from pymatgen.core import Lattice, Molecule, Structure


def test_adsorption_package_roundtrips_mson_and_writes_complete_handoff(tmp_path):
    from llm_matgen.adsorption.package import AdsorptionPackageWriter
    from llm_matgen.generators.adsorption import (
        AdsorptionGenerator,
        AdsorptionInput,
        AdsorptionParams,
    )

    slab = Structure(
        Lattice([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 18.0]]),
        ["Pt"],
        [[0.0, 0.0, 0.5]],
        site_properties={"selective_dynamics": [[False, False, False]]},
    )
    adsorbate = Molecule(["O"], [[0.0, 0.0, 0.0]], charge=0, spin_multiplicity=3)
    result = AdsorptionGenerator().generate(
        AdsorptionInput(
            clean_slab=slab,
            adsorbate=adsorbate,
            anchor_index=1,
            spin_multiplicity=3,
        ),
        AdsorptionParams(
            history_policy="off",
            site_types=("top",),
            max_structures=1,
            max_proposal_attempts=2,
            min_vacuum_each_side=3.0,
        ),
    )

    package = AdsorptionPackageWriter().write(result, tmp_path, run_id="ads-test")
    manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    assert package.viewer_path.is_file()
    assert manifest["viewer"] == "viewer.html"
    assert any(a["format"] == "html" for a in manifest["artifacts"])
    assert package.manifest_path == tmp_path / "ads-test" / "manifest.json"
    assert manifest["configuration_status"] == "initial_configuration"
    assert manifest["hash_schema_versions"] == ["v1", "v2"]
    assert set(manifest["runtime_versions"]) == {"llm-matgen", "pymatgen", "ase", "scipy"}
    assert manifest["retrieval_trace"]["fallback_reason"] == "history_disabled"
    assert manifest["dft_handoff"]["comparison_groups"] == [
        "adsorbed", "clean_slab", "gas_reference"
    ]
    candidate = manifest["candidates"][0]
    restored = MontyDecoder().process_decoded(
        json.loads((tmp_path / "ads-test" / candidate["mson_path"]).read_text(encoding="utf-8"))
    )
    assert restored.site_properties["selective_dynamics"][0] == [False, False, False]
    assert len(restored) == len(slab) + len(adsorbate)
    assert (tmp_path / "ads-test" / candidate["poscar_path"]).is_file()
    assert (tmp_path / "ads-test" / "references" / "clean-slab.mson.json").is_file()
    assert (tmp_path / "ads-test" / "references" / "adsorbate.mson.json").is_file()
    assert (tmp_path / "ads-test" / "references" / "gas-reference.mson.json").is_file()
