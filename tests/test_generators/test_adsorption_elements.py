"""Non-transition substrates must reach the public adsorption handoff."""
import pytest
from pymatgen.core import Lattice, Molecule, Structure

from llm_matgen.generators.adsorption import AdsorptionGenerator, AdsorptionInput, AdsorptionParams


@pytest.mark.parametrize(("element", "height", "plus_u"), [
    ("Li", 1.8, False), ("O", 1.1, False), ("Ce", 1.8, True), ("U", 1.8, True),
])
def test_h_adsorption_classifies_non_transition_elements(element, height, plus_u):
    slab = Structure(Lattice.orthorhombic(3, 3, 20), [element], [[0, 0, .5]])
    result = AdsorptionGenerator().generate(
        AdsorptionInput(clean_slab=slab, adsorbate=Molecule(["H"], [[0, 0, 0]]), spin_multiplicity=2),
        AdsorptionParams(history_policy="off", site_types=("top",), heights=(height,),
                         max_structures=1, max_proposal_attempts=2),
    )
    assert result.dft_handoff.plus_u_sensitive is plus_u
    assert len(result.generated) == 1
    assert len(result.generated[0].structure) == 2
