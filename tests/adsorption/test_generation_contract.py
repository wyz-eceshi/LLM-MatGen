from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from pymatgen.core import Lattice, Molecule, Structure


def tilted_slab() -> Structure:
    return Structure(
        Lattice([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [18.0, 0.0, 0.0]]),
        ["Cu", "Cu", "Cu", "Cu"],
        [[0.50, 0.00, 0.00], [0.50, 0.50, 0.00], [0.50, 0.00, 0.50], [0.50, 0.50, 0.50]],
    )


def oh() -> Molecule:
    return Molecule(
        ["O", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
        charge=0,
        spin_multiplicity=2,
    )


def test_adsorption_params_defaults_are_bounded_and_history_preferred():
    from llm_matgen.generators.adsorption import AdsorptionParams

    params = AdsorptionParams()

    assert params.history_policy == "prefer"
    assert params.slab_state == "relaxed"
    assert params.surface_side == "top"
    assert params.max_structures == 1000
    assert params.max_proposal_attempts >= params.max_structures
    assert params.retrieval_top_k <= 5


def test_adsorption_params_reject_nonfinite_explicit_cartesian_sites():
    from llm_matgen.generators.adsorption import AdsorptionParams

    with pytest.raises(ValidationError, match="explicit"):
        AdsorptionParams(explicit_sites=((float("nan"), 0.0, 0.0),))


def test_adsorption_input_converts_user_anchor_once_and_checks_charge_spin():
    from llm_matgen.generators.adsorption import AdsorptionInput

    value = AdsorptionInput(
        clean_slab=tilted_slab(),
        adsorbate=oh(),
        anchor_index=1,
        charge=0,
        spin_multiplicity=2,
        reference_axis=(0.0, 0.0, 1.0),
    )

    assert value.anchor_index == 1
    assert value.anchor_zero_based == 0
    assert value.charge == 0
    assert value.spin_multiplicity == 2


def test_adsorption_input_rejects_bad_anchor_zero_reference_axis_and_extra_fields():
    from llm_matgen.generators.adsorption import AdsorptionInput

    common = dict(
        clean_slab=tilted_slab(),
        adsorbate=oh(),
        charge=0,
        spin_multiplicity=2,
    )
    with pytest.raises(ValidationError, match="anchor"):
        AdsorptionInput(**common, anchor_index=0, reference_axis=(0.0, 0.0, 1.0))
    with pytest.raises(ValidationError, match="reference axis"):
        AdsorptionInput(**common, anchor_index=1, reference_axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValidationError, match="Extra inputs"):
        AdsorptionInput(
            **common,
            anchor_index=1,
            reference_axis=(0.0, 0.0, 1.0),
            unsupported=True,
        )


def test_adsorption_input_rejects_disconnected_or_multidentate_contracts():
    from llm_matgen.generators.adsorption import AdsorptionInput

    disconnected = Molecule(
        ["O", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        charge=0,
        spin_multiplicity=2,
    )
    with pytest.raises(ValidationError, match="connected"):
        AdsorptionInput(
            clean_slab=tilted_slab(),
            adsorbate=disconnected,
            anchor_index=1,
            charge=0,
            spin_multiplicity=2,
            reference_axis=(0.0, 0.0, 1.0),
        )
    with pytest.raises(ValidationError, match="single-anchor"):
        AdsorptionInput(
            clean_slab=tilted_slab(),
            adsorbate=oh(),
            anchor_index=1,
            charge=0,
            spin_multiplicity=2,
            reference_axis=(0.0, 0.0, 1.0),
            denticity=2,
        )


def test_true_surface_frame_is_right_handed_and_bottom_reverses_normal():
    from llm_matgen.adsorption.proposals import build_surface_frame

    top = build_surface_frame(tilted_slab(), "top")
    bottom = build_surface_frame(tilted_slab(), "bottom")

    assert np.allclose(top.normal, [1.0, 0.0, 0.0])
    assert np.allclose(bottom.normal, [-1.0, 0.0, 0.0])
    assert np.allclose(np.cross(top.tangent1, top.tangent2), top.normal)
    assert np.isclose(np.dot(top.normal, top.tangent1), 0.0)
    assert np.isclose(np.dot(top.normal, top.tangent2), 0.0)


def test_rigid_oh_placement_uses_requested_anchor_and_preserves_roll_and_bond():
    from llm_matgen.adsorption.proposals import LocalFrame, place_adsorbate_rigid

    frame = LocalFrame(
        origin=(2.0, 3.0, 4.0),
        tangent1=(0.0, 1.0, 0.0),
        tangent2=(0.0, 0.0, 1.0),
        normal=(1.0, 0.0, 0.0),
    )
    molecule = oh()
    placed, transform = place_adsorbate_rigid(
        molecule,
        anchor_zero_based=0,
        reference_axis=(0.0, 0.0, 1.0),
        frame=frame,
        height=1.8,
        azimuth=90.0,
        tilt=0.0,
        roll=35.0,
    )

    assert np.allclose(placed.cart_coords[0], [3.8, 3.0, 4.0])
    assert np.isclose(placed.get_distance(0, 1), molecule.get_distance(0, 1))
    assert transform.roll == 35.0
    assert np.isclose(np.linalg.det(np.asarray(transform.rotation)), 1.0)


def test_antiparallel_reference_axis_rotation_is_deterministic():
    from llm_matgen.adsorption.proposals import LocalFrame, place_adsorbate_rigid

    frame = LocalFrame(
        origin=(0.0, 0.0, 0.0),
        tangent1=(1.0, 0.0, 0.0),
        tangent2=(0.0, 1.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    first, first_transform = place_adsorbate_rigid(
        oh(), 0, (0.0, 0.0, -1.0), frame, 2.0, 0.0, 0.0, 0.0
    )
    second, second_transform = place_adsorbate_rigid(
        oh(), 0, (0.0, 0.0, -1.0), frame, 2.0, 0.0, 0.0, 0.0
    )

    assert np.allclose(first.cart_coords, second.cart_coords)
    assert first_transform.rotation == second_transform.rotation
