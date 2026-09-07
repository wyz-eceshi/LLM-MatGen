from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Molecule, Structure


def layered_non_z_slab(*, inplane: float = 4.0, vacuum: float = 18.0) -> Structure:
    return Structure(
        Lattice(
            [
                [0.0, inplane, 0.0],
                [0.0, 0.0, inplane],
                [vacuum, 0.0, 0.0],
            ]
        ),
        ["Cu"] * 6,
        [
            [0.0, 0.0, 0.45],
            [0.5, 0.5, 0.45],
            [0.0, 0.0, 0.50],
            [0.5, 0.5, 0.50],
            [0.0, 0.0, 0.55],
            [0.5, 0.5, 0.55],
        ],
    )


def oh() -> Molecule:
    return Molecule(
        ["O", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
        charge=0,
        spin_multiplicity=2,
    )


def proposal(slab: Structure, *, side="top", height=1.8, coverage=0.5):
    from llm_matgen.adsorption.proposals import (
        AdsorptionProposal,
        AlgorithmicProposalEvidence,
        build_surface_frame,
    )

    top_x = max(np.asarray(slab.cart_coords)[:, 0])
    origin = (float(top_x), 0.0, 0.0)
    return AdsorptionProposal(
        proposal_id=f"p:{side}:{height}",
        source="algorithmic",
        site_id=f"{side}:top:0",
        site_type="top",
        surface_side=side,
        frame=build_surface_frame(slab, side, origin=origin),
        height=height,
        azimuth=0.0,
        tilt=0.0,
        coverage=coverage,
        evidence=AlgorithmicProposalEvidence(method="test-fixture"),
    )


def candidate(slab: Structure, coordinates) -> Structure:
    value = slab.copy()
    for symbol, coordinate in zip(("O", "H"), coordinates):
        value.append(symbol, coordinate, coords_are_cartesian=True)
    return value


def valid_candidate(slab: Structure) -> Structure:
    top_x = max(np.asarray(slab.cart_coords)[:, 0])
    return candidate(slab, [(top_x + 1.8, 0.0, 0.0), (top_x + 2.77, 0.0, 0.0)])


def reason_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_validator_accepts_role_aware_anchor_contact_but_rejects_nonanchor_collision():
    from llm_matgen.adsorption.validation import AdsorptionCandidateValidator

    slab = layered_non_z_slab()
    validator = AdsorptionCandidateValidator(
        slab,
        oh(),
        anchor_zero_based=0,
        anchor_contact_window=(0.75, 1.35),
        min_vacuum_each_side=3.0,
    )
    accepted = validator.validate(valid_candidate(slab), proposal(slab))
    top_x = max(np.asarray(slab.cart_coords)[:, 0])
    collision = candidate(slab, [(top_x + 1.8, 0.0, 0.0), (top_x + 0.4, 0.0, 0.0)])
    rejected = validator.validate(collision, proposal(slab))

    assert accepted.accepted
    assert "slab_adsorbate_collision" in reason_codes(rejected)


def test_validator_uses_element_radius_contact_window_and_full_internal_graph():
    from llm_matgen.adsorption.validation import AdsorptionCandidateValidator

    slab = layered_non_z_slab()
    validator = AdsorptionCandidateValidator(
        slab,
        oh(),
        anchor_zero_based=0,
        anchor_contact_window=(0.75, 1.35),
        min_vacuum_each_side=3.0,
    )
    top_x = max(np.asarray(slab.cart_coords)[:, 0])
    too_far = candidate(slab, [(top_x + 4.0, 0.0, 0.0), (top_x + 4.97, 0.0, 0.0)])
    stretched = candidate(slab, [(top_x + 1.8, 0.0, 0.0), (top_x + 3.3, 0.0, 0.0)])

    assert "anchor_contact_window" in reason_codes(validator.validate(too_far, proposal(slab, height=4.0)))
    stretched_codes = reason_codes(validator.validate(stretched, proposal(slab)))
    assert "internal_distance_changed" in stretched_codes
    assert "internal_bond_graph_changed" in stretched_codes


def test_validator_checks_adsorbate_2d_periodic_images_and_both_vacuum_sides():
    from llm_matgen.adsorption.validation import AdsorptionCandidateValidator

    small = layered_non_z_slab(inplane=1.0)
    small_validator = AdsorptionCandidateValidator(
        small,
        oh(),
        anchor_zero_based=0,
        min_vacuum_each_side=3.0,
    )
    assert "adsorbate_periodic_image" in reason_codes(
        small_validator.validate(valid_candidate(small), proposal(small))
    )

    slab = layered_non_z_slab()
    top_x = max(np.asarray(slab.cart_coords)[:, 0])
    near_edge = candidate(slab, [(17.4, 0.0, 0.0), (top_x + 2.77, 0.0, 0.0)])
    vacuum_validator = AdsorptionCandidateValidator(
        slab,
        oh(),
        anchor_zero_based=0,
        min_vacuum_each_side=3.0,
    )
    assert "insufficient_vacuum" in reason_codes(
        vacuum_validator.validate(near_edge, proposal(slab, height=7.5))
    )


def test_validator_finds_short_2d_image_requiring_large_integer_coefficient():
    from llm_matgen.adsorption.validation import AdsorptionCandidateValidator

    skew = Structure(
        Lattice([[0.0, 4.0, 0.0], [0.0, 40.0, 0.8], [18.0, 0.0, 0.0]]),
        ["Cu", "Cu", "Cu"],
        [[0.0, 0.0, 0.45], [0.0, 0.0, 0.50], [0.0, 0.0, 0.55]],
    )
    top_x = max(np.asarray(skew.cart_coords)[:, 0])
    value = candidate(skew, [(top_x + 1.8, 0.0, 0.0), (top_x + 2.77, 0.0, 0.0)])
    validator = AdsorptionCandidateValidator(
        skew,
        oh(),
        anchor_zero_based=0,
        min_vacuum_each_side=3.0,
    )

    assert "adsorbate_periodic_image" in reason_codes(
        validator.validate(value, proposal(skew))
    )


def test_fixed_layers_follow_true_normal_and_adsorbate_is_always_movable():
    from llm_matgen.adsorption.validation import apply_fixed_bottom_layers

    slab = layered_non_z_slab()
    slab.translate_sites([1], [0.2, 0.0, 0.0], frac_coords=False, to_unit_cell=False)
    structure = valid_candidate(slab)
    fixed, layer_ids = apply_fixed_bottom_layers(
        structure,
        slab_atom_count=len(slab),
        fixed_bottom_layers=1,
    )
    flags = fixed.site_properties["selective_dynamics"]

    assert layer_ids == ((0, 1),)
    assert flags[0] == [False, False, False]
    assert flags[1] == [False, False, False]
    assert all(flags[index] == [True, True, True] for index in range(2, len(structure)))


def test_validator_rejects_wrong_order_bad_fixed_flags_side_and_coverage():
    from llm_matgen.adsorption.validation import AdsorptionCandidateValidator, apply_fixed_bottom_layers

    slab = layered_non_z_slab()
    validator = AdsorptionCandidateValidator(
        slab,
        oh(),
        anchor_zero_based=0,
        fixed_bottom_layers=1,
        expected_coverage=0.5,
        min_vacuum_each_side=3.0,
    )
    fixed, _ = apply_fixed_bottom_layers(
        valid_candidate(slab), slab_atom_count=len(slab), fixed_bottom_layers=1
    )
    wrong_order = fixed.copy()
    wrong_order.replace(0, "O")
    wrong_order.replace(len(slab), "Cu")
    wrong_flags = fixed.copy()
    flags = list(wrong_flags.site_properties["selective_dynamics"])
    flags[-1] = [False, False, False]
    wrong_flags.remove_site_property("selective_dynamics")
    wrong_flags.add_site_property("selective_dynamics", flags)
    bottom_with_top_frame = proposal(slab).model_copy(update={"surface_side": "bottom"})
    bad_coverage = proposal(slab).model_copy(update={"coverage": 0.25})

    assert "species_order" in reason_codes(validator.validate(wrong_order, proposal(slab)))
    assert "fixed_layer_flags" in reason_codes(validator.validate(wrong_flags, proposal(slab)))
    assert "surface_side_normal" in reason_codes(validator.validate(fixed, bottom_with_top_frame))
    assert "coverage_mismatch" in reason_codes(validator.validate(fixed, bad_coverage))


def test_validator_deduplicates_identical_and_symmetry_equivalent_candidates():
    from llm_matgen.adsorption.validation import AdsorptionCandidateValidator

    slab = layered_non_z_slab()
    validator = AdsorptionCandidateValidator(
        slab,
        oh(),
        anchor_zero_based=0,
        min_vacuum_each_side=3.0,
    )
    first = validator.validate(valid_candidate(slab), proposal(slab))
    second = validator.validate(valid_candidate(slab), proposal(slab))

    assert first.accepted
    assert not second.accepted
    assert "duplicate_geometry" in reason_codes(second)
