from pymatgen.core import Lattice, Structure


def make_fixture() -> Structure:
    return Structure(
        Lattice.cubic(4.2),
        ["Li", "Co", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]],
    )


def test_structure_hash_is_stable_and_sensitive_to_coordinates():
    from llm_matgen.utils.structure import structure_sha256

    structure = make_fixture()
    assert structure_sha256(structure) == structure_sha256(structure.copy())

    changed = structure.copy()
    changed.translate_sites([0], [0.01, 0, 0], frac_coords=True)
    assert structure_sha256(changed) != structure_sha256(structure)


def test_site_ids_are_stable_and_include_parent_identity():
    from llm_matgen.utils.structure import assign_site_ids, structure_sha256

    structure = make_fixture()
    parent_hash = structure_sha256(structure)
    first = assign_site_ids(structure, parent_hash)
    second = assign_site_ids(structure.copy(), parent_hash)

    assert first == second
    assert len(first) == len(structure)
    assert len(set(first)) == len(structure)
    assert all(parent_hash[:12] in site_id for site_id in first)


def test_canonical_payload_normalizes_fractional_coordinates_but_not_site_order():
    from llm_matgen.utils.structure import canonical_structure_payload

    structure = make_fixture()
    shifted = structure.copy()
    shifted.translate_sites([0], [1, 0, 0], frac_coords=True)

    assert canonical_structure_payload(structure) == canonical_structure_payload(shifted)

    reordered = Structure(
        structure.lattice,
        list(reversed([site.specie for site in structure])),
        list(reversed([site.frac_coords for site in structure])),
    )
    assert canonical_structure_payload(structure) != canonical_structure_payload(reordered)


def test_versioned_hash_v2_includes_vasp_semantic_site_properties():
    from llm_matgen.utils.structure import structure_sha256

    plain = make_fixture()
    constrained = plain.copy()
    constrained.add_site_property(
        "selective_dynamics",
        [[False, False, False], *([[True, True, True]] * (len(constrained) - 1))],
    )

    assert structure_sha256(plain, version="v1") == structure_sha256(
        constrained, version="v1"
    )
    assert structure_sha256(plain, version="v2") != structure_sha256(
        constrained, version="v2"
    )

    nonperiodic_z = Structure(
        Lattice(plain.lattice.matrix, pbc=(True, True, False)),
        [site.specie for site in plain],
        plain.frac_coords,
    )
    assert structure_sha256(plain, version="v2") != structure_sha256(
        nonperiodic_z, version="v2"
    )
