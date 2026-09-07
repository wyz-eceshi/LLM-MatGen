from __future__ import annotations

import numpy as np
import pytest

from .conftest import job_snapshot, poscar


@pytest.mark.parametrize(
    ("incar", "markers", "stable", "status", "reason"),
    [
        ("NSW=20\nIBRION=2\n", {"ionic_converged": False}, True, "rejected_ionic_unconverged", "ionic_not_converged"),
        ("NSW=2\nIBRION=2\n", {"ionic_converged": False, "ionic_steps": 2}, True, "rejected_ionic_unconverged", "nsw_exhausted"),
        ("NSW=20\nIBRION=2\n", {"electronic_converged": False}, True, "rejected_electronic_unconverged", "electronic_not_converged"),
        ("NSW=20\nIBRION=2\n", {"fatal_warning": True}, True, "rejected_fatal_warning", "fatal_marker"),
        ("NSW=20\nIBRION=5\n", {}, True, "rejected_incomplete", "frequency_run"),
        ("NSW=20\nIBRION=2\nIMAGES=5\n", {}, True, "rejected_incomplete", "neb_run"),
        ("NSW=0\nIBRION=-1\n", {}, True, "rejected_incomplete", "static_run"),
        ("NSW=20\nIBRION=2\n", {}, False, "rejected_incomplete", "unstable_files"),
    ],
)
def test_quality_gates_have_deterministic_rejections(incar, markers, stable, status, reason):
    """Break caught: a known non-training VASP job becoming eligible."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    result = CaseExtractor().extract(job_snapshot(incar=incar, markers=markers, stable=stable))
    assert result.status.value == status
    assert reason in result.audit.rejection_reasons


def test_composition_mismatch_and_missing_clean_parent_are_rejected():
    """Break caught: comparing structures that cannot represent one relaxation."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    mismatched = poscar(["Cu", "O"], [2, 1], [(0, 0, .45), (.5, .5, .45), (0, 0, .55)])
    result = CaseExtractor().extract(job_snapshot(final=mismatched))
    assert result.status.value == "rejected_ambiguous_mapping"
    assert "composition_mismatch" in result.audit.rejection_reasons

    snapshot = job_snapshot()
    missing_parent = snapshot.model_copy(update={"clean_parent_identity": None, "clean_parent_poscar": None})
    result = CaseExtractor().extract(missing_parent)
    assert result.status.value == "rejected_incomplete"
    assert "clean_parent_missing" in result.audit.rejection_reasons


def test_atom_mapping_recovers_reordering_and_rejects_equal_cost_ambiguity():
    """Break caught: arbitrary atom identity swaps corrupting local displacements."""
    from llm_matgen.adsorption.extractor import AmbiguousMappingError, map_atoms

    lattice = np.eye(3) * 10
    initial = np.array([[0.1, 0.1, 0.1], [0.8, 0.8, 0.8]])
    reordered = np.array([[0.81, 0.8, 0.8], [0.11, 0.1, 0.1]])
    assert map_atoms(initial, reordered, ["Cu", "Cu"], ["Cu", "Cu"], lattice) == [1, 0]
    with pytest.raises(AmbiguousMappingError):
        map_atoms(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
            ["H"],
            ["H", "H"],
            lattice,
            allow_count_mismatch=True,
        )


def test_geometry_uses_real_vacuum_axis_minimum_image_and_collective_drift():
    """Break caught: hard-coded z geometry or wrapped displacement artifacts."""
    from llm_matgen.adsorption.extractor import detect_vacuum_axis, minimum_image_displacement, remove_collective_drift

    lattice = np.array([[20.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 1.0, 5.0]])
    fractional = np.array([[0.48, 0.1, 0.1], [0.50, 0.6, 0.7], [0.52, 0.3, 0.4]])
    assert detect_vacuum_axis(lattice, fractional) == 0
    displacement = minimum_image_displacement(np.array([0.95, 0.0, 0.0]), np.array([0.05, 0.0, 0.0]), lattice)
    assert displacement == pytest.approx([2.0, 0.0, 0.0])
    corrected, drift = remove_collective_drift(
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        np.array([[0.01, 0.02, 0.03], [0.51, 0.52, 0.53]]),
        np.eye(3),
    )
    assert drift == pytest.approx([0.01, 0.02, 0.03])
    assert corrected == pytest.approx(np.zeros((2, 3)))


def test_parent_partition_keeps_oxide_oxygen_in_slab():
    """Break caught: classifying every O atom as adsorbate on oxide surfaces."""
    from llm_matgen.adsorption.extractor import partition_against_parent

    lattice = np.eye(3) * 10
    child_species = ["Ti", "O", "O"]
    child = np.array([[0.0, 0.0, 0.5], [0.5, 0.5, 0.5], [0.25, 0.25, 0.65]])
    slab_ids, adsorbate_ids = partition_against_parent(
        child_species,
        child,
        ["Ti", "O"],
        child[:2],
        lattice,
    )
    assert slab_ids == [0, 1]
    assert adsorbate_ids == [2]


def test_mapping_ambiguity_gap_is_accumulated_across_elements():
    """Break caught: an unrelated element displacement making a unique mapping look tied."""
    from llm_matgen.adsorption.extractor import map_atoms

    lattice = np.eye(3) * 10
    initial = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])
    final = np.array([[0.01, 0.0, 0.0], [0.21, 0.0, 0.0], [0.5, 0.2, 0.0]])
    assert map_atoms(initial, final, ["H", "H", "O"], ["H", "H", "O"], lattice) == [0, 1, 2]


def test_skew_cell_uses_plane_normal_and_orthonormal_inplane_coordinates():
    """Break caught: skew lattice vectors contaminating surface height and lateral motion."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    lattice = ((3.0, 0.0, 0.0), (1.0, 3.0, 0.0), (2.0, 0.0, 18.0))
    initial = poscar(
        ["Cu", "H"], [2, 1],
        [(0.0, 0.0, .45), (.5, .5, .45), (0.0, 0.0, .56)],
        lattice=lattice,
    )
    final = poscar(
        ["Cu", "H"], [2, 1],
        [(0.0, 0.0, .451), (.5, .5, .451), (.1, 0.0, .55)],
        lattice=lattice,
    )
    parent = poscar(
        ["Cu"], [2], [(0.0, 0.0, .45), (.5, .5, .45)], lattice=lattice
    )

    result = CaseExtractor().extract(job_snapshot(initial=initial, final=final, parent=parent))
    assert result.status.value == "eligible"
    assert result.features.surface_normal == pytest.approx((0.0, 0.0, 1.0))
    assert result.features.inplane_displacement == pytest.approx((0.278, 0.0), abs=1e-8)


def test_minimum_image_is_the_true_shortest_vector_in_a_skew_cell():
    """Break caught: component-wise wrapping missing the nearest periodic image."""
    from llm_matgen.adsorption.extractor import minimum_image_displacement

    lattice = np.array([[3.0, 0.0, 0.0], [2.9, 0.3, 0.0], [0.0, 0.0, 10.0]])
    displacement = minimum_image_displacement(
        np.array([0.0, 0.0, 0.0]), np.array([0.49, 0.49, 0.0]), lattice
    )

    assert displacement == pytest.approx([-0.009, -0.153, 0.0], abs=1e-12)


def test_final_desorption_and_dissociation_are_rejected():
    """Break caught: an intact initial pose hiding a failed final adsorbate."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    desorbed = poscar(
        ["Cu", "H"], [2, 1],
        [(0.0, 0.0, .451), (.5, .5, .451), (0.0, 0.0, .90)],
    )
    desorption = CaseExtractor().extract(job_snapshot(final=desorbed))
    assert desorption.status.value == "rejected_incomplete"
    assert "desorbed" in desorption.audit.rejection_reasons

    initial_oh = poscar(
        ["Cu", "O", "H"], [2, 1, 1],
        [(0.0, 0.0, .45), (.5, .5, .45), (0.0, 0.0, .56), (0.0, 0.0, .61)],
    )
    final_oh = poscar(
        ["Cu", "O", "H"], [2, 1, 1],
        [(0.0, 0.0, .451), (.5, .5, .451), (0.0, 0.0, .55), (.4, .4, .70)],
    )
    dissociation = CaseExtractor().extract(job_snapshot(initial=initial_oh, final=final_oh))
    assert dissociation.status.value == "rejected_incomplete"
    assert "adsorbate_dissociated" in dissociation.audit.rejection_reasons


def test_final_site_and_transition_are_extracted_from_relaxed_geometry():
    """Break caught: reporting the initial adsorption site after a real site hop."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    initial = poscar(
        ["Cu", "H"], [2, 1],
        [(0.0, 0.0, .45), (.5, .5, .45), (0.0, 0.0, .55)],
    )
    final = poscar(
        ["Cu", "H"], [2, 1],
        [(0.0, 0.0, .451), (.5, .5, .451), (.25, .25, .54)],
    )
    result = CaseExtractor().extract(job_snapshot(initial=initial, final=final))

    assert result.status.value == "eligible"
    assert result.features.initial_site_type == "top"
    assert result.features.final_site_type == "bridge"
    assert result.features.site_transition == "top->bridge"


def test_eligible_extraction_produces_versioned_features_without_outcar_artifact():
    """Break caught: accepting a job without useful local geometry or copying OUTCAR."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    result = CaseExtractor().extract(job_snapshot())
    assert result.status.value == "eligible"
    assert result.features.adsorbate_formula == "H"
    assert result.features.denticity == 1
    assert result.features.vacuum_axis == 2
    assert set(result.artifacts) == {"POSCAR", "CONTCAR", "INCAR", "structures.mson"}
    assert all(e.relative_path != "OUTCAR" or e.content_stored is False for e in result.evidence)
    pose = result.features.pose_correction
    assert pose["slab_indices"] == [0, 1]
    assert pose["adsorbate_indices"] == [2]
    assert pose["anchor_index_initial"] == pose["anchor_index_final"] == 2
    assert set(pose) >= {"local_frame", "initial_local_coordinates", "final_local_coordinates", "local_delta"}


def test_equivalent_fingerprint_ignores_nonsemantic_input_formatting():
    """Break caught: byte formatting preventing staging-copy folding."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    first = CaseExtractor().extract(job_snapshot(incar="NSW=20\nIBRION=2\n"))
    second = CaseExtractor().extract(job_snapshot(incar="NSW = 20 ; IBRION = 2\n"))
    assert first.exact_hash != second.exact_hash
    assert first.equivalent_fingerprint == second.equivalent_fingerprint

    reordered_initial = poscar(
        ["H", "Cu"], [1, 2], [(0.0, 0.0, .56), (0.0, 0.0, .45), (.5, .5, .45)]
    )
    reordered_final = poscar(
        ["H", "Cu"], [1, 2], [(0.02, 0.01, .55), (0.0, 0.0, .451), (.5, .5, .451)]
    )
    reordered = CaseExtractor().extract(
        job_snapshot(initial=reordered_initial, final=reordered_final)
    )
    assert reordered.status.value == "eligible"
    assert first.equivalent_fingerprint == reordered.equivalent_fingerprint


def test_run_type_does_not_treat_year_as_neb_or_relaxation_as_dos():
    """Break caught: broad path/settings heuristics rejecting real relaxations."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    year = CaseExtractor().extract(job_snapshot(job_dir="archive/2024/case"))
    relax_with_dos_output = CaseExtractor().extract(
        job_snapshot(incar="NSW=20\nIBRION=2\nLORBIT=11\nNEDOS=2000\n")
    )
    assert year.status.value == "eligible"
    assert relax_with_dos_output.status.value == "eligible"


def test_variable_cell_relaxation_is_conservatively_rejected():
    """Break caught: applying initial-cell PBC geometry to a changed final cell."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    changed_cell = poscar(
        ["Cu", "H"], [2, 1],
        [(0.0, 0.0, .451), (.5, .5, .451), (.02, .01, .55)],
        lattice=((3.1, 0, 0), (0, 3, 0), (0, 0, 18)),
    )
    result = CaseExtractor().extract(job_snapshot(final=changed_cell))
    assert result.status.value == "rejected_incomplete"
    assert "variable_cell_unsupported" in result.audit.rejection_reasons


def test_collective_drift_uses_true_shortest_images_in_skew_cell():
    """Break caught: component wrapping turning a tiny slab drift into a large vector."""
    from llm_matgen.adsorption.extractor import remove_collective_drift

    lattice = np.array([[3.0, 0.0, 0.0], [2.9, 0.3, 0.0], [0.0, 0.0, 10.0]])
    initial = np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.0]])
    final = np.array([[0.49, 0.49, 0.0], [0.69, 0.69, 0.0]])
    corrected, drift = remove_collective_drift(initial, final, lattice)
    assert drift == pytest.approx([-0.009, -0.153, 0.0], abs=1e-12)
    assert corrected == pytest.approx(np.zeros((2, 3)), abs=1e-12)


def test_mapping_metrics_report_near_degenerate_cost_gap():
    """Break caught: publishing unique mapping without its best-vs-second evidence."""
    from llm_matgen.adsorption.extractor import map_atoms_with_metrics

    lattice = np.eye(3) * 10
    result = map_atoms_with_metrics(
        np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        np.array([[0.09, 0.0, 0.0], [0.11, 0.0, 0.0]]),
        ["Cu", "Cu"], ["Cu", "Cu"], lattice,
    )
    assert result.mapping == [0, 1]
    assert result.cost_gap == pytest.approx(0.4)
    assert result.best_cost == pytest.approx(1.8)


def test_canonical_graph_distinguishes_topology_and_reports_connectivity():
    """Break caught: edge-label bags treating a C4 chain and star as the same molecule."""
    from llm_matgen.adsorption.extractor import canonical_adsorbate_graph

    lattice = np.eye(3) * 20
    chain = np.array([[.1,.1,.1], [.17,.1,.1], [.24,.1,.1], [.31,.1,.1]])
    star = np.array([[.1,.1,.1], [.17,.1,.1], [.065,.1606,.1], [.065,.0394,.1]])
    chain_graph = canonical_adsorbate_graph(("C",)*4, chain, [0,1,2,3], lattice)
    star_graph = canonical_adsorbate_graph(("C",)*4, star, [0,1,2,3], lattice)
    assert chain_graph.connected is True
    assert star_graph.connected is True
    assert chain_graph.fingerprint != star_graph.fingerprint


def test_disconnected_graph_is_rejected_before_canonical_permutations(monkeypatch):
    """Break caught: a disconnected slab fragment causing factorial graph enumeration."""
    import llm_matgen.adsorption.extractor as extractor

    def forbidden_permutations(_group):
        raise AssertionError("disconnected graphs must not be canonically permuted")

    monkeypatch.setattr(extractor, "permutations", forbidden_permutations)
    lattice = np.eye(3) * 100.0
    fractional = np.array([[index / 10.0, 0.0, 0.0] for index in range(10)])

    graph = extractor.canonical_adsorbate_graph(
        ("C",) * 10,
        fractional,
        list(range(10)),
        lattice,
    )

    assert graph.connected is False
    assert graph.canonical_edges == []
    assert graph.fingerprint == "C10:disconnected"


def test_canonical_graph_refines_equivalence_classes_before_permuting(monkeypatch):
    """Break caught: a linear molecule being canonicalized with one n-factorial group."""
    import itertools
    import llm_matgen.adsorption.extractor as extractor

    group_sizes = []

    def bounded_permutations(group):
        group_sizes.append(len(group))
        if len(group) > 2:
            raise AssertionError("color refinement must split the carbon chain first")
        return itertools.permutations(group)

    monkeypatch.setattr(extractor, "permutations", bounded_permutations)
    lattice = np.eye(3) * 100.0
    fractional = np.array([[0.1 + index * 0.014, 0.1, 0.1] for index in range(10)])
    graph = extractor.canonical_adsorbate_graph(("C",) * 10, fractional, list(range(10)), lattice)

    assert graph.connected is True
    assert group_sizes == [2, 2, 2, 2, 2]


def test_highly_symmetric_graph_fails_with_controlled_complexity_audit(monkeypatch):
    """Break caught: a symmetric connected graph exhausting host virtual memory."""
    import llm_matgen.adsorption.extractor as extractor

    def forbidden_permutations(_group):
        raise AssertionError("permutations must not start beyond the resource budget")

    monkeypatch.setattr(extractor, "permutations", forbidden_permutations)
    lattice = np.eye(3) * 100.0
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    radius = 2.7
    cartesian = np.column_stack([20.0 + radius * np.cos(angles), 20.0 + radius * np.sin(angles), np.full(12, 20.0)])

    with pytest.raises(extractor.GraphCanonicalizationLimitError, match="adsorbate_graph_complexity_exceeds_limit"):
        extractor.canonical_adsorbate_graph(("C",) * 12, cartesian / 100.0, list(range(12)), lattice)


def test_anchor_identity_switch_is_rejected_and_pose_frame_is_reconstructible():
    """Break caught: mixing corrections expressed around different molecular anchors."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    initial = poscar(
        ["Cu", "O", "H"], [2,1,1],
        [(0,0,.45),(.5,.5,.45),(0,0,.56),(0,0,.61)],
    )
    switched = poscar(
        ["Cu", "O", "H"], [2,1,1],
        [(0,0,.451),(.5,.5,.451),(0,0,.61),(0,0,.56)],
    )
    rejected = CaseExtractor().extract(job_snapshot(initial=initial, final=switched))
    assert rejected.status.value == "rejected_incomplete"
    assert "anchor_identity_changed" in rejected.audit.rejection_reasons

    eligible = CaseExtractor().extract(job_snapshot())
    pose = eligible.features.pose_correction
    frame = np.array([pose["local_frame"][name] for name in ("e1", "e2", "normal")])
    local_delta = np.array(pose["local_delta"])
    assert frame @ frame.T == pytest.approx(np.eye(3), abs=1e-12)
    assert local_delta @ frame == pytest.approx(pose["translation"], abs=1e-12)


def test_exact_identity_includes_clean_parent_provenance():
    """Break caught: two different clean slabs sharing one exact case identity."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    first = CaseExtractor().extract(job_snapshot())
    other_parent = poscar(["Cu"], [2], [(0,0,.44),(.5,.5,.44)])
    second = CaseExtractor().extract(job_snapshot(parent=other_parent))
    assert first.exact_hash != second.exact_hash


def test_surface_coverage_termination_and_true_2d_period_are_provenance_based():
    """Break caught: bottom layers diluting coverage or short skew translations being missed."""
    from llm_matgen.adsorption.extractor import CaseExtractor, shortest_2d_periodic_distance

    assert shortest_2d_periodic_distance(
        np.array([3.0, 0.0, 0.0]), np.array([14.9, 0.3, 0.0])
    ) == pytest.approx(np.sqrt(.1**2 + .3**2))
    initial = poscar(
        ["Cu", "H"], [4,1],
        [(0,0,.35),(.5,.5,.35),(0,0,.45),(.5,.5,.45),(0,0,.56)],
    )
    final = poscar(
        ["Cu", "H"], [4,1],
        [(0,0,.351),(.5,.5,.351),(0,0,.451),(.5,.5,.451),(.02,.01,.55)],
    )
    parent = poscar(
        ["Cu"], [4], [(0,0,.35),(.5,.5,.35),(0,0,.45),(.5,.5,.45)]
    )
    result = CaseExtractor().extract(job_snapshot(initial=initial, final=final, parent=parent))
    assert result.status.value == "eligible"
    assert result.features.coverage == pytest.approx(0.5)
    assert result.features.termination == "Cu"
    assert result.features.miller_index is None


def test_mapping_gap_is_persisted_and_unknown_ionic_steps_stays_none():
    """Break caught: losing mapping confidence or inventing zero ionic steps."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    result = CaseExtractor().extract(job_snapshot(markers={"ionic_steps": None}))
    assert result.status.value == "eligible"
    assert result.audit.mapping_cost_gap is not None
    assert result.audit.mapping_cost_gap > 0
    assert result.audit.ionic_steps is None


def test_rigid_multiatom_adsorbate_is_supported_but_observed_flexing_is_rejected():
    """Break caught: rejecting all molecules or accepting a materially deformed one."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    initial = poscar(
        ["Cu", "O", "H"], [2,1,1],
        [(0,0,.45),(.5,.5,.45),(0,0,.56),(0,0,.61)],
    )
    rigid_final = poscar(
        ["Cu", "O", "H"], [2,1,1],
        [(0,0,.451),(.5,.5,.451),(.01,0,.55),(.01,0,.60)],
    )
    flexible_final = poscar(
        ["Cu", "O", "H"], [2,1,1],
        [(0,0,.451),(.5,.5,.451),(.01,0,.55),(.01,0,.61)],
    )
    rigid = CaseExtractor().extract(job_snapshot(initial=initial, final=rigid_final))
    flexible = CaseExtractor().extract(job_snapshot(initial=initial, final=flexible_final))
    assert rigid.status.value == "eligible"
    assert flexible.status.value == "rejected_incomplete"
    assert "flexible_adsorbate_unsupported" in flexible.audit.rejection_reasons


def test_rigidity_uses_complete_internal_distance_matrix_not_only_bonds():
    """Break caught: a large H-O-H angle change preserving both O-H bond lengths."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    initial = poscar(
        ["Cu", "O", "H"], [2, 1, 2],
        [(0,0,.45),(.5,.5,.45),(0,0,.56),(.2,0,.56),(0,.2,.56)],
    )
    angle_changed = poscar(
        ["Cu", "O", "H"], [2, 1, 2],
        [(0,0,.451),(.5,.5,.451),(0,0,.56),(.2,0,.56),(.8,0,.56)],
    )
    result = CaseExtractor().extract(job_snapshot(initial=initial, final=angle_changed))
    assert result.status.value == "rejected_incomplete"
    assert "flexible_adsorbate_unsupported" in result.audit.rejection_reasons


def test_equivalent_fingerprint_uses_canonical_bond_indices():
    """Break caught: raw POSCAR atom indices leaking into semantic duplicate identity."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    first_initial = poscar(
        ["Cu", "O", "H"], [2,1,2],
        [(0,0,.45),(.5,.5,.45),(0,0,.56),(.3,0,.56),(0,.3,.56)],
    )
    first_final = poscar(
        ["Cu", "O", "H"], [2,1,2],
        [(0,0,.451),(.5,.5,.451),(.01,0,.56),(.31,0,.56),(.01,.3,.56)],
    )
    reordered_initial = poscar(
        ["H", "Cu", "O"], [2,2,1],
        [(.3,0,.56),(0,.3,.56),(0,0,.45),(.5,.5,.45),(0,0,.56)],
    )
    reordered_final = poscar(
        ["H", "Cu", "O"], [2,2,1],
        [(.31,0,.56),(.01,.3,.56),(0,0,.451),(.5,.5,.451),(.01,0,.56)],
    )
    first = CaseExtractor().extract(job_snapshot(initial=first_initial, final=first_final))
    reordered = CaseExtractor().extract(
        job_snapshot(initial=reordered_initial, final=reordered_final)
    )
    assert first.status.value == reordered.status.value == "eligible"
    assert first.features.internal_bond_graph == reordered.features.internal_bond_graph
    assert first.equivalent_fingerprint == reordered.equivalent_fingerprint


def test_surface_layer_gap_clustering_excludes_close_subsurface_plane():
    """Break caught: a fixed 1.5 A window mixing a plane only 0.9 A below the surface."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    initial = poscar(
        ["Cu", "Pt", "H"], [2,2,1],
        [(0,0,.45),(.5,.5,.45),(0,0,.40),(.5,.5,.40),(0,0,.56)],
    )
    final = poscar(
        ["Cu", "Pt", "H"], [2,2,1],
        [(0,0,.451),(.5,.5,.451),(0,0,.401),(.5,.5,.401),(.02,.01,.55)],
    )
    parent = poscar(
        ["Cu", "Pt"], [2,2],
        [(0,0,.45),(.5,.5,.45),(0,0,.40),(.5,.5,.40)],
    )
    result = CaseExtractor().extract(job_snapshot(initial=initial, final=final, parent=parent))
    assert result.status.value == "eligible"
    assert result.features.coverage == pytest.approx(0.5)
    assert result.features.termination == "Cu"
    assert result.features.surface_layer_composition == {"Cu": 1.0}


def test_surface_side_and_height_follow_the_local_active_site_on_a_rough_slab():
    """Break caught: a recessed bottom site being compared with the global slab extrema."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    initial = poscar(
        ["Cu", "Ni", "C", "O", "H"],
        [4, 1, 1, 1, 1],
        [
            (0.0, 0.0, .10),
            (.5, 0.0, .10),
            (0.0, .5, .10),
            (.5, .5, .10),
            (.25, .25, .45),
            (.25, .25, .75),
            (.25, .25, .37),
            (.25, .25, .31),
        ],
    )
    final = poscar(
        ["Cu", "Ni", "C", "O", "H"],
        [4, 1, 1, 1, 1],
        [
            (0.0, 0.0, .101),
            (.5, 0.0, .101),
            (0.0, .5, .101),
            (.5, .5, .101),
            (.25, .25, .451),
            (.25, .25, .751),
            (.25, .25, .371),
            (.25, .25, .311),
        ],
    )
    parent = poscar(
        ["Cu", "Ni", "C"],
        [4, 1, 1],
        [
            (0.0, 0.0, .10),
            (.5, 0.0, .10),
            (0.0, .5, .10),
            (.5, .5, .10),
            (.25, .25, .45),
            (.25, .25, .75),
        ],
    )

    result = CaseExtractor().extract(job_snapshot(initial=initial, final=final, parent=parent))

    assert result.status.value == "eligible"
    assert result.features.surface_side == "bottom"
    assert result.features.initial_height == pytest.approx(1.44)
    assert result.features.final_height == pytest.approx(1.44)


def test_explicit_surface_provenance_is_parsed_without_guessing():
    """Break caught: dropping an explicit Miller/termination declaration in POSCAR provenance."""
    from llm_matgen.adsorption.extractor import CaseExtractor

    initial = job_snapshot().files["POSCAR"].content.replace(
        "fixture", "fixture MILLER=(1,1,1) TERMINATION=Cu", 1
    )
    result = CaseExtractor().extract(job_snapshot(initial=initial))
    assert result.status.value == "eligible"
    assert result.features.miller_index == (1, 1, 1)
    assert result.features.termination == "Cu"
