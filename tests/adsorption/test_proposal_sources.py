from __future__ import annotations

from types import SimpleNamespace
import sqlite3

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from llm_matgen.adsorption.models import CaseFeatureSet, RetrievalMatch
from llm_matgen.adsorption.store import AdsorptionStoreError


def triangular_non_z_slab() -> Structure:
    return Structure(
        Lattice([[0.0, 3.0, 0.0], [0.0, 1.5, 2.598076211], [18.0, 0.0, 0.0]]),
        ["Pt"],
        [[0.0, 0.0, 0.5]],
    )


def square_non_z_slab() -> Structure:
    return Structure(
        Lattice([[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [18.0, 0.0, 0.0]]),
        ["Cu", "Cu", "Cu", "Cu"],
        [[0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]],
    )


def test_algorithmic_source_uses_asf_on_true_normal_and_normalizes_ontop():
    from llm_matgen.adsorption.proposals import AlgorithmicProposalSource

    proposals = list(
        AlgorithmicProposalSource(
            triangular_non_z_slab(),
            sides=("top",),
            site_types=("top", "bridge", "hollow"),
        ).proposals()
    )

    assert {item.site_type for item in proposals} == {"top", "bridge", "hollow"}
    assert all(item.surface_side == "top" for item in proposals)
    assert all(np.allclose(item.frame.normal, [1.0, 0.0, 0.0]) for item in proposals)
    assert all(item.evidence.method == "pymatgen-asf" for item in proposals)
    assert not any(item.site_type == "ontop" for item in proposals)


def test_algorithmic_source_keeps_top_bottom_flags_independent_and_adds_hollow4_explicit():
    from llm_matgen.adsorption.proposals import AlgorithmicProposalSource

    proposals = list(
        AlgorithmicProposalSource(
            square_non_z_slab(),
            sides=("top", "bottom"),
            site_types=("hollow4", "explicit"),
            explicit_sites=((9.0, 1.0, 1.0),),
        ).proposals()
    )

    assert {(item.surface_side, item.site_type) for item in proposals} == {
        ("top", "hollow4"),
        ("bottom", "hollow4"),
        ("top", "explicit"),
        ("bottom", "explicit"),
    }
    top = next(item for item in proposals if item.surface_side == "top")
    bottom = next(item for item in proposals if item.surface_side == "bottom")
    assert np.allclose(top.frame.normal, -np.asarray(bottom.frame.normal))
    assert top.site_id != bottom.site_id


def test_hollow4_enumerates_all_periodic_fourfold_centres_in_a_supercell():
    from llm_matgen.adsorption.proposals import AlgorithmicProposalSource

    proposals = list(
        AlgorithmicProposalSource(
            square_non_z_slab(),
            sides=("top",),
            site_types=("hollow4",),
        ).proposals()
    )

    assert len(proposals) == 4
    assert all(item.site_type == "hollow4" for item in proposals)


def test_explicit_coordinates_are_consumed_without_repeating_explicit_site_type():
    from llm_matgen.adsorption.proposals import AlgorithmicProposalSource

    proposals = list(
        AlgorithmicProposalSource(
            square_non_z_slab(),
            sides=("top",),
            site_types=("top",),
            explicit_sites=((9.0, 1.0, 1.0),),
        ).proposals()
    )

    assert any(item.site_type == "explicit" for item in proposals)


def base_proposal():
    from llm_matgen.adsorption.proposals import (
        AdsorptionProposal,
        AlgorithmicProposalEvidence,
        LocalFrame,
    )

    return AdsorptionProposal(
        proposal_id="algorithmic:top:top:0000:h1.8:a0:t0:r0",
        source="algorithmic",
        site_id="top:top:0000",
        site_type="top",
        surface_side="top",
        frame=LocalFrame(
            origin=(9.0, 0.0, 0.0),
            tangent1=(0.0, 1.0, 0.0),
            tangent2=(0.0, 0.0, 1.0),
            normal=(1.0, 0.0, 0.0),
        ),
        height=1.8,
        azimuth=0.0,
        tilt=0.0,
        roll=0.0,
        coverage=0.25,
        evidence=AlgorithmicProposalEvidence(method="pymatgen-asf"),
    )


def retrieval_match() -> RetrievalMatch:
    return RetrievalMatch(
        case_id="case-1",
        revision_id="rev-1",
        equivalent_fingerprint="f-1",
        total_score=0.9,
        component_scores={"site_type": 1.0},
        completeness=1.0,
        accepted=True,
    )


def test_retrieved_source_emits_final_pose_and_local_delta_without_global_copy():
    from llm_matgen.adsorption.proposals import RetrievedProposalSource

    features = CaseFeatureSet(
        final_height=1.55,
        pose_correction={
            "slab_indices": [0, 1],
            "adsorbate_indices": [2, 3],
            "initial_local_coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
            "final_local_coordinates": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.96]],
            "local_delta": [0.2, -0.1, -0.25],
            "rotation_degrees": 12.0,
        },
    )
    proposals = list(
        RetrievedProposalSource(
            (base_proposal(),),
            (retrieval_match(),),
            revision_loader=lambda _: SimpleNamespace(features=features, index_revision=7),
            index_revision=7,
        ).proposals()
    )

    assert [item.evidence.mode for item in proposals] == ["final_pose", "local_delta"]
    assert proposals[0].local_adsorbate_coordinates == ((0.0, 0.0, 0.0), (0.1, 0.0, 0.96))
    assert proposals[0].height == 1.55
    assert proposals[1].local_delta == (0.2, -0.1, -0.25)
    assert proposals[1].rotation_degrees == 12.0
    assert all(item.evidence.slab_indices == (0, 1) for item in proposals)
    assert all(item.evidence.adsorbate_indices == (2, 3) for item in proposals)
    assert all(item.frame.origin == (9.0, 0.0, 0.0) for item in proposals)


def test_retrieved_source_rejects_pose_schema_with_inconsistent_adsorbate_rows():
    from llm_matgen.adsorption.proposals import RetrievedProposalSource

    features = CaseFeatureSet(
        pose_correction={
            "slab_indices": [0, 1],
            "adsorbate_indices": [2, 3],
            "initial_local_coordinates": [[0.0, 0.0, 0.0]],
            "final_local_coordinates": [[0.0, 0.0, 0.0]],
            "local_delta": [0.0, 0.0, 0.0],
        },
    )
    with pytest.raises(ValueError, match="adsorbate_indices"):
        RetrievedProposalSource(
            (base_proposal(),),
            (retrieval_match(),),
            revision_loader=lambda _: SimpleNamespace(
                features=features, index_revision=7
            ),
            index_revision=7,
        )


def test_history_off_never_instantiates_store_and_prefer_only_falls_back_on_store_error():
    from llm_matgen.adsorption.proposals import resolve_history

    touched = False

    def forbidden_store():
        nonlocal touched
        touched = True
        raise AssertionError("store must stay unopened")

    off = resolve_history(
        history_policy="off",
        query_features=CaseFeatureSet(),
        store_factory=forbidden_store,
    )
    assert not touched
    assert off.trace.fallback_reason == "history_disabled"

    def broken_store():
        raise AdsorptionStoreError("missing database")

    prefer = resolve_history(
        history_policy="prefer",
        query_features=CaseFeatureSet(),
        store_factory=broken_store,
    )
    assert prefer.trace.fallback_reason.startswith("store_unavailable:")

    sqlite_fallback = resolve_history(
        history_policy="prefer",
        query_features=CaseFeatureSet(),
        store_factory=lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("corrupt")),
    )
    assert sqlite_fallback.trace.fallback_reason.startswith("store_unavailable:")

    filesystem_fallback = resolve_history(
        history_policy="prefer",
        query_features=CaseFeatureSet(),
        store_factory=lambda: (_ for _ in ()).throw(OSError("local store unavailable")),
    )
    assert filesystem_fallback.trace.fallback_reason.startswith("store_unavailable:")

    with pytest.raises(ValueError, match="unexpected"):
        resolve_history(
            history_policy="prefer",
            query_features=CaseFeatureSet(),
            store_factory=lambda: (_ for _ in ()).throw(ValueError("unexpected")),
        )


def test_history_require_fails_clearly_and_resolution_freezes_revision():
    from llm_matgen.adsorption.proposals import HistoryRequiredError, resolve_history

    with pytest.raises(HistoryRequiredError, match="history is required"):
        resolve_history(
            history_policy="require",
            query_features=CaseFeatureSet(),
            store_factory=lambda: (_ for _ in ()).throw(AdsorptionStoreError("broken")),
        )

    class MovingStore:
        listed_at = None

        def status(self):
            return SimpleNamespace(index_revision=11)

        def list_revisions(self, *, index_revision, **_):
            self.listed_at = index_revision
            return []

    store = MovingStore()
    resolved = resolve_history(
        history_policy="prefer",
        query_features=CaseFeatureSet(),
        store_factory=lambda: store,
    )
    assert resolved.index_revision == 11
    assert resolved.trace.index_revision == 11
    assert resolved.trace.query.index_revision == 11
    assert store.listed_at == 11


def test_bounded_stream_is_lazy_deterministic_and_gives_both_sources_a_budget():
    from llm_matgen.adsorption.proposals import bounded_proposal_stream

    consumed = []

    def algorithmic():
        for site in ("s1", "s2", "s1", "s2"):
            consumed.append(f"a:{site}")
            yield SimpleNamespace(site_id=site, source="algorithmic")

    def historical():
        while True:
            consumed.append("h")
            yield SimpleNamespace(site_id="s1", source="retrieved")

    values, audit = bounded_proposal_stream(
        algorithmic(),
        historical(),
        max_attempts=3,
        max_accepted=3,
        accept=lambda _: True,
    )

    assert [(item.source, item.site_id) for item in values] == [
        ("algorithmic", "s1"),
        ("retrieved", "s1"),
        ("algorithmic", "s2"),
    ]
    assert consumed == ["a:s1", "h", "a:s2"]
    assert audit.attempted == 3
    assert audit.truncated
