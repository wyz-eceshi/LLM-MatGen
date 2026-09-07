from __future__ import annotations

from datetime import datetime, timezone


def make_revision(name: str, *, features=None, status="eligible", equivalent=None, quality=1.0, steps=10):
    from llm_matgen.adsorption.models import AdsorptionCaseRevision, CaseAudit, CaseFeatureSet, CaseStatus

    features = features or CaseFeatureSet(
        adsorbate_formula="H",
        adsorbate_graph_fingerprint="H:single",
        anchor_element="H",
        denticity=1,
        active_site_elements=["Pt"],
        local_coordination_signature="Pt-top",
        first_coordination_shell={"Pt": 1},
        surface_layer_composition={"Pt": 1.0},
        initial_site_type="top",
        local_distance_scale=1.8,
        coverage=0.25,
        miller_index=(1, 1, 1),
        termination="Pt",
    )
    return AdsorptionCaseRevision(
        case_id=name,
        revision_id=f"{name}-r1",
        root_id="r",
        relative_job_dir=f"jobs/{name}",
        exact_hash=(name.encode().hex() * 64)[:64].ljust(64, "0"),
        equivalent_fingerprint=equivalent or name,
        source_signature=f"sig-{name}",
        status=CaseStatus(status),
        original_status=CaseStatus(status),
        audit=CaseAudit(case_quality=quality, ionic_steps=steps),
        features=features,
        artifact_relative_path=f"artifacts/{name}/{name}-r1",
        index_revision=1,
        created_at=datetime.now(timezone.utc),
    )


def test_six_scores_exact_match_and_missing_field_penalty():
    """Break caught: hidden weights or missing data receiving free similarity."""
    from llm_matgen.adsorption.models import CaseFeatureSet, RetrievalQuery
    from llm_matgen.adsorption.retrieval import CaseRetriever

    query_features = make_revision("query").features
    query = RetrievalQuery(index_revision=1, features=query_features, threshold=0.0)
    exact = CaseRetriever().retrieve(query, [make_revision("exact")])
    match = exact.matches[0]
    assert match.total_score == 1.0
    assert match.completeness == 1.0
    assert match.component_scores == {
        "first_coordination_shell": 1.0,
        "surface_layer_composition": 1.0,
        "site_type": 1.0,
        "local_distance_scale": 1.0,
        "coverage": 1.0,
        "miller_termination": 1.0,
    }

    missing_features = CaseFeatureSet(**{**query_features.model_dump(), "coverage": None})
    missing = CaseRetriever().retrieve(query, [make_revision("missing", features=missing_features)]).matches[0]
    assert missing.component_scores["coverage"] is None
    assert missing.completeness == 0.9
    assert missing.total_score == 0.9


def test_strict_filters_threshold_top_five_uniqueness_and_determinism():
    """Break caught: chemically incompatible or copied cases entering fallback."""
    from llm_matgen.adsorption.models import CaseFeatureSet, RetrievalQuery
    from llm_matgen.adsorption.retrieval import CaseRetriever

    base = make_revision("query").features
    cases = [make_revision(f"case-{i}", equivalent=f"eq-{i}") for i in range(7)]
    cases.append(make_revision("copy", equivalent="eq-0"))
    cases.append(make_revision("duplicate", status="duplicate"))
    cases.append(make_revision("self"))
    incompatible = CaseFeatureSet(**{**base.model_dump(), "active_site_elements": ["Au"]})
    cases.append(make_revision("wrong-site", features=incompatible))
    wrong_graph = CaseFeatureSet(**{**base.model_dump(), "adsorbate_graph_fingerprint": "OH"})
    cases.append(make_revision("wrong-graph", features=wrong_graph))

    query = RetrievalQuery(index_revision=1, features=base, exclude_case_id="self", threshold=0.65, limit=5)
    first = CaseRetriever().retrieve(query, list(reversed(cases)))
    second = CaseRetriever().retrieve(query, cases)
    assert [m.case_id for m in first.matches] == [m.case_id for m in second.matches]
    assert len(first.matches) == 5
    assert len({m.equivalent_fingerprint for m in first.matches}) == 5
    assert {item.case_id for item in first.rejected} >= {"copy", "duplicate", "self", "wrong-site", "wrong-graph"}


def test_quality_and_ionic_efficiency_only_break_equal_score_ties():
    """Break caught: operational quality silently changing chemical similarity."""
    from llm_matgen.adsorption.models import RetrievalQuery
    from llm_matgen.adsorption.retrieval import CaseRetriever

    query = RetrievalQuery(index_revision=1, features=make_revision("q").features, threshold=0.0)
    low = make_revision("low", quality=0.5, steps=20)
    high = make_revision("high", quality=0.9, steps=8)
    trace = CaseRetriever().retrieve(query, [low, high])
    assert [item.case_id for item in trace.matches] == ["high", "low"]
    assert trace.matches[0].total_score == trace.matches[1].total_score == 1.0


def test_missing_required_hard_filter_fields_never_match_each_other():
    """Break caught: unknown chemistry being treated as strict compatibility."""
    from llm_matgen.adsorption.models import CaseFeatureSet, RetrievalQuery
    from llm_matgen.adsorption.retrieval import CaseRetriever

    incomplete = CaseFeatureSet(denticity=1)
    trace = CaseRetriever().retrieve(
        RetrievalQuery(index_revision=1, features=incomplete, threshold=0.0),
        [make_revision("unknown", features=incomplete)],
    )
    assert trace.matches == []
    assert set(trace.rejected[0].rejection_reasons) >= {
        "adsorbate_graph_missing", "anchor_element_missing",
        "active_site_elements_missing", "local_coordination_missing",
    }
