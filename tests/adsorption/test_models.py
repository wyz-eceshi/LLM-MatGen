from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


def test_contracts_forbid_extra_normalize_utc_and_reject_non_finite():
    """Break caught: permissive or non-portable case JSON entering the index."""
    from llm_matgen.adsorption.models import CaseAudit, CaseFeatureSet, FileEvidence

    evidence = FileEvidence(
        relative_path="POSCAR",
        sha256="a" * 64,
        size=10,
        mtime_ns=1,
        captured_at=datetime.now(timezone.utc),
    )
    assert evidence.captured_at.utcoffset().total_seconds() == 0
    with pytest.raises(ValidationError, match="Extra inputs"):
        evidence.__class__(**{**evidence.model_dump(), "unknown": 1})
    with pytest.raises(ValidationError, match="finite"):
        CaseFeatureSet(local_distance_scale=float("nan"))
    with pytest.raises(ValidationError, match="timezone-aware"):
        FileEvidence(
            relative_path="POSCAR",
            sha256="a" * 64,
            size=10,
            mtime_ns=1,
            captured_at=datetime.now(),
        )
    assert CaseAudit().extractor_version


def test_revision_is_frozen_and_round_trips_json():
    """Break caught: mutating an already published historical revision."""
    from llm_matgen.adsorption.models import (
        AdsorptionCaseRevision,
        CaseAudit,
        CaseFeatureSet,
        CaseStatus,
    )

    revision = AdsorptionCaseRevision(
        case_id="case-1",
        revision_id="rev-1",
        root_id="r",
        relative_job_dir="jobs/one",
        exact_hash="a" * 64,
        equivalent_fingerprint="eq-1",
        source_signature="sig-1",
        status=CaseStatus.ELIGIBLE,
        original_status=CaseStatus.ELIGIBLE,
        audit=CaseAudit(),
        features=CaseFeatureSet(adsorbate_formula="H", adsorbate_graph_fingerprint="H"),
        artifact_relative_path="artifacts/case-1/rev-1",
        index_revision=1,
        created_at=datetime.now(timezone.utc),
    )
    restored = AdsorptionCaseRevision.model_validate_json(revision.model_dump_json())
    assert restored == revision
    with pytest.raises(ValidationError, match="frozen"):
        revision.status = CaseStatus.DUPLICATE


def test_version_provenance_has_central_contract_versions():
    """Break caught: artifacts that cannot identify their extraction contract."""
    from llm_matgen.adsorption.models import CASE_SCHEMA_VERSION, version_provenance

    provenance = version_provenance()
    assert CASE_SCHEMA_VERSION == 1
    assert provenance["case_schema_version"] == 1
    assert provenance["extractor_version"]
    assert provenance["feature_version"]
    assert provenance["adsorption_db_schema_version"] == 3
