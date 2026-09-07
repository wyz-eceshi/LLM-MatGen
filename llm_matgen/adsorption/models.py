"""Strict, JSON-safe contracts for historical adsorption cases."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

from llm_matgen.generators.models import JsonValue

CASE_SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "1.0.0"
FEATURE_VERSION = "1.0.0"


def _require_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("all numeric values must be finite")
    if isinstance(value, dict):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite(item)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _relative_posix(value: str) -> str:
    if not value or "\x00" in value or "\n" in value or "\r" in value or "\\" in value:
        raise ValueError("path must be a safe non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("path must remain relative")
    return path.as_posix()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _finite_values(self):
        _require_finite(self.model_dump(mode="python"))
        return self


class CaseStatus(str, Enum):
    ELIGIBLE = "eligible"
    REJECTED_INCOMPLETE = "rejected_incomplete"
    REJECTED_ELECTRONIC_UNCONVERGED = "rejected_electronic_unconverged"
    REJECTED_IONIC_UNCONVERGED = "rejected_ionic_unconverged"
    REJECTED_FATAL_WARNING = "rejected_fatal_warning"
    REJECTED_AMBIGUOUS_MAPPING = "rejected_ambiguous_mapping"
    DUPLICATE = "duplicate"
    SUPERSEDED_RESTART = "superseded_restart"


class FileEvidence(StrictModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    captured_at: datetime
    stable: bool = True
    content_stored: bool = False
    parsed_evidence: dict[str, JsonValue] = Field(default_factory=dict)
    markers: dict[str, JsonValue] = Field(default_factory=dict)

    _normalize_path = field_validator("relative_path")(_relative_posix)
    _normalize_time = field_validator("captured_at")(_utc)


class CaseAudit(StrictModel):
    files_complete: bool | None = None
    stable: bool | None = None
    ionic_relaxation: bool | None = None
    electronic_converged: bool | None = None
    ionic_converged: bool | None = None
    normal_termination: bool | None = None
    fatal_warning_free: bool | None = None
    composition_match: bool | None = None
    mapping_unique: bool | None = None
    clean_parent_found: bool | None = None
    partition_unique: bool | None = None
    vacuum_axis_unique: bool | None = None
    adsorbate_intact: bool | None = None
    not_desorbed: bool | None = None
    not_subsurface: bool | None = None
    surface_reconstruction_acceptable: bool | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    run_type: str | None = None
    electronic_evidence: dict[str, JsonValue] = Field(default_factory=dict)
    ionic_evidence: dict[str, JsonValue] = Field(default_factory=dict)
    termination_evidence: dict[str, JsonValue] = Field(default_factory=dict)
    mapping_confidence: str | None = None
    mapping_cost: float | None = None
    mapping_cost_gap: float | None = None
    raw_internal_bond_graph: dict[str, JsonValue] | None = None
    ionic_steps: int | None = Field(default=None, ge=0)
    case_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    extractor_version: str = EXTRACTOR_VERSION
    schema_version: int = CASE_SCHEMA_VERSION
    feature_version: str = FEATURE_VERSION


class CaseFeatureSet(StrictModel):
    adsorbate_formula: str | None = None
    adsorbate_graph_fingerprint: str | None = None
    anchor_element: str | None = None
    denticity: PositiveInt | None = None
    active_site_elements: list[str] | None = None
    local_coordination_signature: str | None = None
    surface_side: str | None = None
    surface_normal: tuple[float, float, float] | None = None
    vacuum_axis: int | None = Field(default=None, ge=0, le=2)
    miller_index: tuple[int, int, int] | None = None
    termination: str | None = None
    initial_height: float | None = None
    final_height: float | None = None
    inplane_displacement: tuple[float, float] | None = None
    initial_tilt: float | None = None
    final_tilt: float | None = None
    initial_azimuth: float | None = None
    final_azimuth: float | None = None
    rotation: float | None = None
    internal_bond_graph: dict[str, JsonValue] | None = None
    initial_site_type: str | None = None
    final_site_type: str | None = None
    site_transition: str | None = None
    coordinated_surface_site_ids: list[str] | None = None
    first_coordination_shell: dict[str, float | int] | None = None
    second_coordination_shell: dict[str, float | int] | None = None
    surface_layer_composition: dict[str, float] | None = None
    local_distance_scale: float | None = None
    coverage: float | None = Field(default=None, ge=0.0)
    surface_area: float | None = Field(default=None, gt=0.0)
    periodic_image_distance: float | None = Field(default=None, gt=0.0)
    slab_drift: tuple[float, float, float] | None = None
    slab_reconstruction: bool | None = None
    dissociation: bool | None = None
    desorption: bool | None = None
    subsurface: bool | None = None
    pose_correction: dict[str, JsonValue] | None = None

    @field_validator("active_site_elements")
    @classmethod
    def _sort_elements(cls, value: list[str] | None) -> list[str] | None:
        return sorted(value) if value is not None else None


class AdsorptionCaseRevision(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    revision_id: str
    root_id: str
    relative_job_dir: str
    clean_parent_identity: str | None = None
    exact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    equivalent_fingerprint: str
    source_signature: str
    status: CaseStatus
    original_status: CaseStatus
    audit: CaseAudit
    features: CaseFeatureSet
    evidence: list[FileEvidence] = Field(default_factory=list)
    artifact_relative_path: str
    index_revision: int = Field(ge=1)
    created_at: datetime
    inactive_index_revision: int | None = Field(default=None, ge=1)
    superseded_index_revision: int | None = Field(default=None, ge=1)
    superseded_by_revision_id: str | None = None
    duplicate_of_revision_id: str | None = None

    _normalize_job_dir = field_validator("relative_job_dir")(_relative_posix)
    _normalize_artifact = field_validator("artifact_relative_path")(_relative_posix)
    _normalize_created = field_validator("created_at")(_utc)


class RetrievalQuery(StrictModel):
    index_revision: int = Field(ge=0)
    features: CaseFeatureSet
    exclude_case_id: str | None = None
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=5)


class RetrievalMatch(StrictModel):
    case_id: str
    revision_id: str
    equivalent_fingerprint: str
    total_score: float = Field(ge=0.0, le=1.0)
    component_scores: dict[str, float | None]
    completeness: float = Field(ge=0.0, le=1.0)
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


class RetrievalTrace(StrictModel):
    index_revision: int = Field(ge=0)
    query: RetrievalQuery
    matches: list[RetrievalMatch] = Field(default_factory=list)
    rejected: list[RetrievalMatch] = Field(default_factory=list)
    fallback_reason: str | None = None


class StoreStatus(StrictModel):
    root: str
    schema_version: int
    index_revision: int = Field(ge=0)
    last_scan_status: str | None = None


def version_provenance() -> dict[str, int | str]:
    from llm_matgen.adsorption.schema import ADSORPTION_DB_SCHEMA_VERSION

    return {
        "case_schema_version": CASE_SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "feature_version": FEATURE_VERSION,
        "adsorption_db_schema_version": ADSORPTION_DB_SCHEMA_VERSION,
    }
