"""Deterministic, explainable non-ML retrieval for adsorption cases."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from llm_matgen.adsorption.models import (
    AdsorptionCaseRevision,
    CaseStatus,
    RetrievalMatch,
    RetrievalQuery,
    RetrievalTrace,
)

SCORE_WEIGHTS = {
    "first_coordination_shell": 0.30,
    "surface_layer_composition": 0.20,
    "site_type": 0.15,
    "local_distance_scale": 0.15,
    "coverage": 0.10,
    "miller_termination": 0.10,
}


def _distribution_similarity(
    left: Mapping[str, float | int] | None,
    right: Mapping[str, float | int] | None,
) -> float | None:
    if left is None or right is None:
        return None
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    if left_total <= 0 or right_total <= 0:
        return 1.0 if left_total == right_total else 0.0
    distance = sum(
        abs(float(left.get(key, 0)) / left_total - float(right.get(key, 0)) / right_total)
        for key in keys
    )
    return max(0.0, 1.0 - 0.5 * distance)


def _categorical(left: str | None, right: str | None) -> float | None:
    if left is None or right is None:
        return None
    return 1.0 if left == right else 0.0


def _scale(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = max(abs(left), abs(right), 1.0e-12)
    return max(0.0, 1.0 - abs(left - right) / denominator)


def _miller_termination(query, candidate) -> float | None:
    if (
        query.miller_index is None
        or candidate.miller_index is None
        or query.termination is None
        or candidate.termination is None
    ):
        return None
    miller = 1.0 if tuple(query.miller_index) == tuple(candidate.miller_index) else 0.0
    termination = 1.0 if query.termination == candidate.termination else 0.0
    return 0.5 * (miller + termination)


def _components(query, candidate) -> dict[str, float | None]:
    return {
        "first_coordination_shell": _distribution_similarity(
            query.first_coordination_shell, candidate.first_coordination_shell
        ),
        "surface_layer_composition": _distribution_similarity(
            query.surface_layer_composition, candidate.surface_layer_composition
        ),
        "site_type": _categorical(query.initial_site_type, candidate.initial_site_type),
        "local_distance_scale": _scale(
            query.local_distance_scale, candidate.local_distance_scale
        ),
        "coverage": _scale(query.coverage, candidate.coverage),
        "miller_termination": _miller_termination(query, candidate),
    }


def _score(components: Mapping[str, float | None]) -> tuple[float, float]:
    score = sum(
        SCORE_WEIGHTS[name] * value
        for name, value in components.items()
        if value is not None
    )
    completeness = sum(
        SCORE_WEIGHTS[name] for name, value in components.items() if value is not None
    )
    return round(score, 12), round(completeness, 12)


def _hard_reasons(query: RetrievalQuery, case: AdsorptionCaseRevision) -> list[str]:
    reasons: list[str] = []
    features = case.features
    wanted = query.features
    if case.status is not CaseStatus.ELIGIBLE:
        reasons.append(f"status:{case.status.value}")
    if case.index_revision > query.index_revision:
        reasons.append("outside_index_revision")
    if (
        case.superseded_index_revision is not None
        and case.superseded_index_revision <= query.index_revision
    ):
        reasons.append("superseded")
    if (
        case.inactive_index_revision is not None
        and case.inactive_index_revision <= query.index_revision
    ):
        reasons.append("inactive")
    if query.exclude_case_id and case.case_id == query.exclude_case_id:
        reasons.append("query_self")
    if features.adsorbate_graph_fingerprint is None or wanted.adsorbate_graph_fingerprint is None:
        reasons.append("adsorbate_graph_missing")
    elif features.adsorbate_graph_fingerprint != wanted.adsorbate_graph_fingerprint:
        reasons.append("adsorbate_graph_mismatch")
    if features.anchor_element is None or wanted.anchor_element is None:
        reasons.append("anchor_element_missing")
    elif features.anchor_element != wanted.anchor_element:
        reasons.append("anchor_element_mismatch")
    if features.denticity != 1 or wanted.denticity != 1:
        reasons.append("unsupported_denticity")
    if features.active_site_elements is None or wanted.active_site_elements is None:
        reasons.append("active_site_elements_missing")
    elif features.active_site_elements != wanted.active_site_elements:
        reasons.append("active_site_elements_mismatch")
    if features.local_coordination_signature is None or wanted.local_coordination_signature is None:
        reasons.append("local_coordination_missing")
    elif features.local_coordination_signature != wanted.local_coordination_signature:
        reasons.append("local_coordination_mismatch")
    return reasons


def _decision(
    case: AdsorptionCaseRevision,
    *,
    components: dict[str, float | None] | None = None,
    total_score: float = 0.0,
    completeness: float = 0.0,
    accepted: bool,
    reasons: list[str] | None = None,
) -> RetrievalMatch:
    return RetrievalMatch(
        case_id=case.case_id,
        revision_id=case.revision_id,
        equivalent_fingerprint=case.equivalent_fingerprint,
        total_score=total_score,
        component_scores=components
        or {name: None for name in SCORE_WEIGHTS},
        completeness=completeness,
        accepted=accepted,
        rejection_reasons=reasons or [],
        evidence={
            "relative_job_dir": case.relative_job_dir,
            "case_quality": case.audit.case_quality,
            "ionic_steps": case.audit.ionic_steps,
        },
    )


class CaseRetriever:
    """Apply strict chemical filters followed by six fixed weighted scores."""

    def retrieve(
        self,
        query: RetrievalQuery,
        cases: Iterable[AdsorptionCaseRevision],
    ) -> RetrievalTrace:
        rejected: list[RetrievalMatch] = []
        scored: list[tuple[RetrievalMatch, AdsorptionCaseRevision]] = []
        for case in cases:
            reasons = _hard_reasons(query, case)
            if reasons:
                rejected.append(_decision(case, accepted=False, reasons=reasons))
                continue
            components = _components(query.features, case.features)
            total, completeness = _score(components)
            if total < query.threshold:
                rejected.append(
                    _decision(
                        case,
                        components=components,
                        total_score=total,
                        completeness=completeness,
                        accepted=False,
                        reasons=["below_threshold"],
                    )
                )
                continue
            scored.append(
                (
                    _decision(
                        case,
                        components=components,
                        total_score=total,
                        completeness=completeness,
                        accepted=True,
                    ),
                    case,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0].total_score,
                -item[0].completeness,
                -(item[1].audit.case_quality if item[1].audit.case_quality is not None else -1.0),
                item[1].audit.ionic_steps if item[1].audit.ionic_steps is not None else 10**12,
                item[1].case_id,
                item[1].revision_id,
            )
        )
        unique: list[RetrievalMatch] = []
        fingerprints: set[str] = set()
        for match, case in scored:
            if case.equivalent_fingerprint in fingerprints:
                rejected.append(
                    match.model_copy(
                        update={"accepted": False, "rejection_reasons": ["staging_duplicate"]}
                    )
                )
                continue
            fingerprints.add(case.equivalent_fingerprint)
            if len(unique) < query.limit:
                unique.append(match)
            else:
                rejected.append(
                    match.model_copy(
                        update={"accepted": False, "rejection_reasons": ["outside_top_limit"]}
                    )
                )
        rejected.sort(key=lambda item: (item.case_id, item.revision_id))
        if unique:
            fallback_reason = None
        elif scored:
            fallback_reason = "no_unique_case"
        elif any(item.rejection_reasons == ["below_threshold"] for item in rejected):
            fallback_reason = "below_threshold"
        else:
            fallback_reason = "no_strictly_compatible_case"
        return RetrievalTrace(
            index_revision=query.index_revision,
            query=query,
            matches=unique,
            rejected=rejected,
            fallback_reason=fallback_reason,
        )
