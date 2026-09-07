"""Deterministic adsorption proposal contracts and rigid-body placement helpers."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Callable, Iterable, Iterator, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError as PydanticValidationError,
    field_validator,
    model_validator,
)
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule, Structure
from scipy.spatial import QhullError, Voronoi

from llm_matgen.adsorption.models import (
    AdsorptionCaseRevision,
    CaseFeatureSet,
    RetrievalMatch,
    RetrievalQuery,
    RetrievalTrace,
)
from llm_matgen.adsorption.retrieval import CaseRetriever
from llm_matgen.adsorption.store import AdsorptionStoreError


def _unit(vector, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


class LocalFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin: tuple[float, float, float]
    tangent1: tuple[float, float, float]
    tangent2: tuple[float, float, float]
    normal: tuple[float, float, float]

    @field_validator("origin", "tangent1", "tangent2", "normal")
    @classmethod
    def _finite_triplet(cls, value):
        array = np.asarray(value, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("frame vectors must contain three finite values")
        return tuple(float(item) for item in array)

    @model_validator(mode="after")
    def _orthonormal(self):
        t1 = np.asarray(self.tangent1)
        t2 = np.asarray(self.tangent2)
        normal = np.asarray(self.normal)
        matrix = np.vstack([t1, t2, normal])
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-8):
            raise ValueError("local frame must be orthonormal")
        if not np.allclose(np.cross(t1, t2), normal, atol=1.0e-8):
            raise ValueError("local frame must be right handed")
        return self


class PoseTransform(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    translation: tuple[float, float, float]
    azimuth: float
    tilt: float
    roll: float


def build_surface_frame(
    slab: Structure,
    side: str = "top",
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LocalFrame:
    """Build a real-space frame from the first two lattice vectors and vacuum sign."""

    if side not in {"top", "bottom"}:
        raise ValueError("surface side must be top or bottom")
    lattice = np.asarray(slab.lattice.matrix, dtype=float)
    normal = _unit(np.cross(lattice[0], lattice[1]), name="surface normal")
    if float(np.dot(normal, lattice[2])) < 0.0:
        normal *= -1.0
    if side == "bottom":
        normal *= -1.0
    tangent1 = None
    for vector in lattice:
        projected = vector - float(np.dot(vector, normal)) * normal
        if float(np.linalg.norm(projected)) > 1.0e-10:
            tangent1 = _unit(projected, name="surface tangent")
            break
    if tangent1 is None:
        basis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        tangent1 = _unit(basis - float(np.dot(basis, normal)) * normal, name="fallback tangent")
    tangent2 = _unit(np.cross(normal, tangent1), name="second tangent")
    return LocalFrame(
        origin=origin,
        tangent1=tuple(float(item) for item in tangent1),
        tangent2=tuple(float(item) for item in tangent2),
        normal=tuple(float(item) for item in normal),
    )


def _axis_angle(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = _unit(axis, name="rotation axis")
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle_radians) * cross + (1.0 - math.cos(angle_radians)) * (cross @ cross)


def _align_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _unit(source, name="reference axis")
    target = _unit(target, name="target axis")
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine > 1.0e-12:
        return _axis_angle(cross / sine, math.atan2(sine, cosine))
    if cosine > 0.0:
        return np.eye(3)
    basis = np.eye(3)[int(np.argmin(np.abs(source)))]
    axis = _unit(np.cross(source, basis), name="antiparallel rotation axis")
    return _axis_angle(axis, math.pi)


def place_adsorbate_rigid(
    molecule: Molecule,
    anchor_zero_based: int,
    reference_axis: tuple[float, float, float] | list[float],
    frame: LocalFrame,
    height: float,
    azimuth: float,
    tilt: float,
    roll: float = 0.0,
) -> tuple[Molecule, PoseTransform]:
    """Place a molecule explicitly around its declared anchor without z heuristics."""

    if not 0 <= anchor_zero_based < len(molecule):
        raise ValueError("anchor index is outside adsorbate")
    if not all(math.isfinite(float(item)) for item in (height, azimuth, tilt, roll)):
        raise ValueError("pose values must be finite")
    normal = np.asarray(frame.normal)
    tangent1 = np.asarray(frame.tangent1)
    tangent2 = np.asarray(frame.tangent2)
    azimuth_radians = math.radians(float(azimuth))
    tilt_radians = math.radians(float(tilt))
    inplane = math.cos(azimuth_radians) * tangent1 + math.sin(azimuth_radians) * tangent2
    target_axis = math.cos(tilt_radians) * normal + math.sin(tilt_radians) * inplane
    aligned = _align_rotation(np.asarray(reference_axis, dtype=float), target_axis)
    rolled = _axis_angle(target_axis, math.radians(float(roll))) @ aligned if abs(roll) > 1.0e-15 else aligned
    coordinates = np.asarray(molecule.cart_coords, dtype=float)
    relative = coordinates - coordinates[anchor_zero_based]
    target = np.asarray(frame.origin, dtype=float) + float(height) * normal
    placed_coordinates = relative @ rolled.T + target
    placed = Molecule(
        [site.specie for site in molecule],
        placed_coordinates,
        charge=molecule.charge,
        spin_multiplicity=molecule.spin_multiplicity,
        site_properties=molecule.site_properties,
    )
    translation = target - coordinates[anchor_zero_based] @ rolled.T
    transform = PoseTransform(
        rotation=tuple(tuple(float(item) for item in row) for row in rolled),
        translation=tuple(float(item) for item in translation),
        azimuth=float(azimuth),
        tilt=float(tilt),
        roll=float(roll),
    )
    return placed, transform


@runtime_checkable
class ProposalSource(Protocol):
    def proposals(self) -> Iterator[object]: ...


class AlgorithmicProposalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    surface_atom_indices: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


class RetrievedProposalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    revision_id: str
    index_revision: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    mode: Literal["final_pose", "local_delta"]
    slab_indices: tuple[int, ...]
    adsorbate_indices: tuple[int, ...]


class _RetrievedPoseCorrection(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    slab_indices: tuple[int, ...]
    adsorbate_indices: tuple[int, ...]
    initial_local_coordinates: tuple[tuple[float, float, float], ...]
    final_local_coordinates: tuple[tuple[float, float, float], ...]
    local_delta: tuple[float, float, float]
    rotation_degrees: float | None = None

    @model_validator(mode="after")
    def _consistent_rows(self):
        if not self.slab_indices:
            raise ValueError("pose correction slab_indices must not be empty")
        if not self.adsorbate_indices:
            raise ValueError("pose correction adsorbate_indices must not be empty")
        expected = len(self.adsorbate_indices)
        if (
            len(self.initial_local_coordinates) != expected
            or len(self.final_local_coordinates) != expected
        ):
            raise ValueError(
                "pose correction coordinate rows must match adsorbate_indices"
            )
        numeric = [
            *self.local_delta,
            *(item for row in self.initial_local_coordinates for item in row),
            *(item for row in self.final_local_coordinates for item in row),
        ]
        if self.rotation_degrees is not None:
            numeric.append(self.rotation_degrees)
        if not all(math.isfinite(float(item)) for item in numeric):
            raise ValueError("pose correction numeric values must be finite")
        if any(item < 0 for item in (*self.slab_indices, *self.adsorbate_indices)):
            raise ValueError("pose correction atom indices must be non-negative")
        return self


class AdsorptionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    source: Literal["algorithmic", "retrieved"]
    site_id: str
    site_type: str
    surface_side: Literal["top", "bottom"]
    frame: LocalFrame
    height: float
    azimuth: float
    tilt: float
    roll: float = 0.0
    coverage: float | None = Field(default=None, gt=0.0)
    evidence: AlgorithmicProposalEvidence | RetrievedProposalEvidence
    local_adsorbate_coordinates: tuple[tuple[float, float, float], ...] | None = None
    local_delta: tuple[float, float, float] | None = None
    rotation_degrees: float | None = None

    @model_validator(mode="after")
    def _finite_pose(self):
        values = [self.height, self.azimuth, self.tilt, self.roll]
        if self.rotation_degrees is not None:
            values.append(self.rotation_degrees)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("proposal pose values must be finite")
        return self


def _surface_indices(slab: Structure, normal: np.ndarray, side: str) -> tuple[int, ...]:
    projections = np.asarray(slab.cart_coords, dtype=float) @ normal
    outward = projections if side == "top" else -projections
    highest = float(np.max(outward))
    return tuple(int(index) for index, value in enumerate(outward) if highest - float(value) <= 0.35)


def _canonical_position(slab: Structure, cartesian) -> tuple[float, float, float]:
    fractional = np.asarray(slab.lattice.get_fractional_coords(cartesian), dtype=float) % 1.0
    cartesian_inside = fractional @ np.asarray(slab.lattice.matrix, dtype=float)
    return tuple(float(item) for item in cartesian_inside)


def _deduplicated_positions(slab: Structure, positions: Iterable[object]) -> list[tuple[float, float, float]]:
    unique: dict[tuple[float, float, float], tuple[float, float, float]] = {}
    for position in positions:
        canonical = _canonical_position(slab, position)
        key = tuple(round(item, 8) for item in canonical)
        unique.setdefault(key, canonical)
    return [unique[key] for key in sorted(unique)]


def _site_id(side: str, site_type: str, index: int, origin) -> str:
    digest = sha256(
        f"{side}|{site_type}|".encode()
        + np.asarray(origin, dtype="<f8").round(8).tobytes()
    ).hexdigest()[:12]
    return f"{side}:{site_type}:{index:04d}:{digest}"


class AlgorithmicProposalSource:
    """Finite deterministic site discovery with lazy pose expansion."""

    def __init__(
        self,
        slab: Structure,
        *,
        sides: tuple[str, ...] = ("top",),
        site_types: tuple[str, ...] = ("top", "bridge", "hollow", "hollow4"),
        explicit_sites: tuple[tuple[float, float, float], ...] = (),
        heights: tuple[float, ...] = (1.8,),
        azimuths: tuple[float, ...] = (0.0,),
        tilts: tuple[float, ...] = (0.0,),
        rolls: tuple[float, ...] = (0.0,),
        coverage: float | None = None,
    ):
        if not sides or any(side not in {"top", "bottom"} for side in sides):
            raise ValueError("sides must contain top and/or bottom")
        self.slab = slab.copy()
        self.sides = tuple(dict.fromkeys(sides))
        self.site_types = tuple(dict.fromkeys(site_types))
        self.explicit_sites = explicit_sites
        self.heights = heights
        self.azimuths = azimuths
        self.tilts = tilts
        self.rolls = rolls
        self.coverage = coverage

    def _asf_sites(self, side: str) -> dict[str, list[tuple[float, float, float]]]:
        base = build_surface_frame(self.slab, side)
        working = self.slab.copy()
        indices = set(_surface_indices(working, np.asarray(base.normal), "top"))
        working.add_site_property(
            "surface_properties",
            ["surface" if index in indices else "subsurface" for index in range(len(working))],
        )
        requested = []
        for site_type in self.site_types:
            name = "ontop" if site_type == "top" else site_type
            if name in {"ontop", "bridge", "hollow"} and name not in requested:
                requested.append(name)
        if not requested:
            return {}
        finder = AdsorbateSiteFinder(
            working,
            height=0.9,
            mi_vec=tuple(float(item) for item in base.normal),
        )
        found = finder.find_adsorption_sites(
            distance=0.0,
            positions=tuple(requested),
            symm_reduce=0.01,
            near_reduce=0.01,
        )
        return {
            "top" if name == "ontop" else name: _deduplicated_positions(self.slab, values)
            for name, values in found.items()
            if name != "all"
        }

    def _hollow4(self, side: str) -> list[tuple[float, float, float]]:
        frame = build_surface_frame(self.slab, side)
        indices = _surface_indices(self.slab, np.asarray(frame.normal), "top")
        if len(indices) < 4:
            return []
        lattice = np.asarray(self.slab.lattice.matrix, dtype=float)
        tangent1 = np.asarray(frame.tangent1)
        tangent2 = np.asarray(frame.tangent2)
        basis = np.array(
            [
                [np.dot(lattice[0], tangent1), np.dot(lattice[0], tangent2)],
                [np.dot(lattice[1], tangent1), np.dot(lattice[1], tangent2)],
            ]
        )
        if abs(float(np.linalg.det(basis))) <= 1.0e-12:
            return []
        inverse_basis = np.linalg.inv(basis)
        fractional = np.asarray(self.slab.frac_coords, dtype=float)[list(indices), :2] % 1.0
        expanded = np.array(
            [
                (uv + np.asarray(shift, dtype=float)) @ basis
                for uv in fractional
                for shift in product((-1, 0, 1), repeat=2)
            ]
        )
        try:
            vertices = Voronoi(expanded).vertices
        except QhullError:
            return []

        normal = np.asarray(frame.normal)
        surface_height = float(
            np.mean(np.asarray(self.slab.cart_coords)[list(indices)] @ normal)
        )
        positions = []
        for vertex in vertices:
            uv = vertex @ inverse_basis
            if not np.all((-1.0e-8 <= uv) & (uv < 1.0 - 1.0e-8)):
                continue
            neighbour_vectors = []
            neighbour_distances = []
            for atom_uv in fractional:
                images = np.array(
                    [
                        (atom_uv + np.asarray(shift, dtype=float) - uv) @ basis
                        for shift in product((-1, 0, 1), repeat=2)
                    ]
                )
                norms = np.linalg.norm(images, axis=1)
                nearest = int(np.argmin(norms))
                neighbour_vectors.append(images[nearest])
                neighbour_distances.append(float(norms[nearest]))
            order = np.argsort(neighbour_distances)
            first_four = np.asarray(neighbour_distances)[order[:4]]
            fourth = float(first_four[-1])
            if fourth <= 1.0e-8 or float(np.ptp(first_four)) > max(1.0e-6, 0.08 * fourth):
                continue
            if len(order) > 4 and neighbour_distances[int(order[4])] <= 1.10 * fourth:
                continue
            angles = np.sort(
                np.arctan2(
                    np.asarray(neighbour_vectors)[order[:4], 1],
                    np.asarray(neighbour_vectors)[order[:4], 0],
                )
            )
            gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi])
            if float(np.max(gaps)) >= 0.9 * math.pi:
                continue
            cartesian = uv @ lattice[:2]
            cartesian = cartesian + (surface_height - float(np.dot(cartesian, normal))) * normal
            positions.append(cartesian)
        return _deduplicated_positions(self.slab, positions)

    def _environment_sites(self, side: str, site_type: str) -> list[tuple[float, float, float]]:
        frame = build_surface_frame(self.slab, side)
        indices = _surface_indices(self.slab, np.asarray(frame.normal), "top")
        if not indices:
            return []
        species = [str(self.slab[index].specie) for index in indices]
        if site_type == "doped":
            counts = {symbol: species.count(symbol) for symbol in set(species)}
            majority = max(counts.values())
            chosen = [index for index in indices if counts[str(self.slab[index].specie)] < majority]
        else:
            coordination = {}
            for index in indices:
                coordination[index] = sum(
                    1
                    for other in range(len(self.slab))
                    if other != index and self.slab.get_distance(index, other) <= 3.2
                )
            minimum = min(coordination.values())
            chosen = [index for index in indices if coordination[index] == minimum]
        return _deduplicated_positions(self.slab, [self.slab[index].coords for index in chosen])

    def _sites(self) -> list[tuple[str, str, tuple[float, float, float], AlgorithmicProposalEvidence]]:
        sites = []
        for side in self.sides:
            surface_frame = build_surface_frame(self.slab, side)
            surface_atoms = _surface_indices(self.slab, np.asarray(surface_frame.normal), "top")
            asf = self._asf_sites(side)
            for site_type in ("top", "bridge", "hollow"):
                if site_type not in self.site_types:
                    continue
                for index, position in enumerate(asf.get(site_type, [])):
                    sites.append(
                        (
                            side,
                            site_type,
                            position,
                            AlgorithmicProposalEvidence(
                                method="pymatgen-asf", surface_atom_indices=surface_atoms
                            ),
                        )
                    )
            if "hollow4" in self.site_types:
                for position in self._hollow4(side):
                    sites.append(
                        (
                            side,
                            "hollow4",
                            position,
                            AlgorithmicProposalEvidence(
                                method="fourfold-centroid", surface_atom_indices=surface_atoms
                            ),
                        )
                    )
            for site_type in ("defect", "doped", "undercoordinated"):
                if site_type in self.site_types:
                    for position in self._environment_sites(side, site_type):
                        sites.append(
                            (
                                side,
                                site_type,
                                position,
                                AlgorithmicProposalEvidence(
                                    method="local-environment", surface_atom_indices=surface_atoms
                                ),
                            )
                        )
            if "explicit" in self.site_types or self.explicit_sites:
                for position in _deduplicated_positions(self.slab, self.explicit_sites):
                    sites.append(
                        (
                            side,
                            "explicit",
                            position,
                            AlgorithmicProposalEvidence(method="user-explicit"),
                        )
                    )
        typed = []
        counters: dict[tuple[str, str], int] = {}
        for side, site_type, position, evidence in sites:
            key = (side, site_type)
            index = counters.get(key, 0)
            counters[key] = index + 1
            typed.append((_site_id(side, site_type, index, position), side, site_type, position, evidence))
        return typed

    def proposals(self) -> Iterator[AdsorptionProposal]:
        sites = self._sites()
        for pose in product(self.heights, self.azimuths, self.tilts, self.rolls):
            for site_id, side, site_type, position, evidence in sites:
                height, azimuth, tilt, roll = pose
                frame = build_surface_frame(self.slab, side, origin=position)
                yield AdsorptionProposal(
                    proposal_id=(
                        f"algorithmic:{site_id}:h{height:g}:a{azimuth:g}:t{tilt:g}:r{roll:g}"
                    ),
                    source="algorithmic",
                    site_id=site_id,
                    site_type=site_type,
                    surface_side=side,
                    frame=frame,
                    height=height,
                    azimuth=azimuth,
                    tilt=tilt,
                    roll=roll,
                    coverage=self.coverage,
                    evidence=evidence,
                )


class RetrievedProposalSource:
    def __init__(
        self,
        base_proposals: Iterable[AdsorptionProposal],
        matches: Iterable[RetrievalMatch],
        *,
        revision_loader: Callable[[str], object],
        index_revision: int,
    ):
        self.base_proposals = base_proposals
        self.matches = tuple(matches)
        self.revision_loader = revision_loader
        self.index_revision = index_revision
        self._loaded: dict[str, tuple[object, _RetrievedPoseCorrection]] = {}
        for match in self.matches:
            try:
                revision = self.revision_loader(match.revision_id)
            except LookupError as exc:
                raise ValueError(
                    f"matched history revision {match.revision_id} is unavailable"
                ) from exc
            raw = getattr(getattr(revision, "features", None), "pose_correction", None)
            if not isinstance(raw, dict):
                raise ValueError(
                    f"matched history revision {match.revision_id} has no pose correction"
                )
            try:
                correction = _RetrievedPoseCorrection.model_validate(raw)
            except PydanticValidationError as exc:
                raise ValueError(
                    f"matched history revision {match.revision_id} has invalid pose correction: {exc}"
                ) from exc
            self._loaded[match.revision_id] = (revision, correction)

    def proposals(self) -> Iterator[AdsorptionProposal]:
        for base in self.base_proposals:
            for match in self.matches:
                revision, correction = self._loaded[match.revision_id]
                features = revision.features
                slab_indices = correction.slab_indices
                adsorbate_indices = correction.adsorbate_indices
                initial = correction.initial_local_coordinates
                final = correction.final_local_coordinates
                delta = correction.local_delta
                rotation = correction.rotation_degrees
                common = dict(
                    source="retrieved",
                    site_id=base.site_id,
                    site_type=base.site_type,
                    surface_side=base.surface_side,
                    frame=base.frame,
                    azimuth=base.azimuth,
                    tilt=base.tilt,
                    roll=base.roll,
                    coverage=base.coverage,
                )
                evidence = dict(
                    case_id=match.case_id,
                    revision_id=match.revision_id,
                    index_revision=self.index_revision,
                    score=match.total_score,
                    slab_indices=slab_indices,
                    adsorbate_indices=adsorbate_indices,
                )
                yield AdsorptionProposal(
                    proposal_id=f"retrieved:{match.revision_id}:final:{base.proposal_id}",
                    height=features.final_height if features.final_height is not None else base.height,
                    evidence=RetrievedProposalEvidence(mode="final_pose", **evidence),
                    local_adsorbate_coordinates=final,
                    rotation_degrees=rotation,
                    **common,
                )
                yield AdsorptionProposal(
                    proposal_id=f"retrieved:{match.revision_id}:delta:{base.proposal_id}",
                    height=base.height,
                    evidence=RetrievedProposalEvidence(mode="local_delta", **evidence),
                    local_adsorbate_coordinates=initial,
                    local_delta=delta,
                    rotation_degrees=rotation,
                    **common,
                )


class HistoryRequiredError(RuntimeError):
    pass


@dataclass(frozen=True)
class HistoryResolution:
    index_revision: int
    trace: RetrievalTrace
    revisions: tuple[AdsorptionCaseRevision, ...]


def _empty_trace(
    features: CaseFeatureSet,
    *,
    index_revision: int,
    threshold: float,
    limit: int,
    reason: str,
) -> RetrievalTrace:
    query = RetrievalQuery(
        index_revision=index_revision,
        features=features,
        threshold=threshold,
        limit=limit,
    )
    return RetrievalTrace(index_revision=index_revision, query=query, fallback_reason=reason)


def resolve_history(
    *,
    history_policy: Literal["prefer", "require", "off"],
    query_features: CaseFeatureSet,
    store_factory: Callable[[], object],
    index_revision: int | None = None,
    threshold: float = 0.65,
    limit: int = 5,
    retriever: CaseRetriever | None = None,
) -> HistoryResolution:
    """Freeze one store revision, or branch off before touching the store."""

    if history_policy == "off":
        trace = _empty_trace(
            query_features,
            index_revision=0 if index_revision is None else index_revision,
            threshold=threshold,
            limit=limit,
            reason="history_disabled",
        )
        return HistoryResolution(trace.index_revision, trace, ())
    try:
        store = store_factory()
        frozen = store.status().index_revision if index_revision is None else index_revision
        cases = tuple(store.list_revisions(index_revision=frozen))
        query = RetrievalQuery(
            index_revision=frozen,
            features=query_features,
            threshold=threshold,
            limit=limit,
        )
        trace = (retriever or CaseRetriever()).retrieve(query, cases)
    except (AdsorptionStoreError, sqlite3.DatabaseError, OSError) as exc:
        if history_policy == "require":
            raise HistoryRequiredError(f"history is required but the store is unavailable: {exc}") from exc
        frozen = 0 if index_revision is None else index_revision
        trace = _empty_trace(
            query_features,
            index_revision=frozen,
            threshold=threshold,
            limit=limit,
            reason=f"store_unavailable:{type(exc).__name__}",
        )
        return HistoryResolution(frozen, trace, ())
    if history_policy == "require" and not trace.matches:
        raise HistoryRequiredError(
            f"history is required but no compatible case matched at index revision {frozen}"
        )
    return HistoryResolution(frozen, trace, cases)


@dataclass(frozen=True)
class ProposalStreamAudit:
    attempted: int
    accepted: int
    rejected: int
    truncated: bool


def bounded_proposal_stream(
    algorithmic: Iterable[object],
    historical: Iterable[object],
    *,
    max_attempts: int,
    max_accepted: int,
    accept: Callable[[object], bool],
) -> tuple[list[object], ProposalStreamAudit]:
    """Consume both sources lazily with deterministic geometry-first round robin."""

    accepted: list[object] = []
    attempts = 0
    rejected = 0
    iterators = [iter(algorithmic), iter(historical)]
    active = [True, True]
    turn = 0
    while attempts < max_attempts and len(accepted) < max_accepted and any(active):
        source_index = turn % len(iterators)
        turn += 1
        if not active[source_index]:
            continue
        try:
            proposal = next(iterators[source_index])
        except StopIteration:
            active[source_index] = False
            continue
        attempts += 1
        if accept(proposal):
            accepted.append(proposal)
        else:
            rejected += 1
    truncated = attempts >= max_attempts or len(accepted) >= max_accepted
    return accepted, ProposalStreamAudit(attempts, len(accepted), rejected, truncated)
