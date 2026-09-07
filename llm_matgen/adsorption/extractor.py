"""Pure adsorption-case extraction and conservative quality gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Iterable

import numpy as np
from pymatgen.core import Lattice
from scipy.optimize import linear_sum_assignment

from llm_matgen.adsorption.models import (
    CaseAudit,
    CaseFeatureSet,
    CaseStatus,
    FileEvidence,
)
from llm_matgen.adsorption.source import JobSnapshot

_RELAXATION_IBRION = {1, 2, 3}
_FREQUENCY_IBRION = {5, 6, 7, 8}
_MAX_CANONICAL_GRAPH_PERMUTATIONS = 1_000_000
_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "P": 1.07,
    "Cu": 1.32,
    "Pt": 1.36,
    "Au": 1.36,
    "Ti": 1.60,
    "Ce": 2.04,
}


class ExtractionError(ValueError):
    pass


class AmbiguousMappingError(ExtractionError):
    pass


class AmbiguousVacuumError(ExtractionError):
    pass


class GraphCanonicalizationLimitError(ExtractionError):
    pass


@dataclass(frozen=True)
class ParsedStructure:
    lattice: np.ndarray
    species: tuple[str, ...]
    fractional: np.ndarray


@dataclass(frozen=True)
class CaseExtraction:
    status: CaseStatus
    audit: CaseAudit
    features: CaseFeatureSet
    evidence: tuple[FileEvidence, ...]
    exact_hash: str
    equivalent_fingerprint: str
    clean_parent_identity: str | None
    artifacts: dict[str, str]


@dataclass(frozen=True)
class AtomMappingMetrics:
    mapping: list[int]
    best_cost: float
    cost_gap: float | None


@dataclass(frozen=True)
class CanonicalGraph:
    fingerprint: str
    edges: list[tuple[int, int]]
    canonical_edges: list[tuple[int, int]]
    connected: bool


def _parse_poscar(text: str) -> ParsedStructure:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ExtractionError("POSCAR is incomplete")
    scale = float(lines[1].split()[0])
    raw_lattice = np.array([[float(item) for item in lines[i].split()[:3]] for i in range(2, 5)])
    if scale <= 0:
        raise ExtractionError("negative-volume POSCAR scaling is not supported by this extractor version")
    lattice = raw_lattice * scale
    symbols = lines[5].split()
    try:
        counts = [int(item) for item in lines[6].split()]
    except ValueError as exc:
        raise ExtractionError("VASP 4 POSCAR without element names is unsupported") from exc
    if len(symbols) != len(counts) or any(item <= 0 for item in counts):
        raise ExtractionError("invalid POSCAR species/count header")
    cursor = 7
    if lines[cursor].lower().startswith("s"):
        cursor += 1
    direct = lines[cursor].lower().startswith(("d", "f"))
    cartesian = lines[cursor].lower().startswith(("c", "k"))
    if not direct and not cartesian:
        raise ExtractionError("missing POSCAR coordinate mode")
    cursor += 1
    atom_count = sum(counts)
    if len(lines) < cursor + atom_count:
        raise ExtractionError("POSCAR coordinate count is incomplete")
    coordinates = np.array(
        [[float(item) for item in lines[cursor + i].split()[:3]] for i in range(atom_count)]
    )
    if cartesian:
        coordinates = (coordinates * scale) @ np.linalg.inv(lattice)
    fractional = coordinates % 1.0
    species = tuple(symbol for symbol, count in zip(symbols, counts) for _ in range(count))
    return ParsedStructure(lattice=lattice, species=species, fractional=fractional)


def _fractional_delta(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    delta = np.asarray(final, dtype=float) - np.asarray(initial, dtype=float)
    return delta - np.round(delta)


def minimum_image_displacement(initial: np.ndarray, final: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    lattice_array = np.asarray(lattice, dtype=float)
    _, image = Lattice(lattice_array).get_distance_and_image(initial, final)
    fractional = np.asarray(final, dtype=float) + image - np.asarray(initial, dtype=float)
    return fractional @ lattice_array


def _cost_matrix(initial: np.ndarray, final: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [np.linalg.norm(minimum_image_displacement(left, right, lattice)) for right in final]
            for left in initial
        ]
    )


def map_atoms(
    initial: np.ndarray,
    final: np.ndarray,
    initial_species: list[str] | tuple[str, ...],
    final_species: list[str] | tuple[str, ...],
    lattice: np.ndarray,
    *,
    ambiguity_tolerance: float = 1.0e-8,
    allow_count_mismatch: bool = False,
) -> list[int]:
    """Map atoms by element and PBC distance, rejecting a tied second solution."""

    return map_atoms_with_metrics(
        initial,
        final,
        initial_species,
        final_species,
        lattice,
        ambiguity_tolerance=ambiguity_tolerance,
        allow_count_mismatch=allow_count_mismatch,
    ).mapping


def map_atoms_with_metrics(
    initial: np.ndarray,
    final: np.ndarray,
    initial_species: list[str] | tuple[str, ...],
    final_species: list[str] | tuple[str, ...],
    lattice: np.ndarray,
    *,
    ambiguity_tolerance: float = 1.0e-8,
    allow_count_mismatch: bool = False,
) -> AtomMappingMetrics:
    """Return the unique element-aware assignment and its second-best gap."""

    if not allow_count_mismatch and Counter(initial_species) != Counter(final_species):
        raise ExtractionError("composition mismatch")
    mapping = [-1] * len(initial_species)
    ambiguity_gaps: list[float] = []
    total_best = 0.0
    for element in sorted(set(initial_species)):
        left_ids = [index for index, symbol in enumerate(initial_species) if symbol == element]
        right_ids = [index for index, symbol in enumerate(final_species) if symbol == element]
        if len(right_ids) < len(left_ids) or (not allow_count_mismatch and len(right_ids) != len(left_ids)):
            raise ExtractionError("composition mismatch")
        cost = _cost_matrix(np.asarray(initial)[left_ids], np.asarray(final)[right_ids], lattice)
        rows, columns = linear_sum_assignment(cost)
        best = float(cost[rows, columns].sum())
        total_best += best
        for row, column in zip(rows, columns):
            mapping[left_ids[int(row)]] = right_ids[int(column)]
        for row, column in zip(rows, columns):
            blocked = cost.copy()
            blocked[row, column] = np.inf
            try:
                alt_rows, alt_columns = linear_sum_assignment(blocked)
            except ValueError:
                continue
            alt = float(blocked[alt_rows, alt_columns].sum())
            if math.isfinite(alt):
                ambiguity_gaps.append(alt - best)
    if any(gap <= ambiguity_tolerance for gap in ambiguity_gaps):
        raise AmbiguousMappingError("best and second-best atom mappings are indistinguishable")
    if any(item < 0 for item in mapping):
        raise ExtractionError("incomplete atom mapping")
    gap = min(ambiguity_gaps) if ambiguity_gaps else None
    return AtomMappingMetrics(mapping=mapping, best_cost=total_best, cost_gap=gap)


def remove_collective_drift(
    initial_fractional: np.ndarray,
    final_fractional: np.ndarray,
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    displacements = np.array(
        [
            minimum_image_displacement(left, right, lattice)
            for left, right in zip(initial_fractional, final_fractional)
        ]
    )
    drift = np.median(displacements, axis=0)
    return displacements - drift, drift


def detect_vacuum_axis(
    lattice: np.ndarray,
    fractional: np.ndarray,
    *,
    uniqueness_tolerance: float = 1.0e-6,
) -> int:
    """Choose the lattice axis with the largest real-space periodic empty gap."""

    scores: list[float] = []
    for axis in range(3):
        values = np.sort(np.asarray(fractional)[:, axis] % 1.0)
        if len(values) == 0:
            raise AmbiguousVacuumError("cannot determine vacuum axis without atoms")
        gaps = np.diff(np.concatenate([values, values[:1] + 1.0]))
        scores.append(float(np.max(gaps) * np.linalg.norm(lattice[axis])))
    order = np.argsort(scores)
    if scores[int(order[-1])] - scores[int(order[-2])] <= uniqueness_tolerance:
        raise AmbiguousVacuumError("vacuum direction is not unique")
    return int(order[-1])


def partition_against_parent(
    child_species: list[str] | tuple[str, ...],
    child_fractional: np.ndarray,
    parent_species: list[str] | tuple[str, ...],
    parent_fractional: np.ndarray,
    lattice: np.ndarray,
) -> tuple[list[int], list[int]]:
    mapping = map_atoms(
        np.asarray(parent_fractional),
        np.asarray(child_fractional),
        parent_species,
        child_species,
        lattice,
        allow_count_mismatch=True,
    )
    slab_ids = sorted(mapping)
    adsorbate_ids = sorted(set(range(len(child_species))) - set(slab_ids))
    if not adsorbate_ids:
        raise ExtractionError("no adsorbate remains after clean-parent mapping")
    return slab_ids, adsorbate_ids


def _parse_incar(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        for statement in line.split(";"):
            if "=" in statement:
                key, value = statement.split("=", 1)
                values[key.strip().upper()] = value.strip()
    return values


def _int_setting(settings: dict[str, str], name: str, default: int = 0) -> int:
    try:
        return int(float(settings.get(name, str(default)).split()[0]))
    except (ValueError, IndexError):
        return default


def _run_type(settings: dict[str, str], relative_job_dir: str) -> str:
    nsw = _int_setting(settings, "NSW", 0)
    ibrion = _int_setting(settings, "IBRION", -1)
    if "IMAGES" in settings or any(re.fullmatch(r"\d{2}", part) for part in relative_job_dir.split("/")):
        return "neb"
    if ibrion in _FREQUENCY_IBRION:
        return "frequency"
    if nsw <= 0 or ibrion < 0:
        return "static"
    if ibrion in _RELAXATION_IBRION:
        return "ionic_relaxation"
    if "LORBIT" in settings or "NEDOS" in settings:
        return "dos"
    return "unsupported"


def _hill_formula(symbols: Iterable[str]) -> str:
    counts = Counter(symbols)
    ordered: list[str] = []
    if "C" in counts:
        ordered.append("C")
    if "H" in counts:
        ordered.append("H")
    ordered.extend(sorted(set(counts) - set(ordered)))
    return "".join(symbol + (str(counts[symbol]) if counts[symbol] != 1 else "") for symbol in ordered)


def _radius(symbol: str) -> float:
    return _COVALENT_RADII.get(symbol, 1.25)


def canonical_adsorbate_graph(
    species: tuple[str, ...],
    fractional: np.ndarray,
    ids: list[int],
    lattice: np.ndarray,
) -> CanonicalGraph:
    edges: list[tuple[int, int]] = []
    for left, right in combinations(ids, 2):
        distance = np.linalg.norm(minimum_image_displacement(fractional[left], fractional[right], lattice))
        if distance <= 1.2 * (_radius(species[left]) + _radius(species[right])):
            edges.append((left, right))
    formula = _hill_formula(species[index] for index in ids)
    if len(ids) == 1:
        return CanonicalGraph(
            fingerprint=f"{formula}:single", edges=edges, canonical_edges=[], connected=True
        )
    adjacency = {index: set() for index in ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = set()
    stack = [ids[0]]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency[current] - seen)
    if len(seen) != len(ids):
        return CanonicalGraph(
            fingerprint=f"{formula}:disconnected",
            edges=edges,
            canonical_edges=[],
            connected=False,
        )
    element_colors = {
        element: color
        for color, element in enumerate(sorted({species[index] for index in ids}))
    }
    colors = {index: element_colors[species[index]] for index in ids}
    while True:
        signatures = {
            index: (
                colors[index],
                tuple(sorted(colors[neighbor] for neighbor in adjacency[index])),
            )
            for index in ids
        }
        signature_colors = {
            signature: color
            for color, signature in enumerate(sorted(set(signatures.values())))
        }
        refined = {
            index: signature_colors[signatures[index]]
            for index in ids
        }
        old_groups = {
            frozenset(index for index in ids if colors[index] == color)
            for color in set(colors.values())
        }
        new_groups = {
            frozenset(index for index in ids if refined[index] == color)
            for color in set(refined.values())
        }
        colors = refined
        if old_groups == new_groups:
            break
    groups = [
        [index for index in ids if colors[index] == color]
        for color in sorted(set(colors.values()))
    ]
    permutation_count = 1
    for group in groups:
        permutation_count *= math.factorial(len(group))
        if permutation_count > _MAX_CANONICAL_GRAPH_PERMUTATIONS:
            raise GraphCanonicalizationLimitError(
                "adsorbate_graph_complexity_exceeds_limit"
            )
    best_certificate: str | None = None
    best_order: list[int] | None = None
    for grouped_order in product(*(permutations(group) for group in groups)):
        order = [index for group in grouped_order for index in group]
        labels = ",".join(species[index] for index in order)
        bits = "".join(
            "1" if order[right] in adjacency[order[left]] else "0"
            for left in range(len(order))
            for right in range(left + 1, len(order))
        )
        certificate = f"{labels}|{bits}"
        if best_certificate is None or certificate < best_certificate:
            best_certificate = certificate
            best_order = order
    assert best_certificate is not None and best_order is not None
    canonical_edges = [
        (left, right)
        for left in range(len(best_order))
        for right in range(left + 1, len(best_order))
        if best_order[right] in adjacency[best_order[left]]
    ]
    certificate = best_certificate
    fingerprint = f"{formula}:{hashlib.sha256(certificate.encode()).hexdigest()}"
    return CanonicalGraph(
        fingerprint=fingerprint,
        edges=edges,
        canonical_edges=canonical_edges,
        connected=len(seen) == len(ids),
    )


def _adsorbate_graph(
    species: tuple[str, ...], fractional: np.ndarray, ids: list[int], lattice: np.ndarray
) -> tuple[str, list[tuple[int, int]]]:
    graph = canonical_adsorbate_graph(species, fractional, ids, lattice)
    return graph.fingerprint, graph.edges


def _composition_fraction(symbols: Iterable[str]) -> dict[str, float]:
    counts = Counter(symbols)
    total = sum(counts.values())
    return {symbol: count / total for symbol, count in sorted(counts.items())}


def _surface_frame(lattice: np.ndarray, vacuum_axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inplane_axes = [index for index in range(3) if index != vacuum_axis]
    first_vector = np.asarray(lattice[inplane_axes[0]], dtype=float)
    second_vector = np.asarray(lattice[inplane_axes[1]], dtype=float)
    normal = np.cross(first_vector, second_vector)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        raise AmbiguousVacuumError("surface lattice vectors are degenerate")
    normal /= norm
    if float(np.dot(normal, lattice[vacuum_axis])) < 0.0:
        normal *= -1.0
    basis_first = first_vector / np.linalg.norm(first_vector)
    basis_second = np.cross(normal, basis_first)
    basis_second /= np.linalg.norm(basis_second)
    return normal, basis_first, basis_second


def _surface_contacts(
    species: tuple[str, ...],
    fractional: np.ndarray,
    adsorbate_ids: list[int],
    slab_ids: list[int],
    lattice: np.ndarray,
) -> tuple[list[tuple[float, int, int]], list[tuple[float, int, int]]]:
    distances = sorted(
        (
            float(np.linalg.norm(minimum_image_displacement(fractional[ads_id], fractional[slab_id], lattice))),
            ads_id,
            slab_id,
        )
        for ads_id in adsorbate_ids
        for slab_id in slab_ids
    )
    contacts = [
        item
        for item in distances
        if item[0] <= 1.25 * (_radius(species[item[1]]) + _radius(species[item[2]]))
    ]
    return distances, contacts


def _site_type(species: tuple[str, ...], contacts: list[tuple[float, int, int]]) -> tuple[str, Counter]:
    shell = Counter(species[item[2]] for item in contacts)
    count = sum(shell.values())
    if count <= 1:
        return "top", shell
    if count == 2:
        return "bridge", shell
    return "hollow", shell


def _orientation(
    fractional: np.ndarray,
    species: tuple[str, ...],
    adsorbate_ids: list[int],
    anchor_id: int,
    lattice: np.ndarray,
    normal: np.ndarray,
    basis_first: np.ndarray,
    basis_second: np.ndarray,
) -> tuple[float | None, float | None]:
    other_ids = [index for index in adsorbate_ids if index != anchor_id]
    if not other_ids:
        return None, None
    vectors = [
        minimum_image_displacement(fractional[anchor_id], fractional[index], lattice)
        for index in other_ids
    ]
    vector = max(vectors, key=lambda item: float(np.linalg.norm(item)))
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12:
        return None, None
    cosine = max(-1.0, min(1.0, float(np.dot(vector, normal)) / length))
    tilt = math.degrees(math.acos(cosine))
    azimuth = math.degrees(
        math.atan2(float(np.dot(vector, basis_second)), float(np.dot(vector, basis_first)))
    ) % 360.0
    return tilt, azimuth


def shortest_2d_periodic_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest non-zero vector of a two-dimensional lattice."""

    left = np.asarray(first, dtype=float).copy()
    right = np.asarray(second, dtype=float).copy()
    for _ in range(128):
        if np.linalg.norm(right) < np.linalg.norm(left):
            left, right = right, left
        coefficient = int(round(float(np.dot(left, right) / np.dot(left, left))))
        if coefficient == 0:
            return float(np.linalg.norm(left))
        right = right - coefficient * left
    raise ExtractionError("2D lattice reduction did not converge")


def _outermost_surface_layer_ids(
    slab_ids: list[int], projections: np.ndarray, surface_side: str
) -> list[int]:
    """Cluster the outward plane by deterministic gaps along the true normal."""

    sign = 1.0 if surface_side == "top" else -1.0
    ordered = sorted(
        (
            (-sign * float(projection), slab_id)
            for slab_id, projection in zip(slab_ids, projections)
        ),
        key=lambda item: (item[0], item[1]),
    )
    surface = [ordered[0][1]]
    previous_depth = ordered[0][0]
    for depth, slab_id in ordered[1:]:
        if depth - previous_depth > 0.35:
            break
        surface.append(slab_id)
        previous_depth = depth
    return sorted(surface)


def _structure_provenance(text: str) -> tuple[tuple[int, int, int] | None, str | None]:
    lines = text.splitlines()
    comment = lines[0] if lines else ""
    miller_match = re.search(
        r"MILLER\s*=\s*\(?\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*[, ]\s*(-?\d+)\s*\)?",
        comment,
        flags=re.IGNORECASE,
    )
    termination_match = re.search(
        r"TERMINATION\s*=\s*([A-Za-z0-9_.+\-]+)", comment, flags=re.IGNORECASE
    )
    miller = (
        tuple(int(miller_match.group(index)) for index in (1, 2, 3))
        if miller_match
        else None
    )
    return miller, termination_match.group(1) if termination_match else None


def _hash_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


class CaseExtractor:
    """Conservative pure extractor; all uncertain cases are retained as rejected."""

    required_files = ("POSCAR", "CONTCAR", "INCAR", "OUTCAR")

    @staticmethod
    def _evidence(snapshot: JobSnapshot) -> tuple[FileEvidence, ...]:
        return tuple(
            FileEvidence(
                relative_path=name,
                sha256=item.sha256,
                size=item.size,
                mtime_ns=item.mtime_ns,
                captured_at=item.captured_at,
                stable=item.stable,
                content_stored=name in {"POSCAR", "CONTCAR", "INCAR"} and item.content is not None,
                parsed_evidence={} if name != "INCAR" else {"available": item.content is not None},
                markers=item.markers,
            )
            for name, item in sorted(snapshot.files.items())
        )

    def _result(
        self,
        snapshot: JobSnapshot,
        *,
        status: CaseStatus,
        audit: CaseAudit,
        features: CaseFeatureSet | None = None,
    ) -> CaseExtraction:
        evidence = self._evidence(snapshot)
        exact_hash = _hash_payload(
            {
                "files": [(item.relative_path, item.sha256, item.size) for item in evidence],
                "clean_parent_identity": snapshot.clean_parent_identity,
                "clean_parent_sha256": None
                if snapshot.clean_parent_poscar is None
                else hashlib.sha256(snapshot.clean_parent_poscar.encode()).hexdigest(),
            }
        )
        features = features or CaseFeatureSet()
        equivalent_features = features.model_dump(mode="json")
        equivalent_features.pop("coordinated_surface_site_ids", None)
        equivalent_features.pop("pose_correction", None)
        equivalent_payload = {
            "status": status.value,
            "features": equivalent_features,
        }
        if status is not CaseStatus.ELIGIBLE:
            equivalent_payload["exact_hash"] = exact_hash
        equivalent = _hash_payload(equivalent_payload)
        artifacts = {
            name: item.content
            for name, item in snapshot.files.items()
            if name in {"POSCAR", "CONTCAR", "INCAR"} and item.content is not None
        }
        try:
            initial = _parse_poscar(snapshot.files["POSCAR"].content or "")
            final = _parse_poscar(snapshot.files["CONTCAR"].content or "")
            artifacts["structures.mson"] = json.dumps(
                {
                    "initial": {"lattice": initial.lattice.tolist(), "species": list(initial.species), "fractional": initial.fractional.tolist()},
                    "final": {"lattice": final.lattice.tolist(), "species": list(final.species), "fractional": final.fractional.tolist()},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        except (KeyError, ExtractionError):
            pass
        return CaseExtraction(
            status=status,
            audit=audit,
            features=features,
            evidence=evidence,
            exact_hash=exact_hash,
            equivalent_fingerprint=equivalent,
            clean_parent_identity=snapshot.clean_parent_identity,
            artifacts=artifacts,
        )

    def extract(self, snapshot: JobSnapshot) -> CaseExtraction:
        present = all(name in snapshot.files for name in self.required_files)
        stable = present and all(snapshot.files[name].stable for name in self.required_files)
        audit = CaseAudit(files_complete=present, stable=stable)
        if not present:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(update={"rejection_reasons": ["required_files_missing"]}),
            )
        if not stable:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(update={"rejection_reasons": ["unstable_files"]}),
            )
        if any(snapshot.files[name].content is None for name in ("POSCAR", "CONTCAR", "INCAR")):
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(update={"rejection_reasons": ["small_input_unavailable"]}),
            )

        settings = _parse_incar(snapshot.files["INCAR"].content or "")
        run_type = _run_type(settings, snapshot.relative_job_dir)
        is_relaxation = run_type == "ionic_relaxation"
        audit = audit.model_copy(update={"run_type": run_type, "ionic_relaxation": is_relaxation})
        if not is_relaxation:
            reason = {
                "neb": "neb_run",
                "frequency": "frequency_run",
                "static": "static_run",
                "dos": "dos_run",
            }.get(run_type, "unsupported_run_type")
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(update={"rejection_reasons": [reason]}),
            )

        markers = snapshot.files["OUTCAR"].markers
        fatal_free = not bool(markers.get("fatal_warning", False))
        electronic = bool(markers.get("electronic_converged", False))
        ionic = bool(markers.get("ionic_converged", False))
        normal = bool(markers.get("normal_termination", False))
        raw_ionic_steps = markers.get("ionic_steps")
        ionic_steps = int(raw_ionic_steps) if raw_ionic_steps is not None else None
        audit = audit.model_copy(
            update={
                "fatal_warning_free": fatal_free,
                "electronic_converged": electronic,
                "ionic_converged": ionic,
                "normal_termination": normal,
                "ionic_steps": ionic_steps,
                "electronic_evidence": {"electronic_converged": electronic},
                "ionic_evidence": {"ionic_converged": ionic, "ionic_steps": ionic_steps},
                "termination_evidence": {"normal_termination": normal},
            }
        )
        if not fatal_free:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_FATAL_WARNING,
                audit=audit.model_copy(update={"rejection_reasons": ["fatal_marker"]}),
            )
        if not electronic:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_ELECTRONIC_UNCONVERGED,
                audit=audit.model_copy(update={"rejection_reasons": ["electronic_not_converged"]}),
            )
        if not ionic:
            reasons = ["ionic_not_converged"]
            if ionic_steps is not None and ionic_steps >= _int_setting(settings, "NSW", 0) > 0:
                reasons.append("nsw_exhausted")
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_IONIC_UNCONVERGED,
                audit=audit.model_copy(update={"rejection_reasons": reasons}),
            )
        if not normal:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(update={"rejection_reasons": ["abnormal_termination"]}),
            )

        try:
            initial = _parse_poscar(snapshot.files["POSCAR"].content or "")
            final = _parse_poscar(snapshot.files["CONTCAR"].content or "")
        except ExtractionError:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(update={"rejection_reasons": ["structure_parse_failed"]}),
            )
        composition_match = Counter(initial.species) == Counter(final.species)
        audit = audit.model_copy(update={"composition_match": composition_match})
        if not composition_match:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_AMBIGUOUS_MAPPING,
                audit=audit.model_copy(update={"rejection_reasons": ["composition_mismatch"]}),
            )
        if not np.allclose(initial.lattice, final.lattice, rtol=1.0e-8, atol=1.0e-8):
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={"rejection_reasons": ["variable_cell_unsupported"]}
                ),
            )
        try:
            mapping_metrics = map_atoms_with_metrics(
                initial.fractional,
                final.fractional,
                initial.species,
                final.species,
                initial.lattice,
            )
            mapping = mapping_metrics.mapping
        except (ExtractionError, AmbiguousMappingError):
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_AMBIGUOUS_MAPPING,
                audit=audit.model_copy(
                    update={"mapping_unique": False, "rejection_reasons": ["atom_mapping_ambiguous"]}
                ),
            )
        reordered_final = final.fractional[mapping]
        mapping_cost = mapping_metrics.best_cost
        audit = audit.model_copy(
            update={
                "mapping_unique": True,
                "mapping_confidence": "unique",
                "mapping_cost": mapping_cost,
                "mapping_cost_gap": mapping_metrics.cost_gap,
            }
        )

        if not snapshot.clean_parent_identity or not snapshot.clean_parent_poscar:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={"clean_parent_found": False, "rejection_reasons": ["clean_parent_missing"]}
                ),
            )
        try:
            parent = _parse_poscar(snapshot.clean_parent_poscar)
            slab_ids, adsorbate_ids = partition_against_parent(
                initial.species,
                initial.fractional,
                parent.species,
                parent.fractional,
                initial.lattice,
            )
        except (ExtractionError, AmbiguousMappingError):
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_AMBIGUOUS_MAPPING,
                audit=audit.model_copy(
                    update={
                        "clean_parent_found": True,
                        "partition_unique": False,
                        "rejection_reasons": ["parent_partition_ambiguous"],
                    }
                ),
            )
        audit = audit.model_copy(update={"clean_parent_found": True, "partition_unique": True})

        try:
            vacuum_axis = detect_vacuum_axis(initial.lattice, initial.fractional[slab_ids])
        except AmbiguousVacuumError:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_AMBIGUOUS_MAPPING,
                audit=audit.model_copy(
                    update={
                        "vacuum_axis_unique": False,
                        "rejection_reasons": ["vacuum_axis_ambiguous"],
                    }
                ),
            )
        try:
            normal, basis_first, basis_second = _surface_frame(
                initial.lattice, vacuum_axis
            )
        except AmbiguousVacuumError:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_AMBIGUOUS_MAPPING,
                audit=audit.model_copy(
                    update={
                        "vacuum_axis_unique": False,
                        "rejection_reasons": ["surface_frame_ambiguous"],
                    }
                ),
            )
        slab_corrected, drift = remove_collective_drift(
            initial.fractional[slab_ids],
            reordered_final[slab_ids],
            initial.lattice,
        )
        slab_reconstruction = bool(np.max(np.linalg.norm(slab_corrected, axis=1)) > 1.0)

        try:
            initial_graph = canonical_adsorbate_graph(
                initial.species, initial.fractional, adsorbate_ids, initial.lattice
            )
            final_graph = canonical_adsorbate_graph(
                initial.species, reordered_final, adsorbate_ids, initial.lattice
            )
        except GraphCanonicalizationLimitError:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={
                        "adsorbate_intact": None,
                        "rejection_reasons": [
                            "adsorbate_graph_complexity_exceeds_limit"
                        ],
                    }
                ),
            )
        graph_fingerprint, graph_edges = initial_graph.fingerprint, initial_graph.edges
        final_graph_fingerprint, final_graph_edges = final_graph.fingerprint, final_graph.edges
        audit = audit.model_copy(
            update={
                "raw_internal_bond_graph": {
                    "initial_edges": [[left, right] for left, right in graph_edges],
                    "final_edges": [[left, right] for left, right in final_graph_edges],
                }
            }
        )
        if not initial_graph.connected:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={"adsorbate_intact": False, "rejection_reasons": ["adsorbate_disconnected"]}
                ),
            )
        dissociation = final_graph_fingerprint != graph_fingerprint
        if dissociation:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={
                        "adsorbate_intact": False,
                        "rejection_reasons": ["adsorbate_dissociated"],
                    }
                ),
            )
        flexible = any(
            abs(
                np.linalg.norm(
                    minimum_image_displacement(
                        reordered_final[left], reordered_final[right], initial.lattice
                    )
                )
                - np.linalg.norm(
                    minimum_image_displacement(
                        initial.fractional[left], initial.fractional[right], initial.lattice
                    )
                )
            )
            > max(
                0.15,
                0.15
                * np.linalg.norm(
                    minimum_image_displacement(
                        initial.fractional[left], initial.fractional[right], initial.lattice
                    )
                ),
            )
            for left, right in combinations(adsorbate_ids, 2)
        )
        if flexible:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={
                        "adsorbate_intact": True,
                        "rejection_reasons": ["flexible_adsorbate_unsupported"],
                    }
                ),
            )
        initial_cart = initial.fractional @ initial.lattice
        final_cart = reordered_final @ initial.lattice
        slab_projection = initial_cart[slab_ids] @ normal

        distances, contact_pairs = _surface_contacts(
            initial.species,
            initial.fractional,
            adsorbate_ids,
            slab_ids,
            initial.lattice,
        )
        final_distances, final_contact_pairs = _surface_contacts(
            initial.species,
            reordered_final,
            adsorbate_ids,
            slab_ids,
            initial.lattice,
        )
        nearest_distance, anchor_id, nearest_slab = distances[0]
        final_nearest_distance, final_anchor_id, final_nearest_slab = final_distances[0]
        initial_contact_vector = minimum_image_displacement(
            initial.fractional[nearest_slab],
            initial.fractional[anchor_id],
            initial.lattice,
        )
        signed_initial_height = float(np.dot(initial_contact_vector, normal))
        if abs(signed_initial_height) <= 1.0e-8:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_AMBIGUOUS_MAPPING,
                audit=audit.model_copy(
                    update={
                        "vacuum_axis_unique": True,
                        "rejection_reasons": ["surface_side_ambiguous"],
                    }
                ),
            )
        sign = 1.0 if signed_initial_height > 0.0 else -1.0
        surface_side = "top" if sign > 0.0 else "bottom"
        initial_height = sign * signed_initial_height
        final_contact_vector = minimum_image_displacement(
            reordered_final[final_nearest_slab],
            reordered_final[final_anchor_id],
            initial.lattice,
        )
        final_height = sign * float(np.dot(final_contact_vector, normal))
        anchor_ids = sorted({item[1] for item in contact_pairs})
        final_anchor_ids = sorted({item[1] for item in final_contact_pairs})
        denticity = max(1, len(anchor_ids))
        final_denticity = max(1, len(final_anchor_ids))
        if denticity != 1 or final_denticity != 1:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(update={"rejection_reasons": ["multi_dentate_unsupported"]}),
            )
        if final_anchor_id != anchor_id:
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={"rejection_reasons": ["anchor_identity_changed"]}
                ),
            )
        desorption = final_nearest_distance > 4.0
        subsurface = initial_height < -0.25 or final_height < -0.25
        if desorption or subsurface or slab_reconstruction:
            reasons = []
            if desorption:
                reasons.append("desorbed")
            if subsurface:
                reasons.append("subsurface")
            if slab_reconstruction:
                reasons.append("severe_surface_reconstruction")
            return self._result(
                snapshot,
                status=CaseStatus.REJECTED_INCOMPLETE,
                audit=audit.model_copy(
                    update={
                        "adsorbate_intact": True,
                        "not_desorbed": not desorption,
                        "not_subsurface": not subsurface,
                        "surface_reconstruction_acceptable": not slab_reconstruction,
                        "rejection_reasons": reasons,
                    }
                ),
            )

        axis_others = [index for index in range(3) if index != vacuum_axis]
        ads_delta = minimum_image_displacement(
            initial.fractional[anchor_id],
            reordered_final[anchor_id],
            initial.lattice,
        ) - drift
        outward_normal = normal * sign
        local_second = np.cross(outward_normal, basis_first)
        local_second /= np.linalg.norm(local_second)
        inplane = (
            float(np.dot(ads_delta, basis_first)),
            float(np.dot(ads_delta, local_second)),
        )
        surface_area = float(
            np.linalg.norm(np.cross(initial.lattice[axis_others[0]], initial.lattice[axis_others[1]]))
        )
        inplane_vectors = [initial.lattice[index] for index in axis_others]
        periodic_distance = shortest_2d_periodic_distance(
            inplane_vectors[0], inplane_vectors[1]
        )
        initial_site_contacts = contact_pairs or [(nearest_distance, anchor_id, nearest_slab)]
        final_site_contacts = final_contact_pairs or [
            (final_nearest_distance, final_anchor_id, final_nearest_slab)
        ]
        site_type, first_shell = _site_type(initial.species, initial_site_contacts)
        final_site_type, _ = _site_type(initial.species, final_site_contacts)
        if not first_shell:
            first_shell[initial.species[nearest_slab]] = 1
        first_site_ids = {item[2] for item in initial_site_contacts}
        second_site_ids = {
            other_id
            for first_id in first_site_ids
            for other_id in slab_ids
            if other_id not in first_site_ids
            and np.linalg.norm(
                minimum_image_displacement(
                    initial.fractional[first_id],
                    initial.fractional[other_id],
                    initial.lattice,
                )
            )
            <= 1.25 * (_radius(initial.species[first_id]) + _radius(initial.species[other_id]))
        }
        second_shell = Counter(initial.species[index] for index in second_site_ids)
        surface_ids = _outermost_surface_layer_ids(slab_ids, slab_projection, surface_side)
        miller_index, explicit_termination = _structure_provenance(
            snapshot.files["POSCAR"].content or ""
        )
        termination = explicit_termination or ",".join(
            sorted({initial.species[index] for index in surface_ids})
        )
        initial_tilt, initial_azimuth = _orientation(
            initial.fractional,
            initial.species,
            adsorbate_ids,
            anchor_id,
            initial.lattice,
            outward_normal,
            basis_first,
            local_second,
        )
        final_tilt, final_azimuth = _orientation(
            reordered_final,
            initial.species,
            adsorbate_ids,
            anchor_id,
            initial.lattice,
            outward_normal,
            basis_first,
            local_second,
        )
        rotation = None
        if initial_azimuth is not None and final_azimuth is not None:
            rotation = (final_azimuth - initial_azimuth + 180.0) % 360.0 - 180.0
        features = CaseFeatureSet(
            adsorbate_formula=_hill_formula(initial.species[index] for index in adsorbate_ids),
            adsorbate_graph_fingerprint=graph_fingerprint,
            anchor_element=initial.species[anchor_id],
            denticity=1,
            active_site_elements=sorted(first_shell),
            local_coordination_signature=";".join(
                f"{element}:{count}" for element, count in sorted(first_shell.items())
            ),
            surface_side=surface_side,
            surface_normal=tuple(float(item) for item in outward_normal),
            vacuum_axis=vacuum_axis,
            miller_index=miller_index,
            termination=termination or None,
            initial_height=initial_height,
            final_height=final_height,
            inplane_displacement=inplane,
            initial_tilt=initial_tilt,
            final_tilt=final_tilt,
            initial_azimuth=initial_azimuth,
            final_azimuth=final_azimuth,
            rotation=rotation,
            internal_bond_graph={
                "initial_edges": [list(edge) for edge in initial_graph.canonical_edges],
                "final_edges": [list(edge) for edge in final_graph.canonical_edges],
                "radius_rule_version": "covalent-radii-1",
            },
            initial_site_type=site_type,
            final_site_type=final_site_type,
            site_transition=f"{site_type}->{final_site_type}",
            coordinated_surface_site_ids=[str(item) for item in sorted(first_site_ids)],
            first_coordination_shell=dict(sorted(first_shell.items())),
            second_coordination_shell=dict(sorted(second_shell.items())),
            surface_layer_composition=_composition_fraction(
                initial.species[index] for index in surface_ids
            ),
            local_distance_scale=nearest_distance,
            coverage=1.0 / len(surface_ids),
            surface_area=surface_area,
            periodic_image_distance=periodic_distance,
            slab_drift=tuple(float(item) for item in drift),
            slab_reconstruction=slab_reconstruction,
            dissociation=dissociation,
            desorption=desorption,
            subsurface=subsurface,
            pose_correction={
                "translation": [float(item) for item in ads_delta],
                "mapping": mapping,
                "slab_indices": slab_ids,
                "adsorbate_indices": adsorbate_ids,
                "anchor_index_initial": anchor_id,
                "anchor_index_final": final_anchor_id,
                "local_frame": {
                    "e1": [float(item) for item in basis_first],
                    "e2": [float(item) for item in local_second],
                    "normal": [float(item) for item in outward_normal],
                    "handedness": "right",
                },
                "initial_local_coordinates": [
                    [float(np.dot(vector, axis)) for axis in (basis_first, local_second, outward_normal)]
                    for vector in (
                        minimum_image_displacement(initial.fractional[anchor_id], initial.fractional[index], initial.lattice)
                        for index in adsorbate_ids
                    )
                ],
                "final_local_coordinates": [
                    [float(np.dot(vector, axis)) for axis in (basis_first, local_second, outward_normal)]
                    for vector in (
                        minimum_image_displacement(reordered_final[final_anchor_id], reordered_final[index], initial.lattice)
                        for index in adsorbate_ids
                    )
                ],
                "local_delta": [float(np.dot(ads_delta, axis)) for axis in (basis_first, local_second, outward_normal)],
                "rotation_degrees": rotation,
            },
        )
        audit = audit.model_copy(
            update={
                "vacuum_axis_unique": True,
                "adsorbate_intact": True,
                "not_desorbed": True,
                "not_subsurface": True,
                "surface_reconstruction_acceptable": True,
                "case_quality": 1.0,
            }
        )
        return self._result(
            snapshot,
            status=CaseStatus.ELIGIBLE,
            audit=audit,
            features=features,
        )
