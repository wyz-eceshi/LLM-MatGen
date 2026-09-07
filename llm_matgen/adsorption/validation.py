"""Adsorption-specific chemical, geometric, and execution-contract validation."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Molecule, Structure

from llm_matgen.adsorption.proposals import AdsorptionProposal, build_surface_frame


_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ti": 1.60,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Pt": 1.36,
    "Au": 1.36,
    "Ce": 2.04,
}


def element_radius(symbol: str) -> float:
    """Return a covalent-size radius, with an element-derived fallback."""

    if symbol in _COVALENT_RADII:
        return _COVALENT_RADII[symbol]
    radius = Element(symbol).atomic_radius
    return 1.25 if radius is None else 0.75 * float(radius)


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    atom_indices: tuple[int, ...] = ()
    details: dict[str, float | int | str | bool] = Field(default_factory=dict)


class AdsorptionValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    issues: tuple[ValidationIssue, ...] = ()
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)


def _layer_ids(structure: Structure, slab_atom_count: int) -> tuple[tuple[int, ...], ...]:
    if slab_atom_count <= 0 or slab_atom_count > len(structure):
        raise ValueError("slab_atom_count is outside structure")
    normal = np.asarray(build_surface_frame(structure, "top").normal)
    ordered = sorted(
        (float(np.dot(structure[index].coords, normal)), index)
        for index in range(slab_atom_count)
    )
    layers: list[list[int]] = [[ordered[0][1]]]
    previous = ordered[0][0]
    for projection, index in ordered[1:]:
        if projection - previous > 0.35:
            layers.append([])
        layers[-1].append(index)
        previous = projection
    return tuple(tuple(sorted(layer)) for layer in layers)


def apply_fixed_bottom_layers(
    structure: Structure,
    *,
    slab_atom_count: int,
    fixed_bottom_layers: int,
) -> tuple[Structure, tuple[tuple[int, ...], ...]]:
    """Apply selective dynamics by true-normal layers; adsorbate atoms stay movable."""

    if fixed_bottom_layers < 0:
        raise ValueError("fixed_bottom_layers cannot be negative")
    layers = _layer_ids(structure, slab_atom_count)
    selected = layers[:fixed_bottom_layers]
    fixed = {index for layer in selected for index in layer}
    result = structure.copy()
    existing = result.site_properties.get("selective_dynamics")
    flags = []
    for index in range(len(result)):
        if index >= slab_atom_count:
            flags.append([True, True, True])
        elif index in fixed:
            flags.append([False, False, False])
        elif existing is not None and existing[index] is not None:
            flags.append([bool(item) for item in existing[index]])
        else:
            flags.append([True, True, True])
    if existing is not None:
        result.remove_site_property("selective_dynamics")
    result.add_site_property("selective_dynamics", flags)
    return result, selected


def _bond_graph(symbols: list[str], distances: dict[tuple[int, int], float]) -> set[tuple[int, int]]:
    return {
        pair
        for pair, distance in distances.items()
        if distance <= 1.25 * (element_radius(symbols[pair[0]]) + element_radius(symbols[pair[1]]))
    }


def _gauss_reduce_2d(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=float).copy()
    right = np.asarray(second, dtype=float).copy()
    if np.linalg.norm(np.cross(left, right)) <= 1.0e-12:
        raise ValueError("surface lattice vectors are linearly dependent")
    for _ in range(128):
        if np.linalg.norm(right) < np.linalg.norm(left):
            left, right = right, left
        coefficient = int(round(float(np.dot(left, right) / np.dot(left, left))))
        if coefficient == 0:
            return left, right
        right = right - coefficient * left
    raise ValueError("two-dimensional lattice reduction did not converge")


def shortest_2d_image_distance(
    delta: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    exclude_zero_image: bool,
) -> float:
    """Solve the 2D closest-vector problem after Gauss lattice reduction."""

    reduced = np.vstack(_gauss_reduce_2d(first, second))
    center = np.linalg.lstsq(reduced.T, -np.asarray(delta, dtype=float), rcond=None)[0]
    anchors = np.floor(center).astype(int)
    best = math.inf
    for left_offset in range(-2, 4):
        for right_offset in range(-2, 4):
            coefficients = anchors + np.array([left_offset, right_offset])
            if exclude_zero_image and np.all(coefficients == 0):
                continue
            vector = np.asarray(delta, dtype=float) + coefficients @ reduced
            best = min(best, float(np.linalg.norm(vector)))
    return best


class AdsorptionCandidateValidator:
    """Validate one streamed candidate and remember only accepted geometries."""

    def __init__(
        self,
        clean_slab: Structure,
        adsorbate: Molecule,
        *,
        anchor_zero_based: int,
        anchor_contact_window: tuple[float, float] = (0.75, 1.35),
        min_vacuum_each_side: float = 2.5,
        fixed_bottom_layers: int = 0,
        expected_coverage: float | None = None,
    ):
        if not 0 <= anchor_zero_based < len(adsorbate):
            raise ValueError("anchor index is outside adsorbate")
        lower, upper = anchor_contact_window
        if not (0.0 < lower < upper and all(math.isfinite(value) for value in (lower, upper))):
            raise ValueError("anchor contact window must be finite and increasing")
        if not math.isfinite(min_vacuum_each_side) or min_vacuum_each_side < 0.0:
            raise ValueError("minimum vacuum must be finite and non-negative")
        self.clean_slab = clean_slab.copy()
        self.adsorbate = adsorbate.copy()
        self.anchor_zero_based = anchor_zero_based
        self.anchor_contact_window = anchor_contact_window
        self.min_vacuum_each_side = min_vacuum_each_side
        self.fixed_bottom_layers = fixed_bottom_layers
        self.expected_coverage = expected_coverage
        self._accepted: list[tuple[str, Structure]] = []
        self._matcher = StructureMatcher(
            primitive_cell=False,
            scale=False,
            attempt_supercell=False,
            allow_subset=False,
        )
        self._expected_symbols = [str(site.specie) for site in self.clean_slab] + [
            str(site.specie) for site in self.adsorbate
        ]
        self._expected_distances = {
            (left, right): float(self.adsorbate.get_distance(left, right))
            for left, right in combinations(range(len(self.adsorbate)), 2)
        }
        self._expected_graph = _bond_graph(
            [str(site.specie) for site in self.adsorbate], self._expected_distances
        )

    def _issue(self, issues, code, message, atoms=(), **details):
        issues.append(
            ValidationIssue(
                code=code,
                message=message,
                atom_indices=tuple(int(item) for item in atoms),
                details=details,
            )
        )

    def validate(
        self,
        structure: Structure,
        proposal: AdsorptionProposal,
    ) -> AdsorptionValidationReport:
        issues: list[ValidationIssue] = []
        metrics: dict[str, float | int | str | bool] = {}
        slab_count = len(self.clean_slab)
        ads_count = len(self.adsorbate)
        symbols = [str(site.specie) for site in structure]
        if len(structure) != slab_count + ads_count:
            self._issue(
                issues,
                "atom_count",
                "candidate atom count differs from clean slab plus adsorbate",
                actual=len(structure),
                expected=slab_count + ads_count,
            )
        if symbols != self._expected_symbols:
            self._issue(issues, "species_order", "candidate species order is not slab then adsorbate")

        top_normal = np.asarray(build_surface_frame(self.clean_slab, "top").normal)
        wanted_normal = top_normal if proposal.surface_side == "top" else -top_normal
        normal_alignment = float(np.dot(wanted_normal, np.asarray(proposal.frame.normal)))
        metrics["surface_normal_alignment"] = normal_alignment
        if normal_alignment < 1.0 - 1.0e-8:
            self._issue(
                issues,
                "surface_side_normal",
                "proposal side and local outward normal disagree",
                alignment=normal_alignment,
            )
        if self.expected_coverage is not None and (
            proposal.coverage is None
            or not math.isclose(proposal.coverage, self.expected_coverage, rel_tol=1.0e-8, abs_tol=1.0e-10)
        ):
            self._issue(
                issues,
                "coverage_mismatch",
                "proposal coverage differs from requested coverage",
                actual=-1.0 if proposal.coverage is None else proposal.coverage,
                expected=self.expected_coverage,
            )

        if len(structure) >= slab_count + ads_count:
            anchor_global = slab_count + self.anchor_zero_based
            anchor_symbol = str(self.adsorbate[self.anchor_zero_based].specie)
            contacts = []
            for slab_index in range(slab_count):
                slab_symbol = str(self.clean_slab[slab_index].specie)
                distance = float(structure.get_distance(slab_index, anchor_global))
                scale = element_radius(slab_symbol) + element_radius(anchor_symbol)
                contacts.append((distance / scale, distance, slab_index))
            contact_ratio, contact_distance, contact_index = min(contacts)
            metrics["anchor_contact_ratio"] = contact_ratio
            metrics["anchor_contact_distance"] = contact_distance
            lower, upper = self.anchor_contact_window
            if not lower <= contact_ratio <= upper:
                self._issue(
                    issues,
                    "anchor_contact_window",
                    "anchor to surface contact is outside the radius-scaled window",
                    (anchor_global, contact_index),
                    ratio=contact_ratio,
                    lower=lower,
                    upper=upper,
                )
            for ads_local in range(ads_count):
                ads_global = slab_count + ads_local
                ads_symbol = str(self.adsorbate[ads_local].specie)
                for slab_index in range(slab_count):
                    slab_symbol = str(self.clean_slab[slab_index].specie)
                    distance = float(structure.get_distance(slab_index, ads_global))
                    cutoff = 0.65 * (
                        element_radius(slab_symbol) + element_radius(ads_symbol)
                    )
                    if distance < cutoff:
                        self._issue(
                            issues,
                            "slab_adsorbate_collision",
                            "slab and adsorbate atoms overlap by their role-aware radii",
                            (slab_index, ads_global),
                            distance=distance,
                            cutoff=cutoff,
                            role="anchor" if ads_local == self.anchor_zero_based else "nonanchor",
                        )
                        break

            candidate_distances = {
                (left, right): float(
                    structure.get_distance(slab_count + left, slab_count + right)
                )
                for left, right in combinations(range(ads_count), 2)
            }
            for pair, expected in self._expected_distances.items():
                actual = candidate_distances[pair]
                tolerance = max(0.08, 0.08 * expected)
                if abs(actual - expected) > tolerance:
                    self._issue(
                        issues,
                        "internal_distance_changed",
                        "adsorbate internal distance changed outside rigid tolerance",
                        (slab_count + pair[0], slab_count + pair[1]),
                        actual=actual,
                        expected=expected,
                        tolerance=tolerance,
                    )
            candidate_graph = _bond_graph(
                [str(site.specie) for site in self.adsorbate], candidate_distances
            )
            if candidate_graph != self._expected_graph:
                self._issue(
                    issues,
                    "internal_bond_graph_changed",
                    "adsorbate bond graph differs from the input molecule",
                    expected_edges=len(self._expected_graph),
                    actual_edges=len(candidate_graph),
                )

            inplane = [np.asarray(self.clean_slab.lattice.matrix[index]) for index in (0, 1)]
            ads_coords = np.asarray(structure.cart_coords[slab_count : slab_count + ads_count])
            image_failure = None
            for left in range(ads_count):
                for right in range(left, ads_count):
                    distance = shortest_2d_image_distance(
                        ads_coords[right] - ads_coords[left],
                        inplane[0],
                        inplane[1],
                        exclude_zero_image=left == right,
                    )
                    cutoff = 0.85 * (
                        element_radius(str(self.adsorbate[left].specie))
                        + element_radius(str(self.adsorbate[right].specie))
                    )
                    if distance < cutoff:
                        image_failure = (left, right, distance, cutoff)
                        break
                if image_failure:
                    break
            if image_failure:
                left, right, distance, cutoff = image_failure
                self._issue(
                    issues,
                    "adsorbate_periodic_image",
                    "adsorbate is too close to its two-dimensional periodic image",
                    (slab_count + left, slab_count + right),
                    distance=distance,
                    cutoff=cutoff,
                )

        vacuum_period = abs(float(np.dot(self.clean_slab.lattice.matrix[2], top_normal)))
        if vacuum_period <= 1.0e-8:
            self._issue(issues, "insufficient_vacuum", "vacuum vector has no normal component")
        else:
            projections = (np.asarray(structure.cart_coords) @ top_normal) % vacuum_period
            lower_vacuum = float(np.min(projections))
            upper_vacuum = float(vacuum_period - np.max(projections))
            metrics["vacuum_lower"] = lower_vacuum
            metrics["vacuum_upper"] = upper_vacuum
            if min(lower_vacuum, upper_vacuum) < self.min_vacuum_each_side:
                self._issue(
                    issues,
                    "insufficient_vacuum",
                    "candidate leaves too little normal vacuum on one side",
                    lower=lower_vacuum,
                    upper=upper_vacuum,
                    required=self.min_vacuum_each_side,
                )

        flags = structure.site_properties.get("selective_dynamics")
        if self.fixed_bottom_layers > 0:
            layers = _layer_ids(structure, slab_count)
            fixed = {index for layer in layers[: self.fixed_bottom_layers] for index in layer}
            expected = [
                [False, False, False]
                if index < slab_count and index in fixed
                else [True, True, True]
                for index in range(len(structure))
            ]
            if flags is None or list(flags) != expected:
                self._issue(
                    issues,
                    "fixed_layer_flags",
                    "selective dynamics does not match the true-normal bottom-layer policy",
                )
        elif flags is not None and any(
            list(flags[index]) != [True, True, True]
            for index in range(slab_count, len(structure))
        ):
            self._issue(
                issues,
                "fixed_layer_flags",
                "adsorbate atoms must always remain movable",
            )

        if not issues:
            duplicate = any(
                side == proposal.surface_side and self._matcher.fit(previous, structure)
                for side, previous in self._accepted
            )
            if duplicate:
                self._issue(
                    issues,
                    "duplicate_geometry",
                    "candidate is geometrically or symmetrically equivalent to an accepted candidate",
                )
            else:
                self._accepted.append((proposal.surface_side, structure.copy()))
        return AdsorptionValidationReport(
            accepted=not issues,
            issues=tuple(issues),
            metrics=metrics,
        )
