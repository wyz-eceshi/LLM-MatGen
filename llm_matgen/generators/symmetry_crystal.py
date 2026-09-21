"""Space-group constrained atomic crystal generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from llm_matgen.generators.backends import OptionalDependencyError
from llm_matgen.generators.models import (
    BaseGenerationParams,
    GeneratedStructure,
    GenerationResult,
    Provenance,
    StructureRecord,
)
from llm_matgen.utils.structure import structure_sha256


class SpaceGroupSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1, le=230)
    hall_number: int | None = Field(default=None, ge=1, le=530)

    @model_validator(mode="after")
    def validate_hall_number(self):
        if self.hall_number is not None:
            import spglib

            data = spglib.get_spacegroup_type(self.hall_number)
            if data is None or int(data.number) != self.number:
                raise ValueError("hall_number does not belong to space_group.number")
        return self


class CellSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    setting: Literal["conventional", "hexagonal", "rhombohedral"] = "conventional"
    parameters: tuple[float, float, float, float, float, float]
    unique_axis: Literal["a", "b", "c"] = "b"

    @model_validator(mode="after")
    def validate_parameters(self):
        values = np.asarray(self.parameters, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("cell parameters must be finite")
        if np.any(values[:3] <= 0):
            raise ValueError("cell lengths must be positive")
        if np.any(values[3:] <= 0) or np.any(values[3:] >= 180):
            raise ValueError("cell angles must lie between 0 and 180 degrees")
        lattice = Lattice.from_parameters(*self.parameters)
        if not math.isfinite(lattice.volume) or lattice.volume <= 1e-8:
            raise ValueError("cell volume must be positive")
        return self


class CompositionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reduced: dict[str, PositiveInt] = Field(min_length=1)
    formula_units: PositiveInt | None = None
    formula_units_range: tuple[PositiveInt, PositiveInt] | None = None

    @model_validator(mode="after")
    def validate_composition(self):
        for symbol in self.reduced:
            Element(symbol)
        if (self.formula_units is None) == (self.formula_units_range is None):
            raise ValueError("provide exactly one of formula_units or formula_units_range")
        if self.formula_units_range is not None and self.formula_units_range[0] > self.formula_units_range[1]:
            raise ValueError("formula_units_range must be ordered")
        return self

    def candidate_formula_units(self) -> range:
        if self.formula_units is not None:
            return range(self.formula_units, self.formula_units + 1)
        assert self.formula_units_range is not None
        return range(self.formula_units_range[0], self.formula_units_range[1] + 1)

    def counts(self, formula_units: int) -> dict[str, int]:
        return {element: int(value) * formula_units for element, value in self.reduced.items()}


class CoordinationConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center: str
    neighbor: str
    cutoff_A: PositiveFloat
    min_count: int = Field(default=0, ge=0)
    max_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_elements_and_range(self):
        Element(self.center)
        Element(self.neighbor)
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("coordination max_count must be >= min_count")
        return self


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symprec: PositiveFloat = 0.001
    pair_min_A: dict[str, PositiveFloat] = Field(default_factory=dict)
    coordination: list[CoordinationConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pair_keys(self):
        for key in self.pair_min_A:
            parts = key.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid pair_min_A key: {key!r}")
            Element(parts[0]); Element(parts[1])
        return self


class OrbitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element: str
    representative_fractional: tuple[float, float, float]
    expected_multiplicity: PositiveInt
    wyckoff: str | None = None

    @model_validator(mode="after")
    def validate_orbit(self):
        Element(self.element)
        if not np.isfinite(np.asarray(self.representative_fractional, dtype=float)).all():
            raise ValueError("orbit coordinates must be finite")
        return self


class ExplicitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orbits: list[OrbitSpec] = Field(min_length=1)


class SearchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    candidates: PositiveInt = 5
    max_attempts: PositiveInt = 5000
    fixed_orbits: list[OrbitSpec] = Field(default_factory=list)


class SymmetryCrystalRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: Literal["llm-matgen-symmetry-crystal"] = Field(
        alias="schema", serialization_alias="schema"
    )
    version: Literal[1]
    mode: Literal["explicit", "search"]
    space_group: SpaceGroupSpec
    cell: CellSpec
    composition: CompositionSpec
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    explicit: ExplicitSpec | None = None
    search: SearchSpec | None = None

    @model_validator(mode="after")
    def validate_mode_and_lattice(self):
        if self.mode == "explicit" and (self.explicit is None or self.search is not None):
            raise ValueError("explicit mode requires only the explicit section")
        if self.mode == "search" and (self.search is None or self.explicit is not None):
            raise ValueError("search mode requires only the search section")
        if self.mode == "explicit" and self.composition.formula_units is None:
            raise ValueError("explicit mode requires fixed formula_units")
        _validate_lattice_system(self.space_group.number, self.cell)
        _validate_setting_choice(self.space_group, self.cell)
        return self


class SymmetryCrystalParams(BaseGenerationParams):
    recipe: SymmetryCrystalRecipe
    recipe_source: str = "inline"
    recipe_sha256: str | None = None


def load_recipe_file(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    if path.suffix.lower() == ".json":
        data = json.loads(raw.decode("utf-8-sig"))
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - declared base dependency
            raise RuntimeError("YAML recipes require PyYAML") from exc
        data = yaml.safe_load(raw.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("recipe must contain a mapping object")
    return data, hashlib.sha256(raw).hexdigest()


def _close(left: float, right: float, *, atol: float = 1e-5) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=atol)


def _validate_lattice_system(number: int, cell: CellSpec) -> None:
    crystal_system = str(SpaceGroup.from_int_number(number).crystal_system)
    a, b, c, alpha, beta, gamma = cell.parameters
    right = lambda value: _close(value, 90)
    if crystal_system == "cubic" and not (
        _close(a, b) and _close(b, c) and right(alpha) and right(beta) and right(gamma)
    ):
        raise ValueError("cubic space groups require a=b=c and alpha=beta=gamma=90")
    if crystal_system == "tetragonal" and not (
        _close(a, b) and right(alpha) and right(beta) and right(gamma)
    ):
        raise ValueError("tetragonal space groups require a=b and right angles")
    if crystal_system == "orthorhombic" and not all(map(right, (alpha, beta, gamma))):
        raise ValueError("orthorhombic space groups require right angles")
    if crystal_system == "hexagonal" and not (
        _close(a, b) and right(alpha) and right(beta) and _close(gamma, 120)
    ):
        raise ValueError("hexagonal space groups require a=b, alpha=beta=90, gamma=120")
    if crystal_system == "trigonal":
        rhombohedral = cell.setting == "rhombohedral"
        valid = (
            _close(a, b) and _close(b, c) and _close(alpha, beta) and _close(beta, gamma)
            if rhombohedral else
            _close(a, b) and right(alpha) and right(beta) and _close(gamma, 120)
        )
        if not valid:
            raise ValueError("trigonal cell is incompatible with its selected setting")
    if crystal_system == "monoclinic":
        angles = {"a": (beta, gamma), "b": (alpha, gamma), "c": (alpha, beta)}[cell.unique_axis]
        if not all(map(right, angles)):
            raise ValueError("monoclinic cell must have two right angles for the selected unique axis")


def _validate_setting_choice(space_group: SpaceGroupSpec, cell: CellSpec) -> None:
    crystal_system = str(SpaceGroup.from_int_number(space_group.number).crystal_system)
    if crystal_system == "monoclinic" and cell.unique_axis != "b" and space_group.hall_number is None:
        raise ValueError("nonstandard monoclinic unique axis requires an explicit Hall number")
    if crystal_system == "trigonal" and cell.setting == "rhombohedral":
        symbol = SpaceGroup.from_int_number(space_group.number).symbol
        if not symbol.startswith("R"):
            raise ValueError("rhombohedral setting is only valid for R trigonal space groups")
        if space_group.hall_number is None:
            raise ValueError("rhombohedral setting requires an explicit Hall number")
    if space_group.hall_number is None:
        return
    import spglib

    choice = str(spglib.get_spacegroup_type(space_group.hall_number).choice or "")
    if crystal_system == "monoclinic" and choice:
        axis = choice.lstrip("-")[:1].lower()
        if axis in {"a", "b", "c"} and axis != cell.unique_axis:
            raise ValueError("Hall number setting does not match the selected monoclinic unique axis")
    if crystal_system == "trigonal" and choice in {"H", "R"}:
        expected = "R" if cell.setting == "rhombohedral" else "H"
        if choice != expected:
            raise ValueError("Hall number setting does not match the selected trigonal cell setting")


def _symmetry_operations(recipe: SymmetryCrystalRecipe):
    if recipe.space_group.hall_number is None:
        return SpaceGroup.from_int_number(recipe.space_group.number).symmetry_ops
    import spglib
    from pymatgen.core.operations import SymmOp

    data = spglib.get_symmetry_from_database(recipe.space_group.hall_number)
    return [SymmOp.from_rotation_and_translation(r, t) for r, t in zip(data["rotations"], data["translations"], strict=True)]


def _centering_translations(recipe: SymmetryCrystalRecipe) -> list[list[float]]:
    translations: list[np.ndarray] = []
    for operation in _symmetry_operations(recipe):
        if not np.allclose(operation.rotation_matrix, np.eye(3), atol=1e-8):
            continue
        translation = np.mod(np.asarray(operation.translation_vector, dtype=float), 1.0)
        translation[np.isclose(translation, 1.0, atol=1e-8)] = 0.0
        if np.linalg.norm(translation) < 1e-8:
            continue
        if not any(np.linalg.norm((translation - item) - np.round(translation - item)) < 1e-8 for item in translations):
            translations.append(translation)
    return [list(map(float, item)) for item in sorted(translations, key=lambda value: tuple(np.round(value, 12)))]


def _unique_fractional(points: list[np.ndarray], tolerance: float = 1e-8) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for point in points:
        wrapped = np.mod(point, 1.0)
        if not any(np.linalg.norm((wrapped - other) - np.round(wrapped - other)) < tolerance for other in result):
            result.append(wrapped)
    return sorted(result, key=lambda item: tuple(np.round(item, 12)))


def _expand_explicit(recipe: SymmetryCrystalRecipe) -> tuple[Structure, list[dict[str, Any]]]:
    assert recipe.explicit is not None and recipe.composition.formula_units is not None
    operations = _symmetry_operations(recipe)
    species: list[str] = []
    coords: list[np.ndarray] = []
    orbit_records: list[dict[str, Any]] = []
    for orbit in recipe.explicit.orbits:
        expanded = _unique_fractional([
            np.asarray(operation.operate(orbit.representative_fractional), dtype=float)
            for operation in operations
        ])
        if len(expanded) != orbit.expected_multiplicity:
            raise ValueError(
                f"orbit multiplicity mismatch for {orbit.element}: "
                f"expected {orbit.expected_multiplicity}, generated {len(expanded)}"
            )
        species.extend([orbit.element] * len(expanded)); coords.extend(expanded)
        orbit_records.append({
            "element": orbit.element,
            "representative_fractional": list(orbit.representative_fractional),
            "multiplicity": len(expanded),
            "wyckoff": orbit.wyckoff,
            "_start": len(species) - len(expanded),
            "_end": len(species),
        })
    expected = recipe.composition.counts(recipe.composition.formula_units)
    actual = dict(Counter(species))
    if actual != expected:
        raise ValueError(f"expanded composition mismatch: expected {expected}, generated {actual}")
    structure = Structure(Lattice.from_parameters(*recipe.cell.parameters), species, coords)
    dataset = SpacegroupAnalyzer(structure, symprec=recipe.constraints.symprec).get_symmetry_dataset()
    wyckoffs = list(dataset.wyckoffs) if dataset is not None else []
    for record in orbit_records:
        expected = record["wyckoff"]
        if expected:
            match = re.fullmatch(r"\d*([A-Za-z])", expected.strip())
            if match is None:
                raise ValueError(f"invalid Wyckoff label: {expected!r}")
            actual = set(wyckoffs[record["_start"]:record["_end"]])
            if actual != {match.group(1).lower()}:
                raise ValueError(f"Wyckoff label mismatch for {record['element']}: expected {expected}, got {sorted(actual)}")
        record.pop("_start"); record.pop("_end")
    return structure, orbit_records


def _pair_minima(structure: Structure) -> dict[str, float]:
    symbols = np.asarray([site.specie.symbol for site in structure])
    distances = structure.distance_matrix.copy()
    translations = np.asarray([
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ], dtype=float)
    shortest_periodic = float(np.min(np.linalg.norm(translations @ structure.lattice.matrix, axis=1)))
    np.fill_diagonal(distances, shortest_periodic)
    result: dict[str, float] = {}
    elements = sorted(set(symbols))
    for index, left in enumerate(elements):
        for right in elements[index:]:
            result[f"{left}-{right}"] = float(distances[np.ix_(symbols == left, symbols == right)].min())
    return result


def _translation_residual(structure: Structure, translation: list[float]) -> float:
    from scipy.optimize import linear_sum_assignment

    maximum = 0.0
    for element in sorted({site.specie.symbol for site in structure}):
        coords = np.asarray([
            site.frac_coords for site in structure if site.specie.symbol == element
        ])
        distances = structure.lattice.get_all_distances(
            np.mod(coords + np.asarray(translation), 1.0), coords
        )
        rows, columns = linear_sum_assignment(distances)
        maximum = max(maximum, float(distances[rows, columns].max(initial=0.0)))
    return maximum


def _orbit_summary(structure: Structure, symprec: float) -> list[dict[str, Any]]:
    symmetrized = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetrized_structure()
    result: list[dict[str, Any]] = []
    for indices, wyckoff in zip(
        symmetrized.equivalent_indices,
        symmetrized.wyckoff_symbols,
        strict=True,
    ):
        representative = structure[indices[0]]
        result.append({
            "element": representative.specie.symbol,
            "representative_fractional": [float(value) for value in representative.frac_coords],
            "multiplicity": len(indices),
            "wyckoff": wyckoff,
        })
    return result


def _constraint_key(key: str, available: dict[str, float]) -> str:
    if key in available:
        return key
    left, right = key.split("-")
    reverse = f"{right}-{left}"
    if reverse in available:
        return reverse
    raise ValueError(f"pair constraint references absent elements: {key}")


def _validate_candidate(structure: Structure, recipe: SymmetryCrystalRecipe, counts: dict[str, int]) -> dict[str, Any]:
    actual = Counter(site.specie.symbol for site in structure)
    if dict(actual) != counts:
        raise ValueError(f"candidate composition mismatch: expected {counts}, generated {dict(actual)}")
    if len(structure) > 0 and not np.isfinite(structure.frac_coords).all():
        raise ValueError("candidate contains non-finite coordinates")
    expected_lattice = Lattice.from_parameters(*recipe.cell.parameters)
    actual_parameters = [*structure.lattice.abc, *structure.lattice.angles]
    expected_parameters = [*expected_lattice.abc, *expected_lattice.angles]
    if not np.allclose(actual_parameters, expected_parameters, atol=1e-5):
        raise ValueError("candidate lattice differs from requested fixed cell")
    analyzer = SpacegroupAnalyzer(structure, symprec=recipe.constraints.symprec)
    actual_group = analyzer.get_space_group_number()
    if actual_group != recipe.space_group.number:
        raise ValueError(f"candidate space group mismatch: expected {recipe.space_group.number}, got {actual_group}")
    minima = _pair_minima(structure)
    margins: list[float] = []
    for requested, threshold in recipe.constraints.pair_min_A.items():
        key = _constraint_key(requested, minima)
        if minima[key] < threshold:
            raise ValueError(f"pair distance {key}={minima[key]:.6f} A is below {threshold:.6f} A")
        margins.append(minima[key] / threshold)
    symbols = np.asarray([site.specie.symbol for site in structure])
    distances = structure.distance_matrix
    coordination_penalty = 0
    for rule in recipe.constraints.coordination:
        centers = np.where(symbols == rule.center)[0]
        neighbors = np.where(symbols == rule.neighbor)[0]
        if len(centers) == 0 or len(neighbors) == 0:
            raise ValueError("coordination constraint references absent elements")
        for center in centers:
            count = int(np.count_nonzero(distances[center, neighbors] < rule.cutoff_A))
            if rule.center == rule.neighbor:
                count -= int(distances[center, center] < rule.cutoff_A)
            if count < rule.min_count or (rule.max_count is not None and count > rule.max_count):
                coordination_penalty += 1
    if coordination_penalty:
        raise ValueError(f"coordination constraints failed at {coordination_penalty} sites")
    centering = _centering_translations(recipe)
    centering_residuals = {
        ",".join(f"{value:.12g}" for value in translation): _translation_residual(structure, translation)
        for translation in centering
    }
    if any(value > max(float(recipe.constraints.symprec), 1e-5) * 5 for value in centering_residuals.values()):
        raise ValueError("candidate does not preserve required centering translations")
    if margins:
        distance_margin = min(margins)
    else:
        radius_margins = []
        for key, distance in minima.items():
            left, right = key.split("-")
            radii = [float(Element(symbol).atomic_radius or Element(symbol).atomic_radius_calculated or 1.0) for symbol in (left, right)]
            radius_margins.append(distance / sum(radii))
        distance_margin = min(radius_margins)
    return {
        "space_group_symbol": analyzer.get_space_group_symbol(),
        "space_group_number": actual_group,
        "pair_min_A": minima,
        "distance_margin": float(distance_margin),
        "coordination_penalty": coordination_penalty,
        "centering_translations": centering,
        "centering_residual_A": centering_residuals,
        "orbits": _orbit_summary(structure, float(recipe.constraints.symprec)),
    }


def _import_pyxtal():
    from pyxtal import pyxtal
    from pyxtal.lattice import Lattice as PyXtalLattice
    from pyxtal.symmetry import Group as PyXtalGroup
    return pyxtal, PyXtalLattice, PyXtalGroup


def _search(
    recipe: SymmetryCrystalRecipe,
    max_atoms: int,
) -> tuple[list[tuple[Structure, dict[str, Any]]], dict[str, Any]]:
    assert recipe.search is not None
    try:
        backend = _import_pyxtal()
    except ImportError as exc:
        raise OptionalDependencyError(
            'automatic symmetry search requires PyXtal; install with pip install "llm-matgen[symmetry]"'
        ) from exc
    PyXtal, PyXtalLattice = backend[:2]
    PyXtalGroup = backend[2] if len(backend) > 2 else None
    rng = np.random.default_rng(recipe.search.seed)
    crystal_system = str(SpaceGroup.from_int_number(recipe.space_group.number).crystal_system)
    lattice_type = "hexagonal" if crystal_system == "trigonal" and recipe.cell.setting != "rhombohedral" else crystal_system
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False, scale=False, attempt_supercell=False)
    rejected: Counter[str] = Counter()
    accepted: list[tuple[Structure, dict[str, Any]]] = []
    attempts = 0
    selected_formula_units: int | None = None
    for formula_units in recipe.composition.candidate_formula_units():
        counts = recipe.composition.counts(formula_units)
        if sum(counts.values()) > max_atoms:
            rejected["atom_limit"] += 1
            continue
        species = list(counts)
        num_ions = [counts[element] for element in species]
        if PyXtalGroup is not None:
            group = PyXtalGroup(
                recipe.space_group.hall_number or recipe.space_group.number,
                use_hall=recipe.space_group.hall_number is not None,
            )
            compatible, _ = group.check_compatible(num_ions)
            if not compatible:
                rejected["orbit_incompatible"] += 1
                continue
        selected_formula_units = formula_units
        sites = None
        if recipe.search.fixed_orbits:
            sites = []
            for element in species:
                element_sites: dict[str, Any] = {}
                for orbit in recipe.search.fixed_orbits:
                    if orbit.element != element:
                        continue
                    if not orbit.wyckoff:
                        raise ValueError("search fixed_orbits require Wyckoff labels")
                    value = list(orbit.representative_fractional)
                    if orbit.wyckoff in element_sites:
                        existing = element_sites[orbit.wyckoff]
                        if existing and isinstance(existing[0], (int, float)):
                            element_sites[orbit.wyckoff] = [existing, value]
                        else:
                            existing.append(value)
                    else:
                        element_sites[orbit.wyckoff] = value
                sites.append(element_sites)
        while attempts < recipe.search.max_attempts:
            if len(accepted) >= recipe.search.candidates:
                break
            attempts += 1
            crystal = PyXtal()
            try:
                lattice = PyXtalLattice.from_para(*recipe.cell.parameters, ltype=lattice_type)
                crystal.from_random(
                    3,
                    recipe.space_group.hall_number or recipe.space_group.number,
                    species,
                    num_ions,
                    lattice=lattice,
                    sites=sites,
                    conventional=True,
                    max_count=1,
                    random_state=rng,
                    use_hall=recipe.space_group.hall_number is not None,
                )
                if not crystal.valid:
                    rejected["backend_invalid"] += 1; continue
                structure = crystal.to_pymatgen()
                metrics = _validate_candidate(structure, recipe, counts)
            except Exception as exc:
                message = str(exc).lower()
                if "pair distance" in message:
                    reason = "distance_conflict"
                elif "coordination" in message:
                    reason = "coordination_conflict"
                elif "composition" in message or "wyckoff" in message:
                    reason = "orbit_incompatible"
                elif "space group" in message or "centering" in message:
                    reason = "symmetry_mismatch"
                else:
                    reason = "backend_failure"
                rejected[reason] += 1
                continue
            if any(matcher.fit(structure, previous) for previous, _ in accepted):
                rejected["duplicate"] += 1; continue
            accepted.append((structure, metrics))
        if accepted:
            break
        if selected_formula_units is not None:
            break
    accepted.sort(key=lambda item: (-item[1]["distance_margin"], structure_sha256(item[0])))
    try:
        backend_version = version("pyxtal")
    except PackageNotFoundError:  # pragma: no cover - non-installed test doubles
        backend_version = "test-double"
    return accepted[: recipe.search.candidates], {
        "attempts": attempts,
        "max_attempts": recipe.search.max_attempts,
        "selected_formula_units": selected_formula_units,
        "rejected": dict(sorted(rejected.items())),
        "backend": {"name": "pyxtal", "version": backend_version},
    }


class SymmetryCrystalGenerator:
    defect_name = "symmetry_crystal"
    generator_version = "1.0.0"

    def generate(self, structure: None, params: SymmetryCrystalParams) -> GenerationResult:
        if structure is not None:
            raise ValueError("symmetry-crystal is a source-free generator")
        recipe = params.recipe
        canonical = json.dumps(recipe.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        recipe_hash = params.recipe_sha256 or hashlib.sha256(canonical).hexdigest()
        search_statistics: dict[str, Any] = {
            "attempts": 0,
            "max_attempts": 0,
            "selected_formula_units": recipe.composition.formula_units,
            "rejected": {},
            "backend": {"name": "deterministic-orbit-expansion", "version": self.generator_version},
        }
        if recipe.mode == "explicit":
            candidate, orbits = _expand_explicit(recipe)
            if len(candidate) > params.max_atoms_per_structure:
                raise ValueError(
                    f"expanded structure exceeds atom limit: "
                    f"{len(candidate)} > {params.max_atoms_per_structure}"
                )
            assert recipe.composition.formula_units is not None
            metrics = _validate_candidate(candidate, recipe, recipe.composition.counts(recipe.composition.formula_units))
            candidates = [(candidate, {**metrics, "orbits": orbits})]
        else:
            candidates, search_statistics = _search(recipe, params.max_atoms_per_structure)
            if not candidates:
                raise ValueError(
                    "symmetry search produced no valid candidates; "
                    f"statistics={search_statistics}"
                )
        generated: list[GeneratedStructure] = []
        for rank, (candidate, metrics) in enumerate(candidates, 1):
            identity = structure_sha256(candidate)
            generated.append(GeneratedStructure(
                structure=candidate,
                record=StructureRecord(
                    structure_id=identity,
                    parent_structure_id=f"recipe-{recipe_hash}",
                    formula=candidate.composition.reduced_formula,
                    n_atoms=len(candidate),
                    actual_parameters={
                        "geometry_status": "passed",
                        "relaxation_status": "unknown",
                        "rank": rank,
                        "score": metrics["distance_margin"],
                        "space_group_number": metrics["space_group_number"],
                        "space_group_symbol": metrics["space_group_symbol"],
                        "hall_number": recipe.space_group.hall_number,
                        "cell_setting": recipe.cell.setting,
                        "pair_min_A": metrics["pair_min_A"],
                        "orbits": metrics.get("orbits", []),
                        "centering_translations": metrics["centering_translations"],
                        "centering_residual_A": metrics["centering_residual_A"],
                        "recipe_sha256": recipe_hash,
                        "search_statistics": search_statistics,
                        "backend": search_statistics["backend"],
                    },
                    site_mapping={},
                ),
            ))
        warnings = []
        if recipe.mode == "search" and len(generated) < recipe.search.candidates:
            warnings.append(
                f"search returned {len(generated)} of {recipe.search.candidates} requested candidates; "
                f"statistics={search_statistics}"
            )
        return GenerationResult(
            defect_type=self.defect_name,
            input_count=0,
            skipped_count=sum(search_statistics["rejected"].values()),
            generated=generated,
            warnings=warnings,
            provenance=Provenance(
                generator=self.defect_name,
                generator_version=self.generator_version,
                input_source=params.recipe_source,
                input_structure_hash=recipe_hash,
                input_structure_hashes={"recipe": recipe_hash},
                parameters=params.model_dump(mode="json", by_alias=True),
                seed=recipe.search.seed if recipe.search is not None else None,
                created_at=datetime.now(timezone.utc),
            ),
        )
