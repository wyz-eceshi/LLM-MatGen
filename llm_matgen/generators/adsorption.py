"""Typed contracts for adsorption initial-configuration generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, computed_field, field_validator, model_validator
from pymatgen.core import Element, Lattice, Molecule, Structure

from llm_matgen.adsorption.extractor import canonical_adsorbate_graph
from llm_matgen.adsorption.models import CaseFeatureSet, RetrievalTrace
from llm_matgen.adsorption.proposals import (
    AdsorptionProposal,
    AlgorithmicProposalSource,
    HistoryRequiredError,
    HistoryResolution,
    ProposalStreamAudit,
    RetrievedProposalSource,
    bounded_proposal_stream,
    place_adsorbate_rigid,
    resolve_history,
)
from llm_matgen.adsorption.store import AdsorptionCaseStore
from llm_matgen.adsorption.validation import (
    AdsorptionCandidateValidator,
    AdsorptionValidationReport,
    apply_fixed_bottom_layers,
)
from llm_matgen.generators.models import (
    BaseGenerationParams,
    GeneratedStructure,
    GenerationResult,
    Provenance,
    StructureRecord,
)
from llm_matgen.utils.structure import assign_site_ids, structure_sha256


_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07}


def _radius(symbol: str) -> float:
    return _RADII.get(symbol, 1.25)


class AdsorptionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    clean_slab: Structure
    adsorbate: Molecule
    gas_reference: Molecule | Structure | None = None
    input_source: str = "in-memory"
    input_structure_hash: str | None = None
    anchor_index: PositiveInt = 1
    charge: int | None = None
    spin_multiplicity: PositiveInt = 1
    reference_axis: tuple[float, float, float] | None = None
    denticity: PositiveInt = 1
    rigid: bool = True

    @field_validator("reference_axis")
    @classmethod
    def _axis(cls, value):
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)) or np.linalg.norm(array) <= 1.0e-12:
            raise ValueError("reference axis must be a non-zero finite vector")
        return tuple(float(item) for item in array)

    @model_validator(mode="after")
    def _supported_adsorbate(self):
        if self.anchor_index > len(self.adsorbate):
            raise ValueError("anchor index is outside adsorbate")
        if self.denticity != 1:
            raise ValueError("only a single-anchor adsorbate is supported")
        if not self.rigid:
            raise ValueError("only rigid adsorbates are supported")
        if len(self.adsorbate) > 1 and self.reference_axis is None:
            raise ValueError("multi-atom adsorbate requires a reference axis")
        molecule_charge = int(round(float(self.adsorbate.charge)))
        if self.charge is None:
            self.charge = molecule_charge
        elif self.charge != molecule_charge:
            raise ValueError("charge must match the adsorbate Molecule")
        if self.spin_multiplicity != self.adsorbate.spin_multiplicity:
            raise ValueError("spin multiplicity must match the adsorbate Molecule")
        if len(self.adsorbate) > 1:
            adjacency = {index: set() for index in range(len(self.adsorbate))}
            for left in range(len(self.adsorbate)):
                for right in range(left + 1, len(self.adsorbate)):
                    symbols = (str(self.adsorbate[left].specie), str(self.adsorbate[right].specie))
                    if self.adsorbate.get_distance(left, right) <= 1.25 * sum(_radius(item) for item in symbols):
                        adjacency[left].add(right)
                        adjacency[right].add(left)
            seen: set[int] = set()
            stack = [0]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(adjacency[current] - seen)
            if len(seen) != len(self.adsorbate):
                raise ValueError("adsorbate bond graph must be connected")
        return self

    @computed_field
    @property
    def anchor_zero_based(self) -> int:
        return self.anchor_index - 1


class AdsorptionParams(BaseGenerationParams):
    history_policy: Literal["prefer", "require", "off"] = "prefer"
    slab_state: Literal["relaxed", "preview"] = "relaxed"
    surface_side: Literal["top", "bottom", "both"] = "top"
    site_types: tuple[str, ...] = ("top", "bridge", "hollow", "hollow4", "defect", "doped", "undercoordinated")
    explicit_sites: tuple[tuple[float, float, float], ...] = ()
    max_proposal_attempts: PositiveInt = 10_000
    case_index_revision: int | None = Field(default=None, ge=0)
    retrieval_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(default=5, ge=1, le=5)
    anchor_contact_window: tuple[float, float] = (0.75, 1.35)
    azimuths: tuple[float, ...] = (0.0,)
    tilts: tuple[float, ...] = (0.0,)
    rolls: tuple[float, ...] = (0.0,)
    heights: tuple[float, ...] = (1.8,)
    fixed_bottom_layers: int = Field(default=0, ge=0, le=100)
    coverage: float | None = Field(default=None, gt=0.0, le=1.0)
    gas_reference_box: float = Field(default=20.0, gt=5.0, le=100.0)
    min_vacuum_each_side: float = Field(default=2.5, ge=0.0, le=50.0)

    @field_validator("azimuths", "tilts", "rolls", "heights")
    @classmethod
    def _finite_nonempty(cls, value):
        if not value or not all(math.isfinite(float(item)) for item in value):
            raise ValueError("pose sets must be non-empty and finite")
        return tuple(float(item) for item in value)

    @field_validator("explicit_sites")
    @classmethod
    def _finite_explicit_sites(cls, value):
        if any(
            not all(math.isfinite(float(item)) for item in position)
            for position in value
        ):
            raise ValueError("explicit Cartesian sites must contain finite values")
        return tuple(tuple(float(item) for item in position) for position in value)

    @model_validator(mode="after")
    def _bounds(self):
        if self.max_proposal_attempts < self.max_structures:
            raise ValueError("max_proposal_attempts must not be below max_structures")
        if self.max_proposal_attempts > 1_000_000 or self.max_structures > 100_000:
            raise ValueError("adsorption generation bounds are too large")
        lower, upper = self.anchor_contact_window
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
            raise ValueError("anchor contact window must be finite and increasing")
        allowed_sites = {
            "top",
            "bridge",
            "hollow",
            "hollow4",
            "defect",
            "doped",
            "undercoordinated",
            "explicit",
        }
        if not self.site_types or not set(self.site_types) <= allowed_sites:
            raise ValueError("site_types contains an unsupported adsorption site type")
        if any(value <= 0.0 for value in self.heights):
            raise ValueError("adsorption heights must be positive")
        if any(not 0.0 <= value <= 180.0 for value in self.tilts):
            raise ValueError("tilts must be within 0..180 degrees")
        return self


class DFTHandoffMatrix(BaseModel):
    """Method-consistency checklist only; it contains no energy or run action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_groups: tuple[
        Literal["adsorbed"], Literal["clean_slab"], Literal["gas_reference"]
    ] = ("adsorbed", "clean_slab", "gas_reference")
    fixed_bottom_layers: int = Field(ge=0)
    coverage: float | None = Field(default=None, gt=0.0)
    sidedness: Literal["single", "double"]
    surface_sides_sampled: tuple[Literal["top", "bottom"], ...]
    dipole_correction_recommended: bool
    dispersion_sensitive: bool
    magnetic_sensitive: bool
    plus_u_sensitive: bool
    same_method_required: bool = True
    gas_reference_strategy: Literal["user_provided", "periodic_box"]
    gas_reference_vacuum: float | None = Field(default=None, gt=0.0)
    gas_charge: int
    gas_spin_multiplicity: PositiveInt
    notes: tuple[str, ...] = ()


@dataclass
class AdsorptionGenerationResult(GenerationResult):
    retrieval_trace: RetrievalTrace | None = None
    clean_slab: Structure | None = None
    adsorbate: Molecule | None = None
    gas_reference: Molecule | Structure | None = None
    dft_handoff: DFTHandoffMatrix | None = None
    proposal_audit: ProposalStreamAudit | None = None
    validation_reports: list[AdsorptionValidationReport] = field(default_factory=list)
    configuration_status: Literal["initial_configuration"] = "initial_configuration"

    def combine(self, other: GenerationResult) -> "AdsorptionGenerationResult":
        if not isinstance(other, AdsorptionGenerationResult):
            raise ValueError("cannot combine adsorption result with an untyped generation result")
        if self.defect_type != other.defect_type:
            raise ValueError("cannot combine results with different defect_type")
        if self.clean_slab is None or other.clean_slab is None or (
            structure_sha256(self.clean_slab) != structure_sha256(other.clean_slab)
        ):
            raise ValueError("cannot combine adsorption results from different clean slabs")
        if self.adsorbate is None or other.adsorbate is None or (
            _molecule_digest(self.adsorbate) != _molecule_digest(other.adsorbate)
        ):
            raise ValueError("cannot combine adsorption results from different adsorbates")
        if _gas_reference_digest(self.gas_reference) != _gas_reference_digest(
            other.gas_reference
        ):
            raise ValueError(
                "cannot combine adsorption results with different gas references"
            )
        if self.retrieval_trace != other.retrieval_trace or self.dft_handoff != other.dft_handoff:
            raise ValueError("cannot combine adsorption results with different retrieval or DFT contracts")
        if self.proposal_audit is None or other.proposal_audit is None:
            audit = None
        else:
            audit = ProposalStreamAudit(
                attempted=self.proposal_audit.attempted + other.proposal_audit.attempted,
                accepted=self.proposal_audit.accepted + other.proposal_audit.accepted,
                rejected=self.proposal_audit.rejected + other.proposal_audit.rejected,
                truncated=self.proposal_audit.truncated or other.proposal_audit.truncated,
            )
        return AdsorptionGenerationResult(
            defect_type=self.defect_type,
            input_count=self.input_count + other.input_count,
            skipped_count=self.skipped_count + other.skipped_count,
            generated=[*self.generated, *other.generated],
            warnings=[*self.warnings, *other.warnings],
            provenance=self.provenance or other.provenance,
            retrieval_trace=self.retrieval_trace,
            clean_slab=self.clean_slab.copy(),
            adsorbate=self.adsorbate.copy(),
            gas_reference=None if self.gas_reference is None else self.gas_reference.copy(),
            dft_handoff=self.dft_handoff,
            proposal_audit=audit,
            validation_reports=[*self.validation_reports, *other.validation_reports],
        )


def _molecule_digest(molecule: Molecule) -> str:
    payload = {
        "species": [str(site.specie) for site in molecule],
        "coordinates": np.asarray(molecule.cart_coords, dtype=float).round(10).tolist(),
        "charge": int(round(float(molecule.charge))),
        "spin_multiplicity": molecule.spin_multiplicity,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _gas_reference_digest(value: Molecule | Structure | None) -> str | None:
    if isinstance(value, Molecule):
        return f"molecule:{_molecule_digest(value)}"
    if isinstance(value, Structure):
        return f"structure:{structure_sha256(value)}"
    return None


def _query_features(value: AdsorptionInput, proposal: AdsorptionProposal) -> CaseFeatureSet:
    lattice = np.eye(3) * 50.0
    coordinates = np.asarray(value.adsorbate.cart_coords, dtype=float)
    fractional = (coordinates - coordinates[value.anchor_zero_based] + 25.0) / 50.0
    species = tuple(str(site.specie) for site in value.adsorbate)
    graph = canonical_adsorbate_graph(
        species,
        fractional,
        list(range(len(value.adsorbate))),
        lattice,
    )
    surface_indices = tuple(getattr(proposal.evidence, "surface_atom_indices", ()))
    if not surface_indices:
        distances = [
            np.linalg.norm(np.asarray(site.coords) - np.asarray(proposal.frame.origin))
            for site in value.clean_slab
        ]
        minimum = min(distances)
        surface_indices = tuple(
            index for index, distance in enumerate(distances) if distance <= minimum + 0.35
        )
    shell: dict[str, int] = {}
    for index in surface_indices:
        symbol = str(value.clean_slab[index].specie)
        shell[symbol] = shell.get(symbol, 0) + 1
    return CaseFeatureSet(
        adsorbate_formula=value.adsorbate.composition.alphabetical_formula.replace(" ", ""),
        adsorbate_graph_fingerprint=graph.fingerprint,
        anchor_element=str(value.adsorbate[value.anchor_zero_based].specie),
        denticity=1,
        active_site_elements=sorted(shell),
        local_coordination_signature=";".join(
            f"{symbol}:{count}" for symbol, count in sorted(shell.items())
        ),
        surface_side=proposal.surface_side,
        surface_normal=proposal.frame.normal,
        initial_height=proposal.height,
        initial_site_type=proposal.site_type,
        first_coordination_shell=shell,
        local_distance_scale=proposal.height,
        coverage=proposal.coverage,
    )


def _periodic_gas_reference(molecule: Molecule, box_length: float) -> Structure:
    coordinates = np.asarray(molecule.cart_coords, dtype=float)
    centered = coordinates - np.mean(coordinates, axis=0) + box_length / 2.0
    structure = Structure(
        Lattice.cubic(box_length),
        [site.specie for site in molecule],
        centered,
        coords_are_cartesian=True,
        to_unit_cell=True,
        site_properties=molecule.site_properties,
    )
    structure.add_site_property("gas_reference", [True] * len(structure))
    return structure


def _candidate_from_proposal(
    slab: Structure,
    molecule: Molecule,
    proposal: AdsorptionProposal,
    *,
    anchor_zero_based: int,
    reference_axis: tuple[float, float, float] | None,
) -> tuple[Structure, dict[str, object]]:
    axis = reference_axis or (0.0, 0.0, 1.0)
    if proposal.local_adsorbate_coordinates is not None and proposal.local_delta is None:
        if len(proposal.local_adsorbate_coordinates) != len(molecule):
            raise ValueError("retrieved final pose atom count differs from target adsorbate")
        origin = np.asarray(proposal.frame.origin) + proposal.height * np.asarray(
            proposal.frame.normal
        )
        basis = np.vstack(
            [proposal.frame.tangent1, proposal.frame.tangent2, proposal.frame.normal]
        )
        local = np.asarray(proposal.local_adsorbate_coordinates, dtype=float)
        coordinates = local @ basis + origin
        placed = Molecule(
            [site.specie for site in molecule],
            coordinates,
            charge=molecule.charge,
            spin_multiplicity=molecule.spin_multiplicity,
            site_properties=molecule.site_properties,
        )
        transform = {
            "kind": "retrieved_final_local_pose",
            "local_coordinates": [list(row) for row in proposal.local_adsorbate_coordinates],
        }
    else:
        azimuth = proposal.azimuth + (proposal.rotation_degrees or 0.0)
        placed, rigid = place_adsorbate_rigid(
            molecule,
            anchor_zero_based,
            axis,
            proposal.frame,
            proposal.height,
            azimuth,
            proposal.tilt,
            proposal.roll,
        )
        coordinates = np.asarray(placed.cart_coords, dtype=float)
        if proposal.local_delta is not None:
            basis = np.vstack(
                [proposal.frame.tangent1, proposal.frame.tangent2, proposal.frame.normal]
            )
            coordinates = coordinates + np.asarray(proposal.local_delta) @ basis
            placed = Molecule(
                [site.specie for site in molecule],
                coordinates,
                charge=molecule.charge,
                spin_multiplicity=molecule.spin_multiplicity,
                site_properties=molecule.site_properties,
            )
        transform = {
            "kind": "rigid_body_local_delta" if proposal.local_delta is not None else "rigid_body",
            **rigid.model_dump(mode="json"),
            "local_delta": None
            if proposal.local_delta is None
            else list(proposal.local_delta),
            "history_rotation_degrees": proposal.rotation_degrees,
        }
    candidate = slab.copy()
    for index, site in enumerate(placed):
        properties = {
            name: values[index] for name, values in placed.site_properties.items()
        }
        candidate.append(site.specie, site.coords, coords_are_cartesian=True, properties=properties)
    return candidate, transform


class AdsorptionGenerator:
    defect_name = "adsorption"
    generator_version = "0.1.0"

    def __init__(self, *, store_factory=None):
        self.store_factory = store_factory or AdsorptionCaseStore

    @staticmethod
    def _source(value: AdsorptionInput, params: AdsorptionParams) -> AlgorithmicProposalSource:
        sides = ("top", "bottom") if params.surface_side == "both" else (params.surface_side,)
        return AlgorithmicProposalSource(
            value.clean_slab,
            sides=sides,
            site_types=params.site_types,
            explicit_sites=params.explicit_sites,
            heights=params.heights,
            azimuths=params.azimuths,
            tilts=params.tilts,
            rolls=params.rolls,
            coverage=params.coverage,
        )

    def generate(
        self,
        value: AdsorptionInput,
        params: AdsorptionParams,
    ) -> AdsorptionGenerationResult:
        probe = next(self._source(value, params).proposals(), None)
        if probe is None:
            raise ValueError("adsorption site discovery produced no algorithmic sites")
        resolution: HistoryResolution = resolve_history(
            history_policy=params.history_policy,
            query_features=_query_features(value, probe),
            store_factory=self.store_factory,
            index_revision=params.case_index_revision,
            threshold=params.retrieval_threshold,
            limit=params.retrieval_top_k,
        )
        revisions = {revision.revision_id: revision for revision in resolution.revisions}
        retrieval_trace = resolution.trace
        history_warnings: list[str] = []
        retrieved = iter(())
        if resolution.trace.matches:
            try:
                retrieved = RetrievedProposalSource(
                    self._source(value, params).proposals(),
                    resolution.trace.matches,
                    revision_loader=lambda revision_id: revisions[revision_id],
                    index_revision=resolution.index_revision,
                ).proposals()
            except ValueError as exc:
                message = f"matched history proposal is unavailable: {exc}"
                if params.history_policy == "require":
                    raise HistoryRequiredError(message) from exc
                fallback = f"history_proposal_unavailable:{type(exc).__name__}"
                retrieval_trace = resolution.trace.model_copy(
                    update={"fallback_reason": fallback}
                )
                history_warnings.append(message)
        validator = AdsorptionCandidateValidator(
            value.clean_slab,
            value.adsorbate,
            anchor_zero_based=value.anchor_zero_based,
            anchor_contact_window=params.anchor_contact_window,
            min_vacuum_each_side=params.min_vacuum_each_side,
            fixed_bottom_layers=params.fixed_bottom_layers,
            expected_coverage=params.coverage,
        )
        parent_id = structure_sha256(value.clean_slab)
        parent_sites = assign_site_ids(value.clean_slab, parent_id)
        generated: list[GeneratedStructure] = []
        reports: list[AdsorptionValidationReport] = []
        placement_warnings: list[str] = []

        def accept(proposal: AdsorptionProposal) -> bool:
            try:
                candidate, transform = _candidate_from_proposal(
                    value.clean_slab,
                    value.adsorbate,
                    proposal,
                    anchor_zero_based=value.anchor_zero_based,
                    reference_axis=value.reference_axis,
                )
            except ValueError as exc:
                placement_warnings.append(
                    f"proposal {proposal.proposal_id} rejected during local placement: {exc}"
                )
                return False
            if len(candidate) > params.max_atoms_per_structure:
                raise ValueError(
                    f"adsorption atom limit exceeded: {len(candidate)} > {params.max_atoms_per_structure}"
                )
            candidate, fixed_layers = apply_fixed_bottom_layers(
                candidate,
                slab_atom_count=len(value.clean_slab),
                fixed_bottom_layers=params.fixed_bottom_layers,
            )
            report = validator.validate(candidate, proposal)
            reports.append(report)
            if not report.accepted:
                return False
            child_id = structure_sha256(candidate)
            evidence = proposal.evidence.model_dump(mode="json")
            generated.append(
                GeneratedStructure(
                    structure=candidate,
                    record=StructureRecord(
                        structure_id=child_id,
                        parent_structure_id=parent_id,
                        formula=candidate.composition.reduced_formula,
                        n_atoms=len(candidate),
                        actual_parameters={
                            "configuration_status": "initial_configuration",
                            "proposal_source": proposal.source,
                            "proposal_id": proposal.proposal_id,
                            "proposal_evidence": evidence,
                            "case_revision": evidence.get("revision_id"),
                            "site_id": proposal.site_id,
                            "site_type": proposal.site_type,
                            "surface_side": proposal.surface_side,
                            "surface_normal": list(proposal.frame.normal),
                            "anchor_index_1_based": value.anchor_index,
                            "height": proposal.height,
                            "azimuth": proposal.azimuth,
                            "tilt": proposal.tilt,
                            "roll": proposal.roll,
                            "pose_transform": transform,
                            "coverage": proposal.coverage,
                            "slab_state": params.slab_state,
                            "freeze_policy": {
                                "fixed_bottom_layers": params.fixed_bottom_layers,
                                "fixed_layer_atom_indices": [
                                    list(layer) for layer in fixed_layers
                                ],
                                "adsorbate_movable": True,
                            },
                            "index_revision": resolution.index_revision,
                        },
                        site_mapping={site_id: site_id for site_id in parent_sites},
                    ),
                )
            )
            return True

        _, audit = bounded_proposal_stream(
            self._source(value, params).proposals(),
            retrieved,
            max_attempts=params.max_proposal_attempts,
            max_accepted=params.max_structures,
            accept=accept,
        )
        warnings = [*history_warnings, *placement_warnings]
        if params.slab_state == "preview":
            warnings.append(
                "preview slab selected: generated structures are provisional initial configurations"
            )
        if audit.attempted >= params.max_proposal_attempts:
            warnings.append("adsorption proposal attempt budget exhausted before acceptance limit")
        elif audit.accepted >= params.max_structures:
            warnings.append("adsorption candidates truncated by max_structures")
        gas_reference: Molecule | Structure
        if value.gas_reference is not None:
            gas_reference = value.gas_reference.copy()
            gas_strategy = "user_provided"
            gas_vacuum = None
            gas_notes = ()
            if isinstance(value.gas_reference, Molecule):
                gas_charge = int(round(float(value.gas_reference.charge)))
                gas_spin_multiplicity = value.gas_reference.spin_multiplicity
            else:
                gas_charge = value.charge or 0
                gas_spin_multiplicity = value.spin_multiplicity
        else:
            gas_reference = _periodic_gas_reference(
                value.adsorbate, params.gas_reference_box
            )
            gas_strategy = "periodic_box"
            gas_vacuum = params.gas_reference_box
            gas_charge = value.charge or 0
            gas_spin_multiplicity = value.spin_multiplicity
            gas_notes = (
                "Periodic gas-reference vacuum, spin, and energy convergence still require workflow confirmation.",
            )
        elements = {Element(str(site.specie)) for site in value.clean_slab}
        dft_handoff = DFTHandoffMatrix(
            fixed_bottom_layers=params.fixed_bottom_layers,
            coverage=params.coverage,
            sidedness="single",
            surface_sides_sampled=("top", "bottom")
            if params.surface_side == "both"
            else (params.surface_side,),
            dipole_correction_recommended=True,
            dispersion_sensitive=True,
            magnetic_sensitive=value.spin_multiplicity > 1
            or any(element.is_transition_metal for element in elements),
            plus_u_sensitive=any(
                element.is_transition_metal or element.is_rare_earth_metal
                for element in elements
            ),
            gas_reference_strategy=gas_strategy,
            gas_reference_vacuum=gas_vacuum,
            gas_charge=gas_charge,
            gas_spin_multiplicity=gas_spin_multiplicity,
            notes=gas_notes,
        )
        molecule_id = _molecule_digest(value.adsorbate)
        return AdsorptionGenerationResult(
            defect_type=self.defect_name,
            input_count=1,
            skipped_count=audit.rejected,
            generated=generated,
            warnings=warnings,
            provenance=Provenance(
                generator=self.defect_name,
                generator_version=self.generator_version,
                input_source=value.input_source,
                input_structure_hash=value.input_structure_hash or parent_id,
                input_structure_hashes={
                    "clean_slab": parent_id,
                    "adsorbate": molecule_id,
                },
                parameters=params.model_dump(mode="json"),
                seed=None,
                created_at=datetime.now(timezone.utc),
            ),
            retrieval_trace=retrieval_trace,
            clean_slab=value.clean_slab.copy(),
            adsorbate=value.adsorbate.copy(),
            gas_reference=gas_reference,
            dft_handoff=dft_handoff,
            proposal_audit=audit,
            validation_reports=reports,
        )
