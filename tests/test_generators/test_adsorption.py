from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Molecule, Structure

from llm_matgen.adsorption.store import AdsorptionStoreError


def slab() -> Structure:
    return Structure(
        Lattice([[0.0, 3.0, 0.0], [0.0, 1.5, 2.598076211], [18.0, 0.0, 0.0]]),
        ["Pt"],
        [[0.0, 0.0, 0.5]],
    )


def oxygen() -> Molecule:
    return Molecule(["O"], [[0.0, 0.0, 0.0]], charge=0, spin_multiplicity=3)


def inputs(*, gas_reference=None):
    from llm_matgen.generators.adsorption import AdsorptionInput

    return AdsorptionInput(
        clean_slab=slab(),
        adsorbate=oxygen(),
        gas_reference=gas_reference,
        anchor_index=1,
        charge=0,
        spin_multiplicity=3,
    )


def params(**updates):
    from llm_matgen.generators.adsorption import AdsorptionParams

    values = dict(
        history_policy="off",
        site_types=("top",),
        heights=(1.8,),
        max_structures=10,
        max_proposal_attempts=20,
        min_vacuum_each_side=3.0,
    )
    values.update(updates)
    return AdsorptionParams(**values)


@pytest.mark.parametrize(
    ("side", "expected_sides"),
    [("top", ["top"]), ("bottom", ["bottom"]), ("both", ["top", "bottom"])],
)
def test_generator_history_off_never_opens_store_and_handles_non_z_sides(side, expected_sides):
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    touched = False

    def forbidden_store():
        nonlocal touched
        touched = True
        raise AssertionError("history off must not instantiate the store")

    result = AdsorptionGenerator(store_factory=forbidden_store).generate(
        inputs(), params(surface_side=side)
    )

    assert not touched
    assert [item.record.actual_parameters["surface_side"] for item in result.generated] == expected_sides
    assert all(item.record.actual_parameters["configuration_status"] == "initial_configuration" for item in result.generated)
    assert all(
        np.isclose(abs(item.record.actual_parameters["surface_normal"][0]), 1.0)
        for item in result.generated
    )


def test_generator_prefer_falls_back_require_fails_and_preview_warns():
    from llm_matgen.adsorption.proposals import HistoryRequiredError
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    def broken_store():
        raise AdsorptionStoreError("database missing")

    prefer = AdsorptionGenerator(store_factory=broken_store).generate(
        inputs(), params(history_policy="prefer", slab_state="preview")
    )
    assert prefer.generated_count == 1
    assert prefer.retrieval_trace.fallback_reason.startswith("store_unavailable:")
    assert any("preview" in warning.lower() for warning in prefer.warnings)
    assert prefer.configuration_status == "initial_configuration"

    with pytest.raises(HistoryRequiredError, match="history is required"):
        AdsorptionGenerator(store_factory=broken_store).generate(
            inputs(), params(history_policy="require")
        )


@pytest.mark.parametrize("failure_kind", ["missing_revision", "malformed_pose"])
def test_generator_prefer_falls_back_when_matched_history_cannot_build_proposals(
    monkeypatch, failure_kind
):
    import llm_matgen.generators.adsorption as adsorption_module
    from llm_matgen.adsorption.models import (
        CaseFeatureSet,
        RetrievalMatch,
        RetrievalQuery,
        RetrievalTrace,
    )
    from llm_matgen.adsorption.proposals import HistoryResolution
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    match = RetrievalMatch(
        case_id="case-1",
        revision_id="rev-1",
        equivalent_fingerprint="fingerprint-1",
        total_score=0.9,
        component_scores={"site_type": 1.0},
        completeness=1.0,
        accepted=True,
    )
    trace = RetrievalTrace(
        index_revision=7,
        query=RetrievalQuery(index_revision=7, features=CaseFeatureSet()),
        matches=[match],
    )
    revisions = ()
    if failure_kind == "malformed_pose":
        revisions = (
            SimpleNamespace(
                revision_id="rev-1",
                features=CaseFeatureSet(
                    pose_correction={
                        "slab_indices": [0],
                        "adsorbate_indices": [1, 2],
                        "initial_local_coordinates": [[0.0, 0.0, 0.0]],
                        "final_local_coordinates": [[0.0, 0.0, 0.0]],
                        "local_delta": [0.0, 0.0, 0.0],
                    }
                ),
            ),
        )
    monkeypatch.setattr(
        adsorption_module,
        "resolve_history",
        lambda **_: HistoryResolution(7, trace, revisions),
    )

    result = AdsorptionGenerator().generate(
        inputs(), params(history_policy="prefer", max_structures=10)
    )

    assert result.generated_count == 1
    assert result.retrieval_trace.fallback_reason.startswith(
        "history_proposal_unavailable:"
    )


def test_generator_require_reports_malformed_matched_pose_as_history_error(monkeypatch):
    import llm_matgen.generators.adsorption as adsorption_module
    from llm_matgen.adsorption.models import (
        CaseFeatureSet,
        RetrievalMatch,
        RetrievalQuery,
        RetrievalTrace,
    )
    from llm_matgen.adsorption.proposals import HistoryRequiredError, HistoryResolution
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    match = RetrievalMatch(
        case_id="case-1",
        revision_id="rev-1",
        equivalent_fingerprint="fingerprint-1",
        total_score=0.9,
        component_scores={},
        completeness=1.0,
        accepted=True,
    )
    trace = RetrievalTrace(
        index_revision=7,
        query=RetrievalQuery(index_revision=7, features=CaseFeatureSet()),
        matches=[match],
    )
    revision = SimpleNamespace(
        revision_id="rev-1",
        features=CaseFeatureSet(
            pose_correction={"local_delta": [0.0, 0.0, 0.0]}
        ),
    )
    monkeypatch.setattr(
        adsorption_module,
        "resolve_history",
        lambda **_: HistoryResolution(7, trace, (revision,)),
    )

    with pytest.raises(
        HistoryRequiredError, match="matched history proposal is unavailable"
    ):
        AdsorptionGenerator().generate(
            inputs(), params(history_policy="require", max_structures=10)
        )


def test_attempt_budget_stops_before_accept_limit_and_is_audited():
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    result = AdsorptionGenerator().generate(
        inputs(),
        params(
            heights=(0.1, 1.8),
            max_structures=1,
            max_proposal_attempts=1,
        ),
    )

    assert result.generated_count == 0
    assert result.proposal_audit.attempted == 1
    assert result.proposal_audit.accepted == 0
    assert result.proposal_audit.truncated
    assert any("attempt budget" in warning.lower() for warning in result.warnings)


def test_user_gas_reference_has_priority_and_auto_reference_is_periodic_box():
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    user_gas = oxygen()
    user = AdsorptionGenerator().generate(inputs(gas_reference=user_gas), params())
    automatic = AdsorptionGenerator().generate(inputs(), params(gas_reference_box=22.0))

    assert isinstance(user.gas_reference, Molecule)
    assert user.dft_handoff.gas_reference_strategy == "user_provided"
    assert isinstance(automatic.gas_reference, Structure)
    assert np.allclose(automatic.gas_reference.lattice.abc, [22.0, 22.0, 22.0])
    assert automatic.dft_handoff.gas_reference_strategy == "periodic_box"
    assert automatic.dft_handoff.gas_reference_vacuum == 22.0
    assert any("workflow confirmation" in note.lower() for note in automatic.dft_handoff.notes)


def test_user_molecule_gas_reference_supplies_dft_charge_and_spin():
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    hydride = Molecule(
        ["H"], [[0.0, 0.0, 0.0]], charge=-1, spin_multiplicity=1
    )
    result = AdsorptionGenerator().generate(
        inputs(gas_reference=hydride), params()
    )

    assert result.dft_handoff.gas_charge == -1
    assert result.dft_handoff.gas_spin_multiplicity == 1


def test_dft_handoff_is_comparison_only_and_public_api_is_exported():
    from llm_matgen.generators import (
        AdsorptionGenerationResult,
        AdsorptionGenerator,
        AdsorptionInput,
        AdsorptionParams,
        DFTHandoffMatrix,
    )

    result = AdsorptionGenerator().generate(inputs(), params(surface_side="both"))
    payload = result.dft_handoff.model_dump(mode="json")

    assert isinstance(result, AdsorptionGenerationResult)
    assert isinstance(result.dft_handoff, DFTHandoffMatrix)
    assert result.dft_handoff.comparison_groups == (
        "adsorbed",
        "clean_slab",
        "gas_reference",
    )
    # ``surface_side=both`` samples two independent one-sided structures; it
    # must not be advertised as simultaneous double-sided adsorption.
    assert result.dft_handoff.sidedness == "single"
    assert result.dft_handoff.surface_sides_sampled == ("top", "bottom")
    assert result.dft_handoff.dipole_correction_recommended
    assert result.dft_handoff.same_method_required
    forbidden = {"potcar", "energy", "adsorption_energy", "submit", "submission"}
    assert forbidden.isdisjoint(payload)
    assert AdsorptionInput is not None
    assert AdsorptionParams is not None


def test_typed_adsorption_result_combine_preserves_adsorption_contract():
    from llm_matgen.generators.adsorption import AdsorptionGenerationResult, AdsorptionGenerator

    first = AdsorptionGenerator().generate(inputs(), params())
    second = AdsorptionGenerator().generate(inputs(), params())
    combined = first.combine(second)

    assert isinstance(combined, AdsorptionGenerationResult)
    assert combined.generated_count == 2
    assert combined.retrieval_trace == first.retrieval_trace
    assert combined.dft_handoff == first.dft_handoff
    assert combined.proposal_audit.attempted == 2
    assert len(combined.validation_reports) == 2


def test_generic_pipeline_forces_poscar_mson_and_reference_artifacts(tmp_path):
    import json

    from llm_matgen.generators.adsorption import AdsorptionGenerator
    from llm_matgen.generators.models import OutputFormat
    from llm_matgen.io.exporters import ExportOptions
    from llm_matgen.pipeline import GenerationPipeline

    result = GenerationPipeline(tmp_path).run(
        AdsorptionGenerator(),
        inputs(),
        params(),
        ExportOptions(formats=[OutputFormat.POSCAR]),
        run_id="adsorption-pipeline",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.ok
    assert {item.format for item in result.artifacts} == {
        OutputFormat.POSCAR, OutputFormat.MSON
    }
    assert {item["role"] for item in manifest["reference_artifacts"]} == {
        "clean_slab", "adsorbate", "gas_reference"
    }
    assert manifest["retrieval_trace"]["fallback_reason"] == "history_disabled"
    assert manifest["dft_handoff"]["same_method_required"]


def test_typed_adsorption_result_combine_rejects_different_gas_references():
    from llm_matgen.generators.adsorption import AdsorptionGenerator

    atomic = oxygen()
    molecular = Molecule(
        ["O", "O"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]],
        charge=0,
        spin_multiplicity=3,
    )
    first = AdsorptionGenerator().generate(
        inputs(gas_reference=atomic), params()
    )
    second = AdsorptionGenerator().generate(
        inputs(gas_reference=molecular), params()
    )

    with pytest.raises(ValueError, match="gas references"):
        first.combine(second)


def test_training_protocols_accept_fakes_without_ml_dependencies():
    from llm_matgen.adsorption.training import (
        DatasetBuilder,
        FeatureExtractor,
        ModelProposalSource,
        Trainer,
    )

    class FakeDatasetBuilder:
        dataset_version = "test-dataset-v1"

        def build(self, cases):
            return tuple(cases)

    class FakeFeatureExtractor:
        feature_version = "test-feature-v1"

        def extract(self, dataset):
            return {"count": len(dataset)}

    class FakeTrainer:
        model_version = "test-model-v1"

        def train(self, features):
            return features

    class FakeModelProposalSource:
        model_version = "test-model-v1"

        def proposals(self):
            return iter(())

    assert isinstance(FakeDatasetBuilder(), DatasetBuilder)
    assert isinstance(FakeFeatureExtractor(), FeatureExtractor)
    assert isinstance(FakeTrainer(), Trainer)
    assert isinstance(FakeModelProposalSource(), ModelProposalSource)
    requirements = importlib.metadata.requires("llm-matgen") or []
    lowered = "\n".join(requirements).lower()
    assert "sklearn" not in lowered
    assert "torch" not in lowered
    assert "faiss" not in lowered
