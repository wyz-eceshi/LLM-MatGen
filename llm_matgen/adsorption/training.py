"""Replaceable future-training boundaries; no model implementation lives here."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Protocol, runtime_checkable

from llm_matgen.adsorption.models import AdsorptionCaseRevision
from llm_matgen.adsorption.proposals import AdsorptionProposal


@runtime_checkable
class DatasetBuilder(Protocol):
    dataset_version: str

    def build(self, cases: Iterable[AdsorptionCaseRevision]) -> Any: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    feature_version: str

    def extract(self, dataset: Any) -> Any: ...


@runtime_checkable
class Trainer(Protocol):
    model_version: str

    def train(self, features: Any) -> Any: ...


@runtime_checkable
class ModelProposalSource(Protocol):
    model_version: str

    def proposals(self) -> Iterator[AdsorptionProposal]: ...

