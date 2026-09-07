"""Deterministic structure identity and site lineage helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from pymatgen.core import Structure

_DECIMALS = 10
_HASH_VERSIONS = {"v1", "v2"}
_VASP_SEMANTIC_SITE_PROPERTIES = {
    "selective_dynamics",
    "magmom",
    "charge",
    "velocities",
    "predictor_corrector",
}


def _rounded(values: Any) -> Any:
    return np.asarray(values, dtype=float).round(_DECIMALS).tolist()


def _canonical_property(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_canonical_property(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_property(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            raise ValueError("structure identity properties must contain finite numbers")
        return round(float(value), _DECIMALS)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def canonical_structure_payload(
    structure: Structure, *, version: str = "v1"
) -> dict[str, Any]:
    """Return an order-sensitive, translation-normalized JSON payload."""
    if version not in _HASH_VERSIONS:
        raise ValueError(f"unsupported structure hash version: {version}")
    sites = []
    semantic_names = sorted(
        _VASP_SEMANTIC_SITE_PROPERTIES & set(structure.site_properties)
    )
    for index, site in enumerate(structure):
        species = {
            str(element): round(float(occupancy), _DECIMALS)
            for element, occupancy in sorted(site.species.items(), key=lambda item: str(item[0]))
        }
        payload = {
            "species": species,
            "frac_coords": _rounded(np.mod(site.frac_coords, 1.0)),
        }
        if version == "v2":
            payload["vasp_site_properties"] = {
                name: _canonical_property(structure.site_properties[name][index])
                for name in semantic_names
            }
        sites.append(payload)
    result = {
        "lattice": _rounded(structure.lattice.matrix),
        "sites": sites,
    }
    if version == "v2":
        result["hash_schema"] = "llm-matgen-structure-v2"
        result["pbc"] = [bool(item) for item in structure.lattice.pbc]
    return result


def structure_sha256(structure: Structure, *, version: str = "v1") -> str:
    payload = canonical_structure_payload(structure, version=version)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def versioned_structure_hashes(structure: Structure) -> dict[str, str]:
    return {version: structure_sha256(structure, version=version) for version in ("v1", "v2")}


def assign_site_ids(structure: Structure, parent_structure_id: str) -> list[str]:
    """Assign stable IDs based on parent identity, order, and species."""
    parent_token = parent_structure_id[:12]
    return [
        f"site-{parent_token}-{index:05d}-{str(site.specie)}"
        for index, site in enumerate(structure)
    ]
