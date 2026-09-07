"""Portable, offline ball-and-stick previews of generated structures."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ViewerArtifact:
    path: Path
    sha256: str


def structure_payload(label, structure):
    if not structure.is_ordered:
        raise ValueError('结构查看暂不支持部分占位，请先生成有序构型')
    xyz = np.asarray(structure.cart_coords, dtype=float)
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(structure.lattice.matrix).all():
        raise ValueError('结构查看需要非空且坐标、晶格有效的结构')
    if len(xyz) > 20000:
        raise ValueError('单个预览最多支持 20000 个原子')
    symbols = [site.specie.symbol for site in structure]
    radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols])
    bonds = [[] for _ in xyz]
    tree = cKDTree(xyz)
    # Display only intra-cell bonds; never draw long false bonds across the cell.
    for i, position in enumerate(xyz):
        near = tree.query_ball_point(position, 1.2 * (radii[i] + radii.max()))
        if len(near) > 512:
            raise ValueError('原子过于密集，无法安全生成球棍预览')
        for j in near:
            if j > i and .1 < np.linalg.norm(position - xyz[j]) <= 1.2 * (radii[i] + radii[j]):
                bonds[i].append(j)
                bonds[j].append(i)
    fixed = structure.site_properties.get('selective_dynamics', [[True] * 3] * len(xyz))
    return {'label': str(label), 'formula': structure.composition.reduced_formula,
            'cell': structure.lattice.matrix.tolist(),
            'atoms': [{'element': s, 'number': i + 1, 'position': xyz[i].tolist(),
                       'bonds': bonds[i], 'fixed': not any(fixed[i])} for i, s in enumerate(symbols)]}


def write_viewer(structures, path: Path) -> ViewerArtifact:
    """Embed coordinates and the pinned renderer; no server or network needed."""
    structures = list(structures)
    if not structures or sum(len(s) for _, s in structures) > 200000:
        raise ValueError('预览需要候选结构，且总原子数不得超过 200000')
    data = json.dumps({'entries': [structure_payload(label, s) for label, s in structures]},
                      ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    data = data.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    assets = files(__package__).joinpath('assets')
    library = assets.joinpath('3Dmol-min.js').read_text(encoding='utf-8').replace('</script', '<\\/script')
    template = assets.joinpath('viewer.html').read_text(encoding='utf-8')
    import html as html_module
    license_text = html_module.escape(assets.joinpath('3Dmol-LICENSE.txt').read_text(encoding='utf-8'))
    html = template.replace('__RENDERER__', library).replace('__STRUCTURE_DATA__', data)
    html = html.replace('</footer>', '</footer><details><summary>第三方许可证</summary><pre>' + license_text + '</pre></details>')
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(html, encoding='utf-8', newline='\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return ViewerArtifact(path, hashlib.sha256(path.read_bytes()).hexdigest())
