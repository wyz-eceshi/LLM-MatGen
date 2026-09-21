"""读取随包交付的来源证据，并计算当前实际源码身份；不依赖 Git。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_matgen import __version__


def _hash_map(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    """仅枚举两个生产包；缓存及身份资源自身不参与循环哈希。"""
    result = {}
    for package in ('llm_matgen', 'dft_structure_building'):
        directory = root / package
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob('*')):
            if '__pycache__' in path.parts or path.name == '_build_identity.json' or path.suffix in {'.pyc', '.pyo'}:
                continue
            if path.is_file():
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def get_build_info() -> dict:
    """公开接口：当前源哈希与构建时证据不符时明确返回 modified。"""
    package = Path(__file__).resolve().parent
    current = _source_hashes(package.parent)
    resource = package / '_build_identity.json'
    try:
        recorded = json.loads(resource.read_text(encoding='utf-8'))
    except FileNotFoundError:
        recorded = {}
    upstream = recorded.get('upstream_package_files', {})
    changes = {path: {'upstream': upstream.get(path), 'current': current.get(path)}
               for path in sorted(upstream.keys() | current.keys())
               if upstream.get(path) != current.get(path)}
    expected = recorded.get('package_files', {})
    return {
        'schema': 'llm-matgen-build-identity',
        'version': 1,
        'package_version': __version__,
        'upstream_commit': recorded.get('upstream_commit'),
        'upstream_archive_sha256': recorded.get('upstream_archive_sha256'),
        'source_sha256': _hash_map(current),
        'local_changes_sha256': _hash_map(changes),
        'local_changed_files': sorted(changes),
        'verification': ('verified' if current == expected and __version__ == recorded.get('package_version')
                         else 'modified' if recorded else 'unavailable'),
        'resource': str(resource),
    }
