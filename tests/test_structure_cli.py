"""Public local structure workflow; real small POSCARs, no remote services."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixScaled
from ase.io import read, write

from llm_matgen.__main__ import main


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'POSCAR'
    atoms = Atoms('SiOH', positions=[[0, 0, 0], [2, 2, 2], [4, 4, 4]], cell=[[8, 0, 0], [1, 8, 0], [0, 1, 8]], pbc=True)
    atoms.set_constraint([FixAtoms(indices=[0]), FixScaled(1, mask=[True, False, True])])
    write(path, atoms, format='vasp', direct=True, sort=False)
    return path


def invoke(args):
    try:
        return main(['structure', *map(str, args)])
    except SystemExit as exc:
        return exc.code


def revise(source, *mode, index=2, reason='人工校正', output=None, sidecar=None):
    args = ['adjust-coordinate', '--input', source, '--output', output or source.with_name('POSCAR.revised'), '--index', index, '--reason', reason, *mode]
    if sidecar is not None:
        args += ['--sidecar', sidecar]
    return invoke(args)


def test_public_inspect_and_validate_report_geometry(source, capsys, monkeypatch):
    def forbid_matrix(*args, **kwargs):
        raise AssertionError('全原子距离矩阵不应被调用')
    monkeypatch.setattr(Atoms, 'get_all_distances', forbid_matrix)
    for command in ['inspect', 'validate']:
        assert invoke([command, '--input', source]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report['structure']['atom_count'] == 3
        assert report['structure']['minimum_pair_distance_angstrom'] == pytest.approx(12 ** 0.5)
        assert report['validation_issues'] == []


@pytest.mark.parametrize('mode,values,want', [
    ('--set-cartesian', ['2.1', '2.2', '2.3'], [2.1, 2.2, 2.3]),
    ('--set-fractional', ['1.25', '-0.35', '0.45'], [9.65, -2.35, 3.6]),
    ('--delta-cartesian', ['0.1', '-0.2', '0.3'], [2.1, 1.8, 2.3]),
])
def test_three_modes_preserve_order_cell_constraints_and_hashes(source, mode, values, want):
    original = source.read_bytes()
    assert revise(source, mode, *values) == 0
    output = source.with_name('POSCAR.revised')
    before, after = read(source), read(output, format='vasp')
    np.testing.assert_allclose(after.positions[1], want, atol=1e-12)
    np.testing.assert_allclose(after.positions[[0, 2]], before.positions[[0, 2]], atol=1e-12)
    np.testing.assert_allclose(after.cell, before.cell, atol=1e-12)
    assert after.get_chemical_symbols() == ['Si', 'O', 'H']
    assert repr(before.constraints) == repr(after.constraints)
    assert source.read_bytes() == original
    payload = json.loads(Path(str(output) + '.revision.json').read_text(encoding='utf-8'))
    assert payload['parent_sha256'] == hashlib.sha256(original).hexdigest()
    assert payload['output_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert payload['index_1based'] == 2 and payload['element'] == 'O'
    assert payload['reason'] == '人工校正'
    np.testing.assert_allclose(payload['new_cartesian_angstrom'], want)


@pytest.mark.parametrize('index', [0, -1, 4])
def test_invalid_index_leaves_no_outputs(source, index):
    assert revise(source, '--delta-cartesian', '0', '0', '1', index=index) != 0
    assert sorted(p.name for p in source.parent.iterdir()) == ['POSCAR']


@pytest.mark.parametrize('mode', ['--set-cartesian', '--set-fractional', '--delta-cartesian'])
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_nonfinite_values_leave_no_outputs(source, mode, value):
    assert revise(source, mode, value, '0', '0') != 0
    assert sorted(p.name for p in source.parent.iterdir()) == ['POSCAR']


def test_mode_and_reason_are_required(source):
    assert revise(source) != 0
    assert revise(source, '--set-cartesian', '1', '2', '3', '--delta-cartesian', '1', '2', '3') != 0
    assert revise(source, '--delta-cartesian', '0', '0', '1', reason='  ') != 0
    assert sorted(p.name for p in source.parent.iterdir()) == ['POSCAR']


@pytest.mark.parametrize('conflict', ['input', 'output', 'sidecar', 'same-output-sidecar'])
def test_existing_files_and_path_aliases_never_overwritten(source, conflict):
    original = source.read_bytes()
    output = source.with_name('POSCAR.revised')
    sidecar = Path(str(output) + '.revision.json')
    if conflict == 'input':
        output = source
    elif conflict in ['output', 'sidecar']:
        (output if conflict == 'output' else sidecar).write_bytes(b'existing')
    else:
        sidecar = output
    assert revise(source, '--delta-cartesian', '0', '0', '1', output=output, sidecar=sidecar) != 0
    assert source.read_bytes() == original
    if conflict in ['output', 'sidecar']:
        assert (output if conflict == 'output' else sidecar).read_bytes() == b'existing'
    if conflict != 'output':
        assert not source.with_name('POSCAR.revised').exists()


def test_overlap_validate_fails_but_inspect_reports(source, capsys):
    atoms = read(source)
    atoms.positions[1] = [0.1, 0, 0]
    write(source, atoms, format='vasp', direct=True, sort=False)
    assert invoke(['validate', '--input', source]) != 0
    assert invoke(['inspect', '--input', source]) == 0
    assert '0.1000' in capsys.readouterr().out


def test_sidecar_publish_failure_rolls_back_new_structure(source, monkeypatch):
    from llm_matgen import structure
    original_link = structure.os.link
    def fail_sidecar(src, dst):
        if str(dst).endswith('.revision.json'):
            raise OSError('simulated sidecar publication failure')
        return original_link(src, dst)
    monkeypatch.setattr(structure.os, 'link', fail_sidecar)
    assert revise(source, '--delta-cartesian', '0', '0', '0.2') != 0
    assert sorted(p.name for p in source.parent.iterdir()) == ['POSCAR']


def test_new_sidecar_import_creates_unapproved_revision(source, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert revise(source, '--delta-cartesian', '0', '0', '0.2') == 0
    capsys.readouterr()
    output = source.with_name('POSCAR.revised')
    assert main(['revision', 'import', '--revised', str(output), '--sidecar', str(output) + '.revision.json', '--output-root', str(tmp_path / 'revisions')]) == 0
    result = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
    assert manifest['changed_atom_index_1based'] == 2
    assert manifest['scientific_approval'] == 'required'


def test_build_identity_available_to_project(capsys):
    try:
        status = main(['build-info'])
    except SystemExit as exc:
        status = exc.code
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['upstream_commit'] == 'f7c6c371dfcfa4e07c2224d099dea704ed031a67'
    assert payload['package_version']
    assert len(payload['local_changes_sha256']) == 64
    assert payload['verification'] == 'verified'


@pytest.mark.parametrize('mutation', ['modify', 'add', 'delete'])
def test_build_identity_detects_actual_package_changes(tmp_path, monkeypatch, mutation):
    import shutil
    from llm_matgen import build_identity
    root = Path(build_identity.__file__).resolve().parents[1]
    for package in ['llm_matgen', 'dft_structure_building']:
        shutil.copytree(root / package, tmp_path / package, ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(build_identity, '__file__', str(tmp_path / 'llm_matgen/build_identity.py'))
    before = build_identity.get_build_info()
    assert before['verification'] == 'verified'
    target = tmp_path / 'llm_matgen/structure.py'
    if mutation == 'modify':
        target.write_bytes(target.read_bytes() + b'\n# changed\n')
    elif mutation == 'add':
        (tmp_path / 'llm_matgen/new_local.py').write_text('pass\n')
    else:
        target.unlink()
    after = build_identity.get_build_info()
    assert after['verification'] == 'modified'
    assert before['source_sha256'] != after['source_sha256']
    assert before['local_changes_sha256'] != after['local_changes_sha256']


@pytest.mark.parametrize('value', ['nan', 'inf'])
def test_nonfinite_source_is_rejected_without_revision(source, value):
    atoms = read(source)
    atoms.positions[1, 0] = float(value)
    write(source, atoms, format='vasp', direct=False, sort=False)
    assert invoke(['validate', '--input', source]) != 0
    assert revise(source, '--delta-cartesian', '0', '0', '1') != 0
    assert sorted(p.name for p in source.parent.iterdir()) == ['POSCAR']
