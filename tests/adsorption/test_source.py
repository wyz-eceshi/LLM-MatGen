from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import subprocess
import sys

import pytest


REMOTE_ROOT = "/srv/llm-matgen/cases"
SSH_WRAPPER = "ssh-dft-cluster.ps1"


class SnapshotRunner:
    def __init__(self, *, missing=(), mutate_on_second=()):
        self.missing = set(missing)
        self.mutate_on_second = set(mutate_on_second)
        self.counts = {}
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        command = args[-1]

        class Result:
            returncode = 0
            stderr = ""

        result = Result()
        requested = next(
            (name for name in ("POSCAR", "CONTCAR", "INCAR", "OUTCAR") if f"/{name}'" in command),
            None,
        )
        if requested is None:
            parent = "parent\n1\n3 0 0\n0 3 0\n0 0 18\nCu\n2\nDirect\n0 0 .45\n.5 .5 .45\n"
            files = {}
            for name in ("POSCAR", "CONTCAR", "INCAR", "OUTCAR"):
                if name in self.missing:
                    continue
                content = None if name == "OUTCAR" else f"{name}-content"
                files[name] = {
                    "sha256": sha256(f"{name}-1".encode()).hexdigest(),
                    "size": len(content or "outcar"),
                    "mtime_ns": 1,
                    "stable": not self.mutate_on_second,
                    "content": content,
                    "markers": {},
                }
            result.stdout = json.dumps(
                {
                    "files": files,
                    "clean_parent": {
                        "identity": "clean/POSCAR",
                        "content": parent,
                        "sha256": sha256(parent.encode()).hexdigest(),
                    },
                }
            )
            return result
        self.counts[requested] = self.counts.get(requested, 0) + 1
        if requested in self.missing:
            result.stdout = json.dumps({"exists": False})
            return result
        tick = 2 if requested in self.mutate_on_second and self.counts[requested] > 1 else 1
        content = None if requested == "OUTCAR" else f"{requested}-content"
        result.stdout = json.dumps(
            {
                "exists": True,
                "sha256": sha256(f"{requested}-{tick}".encode()).hexdigest(),
                "size": len(content or "outcar"),
                "mtime_ns": tick,
                "stable": True,
                "content": content,
                "markers": {},
            }
        )
        return result


def test_ssh_source_rejects_root_escape_and_unsafe_paths():
    """Break caught: untrusted job paths escaping the one allowed remote tree."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    with pytest.raises(ValueError, match="allowed root"):
        SSHRemoteFileSource(remote_root="relative/path")
    with pytest.raises(ValueError, match="allowed root"):
        SSHRemoteFileSource(remote_root="/tmp")
    source = SSHRemoteFileSource(remote_root=REMOTE_ROOT, wrapper_path=SSH_WRAPPER)
    for path in ("../escape", "/etc/passwd", "jobs/../escape", "bad\nname", "bad\x00name"):
        with pytest.raises(ValueError):
            source.validate_relative_path(path)
    assert source.validate_relative_path("Ni P/吸附's case") == "Ni P/吸附's case"


def test_snapshot_base64_encodes_untrusted_but_in_root_job_path():
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    runner = SnapshotRunner(missing=("POSCAR", "CONTCAR", "INCAR", "OUTCAR"))
    source = SSHRemoteFileSource(remote_root=REMOTE_ROOT, wrapper_path=SSH_WRAPPER, runner=runner)
    source.snapshot("Ni P/吸附's case", root_id="cluster-a")
    command = runner.calls[0][0][-1]
    assert "Ni P" not in command
    assert "吸附" not in command
    assert "'s case" not in command


def test_ssh_invocation_uses_argument_list_no_shell_and_read_only_discovery():
    """Break caught: shell interpolation or a mutating remote command template."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    source = SSHRemoteFileSource(remote_root=REMOTE_ROOT, wrapper_path=SSH_WRAPPER, runner=runner)
    source.discover(REMOTE_ROOT)
    args, kwargs = calls[0]
    assert isinstance(args, list)
    assert kwargs["shell"] is False
    assert Path(args[0]).name.lower() in {"powershell.exe", "pwsh.exe", "pwsh"}
    command = args[-1]
    assert "find -P" in command
    assert "-print0" in command
    assert not any(token in command for token in ("rm ", "mv ", "cp ", "mkdir", "sed -i", ">"))
    assert Path(source.wrapper_path).name == "ssh-dft-cluster.ps1"


def test_discovery_includes_incomplete_directory_with_only_poscar():
    """Break caught: incomplete calculations disappearing before quality audit."""
    from llm_matgen.adsorption.source import ALLOWED_REMOTE_ROOT, SSHRemoteFileSource

    class Result:
        returncode = 0
        stderr = ""
        stdout = f"{ALLOWED_REMOTE_ROOT}/jobs/incomplete/POSCAR\0"

    def runner(args, **kwargs):
        if "POSCAR" not in args[-1]:
            Result.stdout = ""
        return Result()

    source = SSHRemoteFileSource(runner=runner)
    assert source.discover(ALLOWED_REMOTE_ROOT) == ["jobs/incomplete"]


def test_discovery_maps_marker_file_at_scan_root_to_safe_internal_job():
    from llm_matgen.adsorption.source import (
        ALLOWED_REMOTE_ROOT,
        ROOT_JOB_DIR,
        SSHRemoteFileSource,
    )

    class Result:
        returncode = 0
        stderr = ""
        stdout = f"{ALLOWED_REMOTE_ROOT}/POSCAR\0"

    source = SSHRemoteFileSource(runner=lambda *_args, **_kwargs: Result())
    assert source.discover(ALLOWED_REMOTE_ROOT) == [ROOT_JOB_DIR]
    snapshot_runner = SnapshotRunner(missing=("POSCAR", "CONTCAR", "INCAR", "OUTCAR"))
    SSHRemoteFileSource(runner=snapshot_runner).snapshot(ROOT_JOB_DIR, root_id="cluster-a")
    assert ALLOWED_REMOTE_ROOT not in snapshot_runner.calls[0][0][-1].split("b64:", 1)[-1]


def test_snapshot_keeps_missing_files_for_rejected_incomplete_audit():
    """Break caught: one absent VASP file aborting the entire scan."""
    from llm_matgen.adsorption.extractor import CaseExtractor
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    source = SSHRemoteFileSource(runner=SnapshotRunner(missing={"OUTCAR"}))
    snapshot = source.snapshot("jobs/incomplete", root_id="cluster-a")

    assert set(snapshot.files) == {"POSCAR", "CONTCAR", "INCAR"}
    assert CaseExtractor().extract(snapshot).status.value == "rejected_incomplete"


def test_snapshot_uses_two_whole_job_snapshots_and_marks_cross_pass_change_unstable():
    """Break caught: accepting files that changed between individually stable reads."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    runner = SnapshotRunner(mutate_on_second={"CONTCAR"})
    snapshot = SSHRemoteFileSource(runner=runner).snapshot("jobs/active", root_id="cluster-a")

    assert len(runner.calls) == 1
    assert all(not item.stable for item in snapshot.files.values())


def test_snapshot_finds_read_only_clean_parent_inside_allowed_root():
    """Break caught: every production snapshot being rejected for a missing clean slab."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    snapshot = SSHRemoteFileSource(runner=SnapshotRunner()).snapshot(
        "systems/CN-slab/OH", root_id="cluster-a"
    )

    assert snapshot.clean_parent_identity == "clean/POSCAR"
    assert snapshot.clean_parent_poscar.startswith("parent\n")


def test_batched_snapshots_use_one_runner_call_and_preserve_order():
    """Break caught: full scans paying one SSH handshake for every job."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    payload = json.loads(SnapshotRunner()(["ignored", "batch"], shell=False).stdout)
    calls = []

    class Result:
        returncode = 0
        stderr = ""
        stdout = "\n".join(
            json.dumps({"ok": True, "job": name, "payload": payload})
            for name in ("encoded-a", "encoded-b")
        )

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    source = SSHRemoteFileSource(runner=runner, batch_size=64)
    items = list(source.iter_snapshots(["jobs/a", "jobs/b"], root_id="cluster-a"))

    assert len(calls) == 1
    assert [path for path, _snapshot in items] == ["jobs/a", "jobs/b"]
    assert [snapshot.relative_job_dir for _path, snapshot in items] == ["jobs/a", "jobs/b"]
    command = calls[0][0][-1]
    assert "jobs/a" not in command and "jobs/b" not in command
    assert calls[0][1]["shell"] is False


def test_remote_batch_program_returns_one_json_envelope_per_job(tmp_path):
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    jobs = []
    for name in ("a", "b"):
        job = tmp_path / name
        job.mkdir()
        (job / "POSCAR").write_text("fixture", encoding="utf-8")
        jobs.append(str(job))
    encoded_program = __import__("base64").b64encode(
        SSHRemoteFileSource._job_snapshot_program().encode()
    ).decode()
    encoded_jobs = __import__("base64").b64encode(json.dumps(jobs).encode()).decode()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            SSHRemoteFileSource._batch_snapshot_program(),
            encoded_program,
            str(tmp_path),
            encoded_jobs,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    envelopes = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(envelopes) == 2
    assert all(item["ok"] is True for item in envelopes)
    assert all(set(item["payload"]["files"]) == {"POSCAR"} for item in envelopes)


def test_snapshot_program_hashes_but_does_not_transfer_oversized_small_file(tmp_path):
    """Break caught: accidentally copying an unbounded local-input artifact over SSH."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    path = tmp_path / "POSCAR"
    path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    result = subprocess.run(
        [sys.executable, "-c", SSHRemoteFileSource._snapshot_program(), str(tmp_path), str(path), "POSCAR"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["size"] == 4 * 1024 * 1024 + 1
    assert payload["content"] is None
    assert payload["markers"]["too_large"] is True


def test_outcar_markers_survive_chunk_boundaries_and_failure_is_monotonic(tmp_path):
    """Break caught: a split marker or later clean chunk hiding electronic failure."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    phrase = b"electronic minimization did not converge"
    split = 17
    outcar = tmp_path / "OUTCAR"
    outcar.write_bytes(b"x" * (1024 * 1024 - split) + phrase + b"\n" + b"clean" * 100)
    result = subprocess.run(
        [sys.executable, "-c", SSHRemoteFileSource._snapshot_program(), str(tmp_path), str(outcar), "OUTCAR"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["content"] is None
    assert payload["markers"]["electronic_converged"] is False


def test_one_job_program_accumulates_ionic_steps(tmp_path):
    """Break caught: ionic-step efficiency provenance silently staying at zero."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    job = tmp_path / "job"
    job.mkdir()
    phrase = "FREE ENERGIE OF THE ION-ELECTRON SYSTEM"
    (job / "OUTCAR").write_text(f"{phrase}\nnoise\n{phrase}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", SSHRemoteFileSource._job_snapshot_program(), str(tmp_path), str(job)],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["files"]["OUTCAR"]["markers"]["ionic_steps"] == 2


def test_outcar_uses_only_last_run_segment_and_unknown_steps_are_none(tmp_path):
    """Break caught: an appended failed restart inheriting an older successful run."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    job = tmp_path / "job"
    job.mkdir()
    (job / "OUTCAR").write_text(
        "vasp.6.4 old\nreached required accuracy\nGeneral timing and accounting informations\n"
        "vasp.6.4 new\nelectronic minimization did not converge\nBRMIX: very serious problems\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", SSHRemoteFileSource._job_snapshot_program(), str(tmp_path), str(job)],
        capture_output=True, text=True, check=True,
    )
    markers = json.loads(result.stdout)["files"]["OUTCAR"]["markers"]
    assert markers["normal_termination"] is False
    assert markers["ionic_converged"] is False
    assert markers["electronic_converged"] is False
    assert markers["fatal_warning"] is True
    assert markers["ionic_steps"] is None


def test_snapshot_rejects_required_file_name_set_change_after_read():
    """Break caught: a newly appearing required file leaving a mixed-time snapshot stable."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    runner = SnapshotRunner()
    original = runner.__call__

    def changed_runner(args, **kwargs):
        result = original(args, **kwargs)
        payload = json.loads(result.stdout)
        payload["before_names"] = ["POSCAR", "CONTCAR", "OUTCAR"]
        payload["after_names"] = ["POSCAR", "CONTCAR", "INCAR", "OUTCAR"]
        result.stdout = json.dumps(payload)
        return result

    snapshot = SSHRemoteFileSource(runner=changed_runner).snapshot("jobs/active", root_id="r")
    assert all(not item.stable for item in snapshot.files.values())


@pytest.mark.parametrize(
    "fatal_line",
    ["Disk quota exceeded", "Input/output error", "MPI_ABORT was invoked"],
)
def test_production_outcar_marks_additional_storage_and_mpi_fatals(tmp_path, fatal_line):
    """Break caught: common filesystem/MPI crashes entering the eligible index."""
    from llm_matgen.adsorption.source import SSHRemoteFileSource

    job = tmp_path / "job"
    job.mkdir()
    (job / "OUTCAR").write_text(
        f"vasp.6.4\n{fatal_line}\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-c", SSHRemoteFileSource._job_snapshot_program(), str(tmp_path), str(job)],
        capture_output=True, text=True, check=True,
    )
    markers = json.loads(result.stdout)["files"]["OUTCAR"]["markers"]
    assert markers["fatal_warning"] is True
