from __future__ import annotations

import json
import re
import socket
import sqlite3
import gc
import weakref
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from .conftest import job_snapshot, poscar


class CountingExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, snapshot):
        from llm_matgen.adsorption.extractor import CaseExtractor

        self.calls += 1
        return CaseExtractor().extract(snapshot)


class StreamingSource:
    def __init__(self, count: int):
        self.paths = [f"jobs/case-{index:03d}" for index in range(count)]
        self.references = []
        self.max_live_snapshots = 0

    def discover(self, _root):
        return list(self.paths)

    def snapshot(self, relative_job_dir, *, root_id):
        gc.collect()
        assert sum(reference() is not None for reference in self.references) == 0
        snapshot = job_snapshot(
            relative_job_dir,
            signature=f"signature-{relative_job_dir}",
        ).model_copy(update={"root_id": root_id})
        self.references.append(weakref.ref(snapshot))
        self.max_live_snapshots = max(
            self.max_live_snapshots,
            sum(reference() is not None for reference in self.references),
        )
        return snapshot


class BatchOnlySource:
    def __init__(self):
        self.paths = ["jobs/a", "jobs/b"]
        self.iter_calls = 0

    def discover(self, _root):
        return list(self.paths)

    def snapshot(self, *_args, **_kwargs):
        raise AssertionError("store must use the bounded batch iterator")

    def iter_snapshots(self, paths, *, root_id):
        self.iter_calls += 1
        for path in paths:
            yield path, job_snapshot(
                path, signature=f"signature-{path}"
            ).model_copy(update={"root_id": root_id})


def test_schema_wal_short_connections_and_scan_lock(tmp_path: Path):
    """Break caught: unsafe concurrent writers or a database with weak defaults."""
    from llm_matgen.adsorption.store import AdsorptionCaseStore, ScanLockedError

    store = AdsorptionCaseStore(tmp_path / "cases", stale_lock_seconds=0)
    with store.connect() as first:
        first_id = id(first)
        assert first.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert first.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert first.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
        tables = {row[0] for row in first.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"metadata", "scan_runs", "locations", "case_revisions", "file_evidence"} <= tables
    with store.connect() as second:
        assert second is not first or id(second) != first_id
    with store.scan_lock():
        with pytest.raises(ScanLockedError, match="already active"):
            with store.scan_lock():
                pass

    store.lock_path.write_text(
        json.dumps(
            {
                "pid": 99_999_999,
                "hostname": socket.gethostname(),
                "acquired_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                "token": "stale",
            }
        ),
        encoding="utf-8",
    )
    with store.scan_lock():
        assert store.lock_path.exists()
    assert not store.lock_path.exists()


def test_scan_skips_unchanged_creates_revision_marks_inactive_and_keeps_artifacts_small(tmp_path: Path):
    """Break caught: re-extracting unchanged jobs or overwriting historical evidence."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    store = AdsorptionCaseStore(tmp_path / "cases")
    source = InMemoryRemoteFileSource({"jobs/case-1": job_snapshot()})
    extractor = CountingExtractor()
    assert store.scan(source, extractor, root_id="cluster-a") == 1
    first = store.list_revisions()[0]
    assert extractor.calls == 1
    assert store.scan(source, extractor, root_id="cluster-a") == 2
    assert extractor.calls == 1
    assert len(store.list_revisions()) == 1

    changed = job_snapshot(signature="sig-2")
    source.snapshots["jobs/case-1"] = changed
    assert store.scan(source, extractor, root_id="cluster-a") == 3
    assert extractor.calls == 2
    revisions = store.list_revisions(include_superseded=True)
    assert len(revisions) == 2
    assert store.get_revision(first.revision_id).status.value == "superseded_restart"
    current = store.list_revisions()[0]
    artifact_dir = store.root / current.artifact_relative_path
    assert (artifact_dir / "case.json").is_file()
    assert not (artifact_dir / "OUTCAR").exists()

    source.snapshots.clear()
    assert store.scan(source, extractor, root_id="cluster-a") == 4
    assert store.list_revisions() == []
    assert len(store.list_revisions(include_inactive=True, include_superseded=True)) == 2
    assert len(store.list_revisions(index_revision=1)) == 1


def test_scan_releases_each_large_snapshot_before_fetching_the_next(tmp_path: Path):
    """Break caught: all remote structures accumulating until a full scan raises MemoryError."""
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    store = AdsorptionCaseStore(tmp_path / "cases")
    source = StreamingSource(24)

    assert store.scan(source, CountingExtractor(), root_id="cluster-a") == 1
    assert source.max_live_snapshots == 1
    gc.collect()
    assert all(reference() is None for reference in source.references)
    assert len(store.list_revisions(include_duplicates=True)) == 24


def test_scan_prefers_bounded_batch_iterator_when_source_provides_it(tmp_path: Path):
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    store = AdsorptionCaseStore(tmp_path / "cases")
    source = BatchOnlySource()

    assert store.scan(source, CountingExtractor(), root_id="cluster-a") == 1
    assert source.iter_calls == 1
    assert len(store.list_revisions(include_duplicates=True)) == 2


def test_duplicate_content_preserves_location_but_not_retrieval_weight(tmp_path: Path):
    """Break caught: staging copies multiplying one calculation's influence."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    first = job_snapshot("prod/case", signature="sig-a")
    second = job_snapshot("staging/case", signature="sig-b")
    source = InMemoryRemoteFileSource({"prod/case": first, "staging/case": second})
    store = AdsorptionCaseStore(tmp_path / "cases")
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    revisions = store.list_revisions(include_duplicates=True)
    assert sorted(item.status.value for item in revisions) == ["duplicate", "eligible"]
    duplicate = next(item for item in revisions if item.status.value == "duplicate")
    assert duplicate.original_status.value == "eligible"
    assert len(store.list_revisions()) == 1


def test_v1_database_migrates_presence_history_and_lock_audit_to_v3(tmp_path: Path):
    """Break caught: an existing v1 store missing later presence or lock-audit tables."""
    from llm_matgen.adsorption.schema import INITIAL_SCHEMA, SCAN_LOCK_RECOVERY_AUDIT_SCHEMA
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    root = tmp_path / "cases"
    root.mkdir()
    db = root / "adsorption-cases.sqlite3"
    presence = """CREATE TABLE IF NOT EXISTS location_presence (
    location_id INTEGER NOT NULL REFERENCES locations(location_id) ON DELETE CASCADE,
    active_from_index_revision INTEGER NOT NULL,
    inactive_from_index_revision INTEGER,
    PRIMARY KEY(location_id, active_from_index_revision)
);
"""
    with sqlite3.connect(db) as connection:
        connection.executescript(
            INITIAL_SCHEMA.replace(presence, "").replace(
                SCAN_LOCK_RECOVERY_AUDIT_SCHEMA.strip(), ""
            )
        )
        connection.execute("INSERT INTO metadata(key,value) VALUES ('index_revision','0')")
        connection.execute("INSERT INTO metadata(key,value) VALUES ('schema_version','1')")
        connection.execute("PRAGMA user_version=1")
        connection.commit()

    store = AdsorptionCaseStore(root)
    assert store.status().schema_version == 3
    with store.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='location_presence'"
        ).fetchone()[0] == "location_presence"
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scan_lock_recovery_audit'"
        ).fetchone()[0] == "scan_lock_recovery_audit"


def test_v2_database_migrates_lock_recovery_audit_to_v3(tmp_path: Path):
    """Break caught: a deployed v2 store skipping the new recovery-audit migration."""
    from llm_matgen.adsorption.schema import INITIAL_SCHEMA, SCAN_LOCK_RECOVERY_AUDIT_SCHEMA
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    root = tmp_path / "cases"
    root.mkdir()
    with sqlite3.connect(root / "adsorption-cases.sqlite3") as connection:
        connection.executescript(
            INITIAL_SCHEMA.replace(SCAN_LOCK_RECOVERY_AUDIT_SCHEMA.strip(), "")
        )
        connection.execute("INSERT INTO metadata(key,value) VALUES ('index_revision','0')")
        connection.execute("INSERT INTO metadata(key,value) VALUES ('schema_version','2')")
        connection.execute("PRAGMA user_version=2")
        connection.commit()

    store = AdsorptionCaseStore(root)
    with store.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scan_lock_recovery_audit'"
        ).fetchone()[0] == "scan_lock_recovery_audit"


def test_same_location_reverting_to_historical_content_is_a_restart_not_duplicate(tmp_path: Path):
    """Break caught: a restarted job being confused with a staging copy of itself."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    source = InMemoryRemoteFileSource({"jobs/case-1": job_snapshot(signature="sig-a")})
    store = AdsorptionCaseStore(tmp_path / "cases")
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    source.snapshots["jobs/case-1"] = job_snapshot(
        signature="sig-b", incar="NSW = 21\nIBRION = 2\n"
    )
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    source.snapshots["jobs/case-1"] = job_snapshot(signature="sig-a-returned")
    store.scan(source, CountingExtractor(), root_id="cluster-a")

    revisions = store.list_revisions(include_superseded=True, include_duplicates=True)
    assert len(revisions) == 3
    assert all(item.status.value != "duplicate" for item in revisions)


def test_scan_exception_does_not_publish_or_advance_revision(tmp_path: Path):
    """Break caught: readers observing half a scan after extractor failure."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    class ExplodingExtractor:
        def extract(self, snapshot):
            raise RuntimeError("controlled failure")

    store = AdsorptionCaseStore(tmp_path / "cases")
    source = InMemoryRemoteFileSource({"jobs/case-1": job_snapshot()})
    with pytest.raises(RuntimeError, match="controlled"):
        store.scan(source, ExplodingExtractor(), root_id="cluster-a")
    assert store.status().index_revision == 0
    assert store.list_revisions(include_inactive=True, include_superseded=True) == []
    assert list(store.staging_root.iterdir()) == []


def test_delete_then_reappear_preserves_frozen_presence_history(tmp_path: Path):
    """Break caught: reactivation rewriting whether a case existed in an older index."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    source = InMemoryRemoteFileSource({"jobs/case-1": job_snapshot()})
    store = AdsorptionCaseStore(tmp_path / "cases")
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    source.snapshots.clear()
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    source.snapshots["jobs/case-1"] = job_snapshot()
    store.scan(source, CountingExtractor(), root_id="cluster-a")

    assert store.list_revisions(index_revision=2) == []
    inactive = store.list_revisions(index_revision=2, include_inactive=True)
    assert inactive[0].inactive_index_revision == 2
    assert len(store.list_revisions(index_revision=3)) == 1


def test_future_get_revision_is_rejected_and_next_scan_cleans_orphan_artifact(tmp_path: Path):
    """Break caught: reading unpublished state or retaining crash debris forever."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore, AdsorptionStoreError

    store = AdsorptionCaseStore(tmp_path / "cases")
    source = InMemoryRemoteFileSource({"jobs/case-1": job_snapshot()})
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    revision_id = store.list_revisions()[0].revision_id
    with pytest.raises(AdsorptionStoreError, match="outside published range"):
        store.get_revision(revision_id, index_revision=2)

    orphan = store.artifact_root / "case-crashed" / "rev-crashed"
    orphan.mkdir(parents=True)
    (orphan / "case.json").write_text("{}", encoding="utf-8")
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    assert not orphan.exists()


def test_artifact_move_failure_rolls_back_database_and_all_moved_artifacts(tmp_path: Path, monkeypatch):
    """Break caught: a mid-publish filesystem failure exposing a partial index."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    source = InMemoryRemoteFileSource(
        {
            "jobs/a": job_snapshot("jobs/a", signature="a"),
            "jobs/b": job_snapshot("jobs/b", signature="b", incar="NSW=21\nIBRION=2\n"),
        }
    )
    store = AdsorptionCaseStore(tmp_path / "cases")
    original = Path.replace
    moves = 0

    def fail_second_move(path, target):
        nonlocal moves
        if ".staging" in path.parts:
            moves += 1
            if moves == 2:
                raise OSError("controlled artifact move failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_move)
    with pytest.raises(OSError, match="controlled artifact"):
        store.scan(source, CountingExtractor(), root_id="cluster-a")

    assert store.status().index_revision == 0
    assert store.list_revisions(include_inactive=True, include_superseded=True) == []
    assert list(store.artifact_root.rglob("case.json")) == []
    assert list(store.staging_root.iterdir()) == []


def test_frozen_index_derives_historical_status_without_future_supersede_metadata(tmp_path: Path):
    """Break caught: a frozen reader learning a restart that had not happened yet."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    source = InMemoryRemoteFileSource({"jobs/case": job_snapshot("jobs/case")})
    store = AdsorptionCaseStore(tmp_path / "cases")
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    first_id = store.list_revisions()[0].revision_id
    source.snapshots["jobs/case"] = job_snapshot(
        "jobs/case", signature="changed", incar="NSW=21\nIBRION=2\n"
    )
    store.scan(source, CountingExtractor(), root_id="cluster-a")

    frozen = store.get_revision(first_id, index_revision=1)
    assert frozen.status.value == "eligible"
    assert frozen.superseded_index_revision is None
    assert frozen.superseded_by_revision_id is None
    current = store.get_revision(first_id, index_revision=2)
    assert current.status.value == "superseded_restart"
    assert current.superseded_index_revision == 2


@pytest.mark.parametrize("representative", ["rejected", "inactive", "superseded"])
def test_duplicate_only_uses_current_active_eligible_representative(tmp_path: Path, representative: str):
    """Break caught: unusable historical evidence swallowing a later eligible case."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    store = AdsorptionCaseStore(tmp_path / "cases")
    if representative == "rejected":
        source = InMemoryRemoteFileSource({"old/case": job_snapshot("old/case", stable=False)})
        store.scan(source, CountingExtractor(), root_id="cluster-a")
    else:
        source = InMemoryRemoteFileSource({"old/case": job_snapshot("old/case")})
        store.scan(source, CountingExtractor(), root_id="cluster-a")
        if representative == "inactive":
            source.snapshots.clear()
            store.scan(source, CountingExtractor(), root_id="cluster-a")
        else:
            changed_final = poscar(
                ["Cu", "H"], [2, 1],
                [(0.0, 0.0, .451), (.5, .5, .451), (.25, .25, .54)],
            )
            source.snapshots["old/case"] = job_snapshot(
                "old/case", signature="new", final=changed_final
            )
            store.scan(source, CountingExtractor(), root_id="cluster-a")
    source.snapshots["new/case"] = job_snapshot("new/case", signature="eligible-new")
    store.scan(source, CountingExtractor(), root_id="cluster-a")

    new_case = next(
        item for item in store.list_revisions(include_duplicates=True)
        if item.relative_job_dir == "new/case"
    )
    assert new_case.status.value == "eligible"
    assert new_case.duplicate_of_revision_id is None


def _seed_scan_lease(store, payload):
    with store.connect() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES ('scan_lock', ?)",
            (json.dumps(payload, sort_keys=True),),
        )
        connection.commit()


def test_scan_lock_uses_atomic_lease_for_corruption_cross_host_and_pid_reuse(tmp_path: Path, monkeypatch):
    """Break caught: stale-file recovery deleting a newer contender's lock."""
    from llm_matgen.adsorption.store import AdsorptionCaseStore, ScanLockedError

    store = AdsorptionCaseStore(tmp_path / "cases", stale_lock_seconds=0)
    old_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    _seed_scan_lease(store, {"broken": True})
    with pytest.raises(ScanLockedError, match="corrupt"):
        with store.scan_lock():
            pass

    _seed_scan_lease(
        store,
        {"pid": 99999999, "hostname": "another-host", "acquired_at": old_time,
         "token": "remote", "process_identity": None},
    )
    with pytest.raises(ScanLockedError, match="another host"):
        with store.scan_lock():
            pass

    _seed_scan_lease(
        store,
        {"pid": 4242, "hostname": socket.gethostname(), "acquired_at": old_time,
         "token": "reused", "process_identity": "old-process"},
    )
    monkeypatch.setattr(store, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(store, "_process_identity", lambda pid: "new-process")
    with store.scan_lock():
        with pytest.raises(ScanLockedError, match="already active"):
            with store.scan_lock():
                pass


def test_scan_lock_release_cannot_delete_later_sidecar(tmp_path: Path):
    """Break caught: an old owner unlinking a newer token during release."""
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    store = AdsorptionCaseStore(tmp_path / "cases")
    with store.scan_lock():
        newer = {
            "pid": 1, "hostname": socket.gethostname(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "token": "newer", "process_identity": "newer",
        }
        store.lock_path.write_text(json.dumps(newer), encoding="utf-8")
    assert json.loads(store.lock_path.read_text(encoding="utf-8"))["token"] == "newer"


def test_scan_lock_explicit_recovery_requires_exact_fingerprint_and_records_audit(tmp_path: Path):
    """Break caught: operator recovery deleting a changed owner without durable attribution."""
    from llm_matgen.adsorption.store import AdsorptionCaseStore, ScanLockedError

    store = AdsorptionCaseStore(tmp_path / "cases", stale_lock_seconds=0)
    lease = {
        "pid": 42,
        "hostname": "another-host",
        "acquired_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "token": "remote-owner",
        "process_identity": "remote-process",
    }
    _seed_scan_lease(store, lease)
    with pytest.raises(ScanLockedError) as locked:
        with store.scan_lock():
            pass
    match = re.search(r"recovery_fingerprint=([0-9a-f]{64})", str(locked.value))
    assert match is not None
    fingerprint = match.group(1)

    with pytest.raises(ValueError, match="reason"):
        store.recover_scan_lock(expected_fingerprint=fingerprint, reason="  ")
    later_lease = {**lease, "token": "later-owner"}
    _seed_scan_lease(store, later_lease)
    with pytest.raises(ScanLockedError, match="fingerprint changed"):
        store.recover_scan_lock(
            expected_fingerprint=fingerprint, reason="operator checked original owner"
        )
    with store.connect() as connection:
        current = connection.execute(
            "SELECT value FROM metadata WHERE key='scan_lock'"
        ).fetchone()
        assert json.loads(current[0])["token"] == "later-owner"
        assert connection.execute(
            "SELECT COUNT(*) FROM scan_lock_recovery_audit"
        ).fetchone()[0] == 0

    _seed_scan_lease(store, lease)
    recovery_id = store.recover_scan_lock(
        expected_fingerprint=fingerprint,
        reason="operator verified the remote scheduler is stopped",
    )
    assert recovery_id
    with store.connect() as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='scan_lock'"
        ).fetchone() is None
        audit = connection.execute(
            "SELECT * FROM scan_lock_recovery_audit WHERE recovery_id=?", (recovery_id,)
        ).fetchone()
        assert audit["lease_fingerprint"] == fingerprint
        assert json.loads(audit["lease_json"])["token"] == "remote-owner"
        assert audit["reason"] == "operator verified the remote scheduler is stopped"


@pytest.mark.parametrize(
    "payload",
    [
        ["valid-json-but-not-an-object"],
        {
            "pid": {"nested": 42}, "hostname": ["nested-host"],
            "acquired_at": "not-a-date", "token": "nested", "process_identity": "p",
        },
        {
            "pid": [42], "hostname": {"nested": "host"},
            "acquired_at": ["not-a-date"], "token": "nested", "process_identity": ["p"],
        },
        {
            "pid": True, "hostname": socket.gethostname(),
            "acquired_at": "not-a-date", "token": "bool-pid",
            "process_identity": {"nested": "p"},
        },
    ],
)
def test_explicit_lock_recovery_preserves_malformed_json_and_binds_only_safe_scalars(
    tmp_path: Path, payload
):
    """Break caught: valid nested JSON reaching SQLite as an unsupported bind value."""
    from llm_matgen.adsorption.store import AdsorptionCaseStore, ScanLockedError

    store = AdsorptionCaseStore(tmp_path / "cases", stale_lock_seconds=0)
    _seed_scan_lease(store, payload)
    raw_lease = json.dumps(payload, sort_keys=True)
    with pytest.raises(ScanLockedError) as locked:
        with store.scan_lock():
            pass
    fingerprint = re.search(
        r"recovery_fingerprint=([0-9a-f]{64})", str(locked.value)
    ).group(1)

    recovery_id = store.recover_scan_lock(
        expected_fingerprint=fingerprint, reason="operator retained malformed lease evidence"
    )
    with store.connect() as connection:
        audit = connection.execute(
            "SELECT * FROM scan_lock_recovery_audit WHERE recovery_id=?", (recovery_id,)
        ).fetchone()
    expected_hostname = (
        payload.get("hostname")
        if isinstance(payload, dict) and isinstance(payload.get("hostname"), str)
        else None
    )
    expected_pid = (
        payload.get("pid")
        if isinstance(payload, dict) and type(payload.get("pid")) is int
        else None
    )
    assert audit["lease_json"] == raw_lease
    assert audit["lease_fingerprint"] == fingerprint
    assert audit["lease_hostname"] == expected_hostname
    assert audit["lease_pid"] == expected_pid


def test_malformed_process_identity_cannot_trigger_automatic_pid_reuse_recovery(
    tmp_path: Path, monkeypatch
):
    """Break caught: comparing a nested identity object to a string and stealing its lease."""
    from llm_matgen.adsorption.store import AdsorptionCaseStore, ScanLockedError

    store = AdsorptionCaseStore(tmp_path / "cases", stale_lock_seconds=0)
    payload = {
        "pid": 4242,
        "hostname": socket.gethostname(),
        "acquired_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "token": "malformed-identity",
        "process_identity": {"nested": "not-a-scalar"},
    }
    _seed_scan_lease(store, payload)
    monkeypatch.setattr(store, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(store, "_process_identity", lambda pid: "real-process")
    with pytest.raises(ScanLockedError, match="corrupt scan lease"):
        with store.scan_lock():
            pass
    with store.connect() as connection:
        current = connection.execute(
            "SELECT value FROM metadata WHERE key='scan_lock'"
        ).fetchone()[0]
    assert json.loads(current)["token"] == "malformed-identity"


def test_same_scan_missing_old_location_cannot_be_duplicate_representative(tmp_path: Path):
    """Break caught: a location deleted in this publish swallowing its active replacement."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    store = AdsorptionCaseStore(tmp_path / "cases")
    source = InMemoryRemoteFileSource({"old/case": job_snapshot("old/case")})
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    source.snapshots = {"new/case": job_snapshot("new/case", signature="new-location")}
    store.scan(source, CountingExtractor(), root_id="cluster-a")

    active = store.list_revisions(include_duplicates=True)
    assert len(active) == 1
    assert active[0].relative_job_dir == "new/case"
    assert active[0].status.value == "eligible"
    assert active[0].duplicate_of_revision_id is None


def test_same_scan_superseded_old_location_cannot_be_duplicate_representative(tmp_path: Path):
    """Break caught: a revision replaced in this publish swallowing another new location."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    store = AdsorptionCaseStore(tmp_path / "cases")
    source = InMemoryRemoteFileSource({"old/case": job_snapshot("old/case")})
    store.scan(source, CountingExtractor(), root_id="cluster-a")
    changed = poscar(
        ["Cu", "H"], [2,1],
        [(0,0,.451),(.5,.5,.451),(.25,.25,.54)],
    )
    source.snapshots = {
        "old/case": job_snapshot("old/case", signature="changed", final=changed),
        "new/case": job_snapshot("new/case", signature="new-location"),
    }
    store.scan(source, CountingExtractor(), root_id="cluster-a")

    new_case = next(
        item for item in store.list_revisions(include_duplicates=True)
        if item.relative_job_dir == "new/case"
    )
    assert new_case.status.value == "eligible"
    assert new_case.duplicate_of_revision_id is None


def test_list_limit_is_pushed_down_and_evidence_is_loaded_in_one_query(tmp_path: Path, monkeypatch):
    """Break caught: limited retrieval deserializing every historical revision with N+1 SQL."""
    from llm_matgen.adsorption.source import InMemoryRemoteFileSource
    from llm_matgen.adsorption.store import AdsorptionCaseStore

    source = InMemoryRemoteFileSource({"jobs/case": job_snapshot("jobs/case")})
    store = AdsorptionCaseStore(tmp_path / "cases")
    for index in range(3):
        source.snapshots["jobs/case"] = job_snapshot(
            "jobs/case", signature=f"sig-{index}", incar=f"NSW={20+index}\nIBRION=2\n"
        )
        store.scan(source, CountingExtractor(), root_id="cluster-a")
    with store.connect() as connection:
        oldest = connection.execute(
            "SELECT revision_id FROM case_revisions ORDER BY created_index_revision LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "UPDATE case_revisions SET audit_json='not-json' WHERE revision_id=?", (oldest,)
        )
        connection.commit()

    statements = []
    original_open = store._open

    def traced_open():
        connection = original_open()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(store, "_open", traced_open)
    revisions = store.list_revisions(include_superseded=True, limit=2)
    selects = [item for item in statements if item.lstrip().upper().startswith("SELECT")]
    assert len(revisions) == 2
    assert len(selects) <= 3
