"""Authoritative SQLite and immutable-artifact store for adsorption cases."""

from __future__ import annotations

import json
import os
import ctypes
import shutil
import socket
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from llm_matgen.adsorption.extractor import CaseExtraction
from llm_matgen.adsorption.models import (
    AdsorptionCaseRevision,
    CaseAudit,
    CaseFeatureSet,
    CaseStatus,
    FileEvidence,
    StoreStatus,
)
from llm_matgen.adsorption.schema import (
    ADSORPTION_DB_SCHEMA_VERSION,
    INITIAL_SCHEMA,
    MIGRATE_V1_TO_V2,
    MIGRATE_V2_TO_V3,
)
from llm_matgen.config import default_data_root

DEFAULT_STORE_ROOT = default_data_root() / "adsorption-cases"


class AdsorptionStoreError(RuntimeError):
    pass


class ScanLockedError(AdsorptionStoreError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class AdsorptionCaseStore:
    def __init__(
        self,
        root: Path | str = DEFAULT_STORE_ROOT,
        *,
        busy_timeout_ms: int = 5000,
        stale_lock_seconds: int = 6 * 60 * 60,
    ):
        self.root = Path(root).resolve()
        self.db_path = self.root / "adsorption-cases.sqlite3"
        self.artifact_root = self.root / "artifacts"
        self.staging_root = self.root / ".staging"
        self.lock_path = self.root / "scan.lock"
        self.busy_timeout_ms = busy_timeout_ms
        self.stale_lock_seconds = stale_lock_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._open()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            with self.connect() as connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version > ADSORPTION_DB_SCHEMA_VERSION:
                    raise AdsorptionStoreError(
                        f"adsorption schema {version} is newer than supported "
                        f"{ADSORPTION_DB_SCHEMA_VERSION}"
                    )
                if version == 0:
                    connection.executescript(INITIAL_SCHEMA)
                    connection.execute(
                        "INSERT OR IGNORE INTO metadata(key, value) VALUES ('index_revision', '0')"
                    )
                    connection.execute(
                        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
                        (str(ADSORPTION_DB_SCHEMA_VERSION),),
                    )
                    connection.execute(
                        f"PRAGMA user_version = {ADSORPTION_DB_SCHEMA_VERSION}"
                    )
                    connection.commit()
                else:
                    if version == 1:
                        connection.executescript(MIGRATE_V1_TO_V2)
                        version = 2
                    if version == 2:
                        connection.executescript(MIGRATE_V2_TO_V3)
                        version = 3
                    connection.execute(
                        "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
                        (str(ADSORPTION_DB_SCHEMA_VERSION),),
                    )
                    connection.execute(
                        f"PRAGMA user_version = {ADSORPTION_DB_SCHEMA_VERSION}"
                    )
                    connection.commit()
        except (sqlite3.DatabaseError, OSError) as exc:
            raise AdsorptionStoreError(f"cannot initialize adsorption store: {self.root}") from exc

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _process_identity(pid: int) -> str | None:
        if pid <= 0:
            return None
        try:
            if os.name == "nt":
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if not handle:
                    return None
                creation = ctypes.c_ulonglong()
                exit_time = ctypes.c_ulonglong()
                kernel = ctypes.c_ulonglong()
                user = ctypes.c_ulonglong()
                try:
                    if not kernel32.GetProcessTimes(
                        handle,
                        ctypes.byref(creation),
                        ctypes.byref(exit_time),
                        ctypes.byref(kernel),
                        ctypes.byref(user),
                    ):
                        return None
                    return str(creation.value)
                finally:
                    kernel32.CloseHandle(handle)
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            return stat.split()[21]
        except (OSError, IndexError, AttributeError):
            return None

    def _lease_recoverable(self, payload: object) -> tuple[bool, str]:
        if not isinstance(payload, dict):
            return False, "corrupt scan lease"
        required = {"pid", "hostname", "acquired_at", "token", "process_identity"}
        if not required <= set(payload):
            return False, "corrupt scan lease"
        if (
            type(payload["pid"]) is not int
            or not isinstance(payload["hostname"], str)
            or not isinstance(payload["acquired_at"], str)
            or not isinstance(payload["token"], str)
            or (
                payload["process_identity"] is not None
                and not isinstance(payload["process_identity"], str)
            )
        ):
            return False, "corrupt scan lease"
        try:
            acquired = datetime.fromisoformat(payload["acquired_at"])
        except ValueError:
            return False, "corrupt scan lease"
        pid = payload["pid"]
        if acquired.tzinfo is None:
            return False, "corrupt scan lease"
        if payload["hostname"] != socket.gethostname():
            return False, "scan lease belongs to another host"
        age = (_now() - acquired.astimezone(timezone.utc)).total_seconds()
        if age < self.stale_lock_seconds:
            return False, "an adsorption scan is already active"
        if not self._pid_alive(pid):
            return True, "dead process"
        current_identity = self._process_identity(pid)
        if current_identity is not None and current_identity != payload["process_identity"]:
            return True, "reused pid"
        return False, "an adsorption scan is already active"

    @contextmanager
    def scan_lock(self) -> Iterator[None]:
        token = uuid4().hex
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": _now().isoformat(),
            "token": token,
            "process_identity": self._process_identity(os.getpid()),
        }
        lease_value = _json(payload)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='scan_lock'"
            ).fetchone()
            if row is not None:
                recovery_fingerprint = sha256(row[0].encode("utf-8")).hexdigest()
                try:
                    current = json.loads(row[0])
                except (TypeError, json.JSONDecodeError):
                    connection.rollback()
                    raise ScanLockedError(
                        "corrupt scan lease; "
                        f"recovery_fingerprint={recovery_fingerprint}; lock={self.lock_path}"
                    ) from None
                recoverable, reason = self._lease_recoverable(current)
                if not recoverable:
                    connection.rollback()
                    raise ScanLockedError(
                        f"{reason}; recovery_fingerprint={recovery_fingerprint}; "
                        f"lock={self.lock_path}"
                    )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES ('scan_lock', ?)",
                (lease_value,),
            )
            connection.commit()
        try:
            self.lock_path.write_text(lease_value, encoding="utf-8")
        except OSError:
            with self.connect() as connection:
                connection.execute(
                    "DELETE FROM metadata WHERE key='scan_lock' AND value=?",
                    (lease_value,),
                )
                connection.commit()
            raise
        try:
            yield
        finally:
            try:
                current = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if current.get("token") == token:
                    self.lock_path.unlink()
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                pass
            with self.connect() as connection:
                connection.execute(
                    "DELETE FROM metadata WHERE key='scan_lock' AND value=?",
                    (lease_value,),
                )
                connection.commit()

    def recover_scan_lock(self, *, expected_fingerprint: str, reason: str) -> str:
        """Explicitly clear exactly one operator-reviewed lease and preserve an audit row."""

        reason = reason.strip()
        if not reason:
            raise ValueError("scan lock recovery reason is required")
        if len(expected_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in expected_fingerprint
        ):
            raise ValueError("expected_fingerprint must be a lowercase SHA-256 digest")
        recovery_id = f"lock-recovery-{uuid4().hex}"
        recovered_at = _now().isoformat()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='scan_lock'"
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ScanLockedError("no scan lease exists to recover")
            lease_value = str(row[0])
            actual_fingerprint = sha256(lease_value.encode("utf-8")).hexdigest()
            if actual_fingerprint != expected_fingerprint:
                connection.rollback()
                raise ScanLockedError(
                    "scan lease fingerprint changed; "
                    f"current_fingerprint={actual_fingerprint}"
                )
            try:
                parsed_lease = json.loads(lease_value)
            except (TypeError, json.JSONDecodeError):
                parsed_lease = None
            lease = parsed_lease if isinstance(parsed_lease, dict) else {}
            lease_hostname = lease.get("hostname")
            if not isinstance(lease_hostname, str):
                lease_hostname = None
            lease_pid = lease.get("pid")
            if type(lease_pid) is not int:
                lease_pid = None
            deleted = connection.execute(
                "DELETE FROM metadata WHERE key='scan_lock' AND value=?",
                (lease_value,),
            )
            if deleted.rowcount != 1:
                connection.rollback()
                raise ScanLockedError("scan lease changed during conditional recovery")
            connection.execute(
                """
                INSERT INTO scan_lock_recovery_audit(
                    recovery_id, recovered_at, reason, lease_fingerprint,
                    lease_json, lease_hostname, lease_pid,
                    operator_hostname, operator_pid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recovery_id,
                    recovered_at,
                    reason,
                    actual_fingerprint,
                    lease_value,
                    lease_hostname,
                    lease_pid,
                    socket.gethostname(),
                    os.getpid(),
                ),
            )
            connection.commit()
        try:
            sidecar_value = self.lock_path.read_text(encoding="utf-8")
            if sha256(sidecar_value.encode("utf-8")).hexdigest() == expected_fingerprint:
                self.lock_path.unlink()
        except (FileNotFoundError, OSError):
            pass
        return recovery_id

    def _index_revision(self, connection: sqlite3.Connection | None = None) -> int:
        if connection is not None:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='index_revision'"
            ).fetchone()
            return int(row[0])
        with self.connect() as owned:
            return self._index_revision(owned)

    def status(self) -> StoreStatus:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status FROM scan_runs ORDER BY started_at DESC, scan_run_id DESC LIMIT 1"
            ).fetchone()
            return StoreStatus(
                root=str(self.root),
                schema_version=ADSORPTION_DB_SCHEMA_VERSION,
                index_revision=self._index_revision(connection),
                last_scan_status=row[0] if row else None,
            )

    def _record_scan_start(self, scan_run_id: str, root_id: str, started_at: datetime) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO scan_runs(scan_run_id, root_id, started_at, status)
                VALUES (?, ?, ?, 'running')
                """,
                (scan_run_id, root_id, started_at.isoformat()),
            )
            connection.commit()

    def _record_scan_failure(self, scan_run_id: str, error: Exception) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE scan_runs
                SET status='failed', completed_at=?, error=?
                WHERE scan_run_id=?
                """,
                (_now().isoformat(), f"{type(error).__name__}: {error}", scan_run_id),
            )
            connection.commit()

    def _known_locations(self, root_id: str) -> dict[str, sqlite3.Row]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT l.*, r.case_id
                FROM locations l
                LEFT JOIN case_revisions r ON r.revision_id=l.current_revision_id
                WHERE l.root_id=?
                """,
                (root_id,),
            ).fetchall()
        return {row["relative_job_dir"]: row for row in rows}

    def _cleanup_orphan_artifacts(self) -> None:
        with self.connect() as connection:
            referenced = {
                (self.root / row[0]).resolve()
                for row in connection.execute("SELECT artifact_relative_path FROM case_revisions")
            }
        for case_dir in self.artifact_root.iterdir():
            if not case_dir.is_dir():
                continue
            for revision_dir in case_dir.iterdir():
                if revision_dir.is_dir() and revision_dir.resolve() not in referenced:
                    shutil.rmtree(revision_dir)
            if not any(case_dir.iterdir()):
                case_dir.rmdir()
        for stale_staging in self.staging_root.iterdir():
            if stale_staging.is_dir():
                shutil.rmtree(stale_staging)

    @staticmethod
    def _case_id(root_id: str, relative_job_dir: str) -> str:
        digest = sha256(f"{root_id}\0{relative_job_dir}".encode()).hexdigest()[:24]
        return f"case-{digest}"

    def _write_staged_artifact(
        self,
        staging_scan: Path,
        revision: AdsorptionCaseRevision,
        extraction: CaseExtraction,
    ) -> Path:
        directory = staging_scan / revision.case_id / revision.revision_id
        directory.mkdir(parents=True, exist_ok=False)
        allowed = {"POSCAR", "CONTCAR", "INCAR"}
        for name, content in extraction.artifacts.items():
            if name in allowed or name.lower().endswith(".mson"):
                (directory / name).write_text(content, encoding="utf-8")
        self._write_case_json(directory, revision)
        return directory

    @staticmethod
    def _write_case_json(directory: Path, revision: AdsorptionCaseRevision) -> None:
        (directory / "case.json").write_text(
            json.dumps(
                revision.model_dump(mode="json"),
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def scan(self, source, extractor, *, root_id: str) -> int:
        """Run one complete scan and publish exactly one new index revision."""

        scan_run_id = uuid4().hex
        started_at = _now()
        staging_scan = self.staging_root / scan_run_id
        moved_artifacts: list[Path] = []
        with self.scan_lock():
            self._cleanup_orphan_artifacts()
            self._record_scan_start(scan_run_id, root_id, started_at)
            try:
                known = self._known_locations(root_id)
                discovery_root = getattr(source, "remote_root", root_id)
                discovered = sorted(set(source.discover(discovery_root)))
                with self.connect() as connection:
                    current_index = self._index_revision(connection)
                    publish_index = current_index + 1
                    duplicate_rows = connection.execute(
                        """
                        SELECT r.revision_id, r.exact_hash,
                               r.equivalent_fingerprint, l.root_id,
                               l.relative_job_dir
                        FROM case_revisions r
                        JOIN locations l ON l.location_id=r.location_id
                        WHERE r.original_status=?
                          AND r.duplicate_of_revision_id IS NULL
                          AND (r.superseded_index_revision IS NULL OR r.superseded_index_revision>?)
                          AND EXISTS (
                              SELECT 1 FROM location_presence p
                              WHERE p.location_id=l.location_id
                                AND p.active_from_index_revision<=?
                                AND (p.inactive_from_index_revision IS NULL OR p.inactive_from_index_revision>?)
                          )
                        ORDER BY r.created_index_revision, r.revision_id
                        """,
                        (CaseStatus.ELIGIBLE.value, current_index, current_index, current_index),
                    ).fetchall()

                signatures: dict[str, str] = {}
                pending: dict[str, AdsorptionCaseRevision] = {}
                staging_scan.mkdir(parents=True, exist_ok=False)
                batch_iterator = getattr(source, "iter_snapshots", None)
                snapshot_items = (
                    batch_iterator(discovered, root_id=root_id)
                    if callable(batch_iterator)
                    else (
                        (path, source.snapshot(path, root_id=root_id))
                        for path in discovered
                    )
                )
                seen_snapshot_paths: list[str] = []
                for relative_job_dir, snapshot in snapshot_items:
                    expected_position = len(seen_snapshot_paths)
                    if (
                        expected_position >= len(discovered)
                        or relative_job_dir != discovered[expected_position]
                        or snapshot.relative_job_dir != relative_job_dir
                    ):
                        raise AdsorptionStoreError(
                            "remote snapshot iterator returned an unexpected path or order"
                        )
                    seen_snapshot_paths.append(relative_job_dir)
                    signatures[relative_job_dir] = snapshot.source_signature
                    old = known.get(relative_job_dir)
                    if (
                        old is not None
                        and old["source_signature"] == snapshot.source_signature
                        and old["current_revision_id"] is not None
                    ):
                        continue
                    extraction = extractor.extract(snapshot)
                    case_id = (
                        old["case_id"]
                        if old is not None and old["case_id"]
                        else self._case_id(root_id, relative_job_dir)
                    )
                    revision_id = f"rev-{uuid4().hex}"
                    artifact_relative = (
                        Path("artifacts") / case_id / revision_id
                    ).as_posix()
                    revision = AdsorptionCaseRevision(
                        case_id=case_id,
                        revision_id=revision_id,
                        root_id=root_id,
                        relative_job_dir=relative_job_dir,
                        clean_parent_identity=extraction.clean_parent_identity,
                        exact_hash=extraction.exact_hash,
                        equivalent_fingerprint=extraction.equivalent_fingerprint,
                        source_signature=snapshot.source_signature,
                        status=extraction.status,
                        original_status=extraction.status,
                        audit=extraction.audit,
                        features=extraction.features,
                        evidence=list(extraction.evidence),
                        artifact_relative_path=artifact_relative,
                        index_revision=publish_index,
                        created_at=_now(),
                    )
                    self._write_staged_artifact(staging_scan, revision, extraction)
                    pending[relative_job_dir] = revision
                    del extraction, snapshot
                if seen_snapshot_paths != discovered:
                    raise AdsorptionStoreError(
                        "remote snapshot iterator did not return every discovered path"
                    )

                duplicate_by_hash: dict[str, list[tuple[str, str, str]]] = {}
                duplicate_by_equivalent: dict[str, list[tuple[str, str, str]]] = {}
                for row in duplicate_rows:
                    if row["root_id"] == root_id and (
                        row["relative_job_dir"] not in signatures
                        or row["relative_job_dir"] in pending
                    ):
                        continue
                    candidate = (
                        row["revision_id"],
                        row["root_id"],
                        row["relative_job_dir"],
                    )
                    duplicate_by_hash.setdefault(row["exact_hash"], []).append(candidate)
                    duplicate_by_equivalent.setdefault(
                        row["equivalent_fingerprint"], []
                    ).append(candidate)

                def other_location_candidate(
                    candidates: dict[str, list[tuple[str, str, str]]],
                    fingerprint: str,
                    location: tuple[str, str],
                ) -> str | None:
                    return next(
                        (
                            revision_id
                            for revision_id, candidate_root, candidate_dir
                            in candidates.get(fingerprint, [])
                            if (candidate_root, candidate_dir) != location
                        ),
                        None,
                    )

                for relative_job_dir in discovered:
                    revision = pending.get(relative_job_dir)
                    if revision is None or revision.original_status is not CaseStatus.ELIGIBLE:
                        continue
                    location = (root_id, relative_job_dir)
                    duplicate_of = other_location_candidate(
                        duplicate_by_hash, revision.exact_hash, location
                    )
                    if duplicate_of is None:
                        duplicate_of = other_location_candidate(
                            duplicate_by_equivalent,
                            revision.equivalent_fingerprint,
                            location,
                        )
                    if duplicate_of is not None:
                        revision = revision.model_copy(
                            update={
                                "status": CaseStatus.DUPLICATE,
                                "duplicate_of_revision_id": duplicate_of,
                            }
                        )
                        pending[relative_job_dir] = revision
                        staged = staging_scan / revision.case_id / revision.revision_id
                        self._write_case_json(staged, revision)
                    else:
                        candidate = (revision.revision_id, root_id, relative_job_dir)
                        duplicate_by_hash.setdefault(revision.exact_hash, []).append(candidate)
                        duplicate_by_equivalent.setdefault(
                            revision.equivalent_fingerprint, []
                        ).append(candidate)

                connection = self._open()
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    if self._index_revision(connection) != current_index:
                        raise AdsorptionStoreError("index revision changed during scan")
                    for relative_job_dir in discovered:
                        old = known.get(relative_job_dir)
                        revision = pending.get(relative_job_dir)
                        if old is None:
                            connection.execute(
                                """
                                INSERT INTO locations(
                                    root_id, relative_job_dir, source_signature,
                                    current_revision_id, active, first_seen_index_revision,
                                    last_seen_index_revision, inactive_index_revision
                                ) VALUES (?, ?, ?, ?, 1, ?, ?, NULL)
                                """,
                                (
                                    root_id,
                                    relative_job_dir,
                                    signatures[relative_job_dir],
                                    revision.revision_id if revision else None,
                                    publish_index,
                                    publish_index,
                                ),
                            )
                            location_id = int(
                                connection.execute(
                                    "SELECT last_insert_rowid()"
                                ).fetchone()[0]
                            )
                            connection.execute(
                                "INSERT INTO location_presence(location_id, active_from_index_revision) VALUES (?, ?)",
                                (location_id, publish_index),
                            )
                        else:
                            location_id = int(old["location_id"])
                            if not bool(old["active"]):
                                connection.execute(
                                    "INSERT INTO location_presence(location_id, active_from_index_revision) VALUES (?, ?)",
                                    (location_id, publish_index),
                                )
                            connection.execute(
                                """
                                UPDATE locations
                                SET source_signature=?, active=1,
                                    last_seen_index_revision=?, inactive_index_revision=NULL,
                                    current_revision_id=COALESCE(?, current_revision_id)
                                WHERE location_id=?
                                """,
                                (
                                    signatures[relative_job_dir],
                                    publish_index,
                                    revision.revision_id if revision else None,
                                    location_id,
                                ),
                            )
                        if revision is None:
                            continue
                        if old is not None and old["current_revision_id"]:
                            connection.execute(
                                """
                                UPDATE case_revisions
                                SET status=?, superseded_index_revision=?,
                                    superseded_by_revision_id=?
                                WHERE revision_id=?
                                """,
                                (
                                    CaseStatus.SUPERSEDED_RESTART.value,
                                    publish_index,
                                    revision.revision_id,
                                    old["current_revision_id"],
                                ),
                            )
                        staged = staging_scan / revision.case_id / revision.revision_id
                        final = self.root / revision.artifact_relative_path
                        final.parent.mkdir(parents=True, exist_ok=True)
                        if final.exists():
                            raise AdsorptionStoreError(
                                f"immutable artifact already exists: {final}"
                            )
                        staged.replace(final)
                        moved_artifacts.append(final)
                        connection.execute(
                            """
                            INSERT INTO case_revisions(
                                revision_id, case_id, location_id, original_status,
                                status, exact_hash, equivalent_fingerprint,
                                source_signature, clean_parent_identity, audit_json,
                                features_json, artifact_relative_path,
                                created_index_revision, created_at,
                                duplicate_of_revision_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                revision.revision_id,
                                revision.case_id,
                                location_id,
                                revision.original_status.value,
                                revision.status.value,
                                revision.exact_hash,
                                revision.equivalent_fingerprint,
                                revision.source_signature,
                                revision.clean_parent_identity,
                                _json(revision.audit),
                                _json(revision.features),
                                revision.artifact_relative_path,
                                revision.index_revision,
                                revision.created_at.isoformat(),
                                revision.duplicate_of_revision_id,
                            ),
                        )
                        for evidence in revision.evidence:
                            connection.execute(
                                """
                                INSERT INTO file_evidence(
                                    revision_id, relative_path, sha256, size, mtime_ns,
                                    captured_at, stable, content_stored,
                                    parsed_evidence_json, markers_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    revision.revision_id,
                                    evidence.relative_path,
                                    evidence.sha256,
                                    evidence.size,
                                    evidence.mtime_ns,
                                    evidence.captured_at.isoformat(),
                                    int(evidence.stable),
                                    int(evidence.content_stored),
                                    _json(evidence.parsed_evidence),
                                    _json(evidence.markers),
                                ),
                            )
                    missing = sorted(set(known) - set(discovered))
                    for relative_job_dir in missing:
                        location_id = int(known[relative_job_dir]["location_id"])
                        connection.execute(
                            """
                            UPDATE locations
                            SET active=0, inactive_index_revision=?,
                                last_seen_index_revision=?
                            WHERE root_id=? AND relative_job_dir=?
                            """,
                            (publish_index, publish_index, root_id, relative_job_dir),
                        )
                        connection.execute(
                            """
                            UPDATE location_presence
                            SET inactive_from_index_revision=?
                            WHERE location_id=? AND inactive_from_index_revision IS NULL
                            """,
                            (publish_index, location_id),
                        )
                    connection.execute(
                        "UPDATE metadata SET value=? WHERE key='index_revision'",
                        (str(publish_index),),
                    )
                    connection.execute(
                        """
                        UPDATE scan_runs
                        SET status='published', completed_at=?,
                            published_index_revision=?, error=NULL
                        WHERE scan_run_id=?
                        """,
                        (_now().isoformat(), publish_index, scan_run_id),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    for path in reversed(moved_artifacts):
                        shutil.rmtree(path, ignore_errors=True)
                    raise
                finally:
                    connection.close()
                return publish_index
            except Exception as exc:
                self._record_scan_failure(scan_run_id, exc)
                raise
            finally:
                shutil.rmtree(staging_scan, ignore_errors=True)

    def _evidence_for(
        self, connection: sqlite3.Connection, revision_id: str
    ) -> list[FileEvidence]:
        rows = connection.execute(
            """
            SELECT * FROM file_evidence
            WHERE revision_id=?
            ORDER BY relative_path
            """,
            (revision_id,),
        ).fetchall()
        return [
            FileEvidence(
                relative_path=row["relative_path"],
                sha256=row["sha256"],
                size=row["size"],
                mtime_ns=row["mtime_ns"],
                captured_at=datetime.fromisoformat(row["captured_at"]),
                stable=bool(row["stable"]),
                content_stored=bool(row["content_stored"]),
                parsed_evidence=json.loads(row["parsed_evidence_json"]),
                markers=json.loads(row["markers_json"]),
            )
            for row in rows
        ]

    def _evidence_for_many(
        self, connection: sqlite3.Connection, revision_ids: list[str]
    ) -> dict[str, list[FileEvidence]]:
        grouped = {revision_id: [] for revision_id in revision_ids}
        if not revision_ids:
            return grouped
        placeholders = ",".join("?" for _ in revision_ids)
        rows = connection.execute(
            f"""
            SELECT * FROM file_evidence
            WHERE revision_id IN ({placeholders})
            ORDER BY revision_id, relative_path
            """,
            revision_ids,
        ).fetchall()
        for row in rows:
            grouped[row["revision_id"]].append(
                FileEvidence(
                    relative_path=row["relative_path"],
                    sha256=row["sha256"],
                    size=row["size"],
                    mtime_ns=row["mtime_ns"],
                    captured_at=datetime.fromisoformat(row["captured_at"]),
                    stable=bool(row["stable"]),
                    content_stored=bool(row["content_stored"]),
                    parsed_evidence=json.loads(row["parsed_evidence_json"]),
                    markers=json.loads(row["markers_json"]),
                )
            )
        return grouped

    def _load_revision(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        index_revision: int,
        evidence: list[FileEvidence] | None = None,
    ) -> AdsorptionCaseRevision:
        recorded_superseded = row["superseded_index_revision"]
        superseded_visible = (
            recorded_superseded is not None and index_revision >= recorded_superseded
        )
        if superseded_visible:
            status = CaseStatus.SUPERSEDED_RESTART
            superseded = recorded_superseded
            superseded_by = row["superseded_by_revision_id"]
        else:
            status = (
                CaseStatus.DUPLICATE
                if row["duplicate_of_revision_id"] is not None
                else CaseStatus(row["original_status"])
            )
            superseded = None
            superseded_by = None
        if "inactive_at_target" in row.keys():
            inactive = row["inactive_at_target"]
        else:
            active = connection.execute(
                """
                SELECT 1 FROM location_presence
                WHERE location_id=? AND active_from_index_revision<=?
                  AND (inactive_from_index_revision IS NULL OR inactive_from_index_revision>?)
                """,
                (row["location_id"], index_revision, index_revision),
            ).fetchone()
            inactive = None
            if active is None:
                ended = connection.execute(
                    """
                    SELECT inactive_from_index_revision FROM location_presence
                    WHERE location_id=? AND inactive_from_index_revision<=?
                    ORDER BY inactive_from_index_revision DESC LIMIT 1
                    """,
                    (row["location_id"], index_revision),
                ).fetchone()
                inactive = ended[0] if ended else None
        return AdsorptionCaseRevision(
            case_id=row["case_id"],
            revision_id=row["revision_id"],
            root_id=row["root_id"],
            relative_job_dir=row["relative_job_dir"],
            clean_parent_identity=row["clean_parent_identity"],
            exact_hash=row["exact_hash"],
            equivalent_fingerprint=row["equivalent_fingerprint"],
            source_signature=row["source_signature"],
            status=status,
            original_status=CaseStatus(row["original_status"]),
            audit=CaseAudit.model_validate_json(row["audit_json"]),
            features=CaseFeatureSet.model_validate_json(row["features_json"]),
            evidence=evidence if evidence is not None else self._evidence_for(connection, row["revision_id"]),
            artifact_relative_path=row["artifact_relative_path"],
            index_revision=row["created_index_revision"],
            created_at=datetime.fromisoformat(row["created_at"]),
            inactive_index_revision=inactive,
            superseded_index_revision=superseded,
            superseded_by_revision_id=superseded_by,
            duplicate_of_revision_id=row["duplicate_of_revision_id"],
        )

    def get_revision(
        self,
        revision_id: str,
        *,
        index_revision: int | None = None,
    ) -> AdsorptionCaseRevision | None:
        with self.connect() as connection:
            current = self._index_revision(connection)
            target = current if index_revision is None else index_revision
            if target < 0 or target > current:
                raise AdsorptionStoreError(
                    f"index revision {target} is outside published range 0..{current}"
                )
            row = connection.execute(
                """
                SELECT r.*, l.root_id, l.relative_job_dir, l.inactive_index_revision
                FROM case_revisions r
                JOIN locations l ON l.location_id=r.location_id
                WHERE r.revision_id=? AND r.created_index_revision<=?
                """,
                (revision_id, target),
            ).fetchone()
            return self._load_revision(connection, row, index_revision=target) if row else None

    def list_revisions(
        self,
        *,
        index_revision: int | None = None,
        include_inactive: bool = False,
        include_superseded: bool = False,
        include_duplicates: bool = False,
        status: CaseStatus | None = None,
        limit: int | None = None,
    ) -> list[AdsorptionCaseRevision]:
        with self.connect() as connection:
            current = self._index_revision(connection)
            target = current if index_revision is None else index_revision
            if target < 0 or target > current:
                raise AdsorptionStoreError(
                    f"index revision {target} is outside published range 0..{current}"
                )
            clauses = ["r.created_index_revision<=?"]
            parameters: list[object] = [target]
            if not include_inactive:
                clauses.append(
                    """EXISTS (
                        SELECT 1 FROM location_presence p
                        WHERE p.location_id=l.location_id
                          AND p.active_from_index_revision<=?
                          AND (p.inactive_from_index_revision IS NULL OR p.inactive_from_index_revision>?)
                    )"""
                )
                parameters.extend([target, target])
            if not include_superseded:
                clauses.append(
                    "(r.superseded_index_revision IS NULL OR r.superseded_index_revision>?)"
                )
                parameters.append(target)
            if not include_duplicates:
                clauses.append(
                    "NOT (r.duplicate_of_revision_id IS NOT NULL AND "
                    "(r.superseded_index_revision IS NULL OR r.superseded_index_revision>?))"
                )
                parameters.append(target)
            if status is not None:
                if status is CaseStatus.SUPERSEDED_RESTART:
                    clauses.append(
                        "r.superseded_index_revision IS NOT NULL AND r.superseded_index_revision<=?"
                    )
                    parameters.append(target)
                elif status is CaseStatus.DUPLICATE:
                    clauses.append(
                        "r.duplicate_of_revision_id IS NOT NULL AND "
                        "(r.superseded_index_revision IS NULL OR r.superseded_index_revision>?)"
                    )
                    parameters.append(target)
                else:
                    clauses.append(
                        "r.duplicate_of_revision_id IS NULL AND r.original_status=? AND "
                        "(r.superseded_index_revision IS NULL OR r.superseded_index_revision>?)"
                    )
                    parameters.extend([status.value, target])
            limit_sql = ""
            if limit is not None:
                limit_sql = " LIMIT ?"
                parameters.append(limit)
            rows = connection.execute(
                f"""
                SELECT r.*, l.root_id, l.relative_job_dir,
                       CASE WHEN EXISTS (
                           SELECT 1 FROM location_presence p2
                           WHERE p2.location_id=l.location_id
                             AND p2.active_from_index_revision<=?
                             AND (p2.inactive_from_index_revision IS NULL OR p2.inactive_from_index_revision>?)
                       ) THEN NULL ELSE (
                           SELECT MAX(p3.inactive_from_index_revision)
                           FROM location_presence p3
                           WHERE p3.location_id=l.location_id
                             AND p3.inactive_from_index_revision<=?
                       ) END AS inactive_at_target
                FROM case_revisions r
                JOIN locations l ON l.location_id=r.location_id
                WHERE {' AND '.join(clauses)}
                ORDER BY r.created_index_revision DESC, r.case_id, r.revision_id
                {limit_sql}
                """,
                [target, target, target, *parameters],
            ).fetchall()
            evidence_by_revision = self._evidence_for_many(
                connection, [row["revision_id"] for row in rows]
            )
            revisions = [
                self._load_revision(
                    connection,
                    row,
                    index_revision=target,
                    evidence=evidence_by_revision[row["revision_id"]],
                )
                for row in rows
            ]
        return revisions

    def query(self, **filters) -> list[AdsorptionCaseRevision]:
        return self.list_revisions(**filters)
