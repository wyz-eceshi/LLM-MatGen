"""Independent SQLite schema for historical adsorption cases."""

ADSORPTION_DB_SCHEMA_VERSION = 3

SCAN_LOCK_RECOVERY_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_lock_recovery_audit (
    recovery_id TEXT PRIMARY KEY,
    recovered_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    lease_fingerprint TEXT NOT NULL,
    lease_json TEXT NOT NULL,
    lease_hostname TEXT,
    lease_pid INTEGER,
    operator_hostname TEXT NOT NULL,
    operator_pid INTEGER NOT NULL
);
"""

MIGRATE_V1_TO_V2 = """
CREATE TABLE IF NOT EXISTS location_presence (
    location_id INTEGER NOT NULL REFERENCES locations(location_id) ON DELETE CASCADE,
    active_from_index_revision INTEGER NOT NULL,
    inactive_from_index_revision INTEGER,
    PRIMARY KEY(location_id, active_from_index_revision)
);
INSERT OR IGNORE INTO location_presence(
    location_id, active_from_index_revision, inactive_from_index_revision
)
SELECT location_id, first_seen_index_revision,
       CASE WHEN active=0 THEN inactive_index_revision ELSE NULL END
FROM locations;
"""

MIGRATE_V2_TO_V3 = SCAN_LOCK_RECOVERY_AUDIT_SCHEMA

INITIAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scan_runs (
    scan_run_id TEXT PRIMARY KEY,
    root_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    published_index_revision INTEGER,
    error TEXT
);
CREATE TABLE IF NOT EXISTS scan_lock_recovery_audit (
    recovery_id TEXT PRIMARY KEY,
    recovered_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    lease_fingerprint TEXT NOT NULL,
    lease_json TEXT NOT NULL,
    lease_hostname TEXT,
    lease_pid INTEGER,
    operator_hostname TEXT NOT NULL,
    operator_pid INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS locations (
    location_id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id TEXT NOT NULL,
    relative_job_dir TEXT NOT NULL,
    source_signature TEXT NOT NULL,
    current_revision_id TEXT,
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    first_seen_index_revision INTEGER NOT NULL,
    last_seen_index_revision INTEGER NOT NULL,
    inactive_index_revision INTEGER,
    UNIQUE(root_id, relative_job_dir)
);
CREATE TABLE IF NOT EXISTS case_revisions (
    revision_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    location_id INTEGER NOT NULL REFERENCES locations(location_id),
    original_status TEXT NOT NULL,
    status TEXT NOT NULL,
    exact_hash TEXT NOT NULL,
    equivalent_fingerprint TEXT NOT NULL,
    source_signature TEXT NOT NULL,
    clean_parent_identity TEXT,
    audit_json TEXT NOT NULL,
    features_json TEXT NOT NULL,
    artifact_relative_path TEXT NOT NULL,
    created_index_revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    superseded_index_revision INTEGER,
    superseded_by_revision_id TEXT,
    duplicate_of_revision_id TEXT
);
CREATE TABLE IF NOT EXISTS location_presence (
    location_id INTEGER NOT NULL REFERENCES locations(location_id) ON DELETE CASCADE,
    active_from_index_revision INTEGER NOT NULL,
    inactive_from_index_revision INTEGER,
    PRIMARY KEY(location_id, active_from_index_revision)
);
CREATE TABLE IF NOT EXISTS file_evidence (
    revision_id TEXT NOT NULL REFERENCES case_revisions(revision_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    stable INTEGER NOT NULL CHECK (stable IN (0, 1)),
    content_stored INTEGER NOT NULL CHECK (content_stored IN (0, 1)),
    parsed_evidence_json TEXT NOT NULL,
    markers_json TEXT NOT NULL,
    PRIMARY KEY(revision_id, relative_path)
);
CREATE INDEX IF NOT EXISTS idx_locations_active
    ON locations(root_id, active, relative_job_dir);
CREATE INDEX IF NOT EXISTS idx_revisions_location
    ON case_revisions(location_id, created_index_revision DESC);
CREATE INDEX IF NOT EXISTS idx_revisions_exact
    ON case_revisions(exact_hash);
CREATE INDEX IF NOT EXISTS idx_revisions_equivalent
    ON case_revisions(equivalent_fingerprint);
"""
