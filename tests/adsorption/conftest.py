from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256


def poscar(
    species: list[str],
    counts: list[int],
    fractional: list[tuple[float, float, float]],
    lattice: tuple[tuple[float, float, float], ...] = (
        (3.0, 0.0, 0.0),
        (0.0, 3.0, 0.0),
        (0.0, 0.0, 18.0),
    ),
) -> str:
    lines = [
        "fixture",
        "1.0",
        *(" ".join(str(value) for value in vector) for vector in lattice),
        " ".join(species),
        " ".join(str(value) for value in counts),
        "Direct",
        *(" ".join(str(value) for value in point) for point in fractional),
    ]
    return "\n".join(lines) + "\n"


def remote_file(path: str, content: str | None, *, markers=None, stable=True, tick=1):
    from llm_matgen.adsorption.source import RemoteFileSnapshot

    payload = (content or "").encode()
    return RemoteFileSnapshot(
        relative_path=path,
        sha256=sha256(payload).hexdigest(),
        size=len(payload),
        mtime_ns=tick,
        content=content,
        markers=markers or {},
        stable=stable,
    )


def job_snapshot(
    job_dir: str = "jobs/case-1",
    *,
    initial: str | None = None,
    final: str | None = None,
    incar: str = "NSW = 20\nIBRION = 2\n",
    markers=None,
    stable=True,
    signature="sig-1",
    parent: str | None = None,
):
    from llm_matgen.adsorption.source import JobSnapshot

    initial = initial or poscar(
        ["Cu", "H"], [2, 1], [(0.0, 0.0, 0.45), (0.5, 0.5, 0.45), (0.0, 0.0, 0.56)]
    )
    final = final or poscar(
        ["Cu", "H"], [2, 1], [(0.0, 0.0, 0.451), (0.5, 0.5, 0.451), (0.02, 0.01, 0.55)]
    )
    parent = parent or poscar(
        ["Cu"], [2], [(0.0, 0.0, 0.45), (0.5, 0.5, 0.45)]
    )
    outcar_markers = {
        "normal_termination": True,
        "ionic_converged": True,
        "electronic_converged": True,
        "fatal_warning": False,
        "ionic_steps": 7,
    }
    outcar_markers.update(markers or {})
    files = {
        "POSCAR": remote_file("POSCAR", initial, stable=stable),
        "CONTCAR": remote_file("CONTCAR", final, stable=stable),
        "INCAR": remote_file("INCAR", incar, stable=stable),
        "OUTCAR": remote_file("OUTCAR", None, markers=outcar_markers, stable=stable),
    }
    return JobSnapshot(
        root_id="cluster-a",
        relative_job_dir=job_dir,
        source_signature=signature,
        files=files,
        clean_parent_identity="clean-1",
        clean_parent_poscar=parent,
        captured_at=datetime.now(timezone.utc),
    )

