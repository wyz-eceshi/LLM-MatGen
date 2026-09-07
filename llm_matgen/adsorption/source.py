"""Read-only remote snapshot sources for adsorption calculations."""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
import re
import subprocess
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Callable, Iterator, Protocol, runtime_checkable

from pydantic import Field, field_validator

from llm_matgen.adsorption.models import StrictModel
from llm_matgen.generators.models import JsonValue

ALLOWED_REMOTE_ROOT = "/public/home/zhangwy01/culuyao"
SSH_WRAPPER = r"C:\Users\Administrator\Documents\VASP计算流\scripts\ssh-dft-cluster.ps1"
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
ROOT_JOB_DIR = "__scan_root__"


class RemoteFileSnapshot(StrictModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    content: str | None = None
    markers: dict[str, JsonValue] = Field(default_factory=dict)
    stable: bool = True
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("captured_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(timezone.utc)


class JobSnapshot(StrictModel):
    root_id: str
    relative_job_dir: str
    source_signature: str
    files: dict[str, RemoteFileSnapshot]
    clean_parent_identity: str | None = None
    clean_parent_poscar: str | None = None
    captured_at: datetime

    @field_validator("captured_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(timezone.utc)


@runtime_checkable
class RemoteFileSource(Protocol):
    def discover(self, root: str) -> list[str]:
        """Return normalized job directories below the configured root."""

    def snapshot(self, relative_job_dir: str, *, root_id: str) -> JobSnapshot:
        """Return a stable, bounded snapshot without full OUTCAR content."""


class InMemoryRemoteFileSource:
    """Complete fake source used by tests and offline callers."""

    def __init__(self, snapshots: dict[str, JobSnapshot]):
        self.snapshots = snapshots
        self.discover_calls = 0
        self.snapshot_calls = 0

    def discover(self, root: str) -> list[str]:
        self.discover_calls += 1
        return sorted(self.snapshots)

    def snapshot(self, relative_job_dir: str, *, root_id: str) -> JobSnapshot:
        self.snapshot_calls += 1
        snapshot = self.snapshots[relative_job_dir]
        if snapshot.root_id == root_id:
            return snapshot
        return snapshot.model_copy(update={"root_id": root_id})


class RemoteSourceError(RuntimeError):
    pass


class SSHRemoteFileSource:
    """Production source restricted to one existing SSH wrapper and one root."""

    def __init__(
        self,
        *,
        remote_root: str = ALLOWED_REMOTE_ROOT,
        wrapper_path: str = SSH_WRAPPER,
        runner: Callable[..., object] = subprocess.run,
        timeout_seconds: int = 120,
        batch_size: int = 64,
    ):
        normalized_root = posixpath.normpath(remote_root)
        if normalized_root != ALLOWED_REMOTE_ROOT:
            raise ValueError(f"remote root is not the allowed root: {remote_root}")
        if wrapper_path != SSH_WRAPPER:
            raise ValueError("only the configured ssh-dft-cluster.ps1 wrapper is allowed")
        self.remote_root = normalized_root
        self.wrapper_path = wrapper_path
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        if batch_size < 1 or batch_size > 128:
            raise ValueError("batch_size must be between 1 and 128")
        self.batch_size = batch_size

    @staticmethod
    def validate_relative_path(value: str) -> str:
        if (
            not value
            or "\x00" in value
            or "\n" in value
            or "\r" in value
            or "\\" in value
            or value.startswith("/")
        ):
            raise ValueError("unsafe remote relative path")
        path = PurePosixPath(value)
        if ".." in path.parts or "." in path.parts:
            raise ValueError("remote path escapes the allowed root")
        normalized = posixpath.normpath(value)
        if normalized.startswith("../") or normalized == "..":
            raise ValueError("remote path escapes the allowed root")
        return normalized

    def _invoke(self, command: str, *, timeout_seconds: int | None = None) -> str:
        result = self.runner(
            [
                POWERSHELL,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                self.wrapper_path,
                command,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            check=False,
            shell=False,
        )
        if getattr(result, "returncode", 1) != 0:
            stderr = str(getattr(result, "stderr", "") or "").strip()
            stdout = str(getattr(result, "stdout", "") or "").strip()
            detail = stderr or stdout[-4000:] or "no remote diagnostic output"
            raise RemoteSourceError(
                f"read-only SSH command failed with exit {getattr(result, 'returncode', '?')}: {detail}"
            )
        return str(getattr(result, "stdout", ""))

    def discover(self, root: str) -> list[str]:
        if posixpath.normpath(root) != self.remote_root:
            raise ValueError("discovery root is not the allowed root")
        command = (
            f"find -P '{self.remote_root}' -type f "
            r"\( -name POSCAR -o -name CONTCAR -o -name INCAR -o -name OUTCAR \) -print0"
        )
        output = self._invoke(command)
        directories: set[str] = set()
        prefix = self.remote_root + "/"
        for raw_path in output.split("\x00"):
            if not raw_path:
                continue
            normalized = posixpath.normpath(raw_path)
            if not normalized.startswith(prefix):
                raise RemoteSourceError("remote discovery returned a path outside the allowed root")
            relative = posixpath.dirname(normalized[len(prefix) :])
            if relative == "":
                relative = ROOT_JOB_DIR
            directories.add(self.validate_relative_path(relative))
        return sorted(directories)

    @staticmethod
    def _snapshot_program() -> str:
        return """import base64,hashlib,json,os,sys
root=sys.argv[1]
path_arg=sys.argv[2]
path=base64.b64decode(path_arg[4:]).decode('utf-8') if path_arg.startswith('b64:') else path_arg
kind=sys.argv[3]
root_real=os.path.realpath(root)
real=os.path.realpath(path)
if os.path.commonpath([root_real,real]) != root_real:
    raise SystemExit(42)
if not os.path.isfile(real):
    print(json.dumps({'exists':False},sort_keys=True,separators=(',',':')))
    raise SystemExit(0)
before=os.stat(real)
h=hashlib.sha256()
chunks=[]
markers={}
tail=''
electronic_failed=False
small_limit=4*1024*1024
with open(real,'rb') as handle:
    for chunk in iter(lambda:handle.read(1024*1024),b''):
        h.update(chunk)
        if kind != 'OUTCAR' and before.st_size <= small_limit:
            chunks.append(chunk)
        elif kind == 'OUTCAR':
            text=tail+chunk.decode('utf-8','ignore')
            markers['normal_termination']=markers.get('normal_termination',False) or 'General timing and accounting informations' in text
            markers['ionic_converged']=markers.get('ionic_converged',False) or 'reached required accuracy' in text
            electronic_failed=electronic_failed or 'electronic minimization did not converge' in text
            markers['fatal_warning']=markers.get('fatal_warning',False) or any(x in text for x in ('VERY BAD NEWS','internal error','segmentation fault','No space left on device','Disk quota exceeded','Input/output error','MPI_ABORT'))
            tail=text[-256:]
after=os.stat(real)
if kind == 'OUTCAR':
    markers['electronic_converged']=not electronic_failed
elif before.st_size > small_limit:
    markers['too_large']=True
payload={'exists':True,'sha256':h.hexdigest(),'size':before.st_size,'mtime_ns':before.st_mtime_ns,
         'stable':before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns,
         'content':None if kind=='OUTCAR' or before.st_size>small_limit else b''.join(chunks).decode('utf-8','replace'),
         'markers':markers}
print(json.dumps(payload,sort_keys=True,separators=(',',':')))
"""

    @staticmethod
    def _parent_program() -> str:
        return """import base64,hashlib,json,os,sys
root=sys.argv[1]
job_arg=sys.argv[2]
job=base64.b64decode(job_arg[4:]).decode('utf-8') if job_arg.startswith('b64:') else job_arg
root_real=os.path.realpath(root)
job_real=os.path.realpath(job)
if os.path.commonpath([root_real,job_real]) != root_real:
    raise SystemExit(42)
candidates=[]
cursor=os.path.dirname(job_real)
while cursor != root_real and os.path.commonpath([root_real,cursor]) == root_real:
    for name in ('CONTCAR','POSCAR'):
        candidates.append(os.path.join(cursor,name))
    cursor=os.path.dirname(cursor)
parent=os.path.dirname(job_real)
try:
    siblings=sorted(os.scandir(parent),key=lambda item:item.name)
except OSError:
    siblings=[]
for entry in siblings:
    if not entry.is_dir(follow_symlinks=False) or entry.path == job_real:
        continue
    lowered=entry.name.lower()
    if not any(token in lowered for token in ('clean','slab','surface','substrate')):
        continue
    for name in ('CONTCAR','POSCAR'):
        candidates.append(os.path.join(entry.path,name))
seen=set()
for candidate in candidates:
    real=os.path.realpath(candidate)
    if real in seen or os.path.commonpath([root_real,real]) != root_real or not os.path.isfile(real):
        continue
    seen.add(real)
    before=os.stat(real)
    if before.st_size > 4*1024*1024:
        continue
    with open(real,'rb') as handle:
        data=handle.read()
    after=os.stat(real)
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        continue
    print(json.dumps({'identity':os.path.relpath(real,root_real).replace(os.sep,'/'),
                      'content':data.decode('utf-8','replace'),
                      'sha256':hashlib.sha256(data).hexdigest()},
                     sort_keys=True,separators=(',',':')))
    raise SystemExit(0)
print('null')
"""

    def _read_file(self, relative_job_dir: str, name: str) -> RemoteFileSnapshot | None:
        job_dir = self.validate_relative_path(relative_job_dir)
        relative_file = self.validate_relative_path(f"{job_dir}/{name}")
        absolute = (
            posixpath.join(self.remote_root, name)
            if job_dir == ROOT_JOB_DIR
            else posixpath.join(self.remote_root, relative_file)
        )
        path_token = "b64:" + base64.b64encode(absolute.encode("utf-8")).decode("ascii")
        encoded = base64.b64encode(self._snapshot_program().encode()).decode()
        command = (
            "python3 -c 'import base64,sys;exec(base64.b64decode(sys.argv.pop(1)))' '"
            + encoded
            + "' '"
            + self.remote_root
            + "' '"
            + path_token
            + "' '"
            + name
            + "'"
        )
        payload = json.loads(self._invoke(command))
        if not payload.get("exists", True):
            return None
        return RemoteFileSnapshot(
            relative_path=name,
            sha256=payload["sha256"],
            size=payload["size"],
            mtime_ns=payload["mtime_ns"],
            content=payload["content"],
            markers=payload["markers"],
            stable=payload["stable"],
        )

    def _find_clean_parent(self, relative_job_dir: str) -> dict[str, str] | None:
        job_dir = self.validate_relative_path(relative_job_dir)
        absolute = self.remote_root if job_dir == ROOT_JOB_DIR else posixpath.join(self.remote_root, job_dir)
        path_token = "b64:" + base64.b64encode(absolute.encode("utf-8")).decode("ascii")
        encoded = base64.b64encode(self._parent_program().encode()).decode()
        command = (
            "python3 -c 'import base64,sys;exec(base64.b64decode(sys.argv.pop(1)))' '"
            + encoded
            + "' '"
            + self.remote_root
            + "' '"
            + path_token
            + "'"
        )
        payload = json.loads(self._invoke(command))
        if payload is None:
            return None
        identity = self.validate_relative_path(str(payload["identity"]))
        content = str(payload["content"])
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest != payload["sha256"]:
            raise RemoteSourceError("clean-parent content hash mismatch")
        return {"identity": identity, "content": content, "sha256": digest}

    @staticmethod
    def _job_snapshot_program() -> str:
        return """import base64,hashlib,json,os,sys
root=sys.argv[1]
job_arg=sys.argv[2]
job=base64.b64decode(job_arg[4:]).decode('utf-8') if job_arg.startswith('b64:') else job_arg
root_real=os.path.realpath(root)
job_real=os.path.realpath(job)
if os.path.commonpath([root_real,job_real]) != root_real:
    raise SystemExit(42)
names=('POSCAR','CONTCAR','INCAR','OUTCAR')
paths={}
before={}
for name in names:
    real=os.path.realpath(os.path.join(job_real,name))
    if os.path.commonpath([root_real,real]) != root_real or not os.path.isfile(real):
        continue
    paths[name]=real
    stat=os.stat(real)
    before[name]=(stat.st_size,stat.st_mtime_ns)
files={}
small_limit=4*1024*1024
for name,real in paths.items():
    h=hashlib.sha256(); chunks=[]; markers={}; tail=''; electronic_failed=False; ionic_steps=0
    with open(real,'rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
            if name != 'OUTCAR' and before[name][0] <= small_limit:
                chunks.append(chunk)
            elif name == 'OUTCAR':
                old_tail=len(tail); text=tail+chunk.decode('utf-8','ignore')
                lowered=text.lower(); run_starts=[]; cursor=0
                while True:
                    found=lowered.find('vasp.',cursor)
                    if found < 0: break
                    if found+5 > old_tail: run_starts.append(found)
                    cursor=found+1
                if run_starts:
                    text=text[run_starts[-1]:]; lowered=text.lower()
                    markers={}; electronic_failed=False; ionic_steps=0; old_tail=0
                markers['normal_termination']=markers.get('normal_termination',False) or 'general timing and accounting informations' in lowered
                markers['ionic_converged']=markers.get('ionic_converged',False) or 'reached required accuracy' in lowered
                electronic_failed=electronic_failed or 'electronic minimization did not converge' in lowered
                fatal_tokens=('very bad news','internal error','segmentation fault','no space left on device','disk quota exceeded','input/output error','mpi_abort','brmix: very serious problems','zbrent: fatal','forrtl: severe','out of memory','oom-kill','killed by signal')
                markers['fatal_warning']=markers.get('fatal_warning',False) or any(x in lowered for x in fatal_tokens)
                phrase='FREE ENERGIE OF THE ION-ELECTRON SYSTEM'; start=0
                while True:
                    found=text.find(phrase,start)
                    if found < 0: break
                    if found+len(phrase) > old_tail: ionic_steps+=1
                    start=found+1
                tail=text[-256:]
    if name == 'OUTCAR':
        markers.setdefault('normal_termination',False); markers.setdefault('ionic_converged',False); markers.setdefault('fatal_warning',False)
        markers['electronic_converged']=not electronic_failed; markers['ionic_steps']=ionic_steps if ionic_steps else None
    elif before[name][0] > small_limit:
        markers['too_large']=True
    files[name]={'sha256':h.hexdigest(),'size':before[name][0],'mtime_ns':before[name][1],
                 'content':None if name=='OUTCAR' or before[name][0]>small_limit else b''.join(chunks).decode('utf-8','replace'),
                 'markers':markers}
after={}; after_names=[]
for name in names:
    real=os.path.realpath(os.path.join(job_real,name))
    if os.path.commonpath([root_real,real]) != root_real or not os.path.isfile(real): continue
    after_names.append(name)
    try:
        stat=os.stat(real); after[name]=(stat.st_size,stat.st_mtime_ns)
    except OSError: pass
stable=set(before)==set(after)==set(after_names) and all(before[name]==after[name] for name in before)
for item in files.values(): item['stable']=stable
candidates=[]; cursor=os.path.dirname(job_real)
while cursor != root_real and os.path.commonpath([root_real,cursor]) == root_real:
    for name in ('CONTCAR','POSCAR'): candidates.append(os.path.join(cursor,name))
    cursor=os.path.dirname(cursor)
parent_dir=job_real if job_real == root_real else os.path.dirname(job_real)
try: siblings=sorted(os.scandir(parent_dir),key=lambda item:item.name)
except OSError: siblings=[]
for entry in siblings:
    if entry.is_dir(follow_symlinks=False) and entry.path != job_real and any(x in entry.name.lower() for x in ('clean','slab','surface','substrate')):
        for name in ('CONTCAR','POSCAR'): candidates.append(os.path.join(entry.path,name))
clean=None; seen=set()
for candidate in candidates:
    real=os.path.realpath(candidate)
    if real in seen or os.path.commonpath([root_real,real]) != root_real or not os.path.isfile(real): continue
    seen.add(real); first=os.stat(real)
    if first.st_size>small_limit: continue
    with open(real,'rb') as handle: data=handle.read()
    last=os.stat(real)
    if (first.st_size,first.st_mtime_ns)!=(last.st_size,last.st_mtime_ns): continue
    clean={'identity':os.path.relpath(real,root_real).replace(os.sep,'/'),'content':data.decode('utf-8','replace'),'sha256':hashlib.sha256(data).hexdigest()}; break
print(json.dumps({'files':files,'clean_parent':clean,'before_names':sorted(before),'after_names':sorted(after_names)},sort_keys=True,separators=(',',':')))
"""

    @staticmethod
    def _batch_snapshot_program() -> str:
        return """import base64,json,subprocess,sys
program=base64.b64decode(sys.argv[1]).decode('utf-8')
root=sys.argv[2]
jobs=json.loads(base64.b64decode(sys.argv[3]).decode('utf-8'))
for job in jobs:
    result=subprocess.run([sys.executable,'-c',program,root,job],capture_output=True,text=True,check=False)
    if result.returncode != 0:
        detail=(result.stderr or result.stdout or 'no diagnostic output')[-2000:]
        print(json.dumps({'ok':False,'job':job,'returncode':result.returncode,'error':detail},sort_keys=True,separators=(',',':')))
        raise SystemExit(43)
    try:
        payload=json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(json.dumps({'ok':False,'job':job,'returncode':44,'error':str(exc)},sort_keys=True,separators=(',',':')))
        raise SystemExit(44)
    print(json.dumps({'ok':True,'job':job,'payload':payload},sort_keys=True,separators=(',',':')))
"""

    def _snapshot_from_payload(
        self, payload: dict[str, object], *, relative_job_dir: str, root_id: str
    ) -> JobSnapshot:
        job_dir = self.validate_relative_path(relative_job_dir)
        raw_files = payload.get("files")
        if not isinstance(raw_files, dict):
            raise RemoteSourceError(f"remote snapshot has no file map: {job_dir}")
        name_set_stable = sorted(payload.get("before_names", raw_files)) == sorted(
            payload.get("after_names", raw_files)
        )
        files = {
            str(name): RemoteFileSnapshot(
                relative_path=str(name),
                **{
                    **item,
                    "stable": bool(item.get("stable", True)) and name_set_stable,
                },
            )
            for name, item in raw_files.items()
            if isinstance(item, dict)
        }
        names = ("POSCAR", "CONTCAR", "INCAR", "OUTCAR")
        signature_payload = [
            (name, item.sha256, item.size, item.mtime_ns, item.stable)
            for name, item in sorted(files.items())
        ]
        missing = sorted(set(names) - set(files))
        parent = payload.get("clean_parent")
        if parent is not None and not isinstance(parent, dict):
            raise RemoteSourceError(f"remote clean-parent payload is invalid: {job_dir}")
        signature = hashlib.sha256(
            json.dumps(
                {
                    "files": signature_payload,
                    "missing": missing,
                    "clean_parent": None
                    if parent is None
                    else (parent["identity"], parent["sha256"]),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return JobSnapshot(
            root_id=root_id,
            relative_job_dir=job_dir,
            source_signature=signature,
            files=files,
            clean_parent_identity=None if parent is None else str(parent["identity"]),
            clean_parent_poscar=None if parent is None else str(parent["content"]),
            captured_at=datetime.now(timezone.utc),
        )

    def iter_snapshots(
        self, relative_job_dirs: list[str], *, root_id: str
    ) -> Iterator[tuple[str, JobSnapshot]]:
        """Fetch bounded batches over one SSH connection per batch."""

        validated = [self.validate_relative_path(path) for path in relative_job_dirs]
        for start in range(0, len(validated), self.batch_size):
            batch = validated[start : start + self.batch_size]
            absolute_tokens = []
            for job_dir in batch:
                absolute = (
                    self.remote_root
                    if job_dir == ROOT_JOB_DIR
                    else posixpath.join(self.remote_root, job_dir)
                )
                absolute_tokens.append(
                    "b64:" + base64.b64encode(absolute.encode("utf-8")).decode("ascii")
                )
            encoded_batch = base64.b64encode(self._batch_snapshot_program().encode()).decode()
            encoded_job = base64.b64encode(self._job_snapshot_program().encode()).decode()
            encoded_paths = base64.b64encode(
                json.dumps(absolute_tokens, separators=(",", ":")).encode()
            ).decode()
            command = (
                "python3 -c 'import base64,sys;exec(base64.b64decode(sys.argv.pop(1)))' '"
                + encoded_batch
                + "' '"
                + encoded_job
                + "' '"
                + self.remote_root
                + "' '"
                + encoded_paths
                + "'"
            )
            output = self._invoke(
                command,
                timeout_seconds=max(self.timeout_seconds, 10 * len(batch)),
            )
            lines = [line for line in output.splitlines() if line.strip()]
            if len(lines) != len(batch):
                raise RemoteSourceError(
                    f"remote batch returned {len(lines)} snapshots for {len(batch)} jobs"
                )
            for expected, line in zip(batch, lines):
                try:
                    envelope = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RemoteSourceError(
                        f"remote batch returned invalid JSON for {expected}"
                    ) from exc
                if envelope.get("ok") is not True:
                    raise RemoteSourceError(
                        f"remote batch failed for {expected}: {envelope.get('error', 'unknown error')}"
                    )
                yield (
                    expected,
                    self._snapshot_from_payload(
                        envelope["payload"],
                        relative_job_dir=expected,
                        root_id=root_id,
                    ),
                )

    def snapshot(self, relative_job_dir: str, *, root_id: str) -> JobSnapshot:
        job_dir = self.validate_relative_path(relative_job_dir)
        absolute = self.remote_root if job_dir == ROOT_JOB_DIR else posixpath.join(self.remote_root, job_dir)
        path_token = "b64:" + base64.b64encode(absolute.encode("utf-8")).decode("ascii")
        encoded = base64.b64encode(self._job_snapshot_program().encode()).decode()
        command = (
            "python3 -c 'import base64,sys;exec(base64.b64decode(sys.argv.pop(1)))' '"
            + encoded
            + "' '"
            + self.remote_root
            + "' '"
            + path_token
            + "'"
        )
        return self._snapshot_from_payload(
            json.loads(self._invoke(command)),
            relative_job_dir=job_dir,
            root_id=root_id,
        )
