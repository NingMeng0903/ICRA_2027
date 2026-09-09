"""ICRA_2027 layout, run provenance, and playground import path.

New collections are immutable run directories under ``DATA/runs`` and
``VISU/runs``.  ``DATA/<kind>`` and ``VISU/<kind>`` remain stable latest
symlinks after the first new collection.  Existing flat folders are migrated
only when a collection is started; importing this module never touches them.
The helpers are intentionally hardware-free.
"""

from __future__ import annotations

import shutil
import sys
import json
import hashlib
import importlib.metadata
import os
import re
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable, Mapping

ICRA_ROOT = Path("/media/camp/EXT_DRIVE/ICRA_2027")
FORCE_TEST = ICRA_ROOT / "FORCE_TEST"
DATA = ICRA_ROOT / "DATA"
VISU = ICRA_ROOT / "VISU"
# Select a reviewed controller checkout once for the whole experiment run.
PLAYGROUND = Path(os.environ.get(
    "REALUS_PLAYGROUND_ROOT", "/media/camp/EXT_DRIVE/RealUS_playground"
)).expanduser().resolve()
_ACTIVE_RUN_ID: str | None = None

_SOURCE_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
        ".sh", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".cmake",
    }
)
_CONFIG_SUFFIXES = frozenset({".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"})
_EXCLUDED_PARTS = frozenset(
    {"DATA", "VISU", "meshes", "mesh", ".worktrees", "worktrees", ".git", "__pycache__"}
)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def add_run_arg(parser) -> None:
    """Add the stable run selector shared by every FORCE_TEST entry point."""
    parser.add_argument(
        "--run-id",
        default="",
        help="stable output id; defaults to the collection timestamp",
    )


def sanitize_run_id(value: str | None) -> str:
    raw = str(value or "").strip()
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return raw[:96] or stamp()


def set_active_run_id(value: str | None) -> str:
    global _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = sanitize_run_id(value)
    return _ACTIVE_RUN_ID


def active_run_id(default: str | None = None) -> str:
    return _ACTIVE_RUN_ID or sanitize_run_id(default)


def dry_exit(args) -> bool:
    """True → caller must return 0.  Writes nothing to DATA/ or VISU/."""
    if hasattr(args, "run_id"):
        # Establish one id before either analyze or collect.  The dry path
        # still returns before any directory helper can create a file.
        set_active_run_id(getattr(args, "run_id", "") or active_run_id())
    if hasattr(args, "dof"):
        from dof import set_requested_dof

        set_requested_dof(int(getattr(args, "dof")))
    if not getattr(args, "dry_run", False):
        return False
    print("[DRY] no drive, no DATA, no VISU", flush=True)
    return True


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    VISU.mkdir(parents=True, exist_ok=True)


def configure_roots(*, data: Path | None = None, visu: Path | None = None) -> None:
    """Override roots for offline tests.

    Production scripts keep the constants above.  Tests can point both roots
    at a temporary directory without touching the repository's current DATA or
    VISU folders.
    """
    global DATA, VISU
    if data is not None:
        DATA = Path(data)
    if visu is not None:
        VISU = Path(visu)


def add_playground() -> Path:
    for extra in (PLAYGROUND, PLAYGROUND / "rm75_control", PLAYGROUND / "src"):
        text = str(extra)
        if text not in sys.path:
            sys.path.insert(0, text)
    return PLAYGROUND


def _is_kind(name: str, kind: str) -> bool:
    return name == kind or name.startswith(kind + "_")


def _rm(path: Path) -> None:
    # ``Path.is_dir()`` follows symlinks; unlink a link before considering
    # recursive removal so a compatibility latest link can never delete its
    # run target.
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def wipe_kind(kind: str, *, preserve: Path | None = None) -> list[str]:
    """Deprecated compatibility shim.

    The old implementation deleted prior runs.  Keeping this function as a
    no-op prevents an out-of-tree caller from silently destroying evidence;
    new code uses :func:`kind_dirs` and immutable run directories.
    """
    del kind, preserve
    return []


def _runs_root(root: Path, kind: str) -> Path:
    return Path(root) / "runs" / str(kind)


def _managed_run(path: Path, root: Path, kind: str) -> Path | None:
    try:
        resolved = Path(path).resolve()
        base = _runs_root(root, kind).resolve()
        rel = resolved.relative_to(base)
    except (ValueError, OSError):
        return None
    if len(rel.parts) == 0:
        return None
    return base / rel.parts[0]


def _latest_target(root: Path, kind: str) -> Path:
    link = Path(root) / str(kind)
    if link.is_symlink():
        try:
            return link.resolve()
        except OSError:
            return link
    return link


def resolve_kind_dir(kind: str, *, root: Path | None = None) -> Path:
    """Resolve a latest kind directory, including legacy flat layouts."""
    root = Path(DATA if root is None else root)
    fixed = root / str(kind)
    if fixed.exists() or fixed.is_symlink():
        return fixed
    runs = _runs_root(root, kind)
    if runs.is_dir():
        candidates = sorted((p for p in runs.iterdir() if p.is_dir()), key=lambda p: p.name)
        if candidates:
            return candidates[-1]
    return fixed


def _atomic_latest_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.with_name(f".{link.name}.latest.{os.getpid()}")
    try:
        if tmp.exists() or tmp.is_symlink():
            _rm(tmp)
        rel = os.path.relpath(str(target), str(link.parent))
        os.symlink(rel, tmp)
        os.replace(tmp, link)
    finally:
        if tmp.exists() or tmp.is_symlink():
            _rm(tmp)


def _migrate_legacy(root: Path, kind: str, run_id: str) -> None:
    """Move a flat legacy folder exactly once, only on a new collection."""
    legacy = Path(root) / str(kind)
    if not legacy.is_dir() or legacy.is_symlink():
        return
    dst_root = _runs_root(root, kind)
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / f"legacy_{run_id}"
    suffix = 1
    while dst.exists() or dst.is_symlink():
        dst = dst_root / f"legacy_{run_id}_{suffix}"
        suffix += 1
    shutil.move(str(legacy), str(dst))


def _fresh_run_id(data_root: Path, visu_root: Path, kind: str, run_id: str) -> str:
    """Avoid reusing an existing id when a new collection starts.

    ``stamp()`` has one-second resolution and two axes/experiments can be
    launched in the same second.  Reusing that directory would make the
    second run append or overwrite the first run's evidence.  Analysis of a
    managed CSV bypasses this helper and deliberately reuses its run.
    """
    base = sanitize_run_id(run_id)
    candidate = base
    suffix = 1
    while (
        (_runs_root(data_root, kind) / candidate).exists()
        or (_runs_root(data_root, kind) / candidate).is_symlink()
        or (_runs_root(visu_root, kind) / candidate).exists()
        or (_runs_root(visu_root, kind) / candidate).is_symlink()
    ):
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded_relative(path: Path) -> bool:
    return any(part in _EXCLUDED_PARTS for part in Path(path).parts)


def _source_path(path: Path) -> bool:
    return path.suffix.lower() in _SOURCE_SUFFIXES and not _excluded_relative(path)


def _run_git(root: Path, args: list[str], *, binary: bool = False) -> bytes:
    command = ["git", "-C", str(root), *args]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode != 0:
        return b""
    if binary:
        return bytes(result.stdout)
    return str(result.stdout).encode("utf-8", "replace")


def _git_status(root: Path) -> list[dict[str, str]]:
    raw = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    if not raw:
        return []
    out: list[dict[str, str]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if len(line) < 4:
            continue
        state = line[:2]
        name = line[3:]
        if " -> " in name:
            name = name.rsplit(" -> ", 1)[-1]
        name = name.strip().strip('"')
        rel = Path(name)
        if not name or _excluded_relative(rel):
            continue
        out.append({"status": state, "path": name})
    return out


def _git_record(root: Path, label: str, snapshot_root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    head_raw = _run_git(root, ["rev-parse", "HEAD"])
    head = head_raw.decode("utf-8", "replace").strip() if head_raw else None
    if head and "fatal:" in head:
        head = None
    status = _git_status(root)
    diff_args = [
        "diff",
        "--binary",
        "--no-ext-diff",
        "--",
        ":(exclude)DATA/**",
        ":(exclude)VISU/**",
        ":(exclude)meshes/**",
        ":(exclude).worktrees/**",
        ":(exclude)worktrees/**",
    ]
    diff = _run_git(root, diff_args, binary=True)
    files: list[dict[str, object]] = []
    for item in status:
        rel = Path(item["path"])
        absolute = root / rel
        entry: dict[str, object] = {"path": item["path"], "status": item["status"]}
        if not _source_path(rel):
            continue
        if absolute.is_file() and not absolute.is_symlink():
            try:
                content = absolute.read_bytes()
                target = snapshot_root / label / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                entry.update(
                    {
                        "present": True,
                        "size": len(content),
                        "sha256": _sha256_bytes(content),
                        "snapshot": str(target.relative_to(snapshot_root)),
                    }
                )
            except OSError as exc:
                entry.update({"present": False, "error": str(exc)})
        else:
            entry.update({"present": False, "sha256": None})
        files.append(entry)
    return {
        "root": str(root),
        "head": head,
        "dirty": bool(status),
        "status": status,
        "diff_sha256": _sha256_bytes(diff),
        "source_files": files,
    }


def _copy_file_record(path: Path, snapshot_root: Path, *, category: str) -> dict[str, object]:
    path = Path(path)
    entry: dict[str, object] = {"path": str(path), "present": False}
    if not path.is_file() or path.is_symlink():
        entry["error"] = "missing_or_symlink"
        return entry
    try:
        content = path.read_bytes()
        digest = _sha256_bytes(content)
        # Keep absolute external config paths out of the snapshot layout while
        # retaining a deterministic, collision-resistant file name.
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path).strip("/"))
        target = snapshot_root / category / f"{safe_name}.{digest[:12]}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entry.update(
            {
                "present": True,
                "size": len(content),
                "sha256": digest,
                "snapshot": str(target.relative_to(snapshot_root)),
            }
        )
    except OSError as exc:
        entry["error"] = str(exc)
    return entry


def _config_candidates(
    *,
    config_paths: Iterable[Path] | None,
    argv: list[str],
) -> list[Path]:
    found: dict[str, Path] = {}

    def add(path: Path) -> None:
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            resolved = Path(path).expanduser()
        if resolved.is_file() and resolved.suffix.lower() in _CONFIG_SUFFIXES:
            found[str(resolved)] = resolved

    for path in config_paths or ():
        add(Path(path))
    for index, arg in enumerate(argv):
        text = str(arg)
        if text.startswith("--config="):
            add(Path(text.split("=", 1)[1]))
        elif text in {"--config", "-c"} and index + 1 < len(argv):
            add(Path(argv[index + 1]))
    for key, value in os.environ.items():
        if "CONFIG" not in key.upper():
            continue
        candidate = Path(str(value)).expanduser()
        if candidate.is_file():
            add(candidate)

    # The controller's YAML/JSON configuration tree is small and stable.  A
    # run records all candidates because an ICRA entry point does not always
    # expose the daemon's final resolved config path in its public API.
    for root in (PLAYGROUND / "rm75_control" / "configs", PLAYGROUND / "configs"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not _excluded_relative(path.relative_to(root)):
                add(path)
    return [found[key] for key in sorted(found)]


def _native_binary_candidates(extra: Iterable[Path] | None) -> list[Path]:
    found: dict[str, Path] = {}

    def add(path: Path) -> None:
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            resolved = Path(path).expanduser()
        if resolved.is_file() and (resolved.suffix.startswith(".so") or os.access(resolved, os.X_OK)):
            found[str(resolved)] = resolved

    for path in extra or ():
        add(Path(path))
    for key, value in os.environ.items():
        upper = key.upper()
        if any(token in upper for token in ("NATIVE", "WBC_RT", "BINARY")):
            add(Path(str(value)))
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and str(path).endswith((".so", ".so.1", ".so.2")):
            add(Path(path))
    return [found[key] for key in sorted(found)]


def _dependency_versions() -> dict[str, str | None]:
    names = ("numpy", "scipy", "osqp", "cvxpy", "PyYAML", "peirastic", "rm75-control")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as exc:
            versions[name] = f"error:{exc}"
    return versions


def capture_run_provenance(
    data: Path,
    *,
    kind: str,
    run_id: str,
    repo_roots: Mapping[str, Path] | None = None,
    config_paths: Iterable[Path] | None = None,
    native_binaries: Iterable[Path] | None = None,
    argv: Iterable[str] | None = None,
) -> dict[str, object]:
    """Write one immutable source/config manifest for a new collection.

    The manifest stores copies of changed/untracked source files and config
    files under the run directory, alongside their hashes.  Existing
    ``provenance.json`` files are never refreshed, so later edits to a dirty
    checkout or its config cannot rewrite an earlier run's provenance.
    """

    data = Path(data)
    manifest_path = data / "provenance.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"manifest": str(manifest_path), "error": "existing_manifest_unreadable"}
    snapshot_root = data / "source_snapshot"
    command = list(sys.argv if argv is None else argv)
    roots = dict(
        repo_roots
        or {
            "icra_2027": ICRA_ROOT,
            "realus_playground": PLAYGROUND,
        }
    )
    repositories = {
        str(label): _git_record(Path(root), str(label), snapshot_root)
        for label, root in roots.items()
        if Path(root).exists()
    }
    config_records = [
        _copy_file_record(path, snapshot_root, category="config_snapshot")
        for path in _config_candidates(config_paths=config_paths, argv=command)
    ]
    native_records = []
    for path in _native_binary_candidates(native_binaries):
        try:
            native_records.append(
                {
                    "path": str(path),
                    "present": True,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        except OSError as exc:
            native_records.append({"path": str(path), "present": False, "error": str(exc)})
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": str(kind),
        "run_id": str(run_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": {
            "argv": command,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "python": platform.python_version(),
        },
        "repositories": repositories,
        "configs": config_records,
        "native_binaries": native_records,
        "dependencies": _dependency_versions(),
    }
    data.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _run_metadata(data: Path, *, kind: str, run_id: str) -> None:
    path = Path(data) / "run.json"
    if path.exists():
        return
    meta: dict[str, object] = {
        "kind": str(kind),
        "run_id": str(run_id),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        from dof import current_metadata

        meta.update(current_metadata())
    except Exception:
        meta.update(
            {
                "dof_requested": None,
                "dof_active": None,
                "dof_config_source": "unknown",
                "controller_api_version": "unknown",
            }
        )
    provenance = capture_run_provenance(data, kind=kind, run_id=run_id)
    meta["provenance_file"] = "provenance.json"
    meta["provenance_schema_version"] = provenance.get("schema_version", 1)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def kind_dirs(kind: str, *, preserve: Path | None = None, run_id: str | None = None) -> tuple[Path, Path]:
    """Return an immutable run directory and update latest links safely.

    A collect is recognized by ``preserve is None``.  It migrates a legacy
    flat folder only then.  Analysis of a CSV already inside a managed run
    reuses that run; analysis of an old flat CSV writes a new run and leaves
    the old folder untouched.
    """
    ensure_dirs()
    rid = sanitize_run_id(run_id or active_run_id())
    data_root = Path(DATA)
    visu_root = Path(VISU)
    preserve_path = Path(preserve).resolve() if preserve is not None else None
    managed_data = _managed_run(preserve_path, data_root, kind) if preserve_path else None
    managed_visu = _managed_run(preserve_path, visu_root, kind) if preserve_path else None
    existing = managed_data or managed_visu
    if existing is not None:
        data = existing
        visu = _runs_root(visu_root, kind) / existing.name
        visu.mkdir(parents=True, exist_ok=True)
        data.mkdir(parents=True, exist_ok=True)
        _run_metadata(data, kind=kind, run_id=existing.name)
        return data, visu

    # Only a new collection may move a legacy folder.  ``--csv`` analysis
    # never invokes this migration because it supplies ``preserve``.  An
    # external/legacy CSV still gets a fresh run id when the requested id is
    # already occupied; only a CSV already inside a managed run may reuse it.
    if preserve is None:
        _migrate_legacy(data_root, kind, rid)
        _migrate_legacy(visu_root, kind, rid)
        rid = _fresh_run_id(data_root, visu_root, kind, rid)
        set_active_run_id(rid)
    else:
        rid = _fresh_run_id(data_root, visu_root, kind, rid)
    data = _runs_root(data_root, kind) / rid
    visu = _runs_root(visu_root, kind) / rid
    data.mkdir(parents=True, exist_ok=True)
    visu.mkdir(parents=True, exist_ok=True)
    _run_metadata(data, kind=kind, run_id=rid)
    # A real legacy directory blocks the compatibility link until the next
    # collection migrates it.  This is deliberate: analyze must not mutate it.
    fixed_data = data_root / str(kind)
    fixed_visu = visu_root / str(kind)
    if not (fixed_data.exists() and not fixed_data.is_symlink()):
        _atomic_latest_link(fixed_data, data)
    if not (fixed_visu.exists() and not fixed_visu.is_symlink()):
        _atomic_latest_link(fixed_visu, visu)
    return data, visu


def run_dir(kind: str, when: str | None = None) -> Path:
    data, _visu = kind_dirs(kind, run_id=when)
    return data


def write_readme(folder: Path, body: str) -> Path:
    """Write a run-scoped VISU README; a later analysis may overwrite it.

    A folder with data must hold the latest analysis (numbers + verdict),
    not just what the script is for. Purpose-only text is for uncollected
    stubs.
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "README.md"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def visu_dir(kind: str, when: str | None = None) -> Path:
    return run_visu_dir(kind, run_id=when)


def run_visu_dir(
    kind: str,
    *,
    run_id: str | None = None,
    collect: bool = False,
) -> Path:
    """Create a run-scoped VISU directory without creating a DATA run.

    ``collect=True`` is the one path allowed to migrate a legacy flat VISU
    folder.  Analyze-only callers leave that folder untouched and receive a
    printed run path without replacing the compatibility link.
    """
    ensure_dirs()
    rid = sanitize_run_id(run_id or active_run_id())
    root = Path(VISU)
    fixed = root / str(kind)
    if collect:
        _migrate_legacy(root, kind, rid)
    # Index/report analysis has no source path to identify a managed run, so
    # never reuse an occupied output directory.  This also protects a second
    # collect launched in the same timestamp from appending to the first.
    rid = _fresh_run_id(Path(DATA), root, kind, rid)
    if fixed.is_dir() and not fixed.is_symlink():
        # A legacy visualization folder is left untouched until a collection
        # explicitly migrates it through this function or kind_dirs().
        out = _runs_root(root, kind) / rid
    else:
        out = _runs_root(root, kind) / rid
        _atomic_latest_link(fixed, out)
    out.mkdir(parents=True, exist_ok=True)
    return out
