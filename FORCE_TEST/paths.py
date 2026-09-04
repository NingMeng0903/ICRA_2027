"""ICRA_2027 layout and playground import path.

DATA = raw + JSON tables.  VISU = figures.  Scripts stay in FORCE_TEST.
Each script kind keeps only the latest run: DATA/<kind>/ and VISU/<kind>/.
A new collect or --csv analyze deletes that kind's previous folders first.
Does not drive the arm.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

ICRA_ROOT = Path("/media/camp/EXT_DRIVE/ICRA_2027")
FORCE_TEST = ICRA_ROOT / "FORCE_TEST"
DATA = ICRA_ROOT / "DATA"
VISU = ICRA_ROOT / "VISU"
PLAYGROUND = Path("/media/camp/EXT_DRIVE/RealUS_playground")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def dry_exit(args) -> bool:
    """True → caller must return 0.  Writes nothing to DATA/ or VISU/."""
    if not getattr(args, "dry_run", False):
        return False
    print("[DRY] no drive, no DATA, no VISU", flush=True)
    return True


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    VISU.mkdir(parents=True, exist_ok=True)


def add_playground() -> Path:
    for extra in (PLAYGROUND, PLAYGROUND / "rm75_control", PLAYGROUND / "src"):
        text = str(extra)
        if text not in sys.path:
            sys.path.insert(0, text)
    return PLAYGROUND


def _is_kind(name: str, kind: str) -> bool:
    return name == kind or name.startswith(kind + "_")


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def wipe_kind(kind: str, *, preserve: Path | None = None) -> list[str]:
    """Delete previous DATA/VISU folders for this script kind.

    If *preserve* is a file inside DATA/<kind>/ (the log being re-analyzed),
    that folder is kept so the CSV is not deleted; JSON/figures are overwritten.
    """
    ensure_dirs()
    keep: Path | None = None
    if preserve is not None:
        src = Path(preserve).resolve()
        candidate = (DATA / kind).resolve()
        try:
            src.relative_to(candidate)
            keep = candidate
        except ValueError:
            keep = None
    removed: list[str] = []
    for root in (DATA, VISU):
        if not root.is_dir():
            continue
        for path in list(root.iterdir()):
            if not _is_kind(path.name, kind):
                continue
            if keep is not None and path.resolve() == keep:
                continue
            readme = None
            note = path / "README.md"
            if path.is_dir() and note.is_file():
                readme = note.read_text(encoding="utf-8")
            _rm(path)
            if readme is not None:
                path.mkdir(parents=True, exist_ok=True)
                (path / "README.md").write_text(readme, encoding="utf-8")
            removed.append(str(path))
    if removed:
        print(f"[WIPE] {kind}: removed {len(removed)} old folder(s)", flush=True)
    return removed


def kind_dirs(kind: str, *, preserve: Path | None = None) -> tuple[Path, Path]:
    """Stable latest folders. Wipes older runs of this kind first."""
    wipe_kind(kind, preserve=preserve)
    data = DATA / kind
    visu = VISU / kind
    data.mkdir(parents=True, exist_ok=True)
    visu.mkdir(parents=True, exist_ok=True)
    return data, visu


def run_dir(kind: str, when: str | None = None) -> Path:
    del when
    data, _visu = kind_dirs(kind)
    return data


def write_readme(folder: Path, body: str) -> Path:
    """Write VISU/<kind>/README.md. Survives wipe; analyze overwrites.

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
    del when
    ensure_dirs()
    out = VISU / kind
    out.mkdir(parents=True, exist_ok=True)
    return out
