"""Shared collect / analyze glue for 07–15."""

from __future__ import annotations

from pathlib import Path

from goto_mid import add_movej_args
from id_math import read_json
from paths import DATA, kind_dirs
from window_a import (
    AlignmentError,
    add_window_a_arg,
    copy_window_a,
    merge_logs,
    require_window_a,
    resolve_window_a,
)


def add_contact_args(parser, *, abort_n: float, contact_n: float = 0.40) -> None:
    parser.add_argument("--csv", default="", help="command CSV; Window A sibling or --window-a-csv")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--contact-n", type=float, default=contact_n)
    parser.add_argument("--abort-n", type=float, default=abort_n)
    parser.add_argument("--theta-axis", type=int, default=4, help="ωθ index in the 6-D twist (default wy)")
    parser.add_argument("--scan-axis", type=int, default=0, help="path tangent index (default vx)")
    parser.add_argument("--dry-run", action="store_true")
    add_window_a_arg(parser)
    add_movej_args(parser)


def load_aligned(cmd_csv: Path, window_a_csv: str | Path | None = None):
    data_dir = Path(cmd_csv).parent
    wa = require_window_a(window_a_csv, data_dir)
    return merge_logs(Path(cmd_csv), wa)


def stash_window_a(window_a_csv: str | Path | None, data_dir: Path) -> Path:
    src = resolve_window_a(window_a_csv, data_dir)
    if src is None:
        raise AlignmentError(
            "pass --window-a-csv from Window A --log-csv; "
            "06 already showed MotionBus cannot replace it"
        )
    return copy_window_a(src, data_dir)


def open_kind(kind: str, cmd_csv: Path | None = None) -> tuple[Path, Path]:
    preserve = cmd_csv if cmd_csv is not None and Path(cmd_csv).is_file() else None
    return kind_dirs(kind, preserve=preserve)


KIND_JSON = {
    "08_ke": "ke.json",
    "11_exec_2dof": "exec.json",
    "12_stop_tail": "tail.json",
    "13_contact_hs": "hs.json",
    "14_port_energy": "energy.json",
}


def find_kind_json(kind: str, filename: str | None = None, *, root: Path | None = None) -> Path | None:
    folder = (root or DATA) / kind
    path = folder / (filename or KIND_JSON.get(kind, "out.json"))
    return path if path.is_file() else None


def load_kind_json(kind: str, filename: str | None = None, *, root: Path | None = None) -> dict | None:
    path = find_kind_json(kind, filename, root=root)
    if path is None:
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    payload["_path"] = str(path)
    return payload


def load_id_bundle(*, root: Path | None = None) -> dict[str, dict | None]:
    """Read 08/11–14 payloads.  Missing files stay None — 15 must not fit them."""

    return {kind: load_kind_json(kind, root=root) for kind in KIND_JSON}
