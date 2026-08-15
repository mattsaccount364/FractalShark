"""Shared helpers for the NCU attribution scripts in this folder.

Usage is self-contained: each script locates the ncu_report Python package and
ncu.exe automatically (override with NCU_PYTHON_DIR / NCU_EXE environment
variables, or the --ncu-python-dir / --ncu options of the scripts).
"""

import csv
import io
import os
import re
import sys


def find_ncu_python_dir() -> str:
    env = os.environ.get("NCU_PYTHON_DIR")
    if env:
        return env
    root = r"C:\Program Files\NVIDIA Corporation"
    cands = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.startswith("Nsight Compute"):
                d = os.path.join(root, name, "extras", "python")
                if os.path.isdir(d):
                    cands.append((name, d))
    if not cands:
        raise SystemExit(
            "Could not find Nsight Compute python extras. "
            "Set NCU_PYTHON_DIR.")
    cands.sort(reverse=True)
    return cands[0][1]


def find_ncu_exe() -> str:
    env = os.environ.get("NCU_EXE")
    if env:
        return env
    root = r"C:\Program Files\NVIDIA Corporation"
    cands = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            if name.startswith("Nsight Compute"):
                exe = os.path.join(root, name, "target",
                                   "windows-desktop-win7-x64", "ncu.exe")
                if os.path.isfile(exe):
                    cands.append((name, exe))
    if not cands:
        raise SystemExit("Could not find ncu.exe. Set NCU_EXE.")
    cands.sort(reverse=True)
    return cands[0][1]


def import_ncu_report(ncu_python_dir: str):
    if ncu_python_dir not in sys.path:
        sys.path.insert(0, ncu_python_dir)
    import ncu_report  # noqa: F401
    return ncu_report


def fnum(v) -> int:
    v = str(v).strip().replace(",", "")
    if v in ("", "n/a", "-"):
        return 0
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))
        except ValueError:
            return 0


def load_source_csv(csv_path: str):
    """Load an NCU source-page CSV into [(addr, sass, stalls)] rows.

    `stalls` is a dict of the stall_* column name -> integer sample count.
    Also returns (rows, stall_cols, kernel_name_rows).
    """
    rows = []
    stall_cols = []
    kernels = []
    with io.open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.reader(f)
        idx = None
        for line in rdr:
            if not line:
                continue
            if line[0] == "Kernel Name":
                kernels.append(line)
                continue
            if line[0] == "Address":
                idx = {h.strip(): i for i, h in enumerate(line)}
                stall_cols = [h for h in idx if h.startswith("stall_")]
                continue
            addr = int(line[0].strip().replace("0x", ""), 16)
            sass = line[1].strip()
            stalls = {c: fnum(line[idx[c]]) for c in stall_cols
                      if c in idx}
            rows.append((addr, sass, stalls))
    return rows, stall_cols, kernels


def short_file(file_name: str) -> str:
    return str(file_name).replace("\\", "/").rsplit("/", 1)[-1]


class ReportView:
    """Thin wrapper over an ncu_report action for source/SASS/metric lookups."""

    def __init__(self, report_path: str):
        import_ncu_report(find_ncu_python_dir())
        import ncu_report
        ctx = ncu_report.load_report(report_path)
        self.act = ctx.range_by_idx(0).action_by_idx(0)
        self.name = self._demangle(self.act)

    @staticmethod
    def _demangle(act):
        for attr in ("demangled_name", "name"):
            try:
                v = act.__getattribute__(attr)
                v = v() if callable(v) else v
                v = str(v)
                if v and v != "?":
                    return v
            except Exception:
                continue
        return "?"

    def source_line(self, pc: int):
        """Return (short_file, line) or None for a PC."""
        try:
            si = self.act.source_info(pc)
            fn = si.file_name()
            ln = si.line()
            if ln and int(str(ln)) == 0:
                return None
            return (short_file(fn), int(str(ln))) if (fn and ln) else None
        except Exception:
            return None

    def metric(self, name: str):
        try:
            m = self.act.metric_by_name(name)
            return m.value()
        except Exception:
            return None


def find_barrier_wait_sites(ordered_rows):
    """Return {wait_pc: bar_pc} for the classic grid-sync SASS layout.

    Layout: `BAR.SYNC <imm>; BRA.DIV 0x0` (16 bytes apart), with the
    barrier-wait samples landing on the instruction immediately after the
    pair. Newer ptxas layouts may place the wait directly inside the
    cooperative_groups sync (sync.h / cooperative_groups.h lines) — in that
    case call-site identification has to be done with inspect_hot.py.
    """
    pairs = {}
    for i in range(len(ordered_rows) - 1):
        a0, s0, _ = ordered_rows[i]
        a1, s1, _ = ordered_rows[i + 1]
        if (re.match(r"^\s*BAR\.SYNC", s0) and a1 == a0 + 16
                and re.match(r"^\s*BRA\.DIV", s1)
                and i + 2 < len(ordered_rows)):
            pairs[ordered_rows[i + 2][0]] = a0
    return pairs
