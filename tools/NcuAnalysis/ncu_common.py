"""Shared helpers for the NCU attribution scripts in this folder.

Each script locates the ncu_report package and ncu.exe automatically; override
with NCU_PYTHON_DIR / NCU_EXE. Report, source, and sample data stay scoped to
explicit range/action identities, even when names or instruction PCs repeat.
"""

import collections
import csv
import dataclasses
import hashlib
import io
import json
import os
import pathlib
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
    return int(float(v)) if '.' in v or 'e' in v.lower() else int(v)


def file_sha256(path):
    with open(path, 'rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def nonnegative_index(value):
    import argparse
    index = int(value)
    if index < 0:
        raise argparse.ArgumentTypeError('run indices must be nonnegative')
    return index


def compile_regex(expression):
    # Translate a user-input error into the command-line error contract.
    try:
        return re.compile(expression)
    except re.error as error:
        raise SystemExit('Invalid regular expression: %s' % error) from error


def add_selection_arguments(parser):
    parser.add_argument('--report', required=True)
    parser.add_argument('--list-runs', action='store_true', help='list selected launches without analysis')
    parser.add_argument('--kernel-name', help='exact kernel name, or regex:<expression>')
    parser.add_argument('--kernel-name-base', choices=['function', 'demangled', 'mangled'],
                        default='function')
    parser.add_argument('--range', type=nonnegative_index, help='report range index')
    parser.add_argument('--action', type=nonnegative_index,
                        help='action within --range (range 0 when omitted)')


class Report:
    """Own the report context for the lifetime of all run views."""

    def __init__(self, path, context=None):
        self.path = pathlib.Path(path)
        if context is None:
            api = import_ncu_report(find_ncu_python_dir())
            context = api.load_report(str(self.path))
        self.context = context
        self.runs = []
        self.range_counts = []
        ordinal = 0
        for ri in range(context.num_ranges()):
            capture_range = context.range_by_idx(ri)
            self.range_counts.append(capture_range.num_actions())
            for ai in range(capture_range.num_actions()):
                action = capture_range.action_by_idx(ai)
                if action.workload_type() == action.WorkloadType_KERNEL:
                    self.runs.append(ReportView(self, action, ri, ai, ordinal))
                ordinal += 1

    def select(self, kernel_name, name_base, range_index, action_index):
        effective_range = 0 if action_index is not None and range_index is None else range_index
        if effective_range is not None:
            if effective_range < 0 or effective_range >= len(self.range_counts):
                raise SystemExit('--range out of bounds; use --list-runs')
        if action_index is not None:
            if action_index < 0 or action_index >= self.range_counts[effective_range]:
                raise SystemExit('--action out of bounds; use --list-runs')
        pattern = None
        if kernel_name is not None and kernel_name.startswith('regex:'):
            expression = kernel_name[len('regex:'):]
            pattern = compile_regex(expression)
        selected = []
        for run in self.runs:
            if effective_range is not None and run.range_index != effective_range:
                continue
            if action_index is not None and run.action_index != action_index:
                continue
            name = run.names[name_base]
            if kernel_name is not None:
                if pattern is not None:
                    if pattern.search(name) is None:
                        continue
                elif kernel_name != name:
                    continue
            selected.append(run)
        if not selected:
            raise SystemExit('No kernel launches match the selectors; use --list-runs')
        return selected


def short_file(file_name: str) -> str:
    return str(file_name).replace("\\", "/").rsplit("/", 1)[-1]


class ReportView:
    """One explicit launch; PC mappings never cross run boundaries."""

    def __init__(self, report, action, range_index, action_index, export_index):
        self.report = report
        self.act = action
        self.range_index = range_index
        self.action_index = action_index
        self.export_index = export_index
        self.names = {
            'function': action.name(action.NameBase_FUNCTION),
            'demangled': action.name(action.NameBase_DEMANGLED),
            'mangled': action.name(action.NameBase_MANGLED),
        }
        self.name = self.names['demangled']
        self.metric_names = set(action.metric_names() or [])

    @property
    def key(self):
        return self.range_index, self.action_index

    def source_line(self, pc: int):
        """Return (short_file, line) or None for a PC."""
        si = self.act.source_info(pc)
        if si is None:
            return None
        fn, ln = si.file_name(), si.line()
        return (short_file(fn), int(ln)) if fn and ln else None

    def metric(self, name: str):
        return self.act.metric_by_name(name).value() if name in self.metric_names else None

    def dimensions(self, kind):
        values = tuple(self.metric('launch__%s_dim_%s' % (kind, axis)) for axis in 'xyz')
        return values if all(value is not None for value in values) else None

    def print_header(self):
        print('\n=== Run range=%d action=%d ===' % self.key)
        print(self.name)
        for label, metric in [('device', 'launch__device_id'), ('context', 'launch__context_id'),
                              ('stream', 'launch__stream_id'), ('duration_ns', 'gpu__time_duration.sum'),
                              ('registers/thread', 'launch__registers_per_thread')]:
            value = self.metric(metric)
            print('  %s = %s' % (label, 'unavailable' if value is None else value))
        print('  grid = %s; block = %s' % (self.dimensions('grid'), self.dimensions('block')))


def print_run_summary(runs, full_names):
    print('Selected kernel launches:', len(runs))
    print('Range Action Device Context Stream Duration(ms) Grid Block Function')
    durations = []
    for run in runs:
        duration = run.metric('gpu__time_duration.sum')
        if duration is not None:
            durations.append(duration)
        duration_text = 'unavailable' if duration is None else '%.6f' % (duration / 1e6)
        meta = [run.metric('launch__%s_id' % field) for field in ['device', 'context', 'stream']]
        print('%d %d %s %s %s %s %s %s %s' %
              (run.range_index, run.action_index, *meta, duration_text,
               run.dimensions('grid'), run.dimensions('block'), run.names['function']))
        if full_names:
            print('  specialization:', run.name)
    print('Cumulative kernel duration: %.6f ms (%d/%d durations available; not application elapsed time)'
          % (sum(durations) / 1e6, len(durations), len(runs)))


def selected_runs(args):
    report = Report(args.report)
    runs = report.select(args.kernel_name, args.kernel_name_base, args.range, args.action)
    print_run_summary(runs, args.list_runs)
    return [] if args.list_runs else runs


def resolve_metric(view, regex):
    pattern = re.compile(regex)
    return next((name for name in sorted(view.metric_names)
                 if pattern.match(name) and view.metric(name) is not None), None)


def print_metric_groups(view, groups):
    view.print_header()
    for title, items in groups:
        print('\n== %s ==' % title)
        found = []
        for label, expression in items:
            name = resolve_metric(view, expression)
            if name is not None:
                metric = view.act.metric_by_name(name)
                found.append((label, metric.value(), metric.unit() or '', name))
        if not found:
            print('  (metrics unavailable in this capture)')
        for label, value, unit, name in sorted(found, key=lambda row: abs(row[1]), reverse=True):
            print('  %-48s %s %s [%s]' % (label, value, unit, name))


@dataclasses.dataclass
class SourceSection:
    kernel_name: str
    columns: list = dataclasses.field(default_factory=list)
    records: list = dataclasses.field(default_factory=list)
    rows: list = dataclasses.field(default_factory=list)

    @property
    def stall_columns(self):
        return [column for column in self.columns if column.startswith('stall_')]

    def total(self, column):
        return sum(row[2].get(column, 0) for row in self.rows)


def parse_source_csv(text):
    """Retain native section boundaries, including repeated identical kernel names."""
    sections = []
    current = None
    addresses = set()
    for line in csv.reader(io.StringIO(text.lstrip('\ufeff'))):
        if not line:
            continue
        if line[0] == 'Kernel Name':
            if len(line) < 2:
                raise SystemExit('Malformed Kernel Name header in source CSV')
            current = SourceSection(line[1])
            sections.append(current)
            addresses = set()
            continue
        if current is None:
            continue  # NCU may emit a deployment-warning preamble before the CSV.
        if line[0] == 'Address':
            if current.columns or 'Source' not in line or len(set(line)) != len(line):
                raise SystemExit('Malformed or duplicate Address header in source CSV')
            current.columns = line
            continue
        if not current.columns or len(line) != len(current.columns):
            raise SystemExit('Malformed row in source CSV; re-export with export_source_csv.ps1')
        if re.fullmatch(r'(?:0x)?[0-9a-fA-F]+', line[0].strip()) is None:
            raise SystemExit('Invalid instruction address in source CSV: ' + line[0])
        pc = int(line[0], 16)
        if pc in addresses:
            raise SystemExit('Duplicate PC within one source section: 0x%X' % pc)
        addresses.add(pc)
        values = dict(zip(current.columns, line))
        counters = {name: fnum(value) for name, value in values.items()
                    if name.startswith('stall_') or name in
                    ['Instructions Executed', 'Thread Instructions Executed',
                     'L2 Theoretical Sectors Global', 'L2 Theoretical Sectors Global Excessive']}
        current.records.append(line)
        current.rows.append((pc, values['Source'].strip(), counters))
    if not sections or any(not section.rows for section in sections):
        raise SystemExit('Source CSV contains no SASS or an empty section; re-export with SASS enabled')
    return sections


def verify_source_section(view, section):
    if section.kernel_name != view.name:
        raise SystemExit('CSV kernel specialization does not match range %d/action %d' % view.key)
    for pc, sass, _ in section.rows:
        captured = view.act.sass_by_pc(pc)
        if not captured or ' '.join(captured.split()) != ' '.join(sass.split()):
            raise SystemExit('CSV SASS mismatch at 0x%X; re-export from the selected report' % pc)
    expected = view.metric('sass__inst_executed_per_opcode')
    if expected is not None and 'Instructions Executed' in section.columns:
        if section.total('Instructions Executed') != expected:
            raise SystemExit('CSV executed-instruction total differs from selected launch; re-export')


class SourceCsv:
    """Bind section indices to report-local run identities using a fingerprinted sidecar."""

    def __init__(self, path, report, selected):
        path = pathlib.Path(path)
        self.sections = parse_source_csv(path.read_text(encoding='utf-8-sig'))
        manifest_path = pathlib.Path(str(path) + '.manifest.json')
        self.by_run = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('schema_version') != 1:
                raise SystemExit('Unsupported source manifest version; re-export')
            if manifest.get('report_sha256') != file_sha256(report.path):
                raise SystemExit('Source manifest belongs to a different report; re-export')
            if manifest.get('csv_sha256') != file_sha256(path):
                raise SystemExit('Source CSV changed after export; re-export')
            entries = manifest.get('runs')
            if not isinstance(entries, list) or len(entries) != len(self.sections):
                raise SystemExit('Source manifest section count mismatch; re-export')
            catalog = {run.key: run for run in report.runs}
            for index, (entry, section) in enumerate(zip(entries, self.sections)):
                if not isinstance(entry, dict):
                    raise SystemExit('Malformed source manifest run; re-export')
                key = (entry.get('range_index'), entry.get('action_index'))
                if any(type(value) is not int or value < 0 for value in key):
                    raise SystemExit('Invalid source manifest range/action index; re-export')
                view = catalog.get(key)
                if entry.get('section_index') != index or view is None or key in self.by_run:
                    raise SystemExit('Invalid or duplicate source manifest run identity; re-export')
                if entry.get('mangled_name') != view.names['mangled']:
                    raise SystemExit('Source manifest kernel identity mismatch; re-export')
                verify_source_section(view, section)
                self.by_run[key] = section
        else:
            if len(self.sections) != 1 or len(selected) != 1:
                raise SystemExit('Legacy multi-run CSV is ambiguous; re-export to create a manifest')
            view = selected[0]
            matches = [run for run in report.runs if run.name == self.sections[0].kernel_name]
            if len(matches) != 1 or matches[0].key != view.key:
                raise SystemExit('Legacy CSV cannot identify a repeated launch; re-export with a manifest')
            verify_source_section(view, self.sections[0])
            self.by_run[view.key] = self.sections[0]
            print('Legacy single-section CSV: verified unique specialization and SASS; no fingerprint manifest')
        for run in selected:
            if run.key not in self.by_run:
                raise SystemExit('CSV lacks selected range %d/action %d; re-export those runs' % run.key)


def source_runs(args):
    runs = selected_runs(args)
    if not runs:
        return []
    if not args.csv:
        raise SystemExit('--csv is required for source analysis (not for --list-runs)')
    source = SourceCsv(args.csv, runs[0].report, runs)
    return [(run, source.by_run[run.key]) for run in runs]


_STALL_SUFFIXES = {
    'long_sb': 'long_scoreboard', 'math': 'math_pipe_throttle', 'lg': 'lg_throttle',
    'mio': 'mio_throttle', 'tex': 'tex_throttle', 'no_inst': 'no_instructions',
    'sleep': 'sleeping', 'dispatch': 'dispatch_stall',
}


def canonical_stall_name(column):
    suffix = column.removeprefix('stall_')
    not_issued = suffix.endswith(' (Not Issued)') or suffix.endswith('_not_issued')
    suffix = suffix.removesuffix(' (Not Issued)').removesuffix('_not_issued')
    suffix = _STALL_SUFFIXES.get(suffix, suffix)
    return 'smsp__pcsamp_warps_issue_stalled_' + suffix + ('_not_issued' if not_issued else '')


@dataclasses.dataclass
class Attribution:
    total: int
    canonical: object
    unattributed: int
    per_line: collections.Counter
    pc_lines: dict

    @property
    def residual(self):
        return None if self.canonical is None else self.canonical - self.total

    @property
    def rankable(self):
        return self.unattributed == 0 and (self.residual is None or self.residual >= 0)

    @property
    def scope(self):
        return 'full-run coverage' if self.canonical == self.total else 'CSV subset only'


def attribute_samples(view, section, focus):
    if focus not in section.stall_columns:
        raise SystemExit('Requested column %r is unavailable; columns: %s' %
                         (focus, ', '.join(section.stall_columns)))
    per_line = collections.Counter()
    pc_lines = {}
    unattributed = 0
    pairs = find_barrier_wait_sites(sorted(section.rows)) if focus.startswith('stall_barrier') else {}
    for pc, _, counters in section.rows:
        count = counters[focus]
        if not count:
            continue
        line = view.source_line(pc)
        if line is None and pc in pairs:
            line = view.source_line(pairs[pc])
        if line is None:
            unattributed += count
        else:
            pc_lines[pc] = line
            per_line[line] += count
    total = section.total(focus)
    assert sum(per_line.values()) + unattributed == total
    return Attribution(total, view.metric(canonical_stall_name(focus)),
                       unattributed, per_line, pc_lines)


def print_attribution(result, focus):
    print('Sample family:', focus)
    print('  canonical action total:', 'unavailable' if result.canonical is None else result.canonical)
    print('  CSV total:', result.total)
    print('  attributed within CSV:', sum(result.per_line.values()))
    print('  unattributed within CSV:', result.unattributed)
    print('  canonical minus CSV residual:',
          'unavailable' if result.residual is None else result.residual)
    print('  attribution scope:', result.scope)
    if not result.rankable:
        print('  Rankings suppressed: incomplete CSV join or CSV exceeds canonical count.')


def require_buckets(value, option):
    if not value:
        raise SystemExit(option + ' is required for bucket analysis (not for --list-runs)')
    return [(short_file(name.strip()), int(line))
            for name, line in (bucket.strip().rsplit(':', 1) for bucket in value.split(','))]


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
