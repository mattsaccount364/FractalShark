"""Selection, provenance, and attribution tests; no NCU installation or GPU required."""

import argparse
import contextlib
import csv
import importlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import export_source_csv as E
import ncu_common as C


class Metric:
    def __init__(self, value):
        self.data = value

    def value(self):
        return self.data

    def unit(self):
        return ''


class SourceInfo:
    def file_name(self):
        return 'source/kernel.cuh'

    def line(self):
        return 12


class Action:
    NameBase_FUNCTION = 0
    NameBase_DEMANGLED = 1
    NameBase_MANGLED = 2
    WorkloadType_KERNEL = 0

    def __init__(self, function='foo', specialization='float', samples=8, kind=0):
        self.names = [function, 'void %s<%s>()' % (function, specialization),
                      '_Z%s_%s' % (function, specialization)]
        self.kind = kind
        self.metrics = {'gpu__time_duration.sum': 1000, 'launch__device_id': 0,
                        'launch__context_id': 1, 'launch__stream_id': 13,
                        'sass__inst_executed_per_opcode': 5,
                        'smsp__pcsamp_warps_issue_stalled_barrier': samples}
        for kind, values in [('block', [16, 8, 1]), ('grid', [50, 100, 1])]:
            for axis, value in zip('xyz', values):
                self.metrics['launch__%s_dim_%s' % (kind, axis)] = value
        self.sass = {16: 'MOV R0, R1', 32: 'EXIT'}
        self.source = {16: SourceInfo(), 32: SourceInfo()}

    def name(self, base):
        return self.names[base]

    def workload_type(self):
        return self.kind

    def metric_names(self):
        return self.metrics.keys()

    def metric_by_name(self, name):
        return Metric(self.metrics[name])

    def sass_by_pc(self, pc):
        return self.sass.get(pc, '')

    def source_info(self, pc):
        return self.source.get(pc)


class Range:
    def __init__(self, actions):
        self.actions = actions

    def num_actions(self):
        return len(self.actions)

    def action_by_idx(self, index):
        return self.actions[index]


class Context:
    def __init__(self, ranges):
        self.ranges = [Range(actions) for actions in ranges]

    def num_ranges(self):
        return len(self.ranges)

    def range_by_idx(self, index):
        return self.ranges[index]


def source_text(action, samples):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Kernel Name', action.names[1]])
    writer.writerow(['Address', 'Source', 'Instructions Executed',
                     'stall_barrier', 'stall_barrier (Not Issued)'])
    writer.writerow(['0x10', action.sass[16], '3', str(samples), '1'])
    writer.writerow(['0x20', action.sass[32], '2', '0', '0'])
    return output.getvalue()


def native_text(action):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Kernel Name', 'Context', 'Stream', 'Device',
                     'gpu__time_duration.sum', 'Grid Size', 'Block Size'])
    writer.writerow(['9', action.names[2], '1', '13', '0', '1,000',
                     '(50, 100, 1)', '(16, 8, 1)'])
    return output.getvalue()


class ToolingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='test-ncu-', dir=pathlib.Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = pathlib.Path(self.temporary.name)
        self.report_path = self.directory / 'capture.ncu-rep'
        self.report_path.write_bytes(b'fake report')

    def report(self, ranges):
        return C.Report(self.report_path, Context(ranges))

    def fixture(self, report, counts):
        csv_path = self.directory / 'source.csv'
        csv_path.write_text(''.join(source_text(run.act, count)
                                   for run, count in zip(report.runs, counts)), encoding='utf-8', newline='')
        manifest = {'schema_version': 1, 'report_sha256': C.file_sha256(self.report_path),
                    'csv_sha256': C.file_sha256(csv_path),
                    'runs': [{'section_index': index, 'range_index': run.range_index,
                              'action_index': run.action_index, 'mangled_name': run.names['mangled']}
                             for index, run in enumerate(report.runs)]}
        path = pathlib.Path(str(csv_path) + '.manifest.json')
        path.write_text(json.dumps(manifest), encoding='utf-8')
        return csv_path, path

    def test_default_and_name_selection_across_ranges(self):
        report = self.report([[Action(), Action('other')], [Action(), Action(specialization='double')]])
        self.assertEqual(len(report.select(None, 'function', None, None)), 4)
        self.assertEqual([run.key for run in report.select('foo', 'function', None, None)],
                         [(0, 0), (1, 0), (1, 1)])
        self.assertEqual([run.key for run in report.select('void foo<double>()', 'demangled', None, None)],
                         [(1, 1)])
        self.assertEqual([run.key for run in report.select('_Zfoo_double', 'mangled', None, None)], [(1, 1)])

    def test_action_compatibility_range_filter_and_intersection(self):
        report = self.report([[Action(), Action('other')], [Action()]])
        self.assertEqual(report.select(None, 'function', None, 1)[0].key, (0, 1))
        self.assertEqual(report.select(None, 'function', 1, 0)[0].key, (1, 0))
        self.assertEqual([run.key for run in report.select(None, 'function', 1, None)], [(1, 0)])
        with self.assertRaises(SystemExit):
            report.select('foo', 'function', 0, 1)

    def test_regex_and_invalid_selectors(self):
        report = self.report([[Action(), Action('other')]])
        self.assertEqual(len(report.select('regex:^foo$', 'function', None, None)), 1)
        for ri, ai in [(-1, None), (1, None), (None, -1), (None, 2)]:
            with self.subTest(range=ri, action=ai), self.assertRaises(SystemExit):
                report.select(None, 'function', ri, ai)
        with self.assertRaises(SystemExit):
            report.select('missing', 'function', None, None)
        with self.assertRaisesRegex(SystemExit, 'Invalid regular expression'):
            report.select('regex:[', 'function', None, None)
        with self.assertRaises(argparse.ArgumentTypeError):
            C.nonnegative_index('-1')

    def test_non_kernel_actions_keep_export_ordinals(self):
        report = self.report([[Action(kind=1), Action()], [Action()]])
        self.assertEqual([(run.key, run.export_index) for run in report.runs], [((0, 1), 1), ((1, 0), 2)])
        self.assertIs(report.runs[0].report.context, report.context)

    def test_repeated_names_and_pc_collisions_remain_separate(self):
        report = self.report([[Action(samples=3), Action(samples=7)]])
        csv_path, _ = self.fixture(report, [3, 7])
        loaded = C.SourceCsv(csv_path, report, report.runs)
        self.assertEqual(loaded.by_run[(0, 0)].total('stall_barrier'), 3)
        self.assertEqual(loaded.by_run[(0, 1)].total('stall_barrier'), 7)
        self.assertEqual(C.attribute_samples(report.runs[1], loaded.by_run[(0, 1)], 'stall_barrier').total, 7)

    def test_subset_selection_and_missing_exported_run(self):
        report = self.report([[Action(), Action('other')]])
        csv_path, _ = self.fixture(report, [3, 4])
        loaded = C.SourceCsv(csv_path, report, [report.runs[1]])
        self.assertEqual(loaded.by_run[(0, 1)].kernel_name, report.runs[1].name)
        manifest_report = self.report([[report.runs[0].act]])
        csv_path, _ = self.fixture(manifest_report, [3])
        with self.assertRaisesRegex(SystemExit, 'lacks selected'):
            C.SourceCsv(csv_path, report, report.runs)

    def test_stale_csv_and_report_fingerprints(self):
        report = self.report([[Action()]])
        csv_path, _ = self.fixture(report, [3])
        csv_path.write_text(csv_path.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(SystemExit, 'changed after export'):
            C.SourceCsv(csv_path, report, report.runs)
        csv_path, _ = self.fixture(report, [3])
        self.report_path.write_bytes(b'changed report')
        with self.assertRaisesRegex(SystemExit, 'different report'):
            C.SourceCsv(csv_path, report, report.runs)

    def test_invalid_manifest_and_duplicate_identities(self):
        report = self.report([[Action(), Action()]])
        csv_path, manifest_path = self.fixture(report, [3, 3])
        manifest = json.loads(manifest_path.read_text())
        manifest['schema_version'] = 2
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SystemExit, 'version'):
            C.SourceCsv(csv_path, report, report.runs)
        manifest['schema_version'] = 1
        manifest['runs'][1]['action_index'] = 0
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SystemExit, 'duplicate'):
            C.SourceCsv(csv_path, report, report.runs)
        manifest['runs'][1]['action_index'] = []
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SystemExit, 'range/action index'):
            C.SourceCsv(csv_path, report, report.runs)

    def test_legacy_single_section_requires_unique_run(self):
        action = Action()
        csv_path = self.directory / 'legacy.csv'
        csv_path.write_text(source_text(action, 3))
        report = self.report([[action]])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIn((0, 0), C.SourceCsv(csv_path, report, report.runs).by_run)
        repeated = self.report([[action, Action()]])
        with self.assertRaisesRegex(SystemExit, 'repeated launch'):
            C.SourceCsv(csv_path, repeated, [repeated.runs[0]])
        csv_path.write_text(source_text(action, 3) * 2)
        with self.assertRaisesRegex(SystemExit, 'ambiguous'):
            C.SourceCsv(csv_path, repeated, repeated.runs)

    def test_sass_name_and_instruction_count_mismatch(self):
        report = self.report([[Action()]])
        section = C.parse_source_csv(source_text(report.runs[0].act, 3))[0]
        section.kernel_name = 'wrong'
        with self.assertRaisesRegex(SystemExit, 'specialization'):
            C.verify_source_section(report.runs[0], section)
        section.kernel_name = report.runs[0].name
        report.runs[0].act.sass[16] = 'wrong instruction'
        with self.assertRaisesRegex(SystemExit, 'SASS mismatch'):
            C.verify_source_section(report.runs[0], section)
        report.runs[0].act.sass[16] = 'MOV R0, R1'
        report.runs[0].act.metrics['sass__inst_executed_per_opcode'] = 6
        with self.assertRaisesRegex(SystemExit, 'instruction total'):
            C.verify_source_section(report.runs[0], section)

    def test_malformed_rows_headers_and_duplicate_pcs(self):
        text = source_text(Action(), 3)
        for malformed in ['', text.replace('0x20', '0x10'),
                          text.replace('0x20', 'not-an-address'),
                          text.replace('Address,Source', 'Address,Address'),
                          text + 'garbage\n']:
            with self.subTest(text=malformed[:50]), self.assertRaises(SystemExit):
                C.parse_source_csv(malformed)
        self.assertEqual(len(C.parse_source_csv('==WARNING== fallback\n' + text)), 1)

    def test_partial_coverage_does_not_claim_full_run(self):
        report = self.report([[Action(samples=8)]])
        section = C.parse_source_csv(source_text(report.runs[0].act, 3))[0]
        result = C.attribute_samples(report.runs[0], section, 'stall_barrier')
        self.assertEqual((result.total, result.canonical, result.unattributed, result.residual), (3, 8, 0, 5))
        self.assertTrue(result.rankable)
        self.assertEqual(result.scope, 'CSV subset only')
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            C.print_attribution(result, 'stall_barrier')
        self.assertIn('residual: 5', captured.getvalue())

    def test_unmapped_or_excess_samples_suppress_rankings(self):
        report = self.report([[Action(samples=8)]])
        section = C.parse_source_csv(source_text(report.runs[0].act, 3))[0]
        report.runs[0].act.source.clear()
        result = C.attribute_samples(report.runs[0], section, 'stall_barrier')
        self.assertEqual(result.unattributed, 3)
        self.assertFalse(result.rankable)
        report.runs[0].act.source[16] = SourceInfo()
        report.runs[0].act.metrics['smsp__pcsamp_warps_issue_stalled_barrier'] = 2
        self.assertFalse(C.attribute_samples(report.runs[0], section, 'stall_barrier').rankable)

    def test_missing_column_metric_and_zero_counts(self):
        report = self.report([[Action(samples=0)]])
        section = C.parse_source_csv(source_text(report.runs[0].act, 0))[0]
        with self.assertRaisesRegex(SystemExit, 'unavailable'):
            C.attribute_samples(report.runs[0], section, 'stall_wait')
        result = C.attribute_samples(report.runs[0], section, 'stall_barrier')
        self.assertEqual(result.scope, 'full-run coverage')
        self.assertTrue(result.rankable)
        report.runs[0].metric_names.remove('smsp__pcsamp_warps_issue_stalled_barrier')
        result = C.attribute_samples(report.runs[0], section, 'stall_barrier')
        self.assertIsNone(result.canonical)
        self.assertEqual(result.scope, 'CSV subset only')
        self.assertEqual(C.resolve_metric(report.runs[0], '^launch__device_id$'), 'launch__device_id')
        self.assertIsNone(C.resolve_metric(report.runs[0], '^missing$'))

    def test_aliases_and_not_issued_names(self):
        self.assertEqual(C.canonical_stall_name('stall_long_sb'),
                         'smsp__pcsamp_warps_issue_stalled_long_scoreboard')
        self.assertEqual(C.canonical_stall_name('stall_math (Not Issued)'),
                         'smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued')
        report = self.report([[Action()]])
        report.runs[0].act.metrics['warpsampling:smsp__pcsamp_warps_issue_stalled_barrier'] = 10000
        # Refresh available names after modifying the fake action.
        report.runs[0].metric_names = set(report.runs[0].act.metric_names())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            importlib.import_module('stall_breakdown').analyze(report.runs[0])
        self.assertNotIn('10000', output.getvalue())
        self.assertNotIn('grand total', output.getvalue())

    def test_native_launch_validation(self):
        report = self.report([[Action()]])
        view = report.runs[0]
        text = native_text(view.act)
        self.assertEqual(E.verify_native_launch(text, view), '9')
        view.act.metrics['gpu__time_duration.sum'] = 1000.5
        self.assertEqual(E.verify_native_launch(text.replace('1,000', '1000.5'), view), '9')
        view.act.metrics['gpu__time_duration.sum'] = 1000
        for invalid in [text.replace('1,000', '2,000'), text.replace('16, 8, 1', '8, 8, 1'),
                        text.replace(view.names['mangled'], '_wrong'), text + text.splitlines()[-1] + '\n']:
            with self.assertRaises(SystemExit):
                E.verify_native_launch(invalid, view)

    def test_export_publication_and_repeated_sections(self):
        report = self.report([[Action(samples=3), Action(samples=7)]])
        responses = []
        for run, count in zip(report.runs, [3, 7]):
            responses.extend([native_text(run.act), source_text(run.act, count)])
        output = self.directory / 'export.csv'
        with mock.patch.object(E, 'run_ncu', side_effect=responses) as native:
            with contextlib.redirect_stdout(io.StringIO()):
                E.export_runs(report.runs, output, 'ncu.exe')
        self.assertEqual(native.call_count, 4)
        loaded = C.SourceCsv(output, report, report.runs)
        self.assertEqual([loaded.by_run[run.key].total('stall_barrier') for run in report.runs], [3, 7])
        self.assertIn('--config-file', native.call_args_list[0].args[0])

    def test_export_failure_preserves_existing_pair(self):
        report = self.report([[Action()]])
        output, manifest = self.fixture(report, [3])
        original = output.read_bytes(), manifest.read_bytes()
        with mock.patch.object(E, 'run_ncu', return_value=native_text(Action('wrong'))):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                E.export_runs(report.runs, output, 'ncu.exe')
        self.assertEqual((output.read_bytes(), manifest.read_bytes()), original)

    def test_list_runs_needs_no_source_or_bucket_arguments_in_any_helper(self):
        report = self.report([[Action()]])
        modules = ['metric_probe', 'pipe_analysis', 'memory_analysis', 'stall_breakdown',
                   'join_attribution', 'inspect_hot', 'all_hot_pcs', 'resolve_barsync', 'export_source_csv']
        for module in modules:
            with self.subTest(module=module), mock.patch.object(C, 'Report', return_value=report):
                with mock.patch('sys.argv', [module, '--report', 'fake', '--list-runs']):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        importlib.import_module(module).main()
                self.assertIn('Selected kernel launches: 1', output.getvalue())


if __name__ == '__main__':
    unittest.main()
