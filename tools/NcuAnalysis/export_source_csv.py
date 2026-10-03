"""Export separately verified launches into a source CSV and fingerprint manifest."""

import argparse
import csv
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import ncu_common as C


def run_ncu(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, encoding='utf-8')
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode:
        raise SystemExit('NCU exited with code %d: %s' % (result.returncode, result.stdout[:2000]))
    return result.stdout


def verify_native_launch(text, view):
    """Cross-check CLI import ordinal against the API's range/action catalog."""
    header = None
    launches = []
    for row in csv.reader(io.StringIO(text)):
        if row and row[0] == 'ID':
            header = row
        elif header and row and row[0].isdigit():
            launches.append(dict(zip(header, row)))
    if len(launches) != 1:
        raise SystemExit('Native import did not select exactly one launch; export aborted')
    launch = launches[0]
    if launch.get('Kernel Name') != view.names['mangled']:
        raise SystemExit('Native import ordinal/kernel mismatch; export aborted')
    for label, metric in [('Context', 'launch__context_id'), ('Stream', 'launch__stream_id'),
                          ('Device', 'launch__device_id'), ('gpu__time_duration.sum', 'gpu__time_duration.sum')]:
        expected = view.metric(metric)
        if expected is not None:
            actual = None if label not in launch else float(launch[label].replace(',', ''))
            if actual != expected:
                raise SystemExit('Native import %s mismatch; export aborted' % label)
    for label, kind in [('Grid Size', 'grid'), ('Block Size', 'block')]:
        expected = view.dimensions(kind)
        actual = tuple(int(value) for value in re.findall(r'\d+', launch.get(label, '')))
        if expected is not None and actual != expected:
            raise SystemExit('Native import %s mismatch; export aborted' % label)
    return launch['ID']


def export_runs(runs, output, ncu):
    report = runs[0].report
    report_hash = C.file_sha256(report.path)
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer)
    entries = []
    for index, view in enumerate(runs):
        print('[export] range=%d action=%d: %s' % (*view.key, view.names['function']), flush=True)
        base = [ncu, '--config-file', 'off', '--import', str(report.path),
                '--launch-skip', str(view.export_index), '--launch-count', '1',
                '--rename-kernels', 'off', '--csv', '--print-units', 'base']
        metadata = run_ncu(base + ['--page', 'raw', '--print-kernel-base', 'mangled',
                                   '--metrics', 'gpu__time_duration.sum'])
        native_id = verify_native_launch(metadata, view)
        source = run_ncu(base + ['--page', 'source', '--print-source', 'sass'])
        sections = C.parse_source_csv(source)
        if len(sections) != 1:
            raise SystemExit('Native source export did not contain exactly one section; export aborted')
        section = sections[0]
        C.verify_source_section(view, section)
        writer.writerow(['Kernel Name', section.kernel_name])
        writer.writerow(section.columns)
        writer.writerows(section.records)
        entries.append({'section_index': index, 'range_index': view.range_index,
                        'action_index': view.action_index, 'mangled_name': view.names['mangled'],
                        'native_import_id': native_id})
    if C.file_sha256(report.path) != report_hash:
        raise SystemExit('Report changed during export; export aborted')
    output = pathlib.Path(output)
    manifest_path = pathlib.Path(str(output) + '.manifest.json')
    if output.resolve() == report.path.resolve() or manifest_path.resolve() == report.path.resolve():
        raise SystemExit('Output must not overwrite the input report')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ncu-export-', dir=output.parent) as temporary:
        csv_path = pathlib.Path(temporary) / 'source.csv'
        csv_path.write_text(buffer.getvalue(), encoding='utf-8', newline='')
        manifest = {'schema_version': 1, 'report_sha256': report_hash,
                    'csv_sha256': C.file_sha256(csv_path), 'runs': entries}
        json_path = pathlib.Path(temporary) / 'manifest.json'
        json_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        os.replace(csv_path, output)
        os.replace(json_path, manifest_path)
    print('[export] CSV:', output)
    print('[export] Manifest:', manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    parser.add_argument('--output-csv', help='CSV path; otherwise NCU_OUT_CSV or the system temp directory')
    args = parser.parse_args()
    runs = C.selected_runs(args)
    if not runs:
        return
    output = args.output_csv or os.environ.get('NCU_OUT_CSV') or str(pathlib.Path(tempfile.gettempdir()) / 'ncu_source.csv')
    export_runs(runs, output, C.find_ncu_exe())


if __name__ == '__main__':
    main()
