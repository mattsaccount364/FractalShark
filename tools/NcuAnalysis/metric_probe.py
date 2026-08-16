r"""Discover the metrics an NCU report actually exposes.

NCU renames or splits metrics between releases, so other scripts in this
folder select candidates by regex instead of hardcoding names. This tool is
the fast way to see which candidates exist for a given capture:

Usage:
    py -3 tools\NcuAnalysis\metric_probe.py --report <rep> --pattern 'pipe|issue'
    py -3 tools\NcuAnalysis\metric_probe.py --report <rep> --pattern 'conflict' --values
    py -3 tools\NcuAnalysis\metric_probe.py --report <rep> --pattern '.*' --action 2 --values

Prints metric names (and value + unit with --values) for the selected action.
Action 0 is the first kernel instance; use --action to compare instances.
"""

import argparse
import re

import ncu_common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--pattern", default=r".*",
                    help="case-insensitive regex to filter metric names")
    ap.add_argument("--action", type=int, default=0,
                    help="kernel-instance/action index (report may hold several)")
    ap.add_argument("--values", action="store_true",
                    help="print value + unit for each metric")
    args = ap.parse_args()

    C.import_ncu_report(C.find_ncu_python_dir())
    import ncu_report

    ctx = ncu_report.load_report(args.report)
    rng = ctx.range_by_idx(0)
    count = rng.num_actions()
    if args.action >= count:
        raise SystemExit("report has %d actions; --action out of range" % count)
    act = rng.action_by_idx(args.action)

    names = list(act.metric_names() or [])
    pat = re.compile(args.pattern, re.IGNORECASE)
    shown = [n for n in names if pat.search(n)]

    if args.values:
        for n in sorted(shown):
            try:
                m = act.metric_by_name(n)
                v = m.value()
                unit = ""
                try:
                    unit = (m.unit() or "").strip()
                except Exception:
                    pass
                print(("%s = %s %s" % (n, v, unit)).rstrip())
            except Exception as e:
                print("%s = <error: %s>" % (n, e))
    else:
        for n in sorted(shown):
            print(n)

    print("-- actions in report: %d | total metrics: %d | matched: %d"
          % (count, len(names), len(shown)))


if __name__ == "__main__":
    main()
