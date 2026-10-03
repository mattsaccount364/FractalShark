"""List available metrics and optionally values for each selected launch."""

import argparse

import ncu_common as C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    parser.add_argument('--pattern', default='.*', help='case-insensitive metric-name regex')
    parser.add_argument('--values', action='store_true')
    args = parser.parse_args()
    pattern = C.compile_regex('(?i)' + args.pattern)
    for view in C.selected_runs(args):
        view.print_header()
        shown = sorted(name for name in view.metric_names if pattern.search(name))
        for name in shown:
            if args.values:
                metric = view.act.metric_by_name(name)
                print(('%s = %s %s' % (name, metric.value(), metric.unit() or '')).rstrip())
            else:
                print(name)
        print('-- total metrics: %d; matched: %d' % (len(view.metric_names), len(shown)))


if __name__ == '__main__':
    main()
