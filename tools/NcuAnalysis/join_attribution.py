"""Attribute a sample family per selected launch, with explicit coverage accounting."""

import argparse
import collections

import ncu_common as C


def analyze(view, section, args):
    view.print_header()
    print('SASS instruction rows:', len(section.rows))
    if 'Instructions Executed' in section.columns:
        print('Source-export executed warp instructions:', section.total('Instructions Executed'))
    result = C.attribute_samples(view, section, args.focus)
    C.print_attribution(result, args.focus)
    base_columns = [name for name in section.stall_columns
                    if '(Not Issued)' not in name and not name.endswith('_not_issued')]
    budget = sum(section.total(name) for name in base_columns)
    print('CSV base scheduler-state sample budget:', budget)
    if not result.rankable:
        return
    print('\nTop %d lines by %s samples (%s):' % (args.top, args.focus, result.scope))
    for (name, line), count in result.per_line.most_common(args.top):
        print('%12d %6.2f%% %s:%d' % (count, 100 * count / max(result.total, 1), name, line))
    per_file = collections.Counter()
    for (name, _), count in result.per_line.items():
        per_file[name] += count
    print('Per-file totals (%s):' % result.scope, per_file.most_common())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    parser.add_argument('--csv', help='source CSV exported with matching manifest')
    parser.add_argument('--top', type=C.nonnegative_index, default=30)
    parser.add_argument('--focus', default='stall_barrier')
    args = parser.parse_args()
    for view, section in C.source_runs(args):
        analyze(view, section, args)


if __name__ == '__main__':
    main()
