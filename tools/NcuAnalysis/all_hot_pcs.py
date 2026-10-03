"""Show contributing PCs of requested source buckets, separately per launch."""

import argparse

import ncu_common as C


def analyze(view, section, args, buckets):
    view.print_header()
    result = C.attribute_samples(view, section, args.focus)
    C.print_attribution(result, args.focus)
    if not result.rankable:
        return
    for name, line in buckets:
        rows = [(pc, counters[args.focus], sass) for pc, sass, counters in section.rows
                if result.pc_lines.get(pc) == (name, line)]
        rows.sort(key=lambda row: -row[1])
        print('\n== %s:%d: %d PCs, %d samples (%s) ==' %
              (name, line, len(rows), sum(row[1] for row in rows), result.scope))
        for pc, count, sass in rows[:args.limit]:
            print('  0x%X samples=%d %s' % (pc, count, sass))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    parser.add_argument('--csv')
    parser.add_argument('--buckets', help='comma-separated file:line buckets')
    parser.add_argument('--focus', default='stall_barrier')
    parser.add_argument('--limit', type=C.nonnegative_index, default=15)
    args = parser.parse_args()
    sources = C.source_runs(args)
    if sources:
        buckets = C.require_buckets(args.buckets, '--buckets')
        for view, section in sources:
            analyze(view, section, args, buckets)


if __name__ == '__main__':
    main()
