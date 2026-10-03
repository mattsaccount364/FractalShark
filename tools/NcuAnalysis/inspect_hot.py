"""Show SASS windows around contributing PCs of source buckets, per launch."""

import argparse

import ncu_common as C


def analyze(view, section, args, buckets):
    view.print_header()
    result = C.attribute_samples(view, section, args.focus)
    C.print_attribution(result, args.focus)
    if not result.rankable:
        return
    lookup = {pc: sass for pc, sass, _ in section.rows}
    for name, line in buckets:
        rows = [(pc, counters[args.focus]) for pc, _, counters in section.rows
                if result.pc_lines.get(pc) == (name, line)]
        rows.sort(key=lambda row: -row[1])
        print('\n== %s:%d (%s): %d contributing PCs ==' % (name, line, result.scope, len(rows)))
        for pc, count in rows[:args.top_pcs]:
            print('  PC=0x%X samples=%d' % (pc, count))
            for offset in range(-args.back, args.fwd + 1):
                address = pc + 16 * offset
                source = view.source_line(address) if address in lookup else None
                tag = '%s:%d' % source if source else '--'
                print('    %+d 0x%X %-48s %s' %
                      (offset, address, lookup.get(address, '--'), tag))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    parser.add_argument('--csv')
    parser.add_argument('--buckets', help='comma-separated file:line buckets')
    parser.add_argument('--focus', default='stall_barrier')
    parser.add_argument('--top-pcs', type=C.nonnegative_index, default=2)
    parser.add_argument('--back', type=C.nonnegative_index, default=6)
    parser.add_argument('--fwd', type=C.nonnegative_index, default=3)
    args = parser.parse_args()
    sources = C.source_runs(args)
    if sources:
        buckets = C.require_buckets(args.buckets, '--buckets')
        for view, section in sources:
            analyze(view, section, args, buckets)


if __name__ == '__main__':
    main()
