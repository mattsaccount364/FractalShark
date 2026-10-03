"""Resolve source-bucket barrier samples to nearby SASS and caller context, per launch."""

import argparse

import ncu_common as C


def analyze(view, section, args, bucket):
    view.print_header()
    result = C.attribute_samples(view, section, 'stall_barrier')
    C.print_attribution(result, 'stall_barrier')
    if not result.rankable:
        return
    ordered = sorted(section.rows)
    contributing = [(index, pc, counters['stall_barrier'])
                    for index, (pc, _, counters) in enumerate(ordered)
                    if result.pc_lines.get(pc) == bucket]
    contributing.sort(key=lambda row: -row[2])
    total = sum(row[2] for row in contributing)
    print('\nBucket %s:%d: %d samples, %d PCs (%s)' %
          (*bucket, total, len(contributing), result.scope))
    for index, pc, count in contributing[:args.top]:
        caller = None
        for previous in range(index, max(index - 400, -1), -1):
            source = view.act.source_info(ordered[previous][0])
            if source is not None:
                path = source.file_name().replace('\\', '/')
                if 'cooperative_groups' not in path and C.short_file(path) != 'sync.h':
                    if path.endswith(('.cu', '.cuh', '.h', '.inl')) and source.line():
                        caller = (path, source.line())
                        break
        print('  PC=0x%X samples=%d nearest non-barrier source=%s' % (pc, count, caller))
        for near in range(max(0, index - args.radius), min(len(ordered), index + args.radius + 1)):
            address, sass, _ = ordered[near]
            print('    %s 0x%X %s' % ('>' if address == pc else ' ', address, sass))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    parser.add_argument('--csv')
    parser.add_argument('--bucket', help='file:line bucket')
    parser.add_argument('--top', type=C.nonnegative_index, default=5)
    parser.add_argument('--radius', type=C.nonnegative_index, default=3)
    args = parser.parse_args()
    sources = C.source_runs(args)
    if sources:
        buckets = C.require_buckets(args.bucket, '--bucket')
        if len(buckets) != 1:
            raise SystemExit('--bucket requires exactly one file:line')
        for view, section in sources:
            analyze(view, section, args, buckets[0])


if __name__ == '__main__':
    main()
