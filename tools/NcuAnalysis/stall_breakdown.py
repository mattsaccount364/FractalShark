"""Report canonical scheduler states separately for every selected kernel launch."""

import argparse

import ncu_common as C


def analyze(view):
    view.print_header()
    base = {}
    not_issued = {}
    prefix = 'smsp__pcsamp_warps_issue_stalled_'
    for name in sorted(view.metric_names):
        if not name.startswith(prefix):
            continue
        value = view.metric(name)
        if value is None:
            continue
        target = not_issued if name.endswith('_not_issued') or name.endswith('.not_issued') else base
        target[name] = value
    for title, values in [('All-samples scheduler states', base), ('Not Issued subsets', not_issued)]:
        total = sum(values.values())
        print('\n== %s; total %d ==' % (title, total))
        if not values:
            print('  (canonical metrics unavailable)')
        for name, count in sorted(values.items(), key=lambda item: -item[1]):
            print('%12d %6.2f%% %s' % (count, 100 * count / max(total, 1), name))
    print('Not Issued counts are subsets and are not added to the base budget.')
    print('selected/not_selected are scheduler states, not true stalls; sample shares are not elapsed-time shares.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    args = parser.parse_args()
    for view in C.selected_runs(args):
        analyze(view)


if __name__ == '__main__':
    main()
