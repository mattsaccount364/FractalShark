r"""Enumerate every contributing PC of a hot bucket, with SASS + source line.

Useful for hand-joining a bucket to a specific call site when several
distinct call sites collapse onto one library line (sync.h,
cooperative_groups.h).

Usage:
    py -3 tools\NcuAnalysis\all_hot_pcs.py --report <rep> --csv <csv> \
        --buckets "sync.h:109" [--limit 25]
"""

import argparse

import ncu_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--buckets", required=True,
                    help='comma-separated "file:line" buckets')
    ap.add_argument("--focus", default="stall_barrier")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()

    rows, _, _ = C.load_source_csv(args.csv)
    view = C.ReportView(args.report)

    contrib = {}
    for addr, sass, stalls in rows:
        b = stalls.get(args.focus, 0)
        if b <= 0:
            continue
        lin = view.source_line(addr)
        if lin:
            contrib.setdefault((lin[0], lin[1]), []).append(
                (addr, b, sass))

    for b_ in args.buckets.split(","):
        b_ = b_.strip()
        if not b_:
            continue
        fn, ln = b_.rsplit(":", 1)
        lst = sorted(contrib.get((fn, int(ln)), []), key=lambda t: -t[1])
        print("\n== %s : %d contributing PCs, %d total samples ==" %
              (b_, len(lst), sum(t[1] for t in lst)))
        for pc, b, sass in lst[:args.limit]:
            print("  0x%-18X bar=%-6d %s" % (pc, b, (sass or "")[:60]))


if __name__ == "__main__":
    main()
