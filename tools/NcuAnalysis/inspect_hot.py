r"""Identify the call site behind a hot library line (sync.h /
cooperative_groups.h) by showing the SASS context of its contributing PCs.

When ptxas inlines cg::grid_group::sync(), barrier-wait samples land on the
library's sync.h/cooperative_groups.h lines instead of the user call site.
For each contributing PC this tool prints a small window of preceding
instructions (where the caller's code and the BAR.SYNC live) with source
lines, so the bucket can be joined back to a specific grid.sync() call in
KernelHpSharkReferenceOrbit_cu.h / ReferenceNTT_cu.h.

Usage:
    py -3 tools\NcuAnalysis\inspect_hot.py --report <rep> --csv <csv> \
        --buckets "sync.h:109,cooperative_groups.h:182" [--top-pcs 3]
"""

import argparse
import re
import sys

import ncu_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--buckets", required=True,
                    help='comma-separated "file:line" buckets, e.g. '
                         '"sync.h:109,cooperative_groups.h:182"')
    ap.add_argument("--focus", default="stall_barrier")
    ap.add_argument("--top-pcs", type=int, default=2)
    ap.add_argument("--back", type=int, default=6,
                    help="instructions to show before the hot PC")
    ap.add_argument("--fwd", type=int, default=3,
                    help="instructions to show after the hot PC")
    args = ap.parse_args()

    rows, stall_cols, _ = C.load_source_csv(args.csv)
    lookup = {a: (s, st) for a, s, st in rows}

    view = C.ReportView(args.report)

    contrib = {}
    for addr, sass, stalls in rows:
        b = stalls.get(args.focus, 0)
        if b <= 0:
            continue
        lin = view.source_line(addr)
        if lin:
            contrib.setdefault((lin[0], lin[1]), []).append((addr, b))

    def window(pc):
        out = []
        a = int(pc)
        for d in range(-args.back, args.fwd + 1):
            x = a + 16 * d
            s = lookup.get(x)
            if s is None:
                out.append(("%+d" % d, "--", "--"))
                continue
            ln = view.source_line(x)
            tag = ("%s:%d" % (ln[0], ln[1])) if ln else "--"
            out.append(("%+d" % d, (s[0] or "")[:34], tag))
        return out

    for b_ in args.buckets.split(","):
        b_ = b_.strip()
        if not b_:
            continue
        fn, ln = b_.rsplit(":", 1)
        key = (fn, int(ln))
        lst = sorted(contrib.get(key, []), key=lambda t: -t[1]
                     )[:args.top_pcs]
        if not lst:
            print("\n== %s : no direct contribs (indirect bucket) ==" % b_)
            continue
        for pc, bcount in lst:
            print("\n== %s : pc 0x%X (bar=%d) ==" % (b_, pc, bcount))
            for tag, sass, line in window(pc):
                print("   %-5s %-34s %s" % (tag, sass, line))


if __name__ == "__main__":
    main()
