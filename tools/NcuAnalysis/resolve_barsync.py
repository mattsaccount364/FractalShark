#!/usr/bin/env python3
"""Resolve barrier-stall samples (reported on library lines such as sync.h:109
or cooperative_groups.h:182) to the specific user-level grid.sync() call site.

For every contributing PC of the chosen bucket, this prints:
  * the SASS window around the PC,
  * the nearest preceding user-source line (a real .cu/.h line, not the
    cg runtime), which pins the enclosing loop/function region, and
  * the total samples for that PC.

This lets several grid.sync() sites that collapse onto one library line be
distinguished by their enclosing user-source line. Confirm the final mapping
against the kernel source.

Usage:
    py -3 resolve_barsync.py --report testcuda.ncu-rep --csv ncu_source.csv \
        --bucket "sync.h:109" [--top 5] [--radius 3]
"""
from __future__ import annotations

import argparse

import ncu_common as C


def main() -> int:
    p = argparse.ArgumentParser(description="Resolve a barrier bucket to its user grid.sync() sites.")
    p.add_argument("--report", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--bucket", required=True, help='e.g. "sync.h:109"')
    p.add_argument("--top", type=int, default=5, help="top N contributing PCs (default 5)")
    p.add_argument("--radius", type=int, default=3, help="SASS lines each side (default 3)")
    a = p.parse_args()

    bucket_fn, _, bucket_ln = a.bucket.partition(":")
    bucket_ln = int(bucket_ln)

    view = C.ReportView(a.report)
    print("kernel:", view.name)
    rows, _, _ = C.load_source_csv(a.csv)
    ordered = sorted(rows, key=lambda t: t[0])

    USER = (".cu", ".inl")  # user-source files, not the cg runtime (.h)

    def user_line(addr: int):
        lin = view.source_line(addr)
        if not lin:
            return None
        fn, ln = lin
        if fn.endswith(USER):
            return (fn, ln)
        return None

    # Collect the contributing PCs of the bucket.
    contrib = []
    for k, (addr, sass, stalls) in enumerate(ordered):
        b = stalls.get("stall_barrier", 0) if stalls else 0
        if b <= 0:
            continue
        lin = view.source_line(addr)
        if lin is None or C.short_file(lin[0]) != bucket_fn or int(lin[1]) != bucket_ln:
            continue
        # nearest preceding user-source line for context
        ctx = None
        for j in range(k, max(k - 400, -1), -1):
            ctx = user_line(ordered[j][0])
            if ctx:
                break
        contrib.append((addr, b, sass, ctx, k))

    contrib.sort(key=lambda t: -t[1])
    total = sum(c[1] for c in contrib)
    print(f"\nbucket {a.bucket}: barrier samples = {total} across {len(contrib)} PCs")

    for addr, b, sass, ctx, k in contrib[:a.top]:
        print(f"\n  PC=0x{addr:X}  barrier={b}  "
              f"({100.0 * b / max(total, 1):.2f}%)  nearest user src: {ctx}")
        for j in range(max(0, k - a.radius), min(len(ordered), k + a.radius + 1)):
            tag = ">" if ordered[j][0] == addr else " "
            print(f"    {tag} 0x{ordered[j][0]:X}: {ordered[j][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
