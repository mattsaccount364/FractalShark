r"""Per-line stall attribution for a single-kernel NCU report.

Usage:
    py -3 tools\NcuAnalysis\join_attribution.py --report <rep> --csv <csv>
        [--top 30] [--focus stall_barrier]

What it prints:
  - kernel name, duration, grid/block, registers/thread, dynamic shared,
    local-memory load/store totals;
  - per-instruction stall totals across all stall_* columns and the share
    of the focus column (default stall_barrier);
  - a top-N ranking of source lines by focus-column samples (falling back to
    the BAR.SYNC/BRA.DIV pair heuristic when available);
  - per-file totals and the unattributed remainder.

The join reconciles exactly: attributed sum + unattributed == focus total,
so no samples are lost when the output ends with "unattributed: 0".

Requires the source CSV produced by export_source_csv.ps1 (same report).
"""

import argparse
import collections
import sys

import ncu_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--focus", default="stall_barrier",
                    help="stall_* column to attribute (default stall_barrier)")
    args = ap.parse_args()

    rows, stall_cols, kernels = C.load_source_csv(args.csv)
    if not rows:
        raise SystemExit("source CSV is empty; re-run export_source_csv.ps1")
    print("kernels in csv:", len(kernels))
    for k in kernels[:4]:
        print("  -", k[1] if len(k) > 1 else k)

    view = C.ReportView(args.report)
    print("\nreport kernel:", view.name)
    for name in ("gpu__time_duration.sum", "launch__grid_size",
                 "launch__block_size", "launch__registers_per_thread",
                 "launch__shared_mem_per_block_static",
                 "launch__shared_mem_per_block_dynamic",
                 "local_memory__t_bytes_pipe_lsu_mem_local_op_ld.sum",
                 "local_memory__t_bytes_pipe_lsu_mem_local_op_st.sum"):
        v = view.metric(name)
        print("  %s = %s" % (name, v))

    ordered = sorted(rows, key=lambda t: t[0])
    pairs = C.find_barrier_wait_sites(ordered)
    print("\nSASS instruction rows:", len(ordered),
          " grid-sync wait pairs:", len(pairs))

    focus = args.focus
    if focus not in stall_cols:
        print("warning: focus column %r not found in csv; available: %s"
              % (focus, stall_cols))
        focus = stall_cols[0] if stall_cols else None
        if focus is None:
            raise SystemExit("csv has no stall_* columns")

    total_focus = sum(r[2].get(focus, 0) for r in ordered)
    # Each stall type appears twice in the CSV: the base column and a
    # "(Not Issued)" variant. The stall budget is the sum of the base
    # columns only (matches the methodology used in the GPU reference note).
    base_cols = [c for c in stall_cols if "(Not Issued)" not in c]
    total_base = sum(sum(r[2].get(c, 0) for c in base_cols) for r in ordered)
    print("focus column %s total: %d" % (focus, total_focus))
    print("stall budget (sum of base stall_ columns): %d  (focus share: %.2f%%)"
          % (total_base, 100.0 * total_focus / max(total_base, 1)))
    raw_all = sum(sum(r[2].values()) for r in ordered)
    print("raw sum including (Not Issued) variants: %d" % raw_all)

    per_line = collections.Counter()
    unattr = 0
    for addr, sass, stalls in ordered:
        b = stalls.get(focus, 0)
        if b == 0:
            continue
        lin = view.source_line(addr)
        if lin is None and addr in pairs:
            lin = view.source_line(pairs[addr])
        if lin is None:
            unattr += b
            continue
        per_line[(lin[0], lin[1])] += b

    print("\ntop %d lines by %s samples:" % (args.top, focus))
    for (fn, ln), v in per_line.most_common(args.top):
        print("%12d  %5.2f%%  %s:%s" % (v, 100.0 * v / max(total_focus, 1),
                                        fn, ln))
    print("unattributed:", unattr)
    print("attributed sum:", sum(per_line.values()),
          " rest:", total_focus - sum(per_line.values()))

    pf = collections.Counter()
    for (fn, ln), v in per_line.items():
        pf[fn] += v
    print("per-file totals:",
          pf.most_common())


if __name__ == "__main__":
    main()
