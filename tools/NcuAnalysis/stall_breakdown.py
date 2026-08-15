r"""Report the overall stall composition for a single-kernel NCU report.

This is the fast "what is the kernel even spending its time on" view, separate
from the per-line attribution in join_attribution.py. It reads the report
directly (no source CSV needed) and prints every non-zero `stall_*` reason split
into the issued base column and its `(Not Issued)` variant, with percentages of
the base-column total. It is the way to confirm which stall reason dominates
(typically `stall_barrier`, i.e. `grid.sync()`/`bar.sync`, followed by
`stall_wait` and `stall_long_scoreboard`) before drilling into per-line causes.

Usage:
    py -3 tools\NcuAnalysis\stall_breakdown.py --report <rep>
"""

import argparse

import ncu_common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    C.import_ncu_report(C.find_ncu_python_dir())
    import ncu_report

    ctx = ncu_report.load_report(args.report)
    act = ctx.range_by_idx(0).action_by_idx(0)
    names = list(act.metric_names() or [])

    base = {}
    notissued = {}
    stall_prefix = "smsp__pcsamp_warps_issue_stalled_"
    for n in names:
        low = n.lower()
        # NCU exposes both the canonical metric and a `warpsampling:` alias.  The
        # alias is the same sample count, so accepting both silently doubles the
        # stall budget.  Restrict aggregation to the canonical issue-stall names.
        if not low.startswith(stall_prefix) or "average_warps" in low:
            continue
        try:
            v = act.metric_by_name(n).value()
            v = float(v)
        except Exception:
            continue
        if v <= 0:
            continue
        key = low.replace("_not_issued", "").replace(".not_issued", "")
        if "not_issued" in low:
            notissued[key] = notissued.get(key, 0) + v
        else:
            base[key] = base.get(key, 0) + v

    tb = sum(base.values())
    try:
        print("kernel:", C.ReportView(args.report).name)
        print("  duration =", act.metric_by_name("gpu__time_duration.sum").value(), "ns")
        print("  regs/thread =", act.metric_by_name("launch__registers_per_thread").value())
    except Exception:
        pass
    print("\n== stall reasons (base issued columns), total %d ==" % tb)
    for n, v in sorted(base.items(), key=lambda kv: -kv[1]):
        print("%12d  %6.2f%%  %s" % (v, 100 * v / tb, n))
    tn = sum(notissued.values())
    print("\n== (Not Issued) variants, total %d ==" % tn)
    for n, v in sorted(notissued.items(), key=lambda kv: -kv[1]):
        print("%12d  %6.2f%%  %s" % (v, 100 * v / tn, n))
    print("\ngrand total (base + not-issued):", tb + tn)


if __name__ == "__main__":
    main()
