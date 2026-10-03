r"""Pipe utilization, scheduler issue, and SASS opcode mix for an NCU report.

Answers "is the kernel compute-bound, issue-bound, or divergent?" by reporting
the % of peak sustained per SM pipe (ALU, FMA, LSU, XU, FP64, ...), the
scheduler issue rate, the SASS instruction mix by instruction class, the
per-reason scheduler stall mix, and warp divergence.

Usage:
    py -3 tools\NcuAnalysis\pipe_analysis.py --report <rep> [--kernel-name <name>] [--action <index>]

Metric names are matched by candidate regex because NCU renames pipes across
architectures (e.g. Blackwell exposes `cbu`, `lsu`, `tma`, `uniform`, `fma_type_*`).
Metric families that are not present in the capture are skipped silently
(use `metric_probe.py` to search what a report exposes).
"""

import argparse

import ncu_common as C

_GROUPS = [
    # ("label", [ (display_name, candidate_regex_or_none), ... ])
    ("Pipe utilization (% of peak sustained elapsed)", [
        (n,
         r"^sm__inst_executed_pipe_(%s)\.sum\.pct_of_peak_sustained_elapsed$" % "|".join(cands))
        for n, cands in [
            ("alu", ["alu"]), ("cbu (aluheavy)", ["cbu", "aluheavy"]), ("cbu2", ["cbu2"]),
            ("fma", ["fma"]), ("fma_fp32", ["fma_type_fp32"]), ("fma_int", ["fma_type_int32"]),
            ("fma_fp64", ["fma_type_fp64"]), ("fp64", ["fp64", "dp"]),
            ("lsu (load-store)", ["lsu", "lu"]), ("xu (xu/mufu)", ["xu"]), ("tex", ["tex"]),
            ("tma", ["tma"]), ("uniform (UMOV/UIMAD...)", ["uniform"]),
            ("adu (address)", ["adu"]), ("ipa (int/alu pipe)", ["ipa"]), ("workid", ["workid"]),
        ]
    ]),
    ("Scheduler issue rate", [
        ("issue_active (avg, % of peak)", r"^smsp__issue_active\.avg\.pct_of_peak_sustained_active$"),
        ("issue_active (sum, % of peak)", r"^smsp__issue_active\.sum\.pct_of_peak_sustained_active$"),
        ("ipc (smsp__issue_active.avg.per_cycle_active)", r"^smsp__issue_active\.avg\.per_cycle_active$"),
    ]),
    ("Warp divergence", [
        ("avg active lanes per warp-exec (32.0 = uniform; 1.0 = fully divergent)",
         r"^smsp__thread_inst_executed_per_inst_executed\.ratio$"),
    ]),
    ("Scheduler stall mix (warps stalled per issue-active, by reason)", [
        (r,
         "smsp__average_warps_issue_stalled_%s_per_issue_active\\.ratio" % r)
        for r in [
            "barrier", "wait", "long_scoreboard", "short_scoreboard", "membar",
            "no_instruction", "not_selected", "selected", "branch_resolving",
            "dispatch_stall", "math_pipe_throttle", "mio_throttle", "lg_throttle",
            "tex_throttle", "drain", "sleeping", "misc",
        ]
    ]),
    ("SASS instruction mix (memory classes, absolute)", [
        (r,
         "smsp__sass_inst_executed_op_%s\\.sum" % r)
        for r in [
            "shared", "shared_ld", "shared_st", "dshared", "dshared_ld", "dshared_st",
            "global_ld", "global_st", "local_ld", "local_st",
            "ldgsts_cache_access", "ldgsts_cache_bypass",
            "tma_ld", "tma_st", "tma_red", "tma_atom",
        ]
    ]),
]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    C.add_selection_arguments(parser)
    args = parser.parse_args()
    for view in C.selected_runs(args):
        C.print_metric_groups(view, _GROUPS)


if __name__ == '__main__':
    main()
