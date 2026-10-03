r"""Shared-memory traffic, bank conflicts, global coalescing, DRAM, and occupancy
limiters for an NCU report.

Answers the HWM (harmonic warp manager) questions the source-attribution pass
cannot: how many shared LD/ST wavefronts there are, how many bank conflicts,
how many LDGSTS async-copy wavefronts (the radix-2^15/16 large-stage page
copies), global sector coalescing (1.0 means one 32-byte sector per request;
32.0 means 32 sectors per request), DRAM traffic, I-cache miss
behavior, and which resource (registers, shared mem, warps, threads, barriers)
pins occupancy.

Usage:
    py -3 tools\NcuAnalysis\memory_analysis.py --report <rep> [--kernel-name <name>] [--action <index>]

Metric families are matched by candidate regex because NCU renames metrics
across releases (e.g. Blackwell splits ld/st into op-level metrics, and uses
`launch__occupancy_limit_barriers` on some architectures). Groups not present
in the capture are reported as such.
"""

import argparse

import ncu_common as C

_GROUPS = [
    ("Shared-memory wavefronts (LSU)", [
        ("shared_ld (absolute)", r"^l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld\.sum$"),
        ("shared_ld (% of peak)", r"^l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld\.sum\.pct_of_peak_sustained_elapsed$"),
        ("shared_st (absolute)", r"^l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st\.sum$"),
        ("shared_st (% of peak)", r"^l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st\.sum\.pct_of_peak_sustained_elapsed$"),
        ("ldgsts (async copy, SASS wavefronts)", r"^smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts\.sum$"),
        ("ldgsts (% of peak)", r"^smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts\.sum\.pct_of_peak_sustained_elapsed$"),
    ]),
    ("Shared-memory bank conflicts", [
        ("op_ld (absolute)", r"^l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld\.sum$"),
        ("op_st (absolute)", r"^l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st\.sum$"),
        ("op_ldgsts (absolute)", r"^l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts\.sum$"),
        ("op_atom (absolute)", r"^l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom\.sum$"),
    ]),
    ("Global coalescing (L1TEX sectors per request; 1.0 = one 32B sector, 32.0 = warp-scatter)", [
        ("global_ld sectors (t_sectors)", r"^l1tex__t_sectors_pipe_lsu_mem_global_op_ld\.sum$"),
        ("global_ld requests (t_requests)", r"^l1tex__t_requests_pipe_lsu_mem_global_op_ld\.sum$"),
        ("global_st sectors (t_sectors)", r"^l1tex__t_sectors_pipe_lsu_mem_global_op_st\.sum$"),
        ("global_st requests (t_requests)", r"^l1tex__t_requests_pipe_lsu_mem_global_op_st\.sum$"),
    ]),
    ("DRAM", [
        ("bytes (absolute)", r"^dram__bytes\.sum$"),
        ("bytes/s", r"^dram__bytes\.sum\.per_second$"),
    ]),
    ("L1/L2 cache", [
        ("l1tex hit rate (%)", r"^l1tex__t_sector_hit_rate(?:\.avg)?\.pct$"),
        ("ltc (L2) hit rate (%)", r"^lts__t_sector_hit_rate(?:\.avg)?\.pct$"),
        ("ltc read sectors", r"^lts__t_sectors_srcunit_tex_op_read\.sum$"),
    ]),
    ("I-cache / instruction miss", [
        ("smsp inst_cache_miss (sum)", r"^smsp__inst_cache_miss\..*$"),
        ("icache requests (sm)", r"^sm__inst_cache_miss\..*$"),
    ]),
    ("Occupancy limiters (the metric that pins max warps/CTA to the achieved value)", [
        ("launch limit: barriers", r"^launch__occupancy_limit_barriers$"),
        ("launch limit: blocks", r"^launch__occupancy_limit_blocks$"),
        ("launch limit: registers", r"^launch__occupancy_limit_registers$"),
        ("launch limit: shared_mem", r"^launch__occupancy_limit_shared_mem$"),
        ("launch limit: threads", r"^launch__occupancy_limit_threads$"),
        ("launch limit: warps", r"^launch__occupancy_limit_warps$"),
    ]),
    ("Occupancy (achieved)", [
        ("achieved", r"^sm__warps_active.avg.pct_of_peak_sustained_active$"),
        ("achieved warps/SM", r"^sm__warps_active.avg.per_cycle_active$"),
        ("theoretical max (%)", r"^sm__maximum_warps_per_active_cycle_pct$"),
        ("theoretical max (pct, max)", r"^sm__maximum_warps_per_active_cycle_pct\.max$"),
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
