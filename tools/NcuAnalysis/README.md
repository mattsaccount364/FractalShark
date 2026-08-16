# NcuAnalysis

Python/PowerShell helpers for attributed NCU (Nsight Compute) profile
analysis of FractalShark CUDA kernels. Used for the Reference2
`grid.sync()` / barrier attribution work described in
`Notes/Reference2GpuLargeStageRadixAnalysis.md`.

Most scripts are single-kernel scoped (first range / all actions of the
report summed — divide per-line counts by the captured instance count, 5 in
the current report, for per-instance values). All tools work against the NCU
Python API shipped with Nsight Compute (`extras\python\ncu_report`) or a
pre-exported source-page CSV. On this machine: Nsight Compute 2026.1.1.0,
RTX 5090 (CC 12.0).

## Prerequisites
- Nsight Compute installed (auto-detected under
  `C:\Program Files\NVIDIA Corporation\Nsight Compute*\`; override with
  `NCU_EXE` / `NCU_PYTHON_DIR`).
- Python 3 (`py -3`) for the NCU Python API (the system `python` may be
  2.x and will fail).
- The source CSV lives at `$env:NCU_OUT_CSV` (default `%TEMP%\ncu_source.csv`).
  Note: on this host TEMP is `H:\WindowsTemp`, not the user profile — always
  key off `$env:TEMP`, never a hardcoded profile path.
- NCU renames metrics across releases (Blackwell, e.g., exposes `cbu`, `lsu`,
  `tma`, `fma_type_*` and `launch__occupancy_limit_barriers` but no
  `inst_cache_*` family). The analysis scripts therefore match metric
  families by candidate regex and skip families the capture lacks. Use
  `metric_probe.py` to search what a given report exposes.

## Files
| Script | Purpose |
|---|---|
| `export_source_csv.ps1` | `ncu --import <rep> --page source --csv` wrapper; writes the per-instruction stall CSV to `$env:NCU_OUT_CSV` (default `%TEMP%\ncu_source.csv`) — the input for the CSV-based scripts. |
| `join_attribution.py` | Summary (duration, grid, registers, spills) + per-line attribution of one stall column (`--focus stall_barrier` default; re-run with `--focus stall_wait` / `stall_long_sb` / ...); sums over all captured instances; must reconcile to 0 unattributed. |
| `stall_breakdown.py` | Per-instance stall-reason composition (barrier / wait / long_sb / ...) from the per-instruction base columns; confirms which stall type dominates before per-line work. |
| `pipe_analysis.py` | Pipe utilization (% of peak per pipe), scheduler issue rate, average active lanes per warp (divergence), the per-reason scheduler stall mix, and the SASS memory-op instruction mix. |
| `memory_analysis.py` | Shared-memory LD/ST + LDGSTS wavefronts, shared bank conflicts (ld/st/ldgsts/atom), global sector coalescing (t_sectors vs t_requests), DRAM bytes/s, L1/L2 hit rates, and the launch occupancy limiters (barriers/blocks/registers/shared_mem/warps). |
| `metric_probe.py` | Discovery: lists (or prints values of, with `--values`) the metric names in a report matching a regex — use first against a new NCU/architecture before trusting the others' output. |
| `inspect_hot.py` | For a hot bucket on a library line (`sync.h:109`, `cooperative_groups.h:182`), prints the SASS window of the contributing PCs so the bucket can be joined back to the user-level `grid.sync()` call site. |
| `all_hot_pcs.py` | Dumps every contributing PC of a bucket with SASS + source line for manual joining. |
| `resolve_barsync.py` | Barrier-wait -> `BAR.SYNC` resolver; for each library bucket prints the SASS around its resolved `BAR.SYNC` plus the exact wait samples, so the bucket maps back to a specific user `grid.sync()` site. |
| `ncu_common.py` | Shared helpers (NCU discovery, CSV loading, `ReportView`, barrier-wait site finder, stall-column reconciliation). |

## Typical workflow
```powershell
$rep = "H:\Documents\Programming\FractalShark\testcuda.ncu-rep"

# 1. Export the source-page CSV (needed by join_attribution / inspect_hot /
#    resolve_barsync). Output goes to %TEMP%\ncu_source.csv (override with
#    $env:NCU_OUT_CSV before running).
powershell -NoProfile -File tools\NcuAnalysis\export_source_csv.ps1 -Report $rep

# 2. Per-instance stall composition (no CSV needed; report API only).
py -3 tools\NcuAnalysis\stall_breakdown.py --report $rep

# 3. Per-line attribution for one stall family at a time (each must reconcile
#    to 0 unattributed before trusting the ranking; re-run per family).
py -3 tools\NcuAnalysis\join_attribution.py --report $rep `
    --csv "$env:NCU_OUT_CSV" --focus stall_barrier --top 25
py -3 tools\NcuAnalysis\join_attribution.py --report $rep `
    --csv "$env:NCU_OUT_CSV" --focus stall_wait --top 25
py -3 tools\NcuAnalysis\join_attribution.py --report $rep `
    --csv "$env:NCU_OUT_CSV" --focus stall_long_sb --top 25

# 4. Runtime profile (pipes, issue, divergence, stalls, occupancy).
py -3 tools\NcuAnalysis\pipe_analysis.py  --report $rep
py -3 tools\NcuAnalysis\memory_analysis.py --report $rep

# 5. Join barrier buckets on library lines back to user call sites
#    (both need the exported CSV).
py -3 tools\NcuAnalysis\resolve_barsync.py --report $rep `
    --csv "$env:NCU_OUT_CSV" --bucket "sync.h:109"
py -3 tools\NcuAnalysis\inspect_hot.py --report $rep `
    --csv "$env:NCU_OUT_CSV" `
    --buckets "sync.h:109,cooperative_groups.h:182"
```

## Interpretation notes (method)
- Barrier-wait samples are the rows with nonzero `stall_barrier`. Their sum
  per kernel is the barrier budget; divide by the sum of all *base* `stall_*`
  columns (never the `(Not Issued)` variants — those double-count) for the
  barrier share of the stall budget.
- `grid.sync()` compiles to `BAR.SYNC <imm>; BRA.DIV 0x0` (16 bytes apart).
  In the binary layout captured for this report the wait samples land on the
  instruction immediately after the pair; `join_attribution.py` maps those
  back via `source_info`. In other ptxas layouts the wait samples land
  **inside** the cg runtime (`sync.h` / `cooperative_groups.h` lines) and
  `join_attribution.py` reports those library lines; use `inspect_hot.py` /
  `resolve_barsync.py` to identify which user call site each belongs to.
- The join must reconcile exactly (attributed + unattributed == total)
  before any ranking is considered trustworthy.
- Occupancy math: on this GPU (RTX 5090, CC 12.0, 170 SMs) the kernel runs
  8 warps (256 threads) per CTA. `pipe_analysis`'s "avg active lanes"
  (32.0 uniform, 1.0 fully divergent) and `memory_analysis`'s launch
  limiters together say whether a result is stall-bound (issue << 100%,
  pipes all < ~5% of peak, occupancy pinned by shared memory/registers),
  which is the diagnostic path taken in the Reference2 analysis note:
  issue 14.8% / top pipe (LSU) 3.8% / achieved 16.65% occupancy / shared-
  memory-limited => pure latency-bound, no pipe saturated.

