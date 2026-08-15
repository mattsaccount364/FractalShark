# NcuAnalysis

Python/PowerShell helpers for attributed NCU (Nsight Compute) profile
analysis of FractalShark CUDA kernels. Used for the Reference2
`grid.sync()` / barrier attribution work described in
`Notes/Reference2GpuLargeStageRadixAnalysis.md`.

All scripts are single-kernel scoped (first range / first action of the
report) and work against the NCU Python API shipped with Nsight Compute
(`extras\python\ncu_report`). On this machine: 2026.1.1.0.

## Prerequisites
- Nsight Compute installed (auto-detected under
  `C:\Program Files\NVIDIA Corporation\Nsight Compute*\`; override with
  `NCU_EXE` / `NCU_PYTHON_DIR`).
- Python 3 (`py -3`) for the NCU Python API (the system `python` may be
  2.x and will fail).

## Files
| Script | Purpose |
|---|---|
| `export_source_csv.ps1` | `ncu --import <rep> --page source --csv` wrapper; the per-instruction stall CSV the other scripts consume. |
| `join_attribution.py` | Summary (duration, grid, registers, spills) + per-line attribution of one stall column (default `stall_barrier`), with the `BAR.SYNC`/`BRA.DIV` wait-site correction; must reconcile to 0 unattributed. |
| `inspect_hot.py` | For a hot bucket on a library line (`sync.h:109`, `cooperative_groups.h:182`), prints the SASS window of the contributing PCs so the bucket can be joined back to the user-level `grid.sync()` call site. |
| `all_hot_pcs.py` | Dumps every contributing PC of a bucket with SASS + source line for manual joining. |
| `stall_breakdown.py` | Overall stall-reason composition (barrier vs wait vs long_scoreboard vs ...) from the per-instruction base columns; confirms which stall type dominates before per-line work. |
| `resolve_barsync.py` | Barrier-wait -> `BAR.SYNC` resolver; for each library bucket prints the SASS around its resolved `BAR.SYNC` plus the exact wait samples, so the bucket maps back to a specific user `grid.sync()` site. |
| `ncu_common.py` | Shared helpers (NCU discovery, CSV loading, `ReportView`). |

## Typical workflow
```powershell
$rep = "H:\Documents\Programming\FractalShark\testcuda.ncu-rep"
$csv = "$env:TEMP\ncu_source.csv"

powershell -NoProfile -File tools\NcuAnalysis\export_source_csv.ps1 -Report $rep -OutCsv $csv

py -3 tools\NcuAnalysis\join_attribution.py --report $rep --csv $csv --top 25

py -3 tools\NcuAnalysis\inspect_hot.py --report $rep --csv $csv `
    --buckets "sync.h:109,cooperative_groups.h:182"
```

## Interpretation notes (method)
- Barrier-wait samples are the rows with nonzero `stall_barrier`. Their sum
  per kernel is the barrier budget; divide by the sum of all `stall_*`
  columns for the barrier share of the stall budget.
- `grid.sync()` compiles to `BAR.SYNC <imm>; BRA.DIV 0x0` (16 bytes apart).
  In the binary layout captured in Aug 2026 the wait samples land on the
  instruction immediately after the pair; `join_attribution.py` maps those
  back via `source_info`.
- In other ptxas layouts the wait samples land **inside** the cg runtime
  (`sync.h` / `cooperative_groups.h` lines). `join_attribution.py` then
  reports those library lines; use `inspect_hot.py` to identify which user
  call site each one belongs to.
- The join must reconcile exactly (attributed + unattributed == total)
  before any ranking is considered trustworthy.
