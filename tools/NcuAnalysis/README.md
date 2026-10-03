# NcuAnalysis

Python and PowerShell tools for inspecting Nsight Compute reports containing multiple kernels,
repeated launches, and multiple report ranges. Each launch is analyzed independently. Counts,
percentages, source PCs, and barrier context are never merged across launches.

## Runtime

Use Python 3.10 or newer with the `ncu_report` API from the installed Nsight Compute version.
The tools locate NVIDIA installations automatically. `NCU_PYTHON_DIR` overrides the API directory;
`NCU_EXE` overrides the native CLI executable.

On this Windows host, Nsight Compute 2026.1.1 supplies a compatible Python 3.12 interpreter:

```powershell
$python = 'C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.1.1/host/target-windows-x64/python/bin/python.exe'
$report = 'H:/Documents/Programming/FractalShark/testcuda.ncu-rep'
$csv = Join-Path $env:TEMP 'ncu_source.csv'
```

The PowerShell export wrapper prefers the bundled interpreter automatically, then falls back to
`py -3`. Override that choice with `-PythonExe`. Invoke the wrapper with the installed PowerShell 7,
not the WindowsApps shim or Windows PowerShell 5.1. The bundled interpreter supports the analysis
scripts but lacks `unittest`; unit tests require a full Python installation.

NCU can warn that it could not deploy stock sections into the user Documents directory and then
successfully use its installed sections. The tools do not change HOME or install section files.
Native failures propagate; successful exports retain only CSV data, excluding that warning preamble.

## Selection shared by every tool

| Python option | PowerShell exporter option | Behavior |
|---|---|---|
| `--list-runs` | `-ListRuns` | List matching launches without analysis or export |
| `--kernel-name NAME` | `-KernelName NAME` | Exact name, or `regex:EXPRESSION` for a partial regex match |
| `--kernel-name-base BASE` | `-KernelNameBase BASE` | `function`, `demangled`, or `mangled`; default `function` |
| `--range INDEX` | `-Range INDEX` | Restrict to one report range |
| `--action INDEX` | `-Action INDEX` | Select an action in that range; assumes range 0 if omitted |

Without selectors, every kernel action in every range is analyzed separately. A function-name
filter includes all matching specializations and launches; use the full demangled or mangled name
when specialization matters. Combined selectors must all match. Invalid indices, invalid regexes,
and empty selections produce errors instead of reverting to the first action.

The launch table reports range/action identity, device/context/stream, dimensions, and duration.
Detailed output includes the full specialization and register count. The timing summary is the
sum of available kernel durations, not application elapsed time. Utilization percentages and
sample counts remain per launch. `--list-runs` works without source CSV or bucket arguments.

Range/action indices are report-local handles, not the original UI/profiler launch identifiers.
For the supplied capture, the sole LA v2 action is **range 0, action 9**. The three
`mandel_1x_float` launches are actions **0, 3, and 6**; action 6 uses a different stream.
The tool does not translate an original label such as 765 into an action index.

## Workflow for the LA kernel

```powershell
& $python -B ./tools/NcuAnalysis/metric_probe.py --report $report --list-runs
& $python -B ./tools/NcuAnalysis/metric_probe.py --report $report --action 9 --pattern 'cache|hit_rate' --values

# Export only the matching launches. The output path also becomes NCU_OUT_CSV
# in this PowerShell session when the script is invoked directly.
& ./tools/NcuAnalysis/export_source_csv.ps1 -Report $report `
    -KernelName mandel_1xHDR_float_perturb_lav2 -OutputCsv $csv

& $python -B ./tools/NcuAnalysis/stall_breakdown.py --report $report --action 9
& $python -B ./tools/NcuAnalysis/pipe_analysis.py --report $report --action 9
& $python -B ./tools/NcuAnalysis/memory_analysis.py --report $report --action 9
& $python -B ./tools/NcuAnalysis/join_attribution.py --report $report `
    --csv $csv --action 9 --focus stall_long_sb --top 25
& $python -B ./tools/NcuAnalysis/inspect_hot.py --report $report `
    --csv $csv --action 9 --buckets 'LAKernel.cuh:221' --focus stall_barrier
& $python -B ./tools/NcuAnalysis/all_hot_pcs.py --report $report `
    --csv $csv --action 9 --buckets 'LAKernel.cuh:221'
& $python -B ./tools/NcuAnalysis/resolve_barsync.py --report $report `
    --csv $csv --action 9 --bucket 'LAKernel.cuh:221'
```

The exporter retains the existing `-Report` argument and `NCU_OUT_CSV` environment override.
Without an explicit output path or environment override, output is `%TEMP%/ncu_source.csv`.
A Python-only entry point provides the same behavior:

```powershell
& $python -B ./tools/NcuAnalysis/export_source_csv.py --report $report `
    --kernel-name mandel_1xHDR_float_perturb_lav2 --output-csv $csv
```

## Repeated launches and source provenance

```powershell
$repeatedCsv = Join-Path $env:TEMP 'ncu_repeated_source.csv'
& ./tools/NcuAnalysis/export_source_csv.ps1 -Report $report `
    -KernelName mandel_1x_float -OutputCsv $repeatedCsv
& $python -B ./tools/NcuAnalysis/join_attribution.py --report $report `
    --kernel-name mandel_1x_float --csv $repeatedCsv --focus stall_wait
```

This produces three independent analyses, not one summed source ranking. To analyze all kernels,
export without a selector and then analyze that export without a selector. A CSV that contains
only selected launches cannot supply an analysis of additional launches.

Source export invokes NCU once per launch for native metadata and once for SASS. Import filters
select one launch at a time; native name, device/context/stream, dimensions, duration, SASS, and
available source instruction totals are checked against the API run before publication.
Configuration-file overrides and kernel renaming are disabled for these import commands.

The CSV preserves separate native `Kernel Name`/`Address` sections, including repeated names and
PCs. Its neighboring **`<csv>.manifest.json`** records schema version 1, SHA-256 fingerprints of the
report and CSV, and each section's range/action identity and mangled name. Keep the pair together.
The reader checks fingerprints, identities, section counts, SASS, and available instruction totals;
stale, mismatched, missing, or duplicate sections produce a re-export error. Exports are published
only after all selected launches pass validation. If publication is interrupted between the two
file replacements, the fingerprint check rejects the inconsistent pair.

Legacy native CSVs without a manifest are accepted only for one selected launch and one section,
with a specialization that occurs exactly once in the report and matching SASS/instruction totals.
Repeated-launch legacy exports require re-exporting; matching names or shared PCs cannot identify
which launch produced their samples. Generated CSVs and manifests belong in TEMP or an untracked
validation directory, not in the repository's source history.

## Attribution and interpretation

For each run and requested sample family, source tools print the canonical action total, CSV total,
attributed and unattributed counts within the CSV, and the canonical-minus-CSV residual. The CSV
join must reconcile exactly. A ranking is printed only when **unattributed within CSV is zero**
and the CSV does not exceed the canonical count. When coverage is incomplete or a canonical metric
is unavailable, rankings explicitly describe the **CSV subset only**, never full-run attribution.
Unmapped samples are not assigned to whichever source line happened to rank highest.

In the supplied LA capture, `stall_long_sb` has 32,320 canonical samples but only 17,810 CSV samples:
its within-CSV join has zero unattributed samples, while the separate residual is **14,510**.
`stall_wait` has residual **485**, and `stall_barrier` has residual **875**. A zero within-CSV
remainder does not erase those missing samples. Missing requested columns produce an error rather
than silently selecting another family.

Use canonical `smsp__pcsamp_warps_issue_stalled_*` metrics. `warpsampling:` aliases are excluded;
their equivalence must not be assumed. `(Not Issued)` variants are subsets and are reported
separately, not added to the base sample budget. `selected` and `not_selected` are scheduler
states rather than true stalls, and their shares are not elapsed-time percentages. Source waits
identify consumers; inspect surrounding SASS before attributing them to producers. NVIDIA's
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons)
describes these distinctions.

The old `BAR.SYNC; BRA.DIV` adjacency heuristic is retained for compatible grid-sync layouts.
It is not a universal pattern: inspect SASS for deferred block barriers and compiler-dependent
layouts. The hot-PC tools keep their windows within one selected source section. Occupancy, block
size, and SM counts are reported from the capture; prior reference-kernel values are not defaults
for an unrelated kernel. Recorded zero metrics remain visible; unavailable metric groups are
marked unavailable. Cache hit-rate candidates cover both current `.pct` and older `.avg.pct` names.

## Tools

| Script | Purpose |
|---|---|
| `export_source_csv.ps1` / `export_source_csv.py` | Verified source export plus provenance manifest |
| `metric_probe.py` | Launch listing and metric-name/value discovery |
| `stall_breakdown.py` | Canonical scheduler-state budgets, per launch |
| `pipe_analysis.py` | Pipe utilization, issue rate, lane activity, instruction classes |
| `memory_analysis.py` | Memory traffic, bank conflicts, caches, occupancy |
| `join_attribution.py` | Source-line and file rankings with explicit coverage |
| `inspect_hot.py` | SASS windows around contributing bucket PCs |
| `all_hot_pcs.py` | Contributing PCs and counts for requested buckets |
| `resolve_barsync.py` | Barrier-bucket SASS and nearby caller context |
| `ncu_common.py` | Report ownership, selection, section parsing, provenance, attribution |

## Validation

Standard-library tests use fake API contexts and temporary source exports; no GPU or NCU API is
required. Run them with a full Python 3.10+ interpreter. On this host:

```powershell
& 'C:/Users/Matthew/AppData/Local/Programs/Python/Python310/python.exe' -B `
    -m unittest discover -s ./tools/NcuAnalysis -p 'test_*.py' -v
```

For native validation, list the supplied report's 12 actions, export all of them, exercise each
helper with action 9 and with the repeated `mandel_1x_float` filter, and confirm the LA source
instruction count of **21,975,576,711** and the residuals above. These are capture-specific checks;
new report fingerprints and run catalogs must be used after recapturing. Only the Python and
PowerShell tooling changed; CUDA or renderer rebuilds are not required for these checks.
