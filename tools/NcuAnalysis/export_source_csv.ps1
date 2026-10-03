#Requires -Version 5.1
<#
Export per-launch source sections with a neighboring .manifest.json sidecar.
Run with PowerShell 7. Defaults to NCU_OUT_CSV or %TEMP%/ncu_source.csv.
Examples:
    ./tools/NcuAnalysis/export_source_csv.ps1 -Report testcuda.ncu-rep -ListRuns
    ./tools/NcuAnalysis/export_source_csv.ps1 -Report testcuda.ncu-rep -KernelName mandel_1xHDR_float_perturb_lav2
#>
param(
    [Parameter(Mandatory=$true, Position=0)] [string]$Report,
    [string]$KernelName,
    [ValidateSet('function', 'demangled', 'mangled')] [string]$KernelNameBase = 'function',
    [ValidateRange(0, [int]::MaxValue)] [int]$Range,
    [ValidateRange(0, [int]::MaxValue)] [int]$Action,
    [switch]$ListRuns,
    [string]$OutputCsv,
    [string]$PythonExe
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Report)) { throw "Report not found: $Report" }

if (-not $PythonExe) {
    $ncuRoots = @()
    if ($env:NCU_PYTHON_DIR) {
        $ncuRoots += Split-Path -Parent (Split-Path -Parent $env:NCU_PYTHON_DIR)
    }
    if ($env:NCU_EXE) {
        $ncuRoots += Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $env:NCU_EXE))
    }
    $installRoot = 'C:/Program Files/NVIDIA Corporation'
    if (Test-Path -LiteralPath $installRoot) {
        $ncuRoots += Get-ChildItem -LiteralPath $installRoot -Directory -Filter 'Nsight Compute*' |
            Sort-Object Name -Descending | Select-Object -ExpandProperty FullName
    }
    foreach ($ncuRoot in $ncuRoots) {
        $candidate = Join-Path $ncuRoot 'host/target-windows-x64/python/bin/python.exe'
        if (Test-Path -LiteralPath $candidate) { $PythonExe = $candidate; break }
    }
}

$pythonPrefix = @()
if (-not $PythonExe) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) { throw 'Python 3 not found; supply -PythonExe with a Python 3 executable.' }
    $PythonExe = $launcher.Source
    $pythonPrefix = @('-3')
}

$backendArgs = @('-B', (Join-Path $PSScriptRoot 'export_source_csv.py'), '--report', $Report,
                 '--kernel-name-base', $KernelNameBase)
if ($PSBoundParameters.ContainsKey('KernelName')) { $backendArgs += @('--kernel-name', $KernelName) }
if ($PSBoundParameters.ContainsKey('Range')) { $backendArgs += @('--range', [string]$Range) }
if ($PSBoundParameters.ContainsKey('Action')) { $backendArgs += @('--action', [string]$Action) }
if ($ListRuns) { $backendArgs += '--list-runs' }
if (-not $ListRuns) {
    if (-not $OutputCsv) {
        $OutputCsv = if ($env:NCU_OUT_CSV) { $env:NCU_OUT_CSV } else { Join-Path $env:TEMP 'ncu_source.csv' }
    }
    $env:NCU_OUT_CSV = $OutputCsv
    $backendArgs += @('--output-csv', $OutputCsv)
}
& $PythonExe @pythonPrefix @backendArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
