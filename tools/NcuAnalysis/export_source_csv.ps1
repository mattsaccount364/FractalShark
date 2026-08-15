"""Export the NCU source-page CSV for a .ncu-rep report.

Usage:
    powershell -NoProfile -File tools\NcuAnalysis\export_source_csv.ps1 `
        -Report <path-to.ncu-rep> [-OutCsv <csv>] [-NcuExe <path>]

Notes:
- The profiled binary must have been built with --source-info (the default
  for device code) and NCU must have been able to resolve source lines at
  capture time; this script only exports.
- Output CSV column layout includes per-instruction stall_* sample counts,
  which is what the join_attribution.py / inspect_hot.py scripts consume.
"""
param(
    [Parameter(Mandatory = $true)][string] $Report,
    [string] $OutCsv = (Join-Path $env:TEMP "ncu_source.csv"),
    [string] $NcuExe = ""
)

$ErrorActionPreference = "Stop"

if (-not $NcuExe) {
    $root = "C:\Program Files\NVIDIA Corporation"
    $cands = @(Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "Nsight Compute*" } |
        ForEach-Object { Join-Path $_.FullName "target\windows-desktop-win7-x64\ncu.exe" } |
        Where-Object { Test-Path $_ } |
        Sort-Object -Descending)
    if (-not $cands.Count) {
        Write-Error "Could not find ncu.exe; pass -NcuExe."
    }
    $NcuExe = $cands[0]
}

Write-Host "ncu : $NcuExe"
Write-Host "rep : $Report"
Write-Host "csv : $OutCsv"

& $NcuExe --import $Report --page source --csv | Out-File -FileEncoding utf8 -LiteralPath $OutCsv
if ($LASTEXITCODE -ne 0) {
    Write-Error "ncu export failed with exit code $LASTEXITCODE"
}

"OK: $OutCsv ($((Get-Item -LiteralPath $OutCsv).Length) bytes)"
