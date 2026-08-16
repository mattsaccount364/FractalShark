#Requires -Version 5.1
<#
Export the NCU source-page CSV for a .ncu-rep report.

Usage:
    powershell -NoProfile -File tools\NcuAnalysis\export_source_csv.ps1 `
        -Report <path-to.ncu-rep>

The CSV is written to $env:NCU_OUT_CSV, or %TEMP%\ncu_source.csv by default
(kept outside the repo so it is never committed).

Notes:
  * The source page is huge (tens of MB). Prefer keeping it out of the repo.
  * The CSV contains no kernel durations -- only per-source-line stall counts.
    For duration + occupancy, run ``--page details`` separately.
  * ``param()`` must be the first statement in a script, so this header is a
    comment block rather than a bare string literal (a literal before param()
    is a parse error).
#>
param(
    # Path to the .ncu-rep report (required).
    [Parameter(Mandatory=$true, Position=0)] [string]$Report
)

$ErrorActionPreference = 'Stop'

# Output path first: PowerShell cannot use Join-Path in a default parameter
# value, so resolve it here.
$env:NCU_OUT_CSV = if ($env:NCU_OUT_CSV) { $env:NCU_OUT_CSV } else { Join-Path $env:TEMP "ncu_source.csv" }
if (-not (Test-Path $Report)) { Write-Error "Report not found: $Report"; exit 1 }

# Locate ncu.exe from known install dirs, then PATH.
$ncu = $null
$candidates = @(
    "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\target\windows-desktop-win7-x64\ncu.exe",
    "C:\Program Files\NVIDIA\Nsight Compute 2026.1.1\target\windows-desktop-win7-x64\ncu.exe",
    "C:\Program Files\NVIDIA\Nsight Compute 2026.1\target\windows-desktop-win7-x64\ncu.exe",
    "C:\Program Files\NVIDIA\Nsight Compute 2025.3.1\target\windows-desktop-win7-x64\ncu.exe"
)
foreach ($c in $candidates) { if (Test-Path $c) { $ncu = $c; break } }
if ($null -eq $ncu) {
    $cmd = Get-Command ncu -ErrorAction SilentlyContinue
    if ($cmd) { $ncu = $cmd.Source }
}
if ($null -eq $ncu) { Write-Error "Could not locate ncu.exe in the standard install dirs or PATH."; exit 1 }
Write-Host "[export] Using NCU: $ncu"

Write-Host "[export] Exporting source page for: $Report"
Write-Host "[export] Writing CSV: $env:NCU_OUT_CSV"
# NOTE: -FileEncoding is not a valid Out-File parameter on this host
# (PowerShell 5.1); use -Encoding utf8.
& $ncu --import $Report --page source --csv 2>$null | Out-File -Encoding utf8 $env:NCU_OUT_CSV
if ($LASTEXITCODE -ne 0) {
    Write-Error "ncu exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
$sz = (Get-Item $env:NCU_OUT_CSV).Length
Write-Host "[export] Done. CSV size: $sz bytes"
