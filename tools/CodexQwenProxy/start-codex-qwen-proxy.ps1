[CmdletBinding()]
param(
    [switch]$Background,
    [switch]$Foreground,
    [int]$ParentPid,
    [string]$CleanupModel,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'

if ($Background -and $Foreground) {
    throw 'Choose either -Background or -Foreground, not both.'
}

if ($Background -and $ParentPid -le 0) {
    throw 'Background mode requires -ParentPid; use run-codex-qwen.ps1 for managed sessions.'
}

$proxyDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$proxyScript = Join-Path $proxyDirectory 'codex_qwen_proxy.py'
$proxyHost = '127.0.0.1'
$proxyPort = 11435
$healthUri = "http://$proxyHost`:$proxyPort/healthz"

$listeners = @(Get-NetTCPConnection -LocalPort $proxyPort -State Listen -ErrorAction SilentlyContinue)

if ($Stop) {
    foreach ($listener in $listeners) {
        Stop-Process -Id $listener.OwningProcess -Force
    }
    if ($listeners.Count -eq 0) {
        Write-Output "Codex Qwen proxy was not running."
    } else {
        Write-Output "Stopped Codex Qwen proxy on port $proxyPort."
    }
    exit 0
}

if ($listeners.Count -gt 0) {
    Write-Output "Codex Qwen proxy is already listening on $proxyHost`:$proxyPort."
    exit 0
}

$pythonLauncher = (Get-Command py -ErrorAction Stop).Source
$logPath = Join-Path $env:TEMP 'codex-qwen-proxy.log'
$arguments = @(
    '-3'
    $proxyScript
    '--host'
    $proxyHost
    '--port'
    $proxyPort
    '--ollama-base-url'
    'http://127.0.0.1:11434'
    '--log-file'
    $logPath
)

if ($ParentPid -gt 0) {
    $arguments += @('--parent-pid', $ParentPid)
}

if ($CleanupModel) {
    $arguments += @('--cleanup-model', $CleanupModel)
}

if ($Foreground -or -not $Background) {
    & $pythonLauncher @arguments
    exit $LASTEXITCODE
}

$process = Start-Process -FilePath $pythonLauncher -ArgumentList $arguments -WindowStyle Hidden -PassThru
Start-Sleep -Milliseconds 500
try {
    $health = Invoke-RestMethod -Uri $healthUri -Method Get -TimeoutSec 5
} catch {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    throw "Codex Qwen proxy failed to start. See $logPath. $($_.Exception.Message)"
}

Write-Output "Started Codex Qwen proxy (PID $($process.Id)) on $proxyHost`:$proxyPort. Log: $logPath"
