[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CodexArguments
)

$ErrorActionPreference = 'Stop'

function Stop-ManagedQwenModel {
    param([string]$ModelName)

    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        & ollama stop $ModelName | Out-Null
        $loaded = @(ollama ps | Select-String -SimpleMatch $ModelName)
        if ($loaded.Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Warning "Ollama still reports $ModelName after cleanup attempts."
}

$proxyLauncher = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'start-codex-qwen-proxy.ps1'
$proxyPort = 11435
$qwenModel = 'qwen3.8-copilot-160k-codex:latest'
$listeners = @(Get-NetTCPConnection -LocalPort $proxyPort -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    throw "Codex Qwen proxy is already listening on port $proxyPort. Stop that instance before starting a managed session."
}
$modelWasLoaded = @(ollama ps | Select-String -SimpleMatch $qwenModel).Count -gt 0
$startedByThisScript = $false
$exitCode = 1

try {
    if ($listeners.Count -eq 0) {
        if (-not $modelWasLoaded) {
            & $proxyLauncher -Background -ParentPid $PID -CleanupModel $qwenModel
        } else {
            & $proxyLauncher -Background -ParentPid $PID
        }
        $startedByThisScript = $true
    }

    & codex --profile ollama-launch @CodexArguments
    $exitCode = $LASTEXITCODE
} finally {
    if ($startedByThisScript) {
        & $proxyLauncher -Stop
    }
    if ($startedByThisScript -and -not $modelWasLoaded) {
        Stop-ManagedQwenModel $qwenModel
    }
}

exit $exitCode
