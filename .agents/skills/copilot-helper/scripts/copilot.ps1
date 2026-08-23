# Configure the local Ollama-backed Copilot CLI for unattended, read-only
# secondary reviews. The legacy --codex switch is accepted for compatibility,
# but it is no longer required by callers.

$forwardedArgs = @()
$promptFile = $null
$promptText = $null
$promptWasProvided = $false

for ($argumentIndex = 0; $argumentIndex -lt $args.Count; ++$argumentIndex) {
    $argument = [string]$args[$argumentIndex]
    if ($argument -eq "--codex" -or $argument -eq "-Codex") {
        continue
    }

    if ($argument -eq "--prompt-file") {
        if ($argumentIndex + 1 -ge $args.Count) {
            throw "--prompt-file requires a path"
        }
        $promptFile = $args[++$argumentIndex]
        continue
    }

    if ($argument -eq "--prompt" -or $argument -eq "-p") {
        if ($argumentIndex + 1 -ge $args.Count) {
            throw "$argument requires a prompt"
        }

        $promptParts = [System.Collections.Generic.List[string]]::new()
        $promptParts.Add([string]$args[++$argumentIndex])
        while ($argumentIndex + 1 -lt $args.Count) {
            $promptParts.Add([string]$args[++$argumentIndex])
        }

        $promptText = $promptParts -join ' '
        $promptWasProvided = $true
        break
    }

    if ($argument.StartsWith("--prompt=", [System.StringComparison]::Ordinal)) {
        $promptParts = [System.Collections.Generic.List[string]]::new()
        $promptParts.Add($argument.Substring("--prompt=".Length))
        while ($argumentIndex + 1 -lt $args.Count) {
            $promptParts.Add([string]$args[++$argumentIndex])
        }

        $promptText = $promptParts -join ' '
        $promptWasProvided = $true
        break
    }

    $forwardedArgs += $argument
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path

# Do not use the Env: provider with a wildcard here.  Some Windows hosts
# expose both `Path` and `PATH`; PowerShell then fails while constructing the
# provider's dynamic parameter set before it ever reaches the COPILOT_* names.
foreach ($variableName in @(
        "COPILOT_PROVIDER_TYPE",
        "COPILOT_PROVIDER_BASE_URL",
        "COPILOT_PROVIDER_API_KEY",
        "COPILOT_PROVIDER_WIRE_API",
        "COPILOT_MODEL",
        "COPILOT_PROVIDER_MAX_PROMPT_TOKENS",
        "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS",
        "COPILOT_LARGE_OUTPUT_THRESHOLD_BYTES"
    )) {
    [Environment]::SetEnvironmentVariable($variableName, $null, "Process")
}

function Get-ApplicationPath {
    param(
        [Parameter(Mandatory)]
        [string[]]$CommandNames
    )

    foreach ($commandName in $CommandNames) {
        $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue
        if ($command -and (Test-Path -LiteralPath $command.Source -PathType Leaf)) {
            return $command.Source
        }
    }

    throw "Unable to find any of these applications on PATH: $($CommandNames -join ', ')"
}

function Test-OllamaReady {
    param(
        [Parameter(Mandatory)]
        [string]$OllamaPath
    )

    $ollamaOutput = @(& $OllamaPath ps 2>&1 | ForEach-Object { [string]$_ })
    $ollamaExitCode = $LASTEXITCODE
    if ($ollamaExitCode -ne 0) {
        $details = $ollamaOutput -join [Environment]::NewLine
        throw "Unable to query Ollama with 'ollama ps' (exit code $ollamaExitCode). Is Ollama running? $details"
    }

    $nonEmptyLines = @($ollamaOutput | Where-Object { $_.Trim().Length -gt 0 })
    $modelLines = @()
    if ($nonEmptyLines.Count -gt 0) {
        if ($nonEmptyLines[0] -match '^\s*NAME\s+ID\s+') {
            $modelLines = @($nonEmptyLines | Select-Object -Skip 1)
        } else {
            $modelLines = $nonEmptyLines
        }
    }

    if ($modelLines.Count -gt 0) {
        throw "Ollama already reports a running model. Stop it before starting another Qwen review, then retry:`n$($modelLines -join [Environment]::NewLine)"
    }
}

$env:COPILOT_PROVIDER_TYPE = "openai"
$env:COPILOT_PROVIDER_BASE_URL = "http://127.0.0.1:11434/v1"
$env:COPILOT_PROVIDER_API_KEY = "ollama"
$env:COPILOT_PROVIDER_WIRE_API = "completions"

$env:COPILOT_MODEL = "qwen3.8-copilot-160k"
$env:COPILOT_PROVIDER_MAX_PROMPT_TOKENS = "150000"
$env:COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = "8192"

# Spill large tool output to files sooner instead of stuffing
# all of it into model context.
$env:COPILOT_LARGE_OUTPUT_THRESHOLD_BYTES = "8192"

$copilotArguments = @(
    "-C", $repositoryRoot,
    "--add-dir", $repositoryRoot,
    "--disable-builtin-mcps",
    "--deny-tool=write",
    "--deny-tool=shell",
    "--no-ask-user",
    "--reasoning-effort", "max",
    "--stream", "on",
    "-s"
)

$copilotArguments += $forwardedArgs

if ($null -ne $promptFile) {
    $promptText = Get-Content -LiteralPath $promptFile -Raw
    $promptWasProvided = $true
}

if ($promptWasProvided) {
    $reviewPrompt = @"
Act as an advisory, read-only secondary reviewer. Inspect files and run
non-destructive checks when useful, but do not edit files, commit, push, or run
destructive commands. Report evidence and recommendations to the caller.

User request:
$promptText
"@
    $copilotArguments += "--prompt"
    $copilotArguments += $reviewPrompt
}

$ollamaCommand = Get-ApplicationPath -CommandNames @('ollama.exe', 'ollama')
Test-OllamaReady -OllamaPath $ollamaCommand

$copilotCommand = Get-ApplicationPath -CommandNames @('copilot.exe', 'copilot')

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $copilotCommand
$startInfo.WorkingDirectory = $repositoryRoot
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
foreach ($argument in $copilotArguments) {
    [void]$startInfo.ArgumentList.Add([string]$argument)
}

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
try {
    if (-not $process.Start()) {
        throw "Process.Start returned false"
    }

    # Read both streams asynchronously so a verbose model/tool trace cannot
    # deadlock the wrapper while preserving the CLI output for the caller.
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    [Console]::Write($stdoutTask.GetAwaiter().GetResult())
    [Console]::Error.Write($stderrTask.GetAwaiter().GetResult())
    $exitCode = $process.ExitCode
} catch {
    throw "Failed to start $($copilotCommand): $($_.Exception.Message)"
} finally {
    $process.Dispose()
}

if ($null -eq $exitCode) {
    $exitCode = 0
}

exit $exitCode
