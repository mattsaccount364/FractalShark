# Configure the local Ollama-backed Copilot CLI and provide a Codex-oriented
# profile without changing the default manual invocation.

$codexMode = $false
$forwardedArgs = @()
$promptFile = $null

for ($argumentIndex = 0; $argumentIndex -lt $args.Count; ++$argumentIndex) {
    $argument = $args[$argumentIndex]
    if ($argument -eq "--codex" -or $argument -eq "-Codex") {
        $codexMode = $true
        continue
    }

    if ($argument -eq "--prompt-file") {
        if ($argumentIndex + 1 -ge $args.Count) {
            throw "--prompt-file requires a path"
        }
        $promptFile = $args[++$argumentIndex]
        continue
    }

    $forwardedArgs += $argument
}

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
    "--no-ask-user",
    "--reasoning-effort",
    "max"
)

if ($codexMode) {
    $copilotArguments += "-s"
    # Codex invokes this profile for an explicitly requested secondary review.
    # Noninteractive mode cannot answer Copilot's path/tool permission prompts,
    # so grant the review process access up front; the review prompt must still
    # explicitly prohibit edits and commits.
    $copilotArguments += "--allow-all-paths"
    $copilotArguments += "--allow-all-tools"
}

$copilotArguments += $forwardedArgs
if ($null -ne $promptFile) {
    $promptText = Get-Content -LiteralPath $promptFile -Raw
    $promptArgumentIndex = [Array]::IndexOf($copilotArguments, "--prompt")
    if ($promptArgumentIndex -ge 0) {
        $copilotArguments[$promptArgumentIndex + 1] = $promptText
    } else {
        $copilotArguments += "--prompt"
        $copilotArguments += $promptText
    }
}

$copilotCommand = (Get-Command copilot.exe -CommandType Application -ErrorAction Stop).Source
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $copilotCommand
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
foreach ($argument in $copilotArguments) {
    [void]$startInfo.ArgumentList.Add([string]$argument)
}

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
if (-not $process.Start()) {
    throw "Failed to start $copilotCommand"
}

# Read both streams asynchronously so a verbose model/tool trace cannot deadlock the
# review process while preserving the CLI's output for the caller.
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
$process.WaitForExit()
[Console]::Write($stdoutTask.GetAwaiter().GetResult())
[Console]::Error.Write($stderrTask.GetAwaiter().GetResult())
$exitCode = $process.ExitCode
$process.Dispose()
exit $exitCode
