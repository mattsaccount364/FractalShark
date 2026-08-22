# Configure the local Ollama-backed Copilot CLI and provide a Codex-oriented
# profile without changing the default manual invocation.

$codexMode = $false
$forwardedArgs = @()

foreach ($argument in $args) {
    if ($argument -eq "--codex" -or $argument -eq "-Codex") {
        $codexMode = $true
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

& copilot @copilotArguments
exit $LASTEXITCODE
