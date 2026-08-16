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

Remove-Item Env:COPILOT_* -ErrorAction SilentlyContinue

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
}

$copilotArguments += $forwardedArgs

& copilot @copilotArguments
exit $LASTEXITCODE
