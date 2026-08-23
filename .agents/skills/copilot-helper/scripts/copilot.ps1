# Configure the local Ollama-backed CLI and provide a Codex-oriented profile
# without changing the default manual invocation.
#
# Launches one of two CLIs, both pinned to the same local Qwen model
# (qwen3.8-copilot-160k served by Ollama at 127.0.0.1:11434/v1):
#   copilot   (default)  GitHub Copilot CLI, configured via COPILOT_* variables
#   opencode            OpenCode CLI, configured via a machine-scoped config
#                       generated once by this helper
#
# Selection flags (consumed here, never forwarded to the CLI):
#   --copilot, -c, --cli      run the Copilot CLI (default)
#   --opencode, -o, --oc      run the OpenCode CLI
#
# --codex remains: the Codex secondary-review profile.  It implies the
# Copilot CLI runner plus its permission grants, or --auto on the OpenCode
# runner (the closest equivalent; the review prompt must still prohibit
# edits and commits).

$runner = $null
$codexMode = $false
$forwardedArgs = @()

foreach ($argument in $args) {
    if ($argument -eq "--codex" -or $argument -eq "-Codex") {
        $codexMode = $true
        $runner = "copilot"
        continue
    }

    if ($argument -eq "--copilot" -or $argument -eq "-c" -or $argument -eq "--cli") {
        $runner = "copilot"
        continue
    }

    if ($argument -eq "--opencode" -or $argument -eq "-o" -or $argument -eq "--oc") {
        $runner = "opencode"
        continue
    }

    $forwardedArgs += $argument
}

# Default keeps the existing non-interactive Copilot CLI contract; OpenCode
# is opt-in via --opencode and starts interactively when no prompt is given.
if ($null -eq $runner) {
    $runner = "copilot"
}

# Shared local Qwen model settings, the single source of truth for both
# runner branches.
$modelBase = "http://127.0.0.1:11434/v1"
$modelApiKey = "ollama"
$modelId = "qwen3.8-copilot-160k"
$modelDisplay = "Qwen 3.8 Copilot 160k (local)"
$maxPromptTokens = "150000"
$maxOutputTokens = "8192"

if ($runner -eq "copilot") {
    # Do not use the Env: provider with a wildcard here.  Some Windows hosts
    # expose both `Path` and `PATH`; PowerShell then fails while constructing
    # the provider's dynamic parameter set before it ever reaches the
    # COPILOT_* names.
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
    $env:COPILOT_PROVIDER_BASE_URL = $modelBase
    $env:COPILOT_PROVIDER_API_KEY = $modelApiKey
    $env:COPILOT_PROVIDER_WIRE_API = "completions"

    $env:COPILOT_MODEL = $modelId
    $env:COPILOT_PROVIDER_MAX_PROMPT_TOKENS = $maxPromptTokens
    $env:COPILOT_PROVIDER_MAX_OUTPUT_TOKENS = $maxOutputTokens

    # Spill large tool output to files sooner instead of stuffing
    # all of it into model context.
    $env:COPILOT_LARGE_OUTPUT_THRESHOLD_BYTES = $maxOutputTokens

    $copilotArguments = @(
        "--no-ask-user",
        "--reasoning-effort",
        "max"
    )

    if ($codexMode) {
        $copilotArguments += "-s"
        # Codex invokes this profile for an explicitly requested secondary
        # review.  Noninteractive mode cannot answer Copilot's path/tool
        # permission prompts, so grant the review process access up front;
        # the review prompt must still explicitly prohibit edits and commits.
        $copilotArguments += "--allow-all-paths"
        $copilotArguments += "--allow-all-tools"
    }

    $copilotArguments += $forwardedArgs

    if ($null -eq (Get-Command copilot -ErrorAction SilentlyContinue)) {
        throw "'copilot' was not found on PATH.  Install the GitHub Copilot CLI first."
    }

    & copilot @copilotArguments
    exit $LASTEXITCODE
}

if ($runner -eq "opencode") {
    if ($null -eq (Get-Command opencode -ErrorAction SilentlyContinue)) {
        throw "'opencode' was not found on PATH.  Install it with 'npm i -g opencode-ai'."
    }

    # Register the local Qwen model with OpenCode via a machine-scoped config
    # generated once by this helper.  It is kept when present so hand edits
    # are preserved.  Both the interactive TUI and `opencode run` resolve the
    # same model through OPENCODE_CONFIG.
    $configRoot = Join-Path $env:USERPROFILE ".config\opencode"
    if (-not (Test-Path -LiteralPath $configRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $configRoot -Force | Out-Null
    }
    $configPath = Join-Path $configRoot "copilot-local-qwen.jsonc"

    if (-not (Test-Path -LiteralPath $configPath)) {
        $providerId = "local-qwen"
        # Single-quoted template: no PowerShell expansion, then substitute
        # the shared settings with plain placeholders.
        $configTemplate = @'
{
  "$schema": "https://opencode.ai/config.json",
  "model": "__PROVIDER__/__MODEL__",
  "provider": {
    "__PROVIDER__": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "__DISPLAY__",
      "options": {
        "baseURL": "__BASE__",
        "apiKey": "__APIKEY__"
      },
      "models": {
        "__MODEL__": { "name": "__DISPLAY__" }
      }
    }
  }
}
'@
        $configContent = $configTemplate.Replace("__PROVIDER__", $providerId)
        $configContent = $configContent.Replace("__MODEL__", $modelId)
        $configContent = $configContent.Replace("__DISPLAY__", $modelDisplay)
        $configContent = $configContent.Replace("__BASE__", $modelBase)
        $configContent = $configContent.Replace("__APIKEY__", $modelApiKey)
        Set-Content -LiteralPath $configPath -Value $configContent -Encoding utf8NoBOM
    }

    $env:OPENCODE_CONFIG = $configPath

    # Note: the Copilot branch injects "--reasoning-effort max"; the OpenCode
    # equivalent is a provider-specific "--variant" whose valid values depend
    # on the provider and are not verified for this endpoint.  Pass "--variant
    # <value>" explicitly if a particular reasoning budget is required.
    # OpenCode's top-level command starts the interactive TUI.  Use `run`
    # only when the caller supplied a prompt/command or requested codex mode.
    $opencodeArguments = @()
    if ($codexMode -or $forwardedArgs.Count -gt 0) {
        $opencodeArguments += "run"

        if ($codexMode) {
            # --codex under the OpenCode runner maps to --auto, the closest
            # equivalent of Copilot's non-interactive permission grants.
            $opencodeArguments += "--auto"
            Write-Warning "--codex under the OpenCode runner maps to --auto only."
        }

        $opencodeArguments += $forwardedArgs
    }

    & opencode @opencodeArguments
    exit $LASTEXITCODE
}

throw "Unsupported runner selection: $runner.  Use --copilot or --opencode."
