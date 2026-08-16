$helperPath = Join-Path $PSScriptRoot ".agents\skills\copilot-helper\scripts\copilot.ps1"

if (-not (Test-Path -LiteralPath $helperPath -PathType Leaf)) {
    throw "Copilot helper was not found at $helperPath"
}

& $helperPath @args
exit $LASTEXITCODE
