---
name: copilot-helper
description: Use the repository's local GitHub Copilot CLI backed by Ollama as an independent secondary reviewer for code exploration, algorithm and CUDA reasoning, debugging, test design, and diff review. Use when another opinion would materially improve a difficult investigation or proposed change.
---

# Copilot helper

Use Copilot as an independent, advisory reviewer. Codex remains responsible for
the investigation, architecture, edits, decisions, and validation.

## Invocation

Run from the repository root through the compatibility wrapper:

```powershell
.\copilot.ps1 --codex -p "<PROMPT>"
```

The wrapper configures the local Ollama endpoint and model, adds the silent
Codex profile, and forwards the remaining Copilot arguments unchanged. Use
`-p` for non-interactive prompts. The canonical implementation is in
`.agents/skills/copilot-helper/scripts/copilot.ps1`.

## Workflow

1. Investigate the task independently and record the evidence and current
   hypothesis.
2. Ask Copilot to inspect the relevant files or problem independently. Give it
   raw context and ask it not to edit files.
3. Compare its claims with the evidence. Reproduce or investigate every
   material disagreement rather than accepting either answer automatically.
4. Implement the chosen change yourself and keep Copilot out of the write path.
5. Ask Copilot to review the resulting diff for correctness, races, CUDA
   hazards, regressions, and missing tests.
6. Run the repository's validation yourself and report which Copilot findings
   were confirmed, rejected, or left uncertain.

## Prompt guidance

Prefer prompts that request evidence and alternatives, for example:

```text
Inspect the Reference2Gpu radix-stage implementation and its callers. Do not
edit files. Trace the relevant call chain, identify likely performance or
correctness risks, cite the files and symbols involved, and propose focused
tests or measurements.
```

For a diff review, provide the specific files or diff and ask for independent
criticism rather than asking Copilot to apply a patch.

Do not include credentials, private tokens, or unrelated large output. Treat
Copilot output as advice, not authority. If the local CLI or Ollama service is
unavailable, continue the investigation without it.

## Long-running jobs

The normal `--codex` profile adds a silent mode and should be used directly only
for short, machine-readable checks. The local Ollama-backed model may take
several minutes to produce a final response, so no immediate output does not
mean that the run is stuck or unusable. For a large investigation, request
incremental output and diagnostic logs explicitly:

```powershell
$logDirectory = Join-Path $env:TEMP "copilot-helper-logs"
New-Item -ItemType Directory -Force $logDirectory | Out-Null
.\copilot.ps1 --codex --stream on --log-level info --log-dir $logDirectory `
    -p "<PROMPT>"
```

Use a generous outer execution timeout, 30 minutes by default for a substantial
review, and wait or poll for completion. From a second PowerShell window, use
the following checks while the command is running:

```powershell
$logDirectory = Join-Path $env:TEMP "copilot-helper-logs"
Get-Process -Name copilot,ollama -ErrorAction SilentlyContinue |
    Select-Object ProcessName,Id,StartTime,CPU
ollama ps
Get-ChildItem -LiteralPath $logDirectory -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 Name,Length,LastWriteTime
Get-Content -LiteralPath "<active-log-file>" -Wait
```

Streamed output, growing logs, recent process CPU activity, or a live Copilot
or Ollama process are useful liveness signals. `ollama ps` confirms model
availability but does not provide an exact token-level completion percentage.
Do not discard a delayed response or kill a run solely because it exceeds a
short default timeout. Only use the repository's `.\copilot_stop.ps1` helper
for an intentional cancellation or after repeated liveness checks show that
the run is genuinely stuck. If the process exits with an actual error, record
the failure and continue the investigation without Copilot.
