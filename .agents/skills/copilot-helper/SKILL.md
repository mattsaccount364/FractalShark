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
.\copilot.ps1 -p "<PROMPT>"
```

The wrapper configures the local Ollama endpoint and model, checks `ollama ps`,
scopes file access to the repository, disables write and shell tools, and
forwards the response. Use `-p` for non-interactive prompts. The canonical
implementation is in
`.agents/skills/copilot-helper/scripts/copilot.ps1`.

On Windows, a prompt containing spaces should be supplied through the wrapper's
file form so command-line quoting cannot split it:

```powershell
.\copilot.ps1 --prompt-file <prompt-file>
```

The wrapper reads that file and passes the prompt as one process argument. Do
not invoke `copilot.exe` directly for a review. If the wrapper reports an
existing Ollama model, stop that model and retry.

## Workflow

1. Investigate the task independently and record the evidence and current
   hypothesis.
2. Ask Copilot to inspect the relevant files or problem independently. Give it
   raw context; the wrapper supplies the read-only reviewer instruction.
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
Inspect the Reference2Gpu radix-stage implementation and its callers. Trace the
relevant call chain, identify likely performance or correctness risks, cite the
files and symbols involved, and propose focused tests or measurements.
```

For a diff review, provide the specific files or diff and ask for independent
criticism rather than asking Copilot to apply a patch.

Do not include credentials, private tokens, or unrelated large output. Treat
Copilot output as advice, not authority. If the local CLI or Ollama service is
unavailable, continue the investigation without it.

## Runtime behavior

The wrapper streams the response, forwards the Copilot exit code, and rejects a
busy Ollama instance before starting another review. Copilot output remains
advisory; verify its claims independently and continue without it if the local
service is unavailable.
