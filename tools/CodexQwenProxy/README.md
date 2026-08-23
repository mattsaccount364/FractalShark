# Native Codex/Qwen bridge

This helper keeps Codex's native V1 subagent lifecycle while adapting the
OpenAI Responses namespace format used by Codex to Ollama's flat function-call
format. It is used by the user-global `ollama-launch` profile and does not
run a second CLI; it is the Codex-to-Ollama compatibility layer.

Use the session-scoped wrapper from the repository root (recommended):

```powershell
.\tools\CodexQwenProxy\run-codex-qwen.ps1
```

It starts the proxy only for that Codex session and stops it when Codex exits.
Arguments after `--` are passed to Codex, for example:

```powershell
.\tools\CodexQwenProxy\run-codex-qwen.ps1 -- exec --sandbox read-only "Review the current diff."
```

The lower-level launcher runs in the foreground by default. Background mode
requires an explicit parent PID, so a detached proxy cannot be started without
a lifetime owner. Use `-Stop` only to clean up an older or separately managed
instance.

The proxy serializes `/v1/responses` requests because the local Qwen model is
single-GPU. A request arriving while another request is active waits for up to
15 minutes; if that limit is reached, Codex receives a clear `QWEN_BUSY` error.
The proxy also watches the Codex connection and cancels the upstream Ollama
request when Codex disconnects. The managed proxy also watches its wrapper PID
and exits if that owner disappears unexpectedly.

The model is bounded at 163,840 runtime tokens, with Codex compaction at
140,000 tokens, 95% usable-window headroom, and an 8,192-byte tool truncation
limit. The generated user-global files are:

- `%USERPROFILE%\.codex\ollama-launch.config.toml`
- `%USERPROFILE%\.codex\model.json`
- `%USERPROFILE%\.codex\agents\qwen_reviewer.toml`

The OpenAI-parent/mixed-provider arrangement is not enabled: the installed
Codex release routes spawned children through the parent ChatGPT provider.
Use the Ollama profile for a native Qwen V1 parent/child tree.
