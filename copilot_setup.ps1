@"
FROM qwen3.8:latest
PARAMETER num_ctx 163840
"@ | Set-Content "$env:TEMP\Modelfile.qwen38-copilot"

ollama create qwen3.8-copilot-160k -f "$env:TEMP\Modelfile.qwen38-copilot"
ollama serve qwen3.8-copilot-160k
