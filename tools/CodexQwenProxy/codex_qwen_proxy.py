#!/usr/bin/env python3
"""Adapt Ollama's Responses endpoint for Codex's V1 namespace tools.

Codex V1 represents collaboration tools as Responses API namespaces.  Ollama's
OpenAI-compatible endpoint accepts ordinary function tools, but returns a
namespaced function name as one flat string.  This proxy flattens namespace
tools on the request and restores ``namespace``/``name`` on the response before
Codex's native tool router sees it.

The proxy deliberately serializes Responses requests.  Qwen is hosted by one
local GPU and concurrent requests are more likely to fail or compete for the
same model than to improve throughput.  A second request waits in a bounded
queue and receives an explicit 503 response if the active request outlives it.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
from ctypes import wintypes
import http.client
import json
import logging
import logging.handlers
import os
import select
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Tuple


LOGGER = logging.getLogger("codex_qwen_proxy")
MAX_REQUEST_BYTES = 64 * 1024 * 1024
REQUEST_LOCK = threading.Lock()

_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 0x00000102
_ERROR_ACCESS_DENIED = 5
_KERNEL32 = None
if os.name == "nt":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _KERNEL32.WaitForSingleObject.restype = wintypes.DWORD
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL


def _namespace_function_name(namespace: str, function_name: str) -> str:
    return namespace + "." + function_name


def _flatten_tools(
    tools: Any,
) -> Tuple[Any, Dict[str, Tuple[str, str]]]:
    if not isinstance(tools, list):
        return tools, {}

    flattened: List[Any] = []
    names: Dict[str, Tuple[str, str]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            flattened.append(tool)
            continue

        namespace = tool.get("name")
        nested_tools = tool.get("tools")
        if not isinstance(namespace, str) or not isinstance(nested_tools, list):
            flattened.append(tool)
            continue

        for nested_tool in nested_tools:
            if not isinstance(nested_tool, dict) or nested_tool.get("type") != "function":
                flattened.append(nested_tool)
                continue

            function_name = nested_tool.get("name")
            if not isinstance(function_name, str):
                flattened.append(nested_tool)
                continue

            flattened_tool = copy.deepcopy(nested_tool)
            flattened_tool["name"] = _namespace_function_name(namespace, function_name)
            # Ollama accepts the standard function fields.  output_schema and
            # defer_loading belong to Codex's namespace extension, not to the
            # OpenAI-compatible function-tool contract.
            flattened_tool.pop("output_schema", None)
            flattened_tool.pop("defer_loading", None)
            flattened.append(flattened_tool)
            names[flattened_tool["name"]] = (namespace, function_name)

    return flattened, names


def _rewrite_input_item(value: Any, namespace_names: Mapping[str, Tuple[str, str]]) -> Any:
    if isinstance(value, list):
        return [_rewrite_input_item(item, namespace_names) for item in value]
    if not isinstance(value, dict):
        return value

    rewritten: Dict[str, Any] = {
        key: _rewrite_input_item(item, namespace_names) for key, item in value.items()
    }
    if rewritten.get("type") in ("function_call", "custom_tool_call"):
        namespace = rewritten.get("namespace")
        name = rewritten.get("name")
        if isinstance(namespace, str) and isinstance(name, str):
            rewritten["name"] = _namespace_function_name(namespace, name)
            rewritten.pop("namespace", None)
        elif isinstance(name, str):
            match = _resolve_namespace_name(name, namespace_names)
            if match is not None:
                rewritten["name"] = _namespace_function_name(match[0], match[1])
    return rewritten


def rewrite_request(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, str]]]:
    rewritten: Dict[str, Any] = copy.deepcopy(dict(payload))
    rewritten_tools, namespace_names = _flatten_tools(rewritten.get("tools"))
    rewritten["tools"] = rewritten_tools
    if "input" in rewritten:
        rewritten["input"] = _rewrite_input_item(rewritten["input"], namespace_names)
    return rewritten, namespace_names


def _restore_namespace(value: Any, namespace_names: Mapping[str, Tuple[str, str]]) -> Any:
    if isinstance(value, list):
        return [_restore_namespace(item, namespace_names) for item in value]
    if not isinstance(value, dict):
        return value

    restored: Dict[str, Any] = {
        key: _restore_namespace(item, namespace_names) for key, item in value.items()
    }
    event_type = restored.get("type")
    is_function_call_event = (
        event_type in ("function_call", "custom_tool_call")
        or isinstance(event_type, str) and event_type.startswith("response.function_call")
    )
    if is_function_call_event and "namespace" not in restored:
        name = restored.get("name")
        if isinstance(name, str):
            match = _resolve_namespace_name(name, namespace_names)
            if match is not None:
                if match[0] == "multi_agent_v1":
                    LOGGER.info("received namespaced tool name=%r mapped=%r", name, match)
                restored["namespace"], restored["name"] = match
    elif is_function_call_event:
        namespace = restored.get("namespace")
        name = restored.get("name")
        if isinstance(namespace, str) and isinstance(name, str):
            match = namespace_names.get(_namespace_function_name(namespace, name))
            if match is not None:
                restored["namespace"], restored["name"] = match
    return restored


def _resolve_namespace_name(
    name: str,
    namespace_names: Mapping[str, Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    match = namespace_names.get(name)
    if match is not None:
        return match

    candidates = [
        match
        for qualified_name, match in namespace_names.items()
        if qualified_name.rsplit(".", 1)[-1] == name
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _rewrite_sse_line(line: bytes, namespace_names: Mapping[str, Tuple[str, str]]) -> bytes:
    if not line.startswith(b"data:"):
        return line
    prefix, separator, data = line.partition(b":")
    if not separator:
        return line
    data = data.lstrip()
    if not data or data == b"[DONE]" or not data.rstrip().endswith(b"}"):
        return line
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return line
    restored = _restore_namespace(payload, namespace_names)
    encoded = json.dumps(restored, separators=(",", ":")).encode("utf-8")
    return prefix + b": " + encoded + (b"\n" if line.endswith(b"\n") else b"")


def _read_request_body(handler: BaseHTTPRequestHandler) -> bytes:
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise ValueError("missing Content-Length")
    length = int(raw_length)
    if length < 0 or length > MAX_REQUEST_BYTES:
        raise ValueError("request body exceeds the configured limit")
    body = handler.rfile.read(length)
    if len(body) != length:
        raise ValueError("request body ended before Content-Length")
    return body


class _ClientDisconnectMonitor:
    """Cancel an upstream generation when Codex closes its HTTP connection."""

    def __init__(self, client_socket: socket.socket, upstream_response: Any):
        self._client_socket = client_socket
        self._upstream_response = upstream_response
        self._stop_event = threading.Event()
        self.disconnected = False
        self._thread = threading.Thread(target=self._run, name="qwen-client-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                readable, _, _ = select.select([self._client_socket], [], [], 0.5)
            except (OSError, ValueError):
                self.disconnected = True
                break
            if not readable:
                continue

            try:
                probe = self._client_socket.recv(1, socket.MSG_PEEK)
            except (BlockingIOError, socket.timeout):
                continue
            except OSError:
                self.disconnected = True
                break
            if not probe:
                self.disconnected = True
                break

            # Unexpected pipelined data is not a disconnect.  Avoid spinning if
            # the client has left data readable while the current response runs.
            self._stop_event.wait(0.5)

        if self.disconnected:
            LOGGER.info("Codex client disconnected; cancelling upstream Responses request")
            try:
                self._upstream_response.close()
            except OSError:
                pass


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CodexQwenProxy/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)

    @property
    def proxy_server(self) -> "CodexQwenServer":
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok", "busy": REQUEST_LOCK.locked()})
            return
        self._proxy_without_rewrite()

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/responses"):
            self._proxy_without_rewrite()
            return

        was_busy = REQUEST_LOCK.locked()
        acquired = REQUEST_LOCK.acquire(timeout=self.proxy_server.busy_timeout)
        if not acquired:
            LOGGER.warning("rejecting Responses request after %.1f seconds in the queue", self.proxy_server.busy_timeout)
            self._send_json(
                503,
                {
                    "error": {
                        "type": "server_error",
                        "code": "qwen_busy",
                        "message": (
                            "QWEN_BUSY: another Ollama Responses request remained active for "
                            + str(self.proxy_server.busy_timeout)
                            + " seconds; retry later."
                        ),
                    }
                },
                extra_headers={"Retry-After": "5"},
            )
            return

        try:
            if was_busy:
                LOGGER.info("processing queued Responses request")
            self._proxy_responses_request()
        finally:
            REQUEST_LOCK.release()

    def _proxy_responses_request(self) -> None:
        try:
            body = _read_request_body(self)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Responses request must be a JSON object")
            rewritten, namespace_names = rewrite_request(payload)
            body = json.dumps(rewritten, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(
                400,
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": "qwen_proxy_request",
                        "message": "Codex Qwen proxy could not rewrite the request: " + str(error),
                    }
                },
            )
            return

        LOGGER.info("forwarding Responses request with %d namespace functions flattened", len(namespace_names))
        try:
            response = self.proxy_server.open_upstream("POST", self.path, body, self.headers)
        except urllib.error.HTTPError as error:
            self._send_upstream_error(error)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            self._send_json(
                502,
                {
                    "error": {
                        "type": "server_error",
                        "code": "qwen_proxy_transport",
                        "message": "QWEN_TRANSPORT_ERROR: " + str(error),
                    }
                },
            )
            return

        content_type = response.headers.get("Content-Type", "application/json")
        monitor = _ClientDisconnectMonitor(self.connection, response)
        monitor.start()
        try:
            if "text/event-stream" in content_type:
                self.send_response(response.status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                for line in response:
                    self.wfile.write(_rewrite_sse_line(line, namespace_names))
                    self.wfile.flush()
                return

            response_body = response.read()
            if monitor.disconnected:
                return
            try:
                response_payload = json.loads(response_body)
                response_body = json.dumps(
                    _restore_namespace(response_payload, namespace_names),
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            self._send_bytes(response.status, response_body, content_type)
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError, http.client.HTTPException) as error:
            if monitor.disconnected:
                LOGGER.info("Responses request cancelled after Codex disconnected: %s", error)
            else:
                LOGGER.warning("Responses forwarding ended before completion: %s", error)
        finally:
            monitor.stop()
            response.close()

    def _proxy_without_rewrite(self) -> None:
        try:
            body = _read_request_body(self) if self.command in ("POST", "PUT", "PATCH") else None
            response = self.proxy_server.open_upstream(self.command, self.path, body, self.headers)
        except urllib.error.HTTPError as error:
            self._send_upstream_error(error)
            return
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
            self._send_json(
                502,
                {
                    "error": {
                        "type": "server_error",
                        "code": "qwen_proxy_transport",
                        "message": "QWEN_TRANSPORT_ERROR: " + str(error),
                    }
                },
            )
            return

        try:
            response_body = response.read()
            content_type = response.headers.get("Content-Type", "application/octet-stream")
        finally:
            response.close()
        self._send_bytes(response.status, response_body, content_type)

    def _send_upstream_error(self, error: urllib.error.HTTPError) -> None:
        try:
            body = error.read()
        finally:
            error.close()
        self._send_bytes(error.code, body, error.headers.get("Content-Type", "application/json"))

    def _send_json(
        self,
        status: int,
        payload: Mapping[str, Any],
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, body, "application/json", extra_headers)

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


class CodexQwenServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: Tuple[str, int], ollama_base_url: str, timeout: float):
        super().__init__(server_address, ProxyHandler)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.timeout = timeout
        self.busy_timeout = 900.0

    def open_upstream(
        self,
        method: str,
        path: str,
        body: Optional[bytes],
        headers: Mapping[str, str],
    ) -> Any:
        url = self.ollama_base_url + path
        forwarded_headers = {
            "Accept": headers.get("Accept", "application/json"),
            "Content-Type": headers.get("Content-Type", "application/json"),
        }
        authorization = headers.get("Authorization")
        if authorization:
            forwarded_headers["Authorization"] = authorization
        request = urllib.request.Request(url, data=body, headers=forwarded_headers, method=method)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=self.timeout)


def _process_is_alive(pid: int) -> bool:
    if _KERNEL32 is not None:
        handle = _KERNEL32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            return _KERNEL32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
        finally:
            _KERNEL32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except (ChildProcessError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _stop_owned_model(model: Optional[str]) -> None:
    if not model:
        return
    try:
        result = subprocess.run(
            ["ollama", "stop", model],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("could not stop owned Ollama model %r: %s", model, error)
        return
    if result.returncode != 0:
        LOGGER.warning("ollama stop for owned model %r exited with %d: %s", model, result.returncode, result.stderr.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--busy-timeout", type=float, default=900.0)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--parent-check-interval", type=float, default=1.0)
    parser.add_argument("--cleanup-model")
    parser.add_argument("--log-file")
    args = parser.parse_args()

    if args.parent_pid is not None and args.parent_pid <= 0:
        parser.error("--parent-pid must be positive")
    if args.parent_check_interval <= 0:
        parser.error("--parent-check-interval must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.log_file:
        file_handler = logging.handlers.WatchedFileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(file_handler)
    server = CodexQwenServer((args.host, args.port), args.ollama_base_url, args.timeout)
    server.busy_timeout = args.busy_timeout
    LOGGER.info("listening on http://%s:%d; forwarding to %s", args.host, args.port, args.ollama_base_url)
    if args.parent_pid is not None and not _process_is_alive(args.parent_pid):
        LOGGER.info("parent PID %d is not alive; exiting without serving", args.parent_pid)
        server.server_close()
        _stop_owned_model(args.cleanup_model)
        return 0

    serving_thread: Optional[threading.Thread] = None
    try:
        if args.parent_pid is None:
            server.serve_forever()
        else:
            serving_thread = threading.Thread(target=server.serve_forever, name="qwen-proxy-server")
            serving_thread.start()
            while _process_is_alive(args.parent_pid):
                time.sleep(args.parent_check_interval)
            LOGGER.info("parent PID %d exited; shutting down", args.parent_pid)
            server.shutdown()
            serving_thread.join()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")
        if serving_thread is not None:
            server.shutdown()
    finally:
        server.server_close()
        _stop_owned_model(args.cleanup_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
