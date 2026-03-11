from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModeConfig:
    profile: str
    tool_name: str
    description: str


MODE_CONFIGS: dict[str, ModeConfig] = {
    "high": ModeConfig(
        profile="high",
        tool_name="codex_high",
        description="Run Codex with the high reasoning profile.",
    ),
    "xhigh": ModeConfig(
        profile="xhigh",
        tool_name="codex_xhigh",
        description="Run Codex with the xhigh reasoning profile.",
    ),
    "high_fast": ModeConfig(
        profile="high_fast",
        tool_name="codex_high_fast",
        description="Run Codex with the high_fast profile.",
    ),
    "xhigh_fast": ModeConfig(
        profile="xhigh_fast",
        tool_name="codex_xhigh_fast",
        description="Run Codex with the xhigh_fast profile.",
    ),
}

TOOL_TO_MODE = {config.tool_name: mode for mode, config in MODE_CONFIGS.items()}
DEFAULT_TIMEOUT_S = 600


def build_codex_command(mode: str, prompt: str, cwd: str | None = None) -> list[str]:
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}")
    command = [
        "codex",
        "exec",
        "-p",
        MODE_CONFIGS[mode].profile,
        "--ephemeral",
        "--skip-git-repo-check",
        prompt,
    ]
    if cwd:
        command[2:2] = ["-C", cwd]
    return command


def build_subprocess_kwargs(cwd: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": DEFAULT_TIMEOUT_S,
    }
    if cwd:
        kwargs["cwd"] = cwd
    return kwargs


def run_codex(mode: str, prompt: str, cwd: str | None = None) -> dict[str, Any]:
    command = build_codex_command(mode, prompt, cwd=cwd)
    completed = subprocess.run(command, **build_subprocess_kwargs(cwd))
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return {
        "mode": mode,
        "profile": MODE_CONFIGS[mode].profile,
        "command": command,
        "cwd": cwd or os.getcwd(),
        "exit_code": int(completed.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "ok": completed.returncode == 0,
    }


def build_tool_definitions() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for config in MODE_CONFIGS.values():
        tools.append(
            {
                "name": config.tool_name,
                "description": config.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Instructions to send to Codex.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory for the Codex run.",
                        },
                    },
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            }
        )
    return tools


def dispatch_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    mode = TOOL_TO_MODE.get(tool_name)
    if mode is None:
        raise ValueError(f"Unknown tool: {tool_name}")

    prompt = arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    cwd = arguments.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("cwd must be a string when provided")

    return run_codex(mode, prompt, cwd=cwd)


def _json_rpc_success(result: dict[str, Any], request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _json_rpc_error(code: int, message: str, request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _encode_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload["stdout"] if payload["stdout"] else payload["stderr"]
    if not text:
        text = "(no output)"
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
        "isError": not payload["ok"],
    }


def _handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params", {})

    if method == "initialize":
        return _json_rpc_success(
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "codex-mcp-router", "version": "0.1.0"},
            },
            request_id,
        )

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return _json_rpc_success({}, request_id)

    if method == "tools/list":
        return _json_rpc_success({"tools": build_tool_definitions()}, request_id)

    if method == "tools/call":
        try:
            tool_name = params["name"]
            arguments = params.get("arguments", {})
            payload = dispatch_tool_call(tool_name, arguments)
            return _json_rpc_success(_encode_tool_result(payload), request_id)
        except Exception as exc:  # pragma: no cover - exercised through CLI smoke path
            return _json_rpc_success(
                {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
                request_id,
            )

    if request_id is None:
        return None
    return _json_rpc_error(-32601, f"Method not found: {method}", request_id)


def _read_message(stream) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line == b"\r\n":
            break
        key, _, value = line.decode("utf-8").partition(":")
        headers[key.strip().lower()] = value.strip()

    content_length = int(headers["content-length"])
    body = stream.read(content_length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(stream, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
    stream.write(header)
    stream.write(body)
    stream.flush()


def serve_stdio() -> int:
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    while True:
        message = _read_message(input_stream)
        if message is None:
            return 0
        response = _handle_request(message)
        if response is not None:
            _write_message(output_stream, response)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route four fixed Codex profiles through MCP stdio.")
    parser.add_argument(
        "--smoke-test",
        choices=sorted(MODE_CONFIGS),
        help="Run one local Codex invocation instead of starting the MCP stdio server.",
    )
    parser.add_argument(
        "--cwd",
        help="Optional working directory for --smoke-test runs.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send during --smoke-test.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.smoke_test:
        if not args.prompt:
            raise SystemExit("--smoke-test requires a prompt")
        result = run_codex(args.smoke_test, args.prompt, cwd=args.cwd)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["ok"] else 1
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
