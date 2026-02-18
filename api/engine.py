# api/engine.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

# Path to llama-cli binary on Marshall's server
LLAMA_BIN = (
    Path.home()
    / "implementation"
    / "server-core"
    / "llama.cpp"
    / "build"
    / "bin"
    / "llama-cli"
)

SYSTEM_PROMPT = "You are a helpful assistant."

LOG_OUTPUT = "/home/alspec/implementation/server-core/logs/log_$(date +%Y%m%d_%H%M%S)_${MODEL}.txt"

MODEL_PATH = (
    Path.home()
    / "implementation"
    / "server-core"
    / "model_storage"
    / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
)


def build_cmd(prompt: str) -> list[str]:
    """
    Build the llama-cli command for a *single* user turn.

    This matches what CovalentCarbon described:
      - use --system-prompt
      - send the user text via --prompt
      - add -no-cnv so llama.cpp doesn't enter its chat UI

    We run one process per request, so no REPL / prompt parsing needed.
    """
    if not LLAMA_BIN.is_file():
        raise RuntimeError(f"llama-cli not found at {LLAMA_BIN}")
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")

    prompt = prompt.strip() or "Hello from ALSPEC project"

    return [
        str(LLAMA_BIN),
        "-m",
        str(MODEL_PATH),
        "--system-prompt",
        SYSTEM_PROMPT,
        "--prompt",
        prompt,
        "-n",
        "256",
        "--split-mode",
        "row",
        "--main-gpu",
        "0",
        "-no-cnv" 
    ]


async def send_and_stream(user_msg: str) -> AsyncGenerator[str, None]:
    """
    Run llama-cli once for this user_msg and stream its stdout.

    The WebSocket layer (api/server.py) will:
      - call this generator
      - forward chunks to the client
      - then send [[END_OF_RESPONSE]] when we're done
    """
    cmd = build_cmd(user_msg)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert proc.stdout is not None

    # Stream stdout as it is produced
    while True:
        chunk = await proc.stdout.read(256)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="ignore")
        if text:
            yield text

    # Optional: if you want to debug later, you can read stderr here.
    await proc.wait()
