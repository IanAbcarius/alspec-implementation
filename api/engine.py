# api/engine.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

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

# GLOBAL persistent llama process
_llama_proc: Optional[asyncio.subprocess.Process] = None


async def start_llama_if_needed() -> None:
    """
    Start the llama-cli persistent REPL if not already running.

    We intentionally use llama.cpp's built-in chat interface:
        ./llama-cli -m <model> --system-prompt "..."
    so it remembers conversation history and keeps a KV cache alive.
    """
    global _llama_proc
    if _llama_proc is not None:
        return

    model_path = (
        Path.home()
        / "implementation"
        / "server-core"
        / "model_storage"
        / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    )

    if not LLAMA_BIN.is_file():
        raise RuntimeError(f"llama-cli not found at {LLAMA_BIN}")
    if not model_path.is_file():
        raise RuntimeError(f"Model file not found at {model_path}")

    cmd = [
        str(LLAMA_BIN),
        "-m",
        str(model_path),
        "--system-prompt",
        SYSTEM_PROMPT,
        "-n",
        "256",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
    ]

    _llama_proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Give llama.cpp a moment to print its banner / warmup.
    # We don't try to consume it here; the first user message
    # will flush any remaining startup text.
    await asyncio.sleep(1.0)


async def send_and_stream(user_msg: str) -> AsyncGenerator[str, None]:
    """
    Send a user message into the running llama-cli REPL and stream output.

    We write the text to stdin, then read stdout until we see a new prompt
    ("> " on its own line). This is a best-effort heuristic that matches
    llama.cpp's interactive mode well enough for the demo.
    """
    await start_llama_if_needed()
    assert _llama_proc is not None
    assert _llama_proc.stdin is not None
    assert _llama_proc.stdout is not None

    # Send the user message and a newline so llama.cpp treats it as one turn.
    _llama_proc.stdin.write((user_msg + "\n").encode("utf-8"))
    await _llama_proc.stdin.drain()

    buffer = ""
    while True:
        chunk = await _llama_proc.stdout.read(128)
        if not chunk:
            # Process ended unexpectedly.
            if buffer:
                yield buffer
            break

        text = chunk.decode("utf-8", errors="ignore")
        buffer += text
        yield text

        # Heuristic: when the interactive prompt returns ("\n> "),
        # llama.cpp is waiting for the next user turn.
        if "\n> " in buffer or buffer.endswith("\n> ") or buffer.endswith("> "):
            break
