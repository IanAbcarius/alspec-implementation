# api/mock_chat_server.py
from __future__ import annotations

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="ALSpec Mock Chat Server")


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """
    Very simple mock server for local testing.

    - Echoes the user text back with a prefix
    - Streams word by word
    - Uses the same [[END_OF_RESPONSE]] sentinel
    """
    await ws.accept()
    try:
        while True:
            prompt = await ws.receive_text()
            reply = f"[MOCK] You said: {prompt}"

            # stream word-by-word to exercise the CLI streaming logic
            for word in reply.split():
                await ws.send_text(word + " ")
                await asyncio.sleep(0.02)

            await ws.send_text("[[END_OF_RESPONSE]]")

    except WebSocketDisconnect:
        return
