# api/server.py
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from api.engine import send_and_stream

app = FastAPI(title="ALSpec Chat Server")


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """
    WebSocket protocol:

    Client sends: raw text (one user message)

    Server responds:
      - many text frames with partial output
      - a final frame equal to "[[END_OF_RESPONSE]]"
    """
    await ws.accept()
    try:
        while True:
            user_msg = await ws.receive_text()

            async for chunk in send_and_stream(user_msg):
                await ws.send_text(chunk)

            # Sentinel for the client so it knows this turn is done
            await ws.send_text("[[END_OF_RESPONSE]]")

    except WebSocketDisconnect:
        return
