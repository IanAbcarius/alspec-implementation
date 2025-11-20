import asyncio
import websockets
import os

# Default to localhost for local tests; override with APP_SERVER_URL when
# talking to Marshall's server.
APP_SERVER_URL = os.getenv("APP_SERVER_URL", "http://localhost:8000")
WS_URL = APP_SERVER_URL.replace("http", "ws") + "/ws/chat"


async def chat_session():
    print(f"Connecting to server at {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        print("Connected. Type messages. Ctrl+C to exit.\n")

        while True:
            # User input
            user_msg = input("You: ")

            # Send to server
            await ws.send(user_msg)

            # Collect model’s answer
            chunks = []
            while True:
                chunk = await ws.recv()
                if chunk == "[[END_OF_RESPONSE]]":
                    break
                chunks.append(chunk)

            model_answer = "".join(chunks).strip()
            print(f"\nModel: {model_answer}\n")
