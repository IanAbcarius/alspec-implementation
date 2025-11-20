# Run on server side

uvicorn api.mock_chat_server:app --host 0.0.0.0 --port 8000

server sets up a websocket server endpoint. upon receiving a message, it uses a function from engine.py to 1. start llama (if not already on) 2. feeds the message into llama 3. returns the llama output to server.py which then sends it back to the client
On the local machine

$env:APP_SERVER_URL = "http://[HOSTIP]" <- windows example

python main.py chat

main.py starts the chat_session from chat.py which sets up a websocket endpoint connection, and afterwards, parses the user input and sends it to the server.
